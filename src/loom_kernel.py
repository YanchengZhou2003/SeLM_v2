# ===== triton 内核：CT 值融合计算 =====
import triton
import triton.language as tl
import torch
import math
from itertools import product
# ===== Python 侧包装 =====

def ct_val_triton(
    cnc_loc: torch.Tensor,          # int32 [T, C, D]
    pos_loc: torch.Tensor,          # int32 [T, S, D]
    eu_norm: torch.Tensor,          # f32   [T, S]
    cos_sta_pos: torch.Tensor,      # f32   [T, S, D]
    cos_sta_pos_sum: torch.Tensor,  # f32   [T, S]
    tp: float,
    h: float,
    out: torch.Tensor = None,       # f32   [T, S, C, D]；如为 None 则内部分配
    *,
    BLOCK_S: int = 128,
    BLOCK_CD: int = 128,
    NUM_WARPS: int = 4,
    NUM_STAGES: int = 2,
):
    """
    计算:
        ct_val = (cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + distance(cnc, pos, eu_norm)) / tp
    其中 distance(cnc, pos, eu_norm) 的广播维度是 (T, S, C, D)，本 kernel 将其与加法融合，避免中间写回。

    要求:
      - cnc_loc/pos_loc.dtype == torch.int32 且 最后一维 D 连续 (stride[-1] == 1)
      - eu_norm/cos_sta_pos/cos_sta_pos_sum/out.dtype == torch.float32
      - eu_norm 形状为 (T, S)
    """

    # ----- 校验与整理 dtype/layout -----
    assert cnc_loc.is_cuda and pos_loc.is_cuda and cos_sta_pos.is_cuda and cos_sta_pos_sum.is_cuda
    assert cnc_loc.dtype == torch.int32 and pos_loc.dtype == torch.int32, "cnc_loc/pos_loc 必须是 int32"
    assert cos_sta_pos.dtype == torch.float32 and cos_sta_pos_sum.dtype == torch.float32, "cos_* 必须是 float32"
    assert eu_norm.dtype == torch.float32 and eu_norm.is_cuda, "eu_norm 必须是 float32 且在 CUDA 上"

    # squeeze eu_norm 的最后一维（如存在）
    if eu_norm.dim() == 3 and eu_norm.size(-1) == 1:
        eu_norm = eu_norm.squeeze(-1)

    # 读取形状
    T, C, D = cnc_loc.shape
    _T2, S, _D2 = pos_loc.shape
    assert _T2 == T and _D2 == D, "pos_loc 形状与 cnc_loc 不匹配"
    assert eu_norm.shape == (T, S)
    assert cos_sta_pos.shape == (T, S, D)
    assert cos_sta_pos_sum.shape == (T, S)

    # 保证 D 连续
    assert cnc_loc.stride(-1) == 1 and pos_loc.stride(-1) == 1 and cos_sta_pos.stride(-1) == 1, \
        "最后一维 D 必须连续；请先 .contiguous()"
    cnc_loc = cnc_loc.contiguous()
    pos_loc = pos_loc.contiguous()
    cos_sta_pos = cos_sta_pos.contiguous()
    cos_sta_pos_sum = cos_sta_pos_sum.contiguous()
    eu_norm = eu_norm.contiguous()

    # 输出
    if out is None:
        out = torch.empty((T, S, C, D), device=cnc_loc.device, dtype=torch.float32)
    else:
        assert out.shape == (T, S, C, D) and out.dtype == torch.float32 and out.is_cuda

    # 网格：三维 (T, S_tiles, (C*D)_tiles)
    grid = (
        T,
        triton.cdiv(S, BLOCK_S),
        triton.cdiv(C * D, BLOCK_CD),
    )

    # 启动 kernel
    kernel_ct_val_fused_cd[grid](
        cnc_loc, pos_loc, eu_norm, cos_sta_pos, cos_sta_pos_sum, out,
        T, S, C, D,
        *cnc_loc.stride(),           # cnc_st, cnc_sc, cnc_sd
        *pos_loc.stride(),           # pos_st, pos_ss, pos_sd
        *eu_norm.stride(),           # eun_st, eun_ss
        *cos_sta_pos.stride(),       # csp_st, csp_ss, csp_sd
        *cos_sta_pos_sum.stride(),   # css_st, css_ss
        *out.stride(),               # out_st, out_ss, out_sc, out_sd
        tp, h,
        BLOCK_S=BLOCK_S, BLOCK_CD=BLOCK_CD,
        NUM_WARPS=NUM_WARPS, NUM_STAGES=NUM_STAGES,
    )
    return out


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_S': block_s, 'BLOCK_CD': block_cd}, num_warps=nw, num_stages=ns) for block_s, block_cd, nw, ns in list(product(
#             [32, 64, 128, 256],    # BLOCK_S
#             [32, 64, 128, 256],    # BLOCK_CD
#             [2, 4, 8],                # NUM_WARPS
#             [1, 2, 3, 4]           # NUM_STAGES
#         ))
#     ],
#     key=['T', 'S', 'C', 'D']
# )


@triton.jit
def kernel_ct_val_fused_cd(
    # ------ 指针 ------
    cnc_ptr,            # int32  [T, C, D]
    pos_ptr,            # int32  [T, S, D]
    eun_ptr,            # f32    [T, S]
    csp_ptr,            # f32    [T, S, D]  (cos_sta_pos)
    css_ptr,            # f32    [T, S]     (cos_sta_pos_sum)
    out_ptr,            # f32    [T, S, C, D]

    # ------ 形状（运行时传入，用于边界与折叠） ------
    T, S, C, D,

    # ------ 步长（以元素为单位，不是字节） ------
    cnc_st, cnc_sc, cnc_sd,      # cnc_loc.stride()
    pos_st, pos_ss, pos_sd,      # pos_loc.stride()
    eun_st, eun_ss,              # eu_norm.stride()
    csp_st, csp_ss, csp_sd,      # cos_sta_pos.stride()
    css_st, css_ss,              # cos_sta_pos_sum.stride()
    out_st, out_ss, out_sc, out_sd,  # out.stride()

    # ------ 标量参数 ------
    tp,    # float
    h,     # float

    # ------ tile 大小（编译期常量）------
    BLOCK_S: tl.constexpr,
    BLOCK_CD: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # ---- program_id：三维 ----
    pid_t  = tl.program_id(0)                       # [0, T)
    pid_s  = tl.program_id(1)                       # tile over S
    pid_cd = tl.program_id(2)                       # tile over (C*D)

    # ---- 本 tile 的行/列索引 ----
    offs_s  = pid_s  * BLOCK_S  + tl.arange(0, BLOCK_S)         # (BLOCK_S,)
    offs_cd = pid_cd * BLOCK_CD + tl.arange(0, BLOCK_CD)        # (BLOCK_CD,)

    # ---- 边界 mask ----
    mask_s  = offs_s  < S
    mask_cd = offs_cd < (C * D)

    # ---- 将 (cd) 还原为 (c, d) ----
    offs_c = offs_cd // D        # (BLOCK_CD,)
    offs_d = offs_cd %  D        # (BLOCK_CD,)

    # ---- 指针基址：第 t 个样本 ----
    cnc_base = cnc_ptr + pid_t * cnc_st
    pos_base = pos_ptr + pid_t * pos_st
    eun_base = eun_ptr + pid_t * eun_st
    csp_base = csp_ptr + pid_t * csp_st
    css_base = css_ptr + pid_t * css_st
    out_base = out_ptr + pid_t * out_st

    # =========================================================
    # 载入 pos / cos_sp / css / eun
    #   pos_tile:  (S_tile, CD_tile)  -> int32
    #   csp_tile:  (S_tile, CD_tile)  -> f32
    #   css_vec:   (S_tile,)          -> f32
    #   eun_vec:   (S_tile,)          -> f32
    # =========================================================
    pos_ptrs = pos_base + offs_s[:, None] * pos_ss + offs_d[None, :] * pos_sd
    pos_tile_i32 = tl.load(pos_ptrs, mask=mask_s[:, None] & mask_cd[None, :], other=0)

    csp_ptrs = csp_base + offs_s[:, None] * csp_ss + offs_d[None, :] * csp_sd
    csp_tile = tl.load(csp_ptrs, mask=mask_s[:, None] & mask_cd[None, :], other=0.0)

    css_ptrs = css_base + offs_s * css_ss
    eun_ptrs = eun_base + offs_s * eun_ss
    css_vec  = tl.load(css_ptrs, mask=mask_s, other=0.0)
    eun_vec  = tl.load(eun_ptrs, mask=mask_s, other=0.0)

    # =========================================================
    # 载入 cnc（只依赖 cd，不依赖 s）
    #   cnc_vec:   (CD_tile,)         -> int32
    # =========================================================
    cnc_ptrs = cnc_base + offs_c * cnc_sc + offs_d * cnc_sd
    cnc_vec_i32 = tl.load(cnc_ptrs, mask=mask_cd, other=0)

    # =========================================================
    # distance(cnc, pos, eun):
    #   sg = sign(cnc)*sign(pos)
    #   xor_result = |cnc| XOR |pos|
    #   exp = floor( log2(xor_result + 1) ) + 1
    #   s = exp / h
    #   dist = sg * (1 - s) * eun
    # =========================================================
    # 符号与绝对值
    sgn_pos = (tl.where(pos_tile_i32 >= 0, 1, 0) * 2 - 1).to(tl.int32)           # (S,CD)
    sgn_cnc = (tl.where(cnc_vec_i32   >= 0, 1, 0) * 2 - 1).to(tl.int32)           # (CD,)
    abs_pos = tl.where(pos_tile_i32 >= 0, pos_tile_i32, -pos_tile_i32)            # (S,CD)
    abs_cnc = tl.where(cnc_vec_i32   >= 0, cnc_vec_i32,   -cnc_vec_i32)           # (CD,)

    # 广播到 (S,CD)：按位异或
    xor_scd = (abs_pos ^ abs_cnc[None, :]).to(tl.int32)

    # exp = floor(log2(xor+1)) + 1
    xfp = (xor_scd + 1).to(tl.float32)
    exp_scd = tl.floor(tl.log2(xfp)) + 1.0

    # s = exp / h
    s_scd = exp_scd / h

    # 符号广播 + eu_norm 广播
    sgn_scd = (sgn_pos * sgn_cnc[None, :]).to(tl.float32)                         # (S,CD)
    eun_sc  = eun_vec[:, None].to(tl.float32)                                     # (S,1)

    dist_scd = sgn_scd * (1.0 - s_scd) * eun_sc                                   # (S,CD)

    # =========================================================
    # ct = (css - csp + dist) / tp
    # =========================================================
    css_sc = css_vec[:, None]                                                     # (S,1)
    ct_scd = (css_sc - csp_tile + dist_scd) / tp                                  # (S,CD)

    # =========================================================
    # 写回 out[t, s, c, d]
    # =========================================================
    out_ptrs = out_base + offs_s[:, None] * out_ss + offs_c[None, :] * out_sc + offs_d[None, :] * out_sd
    tl.store(out_ptrs, ct_scd, mask=mask_s[:, None] & mask_cd[None, :])

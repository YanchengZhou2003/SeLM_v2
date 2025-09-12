import math

import torch
import triton
import triton.language as tl


def ct_loss_triton(
    cnc_loc: torch.Tensor,          # int32 [T, C, D]        on CUDA
    pos_loc: torch.Tensor,          # int32 [T, S_tot, D]    on CUDA
    eu_norm: torch.Tensor,          # f32   [T, S_tot]       on CUDA
    cos_sta_pos: torch.Tensor,      # f32   [T, S_tot, D]    on CUDA
    cos_sta_pos_sum: torch.Tensor,  # f32   [T, S_tot]       on CUDA
    eu_val: torch.Tensor,           # f32   [T, S_tot]       on CUDA
    mask: torch.Tensor,             # f32   [T, S_tot]       on CUDA, 0/1
    lth: torch.Tensor,              # f32   [T]              on CUDA

    S_cos: int,                     # 前半段 S_cos
    tp: float,
    h: float,
    ratio_cos: float,
    ratio_cro: float,

    out_cos: torch.Tensor = None,   # f32   [T, C, D]
    out_cro: torch.Tensor = None,   # f32   [T, C, D]
    out_tot: torch.Tensor = None,   # f32   [T, C, D]

    BLOCK_CD: int = 128,
    BLOCK_S:  int = 128,
    NUM_WARPS: int = 4,
    NUM_STAGES: int = 2,
):
    # ---- 基本校验 ----
    for x, name in [
        (cnc_loc, "cnc_loc"), (pos_loc, "pos_loc"),
        (eu_norm, "eu_norm"), (cos_sta_pos, "cos_sta_pos"),
        (cos_sta_pos_sum, "cos_sta_pos_sum"), (eu_val, "eu_val"),
        (mask, "mask"), (lth, "lth"),
    ]:
        assert x.is_cuda, f"{name} 必须在 CUDA 上"

    assert cnc_loc.dtype == torch.int32 and pos_loc.dtype == torch.int32, "cnc_loc/pos_loc 必须为 int32"
    assert eu_norm.dtype == torch.float32
    assert cos_sta_pos.dtype == torch.float32 and cos_sta_pos_sum.dtype == torch.float32
    assert eu_val.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert lth.dtype == torch.float32

    T, C, D = cnc_loc.shape
    T2, S_tot, D2 = pos_loc.shape
    assert T2 == T and D2 == D
    assert eu_norm.shape == (T, S_tot)
    assert cos_sta_pos.shape == (T, S_tot, D)
    assert cos_sta_pos_sum.shape == (T, S_tot)
    assert eu_val.shape == (T, S_tot)
    assert mask.shape == (T, S_tot)
    assert lth.shape == (T,)

    # ---- 确保 D 连续 ----
    assert cnc_loc.stride(-1) == 1 and pos_loc.stride(-1) == 1 and cos_sta_pos.stride(-1) == 1, \
        "最后一维 D 必须连续；请先 .contiguous()"
    cnc_loc       = cnc_loc.contiguous()
    pos_loc       = pos_loc.contiguous()
    cos_sta_pos   = cos_sta_pos.contiguous()
    cos_sta_pos_sum = cos_sta_pos_sum.contiguous()
    eu_norm       = eu_norm.contiguous()
    eu_val        = eu_val.contiguous()
    mask          = mask.contiguous()

    # ---- 输出 ----
    if out_cos is None:
        out_cos = torch.empty((T, C, D), device=cnc_loc.device, dtype=torch.float32)
    if out_cro is None:
        out_cro = torch.empty((T, C, D), device=cnc_loc.device, dtype=torch.float32)
    if out_tot is None:
        out_tot = torch.empty((T, C, D), device=cnc_loc.device, dtype=torch.float32)

    # ---- grid 设置：二维 (T, ceil_div(C*D, BLOCK_CD)) ----
    grid = (T, triton.cdiv(C * D, BLOCK_CD))

    NUM_S_TILES = (S_tot + BLOCK_S - 1) // BLOCK_S  # 作为编译期常量传入

    kernel_ct_loss_fused_cd[grid](
        cnc_loc, pos_loc, eu_norm, cos_sta_pos, cos_sta_pos_sum, eu_val, mask, lth,
        out_cos, out_cro, out_tot,
        T, S_tot, C, D, S_cos,
        *cnc_loc.stride(),
        *pos_loc.stride(),
        *eu_norm.stride(),
        *cos_sta_pos.stride(),
        *cos_sta_pos_sum.stride(),
        *eu_val.stride(),
        *mask.stride(),
        *out_cos.stride(),
        *out_cro.stride(),
        *out_tot.stride(),
        float(tp), float(h), float(ratio_cos), float(ratio_cro),
        BLOCK_CD=BLOCK_CD, BLOCK_S=BLOCK_S,
        NUM_S_TILES=NUM_S_TILES,
        NUM_WARPS=NUM_WARPS, NUM_STAGES=NUM_STAGES,
    )

    return out_cos, out_cro, out_tot



@triton.jit
def kernel_ct_loss_fused_cd(
    # ---------- 指针 ----------
    cnc_ptr,             # int32 [T, C, D]
    pos_ptr,             # int32 [T, S_tot, D]
    eun_ptr,             # f32   [T, S_tot]      (eu_norm)
    csp_ptr,             # f32   [T, S_tot, D]   (cos_sta_pos)
    css_ptr,             # f32   [T, S_tot]      (cos_sta_pos_sum)
    euv_ptr,             # f32   [T, S_tot]      (eu_val (head/tail混合))
    msk_ptr,             # f32   [T, S_tot]      (mask ∈ {0,1})
    lth_ptr,             # f32   [T]             (lth)

    out_cos_ptr,         # f32   [T, C, D]
    out_cro_ptr,         # f32   [T, C, D]
    out_tot_ptr,         # f32   [T, C, D]

    # ---------- 形状 ----------
    T, S_tot, C, D, S_cos,

    # ---------- 步长（元素为单位） ----------
    cnc_st, cnc_sc, cnc_sd,         # cnc_loc.stride()
    pos_st, pos_ss, pos_sd,         # pos_loc.stride()
    eun_st, eun_ss,                 # eu_norm.stride()
    csp_st, csp_ss, csp_sd,         # cos_sta_pos.stride()
    css_st, css_ss,                 # cos_sta_pos_sum.stride()
    euv_st, euv_ss,                 # eu_val.stride()
    msk_st, msk_ss,                 # mask.stride()

    outc_st, outc_sc, outc_sd,      # loss_cos.stride()
    outr_st, outr_sc, outr_sd,      # loss_cro.stride()
    outt_st, outt_sc, outt_sd,      # loss_tot.stride()

    # ---------- 标量 ----------
    tp,     # f32
    h,      # f32
    w_cos,  # f32
    w_cro,  # f32

    # ---------- tile 常量 ----------
    BLOCK_CD: tl.constexpr,
    BLOCK_S:  tl.constexpr,
    NUM_S_TILES: tl.constexpr,  # = ceil_div(S_tot, BLOCK_S)
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_t  = tl.program_id(0)                 # [0, T)
    pid_cd = tl.program_id(1)                 # tile over (C*D)

    offs_cd = pid_cd * BLOCK_CD + tl.arange(0, BLOCK_CD)      # (BLOCK_CD,)
    mask_cd = offs_cd < (C * D)

    # 将 (cd) 还原为 (c, d)
    offs_c = offs_cd // D
    offs_d = offs_cd %  D

    # base 指针（第 t 个样本）
    cnc_base = cnc_ptr + pid_t * cnc_st
    pos_base = pos_ptr + pid_t * pos_st
    eun_base = eun_ptr + pid_t * eun_st
    csp_base = csp_ptr + pid_t * csp_st
    css_base = css_ptr + pid_t * css_st
    euv_base = euv_ptr + pid_t * euv_st
    msk_base = msk_ptr + pid_t * msk_st

    # 输出 base
    outc_base = out_cos_ptr + pid_t * outc_st
    outr_base = out_cro_ptr + pid_t * outr_st
    outt_base = out_tot_ptr + pid_t * outt_st

    # 载入 cnc_vec (与 s 无关)
    cnc_ptrs = cnc_base + offs_c * cnc_sc + offs_d * cnc_sd
    cnc_vec_i32 = tl.load(cnc_ptrs, mask=mask_cd, other=0)

    # 计算需要的符号与绝对值（与 s 无关）
    sgn_cnc = (tl.where(cnc_vec_i32 >= 0, 1, 0) * 2 - 1).to(tl.int32)  # (CD,)
    abs_cnc = tl.where(cnc_vec_i32 >= 0, cnc_vec_i32, -cnc_vec_i32)    # (CD,)

    # 取 lth[t]
    lth_val = tl.load(lth_ptr + pid_t)

    # 累加器
    cos_acc = tl.zeros([BLOCK_CD], dtype=tl.float32)  # square 部分的 ∑_s(...)
    # KL 的三遍扫描：先最大值，再和，再聚合
    neg_inf = tl.full([BLOCK_CD], -1e30, dtype=tl.float32)
    max_ct  = neg_inf
    max_eu  = neg_inf
    sum_ct  = tl.zeros([BLOCK_CD], dtype=tl.float32)
    sum_eu  = tl.zeros([BLOCK_CD], dtype=tl.float32)
    cro_acc = tl.zeros([BLOCK_CD], dtype=tl.float32)

    # --------------------
    # Pass 0: square 累加 & KL 的 max（只对 tail）
    # --------------------
    for it in range(NUM_S_TILES):
        s_base = it * BLOCK_S
        offs_s = s_base + tl.arange(0, BLOCK_S)                # (BLOCK_S,)
        mask_s = offs_s < S_tot

        # 加载 pos(s,d) / csp(s,d)
        pos_ptrs = pos_base + offs_s[:, None] * pos_ss + offs_d[None, :] * pos_sd
        csp_ptrs = csp_base + offs_s[:, None] * csp_ss + offs_d[None, :] * csp_sd
        pos_tile_i32 = tl.load(pos_ptrs, mask=mask_s[:, None] & mask_cd[None, :], other=0)
        csp_tile     = tl.load(csp_ptrs, mask=mask_s[:, None] & mask_cd[None, :], other=0.0)

        # 加载 css(s), eun(s), euv(s), msk(s)
        css_ptrs = css_base + offs_s * css_ss
        eun_ptrs = eun_base + offs_s * eun_ss
        euv_ptrs = euv_base + offs_s * euv_ss
        msk_ptrs = msk_base + offs_s * msk_ss

        css_vec = tl.load(css_ptrs, mask=mask_s, other=0.0)   # (S_t,)
        eun_vec = tl.load(eun_ptrs, mask=mask_s, other=0.0)
        euv_vec = tl.load(euv_ptrs, mask=mask_s, other=0.0)
        msk_vec = tl.load(msk_ptrs, mask=mask_s, other=0.0)

        # distance(cnc, pos, eun):
        sgn_pos = (tl.where(pos_tile_i32 >= 0, 1, 0) * 2 - 1).to(tl.int32)
        abs_pos = tl.where(pos_tile_i32 >= 0, pos_tile_i32, -pos_tile_i32)

        xor_scd = (abs_pos ^ abs_cnc[None, :]).to(tl.int32)        # (S_t, CD)
        xfp     = (xor_scd + 1).to(tl.float32)
        exp_scd = tl.floor(tl.log2(xfp)) + 1.0
        s_scd   = exp_scd / h

        sgn_scd = (sgn_pos * sgn_cnc[None, :]).to(tl.float32)
        eun_sc  = eun_vec[:, None].to(tl.float32)

        dist_scd = sgn_scd * (1.0 - s_scd) * eun_sc

        # ct(s,cd) = (css[s] - csp(s,d) + dist) / tp
        css_sc = css_vec[:, None]
        ct_scd = (css_sc - csp_tile + dist_scd) / tp

        # head: square 累加（offs_s < S_cos）
        head_mask = (offs_s < S_cos) & mask_s
        any_true = tl.sum(head_mask, axis=None) > 0
        if any_true:
            # euv 作为 target；mask 作用在逐项 loss 上（广播到 CD）
            eu_sc  = euv_vec[:, None]           # (S_t,1)
            msk_sc = msk_vec[:, None]           # (S_t,1)

            diff   = (ct_scd - eu_sc)
            sq     = diff * diff
            sq     = sq * msk_sc
            # 对 S 维累加
            cos_acc += tl.sum(sq, axis=0)

        # tail: KL 的 max（offs_s >= S_cos）
        tail_mask = (offs_s >= S_cos) & mask_s
        any_true = tl.sum(tail_mask, axis=None) > 0
        if any_true:
            # ct 的最大值（逐 cd）
            ct_tail = tl.where(tail_mask[:, None], ct_scd, neg_inf[None, :])
            max_ct  = tl.maximum(max_ct, tl.max(ct_tail, axis=0))

            # eu 的最大值（注意 eu 与 cd 无关；按 cd 冗余保存，简单直接）
            eu_tail = tl.where(tail_mask, euv_vec, -1e30)
            eu_tail_max = tl.max(eu_tail, axis=0)
            max_eu = tl.maximum(max_eu, eu_tail_max)

    # 若没有 tail（S_tot == S_cos），则跳过 KL 的后续

    # --------------------
    # Pass 1: KL 的 sum(exp(x - max))
    # --------------------
    for it in range(NUM_S_TILES):
        s_base = it * BLOCK_S
        offs_s = s_base + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S_tot

        tail_mask = (offs_s >= S_cos) & mask_s
        any_true = tl.sum(tail_mask, axis=None) > 0
        if ~any_true:
            continue

        pos_ptrs = pos_base + offs_s[:, None] * pos_ss + offs_d[None, :] * pos_sd
        csp_ptrs = csp_base + offs_s[:, None] * csp_ss + offs_d[None, :] * csp_sd
        pos_tile_i32 = tl.load(pos_ptrs, mask=mask_s[:, None] & mask_cd[None, :], other=0)
        csp_tile     = tl.load(csp_ptrs, mask=mask_s[:, None] & mask_cd[None, :], other=0.0)

        css_vec = tl.load(css_base + offs_s * css_ss, mask=mask_s, other=0.0)
        eun_vec = tl.load(eun_base + offs_s * eun_ss, mask=mask_s, other=0.0)
        euv_vec = tl.load(euv_base + offs_s * euv_ss, mask=mask_s, other=0.0)

        sgn_pos = (tl.where(pos_tile_i32 >= 0, 1, 0) * 2 - 1).to(tl.int32)
        abs_pos = tl.where(pos_tile_i32 >= 0, pos_tile_i32, -pos_tile_i32)

        xor_scd = (abs_pos ^ abs_cnc[None, :]).to(tl.int32)
        xfp     = (xor_scd + 1).to(tl.float32)
        exp_scd = tl.floor(tl.log2(xfp)) + 1.0
        s_scd   = exp_scd / h

        sgn_scd = (sgn_pos * sgn_cnc[None, :]).to(tl.float32)
        eun_sc  = eun_vec[:, None].to(tl.float32)

        dist_scd = sgn_scd * (1.0 - s_scd) * eun_sc
        ct_scd   = (css_vec[:, None] - csp_tile + dist_scd) / tp

        # 只对 tail 位置做 sum(exp(x - max))
        ct_tail = tl.where(tail_mask[:, None], ct_scd, -1e30)
        sum_ct += tl.sum(tl.exp(ct_tail - max_ct[None, :]), axis=0)

        eu_tail = tl.where(tail_mask, euv_vec, -1e30)
        # eu 与 cd 无关，但我们冗余为每个 cd 计算，简化实现
        sum_eu += tl.sum(tl.exp(eu_tail - max_eu), axis=0)

    # 避免 log(0)
    eps = 1e-20
    logZ_ct = max_ct + tl.log(sum_ct + eps)
    logZ_eu = max_eu + tl.log(sum_eu + eps)

    # --------------------
    # Pass 2: KL 聚合：∑ p_eu * (log p_eu - log p_ct) * mask
    # --------------------
    for it in range(NUM_S_TILES):
        s_base = it * BLOCK_S
        offs_s = s_base + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S_tot

        tail_mask = (offs_s >= S_cos) & mask_s
        any_true = tl.sum(tail_mask, axis=None) > 0
        if ~any_true:
            continue

        pos_ptrs = pos_base + offs_s[:, None] * pos_ss + offs_d[None, :] * pos_sd
        csp_ptrs = csp_base + offs_s[:, None] * csp_ss + offs_d[None, :] * csp_sd
        pos_tile_i32 = tl.load(pos_ptrs, mask=mask_s[:, None] & mask_cd[None, :], other=0)
        csp_tile     = tl.load(csp_ptrs, mask=mask_s[:, None] & mask_cd[None, :], other=0.0)

        css_vec = tl.load(css_base + offs_s * css_ss, mask=mask_s, other=0.0)
        eun_vec = tl.load(eun_base + offs_s * eun_ss, mask=mask_s, other=0.0)
        euv_vec = tl.load(euv_base + offs_s * euv_ss, mask=mask_s, other=0.0)
        msk_vec = tl.load(msk_base + offs_s * msk_ss, mask=mask_s, other=0.0)

        sgn_pos = (tl.where(pos_tile_i32 >= 0, 1, 0) * 2 - 1).to(tl.int32)
        abs_pos = tl.where(pos_tile_i32 >= 0, pos_tile_i32, -pos_tile_i32)

        xor_scd = (abs_pos ^ abs_cnc[None, :]).to(tl.int32)
        xfp     = (xor_scd + 1).to(tl.float32)
        exp_scd = tl.floor(tl.log2(xfp)) + 1.0
        s_scd   = exp_scd / h

        sgn_scd = (sgn_pos * sgn_cnc[None, :]).to(tl.float32)
        eun_sc  = eun_vec[:, None].to(tl.float32)

        dist_scd = sgn_scd * (1.0 - s_scd) * eun_sc
        ct_scd   = (css_vec[:, None] - csp_tile + dist_scd) / tp

        log_ct   = ct_scd - logZ_ct[None, :]
        log_eu   = euv_vec[:, None] - logZ_eu[None, :]
        p_eu     = tl.exp(euv_vec[:, None] - logZ_eu[None, :])

        term     = p_eu * (log_eu - log_ct)
        term     = term * msk_vec[:, None]
        cro_acc += tl.sum(term, axis=0)

    # 归一化 + 加权
    loss_cos = cos_acc / lth_val
    loss_cro = cro_acc  # 原实现未对 cro 用 lth 归一
    loss_tot = w_cos * loss_cos + w_cro * loss_cro

    # 写回
    out_idx = outc_base + offs_c * outc_sc + offs_d * outc_sd
    tl.store(out_idx, loss_cos, mask=mask_cd)

    out_idx = outr_base + offs_c * outr_sc + offs_d * outr_sd
    tl.store(out_idx, loss_cro, mask=mask_cd)

    out_idx = outt_base + offs_c * outt_sc + offs_d * outt_sd
    tl.store(out_idx, loss_tot, mask=mask_cd)

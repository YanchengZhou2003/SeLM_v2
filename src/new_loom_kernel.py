import triton
import triton.language as tl
import torch


def ct_loss_triton_sampled_2dteacher(
    cnc_loc: torch.Tensor,          # int32, [T, C, D]
    pos_loc: torch.Tensor,          # int32, [T, S, D]
    eu_norm: torch.Tensor,          # float32, [T, S]
    cos_sta_pos: torch.Tensor,      # float32, [T, S, D]
    cos_sta_pos_sum: torch.Tensor,  # float32, [T, S]
    eu_teacher: torch.Tensor,       # float32, [T, S] 或 [T, S, 1, 1] —— 教师 logits（未缩放）
    tp: float,
    h: float,
    temperature: float,
    N_dynbr: int,                   # 动态区间长度 = N_sampled
    N_topk: int,                    # 前 N_topk 为 top-k
    N_total: int,                   # 总类数 / 总邻居数，用于 HT 校正
    cur_type: str = "train",        # "train" or "eval"
    cur_tar: torch.Tensor | None = None,  # [T]，静态区间 target：0 ~ (S - N_dynbr - 1)
    scale: float = 20.0,            # logits 缩放中的 20
    *,
    BLOCK_S: int = 128,
    BLOCK_CD: int = 128,
    MAX_S: int = 4096,
    NUM_WARPS: int = 4,
    NUM_STAGES: int = 2,
):
    """
    计算：
      - 动态区间 [0:N_dynbr) 的 HT sampled softmax KL：
          logits_dyn_eu_adj = eu_teacher * (20/temperature) + log_w
          logits_dyn_ct_adj = ct_val(...) * (20/temperature) + log_w
          KL( softmax(logits_dyn_eu_adj) || softmax(logits_dyn_ct_adj) )
      - 静态区间 [N_dynbr:S) 的 CE（仅 train 模式）：
          -log softmax(ct_val * 20/temperature)[cur_tar]
    不会物化任何 [T, S, C, D] 的教师 logits 或 ct_val。
    返回：
      loss_dyn_dyn: [T, C, D]
      loss_dyn_sta: [T, C, D] 或 None
    """
    device = cnc_loc.device
    assert cnc_loc.is_cuda and pos_loc.is_cuda and eu_norm.is_cuda
    assert cos_sta_pos.is_cuda and cos_sta_pos_sum.is_cuda

    # ---- 形状检查 ----
    T, C, D = cnc_loc.shape
    T2, S, D2 = pos_loc.shape
    assert T2 == T and D2 == D, "pos_loc 形状与 cnc_loc 不匹配"
    assert eu_norm.shape == (T, S)
    assert cos_sta_pos.shape == (T, S, D)
    assert cos_sta_pos_sum.shape == (T, S)
    assert 0 <= N_dynbr <= S
    assert N_topk <= N_dynbr <= N_total

    # ---- teacher logits: [T,S] 或 [T,S,1,1] ----
    if eu_teacher.dim() == 4:
        assert eu_teacher.shape[0] == T and eu_teacher.shape[1] == S
        assert eu_teacher.shape[2] == 1 and eu_teacher.shape[3] == 1
        eu_teacher = eu_teacher.squeeze(-1).squeeze(-1)
    assert eu_teacher.shape == (T, S), f"eu_teacher 期望 [T,S]，得到 {eu_teacher.shape}"
    assert eu_teacher.dtype == torch.float32

    # ---- train / eval 模式 ----
    if cur_type == "train":
        assert cur_tar is not None, "train 模式必须提供 cur_tar"
        assert cur_tar.shape == (T,)
        assert N_dynbr < S, "train 模式要求有静态区间：N_dynbr < S"
        has_static_loss = 1
    else:
        cur_tar = None
        has_static_loss = 0

    assert S <= MAX_S, f"S={S} 超过 MAX_S={MAX_S}，需要增大 MAX_S"

    # ---- contiguous + dtype ----
    cnc_loc         = cnc_loc.to(torch.int32).contiguous()
    pos_loc         = pos_loc.to(torch.int32).contiguous()
    eu_norm         = eu_norm.to(torch.float32).contiguous()
    cos_sta_pos     = cos_sta_pos.to(torch.float32).contiguous()
    cos_sta_pos_sum = cos_sta_pos_sum.to(torch.float32).contiguous()
    eu_teacher      = eu_teacher.to(torch.float32).contiguous()
    if cur_tar is not None:
        cur_tar = cur_tar.to(torch.int32).contiguous()
    else:
        cur_tar = torch.zeros((1,), device=device, dtype=torch.int32)  # 占位

    # ---- HT 权重 log_w: (N_dynbr,) ----
    n_random = N_dynbr - N_topk
    total_random_pool = N_total - N_topk
    w = torch.ones(N_dynbr, device=device, dtype=torch.float32)
    if n_random > 0:
        pi = n_random / float(total_random_pool)
        w[N_topk:N_dynbr] = 1.0 / pi
    log_w = torch.log(w).contiguous()  # [N_dynbr]
    logw_ss = log_w.stride(0)

    # ---- 输出 ----
    loss_dyn_dyn = torch.empty((T, C, D), device=device, dtype=torch.float32)
    if has_static_loss:
        loss_dyn_sta = torch.empty((T, C, D), device=device, dtype=torch.float32)
    else:
        loss_dyn_sta = torch.empty((1, 1, 1), device=device, dtype=torch.float32)  # 占位

    # ---- teacher strides ----
    eu_st, eu_ss = eu_teacher.stride()

    # ---- logits 缩放因子 ----
    logit_scale = float(scale) / float(temperature)

    CD = C * D
    grid = (
        T,
        triton.cdiv(CD, BLOCK_CD),
    )

    kernel_ct_loss_sampled_2dteacher[grid](
        cnc_loc, pos_loc, eu_norm,
        cos_sta_pos, cos_sta_pos_sum,
        eu_teacher,
        cur_tar,
        log_w,
        loss_dyn_dyn,
        loss_dyn_sta,
        T, S, C, D, N_dynbr,
        has_static_loss,
        # strides
        *cnc_loc.stride(),          # cnc_st, cnc_sc, cnc_sd
        *pos_loc.stride(),          # pos_st, pos_ss, pos_sd
        *eu_norm.stride(),          # eun_st, eun_ss
        *cos_sta_pos.stride(),      # csp_st, csp_ss, csp_sd
        *cos_sta_pos_sum.stride(),  # css_st, css_ss
        eu_st, eu_ss,               # teacher [T,S]
        logw_ss,                    # log_w stride
        *loss_dyn_dyn.stride(),     # ldd_st, ldd_sc, ldd_sd
        *loss_dyn_sta.stride(),     # lds_st, lds_sc, lds_sd
        tp, h,
        logit_scale,
        MAX_S=MAX_S,
        BLOCK_S=BLOCK_S,
        BLOCK_CD=BLOCK_CD,
        NUM_WARPS=NUM_WARPS,
        NUM_STAGES=NUM_STAGES,
    )

    if has_static_loss:
        return loss_dyn_dyn, loss_dyn_sta
    else:
        return loss_dyn_dyn, None


@triton.jit
def kernel_ct_loss_sampled_2dteacher(
    # ------ 指针 ------
    cnc_ptr, pos_ptr, eun_ptr,
    csp_ptr, css_ptr,
    eu_ptr,            # teacher: [T, S]
    cur_tar_ptr,
    logw_ptr,          # [N_dynbr]
    loss_dyn_dyn_ptr,  # [T, C, D]
    loss_dyn_sta_ptr,  # [T, C, D] 或 dummy
    # ------ 尺寸 ------
    T, S, C, D, N_dynbr,
    has_static_loss,
    # ------ 步长（元素） ------
    cnc_st, cnc_sc, cnc_sd,
    pos_st, pos_ss, pos_sd,
    eun_st, eun_ss,
    csp_st, csp_ss, csp_sd,
    css_st, css_ss,
    eu_st, eu_ss,          # teacher [T, S]
    logw_ss,
    ldd_st, ldd_sc, ldd_sd,
    lds_st, lds_sc, lds_sd,
    # ------ 标量 ------
    tp, h,
    logit_scale,
    # ------ 编译期常量 ------
    MAX_S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_CD: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    i64 = tl.int64

    pid_t  = tl.program_id(0).to(i64)
    pid_cd = tl.program_id(1).to(i64)

    T_i    = T.to(i64)
    S_i    = S.to(i64)
    C_i    = C.to(i64)
    D_i    = D.to(i64)
    Ndyn_i = N_dynbr.to(i64)


    CD_i = C_i * D_i

    offs_cd = pid_cd * BLOCK_CD + tl.arange(0, BLOCK_CD).to(i64)
    mask_cd = offs_cd < CD_i

    offs_c = offs_cd // D_i
    offs_d = offs_cd %  D_i

    cnc_base = cnc_ptr + pid_t * cnc_st
    pos_base = pos_ptr + pid_t * pos_st
    eun_base = eun_ptr + pid_t * eun_st
    csp_base = csp_ptr + pid_t * csp_st
    css_base = css_ptr + pid_t * css_st
    eu_base  = eu_ptr  + pid_t * eu_st

    ldd_base = loss_dyn_dyn_ptr + pid_t * ldd_st
    lds_base = loss_dyn_sta_ptr + pid_t * lds_st

    # ---- 当前 (t, cd_tile) 的 cnc ----
    cnc_ptrs    = cnc_base + offs_c * cnc_sc + offs_d * cnc_sd
    cnc_vec_i32 = tl.load(cnc_ptrs, mask=mask_cd, other=0)
    sgn_cnc     = (tl.where(cnc_vec_i32 >= 0, 1, -1)).to(tl.int32)
    abs_cnc     = tl.where(cnc_vec_i32 >= 0, cnc_vec_i32, -cnc_vec_i32)

    minus_inf = -1.0e30

    # ======================================================================
    # 动态区间 [0:N_dynbr): HT sampled softmax KL
    # teacher logits: eu_teacher[t,s]，在 tile 内广播到 [BLOCK_S, BLOCK_CD]
    # ======================================================================
    max_eu_dyn = tl.full((BLOCK_CD,), minus_inf, tl.float32)
    max_ct_dyn = tl.full((BLOCK_CD,), minus_inf, tl.float32)

    # ---------- pass 1: max ----------
    for s_start in range(0, MAX_S, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S).to(i64)
        mask_s_dyn = offs_s < Ndyn_i
        mask_scd_dyn = mask_s_dyn[:, None] & mask_cd[None, :]

        # teacher: [T,S] -> eu_vec: [BLOCK_S]
        eu_vec_ptrs = eu_base + offs_s * eu_ss
        eu_vec = tl.load(eu_vec_ptrs, mask=mask_s_dyn, other=minus_inf)  # [BLOCK_S]
        eu_tile = eu_vec[:, None]                                        # [BLOCK_S, 1] broadcast
        eu_tile = tl.where(mask_scd_dyn, eu_tile, minus_inf)

        # pos / cos / sum / eun
        pos_ptrs = pos_base + offs_s[:, None] * pos_ss + offs_d[None, :] * pos_sd
        pos_tile_i32 = tl.load(pos_ptrs, mask=mask_scd_dyn, other=0)

        csp_ptrs = csp_base + offs_s[:, None] * csp_ss + offs_d[None, :] * csp_sd
        csp_tile = tl.load(csp_ptrs, mask=mask_scd_dyn, other=0.0)

        css_ptrs = css_base + offs_s * css_ss
        eun_ptrs = eun_base + offs_s * eun_ss
        css_vec  = tl.load(css_ptrs, mask=mask_s_dyn, other=0.0)
        eun_vec  = tl.load(eun_ptrs, mask=mask_s_dyn, other=0.0)

        # distance(cnc,pos,eun)
        sgn_pos = (tl.where(pos_tile_i32 >= 0, 1, -1)).to(tl.int32)
        abs_pos = tl.where(pos_tile_i32 >= 0, pos_tile_i32, -pos_tile_i32)
        xor_scd = (abs_pos ^ abs_cnc[None, :]).to(tl.int32)
        xfp     = (xor_scd + 1).to(tl.float32)
        exp_scd = tl.floor(tl.log2(xfp)) + 1.0
        s_scd   = exp_scd / h

        sgn_scd = (sgn_pos * sgn_cnc[None, :]).to(tl.float32)
        eun_sc  = eun_vec[:, None].to(tl.float32)
        css_sc  = css_vec[:, None]
        dist_scd = sgn_scd * (1.0 - s_scd) * eun_sc

        ct_scd = (css_sc - csp_tile + dist_scd) / tp
        ct_scd = tl.where(mask_scd_dyn, ct_scd, minus_inf)

        # log_w: [N_dynbr] -> [BLOCK_S,1]
        logw_ptrs = logw_ptr + offs_s * logw_ss
        logw_s    = tl.load(logw_ptrs, mask=mask_s_dyn, other=0.0)
        logw_sc   = logw_s[:, None]

        # 调整后的 logits
        eu_adj = eu_tile * logit_scale + logw_sc
        ct_adj = ct_scd * logit_scale + logw_sc

        max_eu_dyn = tl.maximum(max_eu_dyn, tl.max(eu_adj, axis=0))
        max_ct_dyn = tl.maximum(max_ct_dyn, tl.max(ct_adj, axis=0))

    # ---------- pass 2: sumexp ----------
    sumexp_eu_dyn = tl.zeros((BLOCK_CD,), dtype=tl.float32)
    sumexp_ct_dyn = tl.zeros((BLOCK_CD,), dtype=tl.float32)

    for s_start in range(0, MAX_S, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S).to(i64)
        mask_s_dyn = offs_s < Ndyn_i
        mask_scd_dyn = mask_s_dyn[:, None] & mask_cd[None, :]

        eu_vec_ptrs = eu_base + offs_s * eu_ss
        eu_vec = tl.load(eu_vec_ptrs, mask=mask_s_dyn, other=minus_inf)
        eu_tile = eu_vec[:, None]
        eu_tile = tl.where(mask_scd_dyn, eu_tile, minus_inf)

        pos_ptrs = pos_base + offs_s[:, None] * pos_ss + offs_d[None, :] * pos_sd
        pos_tile_i32 = tl.load(pos_ptrs, mask=mask_scd_dyn, other=0)

        csp_ptrs = csp_base + offs_s[:, None] * csp_ss + offs_d[None, :] * csp_sd
        csp_tile = tl.load(csp_ptrs, mask=mask_scd_dyn, other=0.0)

        css_ptrs = css_base + offs_s * css_ss
        eun_ptrs = eun_base + offs_s * eun_ss
        css_vec  = tl.load(css_ptrs, mask=mask_s_dyn, other=0.0)
        eun_vec  = tl.load(eun_ptrs, mask=mask_s_dyn, other=0.0)

        sgn_pos = (tl.where(pos_tile_i32 >= 0, 1, -1)).to(tl.int32)
        abs_pos = tl.where(pos_tile_i32 >= 0, pos_tile_i32, -pos_tile_i32)
        xor_scd = (abs_pos ^ abs_cnc[None, :]).to(tl.int32)
        xfp     = (xor_scd + 1).to(tl.float32)
        exp_scd = tl.floor(tl.log2(xfp)) + 1.0
        s_scd   = exp_scd / h

        sgn_scd = (sgn_pos * sgn_cnc[None, :]).to(tl.float32)
        eun_sc  = eun_vec[:, None].to(tl.float32)
        css_sc  = css_vec[:, None]
        dist_scd = sgn_scd * (1.0 - s_scd) * eun_sc

        ct_scd = (css_sc - csp_tile + dist_scd) / tp
        ct_scd = tl.where(mask_scd_dyn, ct_scd, minus_inf)

        logw_ptrs = logw_ptr + offs_s * logw_ss
        logw_s    = tl.load(logw_ptrs, mask=mask_s_dyn, other=0.0)
        logw_sc   = logw_s[:, None]

        eu_adj = eu_tile * logit_scale + logw_sc
        ct_adj = ct_scd * logit_scale + logw_sc

        eu_shift = eu_adj - max_eu_dyn[None, :]
        ct_shift = ct_adj - max_ct_dyn[None, :]

        exp_eu = tl.exp(eu_shift)
        exp_ct = tl.exp(ct_shift)

        sumexp_eu_dyn += tl.sum(exp_eu, axis=0)
        sumexp_ct_dyn += tl.sum(exp_ct, axis=0)

    log_sumexp_eu_dyn = tl.log(sumexp_eu_dyn)
    log_sumexp_ct_dyn = tl.log(sumexp_ct_dyn)

    # ---------- pass 3: KL ----------
    loss_dyn_dyn_local = tl.zeros((BLOCK_CD,), dtype=tl.float32)

    for s_start in range(0, MAX_S, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S).to(i64)
        mask_s_dyn = offs_s < Ndyn_i
        mask_scd_dyn = mask_s_dyn[:, None] & mask_cd[None, :]

        eu_vec_ptrs = eu_base + offs_s * eu_ss
        eu_vec = tl.load(eu_vec_ptrs, mask=mask_s_dyn, other=minus_inf)
        eu_tile = eu_vec[:, None]
        eu_tile = tl.where(mask_scd_dyn, eu_tile, minus_inf)

        pos_ptrs = pos_base + offs_s[:, None] * pos_ss + offs_d[None, :] * pos_sd
        pos_tile_i32 = tl.load(pos_ptrs, mask=mask_scd_dyn, other=0)

        csp_ptrs = csp_base + offs_s[:, None] * csp_ss + offs_d[None, :] * csp_sd
        csp_tile = tl.load(csp_ptrs, mask=mask_scd_dyn, other=0.0)

        css_ptrs = css_base + offs_s * css_ss
        eun_ptrs = eun_base + offs_s * eun_ss
        css_vec  = tl.load(css_ptrs, mask=mask_s_dyn, other=0.0)
        eun_vec  = tl.load(eun_ptrs, mask=mask_s_dyn, other=0.0)

        sgn_pos = (tl.where(pos_tile_i32 >= 0, 1, -1)).to(tl.int32)
        abs_pos = tl.where(pos_tile_i32 >= 0, pos_tile_i32, -pos_tile_i32)
        xor_scd = (abs_pos ^ abs_cnc[None, :]).to(tl.int32)
        xfp     = (xor_scd + 1).to(tl.float32)
        exp_scd = tl.floor(tl.log2(xfp)) + 1.0
        s_scd   = exp_scd / h

        sgn_scd = (sgn_pos * sgn_cnc[None, :]).to(tl.float32)
        eun_sc  = eun_vec[:, None].to(tl.float32)
        css_sc  = css_vec[:, None]
        dist_scd = sgn_scd * (1.0 - s_scd) * eun_sc

        ct_scd = (css_sc - csp_tile + dist_scd) / tp
        ct_scd = tl.where(mask_scd_dyn, ct_scd, minus_inf)

        logw_ptrs = logw_ptr + offs_s * logw_ss
        logw_s    = tl.load(logw_ptrs, mask=mask_s_dyn, other=0.0)
        logw_sc   = logw_s[:, None]

        eu_adj = eu_tile * logit_scale + logw_sc
        ct_adj = ct_scd * logit_scale + logw_sc

        eu_shift = eu_adj - max_eu_dyn[None, :]
        ct_shift = ct_adj - max_ct_dyn[None, :]

        log_p_eu = eu_shift - log_sumexp_eu_dyn[None, :]
        log_p_ct = ct_shift - log_sumexp_ct_dyn[None, :]

        p_eu = tl.exp(log_p_eu)
        kl_tile = p_eu * (log_p_eu - log_p_ct)

        loss_dyn_dyn_local += tl.sum(kl_tile, axis=0)

    ldd_ptrs = ldd_base + offs_c * ldd_sc + offs_d * ldd_sd
    tl.store(ldd_ptrs, loss_dyn_dyn_local, mask=mask_cd)

    # ======================================================================
    # 静态区间 [N_dynbr : S): cross-entropy (未采样)
    # ======================================================================
    if has_static_loss != 0:
        if Ndyn_i < S_i:
            tar   = tl.load(cur_tar_ptr + pid_t)
            tar_i = tar.to(i64)
            s_target = Ndyn_i + tar_i

            # pass 1: max
            max_ct_sta = tl.full((BLOCK_CD,), minus_inf, tl.float32)
            for s_start in range(0, MAX_S, BLOCK_S):
                offs_s = s_start + tl.arange(0, BLOCK_S).to(i64)
                mask_s_sta = (offs_s >= Ndyn_i) & (offs_s < S_i)
                mask_scd_sta = mask_s_sta[:, None] & mask_cd[None, :]

                pos_ptrs = pos_base + offs_s[:, None] * pos_ss + offs_d[None, :] * pos_sd
                pos_tile_i32 = tl.load(pos_ptrs, mask=mask_scd_sta, other=0)

                csp_ptrs = csp_base + offs_s[:, None] * csp_ss + offs_d[None, :] * csp_sd
                csp_tile = tl.load(csp_ptrs, mask=mask_scd_sta, other=0.0)

                css_ptrs = css_base + offs_s * css_ss
                eun_ptrs = eun_base + offs_s * eun_ss
                css_vec  = tl.load(css_ptrs, mask=mask_s_sta, other=0.0)
                eun_vec  = tl.load(eun_ptrs, mask=mask_s_sta, other=0.0)

                sgn_pos = (tl.where(pos_tile_i32 >= 0, 1, -1)).to(tl.int32)
                abs_pos = tl.where(pos_tile_i32 >= 0, pos_tile_i32, -pos_tile_i32)
                xor_scd = (abs_pos ^ abs_cnc[None, :]).to(tl.int32)
                xfp     = (xor_scd + 1).to(tl.float32)
                exp_scd = tl.floor(tl.log2(xfp)) + 1.0
                s_scd   = exp_scd / h

                sgn_scd = (sgn_pos * sgn_cnc[None, :]).to(tl.float32)
                eun_sc  = eun_vec[:, None].to(tl.float32)
                css_sc  = css_vec[:, None]
                dist_scd = sgn_scd * (1.0 - s_scd) * eun_sc

                ct_scd = (css_sc - csp_tile + dist_scd) / tp
                ct_scd = tl.where(mask_scd_sta, ct_scd, minus_inf)

                ct_scaled = ct_scd * logit_scale
                max_ct_sta = tl.maximum(max_ct_sta, tl.max(ct_scaled, axis=0))

            # pass 2: sumexp
            sumexp_ct_sta = tl.zeros((BLOCK_CD,), dtype=tl.float32)
            for s_start in range(0, MAX_S, BLOCK_S):
                offs_s = s_start + tl.arange(0, BLOCK_S).to(i64)
                mask_s_sta = (offs_s >= Ndyn_i) & (offs_s < S_i)
                mask_scd_sta = mask_s_sta[:, None] & mask_cd[None, :]

                pos_ptrs = pos_base + offs_s[:, None] * pos_ss + offs_d[None, :] * pos_sd
                pos_tile_i32 = tl.load(pos_ptrs, mask=mask_scd_sta, other=0)

                csp_ptrs = csp_base + offs_s[:, None] * csp_ss + offs_d[None, :] * csp_sd
                csp_tile = tl.load(csp_ptrs, mask=mask_scd_sta, other=0.0)

                css_ptrs = css_base + offs_s * css_ss
                eun_ptrs = eun_base + offs_s * eun_ss
                css_vec  = tl.load(css_ptrs, mask=mask_s_sta, other=0.0)
                eun_vec  = tl.load(eun_ptrs, mask=mask_s_sta, other=0.0)

                sgn_pos = (tl.where(pos_tile_i32 >= 0, 1, -1)).to(tl.int32)
                abs_pos = tl.where(pos_tile_i32 >= 0, pos_tile_i32, -pos_tile_i32)
                xor_scd = (abs_pos ^ abs_cnc[None, :]).to(tl.int32)
                xfp     = (xor_scd + 1).to(tl.float32)
                exp_scd = tl.floor(tl.log2(xfp)) + 1.0
                s_scd   = exp_scd / h

                sgn_scd = (sgn_pos * sgn_cnc[None, :]).to(tl.float32)
                eun_sc  = eun_vec[:, None].to(tl.float32)
                css_sc  = css_vec[:, None]
                dist_scd = sgn_scd * (1.0 - s_scd) * eun_sc

                ct_scd = (css_sc - csp_tile + dist_scd) / tp
                ct_scd = tl.where(mask_scd_sta, ct_scd, minus_inf)

                ct_scaled = ct_scd * logit_scale
                ct_shift  = ct_scaled - max_ct_sta[None, :]
                exp_ct    = tl.exp(ct_shift)

                sumexp_ct_sta += tl.sum(exp_ct, axis=0)

            log_sumexp_ct_sta = tl.log(sumexp_ct_sta)
            logZ_ct_sta       = log_sumexp_ct_sta + max_ct_sta

            # pass 3: 在包含 s_target 的 tile 内更新 loss_dyn_sta_local
            loss_dyn_sta_local = tl.zeros((BLOCK_CD,), dtype=tl.float32)

            for s_start in range(0, MAX_S, BLOCK_S):
                offs_s = s_start + tl.arange(0, BLOCK_S).to(i64)
                mask_s_sta = (offs_s >= Ndyn_i) & (offs_s < S_i)
                mask_scd_sta = mask_s_sta[:, None] & mask_cd[None, :]

                # 计算 ct_scaled（与之前完全相同）
                pos_ptrs = pos_base + offs_s[:, None] * pos_ss + offs_d[None, :] * pos_sd
                pos_tile_i32 = tl.load(pos_ptrs, mask=mask_scd_sta, other=0)

                csp_ptrs = csp_base + offs_s[:, None] * csp_ss + offs_d[None, :] * csp_sd
                csp_tile = tl.load(csp_ptrs, mask=mask_scd_sta, other=0.0)

                css_ptrs = css_base + offs_s * css_ss
                eun_ptrs = eun_base + offs_s * eun_ss
                css_vec  = tl.load(css_ptrs, mask=mask_s_sta, other=0.0)
                eun_vec  = tl.load(eun_ptrs, mask=mask_s_sta, other=0.0)

                sgn_pos = (tl.where(pos_tile_i32 >= 0, 1, -1))
                abs_pos = tl.where(pos_tile_i32 >= 0, pos_tile_i32, -pos_tile_i32)
                xor_scd = (abs_pos ^ abs_cnc[None, :])
                xfp     = (xor_scd + 1).to(tl.float32)
                exp_scd = tl.floor(tl.log2(xfp)) + 1.0
                s_scd   = exp_scd / h

                sgn_scd = (sgn_pos * sgn_cnc[None, :]).to(tl.float32)
                eun_sc  = eun_vec[:, None].to(tl.float32)
                css_sc  = css_vec[:, None]
                dist_scd = sgn_scd * (1.0 - s_scd) * eun_sc

                ct_scd = (css_sc - csp_tile + dist_scd) / tp
                ct_scd = tl.where(mask_scd_sta, ct_scd, minus_inf)

                ct_scaled = ct_scd * logit_scale

                # ---- 用 mask 选出 target 行，无需 if / 索引 ----
                target_mask = (offs_s == s_target) & mask_s_sta            # [BLOCK_S]
                target_mask_2d = target_mask[:, None] & mask_scd_sta       # [BLOCK_S, BLOCK_CD]

                ct_row = tl.sum(ct_scaled * target_mask_2d, axis=0)        # [BLOCK_CD]
                has_target = tl.where(tl.sum(target_mask, axis=0) != 0, 1.0, 0.0)

                contrib = -(ct_row - logZ_ct_sta) * has_target
                loss_dyn_sta_local += contrib

            lds_ptrs = lds_base + offs_c * lds_sc + offs_d * lds_sd
            tl.store(lds_ptrs, loss_dyn_sta_local, mask=mask_cd)


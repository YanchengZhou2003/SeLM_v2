# triton_loom.py
import pdb
from typing import Tuple

import torch
import triton
import triton.language as tl

from src.para import (batch_size, block_size, division_fact, factor, h, tp,
                      vocab_size)

# @triton.jit
# def _get_distance_kernel(coord1, coord2, lut_ptr):
#     # sg = (((coord1 >= 0) << 1) - 1) * (((coord2 >= 0) << 1) - 1)
#     s1 = (coord1 >= 0).to(tl.int32) * 2 - 1
#     s2 = (coord2 >= 0).to(tl.int32) * 2 - 1
#     xorv = tl.abs(coord1) ^ tl.abs(coord2)
#     # s = tl.load(lut_ptr + xorv.to(tl.int64))
    
#     return (s1 * s2).to(s.dtype) * (1.0 - s)

@triton.jit
def _get_distance_kernel(coord1, coord2, H: tl.constexpr):
    # 符号部分
    s1 = (coord1 >= 0).to(tl.int32) * 2 - 1
    s2 = (coord2 >= 0).to(tl.int32) * 2 - 1
    sg = (s1 * s2).to(tl.float32)

    # 绝对值 XOR
    xorv = tl.abs(coord1) ^ tl.abs(coord2)

    # 替换 LUT：log2 计算
    val = (xorv + 1).to(tl.float32)
    exp = tl.floor(tl.log2(val)) + 1.0
    s = exp / H

    return sg * (1.0 - s)



@triton.jit
def _loom_kernel(
    # in
    sta_loc_ptr,         # int64 [B,T,D]
    cnc_loc_ptr,         # int64 [B,T,C,D]
    logits_ptr,          # fp32  [B,T,T]
    x_norm_ptr,          # fp32  [B,T,T]
    lg_ptr,              # int64 [B,T]
    mask_ptr,            # bool  [B,T,T]
    
    # out
    sel_ptr,             # int64 [B,T,D]
    loss_b_ptr,          # fp32  [B]  — 原地累加，每(b,t)加上min_{c} loss的D维和
    
    
    # strides (elements)
    sta_sb: tl.int32, sta_st: tl.int32, sta_sd: tl.int32,
    cnc_sb: tl.int32, cnc_st: tl.int32, cnc_sc: tl.int32, cnc_sd: tl.int32,
    log_sb: tl.int32, log_st: tl.int32, log_st2: tl.int32,
    x_norm_sb: tl.int32, x_norm_st: tl.int32, x_norm_st2: tl.int32,
    lg_sb:  tl.int32, lg_st:  tl.int32,
    msk_sb: tl.int32, msk_st: tl.int32, msk_st2: tl.int32,
    sel_sb: tl.int32, sel_st: tl.int32, sel_sd: tl.int32,

    # tiling
    BLOCK_T2: tl.constexpr,   # e.g. 64
    BLOCK_C:  tl.constexpr,   # e.g. 64
    BLOCK_D:  tl.constexpr,   # must be >= D; for D=3 set 4 or 8
    T: tl.constexpr, C: tl.constexpr, D: tl.constexpr, TP: tl.constexpr, H: tl.constexpr,
    loss_calc: tl.constexpr
):
    # pdb.set_trace()
    
    b = tl.program_id(0)
    t = tl.program_id(1)

    # d vector in register
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    # load sta_loc(b,t, d)
    sta_t_ptr = sta_loc_ptr + (b * sta_sb + t * sta_st + d_off * sta_sd)
    sta_t = tl.load(sta_t_ptr, mask=d_mask, other=0)

    # accumulators for running-argmin over C for each d
    best_val = tl.full((BLOCK_D,), 1e30, tl.float32)
    best_idx = tl.full((BLOCK_D,), -1, tl.int32)

    # pre-load lg(b,t)
    lg_bt = tl.load(lg_ptr + b * lg_sb + t * lg_st).to(tl.float32)
    tp_f = tl.full((1,), TP, tl.float32)

    # loop over C in tiles
    for c0 in range(0, C, BLOCK_C):
        c_off = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_off < C

        # cnc_loc(b,t,c,d)  -> shape (BLOCK_D, BLOCK_C)
        cnc_ptr = cnc_loc_ptr + (b * cnc_sb + t * cnc_st + c_off[None, :] * cnc_sc + d_off[:, None] * cnc_sd)
        cnc_dc = tl.load(cnc_ptr, mask=(d_mask[:, None] & c_mask[None, :]), other=0)

        # partial sum over t2 for |delt|, one accumulator per (d,c)
        part = tl.zeros((BLOCK_D, BLOCK_C), tl.float32)

        # loop over t2 in tiles
        for t20 in range(0, T, BLOCK_T2):
            t2_off = t20 + tl.arange(0, BLOCK_T2)
            t2_mask = t2_off < T

            # load sta_loc(b,t2,d) -> (BLOCK_D, BLOCK_T2)
            sta_t2_ptr = sta_loc_ptr + (b * sta_sb + t2_off[None, :] * sta_st + d_off[:, None] * sta_sd)
            sta_t2 = tl.load(sta_t2_ptr, mask=(d_mask[:, None] & t2_mask[None, :]), other=0)

            # load x_norm(b,t,t2) -> (BLOCK_T2,)
            xnm_ptr = x_norm_ptr + (b * x_norm_sb + t * x_norm_st + t2_off * x_norm_st2)
            xnm_t2 = tl.load(xnm_ptr, mask=t2_mask, other=0.0)
            
            # dis_sta_pos(b,t,t2,d) = dist(sta_t[d], sta_t2[d])
            dis_t_t2_d = _get_distance_kernel(
                tl.broadcast_to(sta_t[:, None], (BLOCK_D, BLOCK_T2)),
                sta_t2,
                H
            )
            # sum over D -> [T2]
            dis_sum_t2 = tl.sum(dis_t_t2_d, axis=0)
            dis_sum_t2 = dis_sum_t2 * xnm_t2

            # distance to cnc: dist(sta_t2[d, T2], cnc_dc[d, C])
            # result shape (D, T2, C) but we stream-reduce along T2
            # compute in two mat-broadcasts without materializing full tensor
            # logits_ct = (dist - dis_t_t2_d + dis_sum_t2) / TP
            # delt = logits_ct - logits[b,t,t2]
            # apply mask[b,t,t2]

            # load logits and mask once per T2 tile
            log_ptr = logits_ptr + (b * log_sb + t * log_st + t2_off * log_st2)
            logits_t2 = tl.load(log_ptr, mask=t2_mask, other=0.0)

            msk_ptr = mask_ptr + (b * msk_sb + t * msk_st + t2_off * msk_st2)
            msk_t2 = tl.load(msk_ptr, mask=t2_mask, other=0).to(tl.int1)

            # 矢量化处理整个 T2-tile
            # sta_t2: (D, BLOCK_T2)
            # dis_t_t2_d: (D, BLOCK_T2)
            # dis_sum_t2: (BLOCK_T2,)
            # logits_t2: (BLOCK_T2,)
            # msk_t2: (BLOCK_T2,) 0/1
            # cnc_dc: (D, C)

            sta_bc  = tl.broadcast_to(sta_t2[:, :, None], (BLOCK_D, BLOCK_T2, BLOCK_C))
            cnc_bc  = tl.broadcast_to(cnc_dc[:, None, :], (BLOCK_D, BLOCK_T2, BLOCK_C))

            dist_dtc = _get_distance_kernel(sta_bc, cnc_bc, H) # (D, T2, C)
            dist_dtc = dist_dtc * xnm_t2[None, :, None]

            dis_pos_bc = tl.broadcast_to(dis_t_t2_d[:, :, None], (BLOCK_D, BLOCK_T2, BLOCK_C))
            dis_sum_bc = tl.broadcast_to(dis_sum_t2[None, :, None], (BLOCK_D, BLOCK_T2, BLOCK_C))
            log_bc     = tl.broadcast_to(logits_t2[None, :, None], (BLOCK_D, BLOCK_T2, BLOCK_C))
            mask_bc    = tl.broadcast_to(msk_t2[None, :, None], (BLOCK_D, BLOCK_T2, BLOCK_C))

            logits_ct = (dist_dtc - dis_pos_bc + dis_sum_bc) / tp_f
            delt      = logits_ct - log_bc

            # 应用 mask，累加到 part
            delt = tl.where(mask_bc, delt, 0.0)
            if loss_calc == 0:
                part += tl.sum(tl.abs(delt), axis=1)  # sum over T2
            else:
                part += tl.sum(delt * delt, axis=1)


        # normalize by lg(b,t)
        part = part / lg_bt

        # argmin across C-tile, maintain global best
        # per-d tile min
        tile_min = tl.min(part, axis=1)                     # (D,)
        # indices inside tile (first occurrence)
        is_min = part == tile_min[:, None]
        # take first index by converting mask to int and doing argmax on reversed cummax
        # simpler: reduce by tl.where to pick the first true
        idx_local = tl.argmax(is_min, axis=1)               # (D,)
        val_new = tile_min
        idx_new = (c0 + idx_local).to(tl.int32)

        # update when strictly better
        better = val_new < best_val
        best_val = tl.where(better, val_new, best_val)
        best_idx = tl.where(better, idx_new, best_idx)

    # write selected_locs(b,t,d) from cnc_loc using best_idx[d]
    # gather per d
    gather_ptr = cnc_loc_ptr + (b * cnc_sb + t * cnc_st + best_idx[None, :] * cnc_sc + d_off[:, None] * cnc_sd)
    # diagonal load (d,d) -> use arange and take diag
    sel_vals = tl.load(gather_ptr, mask=(d_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] == tl.arange(0, BLOCK_D)[:, None])), other=0)
    # sel_vals is square with only diagonal valid, sum rows to get (D,)
    sel_vals = tl.sum(sel_vals, axis=1)

    out_ptr = sel_ptr + (b * sel_sb + t * sel_st + d_off * sel_sd)
    tl.store(out_ptr, sel_vals, mask=d_mask)

    # accumulate per-b loss: sum_d best_val[d]
    loss_add = tl.sum(best_val, axis=0)
    tl.atomic_add(loss_b_ptr + b, loss_add)


def triton_loom_wrapper(
    sta_loc: torch.Tensor,       # int64 [B,T,D]
    cnc_loc: torch.Tensor,       # int64 [B,T,C,D]
    logits:  torch.Tensor,       # fp32  [B,T,T]
    x_norm:  torch.Tensor,       # fp32  [B,T,T]
    lg:      torch.Tensor,       # int64 [B,T]
    mask:    torch.Tensor,       # bool  [B,T,T]
) -> Tuple[torch.Tensor, torch.Tensor]:
    now_batch_size = sta_loc.shape[0]
    
    assert sta_loc.is_cuda and cnc_loc.is_cuda and logits.is_cuda and lg.is_cuda and mask.is_cuda
    selected_locs = torch.empty((now_batch_size, block_size+vocab_size, tp), dtype=torch.int64, device=sta_loc.device)
    loss_b = torch.zeros((now_batch_size,), dtype=torch.float32, device=sta_loc.device)

    # print(cnc_loc.shape[2])
    # assert cnc_loc.shape[2] == 2*h*int(c*h//division_fact)+1
    
    # exit(0)
    
    # strides (in elements)
    sta_sb, sta_st, sta_sd = sta_loc.stride()

    cnc_sb, cnc_st, cnc_sc, cnc_sd = cnc_loc.stride()
    log_sb, log_st, log_st2 = logits.stride()
    x_norm_sb, x_norm_st, x_norm_st2 = x_norm.stride()
    lg_sb, lg_st = lg.stride()
    msk_sb, msk_st, msk_st2 = mask.stride()
    sel_sb, sel_st, sel_sd = selected_locs.stride()

    grid = (now_batch_size, block_size+vocab_size)
    _loom_kernel[grid](
        sta_loc, cnc_loc, logits, x_norm, lg, mask,
        selected_locs, loss_b,
        sta_sb, sta_st, sta_sd,
        cnc_sb, cnc_st, cnc_sc, cnc_sd,
        log_sb, log_st, log_st2,
        x_norm_sb, x_norm_st, x_norm_st2,
        lg_sb, lg_st,
        msk_sb, msk_st, msk_st2,
        sel_sb, sel_st, sel_sd,
        BLOCK_T2=64, BLOCK_C=32, BLOCK_D=2,
        T=block_size+vocab_size, C=2*h*int(factor*h//division_fact)+1, D=tp, TP=tp, H=h,
        loss_calc=1,
        num_warps=8, num_stages=4,
        
    )
    # 将累加的和变为“平均每(b)的 min 损失”：mean over (T*D)
    loss_b = loss_b / float((block_size+vocab_size) * tp)
    # print(f"s: {selected_locs.device}", selected_locs.dtype, selected_locs.shape)
    # print(f"l: {loss_b.device}", loss_b.dtype, loss_b.shape)
    return selected_locs, loss_b

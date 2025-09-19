import os
import queue
import sys
import threading
import time
from datetime import datetime
from time import sleep

import matplotlib.pyplot as plt
import torch
# torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
from matplotlib.colors import PowerNorm
from torch.nn import functional as F
from tqdm import tqdm

from src.gettime import CUDATimer, gettime, mark
from src.loom_kernel import ct_val_triton, kernel_ct_val_fused_cd
from src.loom_kernel_full import ct_loss_triton
from src.loss import *
from src.para import *
from src.sampler import *
from src.utils import *
from src.vis import visualize_pair_bihclust, visualize_similarity

main_device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
from typing import List, Tuple


class CritiGraph(torch.nn.Module):
    def __init__(self):
        super().__init__() 
    
        ### 1. 基本参数
        self.h : int = h
        self.tp: int = tp
        self.n : int = n
        
        ### 2. CT Space Embeddings 初始化
        self.train_locations = torch.randint(
            1 - self.n, self.n, (N_train, self.tp), 
            dtype=torch.int64, device=main_device, generator=generators[0]
        )
        self.vocab_locations = torch.randint(
            1 - self.n, self.n, (N_vocab, self.tp), 
            dtype=torch.int64, device=main_device, generator=generators[0]
        )
        self.valid_locations = torch.randint(
            1 - self.n, self.n, (N_valid, self.tp), 
            dtype=torch.int64, device=main_device, generator=generators[0]
        )

        ### 3. 额外参数  
        self.timer = CUDATimer()
    
    
    def connection_masks(self, sz, dev_num=0):
        device = devices[dev_num] if dev_num >= 0 else 'cpu'

        upper_bounds   = 2 ** torch.arange(self.h, dtype=torch.int64, device=device)
        random_numbers = torch.randint(
            0, self.n, 
            (self.h, sz, N_K, self.tp), 
            dtype=torch.int64, device=device, generator=generators[dev_num]
        ) # (H, B*T, K, D)
        masks = random_numbers & (upper_bounds.view(-1, 1, 1, 1) - 1)
        # masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3) # (B*T, H, K, D)
    
    def connection(self, ori_int: torch.Tensor, dev_num=0):
        device = devices[dev_num] if dev_num >= 0 else 'cpu'
        
        flip_masks = (1 << torch.arange(self.h, device=device, dtype=ori_int.dtype)).unsqueeze(0).unsqueeze(2)
        flipped_ints = ori_int.unsqueeze(1) ^ flip_masks # (B*T1, H, D)
        random_masks = self.connection_masks(flipped_ints.size(0), dev_num=dev_num)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h * N_K, self.tp)
        # (B*T1, H, 1, D) ^ (B*T1, H, K, D) -> (B*T1, H*K, D)
        loc = torch.cat((result, ori_int.unsqueeze(1), -result), dim=1) # (B*T1, H*K + 1 + H*K, D)
        return loc
              
    def converge_mask(self, size0: int, size1: int, dev_num=0):
        device  = devices[dev_num]
        mask   = torch.ones((size0, size1), dtype=torch.bool, device=device)
        
        if not self.converge:
            choosing_mask = (torch.rand((size0), device=device) > 0.2)  
            sel_idx = (~choosing_mask).nonzero(as_tuple=False).squeeze(1)
            if sel_idx.numel() > 0:
                mask[sel_idx, :] = False
                rows = sel_idx.repeat_interleave(sample_k)
                cols = torch.randint(0, size1, (rows.numel(),), device=device, generator=generators[dev_num])
                mask[rows, cols] = True

        return mask


    def get_best_loc(
        self, 
        cnc_loc : torch.Tensor, # (T, C, D)
        loss_cos: torch.Tensor, 
        loss_cro: torch.Tensor,
        loss_tot: torch.Tensor,
        sid     : int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        device = devices[sid]
        T      = cnc_loc.size(0)
        
        # step1: 每个样本在 [0, tp) 之间随机生成一个排列（通过排序 trick）
        rand_vals     = torch.rand((T, self.tp), device=device, generator=generators[sid]) # (T, D)
        rand_cols     = rand_vals.argsort(dim=1)[:, :cur_tp]           # (T, cur_tp)
        # print(rand_cols[:4, :2])
        
        # step2: 每个 (t, dim) 的 argmin
        argmin_all    = torch.argmin(loss_tot, dim=1)                  # (T, D)

        # step3: 初始化为 "保持不更新"
        cnc_indices   = torch.full_like(argmin_all, N_K * self.h)   # (T, D)

        # step4: 高级索引直接填充
        t_idx = torch.arange(T, device=loss_tot.device)[:, None].expand(T, cur_tp)  # (T, upd)
        t_mask = torch.rand(T, device=device, generator=generators[sid]) < cur_portion  # (T,)
        t_idx, rand_cols = t_idx[t_mask], rand_cols[t_mask]  # (T_upd, ), (T_upd, upd)
        
        cnc_indices[t_idx, rand_cols] = argmin_all[t_idx, rand_cols]

        

        T_indices     = torch.arange(T              ,    device=device)[:, None]         # (T, 1)
        dim_indices   = torch.arange(self.tp    ,        device=device)[None  :]         # (1, D)
        selected_locs = cnc_loc [T_indices, cnc_indices,  dim_indices]                   # (T, D)
        
        real_loss_cos = loss_cos[T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, )
        real_loss_cro = loss_cro[T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, )
        real_loss_tot = loss_tot[T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, ) 
        
        return selected_locs, real_loss_cos, real_loss_cro, real_loss_tot


    def cos_similarity(self, coord1: torch.Tensor, coord2: torch.Tensor) -> torch.Tensor:
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        xor_result = torch.abs(coord1) ^ torch.abs(coord2)
        _, exp = torch.frexp((xor_result + 1).to(torch.float32))
        s = exp.float() / self.h
        return sg * (1 - s)
    

    def loom_train(
        self, 
        cur_tar: torch.Tensor,   # (T_train, )
        
        cnc_loc: torch.Tensor,   # (T_train, N_C    , dim_ct)
        
        sta_loc: torch.Tensor,   # (T_train, dim_ct)
        ttn_loc: torch.Tensor,   # (T_train, N_ttnbr, dim_ct)
        voc_loc: torch.Tensor,   # (T_train, K_vocab, dim_ct)

        sta_emb: torch.Tensor,   # (T_train, dim_eu)
        ttn_emb: torch.Tensor,   # (T_train, N_ttnbr, dim_eu)
        voc_emb: torch.Tensor,   # (T_train, K_vocab, dim_eu)
        
        mask   : torch.Tensor,   # (T_train, N_ttnbr + N_tvnbr)
        sid    : int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor,]:
        device  = devices[sid]
        pos_loc = torch.cat([ttn_loc, voc_loc], dim=1)  # (T_train, N_ttnbr + K_vocab, dim_ct)
        pos_emb = torch.cat([ttn_emb, voc_emb], dim=1)  # (T_train, N_ttnbr + K_vocab, dim_eu)
        
        ### step 1: 计算欧氏空间的相似度
        eu_val = 20. * temperature * (sta_emb[:, None, :] @ pos_emb.transpose(-1, -2)).squeeze()  
            ### (T_train, 1, dim_eu) @ (T_train, dim_eu, N_ttnbr + K_vocab) -> (T_train, 1, N_ttnbr + K_vocab) -> (T_train, N_ttnbr + K_vocab)
        
        # if use_eu_norm == True:
        #     eu_nrm = (torch.norm(sta_emb[:, None, :], dim=-1, keepdim=True) @ torch.norm(pos_emb, dim=-1, keepdim=True).transpose(-1, -2)).squeeze()  
        #     ### (T_train, 1, 1) @ (T_train, 1, N_ttnbr + K_vocab) -> (T_train, 1, N_ttnbr + K_vocab) -> (T_train, N_ttnbr + K_vocab)
        # else:
        #     eu_nrm = torch.ones((T_train, N_ttnbr + K_vocab), device=device) * 20.
        

        ### step 2: 计算 CT 空间的相似度
        cos_sta_pos     = self.cos_similarity(
            sta_loc[:, None, :],    pos_loc[:, :, :]      
        )                                        # (T_train, N_ttnbr + K_vocab,      dim_ct)
        cos_sta_pos_sum = cos_sta_pos.sum(dim=-1) 
                                                 # (T_train, N_ttnbr + K_vocab             )
        # cos_cnc_pos     = self.cos_similarity(
        #     cnc_loc[:, None, :, :], pos_loc[:, :, None,:]
        # )                                        # (T_train, N_trnbr + K_vocab, N_C, dim_ct)
        # ct_val          = (
        #     cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + cos_cnc_pos
        # ) / self.tp * eu_nrm[:, :, None, None]         
        
        ct_val = ct_val_triton(
            cnc_loc.to(torch.int32).contiguous(),
            pos_loc.to(torch.int32).contiguous(),
            torch.ones((T_train, N_ttnbr + K_vocab), device=device, dtype=torch.float32).contiguous(),
            cos_sta_pos.contiguous(),
            cos_sta_pos_sum.contiguous(),
            tp=float(self.tp),
            h=float(self.h),
            out=None,                # 或传入你复用的 out 缓冲
            BLOCK_S=32,
            BLOCK_CD=32,
            NUM_WARPS=8,
            NUM_STAGES=2,
        ) * 20. * temperature
        
        
        # (T_train, N_ttnbr + K_vocab, N_C, dim_ct)
        ## 对于 T 个 starting point，向 S_tot 个 positive sample 连边。此时，我们把其中某个 positive sample 替换为 connected sample，共有 C 个；此时，D 个维度上的的距离是多少？

        ### step 3: 计算 loss
        ### step 3.1: 计算 dyn-dyn cross-entropy loss
        logits_ct_dyn    = ct_val[:, 0:N_ttnbr]                 # (T_train, N_ttnbr, N_C, dim_ct)
        logits_eu_dyn    = eu_val[:, 0:N_ttnbr]                 # (T_train, N_ttnbr)
        loss_dyn_dyn     = (
            F.softmax(logits_eu_dyn, dim=1)[..., None, None] * (
                F.log_softmax(logits_eu_dyn, dim=1)[..., None, None] - 
                F.log_softmax(logits_ct_dyn, dim=1)
            )
        ).sum(dim=1)                                            # (T_train, N_C, dim_ct)  
                                                             

        ### step 3.2: 计算 dyn-sta cross-entropy loss
        logits_ct_sta    = ct_val[:, N_ttnbr:] / temperature                 # (T_train, K_vocab, N_C, dim_ct)
        # 使用 F.log_softmax 和 cur_tar 计算交叉熵
        # logits_ct_sta: (T_train, K_vocab, N_C, dim_ct)
        # cur_tar: (T_train,)
        # 目标是对每个样本，取 cur_tar 作为 target，计算 cross entropy
        # 先将 logits_ct_sta 变为 (T_train, K_vocab, N_C*dim_ct)，再对 K_vocab 做 softmax
        logits_flat = logits_ct_sta.reshape(T_train, K_vocab, -1)  # (T_train, K_vocab, N_C*dim_ct)
        log_probs = F.log_softmax(logits_flat, dim=1)              # (T_train, K_vocab, N_C*dim_ct)
        # cur_tar 作为 target，gather
        cur_tar_exp = cur_tar.view(-1, 1, 1).expand(-1, 1, N_C*self.tp)  # (T_train, 1, N_C*dim_ct)
        target_log_probs = torch.gather(log_probs, 1, cur_tar_exp).squeeze(1)  # (T_train, N_C*dim_ct)
        loss_dyn_sta = -target_log_probs.reshape(T_train, N_C, self.tp)  # (T_train, N_C, dim_ct)

        return loss_dyn_dyn, loss_dyn_sta
    
    '''     
    def loom_vocab_cro(
        self, 
        cur_tar: torch.Tensor,   # (T_vonbr, )
        
        voc_loc: torch.Tensor,   # (N_vocab, dim_ct)
        cnc_loc: torch.Tensor,   # (N_vocab, N_C    , dim_ct)
        nei_loc: torch.Tensor,   # (T_vonbr, dim_ct)    
        
        voc_emb: torch.Tensor,   # (N_vocab, dim_eu)
        nei_emb: torch.Tensor,   # (T_vonbr, dim_eu)
        
        sid: int
    ) -> torch.Tensor:
        device = devices[sid]
    
        ### step 1: 计算欧式空间的 norm
        if use_eu_norm == True:
            eu_cro_nrm = (torch.norm(nei_emb, dim=-1, keepdim=True) @ torch.norm(voc_emb, dim=-1, keepdim=True).t()).squeeze()  # (T_vonbr, 1) @ (1, N_vocab) -> (T_vonbr, N_vocab,) -> (T_vonbr, N_vocab)
        else:
            eu_cro_nrm = torch.ones((T_vonbr, N_vocab), device=device) * 20.

        ### step 2: 计算 CT 空间的相似度
        cos_nei_voc     = self.cos_similarity(nei_loc[:, None, :], voc_loc[None, :, :])  # (T_vonbr, N_vocab, dim_ct)
        logits_ori      = cos_nei_voc.mean(dim=-1) * eu_cro_nrm                          # (T_vonbr, N_vocab)
        
        cos_nei_voc_sum = cos_nei_voc.sum(dim=-1)                                        # (T_vonbr, N_vocab)
        cos_nei_cnc = self.cos_similarity(nei_loc[:, None, None, :], cnc_loc[None, :, :, :])  
                                                                                         # (T_vonbr, N_vocab, N_C, dim_ct)
        logits_upd      = (
            cos_nei_voc_sum[:, :, None, None] - cos_nei_voc[:, :, None, :] + cos_nei_cnc
        ) / self.tp * eu_cro_nrm[:, :, None, None]                                       # (T_vonbr, N_vocab, N_C, dim_ct)
        
        ### 数值稳定性
        m = torch.maximum(
            logits_ori[:, :, None, None], 
            logits_upd
        ).max(dim=1).values[:, None, :, :]                                               # (T_vonbr, 1,       N_C, dim_ct)

        exp_ori         = torch.exp(logits_ori[:, :, None, None] - m)                    # (T_vonbr, N_vocab, N_C, dim_ct)
        exp_upd         = torch.exp(logits_upd                   - m)                    # (T_vonbr, N_vocab, N_C, dim_ct)
        
        sum_exp_ori     = exp_ori.sum(dim=1, keepdim=True)                               # (T_vonbr, 1, N_C, dim_ct)
        log_sum_exp     = m + torch.log(
            sum_exp_ori - exp_ori + exp_upd + 1e-9
        )                                                                                # (T_vonbr, N_vocab, N_C, dim_ct)
        
        mask            = (cur_tar[:, None] == torch.arange(N_vocab, device=device)[None, :])[:, :, None, None]  
                                                                                         # (T_vonbr, N_vocab, 1  , 1)
        targets_val     = (
            torch.gather(logits_ori, dim=1, index=cur_tar[:, None])[:, :, None, None] * (~mask) +
            logits_upd * mask
        )   # (T_vonbr, N_vocab, N_C, dim_ct)
        
        loss_cro        = (-targets_val + log_sum_exp).sum(dim=0)                         # (N_vocab, N_C, dim_ct)
        
        return loss_cro
    '''
    

    def loom_vocab(
        self, 
        cnc_loc: torch.Tensor,   # (T_vocab, N_C    , dim_ct)
        
        sta_loc: torch.Tensor,   # (T_vocab, dim_ct)
        pos_loc: torch.Tensor,   # (T_vocab, N_vvnbr, dim_ct)

        sta_emb: torch.Tensor,   # (T_vocab, dim_eu)
        pos_emb: torch.Tensor,   # (T_vocab, N_vvnbr, dim_eu)
        
        mask   : torch.Tensor,   # (T_vocab, N_vtnbr + N_vvnbr)
        sid    : int = 0,
    ) -> torch.Tensor:
        device = devices[sid]
        
        ### step 1: 计算欧氏空间的相似度
        eu_val = 20. *  temperature * (sta_emb[:, None, :] @ pos_emb.transpose(-1, -2)).squeeze()  
            ### (T_vocab, 1, dim_eu) @ (T_vocab, dim_eu, N_vvnbr) -> (T_vocab, 1,  N_vvnbr) -> (T_vocab, N_vvnbr)
        # eu_nrm = (torch.norm(sta_emb[:, None, :], dim=-1, keepdim=True) @ torch.norm(pos_emb, dim=-1, keepdim=True).transpose(-1, -2)).squeeze() 
            ### (T_vocab, 1, 1) @ (T_vocab, 1, N_vvnbr) -> (T_vocab, 1,  N_vvnbr) -> (T_vocab, N_vvnbr)

        ### step 2: 计算 CT 空间的相似度
        cos_sta_pos     = self.cos_similarity(
            sta_loc[:, None, :],    pos_loc[:, :, :]      
        )                                        # (T_vocab, N_vvnbr,      dim_ct)
        cos_sta_pos_sum = cos_sta_pos.sum(dim=-1) 
                                                 # (T_vocab, NN_vvnbr             )
        # cos_cnc_pos     = self.cos_similarity(
        #     cnc_loc[:, None, :, :], pos_loc[:, :, None,:]
        # )                                        # (T_vocab, N_vvnbr, N_C, dim_ct)
        # ct_val          = (
        #     cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + cos_cnc_pos
        # ) / self.tp          

        ct_val = ct_val_triton(
            cnc_loc.to(torch.int32).contiguous(),
            pos_loc.to(torch.int32).contiguous(),
            torch.ones((T_vocab, N_vvnbr), device=device, dtype=torch.float32).contiguous(),
            cos_sta_pos.contiguous(),
            cos_sta_pos_sum.contiguous(),
            tp=float(self.tp),
            h=float(self.h),
            out=None,                # 或传入你复用的 out 缓冲
            BLOCK_S=32,
            BLOCK_CD=32,
            NUM_WARPS=8,
            NUM_STAGES=2,
        ) * 20. * temperature
        # (T_vocab, N_vvnbr, N_C, dim_ct)


        ### step 3: 计算 dyn-dyn cross-entropy loss
        logits_ct_dyn    = ct_val                 # (T_vocab, N_vvnbr, N_C, dim_ct)
        logits_eu_dyn    = eu_val                 # (T_vocab, N_vvnbr)
        loss_dyn_dyn     = (
            F.softmax(logits_eu_dyn, dim=1)[..., None, None] * (
                F.log_softmax(logits_eu_dyn, dim=1)[..., None, None] - 
                F.log_softmax(logits_ct_dyn, dim=1)
            )
        ).sum(dim=1)                                       # (T_train, N_C, dim_ct)  
    
        return loss_dyn_dyn
    
    
    def loom_valid(
        self, 
        cnc_loc: torch.Tensor,   # (T_valid, N_C    , dim_ct)
        
        sta_loc: torch.Tensor,   # (T_valid, dim_ct)
        pos_loc: torch.Tensor,   # (T_valid, N_vanbr, dim_ct)

        sta_emb: torch.Tensor,   # (T_valid, dim_eu)
        pos_emb: torch.Tensor,   # (T_valid, N_vanbr, dim_eu)
        
        mask   : torch.Tensor,   # (T_valid, N_vanbr)
        sid    : int = 0,
    ) -> torch.Tensor:
        device = devices[sid]
        
        
        ### step 1: 计算欧氏空间的相似度
        eu_val = 20. * temperature * (sta_emb[:, None, :] @ pos_emb.transpose(-1, -2)).squeeze()  
            ### (T_valid, 1, dim_eu) @ (T_valid, dim_eu, N_vanbr) -> (T_valid, 1, N_vanbr) -> (T_valid, N_vanbr)
        # eu_nrm = (torch.norm(sta_emb[:, None, :], dim=-1, keepdim=True) @ torch.norm(pos_emb, dim=-1, keepdim=True).transpose(-1, -2)).squeeze()  
            ### (T_valid, 1, 1) @ (T_valid, 1, N_vanbr) -> (T_valid, 1, N_vanbr) -> (T_valid, N_vanbr)

        ### step 2: 计算 CT 空间的相似度
        cos_sta_pos     = self.cos_similarity(
            sta_loc[:, None, :],    pos_loc[:, :, :]      
        )                                        # (T_valid, N_vanbr, dim_ct)
        cos_sta_pos_sum = cos_sta_pos.sum(dim=-1) 
                                                 # (T_valid, N_vanbr)
        # cos_cnc_pos     = self.cos_similarity(
        #     cnc_loc[:, None, :, :], pos_loc[:, :, None,:]
        # )                                        # (T_valid, N_vanbr, N_C, dim_ct)
        # ct_val          = (
        #     cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + cos_cnc_pos
        # ) / self.tp          

        ct_val = ct_val_triton(
            cnc_loc.to(torch.int32).contiguous(),
            pos_loc.to(torch.int32).contiguous(),
            torch.ones((T_valid, N_vanbr), device=device, dtype=torch.float32).contiguous(),
            cos_sta_pos.contiguous(),
            cos_sta_pos_sum.contiguous(),
            tp=float(self.tp),
            h=float(self.h),
            out=None,                # 或传入你复用的 out 缓冲
            BLOCK_S=32,
            BLOCK_CD=32,
            NUM_WARPS=8,
            NUM_STAGES=2,
        ) * 20. * temperature
         # (T_valid, N_vanbr, N_C, dim_ct)


        ### step 3: 计算 dyn-dyn cross-entropy loss
        logits_ct_dyn    = ct_val                 # (T_valid, N_vanbr, N_C, dim_ct)
        logits_eu_dyn    = eu_val                 # (T_valid, N_vanbr)
        loss_dyn_dyn     = (
            F.softmax(logits_eu_dyn, dim=1)[..., None, None] * (
                F.log_softmax(logits_eu_dyn, dim=1)[..., None, None] - 
                F.log_softmax(logits_ct_dyn, dim=1)
            )
        ).sum(dim=1)                                       # (T_valid, N_C, dim_ct)  
    
        return loss_dyn_dyn
    
    def train_train_blocks(self, sid: int):
        device = devices[sid]
        
        for train_block_id in train4sid[sid]:
            train_block = train_blocks[train_block_id]
            train_slice = slice(train_block[0], train_block[1])
            
            ### step.1 准备数据
            with torch.cuda.device(0), torch.cuda.stream(data_streams[sid]):
                _cur_tar = self.train_tar[train_slice]                      # (T_train, )
                _ttn_idx = self.train_sampler.get_cos_connection(train_block, "train_train")   # (T_train, N_ttnbr)
                _voc_idx = self.train_sampler.get_cro_connection(_cur_tar)                     # (T_train, K_vocab)

                _sta_loc = self.train_locations[train_slice]                # (T_train, dim_ct)
                _ttn_loc = self.train_locations[_ttn_idx]                   # (T_train, N_ttnbr, dim_ct)
                _voc_loc = self.vocab_locations[_voc_idx]                   # (T_train, K_vocab, dim_ct)

                _sta_emb = self.train_emb[train_slice]                      # (T_train, dim_eu)
                _ttn_emb = self.train_emb[_ttn_idx]                         # (T_train, N_ttnbr, dim_eu)
                _voc_emb = self.train_emb[_voc_idx]                         # (T_train, K_vocab, dim_eu)
                
                data_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                data_ready_event.record(data_streams[sid])

            ### step.2 计算 loss 并选择最佳位置
            with torch.cuda.device(device), torch.cuda.stream(defa_streams[sid]):
                defa_streams[sid].wait_event(data_ready_event)
                
                cur_tar, sta_loc, ttn_loc, voc_loc, sta_emb, ttn_emb, voc_emb = (
                    _cur_tar.to(device, non_blocking=True),
                    _sta_loc.to(device, non_blocking=True),
                    _ttn_loc.to(device, non_blocking=True),
                    _voc_loc.to(device, non_blocking=True),
                    _sta_emb.to(device, non_blocking=True),
                    _ttn_emb.to(device, non_blocking=True),
                    _voc_emb.to(device, non_blocking=True),
                )
                
                cnc_loc   = self.connection(sta_loc, dev_num=sid)
                mask      = self.converge_mask(T_train, N_ttnbr, sid)

                loss_dyn_dyn, loss_dyn_sta = self.loom_train(
                    cur_tar,
                    cnc_loc,
                    sta_loc, ttn_loc, voc_loc,
                    sta_emb, ttn_emb, voc_emb,
                    mask, sid
                )
                
                loss_tot = loss_strategy['train_ratio_cos'] * loss_dyn_dyn + loss_strategy['train_ratio_cro'] * loss_dyn_sta
                selected_locs, loss_cos_T, loss_cro_T, loss_tot_T  = self.get_best_loc(cnc_loc, loss_dyn_dyn, loss_dyn_sta, loss_tot, sid) # (T_train, ) * 3, (T_train, dim_ct)
                
                comp_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                comp_ready_event.record(defa_streams[sid])
            
            with torch.cuda.device(0), torch.cuda.stream(data_streams[sid]):
                data_streams[sid].wait_event(comp_ready_event)
                
                self.new_train_locations[train_slice] = selected_locs.to(main_device, non_blocking=True)
                self.train_loss_cos_buf [train_slice] = loss_cos_T.to(main_device, non_blocking=True)
                self.train_loss_cro_buf [train_slice] = loss_cro_T.to(main_device, non_blocking=True)
                self.train_loss_tot_buf [train_slice] = loss_tot_T.to(main_device, non_blocking=True)

    def train_vocab_blocks(self, sid: int):
        device = devices[sid]

        for vocab_block_id in vocab4sid[sid]:
            vocab_block = vocab_blocks[vocab_block_id]
            vocab_slice = slice(vocab_block[0], vocab_block[1])
            
            ### step.1 准备数据
            with torch.cuda.device(0), torch.cuda.stream(data_streams[sid]):
                _vvn_idx = self.vocab_sampler.get_cos_connection(vocab_block, "vocab_vocab")   # (T_vocab, N_vvnbr)

                _sta_loc = self.vocab_locations[vocab_slice]                 # (T_vocab, dim_ct)
                _vvn_loc = self.vocab_locations[_vvn_idx]                    # (T_vocab, N_vvnbr, dim_ct)

                _sta_emb = self.vocab_emb[vocab_slice]                       # (T_vocab, dim_eu)
                _vvn_emb = self.vocab_emb[_vvn_idx]                          # (T_vocab, N_vvnbr, dim_eu)
                
                data_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                data_ready_event.record(data_streams[sid])

            ### step.2 计算 loss 并选择最佳位置
            with torch.cuda.device(device), torch.cuda.stream(defa_streams[sid]):
                defa_streams[sid].wait_event(data_ready_event)
                
                sta_loc, vvn_loc, sta_emb, vvn_emb = (
                    _sta_loc.to(device, non_blocking=True),
                    _vvn_loc.to(device, non_blocking=True),
                    _sta_emb.to(device, non_blocking=True),
                    _vvn_emb.to(device, non_blocking=True),
                )
                
                cnc_loc   = self.connection(sta_loc, dev_num=sid)
                mask      = self.converge_mask(T_vocab,  N_vvnbr, sid)

                loss_cos  = self.loom_vocab(
                    cnc_loc,
                    sta_loc, vvn_loc,
                    sta_emb, vvn_emb,
                    mask, sid
                )
                
                selected_locs, loss_cos_T, _, _  = self.get_best_loc(cnc_loc, loss_cos, loss_cos, loss_cos, sid) # (T_vocab, ) * 3, (T_vocab, dim_ct)
                
                comp_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                comp_ready_event.record(defa_streams[sid])
            
            with torch.cuda.device(0), torch.cuda.stream(data_streams[sid]):
                data_streams[sid].wait_event(comp_ready_event)
                
                self.new_vocab_locations[vocab_slice] = selected_locs.to(main_device, non_blocking=True)
                self.vocab_loss_cos_buf [vocab_slice] = loss_cos_T.to(main_device, non_blocking=True)
    
    def train_valid_blocks(self, sid: int):
        device = devices[sid]

        for valid_block_id in valid4sid[sid]:
            valid_block = valid_blocks[valid_block_id]
            valid_slice = slice(valid_block[0], valid_block[1])
            
            ### step.1 准备数据
            with torch.cuda.device(0), torch.cuda.stream(data_streams[sid]):
                _pos_idx = self.valid_sampler.get_cos_connection(valid_block)# (T_valid, N_vanbr)

                _sta_loc = self.valid_locations[valid_slice]                 # (T_valid, dim_ct)
                _pos_loc = self.train_locations[_pos_idx]                    # (T_valid, N_vanbr, dim_ct)
                
                _sta_emb = self.valid_emb[valid_slice]                       # (T_valid, dim_eu)
                _pos_emb = self.train_emb[_pos_idx]                          # (T_valid, N_vanbr, dim_eu)
                
                data_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                data_ready_event.record(data_streams[sid])

            ### step.2 计算 loss 并选择最佳位置
            with torch.cuda.device(device), torch.cuda.stream(defa_streams[sid]):
                defa_streams[sid].wait_event(data_ready_event)
                
                sta_loc, pos_loc, sta_emb, pos_emb = (
                    _sta_loc.to(device, non_blocking=True),
                    _pos_loc.to(device, non_blocking=True),
                    _sta_emb.to(device, non_blocking=True),
                    _pos_emb.to(device, non_blocking=True),
                )
                
                cnc_loc   = self.connection(sta_loc, dev_num=sid)
                mask      = self.converge_mask(T_valid, N_vanbr, sid)

                loss_cos  = self.loom_valid(
                    cnc_loc,
                    sta_loc, pos_loc,
                    sta_emb, pos_emb,
                    mask, sid
                )
                
                selected_locs, loss_cos_T, _, _  = self.get_best_loc(cnc_loc, loss_cos, loss_cos, loss_cos, sid) # (T_valid, ) * 3, (T_valid, dim_ct)
                
                comp_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                comp_ready_event.record(defa_streams[sid])
            
            with torch.cuda.device(0), torch.cuda.stream(data_streams[sid]):
                data_streams[sid].wait_event(comp_ready_event)
                
                self.new_valid_locations[valid_slice] = selected_locs.to(main_device, non_blocking=True)
                self.valid_loss_cos_buf [valid_slice] = loss_cos_T.to(main_device, non_blocking=True)

    @thread_guard
    def train_epoch(self, sid: int):
        for cur_epoch in range(train_epoch_num):
            self.epoch_barrier.wait()
            
            ### step 1: 考虑 train_blocks
            self.train_train_blocks(sid)
            # self.epoch_barrier.wait()
            # self._synchronize_all_streams()
            
            ### step 2: 考虑 vocab_blocks
            self.train_vocab_blocks(sid)
            self.epoch_barrier.wait()
            self._synchronize_all_streams()
    
    @thread_guard
    def test_time_train_epoch(self, sid: int):
        for cur_epoch in range(valid_epoch_num):
            self.epoch_barrier.wait()
            self.train_valid_blocks(sid)
            self.epoch_barrier.wait()
            self._synchronize_all_streams()
                
                    

    @gettime(fmt='ms', pr=True)
    def train_all(
        self,        
        train_emb : torch.Tensor, # (N_train, dim)
        vocab_emb : torch.Tensor, # (N_vocab, dim)
        train_tar : torch.Tensor, # (N_train, )   
    ):  
        """
        我们需要拟合的是一个:
        """
        mark(ST, "all")
        mark(ST, "all_preparation")
        
        ### step.2 准备数据分块与采样器
        mark(ST, "all_preparation_2")
        self.train_tar     = train_tar.to(main_device)
        self.train_emb     = train_emb.to(main_device)
        self.vocab_emb     = vocab_emb.to(main_device)
        
        self.train_sampler = TrainSampler()
        self.vocab_sampler = VocabSampler()

        self.train_loss_cos_buf  = torch.zeros(N_train, device=main_device)
        self.train_loss_cro_buf  = torch.zeros(N_train, device=main_device)
        self.train_loss_tot_buf  = torch.zeros(N_train, device=main_device)
        
        self.vocab_loss_cos_buf  = torch.zeros(N_vocab, device=main_device)

        self._synchronize_all_streams()
        mark(ED, "all_preparation_2", father="all_preparation")
        
        
        ### step 3: 开启多线程
        mark(ST, "all_preparation_3")
        self.epoch_barrier  = threading.Barrier(num_devices + 1) # num_devices 个生产线程，num_devices 个消费线程，1 个主线程
        threads: List[threading.Thread] = []
        for i, _ in enumerate(devices):
            thread = threading.Thread(target=self.train_epoch, args=(i,))
            thread.start()
            threads.append(thread)
        mark(ED, "all_preparation_3", father="all_preparation")
        mark(ED, "all_preparation", father="all")
        
        
        mark(ST, "all_epoch")
        ### step 3: 遍历所有 epoch
        for cur_epoch in range(train_epoch_num):
            mark(ST, "epoch")
            
            ### step 3.1: epoch 前的准备工作
            mark(ST, "epoch_preparation")
            self.cur_epoch = cur_epoch
            self.converge  = loss_strategy['train_converge'] is not None and (self.cur_epoch >= loss_strategy['train_converge'])
            loss_split_record = {
                "train_cos_loss":  0.,
                "train_cro_loss":  0.,
                "train_tot_loss":  0.,
            }
            
            if cur_epoch % train_graph_reset == 0 and cur_epoch != 0:
                self.train_sampler.reset_indices()
            if cur_epoch % vocab_graph_reset == 0 and cur_epoch != 0:
                self.vocab_sampler.reset_indices()
            
            self.new_train_locations = self.train_locations.clone()
            self.new_vocab_locations = self.vocab_locations.clone()
            
            self._synchronize_all_streams()
            
            mark(ED, "epoch_preparation", father="epoch")
            
            
            ### step 3.2: 训练
            mark(ST, "epoch_train")
            
            self.epoch_barrier.wait()          # 第一阶段：train_train
            # self.epoch_barrier.wait()          # 第二阶段：train_vocab        
            self.epoch_barrier.wait()          # 第三阶段：整合数据
            
            mark(ED, "epoch_train", father="epoch")
            
            
            ### step 3.3: 位置更新（oos train）
            mark(ST, "epoch_pos_train")
            
            self._synchronize_all_streams()
            self.train_locations = self.new_train_locations
            self.vocab_locations = self.new_vocab_locations

            mark(ED, "epoch_pos_train", father="epoch")
            
            
            ### step 3.4: 记录与打印
            loss_split_record[f"train_cos_loss"] = self.train_loss_cos_buf.sum().item() / N_train
            loss_split_record[f"train_cro_loss"] = self.train_loss_cro_buf.sum().item() / N_train
            loss_split_record[f"train_tot_loss"] = self.train_loss_tot_buf.sum().item() / N_train
            loss_split_record[f"vocab_cos_loss"] = self.vocab_loss_cos_buf.sum().item() / N_vocab
            
            print(f"epoch {cur_epoch:3d} summary:", end=" ")
            for k, v in loss_split_record.items():    
                if "cos" in k: 
                    print(f"{k:15s}: {v:.6f}", end=", ")
                else:
                    print(f"{k:15s}: {v:.4f}", end=", ")
            print()
            sys.stdout.flush()
            
            ### step 3.5: 验证
            mark(ST, "epoch_valid")
            if cur_epoch % val_interval == 0 or cur_epoch == train_epoch_num - 1: 
                self.validate(cur_epoch, "train")
            mark(ED, "epoch_valid", father="epoch")
            mark(ED, "epoch")
            
            ### step 3.6: 可视化
            if cur_epoch % vis_interval == 0 or cur_epoch == train_epoch_num - 1:
                self.visualize(cur_epoch, "train")

        
        for thread in threads:
            thread.join()

        torch.save( 
            {
                "train_locations": self.train_locations.cpu(),
                "vocab_locations": self.vocab_locations.cpu(),
            },
            train_save_path
        )

        mark(ED, "all_epoch", father="all")
        mark(ED, "all")
    

    @gettime(fmt='ms', pr=True)
    def test_time_train_all(
        self,
        train_emb : torch.Tensor, # (N_train, dim)
        valid_emb : torch.Tensor, # (N_train, dim)
        vocab_emb : torch.Tensor, # (N_vocab, dim)
        valid_tar : torch.Tensor, # (N_train, )  
    ):
        mark(ST, "all")
        mark(ST, "all_preparation")
        
        ### step.2 准备数据分块与采样器
        mark(ST, "all_preparation_2")
        self.train_emb     = train_emb.to(main_device)
        self.valid_emb     = valid_emb.to(main_device)
        self.vocab_emb     = vocab_emb.to(main_device)
        self.valid_tar     = valid_tar.long().to(main_device)
        
        self.valid_sampler = ValidSampler()

        self.valid_loss_cos_buf  = torch.zeros(N_valid, device=main_device)

        locations = torch.load(train_save_path)
        self.train_locations = locations['train_locations'].to(main_device)
        self.vocab_locations = locations['vocab_locations'].to(main_device)
        self.new_valid_locations = self.valid_locations.clone()
        
        self._synchronize_all_streams()
        mark(ED, "all_preparation_2", father="all_preparation")
        
        
        ### step 3: 开启多线程
        mark(ST, "all_preparation_3")
        self.epoch_barrier  = threading.Barrier(num_devices + 1) # num_devices 个生产线程，num_devices 个消费线程，1 个主线程
        threads: List[threading.Thread] = []
        for i, _ in enumerate(devices):
            thread = threading.Thread(target=self.test_time_train_epoch, args=(i,))
            thread.start()
            threads.append(thread)
        mark(ED, "all_preparation_3", father="all_preparation")
        mark(ED, "all_preparation", father="all")
        
        
        mark(ST, "all_epoch")
        ### step 3: 遍历所有 epoch
        for cur_epoch in range(valid_epoch_num):
            mark(ST, "epoch")
            
            ### step 3.1: epoch 前的准备工作
            mark(ST, "epoch_preparation")
            self.cur_epoch = cur_epoch
            self.converge  = loss_strategy['valid_converge'] is not None and (self.cur_epoch >= loss_strategy['valid_converge'])
            loss_split_record = {
                "valid_cos_loss":  0.,
            }
            
            if cur_epoch % valid_graph_reset == 0 and cur_epoch != 0:
                self.valid_sampler.reset_indices()
            
            self._synchronize_all_streams()
            
            mark(ED, "epoch_preparation", father="epoch")
            
            ### step 3.2: 训练
            mark(ST, "epoch_train")
            
            self.epoch_barrier.wait()          # 第一阶段：train_train       
            self.epoch_barrier.wait()          # 第二阶段：整合数据
            
            mark(ED, "epoch_train", father="epoch")
            
            
            ### step 3.3: 位置更新（pos train）
            mark(ST, "epoch_pos_train")
        
            self._synchronize_all_streams()
            self.valid_locations = self.new_valid_locations
            
            mark(ED, "epoch_pos_train", father="epoch")
            
            ### step 3.4: 记录与打印
            loss_split_record["valid_cos_loss"] = self.valid_loss_cos_buf.sum().item() / N_valid

            print(f"epoch {cur_epoch:3d} summary:", end=" ")
            for k, v in loss_split_record.items():    
                print(f"{k:15s}: {v:.6f}", end=", ")
            print()
            sys.stdout.flush()
            
            ### step 3.5: 验证
            mark(ST, "epoch_valid")
            if cur_epoch % val_interval == 0 or cur_epoch == train_epoch_num - 1:
                self.validate(cur_epoch, "valid")
            mark(ED, "epoch_valid", father="epoch")
            mark(ED, "epoch")
            
            ### step 3.6: 可视化
            if cur_epoch % vis_interval == 0 or cur_epoch == train_epoch_num - 1:
                self.visualize(cur_epoch, "valid")
        
        for thread in threads:
            thread.join()

        torch.save(
            {
                "train_locations": self.train_locations.cpu(),
                "vocab_locations": self.vocab_locations.cpu(),
                "valid_locations": self.valid_locations.cpu(),
            },       
            valid_save_path
        )

        mark(ED, "all_epoch", father="all")
        mark(ED, "all")
    
    def validate(self, epoch: int, cur_type: str):
        loss     = 0
        accuracy = 0
        
        splits   = train_blocks if cur_type == "train" else valid_blocks
        for i, block in enumerate(splits):
            targets     = self.train_tar[block[0]:block[1]] if cur_type == "train" else self.valid_tar[block[0]:block[1]]
            sta_loc     = self.train_locations[block[0]:block[1]] if cur_type == "train" else self.valid_locations[block[0]:block[1]]
            voc_loc     = self.vocab_locations                   # (V, D)
            sta_emb     = self.train_emb[block[0]:block[1]] if cur_type == "train" else self.valid_emb[block[0]:block[1]]  
                                                                  # (T, dim)
            voc_emb     = self.vocab_emb                          # (V, dim)
            
            ct_val      = self.cos_similarity(
                sta_loc[:, None, :], # (T, 1, D) 
                voc_loc[None, :, :], # (1, V, D)   
            ).mean(dim=-1) # (T, V)
            
            if use_eu_norm:
                eu_cro_nrm = (torch.norm(sta_emb, dim=-1, keepdim=True) @ torch.norm(voc_emb, dim=-1, keepdim=True).t()).squeeze()  # (T, 1) @ (1, V) -> (T, V)
            else:
                eu_cro_nrm = torch.ones_like(ct_val) * 20.
            
            ct_logits   = ct_val * eu_cro_nrm
            
            loss       += F.cross_entropy(ct_logits, targets, reduction='sum').item()
            accuracy   += (ct_logits.argmax(dim=-1) == targets).sum().item()

        loss     /= (N_train if cur_type == "train" else N_valid)
        accuracy /= (N_train if cur_type == "train" else N_valid)

        print(f"{cur_type:5s} loss: {loss:.4f}, accuracy: {accuracy:.4f}")
        sys.stdout.flush()
        
        return loss

    
    def visualize(self, epoch: int, cur_type: str):
        if cur_type == "train":
            train_eu_emb = self.train_emb[:256]                           # (256, dim)
            train_ct_emb = self.train_locations[:256]                    # (256, tp)
            vocab_eu_emb = self.vocab_emb[:256]                           # (256, dim)
            vocab_ct_emb = self.vocab_locations[:256]                    # (256, tp)
            
            S_tt_eu      = normalized_matmul(train_eu_emb, train_eu_emb.t())[0].cpu().numpy()
            S_tt_ct      = self.cos_similarity(train_ct_emb[:, None, :], train_ct_emb[None, :, :]).mean(dim=-1).cpu().numpy()
            visualize_similarity(S_tt_eu, S_tt_ct, meta_name="{}" + "by_eu_train_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))
            visualize_similarity(S_tt_ct, S_tt_ct, meta_name="{}" + "by_ct_train_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))
            
            # S_vv_eu      = normalized_matmul(vocab_eu_emb, vocab_eu_emb.t())[0].cpu().numpy()
            # S_vv_ct      = self.cos_similarity(vocab_ct_emb[:, None, :], vocab_ct_emb[None, :, :]).mean(dim=-1).cpu().numpy()
            # visualize_similarity(S_vv_eu, S_vv_ct, meta_name="{}" + "vocab_vocab_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))
            
            # S_tv_eu      = normalized_matmul(train_eu_emb, vocab_eu_emb.t())[0].cpu().numpy()
            # S_tv_ct      = self.cos_similarity(train_ct_emb[:, None, :], vocab_ct_emb[None, :, :]).mean(dim=-1).cpu().numpy()
            # visualize_pair_bihclust(S_tv_eu, S_tv_ct, meta_name="{}" + "train_vocab_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))
            
            

        elif cur_type == "valid":
            train_eu_emb = self.train_emb[:256]                        # (256, dim)
            train_ct_emb = self.train_locations[:256]                         # (256, tp)
            valid_eu_emb = self.valid_emb[:256]  # (256, dim)
            valid_ct_emb = self.valid_locations[:256]  # (256, tp)
        
            S_vt_eu      = normalized_matmul(valid_eu_emb, train_eu_emb.t())[0].cpu().numpy()
            S_vt_ct      = self.cos_similarity(valid_ct_emb[:, None, :], train_ct_emb[None, :, :]).mean(dim=-1).cpu().numpy()

            visualize_pair_bihclust(S_vt_eu, S_vt_ct, meta_name="{}" + "valid_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))

    def _synchronize_all_streams(self):
        for id in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{id}') 
            torch.cuda.synchronize(device)
            torch.cuda.default_stream(device).synchronize()
            
        for stream in defa_streams:
            stream.synchronize()
        
        for stream in data_streams:
            stream.synchronize()

        
if __name__ == "__main__":
    pass

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
from src.new_loom_kernel import ct_loss_triton_sampled_2dteacher
from src.para import *
from src.sampler import *
from src.utils import *
from src.vis import visualize_pair_bihclust, visualize_similarity
from dataclasses import dataclass

main_device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
from typing import List, Tuple

@dataclass
class TrainLossRecord:
    dyn_dyn_loss: float = 0.0
    dyn_sta_loss: float = 0.0
    tot_dyn_loss: float = 0.0
    sta_sta_loss: float = 0.0

class ValidLossRecord:
    tot_dyn_loss: float = 0.0

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
            dtype=torch.int64, device=main_device, generator=main_generator
        )
        self.vocab_locations = torch.randint(
            1 - self.n, self.n, (N_vocab, self.tp), 
            dtype=torch.int64, device=main_device, generator=main_generator
        )
        self.valid_locations = torch.randint(
            1 - self.n, self.n, (N_valid, self.tp), 
            dtype=torch.int64, device=main_device, generator=main_generator
        )

        ### 3. 额外参数  
        self.timer = CUDATimer()
     
    def connection_masks(self, sz, dev_num=0):
        device = devices[dev_num] if dev_num >= 0 else main_device
        generator = generators[dev_num] if dev_num >= 0 else main_generator
        
        upper_bounds   = 2 ** torch.arange(self.h, dtype=torch.int64, device=device)
        random_numbers = torch.randint(
            0, self.n, 
            (self.h, sz, N_K, self.tp), 
            dtype=torch.int64, device=device, generator=generator
        ) # (H, B*T, K, D)
        masks = random_numbers & (upper_bounds.view(-1, 1, 1, 1) - 1)
        # masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3) # (B*T, H, K, D)
    
    def connection(self, ori_int: torch.Tensor, dev_num=0):
        device = devices[dev_num] if dev_num >= 0 else main_device
        
        flip_masks = (1 << torch.arange(self.h, device=device, dtype=ori_int.dtype)).unsqueeze(0).unsqueeze(2)
        flipped_ints = ori_int.unsqueeze(1) ^ flip_masks # (B*T1, H, D)
        random_masks = self.connection_masks(flipped_ints.size(0), dev_num=dev_num)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h * N_K, self.tp)
        # (B*T1, H, 1, D) ^ (B*T1, H, K, D) -> (B*T1, H*K, D)
        loc = torch.cat((result, ori_int.unsqueeze(1), -result), dim=1) # (B*T1, H*K + 1 + H*K, D)
        return loc
                    
    def converge_mask(self, size0: int, size1: int, dev_num=0):
        device  = devices[dev_num] if dev_num >= 0 else main_device
        generator = generators[dev_num] if dev_num >= 0 else main_generator
        mask   = torch.ones((size0, size1), dtype=torch.bool, device=device)
        
        if not self.converge:
            choosing_mask = (torch.rand((size0), device=device) > 0.2)  
            sel_idx = (~choosing_mask).nonzero(as_tuple=False).squeeze(1)
            if sel_idx.numel() > 0:
                mask[sel_idx, :] = False
                rows = sel_idx.repeat_interleave(sample_k)
                cols = torch.randint(0, size1, (rows.numel(),), device=device, generator=generator)
                mask[rows, cols] = True

        return mask

    def get_best_loc(
        self, 
        cnc_loc : torch.Tensor, # (T, C, D)
        loss    : torch.Tensor,
        loss_1  : torch.Tensor,
        loss_2  : torch.Tensor,
        sid     : int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = devices[sid] if sid >= 0 else main_device
        generator = generators[sid] if sid >= 0 else main_generator
        T      = cnc_loc.size(0)
        
        # step1: 每个样本在 [0, tp) 之间随机生成一个排列（通过排序 trick）
        rand_vals     = torch.rand((T, self.tp), device=device, generator=generator) # (T, D)
        rand_cols     = rand_vals.argsort(dim=1)[:, :cur_tp]           # (T, cur_tp)
        # print(rand_cols[:4, :2])
        
        # step2: 每个 (t, dim) 的 argmin
        argmin_all    = torch.argmin(loss, dim=1)                  # (T, D)

        # step3: 初始化为 "保持不更新"
        cnc_indices   = torch.full_like(argmin_all, N_K * self.h)   # (T, D)

        # step4: 高级索引直接填充
        t_idx = torch.arange(T, device=loss.device)[:, None].expand(T, cur_tp)  # (T, upd)
        t_mask = torch.rand(T, device=device, generator=generator) < cur_portion  # (T,)
        t_idx, rand_cols = t_idx[t_mask], rand_cols[t_mask]  # (T_upd, ), (T_upd, upd)
        
        cnc_indices[t_idx, rand_cols] = argmin_all[t_idx, rand_cols]

        

        T_indices     = torch.arange(T              ,    device=device)[:, None]         # (T, 1)
        dim_indices   = torch.arange(self.tp    ,        device=device)[None  :]         # (1, D)
        selected_locs = cnc_loc [T_indices, cnc_indices,  dim_indices]                   # (T, D)
        
        real_loss     = loss    [T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, )
        real_loss1    = loss_1  [T_indices, cnc_indices, dim_indices].mean(dim=-1) if loss_1 is not None else None       
                                                                                         # (T, )
        real_loss2    = loss_2  [T_indices, cnc_indices, dim_indices].mean(dim=-1) if loss_2 is not None else None        
                                                                                         # (T, )
        
        return selected_locs, real_loss, real_loss1, real_loss2

    def cos_similarity(self, coord1: torch.Tensor, coord2: torch.Tensor) -> torch.Tensor:
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        xor_result = torch.abs(coord1) ^ torch.abs(coord2)
        _, exp = torch.frexp((xor_result + 1).to(torch.float32))
        s = exp.float() / self.h
        return sg * (1 - s)
    
    def loom_dyn(
        self, 
        cur_tar: torch.Tensor,   # (T_train, )
        cnc_loc: torch.Tensor,   # (T_train, N_C    , dim_ct)
        
        sta_loc: torch.Tensor,   # (T_train, dim_ct)
        pos_loc: torch.Tensor,   # (T_train, N_nbr, dim_ct)

        sta_emb: torch.Tensor,   # (T_train, dim_eu)
        pos_emb: torch.Tensor,   # (T_train, N_nbr, dim_eu)
        
        mask   : torch.Tensor,   # (T_train, N_nbr + N_tvnbr)
        sid    : int = 0,
        cur_type: str = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device  = devices[sid]
        ### step 1: 计算欧氏空间的相似度
        eu_val = (sta_emb[:, None, :] @ pos_emb.transpose(-1, -2)).squeeze()[:, :, None, None]  
            ### (T_train, 1, dim_eu) @ (T_train, dim_eu, N_nbr) -> (T_train, 1, N_nbr) -> (T_train, N_nbr, 1, 1)
        # print("what the fuck")
        # print(eu_val[:, :, 0, 0])
        

        ### step 2: 计算 CT 空间的相似度
        cos_sta_pos     = self.cos_similarity(
            sta_loc[:, None, :],    pos_loc[:, :, :]      
        )                                        # (T_train, N_nbr + K_vocab,      dim_ct)
        cos_sta_pos_sum = cos_sta_pos.sum(dim=-1) 
                                                 # (T_train, N_nbr + K_vocab             )
        # cos_cnc_pos     = self.cos_similarity(
        #     cnc_loc[:, None, :, :], pos_loc[:, :, None,:]
        # )                                        # (T_train, N_trnbr + K_vocab, N_C, dim_ct)
        # ct_val          = (
        #     cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + cos_cnc_pos
        # ) / self.tp * eu_nrm[:, :, None, None]         
        
        
        ct_val = ct_val_triton(
            cnc_loc.to(torch.int32).contiguous(),
            pos_loc.to(torch.int32).contiguous(),
            torch.ones((T_train if cur_type=="train" else T_valid, N_nbr if cur_type=="train" else N_dynbr), device=device, dtype=torch.float32).contiguous(),
            cos_sta_pos.contiguous(),
            cos_sta_pos_sum.contiguous(),
            tp=float(self.tp),
            h =float(self.h),
            out=None,                # 或传入你复用的 out 缓冲
            BLOCK_S=32,
            BLOCK_CD=32,
            NUM_WARPS=8,
            NUM_STAGES=2,
        )
        
        
        # (T_train, N_ttnbr + K_vocab, N_C, dim_ct)
        ## 对于 T 个 starting point，向 S_tot 个 positive sample 连边。此时，我们把其中某个 positive sample 替换为 connected sample，共有 C 个；此时，D 个维度上的的距离是多少？

        ### step 3: 计算 loss
        # loss_dyn_dyn = (
        #     torch.square(eu_val[:, 0:N_dynbr, :, :] - ct_val[:, :N_dynbr, :, :]) 
        #     * torch.abs(eu_val[:, :N_dynbr, :, :]) # (T_train, N_dynbr + N_stnbr, N_C, dim_ct)
        # ).sum(dim=1)  # (T_train, N_C, dim_ct)
        # loss_dyn_dyn = (
        #     ( F.log_softmax(eu_val[:, :N_dynbr, :, :] * 20 / temperature, dim=1) 
        #     - F.log_softmax(ct_val[:, :N_dynbr, :, :] * 20 / temperature, dim=1)) 
        #     * F.softmax(eu_val[:, :N_dynbr, :, :] * 20 / temperature, dim=1)
        # ).sum(dim=1)  # (T_train, N_C, dim_ct)
        loss_dyn_dyn = sampled_softmax_loss(
            eu_val[:, :N_dynbr, :, :], 
            ct_val[:, :N_dynbr, :, :], 
            N_dynbr, N_top, temperature, N_train
        ) # (T_train, N_C, dim_ct)
        if cur_type == "train":
            loss_dyn_sta = (- F.log_softmax(ct_val[:, N_dynbr:, :, :] * 20 / temperature, dim=1)).gather(
                dim=1,
                index=cur_tar[:, None, None, None].expand(-1, -1, N_C, tp)
            ).squeeze(1) # (T_train, N_C, dim_ct)
        else:
            loss_dyn_sta = None
        
        # T = T_train if cur_type == "train" else T_valid
        # S = N_nbr if cur_type == "train" else N_dynbr
        
        # loss_dyn_dyn, loss_dyn_sta = ct_loss_triton_sampled_2dteacher(
        #     cnc_loc=cnc_loc,
        #     pos_loc=pos_loc,
        #     eu_norm=torch.ones((T, S), device=device, dtype=torch.float32).contiguous(),
        #     cos_sta_pos=cos_sta_pos,
        #     cos_sta_pos_sum=cos_sta_pos_sum,
        #     eu_teacher=eu_val,     # [T,S] 或 [T,S,1,1]
        #     tp=float(self.tp),
        #     h=float(self.h),
        #     temperature=temperature,
        #     N_dynbr=N_dynbr,
        #     N_topk=N_top,
        #     N_total=N_train,               # 你采样的总池大小
        #     cur_type=cur_type,             # "train" / "valid"
        #     cur_tar=cur_tar if cur_type == "train" else None,
        #     scale=20.0,
        #     BLOCK_S=128,
        #     BLOCK_CD=64,
        # )

                                    
        return loss_dyn_dyn, loss_dyn_sta

    def loom_sta(
        self, 
        cnc_loc: torch.Tensor,   # (T_vocab, N_C    , dim_ct)
        
        sta_loc: torch.Tensor,   # (T_vocab, dim_ct)
        sta_emb: torch.Tensor,   # (T_vocab, dim_eu)
        
        sid    : int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device  = devices[sid]
        voc_loc = self.voc_loc[sid].unsqueeze(1).repeat(1, N_vocab, 1)       # (T_vocab, dim_ct) -> (T_vocab, N_vocab, dim_ct)
        voc_emb = self.voc_emb[sid].unsqueeze(0).repeat(T_vocab, 1, 1) # (N_vocab, dim_eu) -> (T_vocab, N_vocab, dim_eu)

        ### step 1: 计算欧氏空间的相似度
        eu_val = (sta_emb[:, None, :] @ voc_emb.transpose(-1, -2)).squeeze()[:, :, None, None]  
            ### (T_vocab, 1, dim_eu) @ (T_vocab, dim_eu, N_vocab) -> (T_vocab, 1, N_vocab) -> (T_vocab, N_vocab, 1, 1)
        

        ### step 2: 计算 CT 空间的相似度
        cos_sta_pos     = self.cos_similarity(
            sta_loc[:, None, :],    voc_loc[:, :, :]      
        )                                        # (T_vocab, N_vocab,      dim_ct)
        cos_sta_pos_sum = cos_sta_pos.sum(dim=-1) 
                                                 # (T_vocab, N_vocab             )
        # cos_cnc_pos     = self.cos_similarity(
        #     cnc_loc[:, None, :, :], voc_loc[:, :, None,:]
        # )                                        # (T_vocab, N_vocab, N_C, dim_ct)
        # ct_val          = (
        #     cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + cos_cnc_pos
        # ) / self.tp * eu_nrm[:, :, None, None]         
        
        ct_val = ct_val_triton(
            cnc_loc.to(torch.int32).contiguous(),
            voc_loc.to(torch.int32).contiguous(),
            torch.ones((T_vocab, N_vocab), device=device, dtype=torch.float32).contiguous(),
            cos_sta_pos.contiguous(),
            cos_sta_pos_sum.contiguous(),
            tp=float(self.tp),
            h =float(self.h),
            out=None,                # 或传入你复用的 out 缓冲
            BLOCK_S=32,
            BLOCK_CD=32,
            NUM_WARPS=8,
            NUM_STAGES=2,
        )
        
        ### step 3: 计算 loss
        loss_sta_sta = (
             (F.log_softmax(eu_val * 20 / temperature, dim=1) 
            - F.log_softmax(ct_val * 20 / temperature, dim=1)) 
            * F.softmax(eu_val     * 20 / temperature, dim=1)
        ).sum(dim=1)  # (T_vocab, N_C, dim_ct)
        
                                          
        return loss_sta_sta

    ''' 下面的内容为 sta-dyn 对齐使用的，目前不开启
    def loom_sta(
        self, 
        cur_tar: torch.Tensor,   # (T_vtnbr, )
        pos_loc: torch.Tensor,   # (T_vtnbr, dim_ct)
        pos_emb: torch.Tensor,   # (T_vtnbr, dim_eu)        
        sid: int
    ) -> torch.Tensor:
        device = devices[sid]
    
        voc_loc = self.voc_loc[sid]
        voc_emb = self.vocab_emb[sid]
        cnc_loc = self.voc_cnc_loc[sid]

        ### step 1: 计算欧氏空间的相似度
        eu_val = (sta_emb[:, None, :] @ pos_emb.transpose(-1, -2)).squeeze()[:, :, None, None]  
            ### (T_train, 1, dim_eu) @ (T_train, dim_eu, N_nbr) -> (T_train, 1, N_nbr) -> (T_train, N_nbr, 1, 1)
        

        ### step 2: 计算 CT 空间的相似度
        cos_sta_pos     = self.cos_similarity(
            sta_loc[:, None, :],    pos_loc[:, :, :]      
        )                                        # (T_train, N_nbr + K_vocab,      dim_ct)
        cos_sta_pos_sum = cos_sta_pos.sum(dim=-1) 
                                                 # (T_train, N_nbr + K_vocab             )
        # cos_cnc_pos     = self.cos_similarity(
        #     cnc_loc[:, None, :, :], pos_loc[:, :, None,:]
        # )                                        # (T_train, N_trnbr + K_vocab, N_C, dim_ct)
        # ct_val          = (
        #     cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + cos_cnc_pos
        # ) / self.tp * eu_nrm[:, :, None, None]         
        
        ct_val = ct_val_triton(
            cnc_loc.to(torch.int32).contiguous(),
            pos_loc.to(torch.int32).contiguous(),
            torch.ones((T_train if cur_type=="train" else T_valid, N_nbr), device=device, dtype=torch.float32).contiguous(),
            cos_sta_pos.contiguous(),
            cos_sta_pos_sum.contiguous(),
            tp=float(self.tp),
            h =float(self.h),
            out=None,                # 或传入你复用的 out 缓冲
            BLOCK_S=32,
            BLOCK_CD=32,
            NUM_WARPS=8,
            NUM_STAGES=2,
        )
        
        
        # (T_train, N_ttnbr + K_vocab, N_C, dim_ct)
        ## 对于 T 个 starting point，向 S_tot 个 positive sample 连边。此时，我们把其中某个 positive sample 替换为 connected sample，共有 C 个；此时，D 个维度上的的距离是多少？

        ### step 3: 计算 loss
        # loss_dyn_dyn = (
        #     torch.square(eu_val[:, 0:N_dynbr, :, :] - ct_val[:, :N_dynbr, :, :]) 
        #     * torch.abs(eu_val[:, :N_dynbr, :, :]) # (T_train, N_dynbr + N_stnbr, N_C, dim_ct)
        # ).sum(dim=1)  # (T_train, N_C, dim_ct)
        loss_dyn_dyn = (
            (F.log_softmax(eu_val[:, :N_dynbr, :, :] * 20 / temperature, dim=1) 
            - F.log_softmax(ct_val[:, :N_dynbr, :, :] * 20 / temperature, dim=1)) 
            * F.softmax(eu_val[:, :N_dynbr, :, :] * 20 / temperature, dim=1)
        ).sum(dim=1)  # (T_train, N_C, dim_ct)
        # loss_dyn_sta = loss[:, N_dynbr:, :, :].sum(dim=1) / N_stnbr  # (T_train, N_C, dim_ct) 

        
        # ### step 1: 计算欧式空间的 norm
        # if use_eu_norm == True:
        #     eu_cro_nrm = (torch.norm(pos_emb, dim=-1, keepdim=True) @ torch.norm(voc_emb, dim=-1, keepdim=True).t()).squeeze()  # (T_vtnbr, 1) @ (1, N_vocab) -> (T_vtnbr, N_vocab,) -> (T_vtnbr, N_vocab)
        # else:
        #     eu_cro_nrm = torch.ones((T_vtnbr, N_vocab), device=device) * 20.

        # ### step 2: 计算 CT 空间的相似度
        # cos_nei_voc     = self.cos_similarity(pos_loc[:, None, :], voc_loc[None, :, :])  # (T_vtnbr, N_vocab, dim_ct)
        # logits_ori      = cos_nei_voc.mean(dim=-1) * eu_cro_nrm                          # (T_vtnbr, N_vocab)
        
        # cos_nei_voc_sum = cos_nei_voc.sum(dim=-1)                                        # (T_vtnbr, N_vocab)
        # cos_nei_cnc = self.cos_similarity(pos_loc[:, None, None, :], cnc_loc[None, :, :, :])  
        #                                                                                  # (T_vtnbr, N_vocab, N_C, dim_ct)
        # logits_upd      = (
        #     cos_nei_voc_sum[:, :, None, None] - cos_nei_voc[:, :, None, :] + cos_nei_cnc
        # ) / self.tp * eu_cro_nrm[:, :, None, None]                                       # (T_vtnbr, N_vocab, N_C, dim_ct)
        
        # ### 数值稳定性
        # m = torch.maximum(
        #     logits_ori[:, :, None, None], 
        #     logits_upd
        # ).max(dim=1).values[:, None, :, :]                                               # (T_vtnbr, 1,       N_C, dim_ct)

        # exp_ori         = torch.exp(logits_ori[:, :, None, None] - m)                    # (T_vtnbr, N_vocab, N_C, dim_ct)
        # exp_upd         = torch.exp(logits_upd                   - m)                    # (T_vtnbr, N_vocab, N_C, dim_ct)
        
        # sum_exp_ori     = exp_ori.sum(dim=1, keepdim=True)                               # (T_vtnbr, 1, N_C, dim_ct)
        # log_sum_exp     = m + torch.log(
        #     sum_exp_ori - exp_ori + exp_upd + 1e-9
        # )                                                                                # (T_vtnbr, N_vocab, N_C, dim_ct)
        
        # mask            = (cur_tar[:, None] == torch.arange(N_vocab, device=device)[None, :])[:, :, None, None]  
        #                                                                                  # (T_vtnbr, N_vocab, 1  , 1)
        # targets_val     = (
        #     torch.gather(logits_ori, dim=1, index=cur_tar[:, None])[:, :, None, None] * (~mask) +
        #     logits_upd * mask
        # )   # (T_vtnbr, N_vocab, N_C, dim_ct)
        
        # loss_sta_dyn    = (-targets_val + log_sum_exp).sum(dim=0)                        # (N_vocab, N_C, dim_ct)
        
        # return loss_sta_dyn

        '''
    
    ''' 什么古早版本的代码留着做啥呢
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
    
    
    # def loom_valid(
    #     self, 
    #     cnc_loc: torch.Tensor,   # (T_valid, N_C    , dim_ct)
        
    #     sta_loc: torch.Tensor,   # (T_valid, dim_ct)
    #     pos_loc: torch.Tensor,   # (T_valid, N_vanbr, dim_ct)

    #     sta_emb: torch.Tensor,   # (T_valid, dim_eu)
    #     pos_emb: torch.Tensor,   # (T_valid, N_vanbr, dim_eu)
        
    #     mask   : torch.Tensor,   # (T_valid, N_vanbr)
    #     sid    : int = 0,
    # ) -> torch.Tensor:
    #     device = devices[sid]
        
        
    #     ### step 1: 计算欧氏空间的相似度
    #     eu_val = 20. * (sta_emb[:, None, :] @ pos_emb.transpose(-1, -2)).squeeze()  
    #         ### (T_valid, 1, dim_eu) @ (T_valid, dim_eu, N_vanbr) -> (T_valid, 1, N_vanbr) -> (T_valid, N_vanbr)

    #     ### step 2: 计算 CT 空间的相似度
    #     cos_sta_pos     = self.cos_similarity(
    #         sta_loc[:, None, :],    pos_loc[:, :, :]      
    #     )                                        # (T_valid, N_vanbr, dim_ct)
    #     cos_sta_pos_sum = cos_sta_pos.sum(dim=-1) 
    #                                              # (T_valid, N_vanbr)
    #     # cos_cnc_pos     = self.cos_similarity(
    #     #     cnc_loc[:, None, :, :], pos_loc[:, :, None,:]
    #     # )                                        # (T_valid, N_vanbr, N_C, dim_ct)
    #     # ct_val          = (
    #     #     cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + cos_cnc_pos
    #     # ) / self.tp          

    #     ct_val = ct_val_triton(
    #         cnc_loc.to(torch.int32).contiguous(),
    #         pos_loc.to(torch.int32).contiguous(),
    #         torch.ones((T_valid, N_nbr), device=device, dtype=torch.float32).contiguous(),
    #         cos_sta_pos.contiguous(),
    #         cos_sta_pos_sum.contiguous(),
    #         tp=float(self.tp),
    #         h=float(self.h),
    #         out=None,                # 或传入你复用的 out 缓冲
    #         BLOCK_S=32,
    #         BLOCK_CD=32,
    #         NUM_WARPS=8,
    #         NUM_STAGES=2,
    #     ) * 20.
    #      # (T_valid, N_vanbr, N_C, dim_ct)


    #     ### step 3: 计算 dyn-dyn cross-entropy loss
    #     loss_dyn_dyn     = (
    #         F.softmax(eu_val, dim=1)[..., None, None] * (
    #             F.log_softmax(eu_val, dim=1)[..., None, None] / temperature  - 
    #             F.log_softmax(ct_val, dim=1) / temperature
    #         )
    #     ).sum(dim=1)                                       # (T_valid, N_C, dim_ct)  
    
    #     return loss_dyn_dyn
    '''

    def train_dyn_blocks(self, sid: int):
        device = devices[sid]

        read_stream  = read_streams[sid]
        comp_stream  = comp_streams[sid]
        write_stream = write_streams[sid]

        
        # ======================================================
        # read stream k 必须等待 write stream k - prefetch 完成
        # read stream 0 不需要等待，所以提前插入 read stream 自身 event
        # 也即，插入占位 event；此时无需等待任何 write stream
        # ======================================================
        prev_write_done_list = []
        for _ in range(prefetch + 1):
            e = torch.cuda.Event(enable_timing=False, blocking=False)
            e.record(read_stream)
            prev_write_done_list.append(e)

        for train_block_id in train4sid[sid]:

            train_block = train_blocks[train_block_id]
            train_slice = slice(train_block[0], train_block[1])

            # ======================================================
            # Step 1: READ
            # ======================================================
            with torch.cuda.device(main_device), torch.cuda.stream(read_stream):

                torch.cuda.nvtx.range_push(f"READ block {train_block_id} sid {sid}")

                read_stream.wait_event(prev_write_done_list[0])

                _dyn_idx, _voc_idx = self.train_sampler.get_connection(train_block)

                _cur_tar = self.train_tar[train_slice]
                _sta_loc = self.train_locations[train_slice]
                _pos_loc = torch.cat([
                    self.train_locations[_dyn_idx],
                    self.vocab_locations[_voc_idx],
                ], dim=1)

                _sta_emb = self.train_emb[train_slice]
                _pos_emb = torch.cat([
                    self.train_emb[_dyn_idx],
                    self.vocab_emb[_voc_idx]
                ], dim=1)

                data_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                data_ready_event.record(read_stream)

                torch.cuda.nvtx.range_pop()

            # ======================================================
            # Step 2: COMPUTE
            # ======================================================
            with torch.cuda.device(device), torch.cuda.stream(comp_stream):

                torch.cuda.nvtx.range_push(f"COMPUTE block {train_block_id} sid {sid}")

                comp_stream.wait_event(data_ready_event)

                cur_tar = _cur_tar.to(device, non_blocking=True)
                sta_loc = _sta_loc.to(device, non_blocking=True)
                pos_loc = _pos_loc.to(device, non_blocking=True)
                sta_emb = _sta_emb.to(device, non_blocking=True)
                pos_emb = _pos_emb.to(device, non_blocking=True)

                cnc_loc = self.connection(sta_loc, dev_num=sid)
                loss_dyn_dyn, loss_dyn_sta = self.loom_dyn(
                    cur_tar,
                    cnc_loc,
                    sta_loc, pos_loc,
                    sta_emb, pos_emb,
                    None, sid, "train"
                )

                loss_total = ratio_dyn * loss_dyn_dyn + ratio_sta * loss_dyn_sta

                selected_locs, tot_dyn_loss, dyn_dyn_loss, dyn_sta_loss = self.get_best_loc(
                    cnc_loc, loss_total, loss_dyn_dyn, loss_dyn_sta, sid
                )

                comp_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                comp_ready_event.record(comp_stream)

                torch.cuda.nvtx.range_pop()

            # ======================================================
            # Step 3: WRITE
            # ======================================================
            with torch.cuda.device(main_device), torch.cuda.stream(write_stream):

                torch.cuda.nvtx.range_push(f"WRITE block {train_block_id} sid {sid}")

                write_stream.wait_event(comp_ready_event)

                self.new_train_locations[train_slice] = selected_locs.to(main_device, non_blocking=True)
                self.tot_dyn_loss[train_slice]        = tot_dyn_loss.to(main_device, non_blocking=True)
                self.dyn_dyn_loss[train_slice]        = dyn_dyn_loss.to(main_device, non_blocking=True)
                self.dyn_sta_loss[train_slice]        = dyn_sta_loss.to(main_device, non_blocking=True)

                write_done_event = torch.cuda.Event(enable_timing=False, blocking=False)
                write_done_event.record(write_stream)

                prev_write_done_list.pop(0)
                prev_write_done_list.append(write_done_event)

                torch.cuda.nvtx.range_pop()

    def train_sta_blocks(self, sid: int):
        device = devices[sid]

        read_stream  = read_streams[sid]
        comp_stream  = comp_streams[sid]
        write_stream = write_streams[sid]

        # 初始化 prefetch+1 个已完成事件（pipeline 起点）
        prev_write_done_list = []
        for _ in range(prefetch + 1):
            e = torch.cuda.Event(enable_timing=False, blocking=False)
            e.record(read_stream)     # 立即完成
            prev_write_done_list.append(e)

        # 主循环：遍历 vocab block
        for vocab_block_id in vocab4sid[sid]:

            vocab_block = vocab_blocks[vocab_block_id]
            vocab_slice = slice(vocab_block[0], vocab_block[1])

            # ======================================================
            # Step 1: READ (in cuda:0, read_stream)
            # ======================================================
            with torch.cuda.device(main_device), torch.cuda.stream(read_stream):

                # 限制 read 超前（防止 device(0) 内存爆炸）
                read_stream.wait_event(prev_write_done_list[0])

                _sta_loc = self.vocab_locations[vocab_slice]   # (T_vocab, dim_ct)
                _sta_emb = self.vocab_emb      [vocab_slice]   # (T_vocab, dim_eu)

                data_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                data_ready_event.record(read_stream)

            # ======================================================
            # Step 2: COMPUTE (in cuda:sid, comp_stream)
            # ======================================================
            with torch.cuda.device(device), torch.cuda.stream(comp_stream):

                comp_stream.wait_event(data_ready_event)

                sta_loc = _sta_loc.to(device, non_blocking=True)
                sta_emb = _sta_emb.to(device, non_blocking=True)

                cnc_loc = self.voc_cnc_loc[sid]

                loss = self.loom_sta(
                    cnc_loc,
                    sta_loc, sta_emb,
                    sid
                )

                selected_locs, sta_sta_loss, _, _ = self.get_best_loc(
                    cnc_loc, loss, None, None, sid
                )

                comp_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                comp_ready_event.record(comp_stream)

            # ======================================================
            # Step 3: WRITE (in cuda:0, write_stream)
            # ======================================================
            with torch.cuda.device(main_device), torch.cuda.stream(write_stream):

                write_stream.wait_event(comp_ready_event)

                self.new_vocab_locations[vocab_slice] = selected_locs.to(main_device, non_blocking=True)
                self.sta_sta_loss      [vocab_slice] = sta_sta_loss.to(main_device, non_blocking=True)

                # 更新写完成事件（滑动窗口）
                write_done_event = torch.cuda.Event(enable_timing=False, blocking=False)
                write_done_event.record(write_stream)

                prev_write_done_list.pop(0)
                prev_write_done_list.append(write_done_event)



    '''旧版本的 sta-dyn 对齐代码
    def train_sta_blocks(self, sid: int):
        device = devices[sid]
        
        for vtnbr_block_id in vtnbr4sid[sid]:
            vtnbr_block = vtnbr_blocks[vtnbr_block_id]
            vtnbr_slice = slice(vtnbr_block[0], vtnbr_block[1])
            
            ### step.1 准备数据
            with torch.cuda.device(0), torch.cuda.stream(data_streams[sid]):
                _cur_tar = self.train_tar      [vtnbr_slice]                 # (T_vtnbr, )
                _pos_loc = self.train_locations[vtnbr_slice]                 # (T_vtnbr, dim_ct)
                _pos_emb = self.train_emb      [vtnbr_slice]                 # (T_vtnbr, dim_eu)
                
                data_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                data_ready_event.record(data_streams[sid])

            ### step.2 计算 loss 并选择最佳位置
            with torch.cuda.device(device), torch.cuda.stream(defa_streams[sid]):
                defa_streams[sid].wait_event(data_ready_event)
                
                cur_tar, pos_loc, pos_emb = (
                    _cur_tar.to(device, non_blocking=True),
                    _pos_loc.to(device, non_blocking=True),
                    _pos_emb.to(device, non_blocking=True),
                )

                loss  = self.loom_sta(
                    cur_tar,
                    pos_loc, pos_emb,
                    sid
                )
                
                self._voc_loss_buf[sid] += loss
        
        self.epoch_barrier.wait()
        self._synchronize_all_streams()
        
        if sid == 0: # 主进程
            loss = torch.zeros((N_vocab, N_C, self.tp), device=main_device)
            for i in range(num_devices):
                loss += self._voc_loss_buf[i].to(main_device, non_blocking=False)
            loss /= N_train
            
            selected_locs, vocab_loss, _, _ = self.get_best_loc(self.voc_cnc_loc[0].to(main_device), loss, loss, loss, -1) # (N_vocab, ), (N_vocab, dim_ct)
            
            self.new_vocab_locations  = selected_locs
            self.vocab_cro_loss       = vocab_loss
    '''
    
      
    def valid_dyn_blocks(self, sid: int):
        device = devices[sid]

        read_stream  = read_streams[sid]
        comp_stream  = comp_streams[sid]
        write_stream = write_streams[sid]

        # 初始化 prefetch+1 个已完成事件（用于 pipeline 起始）
        prev_write_done_list = []
        for _ in range(prefetch + 1):
            e = torch.cuda.Event(enable_timing=False, blocking=False)
            e.record(read_stream)  # 此刻 read_stream 为空 → event立即完成
            prev_write_done_list.append(e)

        # 遍历 valid blocks
        for valid_block_id in valid4sid[sid]:

            valid_block = valid_blocks[valid_block_id]
            valid_slice = slice(valid_block[0], valid_block[1])

            # ======================================================
            # Step 1: READ (cuda:0, read_stream)
            # ======================================================
            with torch.cuda.device(main_device), torch.cuda.stream(read_stream):

                # 限制 read 的最大超前
                read_stream.wait_event(prev_write_done_list[0])

                _dyn_idx = self.valid_sampler.get_connection(valid_block)  # (T_valid, N_dynbr)

                _sta_loc = self.valid_locations[valid_slice]
                _pos_loc = self.train_locations[_dyn_idx]

                _sta_emb = self.valid_emb[valid_slice]
                _pos_emb = self.train_emb[_dyn_idx]

                data_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                data_ready_event.record(read_stream)

            # ======================================================
            # Step 2: COMPUTE (cuda:sid, comp_stream)
            # ======================================================
            with torch.cuda.device(device), torch.cuda.stream(comp_stream):

                comp_stream.wait_event(data_ready_event)

                sta_loc = _sta_loc.to(device, non_blocking=True)
                pos_loc = _pos_loc.to(device, non_blocking=True)
                sta_emb = _sta_emb.to(device, non_blocking=True)
                pos_emb = _pos_emb.to(device, non_blocking=True)

                cnc_loc = self.connection(sta_loc, dev_num=sid)

                loss_dyn_dyn, _ = self.loom_dyn(
                    None,
                    cnc_loc,
                    sta_loc, pos_loc,
                    sta_emb, pos_emb,
                    None, sid,
                    "valid"
                )

                loss_total = loss_dyn_dyn

                selected_locs, dyn_dyn_loss, _, _ = self.get_best_loc(
                    cnc_loc, loss_total, None, None, sid
                )

                comp_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                comp_ready_event.record(comp_stream)

            # ======================================================
            # Step 3: WRITE (cuda:0, write_stream)
            # ======================================================
            with torch.cuda.device(main_device), torch.cuda.stream(write_stream):

                write_stream.wait_event(comp_ready_event)

                self.new_valid_locations[valid_slice] = selected_locs.to(main_device, non_blocking=True)
                self.tot_dyn_loss      [valid_slice] = dyn_dyn_loss.to(main_device, non_blocking=True)

                # 更新写完成事件窗口
                write_done_event = torch.cuda.Event(enable_timing=False, blocking=False)
                write_done_event.record(write_stream)

                prev_write_done_list.pop(0)
                prev_write_done_list.append(write_done_event)


    @thread_guard
    def train_epoch(self, sid: int):
        for cur_epoch in range(train_epoch_num):
            self.epoch_barrier.wait()
            
            ### step 1: 考虑 train_blocks
            if cur_epoch % step_dyn == 0:
                self.train_dyn_blocks(sid)
            else:
                self.dyn_dyn_loss.zero_()
                self.dyn_sta_loss.zero_()
                self.tot_dyn_loss.zero_()
                
            
            self.epoch_barrier.wait()
            self._synchronize_all_streams()
            
            
            ### step 2: 考虑 vocab_blocks
            self.train_sta_blocks(sid)
            
            self.epoch_barrier.wait()
            self._synchronize_all_streams()

    @thread_guard
    def valid_epoch(self, sid: int):
        for cur_epoch in range(valid_epoch_num):
            self.epoch_barrier.wait()
            self._synchronize_all_streams()
            
            self.valid_dyn_blocks(sid)
            
            self.epoch_barrier.wait()
            self._synchronize_all_streams()
                
                    
    @gettime(fmt='ms', pr=True)
    def train_all(
        self,        
        train_emb : torch.Tensor, # (N_train, dim)
        train_top : torch.Tensor, # (N_train, num_rk)
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
        self.voc_emb       = [vocab_emb.to(device, non_blocking=True) for device in devices]
        
        self.train_sampler = TrainSampler(train_top)

        self.dyn_dyn_loss  = torch.zeros(N_train, device=main_device)
        self.dyn_sta_loss  = torch.zeros(N_train, device=main_device)
        self.tot_dyn_loss  = torch.zeros(N_train, device=main_device)
        self.sta_sta_loss  = torch.zeros(N_vocab, device=main_device)

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
            self.converge  = train_converge is not None and (self.cur_epoch >= train_converge)
            loss_split_record = TrainLossRecord()
            
            if cur_epoch % train_graph_reset == 0 and cur_epoch != 0:
                self.train_sampler.reset_indices()
            
            self.new_train_locations = self.train_locations.clone()
            self.new_vocab_locations = self.vocab_locations.clone()
            
            self.voc_loc       = [self.vocab_locations.to(device, non_blocking=True) for device in devices]
            voc_cnc_loc        = self.connection(self.vocab_locations, dev_num=-1)
            self.voc_cnc_loc   = [voc_cnc_loc.to(device, non_blocking=True) for device in devices]
            self._voc_loss_buf = [torch.zeros((N_vocab, N_C, self.tp), device=device) for device in devices]

            self._synchronize_all_streams()
            
            mark(ED, "epoch_preparation", father="epoch")
            
            
            ### step 3.2: 训练
            mark(ST, "epoch_train")
            
            self.epoch_barrier.wait()          # 第一阶段：train_train
            self.epoch_barrier.wait()          # 第二阶段：train_vocab
            self.epoch_barrier.wait()          # 第三阶段：整合数据
            
            mark(ED, "epoch_train", father="epoch")
            
            
            ### step 3.3: 位置更新（oos train）
            mark(ST, "epoch_pos_train")
            
            self._synchronize_all_streams()
            self.train_locations = self.new_train_locations
            self.vocab_locations = self.new_vocab_locations

            mark(ED, "epoch_pos_train", father="epoch")
            
            
            ### step 3.4: 记录与打印
            loss_split_record.dyn_dyn_loss = self.dyn_dyn_loss.sum().item() / N_train
            loss_split_record.dyn_sta_loss = self.dyn_sta_loss.sum().item() / N_train
            loss_split_record.tot_dyn_loss = self.tot_dyn_loss.sum().item() / N_train
            loss_split_record.sta_sta_loss = self.sta_sta_loss.sum().item() / N_vocab
            
            print(f"epoch {cur_epoch:3d} summary:", end=" ")
            for k, v in loss_split_record.__dict__.items():
                print(f"{k:15s}: {v:.5f}", end=", ")
            print()
            sys.stdout.flush()
            
            ### step 3.5: 验证
            mark(ST, "epoch_valid")
            if cur_epoch % val_interval == 0 or cur_epoch == train_epoch_num - 1: 
                self.print_cross_entropy_loss(cur_epoch, "train")
            mark(ED, "epoch_valid", father="epoch")
            mark(ED, "epoch")
            
            ### step 3.6: 可视化
            if cur_epoch % vis_interval == 0 or cur_epoch == train_epoch_num - 1:
                self.visualize(cur_epoch, "train")
            
            ### step 3.7: 保存
            if (cur_epoch + 1) % save_interval == 0:
                torch.save( 
                    {
                        "train_locations": self.train_locations.cpu(),
                        "expander_graph" : self.train_sampler.expander_graph.cpu(),
                        "vocab_locations": self.vocab_locations.cpu(),
                    },
                    train_save_path.replace(".pt", f"_epoch{cur_epoch + 1}.pt")
                )

        
        for thread in threads:
            thread.join()

        torch.save( 
            {
                "train_locations": self.train_locations.cpu(),
                "expander_graph" : self.train_sampler.expander_graph.cpu(),
                "vocab_locations": self.vocab_locations.cpu(),
            },
            train_save_path
        )

        mark(ED, "all_epoch", father="all")
        mark(ED, "all")
    

    @gettime(fmt='ms', pr=True)
    def valid_all(
        self,
        train_emb : torch.Tensor, # (N_train, dim)
        valid_emb : torch.Tensor, # (N_train, dim)
        valid_top : torch.Tensor, # (N_valid, *)
        vocab_emb : torch.Tensor, # (N_vocab, dim)
        valid_tar : torch.Tensor, # (N_train, ) 
        train_epoch: int,
    ):
        mark(ST, "all")
        mark(ST, "all_preparation")
        
        ### step.2 准备数据分块与采样器
        mark(ST, "all_preparation_2")
        train_data         = torch.load(train_save_path.replace(".pt", f"_epoch{train_epoch}.pt"))
        
        self.train_emb     = train_emb.to(main_device)
        self.valid_emb     = valid_emb.to(main_device)
        self.vocab_emb     = vocab_emb.to(main_device)
        self.valid_tar     = valid_tar.long().to(main_device)
        
        self.valid_sampler = ValidSampler(train_data['expander_graph'], valid_top)

        self.tot_dyn_loss  = torch.zeros(N_valid, device=main_device)

        self.train_locations     = train_data['train_locations'].to(main_device)
        self.vocab_locations     = train_data['vocab_locations'].to(main_device)
        self.new_valid_locations = self.valid_locations.clone()
        
        self._synchronize_all_streams()
        mark(ED, "all_preparation_2", father="all_preparation")
        
        
        ### step 3: 开启多线程
        mark(ST, "all_preparation_3")
        self.epoch_barrier  = threading.Barrier(num_devices + 1) # num_devices 个生产线程，num_devices 个消费线程，1 个主线程
        threads: List[threading.Thread] = []
        for i, _ in enumerate(devices):
            thread = threading.Thread(target=self.valid_epoch, args=(i,))
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
            self.converge  = valid_converge is not None and (self.cur_epoch >= valid_converge)
            loss_split_record = ValidLossRecord()
            loss_split_record.tot_dyn_loss = 0.0
            
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
            loss_split_record.tot_dyn_loss = self.tot_dyn_loss.sum().item() / N_valid
            
            print(f"epoch {cur_epoch:3d} summary:", end=" ")
            for k, v in loss_split_record.__dict__.items():
                print(f"{k:15s}: {v:.5f}", end=", ")
            print()
            sys.stdout.flush()
            
            ### step 3.5: 验证
            mark(ST, "epoch_valid")
            if cur_epoch % val_interval == 0 or cur_epoch == train_epoch_num - 1:
                self.print_cross_entropy_loss(cur_epoch, "valid")
            mark(ED, "epoch_valid", father="epoch")
            mark(ED, "epoch")
            
            ### step 3.6: 可视化
            if cur_epoch % vis_interval == 0 or cur_epoch == train_epoch_num - 1:
                self.visualize(cur_epoch, "valid")
        
        for thread in threads:
            thread.join()

        # torch.save(
        #     {
        #         "train_locations": self.train_locations.cpu(),
        #         "vocab_locations": self.vocab_locations.cpu(),
        #         "valid_locations": self.valid_locations.cpu(),
        #     },       
        #     valid_save_path
        # )

        mark(ED, "all_epoch", father="all")
        mark(ED, "all")
    
    
    
    
    
    def print_cross_entropy_loss(self, epoch: int, cur_type: str):
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

        print(f"{cur_type:5s} loss: {loss:.6f}, {cur_type:5s} acc: {accuracy:.4f}")
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
            
        for stream in read_streams + comp_streams + write_streams:
            stream.synchronize()

        
if __name__ == "__main__":
    pass

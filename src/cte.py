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
from src.vis import *
from dataclasses import dataclass

main_device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
from typing import List, Tuple

@dataclass
class TrainLossRecord:
    dyn_dyn_loss: float = 0.0
    dyn_sta_loss: float = 0.0
    tot_dyn_loss: float = 0.0
    sta_sta_loss: float = 0.0

@dataclass
class ValidLossRecord:
    dyn_dyn_loss: float = 0.0
    dyn_sta_loss: float = 0.0
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
        
        self.use_dyn_sta = True
     
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
        flipped_ints = ori_int.unsqueeze(1) ^ flip_masks # (N, h, tp)
        random_masks = self.connection_masks(flipped_ints.size(0), dev_num=dev_num) # (N, h, K, tp)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h * N_K, self.tp)
        # (N, h, 1, tp) ^ (N, h, K, tp) -> (N, h, K, tp) -> (N, h*K, tp)
        loc = torch.cat((result, ori_int.unsqueeze(1), -result), dim=1) # (B*T1, H*K + 1 + H*K, D)
        return loc
    
    '''
    def _get_max_flip_level(self, device):
        """
        返回当前允许翻转的最高 bit 层数 l_max \in [0, h-1]。
        epoch 越大，l_max 越小 -> 搜索半径越小。
        """

        if self.cur_epoch < 60:
            max_level = self.h - 1
        elif self.cur_epoch < 80:
            max_level = self.h // 2
        else:
            max_level = self.h // 4
        print(f"Current epoch: {self.cur_epoch}, max flip level: {max_level}")
        
        return max_level

    def connection(self, ori_int: torch.Tensor, dev_num=0):
        device = devices[dev_num] if dev_num >= 0 else main_device

        # 1. 原来的 flip + random 逻辑，先生成完整 result
        flip_masks = (1 << torch.arange(self.h, device=device, dtype=ori_int.dtype)).unsqueeze(0).unsqueeze(2)
        flipped_ints = ori_int.unsqueeze(1) ^ flip_masks  # (B*T1, H, D)

        random_masks = self.connection_masks(flipped_ints.size(0), dev_num=dev_num)
        # random_masks: (B*T1, H, K, D)

        BT1 = flipped_ints.size(0)

        # 先保持和原来一样的形状: (B*T1, H, K, D)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(BT1, self.h, N_K, self.tp)

        # 2. 根据 epoch 限制可用的最大翻转层 l_max
        l_max = self._get_max_flip_level(device)

        # 如果 l_max < h-1，禁用更高的层（让这些层的候选点退化为原坐标）
        if l_max < self.h - 1:
            levels = torch.arange(self.h, device=device)
            disabled = levels > l_max                    # shape: (H,)
            if disabled.any():
                # (B*T1, 1, 1, D)，可广播到禁用层
                ori_expanded = ori_int.unsqueeze(1).unsqueeze(2)  # (B*T1, 1, 1, D)
                # 对于被禁用的层：正分支候选 = 原坐标
                result[:, disabled, :, :] = ori_expanded

        # 3. 展开成 (B*T1, H*K, D)，并构造正/原/负三段
        result_flat = result.view(BT1, self.h * N_K, self.tp)

        # 注意：负分支我们也希望在禁用层上“保持原位”（不做大步长）
        neg_result_flat = -result_flat.clone()
        if l_max < self.h - 1:
            levels = torch.arange(self.h, device=device)
            disabled = levels > l_max
            if disabled.any():
                ori_expanded = ori_int.unsqueeze(1).unsqueeze(2)  # (B*T1, 1, 1, D)
                neg_result = neg_result_flat.view(BT1, self.h, N_K, self.tp)
                neg_result[:, disabled, :, :] = ori_expanded
                neg_result_flat = neg_result.view(BT1, self.h * N_K, self.tp)

        loc = torch.cat((result_flat, ori_int.unsqueeze(1), neg_result_flat), dim=1)
        # 形状仍然是 (B*T1, H*K + 1 + H*K, D)，即 (B*T1, N_C, D)
        return loc
    '''
                    
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
        loss    : torch.Tensor, # (T, C, D)
        loss_1  : torch.Tensor,
        loss_2  : torch.Tensor,
        loss_3  : torch.Tensor,
        sid     : int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ### cnc_loc[i, c, k]: 第 i 个起始样本，在第 k 个维度上，我们将其 loc 选为 cnc_loc[i, c, k]
        ### loss   [i, c, k]: 如果对于第 i 个起始样本，我们只将第 k 个维度的 loc 选为 cnc_loc[i, c, k]，那么对应的 loss 是 loss[i, c, k]
        ### 所以，最稳妥的方式是：对于每个样本 i，将哪个维度 k 上的 loc 选为 cnc_loc[i, c, k] 时， loss 最小，就选哪个 (k, c) 对
        
        
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
        t_idx  = torch.randperm(T, device=device, generator=generator)[:, None].expand(T, cur_tp)  # (T, cur_tp)
        t_mask = torch.rand    (T, device=device, generator=generator) < cur_portion  # (T,)
        t_idx, rand_cols = t_idx[t_mask],   rand_cols[t_mask]  # (T_upd, cur_tp), (T_upd, cur_tp)
        # t_idx, rand_cols = t_idx[:cur_num], rand_cols[:cur_num]  # (cur_num, cur_tp), (cur_num, cur_tp)
        
        # if cur_num == 1 and loss_2 == None and loss_3 == None:
        #     t_index = t_idx[0].item()
        #     d_index = rand_cols[0, 0].item()
        #     ori_loc = cnc_loc[t_index, N_K * self.h, d_index].item()
        #     upd_loc = cnc_loc[t_index, argmin_all[t_index, d_index], d_index].item()
        #     print(f"Static 选择中，被选中的是，第 {t_index} 个样本，在第 {d_index} 个维度上")
        #     print(f"原本的位置为 {ori_loc}")
        #     print(f"更新的位置为 {upd_loc}")
        #     print(f"原本的 loss 为 {loss[t_index, N_K * self.h, d_index].item():.6f}")
        #     print(f"更新的 loss 为 {loss[t_index, argmin_all[t_index, d_index], d_index].item():.6f}")
            
        
        cnc_indices[t_idx, rand_cols] = argmin_all[t_idx, rand_cols]

        

        T_indices     = torch.arange(T              ,    device=device)[:, None]         # (T, 1)
        dim_indices   = torch.arange(self.tp    ,        device=device)[None  :]         # (1, D)
        selected_locs = cnc_loc [T_indices, cnc_indices,  dim_indices]                   # (T, D)
        
        real_loss     = loss    [T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, )
        real_loss1    = loss_1  [T_indices, cnc_indices, dim_indices].mean(dim=-1) if loss_1 is not None else None       
                                                                                         # (T, )
        real_loss2    = loss_2  [T_indices, cnc_indices, dim_indices].mean(dim=-1) if loss_2 is not None else None        
                                                                                         # (T, )
        real_loss3    = loss_3  [T_indices, cnc_indices, dim_indices].mean(dim=-1) if loss_3 is not None else None        
        
        ### 这个 mean 到底在干什么？？？
        
        return selected_locs, real_loss, real_loss1, real_loss2, real_loss3

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
            torch.ones((T_train if cur_type=="train" else T_valid, N_nbr if cur_type=="train" else N_nbr_v), device=device, dtype=torch.float32).contiguous(),
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
        
        # assert -1.0 <= ct_val.min() - 1e-3 and ct_val.max() - 1e-3 <= 1.0, f"ct_val 超出范围: [{ct_val.min().item()}, {ct_val.max().item()}]"
        # assert -1.0 <= eu_val.min() - 1e-3 and eu_val.max() - 1e-3 <= 1.0, f"eu_val 超出范围: [{eu_val.min().item()}, {eu_val.max().item()}]"
        
        # ori_ct_val = ct_val.clone()
        # ct_val = torch.sqrt((1. - (ct_val - 1e-3)) / 2.)
        # ori_eu_val = eu_val.clone()
        # eu_val = torch.sqrt((1. - (eu_val - 1e-3)) / 2.)
        
        # (T_train, N_ttnbr + K_vocab, N_C, dim_ct)
        ## i∈[0, T), j∈[0, S), c∈[0, N_C), k∈[0, tp)
        ## 对于第 i 的起始样本的第 k 维度上的 loc，向第 (i, j) 个目标样本的第 k 维度的 loc 计算相似度，得到 (i, j, k) 的相似度
        ## cnc_loc[i, c, k] 表示的则是，第 i 个起始样本在第 k 个维度上的第 c 个候选 loc，是什么？
        ## 最终算出来的 ct_val[i, j, c, k]，表示的是，第 i 个起始样本，如果只将第 k 维度的 loc 改为 cnc_loc[i, c, k]，那么，它与第 (i, j) 个目标样本的相似度（对所有维度平均）是多少？
        ## 缺省值被设置为 c = N_K * h，即“保持不变”的 loc; 所以，ct_val[i, j, N_K * h, k] 表示的就是，第 i 个起始样本，在第 k 维度上保持不变时，与第 (i, j) 个目标样本的相似度。

        num = N_dynbr if cur_type == "train" else N_dynbr_v
        ### step 3: 计算 loss
        # loss_dyn_dyn = (
        #     torch.abs(ori_eu_val[:, :num :, :] - ori_ct_val[:, :num, :, :]) * F.softmax(ori_eu_val[:, :num, :, :] * 20 / temperature, dim=1)
        #     #* torch.abs(ori_eu_val[:, :num, :, :]) # (T_train, N_dynbr + N_stnbr, N_C, dim_ct)
        # ).mean(dim=1)  # (T_train, N_C, dim_ct)
        # loss_dyn_dyn = (
        #     ( F.log_softmax(eu_val[:, :N_dynbr, :, :] * 20 / temperature, dim=1) 
        #     - F.log_softmax(ct_val[:, :N_dynbr, :, :] * 20 / temperature, dim=1)) 
        #     * F.softmax(eu_val[:, :N_dynbr, :, :] * 20 / temperature, dim=1)
        # ).sum(dim=1)  # (T_train, N_C, dim_ct)
        
        
        loss_dyn_dyn = sampled_softmax_loss(
            eu_val[:, :num, :, :], 
            ct_val[:, :num, :, :], 
            num, N_top, temperature, N_train
        ) # (T_train, N_C, dim_ct)
        
    
        loss_dyn_sta = (
            ( F.log_softmax(eu_val[:, num:, :, :] * 20 / temperature, dim=1) 
            - F.log_softmax(ct_val[:, num:, :, :] * 20 / temperature, dim=1)) * 
                F.softmax(eu_val[:, num:, :, :]   * 20 / temperature, dim=1)
        ).sum(dim=1)  # (T_train, N_C, dim_ct)
        
        # if cur_type == "train":
        #     # loss_dyn_sta = (
        #     #     torch.abs(ori_eu_val[:, num:, :, :] - ori_ct_val[:, num:, :, :]) * F.softmax(ori_eu_val[:, num:, :, :] * 20 / gt_temperature, dim=1)
        #     # ).mean(dim=1)       
            
            
            
        #     # loss_dyn_gt = - F.log_softmax(ct_val[:, num:, :, :] * 20 / gt_temperature, dim=1).gather(
        #     #     dim=1,
        #     #     index=cur_tar[:, None, None, None].expand(-1, -1, N_C, tp)
        #     # ).squeeze(1) # (T_train, N_C, dim_ct)
            
        #     # loss_dyn_gt = torch.zeros_like(loss_dyn_sta)
        # else:
        #     loss_dyn_sta = None
        #     # loss_dyn_gt  = None
        
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
        voc_loc = self.voc_loc[sid].unsqueeze(0).repeat(T_vocab, 1, 1) # (N_vocab, dim_ct) -> (T_vocab, N_vocab, dim_ct)
        voc_emb = self.voc_emb[sid].unsqueeze(0).repeat(T_vocab, 1, 1) # (N_vocab, dim_eu) -> (T_vocab, N_vocab, dim_eu)

        ### step 1: 计算欧氏空间的相似度
        eu_val = (sta_emb[:, None, :] @ voc_emb.transpose(-1, -2)).squeeze()[:, :, None, None]  
            ### (T_vocab, 1, dim_eu) @ (T_vocab, dim_eu, N_vocab) -> (T_vocab, 1, N_vocab) -> (T_vocab, N_vocab, 1, 1)
        # print(eu_val[:4, :4, 0, 0])

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
        
        # assert -1.0 <= ct_val.min() - 1e-3 and ct_val.max() - 1e-3 <= 1.0, f"ct_val 超出范围: [{ct_val.min().item()}, {ct_val.max().item()}]"
        # assert -1.0 <= eu_val.min() - 1e-3 and eu_val.max() - 1e-3 <= 1.0, f"eu_val 超出范围: [{eu_val.min().item()}, {eu_val.max().item()}]"
        
        # ori_ct_val = ct_val.clone()
        # ct_val = torch.sqrt((1. - (ct_val - 1e-3)) / 2.)
        # ori_eu_val = eu_val.clone()
        # eu_val = torch.sqrt((1. - (eu_val - 1e-3)) / 2.)

        ### step 3: 计算 loss
        # loss_sta_sta = (
        #     torch.abs(ori_eu_val - ori_ct_val) * torch.abs(ori_eu_val) * F.softmax(ori_eu_val * 20 / temperature, dim=1)
        #     # * torch.abs(eu_val) # (T_train, N_dynbr + N_stnbr, N_C, dim_ct)
        # ).mean(dim=1)  # (T_train, N_C, dim_ct)
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
                # print("dyn_cnc_loc", cnc_loc[0, :5, 0])
                
                loss_dyn_dyn, loss_dyn_sta = self.loom_dyn(
                    cur_tar,
                    cnc_loc,
                    sta_loc, pos_loc,
                    sta_emb, pos_emb,
                    None, sid, "train"
                )
                if self.use_dyn_sta:
                    loss_total = ratio_dyn * loss_dyn_dyn + ratio_sta * loss_dyn_sta # + ratio_gt * loss_dyn_gt
                else:
                    loss_total = loss_dyn_dyn

                selected_locs, tot_dyn_loss, dyn_dyn_loss, dyn_sta_loss, _ = self.get_best_loc(
                    cnc_loc, loss_total, loss_dyn_dyn, loss_dyn_sta, None, sid
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
                # self.dyn_gt_loss [train_slice]        = dyn_gt_loss .to(main_device, non_blocking=True)

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

                selected_locs, sta_sta_loss, _, _, _ = self.get_best_loc(
                    cnc_loc, loss, None, None, None, sid
                )

                comp_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                comp_ready_event.record(comp_stream)

            # ======================================================
            # Step 3: WRITE (in cuda:0, write_stream)
            # ======================================================
            with torch.cuda.device(main_device), torch.cuda.stream(write_stream):

                write_stream.wait_event(comp_ready_event)

                self.new_vocab_locations[vocab_slice] = selected_locs.to(main_device, non_blocking=True)
                self.sta_sta_loss       [vocab_slice] = sta_sta_loss.to(main_device, non_blocking=True)

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

                _dyn_idx, _voc_idx = self.valid_sampler.get_connection(valid_block)  # (T_valid, N_dynbr), (T_valid, N_stnbr)

                _sta_loc = self.valid_locations[valid_slice]
                _pos_loc = torch.cat([
                    self.train_locations[_dyn_idx],
                    self.vocab_locations[_voc_idx],
                ], dim=1)

                _sta_emb = self.valid_emb[valid_slice]
                _pos_emb = torch.cat([
                    self.train_emb[_dyn_idx],
                    self.vocab_emb[_voc_idx],
                ], dim=1)

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

                loss_dyn_dyn, loss_dyn_sta = self.loom_dyn(
                    None,
                    cnc_loc,
                    sta_loc, pos_loc,
                    sta_emb, pos_emb,
                    None, sid,
                    "valid"
                )

                loss_total = ratio_dyn * loss_dyn_dyn + ratio_sta * loss_dyn_sta

                selected_locs, tot_dyn_loss, dyn_dyn_loss, dyn_sta_loss, _ = self.get_best_loc(
                    cnc_loc, loss_total, loss_dyn_dyn, loss_dyn_sta, None, sid
                )

                comp_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                comp_ready_event.record(comp_stream)

            # ======================================================
            # Step 3: WRITE (cuda:0, write_stream)
            # ======================================================
            with torch.cuda.device(main_device), torch.cuda.stream(write_stream):

                write_stream.wait_event(comp_ready_event)

                self.new_valid_locations[valid_slice] = selected_locs.to(main_device, non_blocking=True)
                self.tot_dyn_loss       [valid_slice] = tot_dyn_loss.to(main_device, non_blocking=True)
                self.dyn_dyn_loss       [valid_slice] = dyn_dyn_loss.to(main_device, non_blocking=True)
                self.dyn_sta_loss       [valid_slice] = dyn_sta_loss.to(main_device, non_blocking=True)

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
        torch.cuda.nvtx.range_push("all")
        
        mark(ST, "all_preparation")
        torch.cuda.nvtx.range_push("all_preparation")
        
        ### step.2 准备数据分块与采样器
        mark(ST, "all_preparation_2")
        self.train_tar     = train_tar.to(main_device)
        self.train_emb     = train_emb.to(main_device)
        self.vocab_emb     = vocab_emb.to(main_device)
        self.voc_emb       = [vocab_emb.to(device, non_blocking=True) for device in devices]
        
        self.train_sampler = TrainSampler(train_top)

        self.dyn_dyn_loss  = torch.zeros(N_train, device=main_device)
        self.dyn_sta_loss  = torch.zeros(N_train, device=main_device)
        # self.dyn_gt_loss   = torch.zeros(N_train, device=main_device)
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
        torch.cuda.nvtx.range_pop() # all_preparation
        
        
        ### step 3.5: 验证
        print("Initial validation:")
        self.print_cross_entropy_loss(-1, "train")
        self.visualize(-1, "train")
        
        
        mark(ST, "all_epoch")
        torch.cuda.nvtx.range_push("all_epoch")
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
            # print("voc_cnc_loc", voc_cnc_loc[0, :50, 0])
            self.voc_cnc_loc   = [voc_cnc_loc.to(device, non_blocking=True) for device in devices]

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
            # loss_split_record.dyn_gt_loss  = self.dyn_gt_loss .sum().item() / N_train
            loss_split_record.tot_dyn_loss = self.tot_dyn_loss.sum().item() / N_train
            loss_split_record.sta_sta_loss = self.sta_sta_loss.sum().item() / N_vocab
            
            print(f"epoch {cur_epoch:3d} summary:", end=" ")
            for k, v in loss_split_record.__dict__.items():
                print(f"{k:15s}: {v:.6f}", end=", ")
            print()
            sys.stdout.flush()
            
            # if loss_split_record.dyn_sta_loss < 0.02:
            #     self.use_dyn_sta = False
            # else:
            #     self.use_dyn_sta = True
            
            ### step 3.5: 验证
            mark(ST, "epoch_valid")
            if cur_epoch % val_interval == 0 or cur_epoch == train_epoch_num - 1: 
                self.print_cross_entropy_loss(cur_epoch, "train")
            mark(ED, "epoch_valid", father="epoch")
            mark(ED, "epoch")
            
            ### step 3.6: 可视化
            if cur_epoch % vis_interval == 0 or cur_epoch == train_epoch_num - 1:
                self.visualize(cur_epoch, "train")
                self.print_dyn_dyn_tot_loss(cur_epoch, "train")
                self.print_dyn_sta_tot_loss(cur_epoch, "train")
            
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
        torch.cuda.nvtx.range_pop() # all_epoch
        
        mark(ED, "all")
        torch.cuda.nvtx.range_pop() # all
    

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

        self.dyn_dyn_loss  = torch.zeros(N_valid, device=main_device)
        self.dyn_sta_loss  = torch.zeros(N_valid, device=main_device)
        self.tot_dyn_loss  = torch.zeros(N_valid, device=main_device)

        self.train_locations     = train_data['train_locations'].to(main_device)
        self.vocab_locations     = train_data['vocab_locations'].to(main_device)
        self.valid_locations     = torch.randint(
            1 - self.n, self.n, (N_valid, self.tp), 
            dtype=torch.int64, device=main_device, generator=main_generator
        )
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
            loss_split_record.dyn_dyn_loss = 0.0
            loss_split_record.dyn_sta_loss = 0.0
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
            loss_split_record.dyn_dyn_loss = self.dyn_dyn_loss.sum().item() / N_valid
            loss_split_record.dyn_sta_loss = self.dyn_sta_loss.sum().item() / N_valid
            loss_split_record.tot_dyn_loss = self.tot_dyn_loss.sum().item() / N_valid
            
            print(f"epoch {cur_epoch:3d} summary:", end=" ")
            for k, v in loss_split_record.__dict__.items():
                print(f"{k:15s}: {v:.5f}", end=", ")
            print()
            sys.stdout.flush()
            

            
            
            ### step 3.5: 验证
            mark(ST, "epoch_valid")
            if cur_epoch % val_interval == 0 or cur_epoch == valid_epoch_num - 1:
                self.print_cross_entropy_loss(cur_epoch, "valid")
            mark(ED, "epoch_valid", father="epoch")
            mark(ED, "epoch")
            
            ### step 3.6: 可视化
            if cur_epoch % vis_interval == 0 or cur_epoch == valid_epoch_num - 1:
                self.visualize(cur_epoch, "valid")
                self.print_dyn_dyn_tot_loss(cur_epoch, "valid")
                self.print_dyn_sta_tot_loss(cur_epoch, "valid")
                # self.visualize_dyn_sta_mds_tsne(cur_epoch)
        
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
    
    
    
    def print_dyn_dyn_tot_loss(self, epoch: int, cur_type: str):
        # === 计算 dyn-dyn 总 loss（softmax 版）===
        # (N_train, N_train) or (N_valid, N_train)
        # 使用 softmax(KL) 而不是 sqrt 后的 L1

        def get_eu_ct_val(o_block, i_block, cur_type: str, type: str = "ori"):
            if cur_type == "train":
                sta_loc = self.train_locations[o_block[0]:o_block[1]]
                sta_emb = self.train_emb[o_block[0]:o_block[1]]
            else:
                sta_loc = self.valid_locations[o_block[0]:o_block[1]]
                sta_emb = self.valid_emb[o_block[0]:o_block[1]]

            pos_loc = self.train_locations[i_block[0]:i_block[1]]
            pos_emb = self.train_emb[i_block[0]:i_block[1]]

            eu_val = sta_emb @ pos_emb.t()    # (T_outer, T_inner)
            ct_val = self.cos_similarity(
                sta_loc[:, None, :],
                pos_loc[None, :, :],
            ).mean(dim=-1)

            assert -1.0 <= eu_val.min() - 1e-3 and eu_val.max() - 1e-3 <= 1.0
            assert -1.0 <= ct_val.min() - 1e-3 and ct_val.max() - 1e-3 <= 1.0

            if type == "ori":
                return eu_val, ct_val
            else:
                raise ValueError("dyn-dyn only uses ori values")

        # ------------------------------------------------
        outer_splits = make_splits(0, N_train, 1024) if cur_type == "train" \
                    else make_splits(0, N_valid, 1024)
        inner_splits = make_splits(0, N_train, 1024)

        total_loss = 0.0
        total_outer = 0

        # 直方图：统计每个 outer 的 KL
        num_bins = 1000
        hist_range = (0.0, 1.0)
        bin_width = (hist_range[1] - hist_range[0]) / num_bins
        kl_hist = torch.zeros(num_bins, device=main_device, dtype=torch.float64)

        def update_hist(values: torch.Tensor, hist_tensor: torch.Tensor):
            v = torch.clamp(values, 0.0, 1.0)
            idx = (v / bin_width).long().clamp(0, num_bins - 1)
            hist_tensor += torch.bincount(idx, minlength=num_bins).to(hist_tensor.dtype)

        # ------------------------------------------------
        for o_block in outer_splits:
            T_outer = o_block[1] - o_block[0]

            # ---------- Pass 1: 计算 softmax 分母 ----------
            exp_sum_eu = torch.zeros(T_outer, device=main_device)
            exp_sum_ct = torch.zeros(T_outer, device=main_device)

            for i_block in inner_splits:
                eu_val, ct_val = get_eu_ct_val(o_block, i_block, cur_type, type="ori")

                eu_logits = 20.0 * eu_val / temperature          # (T_outer, T_inner)
                ct_logits = 20.0 * ct_val / temperature

                exp_sum_eu += torch.exp(eu_logits).sum(dim=1)
                exp_sum_ct += torch.exp(ct_logits).sum(dim=1)

            # ---------- Pass 2: 累 KL ----------
            kl_outer = torch.zeros(T_outer, device=main_device)

            for i_block in inner_splits:
                eu_val, ct_val = get_eu_ct_val(o_block, i_block, cur_type, type="ori")

                eu_logits = 20.0 * eu_val / temperature
                ct_logits = 20.0 * ct_val / temperature

                exp_eu = torch.exp(eu_logits)
                exp_ct = torch.exp(ct_logits)

                p_eu = exp_eu / exp_sum_eu[:, None]

                # KL 累加项
                kl_outer += (p_eu * (
                    (eu_logits - torch.log(exp_sum_eu[:, None])) -
                    (ct_logits - torch.log(exp_sum_ct[:, None]))
                )).sum(dim=1)

            total_loss += kl_outer.sum().item()
            total_outer += T_outer

            update_hist(kl_outer.detach(), kl_hist)


        # visualize_loss_hist(
        #     kl_hist,
        #     "KL",
        #     num_bins=100,
        #     save_path=os.path.join(vis_path, f"dd_{cur_type}_kl_hist_epoch_{epoch:04d}.png")
        # )

        avg_loss = total_loss / total_outer
        print(f"{cur_type:5s} dyn-dyn KL loss: {avg_loss:.6f}")

    def print_dyn_sta_tot_loss(self, epoch: int, cur_type: str):
        # === 计算 dyn-sta 总 loss ===
        # (N_train, N_vocab), (N_valid, N_vocab) 的分布
        
        def get_eu_ct_val(block, cur_type: str, type: str = 'metric'):
            if cur_type == "train":
                sta_loc = self.train_locations[block[0]:block[1]]
                sta_emb = self.train_emb      [block[0]:block[1]]
            else:
                sta_loc = self.valid_locations[block[0]:block[1]]
                sta_emb = self.valid_emb      [block[0]:block[1]]
            
            pos_loc = self.vocab_locations
            pos_emb = self.vocab_emb      
            
            
            eu_val = sta_emb @ pos_emb.t()    # (T_outer, T_inner)
            ct_val = self.cos_similarity(
                sta_loc[:, None, :],
                pos_loc[None, :, :],
            ).mean(dim=-1)   # (T_outer, T_inner)
            
            if type == 'metric':
                assert -1.0 <= ct_val.min() - 1e-3 and ct_val.max() - 1e-3 <= 1.0, f"ct_val 超出范围: [{ct_val.min().item()}, {ct_val.max().item()}]"
                assert -1.0 <= eu_val.min() - 1e-3 and eu_val.max() - 1e-3 <= 1.0, f"eu_val 超出范围: [{eu_val.min().item()}, {eu_val.max().item()}]"
                
                eu_val = torch.sqrt((1. - (eu_val - 1e-3)) / 2.)
                ct_val = torch.sqrt((1. - (ct_val - 1e-3)) / 2.)
            
            
            
            return eu_val, ct_val


        # # ============================================================
        # # 追加可视化：抽取若干 token，看它们与 static 的损失曲线
        # # ============================================================

        def visualize_dyn_sta_detail(cur_type: str, epoch: int, num_samples: int = 5):
            # 1. 获取数据源
            if cur_type == "train":
                total_N = N_train
                targets_all = self.train_tar
                sta_emb_all = self.train_emb
                sta_loc_all = self.train_locations
            else:
                total_N = N_valid
                targets_all = self.valid_tar
                sta_emb_all = self.valid_emb
                sta_loc_all = self.valid_locations

            voc_emb_all = self.vocab_emb
            vocab_loc_all = self.vocab_locations

            # 2. 均匀采样
            sample_indices = torch.linspace(0, total_N - 1, num_samples, dtype=torch.long)

            # 三行可视化
            plt.figure(figsize=(6 * num_samples, 15))

            for idx_i, idx in enumerate(sample_indices):
                block = (idx.item(), idx.item() + 1)
                target = targets_all[idx].unsqueeze(0)  # (1,)

                # ======== (A) 获取原始相似度 sim ∈ [-1,1] ========
                eu_ori, ct_ori = get_eu_ct_val(block, cur_type, type='ori')
                eu_ori = eu_ori[0]      # (V,)
                ct_ori = ct_ori[0]      # (V,)

                # ======== 构造 teacher logits ========
                sta_emb = sta_emb_all[idx:idx+1]          # (1, dim)
                voc_emb = voc_emb_all                     # (V, dim)
                eu_logits = 20.0 * (sta_emb @ voc_emb.t())  # (1, V)
                eu_logits_1d = eu_logits[0]  # (V,)

                # ======== 构造 student logits ========
                if use_eu_norm:
                    eu_norm = (torch.norm(sta_emb, dim=-1, keepdim=True)
                               @ torch.norm(voc_emb, dim=-1, keepdim=True).t())  # (1, V)
                else:
                    eu_norm = torch.ones_like(eu_logits) * 20.

                ct_logits = ct_ori.unsqueeze(0) * eu_norm   # (1, V)
                ct_logits_1d = ct_logits[0]

                # ======== (B) sqrt 转换版本 ========
                eu_sqrt = torch.sqrt((1. - (eu_ori - 1e-3)) / 2.0)
                ct_sqrt = torch.sqrt((1. - (ct_ori - 1e-3)) / 2.0)
                # ×20 对齐范围
                eu_sqrt_20 = 20.0 * eu_sqrt
                ct_sqrt_20 = 20.0 * ct_sqrt

                # ======== (C) softmax 值 ========
                p_eu = torch.softmax(eu_logits, dim=-1)[0]
                p_ct = torch.softmax(ct_logits, dim=-1)[0]

                # ======== 排序依据：原始 logits ========
                sort_idx = torch.argsort(eu_logits_1d, descending=True)

                eu_sorted      = eu_logits_1d[sort_idx].cpu().numpy()
                ct_sorted      = ct_logits_1d[sort_idx].cpu().numpy()
                eu_sqrt_sorted = eu_sqrt_20[sort_idx].cpu().numpy()
                ct_sqrt_sorted = ct_sqrt_20[sort_idx].cpu().numpy()
                p_eu_sorted    = p_eu[sort_idx].cpu().numpy()
                p_ct_sorted    = p_ct[sort_idx].cpu().numpy()

                # ======== 计算 CE ========
                ce_teacher = F.cross_entropy(eu_logits, target).item()
                ce_student = F.cross_entropy(ct_logits, target).item()

                # ============================================================
                # 第一行：原始 logits [-20, 20]
                # ============================================================
                plt.subplot(3, num_samples, idx_i + 1)
                plt.plot(eu_sorted, label="eu (blue)", color="blue")
                plt.scatter(np.arange(len(eu_sorted)), eu_sorted, s=5, color="blue")
                plt.plot(ct_sorted, label="ct (red)", color="red")

                plt.title(f"{cur_type} sample #{idx.item()}")
                plt.xlabel("sorted by logits (desc)")
                plt.ylabel("logits value")
                plt.ylim(-20.0, 20.0)

                plt.text(
                    0.02, 0.95,
                    f"CE_teacher = {ce_teacher:.4f}\nCE_student = {ce_student:.4f}",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment='top'
                )

                if idx_i == 0:
                    plt.legend()

                # ============================================================
                # 第二行：sqrt 变换后 × 20 的值  [0, 20]
                # ============================================================
                plt.subplot(3, num_samples, num_samples + idx_i + 1)
                plt.plot(eu_sqrt_sorted, label="eu_sqrt", color="blue")
                plt.scatter(np.arange(len(eu_sqrt_sorted)), eu_sqrt_sorted, s=4, color="blue")
                plt.plot(ct_sqrt_sorted, label="ct_sqrt", color="red")
                plt.scatter(np.arange(len(ct_sqrt_sorted)), ct_sqrt_sorted, s=4, color="red")

                plt.xlabel("sorted by logits (desc)")
                plt.ylabel("sqrt value ×20")
                plt.ylim(0.0, 20.0)

                # ============================================================
                # 第三行：softmax prob [0, 1]
                # ============================================================
                plt.subplot(3, num_samples, 2 * num_samples + idx_i + 1)
                plt.plot(p_eu_sorted, label="p_eu", color="blue")
                plt.scatter(np.arange(len(p_eu_sorted)), p_eu_sorted, s=4, color="blue")
                plt.plot(p_ct_sorted, label="p_ct", color="red")
                plt.scatter(np.arange(len(p_ct_sorted)), p_ct_sorted, s=4, color="red")

                plt.xlabel("sorted by logits (desc)")
                plt.ylabel("softmax prob")
                plt.ylim(0.0, 1.0)

            # 保存图像
            out_path = os.path.join(vis_path, f"{cur_type}_sample_curve_epoch_{epoch:04d}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            # print(f"[OK] saved dyn-sta detail vis to {out_path}")


        visualize_dyn_sta_detail(cur_type, epoch, num_samples=5)


    def print_cross_entropy_loss(self, epoch: int, cur_type: str):
        loss     = 0
        accuracy = 0

        # === 收集“原来=eu_logits预测”和“现在=ct_logits预测” ===
        all_old_pred = []
        all_now_pred = []

        splits   = train_blocks if cur_type == "train" else valid_blocks

        for i, block in enumerate(splits):
            targets     = self.train_tar[block[0]:block[1]] if cur_type == "train" else self.valid_tar[block[0]:block[1]]
            sta_loc     = self.train_locations[block[0]:block[1]] if cur_type == "train" else self.valid_locations[block[0]:block[1]]
            voc_loc     = self.vocab_locations
            sta_emb     = self.train_emb[block[0]:block[1]] if cur_type == "train" else self.valid_emb[block[0]:block[1]]
            voc_emb     = self.vocab_emb

            # ============ 现在：ct space logits ============
            ct_val = self.cos_similarity(
                sta_loc[:, None, :],
                voc_loc[None, :, :],
            ).mean(dim=-1)   # (T, V)

            if use_eu_norm:
                eu_cro_nrm = (torch.norm(sta_emb, dim=-1, keepdim=True) @
                            torch.norm(voc_emb, dim=-1, keepdim=True).t()).squeeze()
            else:
                eu_cro_nrm = torch.ones_like(ct_val) * 20.

            ct_logits = ct_val * eu_cro_nrm                # (T, V) —— student logits

            # ============ 原来：eu space logits ============
            # 完全遵循之前 GPT 推理的定义
            eu_logits = 20. * (sta_emb @ voc_emb.t())      # (T, V) —— teacher logits

            # ============ loss / acc（你的 student）===========
            loss       += F.cross_entropy(ct_logits, targets, reduction='sum').item()
            pred_now    = ct_logits.argmax(dim=-1)
            accuracy   += (pred_now == targets).sum().item()

            # ============ 保存 ============
            pred_old = eu_logits.argmax(dim=-1)

            all_old_pred.append(pred_old)
            all_now_pred.append(pred_now)


        # ============ 聚合全部预测结果 ============
        all_old_pred = torch.cat(all_old_pred, dim=0)
        all_now_pred = torch.cat(all_now_pred, dim=0)

        total_N = (N_train if cur_type == "train" else N_valid)

        loss     /= total_N
        accuracy /= total_N

        print(f"{cur_type:5s} loss: {loss:.6f}, {cur_type:5s} acc: {accuracy:.4f}")
        sys.stdout.flush()

        if cur_type == "valid" and epoch == valid_epoch_num - 1:
        
            # ============ 构造 teacher → student 的 2×2 转移表 ============
            full_targets = self.train_tar if cur_type=="train" else self.valid_tar

            old_right = (all_old_pred == full_targets)    # eu space
            now_right = (all_now_pred == full_targets)    # ct space

            A = (( now_right &  old_right)).sum().item()   # 原来对，现在对
            B = (( now_right & ~old_right)).sum().item()   # 原来错，现在对（好）
            C = ((~now_right &  old_right)).sum().item()   # 原来对，现在错（不好）
            D = ((~now_right & ~old_right)).sum().item()   # 原来错，现在错

            pctA = A / total_N * 100
            pctB = B / total_N * 100
            pctC = C / total_N * 100
            pctD = D / total_N * 100

            # ----------- 表格 1：数值 -----------
            print("\nTable 1: Prediction Transition (Count)")
            print("+----------+--------------+--------------+")
            print("|          | 原来对了     | 原来错了     |")
            print("+----------+--------------+--------------+")
            print(f"| 现在对了 | {A:10d}     | {B:10d}     |")
            print("+----------+--------------+--------------+")
            print(f"| 现在错了 | {C:10d}     | {D:10d}     |")
            print("+----------+--------------+--------------+\n")

            # ----------- 表格 2：百分比 -----------
            print("Table 2: Prediction Transition (Percentage)")
            print("+----------+------------------+------------------+")
            print("|          | 原来对了         | 原来错了         |")
            print("+----------+------------------+------------------+")
            print(f"| 现在对了 | {pctA:6.2f}%         | {pctB:6.2f}%         |")
            print("+----------+------------------+------------------+")
            print(f"| 现在错了 | {pctC:6.2f}%         | {pctD:6.2f}%         |")
            print("+----------+------------------+------------------+\n")

            sys.stdout.flush()

        return loss


    def visualize(self, epoch: int, cur_type: str):
        if cur_type == "train":
            train_eu_emb = self.train_emb[:256]                    # (256, dim)
            train_ct_emb = self.train_locations[:256]              # (256, tp)
            vocab_eu_emb = self.vocab_emb                          # (256, dim)
            vocab_ct_emb = self.vocab_locations                    # (256, tp)
            
            S_dd_eu      = normalized_matmul(train_eu_emb, train_eu_emb.t())[0].cpu().numpy()
            S_dd_ct      = self.cos_similarity(train_ct_emb[:, None, :], train_ct_emb[None, :, :]).mean(dim=-1).cpu().numpy()
            S_ss_eu      = normalized_matmul(vocab_eu_emb, vocab_eu_emb.t())[0].cpu().numpy()
            S_ss_ct      = self.cos_similarity(vocab_ct_emb[:, None, :], vocab_ct_emb[None, :, :]).mean(dim=-1).cpu().numpy()
            S_ds_eu      = normalized_matmul(train_eu_emb, vocab_eu_emb.t())[0].cpu().numpy()
            S_ds_ct      = self.cos_similarity(train_ct_emb[:, None, :], vocab_ct_emb[None, :, :]).mean(dim=-1).cpu().numpy()
            
            visualize_similarity   (S_dd_eu, S_dd_ct, meta_name="train_et_dd_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))
            visualize_similarity   (S_ss_eu, S_ss_ct, meta_name="train_et_ss_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))
            visualize_pair_bihclust(S_ds_eu, S_ds_ct, meta_name="train_et_ds_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))

            

        elif cur_type == "valid":
            train_eu_emb = self.train_emb[:256]                    # (256, dim)
            train_ct_emb = self.train_locations[:256]              # (256, tp)
            valid_eu_emb = self.valid_emb[:256]                    # (256, dim)
            valid_ct_emb = self.valid_locations[:256]              # (256, tp)
            vocab_eu_emb = self.vocab_emb                          # (256, dim)
            vocab_ct_emb = self.vocab_locations                    # (256, tp)

            S_dd_eu      = normalized_matmul(valid_eu_emb, train_eu_emb.t())[0].cpu().numpy()
            S_dd_ct      = self.cos_similarity(valid_ct_emb[:, None, :], train_ct_emb[None, :, :]).mean(dim=-1).cpu().numpy()
            S_ds_eu      = normalized_matmul(valid_eu_emb, vocab_eu_emb.t())[0].cpu().numpy()
            S_ds_ct      = self.cos_similarity(valid_ct_emb[:, None, :], vocab_ct_emb[None, :, :]).mean(dim=-1).cpu().numpy()
            
            visualize_pair_bihclust(S_dd_eu, S_dd_ct, meta_name="valid_et_dd_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))
            visualize_pair_bihclust(S_ds_eu, S_ds_ct, meta_name="valid_et_ds_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))
            
    
    def visualize_dyn_sta_mds_tsne(
        self,
        epoch: int,
        K_train: int = 1024,
        K_valid: int = 1024,
    ):
        """
        MDS + t-SNE visualization for train / valid / vocab
        Only called at valid stage.
        """

        # ------------------------------------------------
        # 1. 采样点
        # ------------------------------------------------
        train_idx = torch.randperm(N_train)[:K_train]
        valid_idx = torch.randperm(N_valid)[:K_valid]
        vocab_idx = torch.arange(N_vocab)

        # embeddings / locations
        train_emb = self.train_emb[train_idx]
        valid_emb = self.valid_emb[valid_idx]
        vocab_emb = self.vocab_emb

        train_loc = self.train_locations[train_idx]
        valid_loc = self.valid_locations[valid_idx]
        vocab_loc = self.vocab_locations

        # ------------------------------------------------
        # 2. 构造联合集合
        # ------------------------------------------------
        all_emb = torch.cat([train_emb, valid_emb, vocab_emb], dim=0)
        all_loc = torch.cat([train_loc, valid_loc, vocab_loc], dim=0)

        N_total = all_emb.shape[0]

        # ------------------------------------------------
        # 3. 计算 pairwise similarity / distance
        # ------------------------------------------------
        # similarity in [-1,1]
        sim = self.cos_similarity(
            all_loc[:, None, :],
            all_loc[None, :, :]
        ).mean(dim=-1)   # (N_total, N_total)

        dist = (1 - (sim - 1e-3)) / 2.

        dist = dist.cpu().numpy()

        # ------------------------------------------------
        # 4. 计算 valid 的 old / now right
        # ------------------------------------------------
        # 复用你已有的逻辑
        valid_targets = self.valid_tar[valid_idx]

        # teacher
        eu_logits = 20. * (valid_emb @ vocab_emb.t())
        pred_old = eu_logits.argmax(dim=-1)

        # student
        ct_val = self.cos_similarity(
            valid_loc[:, None, :],
            vocab_loc[None, :, :]
        ).mean(dim=-1)

        if use_eu_norm:
            eu_norm = (
                torch.norm(valid_emb, dim=-1, keepdim=True)
                @ torch.norm(vocab_emb, dim=-1, keepdim=True).t()
            )
        else:
            eu_norm = torch.ones_like(ct_val) * 20.

        ct_logits = ct_val * eu_norm
        pred_now = ct_logits.argmax(dim=-1)

        old_right = (pred_old == valid_targets)
        now_right = (pred_now == valid_targets)

        # ------------------------------------------------
        # 5. 计算 MDS / t-SNE
        # ------------------------------------------------
        from sklearn.manifold import MDS, TSNE

        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=0,
        )
        xy_mds = mds.fit_transform(dist)

        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            init="random",
            random_state=0,
            perplexity=30,
        )
        xy_tsne = tsne.fit_transform(dist)

        # ------------------------------------------------
        # 6. 绘图（示例：MDS）
        # ------------------------------------------------
        def plot_scatter(xy, name):
            plt.figure(figsize=(8, 8))

            # train
            plt.scatter(
                xy[:K_train, 0], xy[:K_train, 1],
                c="blue", s=5, marker="o", label="train"
            )

            # valid
            v_start = K_train
            for i in range(K_valid):
                x, y = xy[v_start + i]
                if old_right[i] and now_right[i]:
                    plt.scatter(x, y, c="red", marker="o", s=30)
                elif (not old_right[i]) and now_right[i]:
                    plt.scatter(x, y, c="red", marker="s", s=30)
                elif (not old_right[i]) and (not now_right[i]):
                    plt.scatter(x, y, edgecolors="red", facecolors="none", marker="o", s=30)
                else:  # old right, now wrong
                    plt.scatter(x, y, edgecolors="red", facecolors="none", marker="s", s=30)

            # vocab
            v_start = K_train + K_valid
            plt.scatter(
                xy[v_start:, 0], xy[v_start:, 1],
                c="green", s=30, marker="o", label="vocab"
            )

            plt.title(f"{name}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    vis_path,
                    f"scatter_{name}_epoch_{epoch:04d}.png"
                ),
                dpi=150
            )

        plot_scatter(xy_mds, "MDS")
        plot_scatter(xy_tsne, "t-SNE")

            
    def _synchronize_all_streams(self):
        for id in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{id}') 
            torch.cuda.synchronize(device)
            torch.cuda.default_stream(device).synchronize()
            
        for stream in read_streams + comp_streams + write_streams:
            stream.synchronize()

        
if __name__ == "__main__":
    pass

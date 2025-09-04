import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import PowerNorm
from scipy.cluster.hierarchy import (dendrogram, leaves_list, linkage,
                                     optimal_leaf_ordering)
from scipy.spatial.distance import pdist, squareform
from torch.nn import functional as F

from src.loom_kernel import triton_loom_wrapper
from src.loss import compute_loss
from src.para import batch_size, block_size, sample_k, vocab_size
from src.utils import *

main_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from typing import List, Tuple

# ------------

torch.manual_seed(1337)

class CritiGraph(torch.nn.Module):
    main_distance_lookup_table: torch.Tensor
    main_locations: torch.Tensor
    
    def __init__(self, h, tp, c, eps, epoch, batch_size, convergence, emb_size, blk_size, division_fact, loss_type):
        super().__init__() 
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
        self.k = int(c*h // division_fact)
        
        ### ---- 设备/进程信息 ---- ###
        self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.streams = [torch.cuda.Stream(device=dev) for dev in self.devices]
        
        # self.k = 1
        self.eps = eps
        self.epoch = epoch  
        self.batch_size = batch_size
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=main_device)).unsqueeze(0).unsqueeze(2) # (1, H, 1)
        self.flip_masks = [self.flip_masks.clone().to(dev) for dev in self.devices] 
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.distance_lookup_table = [self.distance_lookup_table.clone().to(dev) for dev in self.devices]
        self.convergence = convergence
        self.emb_size = emb_size
        self.blk_size = blk_size
        
        self.raw_locations = torch.randint(1 - self.n, self.n, (self.emb_size, self.tp), dtype=torch.int64, device='cpu')
        self.locations = [torch.empty_like(self.raw_locations, device=dev) for dev in self.devices]
        for i in range(0, len(self.locations)):
            self.locations[i].copy_(self.raw_locations, non_blocking=True)
        torch.cuda.synchronize()
        
        self.loss_type   = loss_type
        self.sta_TTT     = [None for i in range(len(self.devices))]
        self.sta_loc_TTT = [None for i in range(len(self.devices))]
        self.cnc_loc_TTT = [None for i in range(len(self.devices))]
        self.TTT_epoch   = -1
        self.sample_k    = sample_k
        self.TTT_T       = -1
        self.TTT_ci      = -1


        self.register_buffer('main_locations', self.locations[0].clone().cpu())
        self.main_distance_lookup_table = self.distance_lookup_table[0].clone().cpu()
        self.debug_epoch = False
        
        torch.cuda.empty_cache()
        
        
    def generate_distance_lookup_table(self):
        xor_results = torch.arange(self.n, dtype=torch.int64, device=main_device)
        return (torch.floor(torch.log2(xor_results.float() + 1)) + 1) / self.h
    
    # @torch.jit
    def distance(self, coord1: torch.Tensor, coord2: torch.Tensor, norm: torch.Tensor, dev_num=0):
        # sg = torch.sign(coord1) * torch.sign(coord2)
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        # sg = 1 - (((coord1 >= 0) ^ (coord2 >= 0)).to(torch.int16) << 1)
        xor_result = torch.abs(coord1) ^ torch.abs(coord2)
        # _, exp = torch.frexp((xor_result + 1).to(torch.float32))
        # s = exp.float() / self.h
        s = self.distance_lookup_table[dev_num][xor_result]
        # coord1_norm = self.distance_lookup_table[dev_num][torch.abs(coord1)]
        # coord2_norm = self.distance_lookup_table[dev_num][torch.abs(coord2)]
        # sg * (1 - s): shape = (B, T, T, D) 或者 (B, T, T, C, D)
        return sg * (1 - s) * norm
        # return sg * (1 - s) * coord1_norm * coord2_norm * 5.
    
    def main_distance(self, coord1: torch.Tensor, coord2: torch.Tensor, norm: torch.Tensor,):
        coord1, coord2, x_norm = coord1.detach().clone().cpu(), coord2.detach().clone().cpu(), norm.detach().clone().cpu()
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        xor_result = torch.bitwise_xor(torch.abs(coord1), torch.abs(coord2))
        s = self.main_distance_lookup_table[xor_result]
        return sg * (1 - s) * x_norm
    
    def generate_random_masks(self, sz, dev_num=0):
        upper_bounds = 2**torch.arange(self.h, dtype=torch.int64, device=self.devices[dev_num])
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k, self.tp), dtype=torch.int64, device=self.devices[dev_num]) # (H, B*T, K, D)
        masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3) # (B*T, H, K, D)
    def connection(self, ori_int, dev_num=0):
        flipped_ints = ori_int.unsqueeze(1) ^ self.flip_masks[dev_num] # (B*T1, H, D)
        random_masks = self.generate_random_masks(flipped_ints.size(0), dev_num=dev_num)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k, self.tp)
        # (B*T1, H, 1, D) ^ (B*T1, H, K, D) -> (B*T1, H*K, D)
        loc = torch.cat((result, ori_int.unsqueeze(1), -result), dim=1) # (B*T1, H*K + 1 + H*K, D)
        indices = torch.randperm(loc.size(1), device=self.devices[dev_num]) 
        loc = loc[:, indices, :]
        return loc
        
    def calc_loss(self, 
                  ct_val : torch.Tensor, eu_val : torch.Tensor, 
                  mask   : torch.Tensor, lth    : torch.Tensor,
                  epoch  : int = 0     , mode   : str = 'dyn' ,
                  sum_dim: int = 2
    ) -> torch.Tensor:
        ### ct_val: (B, T1, T2, C, D), eu_val: (B, T1, T2, C, D)
        ### mask  : (B, T1, T2, 1, 1), lth   : (B, T1, 1, 1)
        
        if mode == 'dyn' or mode == 'TTT':
            loss = compute_loss(self.loss_type['dyn_loss'],  ct_val, eu_val, lth, mask, sum_dim) # type: ignore
        elif mode == 'sta':
            loss = compute_loss(self.loss_type['sta_loss'],  ct_val, eu_val, lth, mask, sum_dim) # type: ignore  
        elif mode == 'prob':
            loss = compute_loss(self.loss_type['prob_loss'], ct_val, eu_val, lth, mask, sum_dim) # type: ignore
        elif mode == 'weighted':
            loss_dyn  = compute_loss(
                self.loss_type['dyn_loss'],          # type: ignore
                ct_val[:, :, :block_size, :, :], 
                eu_val[:, :, :block_size, :, :], 
                lth, 
                mask  [:, :, :block_size, :, :],
                sum_dim
            ) # (B, T1, C, tp)
            loss_prob = compute_loss(
                self.loss_type['prob_loss'],         # type: ignore
                ct_val[:, :, block_size:, :, :], 
                eu_val[:, :, block_size:, :, :], 
                lth, 
                mask  [:, :, block_size:, :, :],
                sum_dim
            ) # (B, T1, C, tp)
            loss = self.loss_strategy['ratio_dyn'] * loss_dyn + self.loss_strategy['ratio_prob'] * loss_prob # type: ignore

        return loss # (B, T1, C, tp)
        
        
    
    @torch.no_grad()
    def loom(self, 
             epoch     : int,           #                , 代表当前是第几个 epoch 
             b_sta     : torch.Tensor,  # (cB, T1)       , 代表 每个样本在 locations 中的 id        (v 代指 value)
             b_pos     : torch.Tensor,  # (cB, T2)       , 代表 每个 pos 样本在 locations 中的 id               
             b_val_v   : torch.Tensor,  # (cB, T1, T2)   , 代表 需要拟合的值
             # b_val_m   : torch.Tensor,  # (cB, T1, T2)   , 代表 对应位置的值是否有效
             b_val_n   : torch.Tensor, # (cB, T1, T2)   , 代表 欧式空间的范数乘积
             # cB 是 current batch size 的意思
             mode: str,
    ):
        train_mode = self.loss_strategy['target'].startswith(mode)
        
        cB, T1, T2, D = b_val_v.shape[0], b_val_v.shape[1], b_val_v.shape[2], self.tp
        splits = torch.linspace(0, cB, len(self.devices) + 1, dtype=torch.int64).tolist()
        splits = list(map(int, splits))
        all_selected_locs = torch.zeros((cB, T1, self.tp)    , dtype=torch.int64  , pin_memory=True)
        all_avg_loss      = torch.zeros((len(self.devices), ), dtype=torch.float32, pin_memory=True)
        
        for i, (dev, stream, (s, e)) in enumerate(zip(self.devices, self.streams, zip(splits[:-1], splits[1:]))):
            if s == e: continue
            B = e - s
            dev_num = i
            
            with torch.cuda.device(dev), torch.cuda.stream(stream): # type: ignore
                ### 通信1：数据传输开始 ###        
                sta, pos, val_v, val_n = to_dev(
                    b_sta, b_pos, b_val_v, b_val_n, 
                    device=dev, s=s, e=e
                )
                ### 通信1：数据传输结束 ###
                
                
                ### 计算：计算开始 ###
                #### step 1: 获取基本信息
                sta_loc = self.locations[i][sta] # (B, T1, tp)
                pos_loc = self.locations[i][pos] # (B, T2, tp)
                dis_sta_pos     = self.distance(
                    sta_loc[:, :, None, :]   , pos_loc[:, None, :, :]     , val_n[..., None]      , dev_num=dev_num
                )               # (B, T1, T2, tp)
                '''
                # 用 Triton 算子替换掉之前的所有计算, 传递所有需要的输入张量
                # selected_locs, min_loss = triton_loom_wrapper(
                #     sta_loc=sta_loc,
                #     cnc_loc=cnc_loc,
                #     logits=logits,
                #     x_norm=x_norm,
                #     lg=lg,
                #     mask=mask,
                # )
                # tl = min_loss.mean() * B
                '''
                if train_mode:
                    #### step 2: 获取候选位置（的值）
                    cnc_loc = self.connection(
                        torch.abs(sta_loc.view(-1, self.tp)), 
                        dev_num=dev_num
                    ).view(B, T1, -1, self.tp)       # (B, T1, C, tp)
                    dis_sta_pos_sum = dis_sta_pos.sum(dim=-1) 
                                # (B, T1, T2)
                    dis_cnc_pos     = self.distance(
                        cnc_loc[:, :, None, :, :], pos_loc[:, None, :, None,:], val_n[..., None, None], dev_num=dev_num
                    )            # (B, T1, T2, C, tp)
                    ct_val          = (
                        dis_sta_pos_sum[:, :, :, None, None] - dis_sta_pos[:, :, :, None, :] + dis_cnc_pos
                    ) / self.tp  #   (B, T1, T2, C, tp)
                                #    对于 B 个 batch，T1 个 starting point，向 T2 个 positive sample 连边。此时，我们把其中某个 positive sample 替换为 connected sample，共有 C 个；此时，D 个维度上的的距离是多少？
                    
                    #### step 3: 计算 loss
                    ct_val    = ct_val                                                                # (B, T1, T2, C, tp)
                    eu_val    = val_v[..., None, None].expand(ct_val.shape)                           # (B, T1, T2, C, tp)
                    mask, lth = self.neighbor_batch(B, T1, T2, epoch, dev_num=dev_num,)               # (B, T1, T2)
                    loss      = self.calc_loss(ct_val, eu_val, mask[..., None, None], lth[..., None, None], 
                                               epoch=epoch, mode=mode)                                # (B, T1, C, tp)
                    
                    
                    #### step 4: 挑选 loss 最小的位置
                    indices       = torch.argmin(loss, dim=2)                                         # (B, T1, tp)
                    batch_indices = torch.arange(B,       device=dev)[:, None, None]
                    time_indices  = torch.arange(T1,      device=dev)[None, :, None]
                    dim_indices   = torch.arange(self.tp, device=dev)[None, None, :]
                    selected_locs = cnc_loc[batch_indices, time_indices, indices, dim_indices]        # (B, T1, tp)
                    avg_loss      = loss   [batch_indices, time_indices, indices, dim_indices].mean() * B
                else:
                    ct_val        = dis_sta_pos.mean(dim=-1)                                          # (B, T1, T2)
                    eu_val        = val_v                                                             # (B, T1, T2)
                    mask, lth     = self.neighbor_batch(B, T1, T2, -1, dev_num=dev_num,)        # (B, T1, T2)
                    loss          = self.calc_loss(ct_val, eu_val, mask, lth, 
                                                   epoch=epoch, mode=mode)                            # (B, T1)
                    avg_loss      = loss.mean() * B
                    selected_locs = torch.empty((B, T1, self.tp), dtype=torch.int64, device=dev)      # (B, T1, tp)
                                                            
                
                ### 计算：计算结束 ###
                
                ### 通信2：数据传输开始 ###
                all_selected_locs[s:e].copy_(selected_locs.to(dtype=torch.int64), non_blocking=True)
                all_avg_loss[i].copy_(avg_loss, non_blocking=True)
                ### 通信2：数据传输结束 ###
        
        for i, stream in enumerate(self.streams):
            stream.synchronize()
        
        if train_mode:
            ### 更新与计算 ###
            for i, dev in enumerate(self.devices):
                self.locations[i].index_copy_(
                    0,                                             # dim=0, 沿行更新
                    b_sta.to(dev, non_blocking=True).view(-1),     # 哪些行
                    all_selected_locs.to(dev, non_blocking=True).view(-1, self.tp)   # 更新的数据
                )
        
        return all_avg_loss.sum().item() / cB


    # @timeit(name=f'neighbor_batch 函数主体')
    def neighbor_batch(self, B: int, T1: int, T2: int, epoch: int = 0, dev_num: int = 0, mark: Optional[Mark] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.devices[dev_num]

        # 收敛期：建议缓存这两个返回，避免每次重分配
        if (self.loss_strategy['converge'] is not None and epoch > self.loss_strategy['converge']) or epoch == -1:
            valid_mask = torch.ones((B, T1, T2), dtype=torch.bool, device=device)
            counts = torch.full((B, T1), T2, dtype=torch.float32, device=device)
            return valid_mask, counts

        # 80% 行全选；20% 行仅选 1 个
        choosing_mask = (torch.rand((B, T1), device=device) > 0.2)                 # (B,T1)

        # 为“仅选1个”的行生成 one-hot，不做任何散乱写
        t2_idx   = torch.randint(T2, (B, T1), device=device)                        # (B,T1)
        one_hot  = torch.nn.functional.one_hot(t2_idx, num_classes=T2).to(torch.bool)  # (B,T1,T2)

        # 扩展 choosing_mask 并合成最终 mask：选中行→全 True；未选中行→对应 one-hot
        choosing_mask_3d = choosing_mask.unsqueeze(-1)                              # (B,T1,1)
        valid_mask = torch.where(choosing_mask_3d, 
                                torch.ones((B, T1, T2), dtype=torch.bool, device=device),
                                one_hot)

        # 计数同样用分支消除
        counts = torch.where(choosing_mask, 
                            torch.full((B, T1), T2, dtype=torch.float32, device=device),
                            torch.ones((B, T1), dtype=torch.float32, device=device))
        return valid_mask, counts


    # @timeit(name=f'cte 函数主体')
    def forward(self,        
                sta     : torch.Tensor, # (B, T+V)     , 代表 每个 sta 样本在 locations 中的 id 
                pos     : torch.Tensor, # (B, T+V)     , 代表 每个 pos 样本在 locations 中的 id 
                val_v   : torch.Tensor, # (B, T+V, T+V), 代表 欧式空间待拟合的值              
                val_m   : torch.Tensor, # (B, T+V, T+V), 代表 余弦相似度是否有效
                val_n   : torch.Tensor, # (B, T+V, T+V), 代表 欧式空间的范数乘积，除了 prob 对应的位置之外用 1.0 填充
                targets : Optional[torch.Tensor] = None, 
                mark    : Optional[Mark] = None
                # (0:B, 0:T  , 0:T)   代表着 dynamic embeddings 的内积
                # (0:B, T:T+V, T:T+V) 代表着 static  embeddings 的内积
                # (0:B, 0:T  , T:T+V) 代表着 logits, 准备用于概率分布计算
        ): 
        # assert mark is not None # 保证计时器存在
        T, V = block_size, vocab_size
        ### 1. 生成用于传输数据的张量
        # mark("pinned 张量生成")
        pinned = pinned_copy_by_name(
            named(sta=sta, pos=pos, 
                  val_v=val_v, val_m=val_m, val_n=val_n)
        )
        (sta, pos, val_v, val_m, val_n) = (
            pinned['sta'], pinned['pos'], 
            pinned['val_v'], pinned['val_m'], pinned['val_n']
        ) 

        dyn_loss, sta_loss, prob_loss = 0.0, 0.0, 0.0
        ### 2. 遍历每个 epoch
        for epoch in range(self.epoch): 
            # if epoch < self.epoch * 0.85:  
            
            self.loss_strategy: PhaseConfig = get_strategy(self.loss_type, epoch)
            if self.loss_strategy['target'] == 'TTT_only':
                TTT_loss = self.loom(        epoch, sta            , pos            , val_v                   , val_n                   ,
                mode='TTT')
                if epoch == self.epoch - 1:
                    print(f"current epoch: {epoch}, TTT_loss: {fmt6w(TTT_loss)}")
                continue
            
            if self.loss_strategy['target'] == 'weighted_dyn_prob':
                self.loom(        epoch, sta[:,  :T]    , pos[:,   :]    , val_v[:,  :T,    : ]    , val_n[:,  :T,    : ]    ,
                mode='weighted')
            
            dyn_loss  = self.loom(epoch, sta[:,  :T]    , pos[:,   :T]   , val_v[:,  :T,    :T]    , val_n[:,  :T,    :T]    ,
                mode='dyn')
            sta_loss  = self.loom(epoch, sta[0:1, T:T+V], pos[0:1, T:T+V], val_v[0:1, T:T+V, T:T+V], val_n[0:1, T:T+V, T:T+V],
                mode='sta')
            prob_loss = self.loom(epoch, sta[:,  :T]    , pos[:, T:T+V]  , val_v[:,  :T,   T:T+V]  , val_n[:,  :T,   T:T+V]  ,
                mode='prob')
            print(f"current epoch: {epoch}, dyn_loss: {fmt6w(dyn_loss)}, sta_loss: {fmt6w(sta_loss)}, prob_loss: {fmt6w(prob_loss)}")
        
        
        

        ### 3. 保存并返回 (B, T, V) 的 logits
        self.main_locations = self.locations[0].clone().cpu()
        
        if self.loss_strategy['target'] != 'TTT_only':
            return self.distance(self.locations[0][sta[:, :T].to(self.devices[0])].unsqueeze(2),    # (B, T, 1, tp)
                                self.locations[0][pos[:, T:T+V].to(self.devices[0])].unsqueeze(1), # (B, 1, V, tp)
                                val_n[:, :T, T:T+V, None].to(self.devices[0]),                     # (B, T, V, 1)
                                dev_num=0).mean(dim=-1)                        # (B, T, V)

    # @timeit(name=f'cte 函数主体')
    def forward_TTT(
        self,        
        sta     : torch.Tensor, # (B)     , 代表 每个 sta 样本在 locations 中的 id 
        pos     : torch.Tensor, # (T)     , 代表 每个 pos 样本在 locations 中的 id 
        val_v   : torch.Tensor, # (B, T)  , 代表 欧式空间待拟合的值              
        val_n   : torch.Tensor, # (B, T)  , 代表 欧式空间的范数乘积，除了 prob 对应的位置之外用 1.0 填充,
    ):        
        # assert mark is not None # 保证计时器存在
        B, T, V = sta.size(0), block_size, vocab_size
        ### 1. 生成用于传输数据的张量
        # mark("pinned 张量生成")
        pinned = pinned_copy_by_name(
            named(sta=sta, pos=pos, 
                  val_v=val_v, val_n=val_n)
        )
        (sta, pos, val_v, val_n) = (
            pinned['sta'], pinned['pos'], 
            pinned['val_v'], pinned['val_n']
        ) 


        ### 2. 
        self.loss_strategy: PhaseConfig = get_strategy(self.loss_type, -1)
        if self.loss_strategy['target'] == 'TTT_only':
            TTT_loss = self.loom_TTT(sta, pos, val_v, val_n, mode='TTT') # (B, C, tp)
            return TTT_loss    
        
    def update_TTT(
        self, 
        TTT_loss: torch.Tensor # (B, C, tp)
    ):
        assert self.sta_TTT[0] is not None, "请先运行 forward_TTT 方法以初始化 TTT 状态"
        assert self.cnc_loc_TTT[0] is not None, "请先运行 forward_TTT 方法以初始化 TTT 状态"



        TTT_loss      = TTT_loss.to(self.devices[0])
        indices       = torch.argmin(TTT_loss, dim=1)                                                # (B, tp)
        batch_indices = torch.arange(self.sta_TTT[0].size(0), device=self.devices[0])[:, None]       # (B, 1)
        dim_indices   = torch.arange(self.tp    ,             device=self.devices[0])[None :]        # (1, tp)
        selected_locs = self.cnc_loc_TTT[0][batch_indices, indices, dim_indices]        # (B, tp)

        for i, dev in enumerate(self.devices):
            self.locations[i].index_copy_(
                0,                                             # dim=0, 沿行更新
                self.sta_TTT[i].to(dev, non_blocking=True).view(-1),     # 哪些行
                selected_locs.to(dev, non_blocking=True).view(-1, self.tp)   # 更新的数据
            )
        self.main_locations = self.locations[0].clone().cpu()
        
        # 全局重置
        self.sta_TTT      = [None for _ in range(len(self.devices))]
        self.sta_loc_TTT  = [None for _ in range(len(self.devices))]
        self.cnc_loc_TTT  = [None for _ in range(len(self.devices))]
        self.valid_samples = None

if __name__ == "__main__":
    pass

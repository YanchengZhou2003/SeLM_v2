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
from src.para import T1_block_size, T2_block_size
from src.utils import *

main_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from typing import List, Tuple

# ------------

torch.manual_seed(1337)

class CritiGraph(torch.nn.Module):
    main_distance_lookup_table: torch.Tensor
    main_locations: torch.Tensor
    
    def __init__(self, h, tp, c, emb_size, division_fact, 
                 loss_strategy, sample_k, epoch_num):
        super().__init__() 
        ### 1：设备信息
        self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.streams = [torch.cuda.Stream(device=dev) for dev in self.devices]        
    
        ### 2. 基本参数
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
        self.k = int(c*h // division_fact)
        self.loss_strategy = loss_strategy
        self.emb_size = emb_size
        self.epoch_num = epoch_num
        
        ### 3. CT Space Embeddings 初始化
        self.main_locations = torch.randint(1 - self.n, self.n, (self.emb_size, self.tp), dtype=torch.int64, device='cpu', pin_memory=True)
        self.locations = [self.main_locations.clone().to(dev, non_blocking=True) for dev in self.devices]
        torch.cuda.synchronize()

        ### 4. 训练时参数
        self.epoch = -1
        self.sample_k = sample_k

    
    @torch.compile()
    def distance(self, coord1: torch.Tensor, coord2: torch.Tensor, norm: torch.Tensor):
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        xor_result = torch.abs(coord1) ^ torch.abs(coord2)
        _, exp = torch.frexp((xor_result + 1).to(torch.float32))
        s = exp.float() / self.h
        return sg * (1 - s) * norm
    
    
    def generate_random_masks(self, sz, dev_num=0):
        device = self.devices[dev_num] if dev_num >= 0 else 'cpu'

        upper_bounds   = 2 ** torch.arange(self.h, dtype=torch.int64, device=device)
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k, self.tp), dtype=torch.int64, device=device) # (H, B*T, K, D)
        masks = random_numbers & (upper_bounds.view(-1, 1, 1, 1) - 1)
        # masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3) # (B*T, H, K, D)
    
    def connection(self, ori_int, dev_num=0):
        device = self.devices[dev_num] if dev_num >= 0 else 'cpu'
        
        flip_masks = (1 << torch.arange(self.h, device=device, dtype=ori_int.dtype)).unsqueeze(0).unsqueeze(2)
        flipped_ints = ori_int.unsqueeze(1) ^ flip_masks # (B*T1, H, D)
        random_masks = self.generate_random_masks(flipped_ints.size(0), dev_num=dev_num)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k, self.tp)
        # (B*T1, H, 1, D) ^ (B*T1, H, K, D) -> (B*T1, H*K, D)
        loc = torch.cat((result, ori_int.unsqueeze(1), -result), dim=1) # (B*T1, H*K + 1 + H*K, D)
        indices = torch.randperm(loc.size(1), device=device) 
        loc = loc[:, indices, :]
        return loc
        
    def calc_loss(self, 
                  ct_val : torch.Tensor, eu_val : torch.Tensor, 
                  mask   : torch.Tensor, lth    : torch.Tensor,
                  loss_type: str,
                  sum_dim  : int = 1
    ) -> torch.Tensor:

        loss = compute_loss(loss_type,  ct_val, eu_val, lth, mask, sum_dim) # type: ignore
        return loss # (B, T1, C, tp)
        
        
    
    @torch.no_grad()
    def loom(self, 
             T1_block: Tuple[int, int], T2_block: Tuple[int, int],
             pos: torch.Tensor, emb: torch.Tensor,
             loss_type: str,
             _all_loss: torch.Tensor
    ):
        ### step 1: 获取基本信息
        T1 = T1_block[1] - T1_block[0]
        cT2 = T2_block[1] - T2_block[0]
        C, D   = 2 * self.k * self.h + 1, self.tp
        
        ### step 2: 获取切片
        splits = torch.linspace(0, cT2, len(self.devices) + 1, dtype=torch.int64)
        splits = list(map(int, splits.tolist()))
        
        ### step 3: 开始计算
        for i, (dev, stream, (s, e)) in enumerate(zip(self.devices, self.streams, zip(splits[:-1], splits[1:]))):
            if s == e: 
                _all_loss[i].copy_(torch.zeros_like(_all_loss[i], device=_all_loss[i].device), non_blocking=True)
                continue
            T2 = e - s
            r_s, r_e = s + T2_block[0], e + T2_block[0]
            dev_num = i
            
            with torch.cuda.device(dev), torch.cuda.stream(stream): # type: ignore
                ### 通信1：数据传输开始 ###        
                pos_T2, emb_T2 = to_dev(
                    pos, emb, 
                    device=dev, s=s, e=e
                )
                ### 通信1：数据传输结束 ###
                 
                
                ### 计算：计算开始 ###
                #### step 3.0: 计算欧式空间的值
                emb_T1 = self.emb_T1[i] # (T1, dim)                
                val_v, val_n = normalized_matmul(emb_T1, emb_T2.t(), ori_prod=(loss_type == 'kl' or loss_type == 'js')) # (T1, T2) 
                if loss_type == 'kl' or loss_type == 'js':
                    val_v = self.targets_T1[i][T1_block[0]:T1_block[1], s:e].to(torch.float32) # (T1, T2)
                
                #### step 3.1: 获取基本信息
                sta_loc = self.sta_loc_T1[i] # (T1, tp)
                pos_loc = self.locations[i][pos_T2] # (T2, tp)
                dis_sta_pos     = self.distance(
                    sta_loc[:, None, :]   , pos_loc[None, :, :]     , val_n[..., None]      
                )               # (T1, T2, tp)
                
                #### step 2: 获取候选位置（的值）
                cnc_loc = self.cnc_loc_T1[i] # (T1, C, tp)
                dis_sta_pos_sum = dis_sta_pos.sum(dim=-1) 
                             # (T1, T2)
                dis_cnc_pos     = self.distance(
                    cnc_loc[:, None, :, :], pos_loc[None, :, None,:], val_n[..., None, None]
                )            # (T1, T2, C, tp)
                ct_val          = (
                    dis_sta_pos_sum[:, :, None, None] - dis_sta_pos[:, :, None, :] + dis_cnc_pos
                ) / self.tp  # (T1, T2, C, tp)
                             # 对于 T1 个 starting point，向 T2 个 positive sample 连边。此时，我们把其中某个 positive sample 替换为 connected sample，共有 C 个；此时，D 个维度上的的距离是多少？
                
                #### step 3: 计算 loss
                ct_val    = ct_val                                                                # (T1, T2, C, tp)
                eu_val    = val_v[..., None, None].expand(ct_val.shape)                           # (T1, T2, C, tp)
                mask, lth = self.get_neighbor(T1, r_s, r_e, -1, -1, dev_num=dev_num,)             # (T1, T2)
                loss      = self.calc_loss(ct_val, eu_val, mask[..., None, None], lth[..., None, None], 
                                           loss_type=loss_type)                                   # (T1, C, tp)           
                ### 计算：计算结束 ###
                
                ### 通信2：数据传输开始 ###
                _all_loss[i].copy_(loss, non_blocking=True)
                ### 通信2：数据传输结束 ###
        
        for i, stream in enumerate(self.streams):
            stream.synchronize()
        
        return _all_loss.sum(dim=0) # (T1, C, tp)


    def get_neighbor(
        self,
        T1: int, T2_l: int, T2_r: int,
        idx_s: int, idx_e: int,
        dev_num: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            valid_mask: (B, cT)  布尔张量
            counts    : (B,)     float32
        说明:
            - 内部持久化:
                self.choosing_mask: (B,)  bool
                self.valid_samples: (B, k) int64，"全选"行填 -1
                self.all_idx: (n_all,) long，全选批次索引
                self.sel_idx: (n_sel,) long，非全选批次索引
                self.samples: (n_sel, k) int64，非全选批次的全局采样
        """
        device = self.devices[dev_num] if dev_num >= 0 else 'cpu'
        cT2 = int(T2_r - T2_l)

        # 收敛期：与旧逻辑一致
        if (self.loss_strategy['converge'] is not None and self.epoch > self.loss_strategy['converge']) or self.epoch == -1:
            valid_mask = torch.ones((T1, cT2), dtype=torch.bool, device=device)
            counts = torch.full((T1,), float(cT2), dtype=torch.float32, device=device)
            return valid_mask, counts

        # ---------- 初始化 choosing_mask 与 valid_samples ----------
        if getattr(self, "valid_samples", None) is None:
            # 80% 全选；20% 只用 k 个全局位置
            self.choosing_mask = (torch.rand((T1,), device=device) > 0.2)  # True=全选
            k_eff = self.sample_k

            # 占位: -1
            self.valid_samples = torch.full(
                (T1, k_eff),
                -1,
                dtype=torch.int64,
                device=device,
            )

            # 分批次索引持久化
            self.all_idx = self.choosing_mask.nonzero(as_tuple=False).squeeze(1)   # 全选批次
            self.sel_idx = (~self.choosing_mask).nonzero(as_tuple=False).squeeze(1)  # 非全选批次

            # 对 20% 的“非全选”批次抽样
            picked = torch.randint(
                low=idx_s,
                high=idx_e,
                size=(self.sel_idx.numel(), k_eff),
                device=device,
                dtype=torch.int64
            )  # (num_sel, k_eff)
            self.valid_samples[self.sel_idx] = picked

            # 预存非全选的采样
            self.samples = self.valid_samples.index_select(0, self.sel_idx)  # (m, k_eff)
            
            # 将所有生成的 self.xxx 变量迁移到各个设备
            self.choosing_mask = [self.choosing_mask.to(device) for device in self.devices]
            self.valid_samples = [self.valid_samples.to(device) for device in self.devices]
            self.all_idx = [self.all_idx.to(device) for device in self.devices]
            self.sel_idx = [self.sel_idx.to(device) for device in self.devices]
            self.samples = [self.samples.to(device) for device in self.devices]
            
            return None, None

        # ---------- 基于当前切片 [cT_l, cT_r) 生成局部 mask 与 counts ----------
        
        pos = self.samples[dev_num] - int(T2_l)                   # (m, k)
        in_slice = (pos >= 0) & (pos < cT2)               # (m, k)
        if not in_slice.any():
            valid_mask = torch.ones((T1, cT2), dtype=torch.bool, device=device)
            counts = torch.full((T1,), float(cT2), dtype=torch.float32, device=device)
            return valid_mask, counts
        
        # 存在局部采样
        valid_mask = torch.zeros((T1, cT2), dtype=torch.bool, device=device)
        counts = torch.zeros((T1,), dtype=torch.float32, device=device)
        
        valid_mask[self.all_idx[dev_num]] = True
        counts[self.all_idx[dev_num]] = float(cT2)
        counts[self.sel_idx[dev_num]] = in_slice.sum(dim=1).to(torch.float32) + 1e-12
        
        rc = torch.nonzero(in_slice, as_tuple=False) # (p, 2)
        b_idx = self.sel_idx[dev_num][rc[:, 0]]               # (p,)
        cols = pos[rc[:, 0], rc[:, 1]].to(torch.int64)  # (p,)
        valid_mask[b_idx, cols] = True

        return valid_mask, counts

    @timeit(name=f'cte 函数主体')
    def forward(self,        
                t_eu_emb  : torch.Tensor, # (N_train, dim), pinned memory
                t_targets : torch.Tensor, # (N_train, )   , pinned memory
                v_eu_emb  : torch.Tensor, # (N_valid, dim), pinned memory
                v_targets : torch.Tensor, # (N_valid, )   , pinned memory
                vocab_emb : torch.Tensor, # (vocab_size, dim), pinned memory
                t_idx     : torch.Tensor, # (N_train, )   , pinned memory
                v_idx     : torch.Tensor, # (N_valid, )   , pinned memory
                vocab_idx : torch.Tensor, # (vocab_size, ), pinned memory
                mark      : Optional[Mark] = None,
                ### 以上张量全部放在 cpu，避免占用 gpu 内存
        ): 
        ### step 1: 获取基本信息
        N_train, N_valid, vocab_size = t_eu_emb.size(0), v_eu_emb.size(0), vocab_idx.size(0)
        # 在这里，我们需要拟合的是一个 (N_train + N_valid, N_train + vocab_size) 的矩阵
        # 所以，外层循环遍历第一维，根据 T1_block_size 来划分
        # 内层循环遍历第二维，根据 T2_block_size 来划分
        t_T1_splits     = make_splits(0, N_train, T1_block_size)
        v_T1_splits     = make_splits(N_train, N_train + N_valid, T1_block_size)
        vocab_T1_splits = [(N_train + N_valid, N_train + N_valid + vocab_size)] 
        T1_splits       = t_T1_splits + v_T1_splits + vocab_T1_splits
        
        t_T2_splits     = make_splits(0, N_train, T2_block_size)
        vocab_T2_splits = [(N_train, N_train + vocab_size)]
        T2_splits       = t_T2_splits + vocab_T2_splits
        
        ### step 2: 准备训练时变量
        cur_T1_type, cur_T2_type, cur_loss_type = "train", "train", "dyn" 
        neighbor_idx = {"train": (0, N_train + vocab_size), "valid": (0, N_train), "vocab": (N_train, N_train + vocab_size)}
        _all_loss = torch.empty((len(self.devices), T1_block_size, 2 * self.k * self.h + 1, self.tp), dtype=torch.float32, pin_memory=True)
        # (num_dev, T1_block_size, C, tp)
        
        ### step 3: 遍历所有 epoch
        for epoch in range(self.epoch_num):
            self.epoch = epoch

            ### step 3.1: 训练
            for T1_block in T1_splits:
                all_loss = torch.zeros((T1_block_size, 2 * self.k * self.h + 1, self.tp), dtype=torch.float32, device=self.devices[0])
                
                cur_T1_type   = get_T1_type(T1_block, N_train, N_valid, vocab_size)
                emb_T1        = get_T1_emb (T1_block, t_eu_emb, v_eu_emb, vocab_emb, N_train, N_valid, vocab_size) # (T1_block_size, dim)
                sta_T1        = get_T1_idx (T1_block, t_idx, v_idx, vocab_idx, N_train, N_valid, vocab_size)       # (T1_block_size, )
                targets_T1    = get_T1_targets (T1_block, t_targets, N_train, N_valid, vocab_size)                 # (T1_block_size, )  
                targets_T1    = F.one_hot(targets_T1, num_classes=vocab_size) * 1e2                                # (T1_block_size, vocab_size)   
                self.get_neighbor(T1_block[1] - T1_block[0], -1, -1, 
                                *neighbor_idx[cur_T1_type],
                                dev_num=-1) 
                
                sta_loc_T1    = self.main_locations[sta_T1]                          # (T1_block_size, tp)
                cnc_loc_T1    = self.connection(sta_loc_T1, dev_num=-1)              # (T1_block_size, C, tp)
                
                
                self.emb_T1     = [emb_T1.to(dev, non_blocking=True) for dev in self.devices]
                self.targets_T1 = [targets_T1.to(dev, non_blocking=True) for dev in self.devices]
                self.sta_T1     = [sta_T1.to(dev, non_blocking=True) for dev in self.devices]
                self.sta_loc_T1 = [sta_loc_T1.to(dev, non_blocking=True) for dev in self.devices]
                self.cnc_loc_T1 = [cnc_loc_T1.to(dev, non_blocking=True) for dev in self.devices]
                
                torch.cuda.synchronize()
                
                for T2_block in T2_splits:
                    cur_T2_type   = get_T2_type(T2_block, N_train, vocab_size)
                    
                    if (cur_T1_type == "valid" and cur_T2_type == "vocab") or \
                    (cur_T1_type == "vocab" and cur_T2_type == "train"):
                        continue
                    
                    cur_loss_type = get_loss_type(cur_T1_type, cur_T2_type, self.loss_strategy)
                    emb_T2      = get_T2_emb (T2_block, t_eu_emb, vocab_emb, N_train, vocab_size) # (T2_block_size, dim)
                    pos_T2      = get_T2_idx (T2_block, t_idx, vocab_idx, N_train, vocab_size)    # (T2_block_size, )
                    cur_loss    = self.loom(T1_block, T2_block, pos_T2, emb_T2, cur_loss_type, _all_loss) # (T1_block_size, C, tp)

                    if cur_T1_type == "train" and T2_block == vocab_T2_splits[-1]:
                        all_loss    = self.loss_strategy['ratio_dyn']  * all_loss / N_train + \
                                      self.loss_strategy['ratio_prob'] * cur_loss.to(all_loss.device, non_blocking=True) / vocab_size                    
                    else:
                        all_loss   += cur_loss.to(all_loss.device, non_blocking=True)
                        if cur_T1_type == "valid" and T2_block == t_T2_splits[-1]:
                            all_loss = all_loss / N_valid 
                        elif cur_T1_type == "vocab" and T2_block == vocab_T2_splits[-1]:
                            all_loss = all_loss / N_valid if cur_T1_type == "valid" else all_loss / vocab_size
                        
                self.update(all_loss)
            
            ### step 3.2: 验证
            for split in ['train', 'valid']:
                splits  = t_T1_splits if split == 'train' else v_T1_splits
                dyn_idx = t_idx       if split == 'train' else v_idx
                dyn_emb = t_eu_emb    if split == 'train' else v_eu_emb
                targets = t_targets   if split == 'train' else v_targets
                N       = N_train     if split == 'train' else N_valid
                loss    = 0
                
                for block in splits:
                    Tl, Tr  = block
                    dyn_loc     = self.locations[0][dyn_idx[Tl:Tr].to(self.devices[0], non_blocking=True)] # (T1, tp)
                    vocab_loc   = self.locations[0][vocab_idx.to(self.devices[0], non_blocking=True)] # (N_vocab, tp)
                    _, norm = normalized_matmul(
                        dyn_emb. to(self.devices[0], non_blocking=True), 
                        vocab_emb.to(self.devices[0], non_blocking=True).t()
                    ) # (T1, N_vocab)
                    ct_val  = self.distance(dyn_loc[:, None, :], vocab_loc[None, :, :], norm[:, None])
                    loss  = F.cross_entropy(ct_val, targets[Tl:Tr].to(self.devices[0], non_blocking=True), reduction='mean').item() * (Tr - Tl)
                
                loss = loss / N
                print(f"epoch {epoch:3d}, {split:5s} loss: {loss:.4f}")
        

    def update(
        self, 
        loss: torch.Tensor # (T1, C, tp)
    ):

        indices       = torch.argmin(loss, dim=1)                                                # (T1, tp)
        batch_indices = torch.arange(self.sta_loc_T1[0].size(0), device=self.devices[0])[:, None]   # (T1, 1)
        dim_indices   = torch.arange(self.tp    ,             device=self.devices[0])[None :]    # (1, tp)
        selected_locs = self.cnc_loc_T1[0][batch_indices, indices, dim_indices]                  # (T1, tp)

        for i, dev in enumerate(self.devices):
            self.locations[i].index_copy_(
                0,                                             # dim=0, 沿行更新
                self.sta_T1[i].view(-1),     # 哪些行
                selected_locs.to(dev, non_blocking=True).view(-1, self.tp)   # 更新的数据
            )
        self.main_locations.index_copy_(
            0,
            self.sta_T1[0].cpu().view(-1),
            selected_locs.cpu().view(-1, self.tp)
        )
        
        # 全局重置
        self.sta_T1        = [None for _ in range(len(self.devices))]
        self.sta_loc_T1    = [None for _ in range(len(self.devices))]
        self.cnc_loc_T1    = [None for _ in range(len(self.devices))]
        self.valid_samples = None

if __name__ == "__main__":
    pass

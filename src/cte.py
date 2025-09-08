import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import PowerNorm
from torch.nn import functional as F
from tqdm import tqdm

from src.gettime import gettime, mark
from src.loom_kernel import triton_loom_wrapper
from src.loss import compute_loss
from src.para import ED, ST, T1_block_size, T2_block_size
from src.sample import BaseSample, CTE_Sample, CTE_Sort_Sample, Expander_Sample
from src.utils import *
from src.vis import visualize_pair_bihclust, visualize_similarity

main_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from typing import List, Tuple

# ------------

torch.manual_seed(1337)

class CritiGraph(torch.nn.Module):
    main_distance_lookup_table: torch.Tensor
    main_locations: torch.Tensor
    
    def __init__(self, h, tp, f, emb_size, division_fact, 
                 loss_strategy, sample_k, epoch_num):
        super().__init__() 
        ### 1：设备信息
        self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.streams = [torch.cuda.Stream(device=dev) for dev in self.devices]        
    
        ### 2. 基本参数
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = f
        self.k = int(f*h // division_fact)
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
        
        self._valid_mask_cache = {}

    
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
                  mask   : torch.Tensor,
                  loss_type: str,
                  sum_dim  : int = 1
    ) -> torch.Tensor:

        loss = compute_loss(loss_type,  ct_val, eu_val, mask, sum_dim) # type: ignore
        return loss # (B, T1, C, tp)
        

    def get_neighbor(
        self,
        T1: int, T2_l: int, T2_r: int,
        idx_s: int, idx_e: int,
        dev_num: int = 0
    ) -> torch.Tensor:
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
            
            return None

        
        # 收敛期：与旧逻辑一致
        if self.converge:
            return torch.ones((T1, cT2), dtype=torch.bool, device=device)


        # ---------- 基于当前切片 [cT_l, cT_r) 生成局部 mask 与 counts ----------
        pos = self.samples[dev_num] - int(T2_l)                   # (m, k)
        in_slice = (pos >= 0) & (pos < cT2)               # (m, k)
        if not in_slice.any():
            if (T1, cT2, dev_num) not in self._valid_mask_cache:
                valid_mask = torch.where(self.choosing_mask[dev_num].unsqueeze(1).expand(-1, cT2), True, False)
                self._valid_mask_cache[(T1, cT2, dev_num)] = valid_mask

            return self._valid_mask_cache[(T1, cT2, dev_num)]
        
        # 存在局部采样
        valid_mask = torch.zeros((T1, cT2), dtype=torch.bool, device=device)
        valid_mask[self.all_idx[dev_num]] = True
        
        rc = torch.nonzero(in_slice, as_tuple=False) # (p, 2)
        b_idx = self.sel_idx[dev_num][rc[:, 0]]               # (p,)
        cols = pos[rc[:, 0], rc[:, 1]].to(torch.int64)  # (p,)
        valid_mask[b_idx, cols] = True

        return valid_mask
      
    
    @torch.no_grad()
    def loom(self, 
             T1_block: Tuple[int, int], T2_block: Union[Tuple[int, int], BaseSample],
             pos: torch.Tensor, # (T2, )    or (T1, T2, )
             emb: torch.Tensor, # (T2, dim) or (T1, T2, dim)
             loss_type: str,
             _all_loss: torch.Tensor
    ):
        sample_flag = False
        if isinstance(T2_block, BaseSample):
            T2_block = T2_block.get_size()
            sample_flag = True        
            
        ### step 1: 获取基本信息
        T1 = T1_block[1] - T1_block[0]
        cT2 = T2_block[1] - T2_block[0]
        C, D   = 2 * self.k * self.h + 1, self.tp
        
        ### step 2: 获取切片
        if loss_type == 'kl' or loss_type == 'js':
            splits = [0, cT2] + [cT2] * (len(self.devices) - 1)
        else:
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
                    device=dev, s=s, e=e,
                    dim = 1 if sample_flag else 0
                )
                ### 通信1：数据传输结束 ###
                 
                
                ### 计算：计算开始 ###
                #### step 3.0: 计算欧式空间的值
                emb_T1 = self.emb_T1[i] # (T1, dim)   
                if sample_flag:
                    val_v, val_n = normalized_matmul(emb_T1[:, None, :], emb_T2.transpose(1, 2), ori_prod=(loss_type == 'kl' or loss_type == 'js')) 
                    val_v, val_n = val_v.squeeze(1), val_n.squeeze(1)
                    # (T1, 1, T2) @ (T1, dim, T2) -> (T1, 1, T2) -> (T1, T2) 
                else:             
                    val_v, val_n = normalized_matmul(emb_T1, emb_T2.t(), ori_prod=(loss_type == 'kl' or loss_type == 'js')) # (T1, T2) 
                    
                if loss_type == 'kl' or loss_type == 'js':
                    val_v = self.targets_T1[i][:, s:e].to(torch.float32) # (T1, T2)
                    val_n = 20 * torch.ones_like(val_n) # (T1, T2)
                else:
                    val_n = torch.ones_like(val_n) # (T1, T2)
                
                # (T1, dim) -> (T1, 1, dim) @ (T1, dim, T2) -> (T1, 1, T2) -> (T1, T2)
                
                #### step 3.1: 获取基本信息
                sta_loc = self.sta_loc_T1[i]        # (T1, tp)
                pos_loc = self.locations[i][pos_T2] # (T2, tp) or (T1, T2, tp)
                if not sample_flag:
                    pos_loc = pos_loc.unsqueeze(0).expand(T1, -1, -1) # (T1, T2, tp) 
                
                dis_sta_pos     = self.distance(
                    sta_loc[:, None, :]   , pos_loc[:, :, :]     , val_n[..., None]      
                )               # (T1, T2, tp)
                
                #### step 2: 获取候选位置（的值）
                cnc_loc = self.cnc_loc_T1[i] # (T1, C, tp)
                dis_sta_pos_sum = dis_sta_pos.sum(dim=-1) 
                             # (T1, T2)
                dis_cnc_pos     = self.distance(
                    cnc_loc[:, None, :, :], pos_loc[:, :, None,:], val_n[..., None, None]
                )            # (T1, T2, C, tp)
                ct_val          = (
                    dis_sta_pos_sum[:, :, None, None] - dis_sta_pos[:, :, None, :] + dis_cnc_pos
                ) / self.tp  # (T1, T2, C, tp)
                             # 对于 T1 个 starting point，向 T2 个 positive sample 连边。此时，我们把其中某个 positive sample 替换为 connected sample，共有 C 个；此时，D 个维度上的的距离是多少？
                
                #### step 3: 计算 loss
                ct_val    = ct_val                                                                # (T1, T2, C, tp)
                eu_val    = val_v[..., None, None].expand(ct_val.shape)                           # (T1, T2, C, tp)
                mask      = self.get_neighbor(T1, r_s, r_e, -1, -1, dev_num=dev_num,)             # (T1, T2)
                loss      = self.calc_loss(ct_val, eu_val, mask[..., None, None],  
                                           loss_type=loss_type)                                   # (T1, C, tp)           
                ### 计算：计算结束 ###
                
                ### 通信2：数据传输开始 ###
                _all_loss[i].copy_(loss, non_blocking=True)
                ### 通信2：数据传输结束 ###
        
        for i, stream in enumerate(self.streams):
            stream.synchronize()
        
        return _all_loss.sum(dim=0) # (T1, C, tp)


    @gettime(fmt='ms', pr=False)
    def forward(self,        
                t_eu_emb  : torch.Tensor, # (N_train, dim),    pinned memory
                t_targets : torch.Tensor, # (N_train, )   ,    pinned memory
                v_eu_emb  : torch.Tensor, # (N_valid, dim),    pinned memory
                v_targets : torch.Tensor, # (N_valid, )   ,    pinned memory
                vocab_emb : torch.Tensor, # (vocab_size, dim), pinned memory
                t_idx     : torch.Tensor, # (N_train, )   ,    pinned memory
                v_idx     : torch.Tensor, # (N_valid, )   ,    pinned memory
                vocab_idx : torch.Tensor, # (vocab_size, ),    pinned memory
                sample_factor: float = 1.,
                ### 以上张量全部放在 cpu，避免占用 gpu 内存
        ): 
        mark(ST, "total")
        mark(ST, "preparation")
        ### step 1: 获取基本信息
        N_train, N_valid, vocab_size = t_eu_emb.size(0), v_eu_emb.size(0), vocab_idx.size(0)
        # 在这里，我们需要拟合的是一个 (N_train + N_valid, N_train + vocab_size) 的矩阵
        # 所以，外层循环遍历第一维，根据 T1_block_size 来划分
        # 内层循环遍历第二维，根据 T2_block_size 来划分
        t_T1_splits     = make_splits(0, N_train, T1_block_size)
        v_T1_splits     = make_splits(N_train, N_train + N_valid, T1_block_size)
        vocab_T1_splits = [(N_train + N_valid, N_train + N_valid + vocab_size)] 
        T1_splits       = t_T1_splits + v_T1_splits + vocab_T1_splits
        
        t_T2_splits4t   = make_splits(0, N_train, T2_block_size)
        t_T2_splits4t   = Expander_Sample(N_train, int(int(math.log2(N_train)) ** 2 * sample_factor))
        # t_T2_splits4t   = CTE_Sort_Sample(int(int(math.log2(N_train)) ** 2 * sample_factor))
        t_T2_splits4t.generate_connection()
        for T1_block in T1_splits:
            t_T2_splits4t[T1_block] = t_T2_splits4t.get_connection(t_idx[T1_block[0]:T1_block[1]])
        t_T2_splits4v   = make_splits(0, N_train, T2_block_size)
        vocab_T2_splits = [(N_train, N_train + vocab_size)]
        
        ### step 2: 准备训练时常量
        neighbor_idx = {"train": (0, t_T2_splits4t.c if isinstance(t_T2_splits4t, BaseSample) else N_train + vocab_size), "valid": (0, N_train), "vocab": (N_train, N_train + vocab_size)}

        mark(ED, "preparation", father="total")
        ### step 3: 遍历所有 epoch
        mark(ST, "all_epoch")
        for epoch in range(self.epoch_num):
            mark(ST, "per_epoch")
            mark(ST, "epoch_preparation")
            self.epoch = epoch
            self.converge = self.loss_strategy['converge'] is not None and (self.epoch  >= self.loss_strategy['converge'])
            loss_split_record = {
                "train_dyn_loss" :  0,
                "train_prob_loss":  0,
                "valid_dyn_loss" :  0,
                "vocab_dyn_loss" :  0,
                "vocab_sta_loss" :  0
            }
            cur_T1_type, cur_T2_type, cur_loss_type = "train", "train", "dyn" 
            new_locations = self.main_locations.clone().pin_memory() # (emb_size, tp)
            
            # t_T2_splits4t.generate_connection(self.main_locations[:N_train])
            # t_T2_splits4t.reset_all()
            # for T1_block in T1_splits:
            #     t_T2_splits4t[T1_block] = t_T2_splits4t.get_connection(t_idx[T1_block[0]:T1_block[1]])
            # else:
            
            if epoch % 20 == 0:
                t_T2_splits4t.reset_indices()
                for T1_block in T1_splits:
                    t_T2_splits4t[T1_block] = t_T2_splits4t.get_connection(t_idx[T1_block[0]:T1_block[1]])
            
    
            mark(ED, "epoch_preparation", father="per_epoch")
            mark(ST, "epoch_train")
            ### step 3.1: 训练
            for T1_block in T1_splits:
                mark(ST, "per_T1_block")
                mark(ST, "T1_preparation")
                cur_T1_type   = get_T1_type(T1_block, N_train, N_valid, vocab_size)
                emb_T1        = get_T1_emb (T1_block, t_eu_emb, v_eu_emb, vocab_emb, N_train, N_valid, vocab_size)  # (T1_block_size, dim)
                sta_T1        = get_T1_idx (T1_block, t_idx, v_idx, vocab_idx, N_train, N_valid, vocab_size)        # (T1_block_size, )
                targets_T1    = get_T1_targets (T1_block, t_targets, N_train, N_valid, vocab_size)                  # (T1_block_size, )  
                targets_T1    = F.one_hot(targets_T1, num_classes=vocab_size) * 1e2 if not (targets_T1 == -1).any() else None     
                                                                                                                    # (T1_block_size, vocab_size)   
                self.get_neighbor(T1_block[1] - T1_block[0], -1, -1, 
                                *neighbor_idx[cur_T1_type],
                                dev_num=-1) 


                
                sta_loc_T1    = self.main_locations[sta_T1]                          # (T1_block_size, tp)
                cnc_loc_T1    = self.connection(sta_loc_T1, dev_num=-1)              # (T1_block_size, C, tp)
                
                
                self.emb_T1     = [emb_T1.to(dev, non_blocking=True) for dev in self.devices]
                self.targets_T1 = [targets_T1.to(dev, non_blocking=True) for dev in self.devices] if targets_T1 is not None else [None for dev in self.devices]
                self.sta_T1     = [sta_T1.to(dev, non_blocking=True) for dev in self.devices]
                self.sta_loc_T1 = [sta_loc_T1.to(dev, non_blocking=True) for dev in self.devices]
                self.cnc_loc_T1 = [cnc_loc_T1.to(dev, non_blocking=True) for dev in self.devices]
                
                torch.cuda.synchronize()
                
                _all_loss = torch.empty((len(self.devices), T1_block[1] - T1_block[0], 2 * self.k * self.h + 1, self.tp), dtype=torch.float32, pin_memory=True)
                all_loss = torch.zeros((T1_block[1] - T1_block[0], 2 * self.k * self.h + 1, self.tp), dtype=torch.float32, device=self.devices[0])

                T2_splits       = t_T2_splits4v + vocab_T2_splits if cur_T1_type in ["valid", "vocab"] else [t_T2_splits4t] + vocab_T2_splits
                
                mark(ED, "T1_preparation", father="per_T1_block")
                mark(ST, "T14T2")
                for T2_block in T2_splits:
                    mark(ST, "per_T2_block")
                    mark(ST, "per_T2_preparation")
                    cur_T2_type   = get_T2_type(T2_block, N_train, vocab_size)
                    
                    if (cur_T1_type == "valid" and cur_T2_type == "vocab"):
                        continue
                    
                    cur_loss_type = get_loss_type(cur_T1_type, cur_T2_type, self.loss_strategy)
                    emb_T2        = get_T2_emb (T1_block, T2_block, t_eu_emb, vocab_emb, N_train, vocab_size) # (T2, dim) or (T1, T2, dim)
                    pos_T2        = get_T2_idx (T1_block, T2_block, t_idx, vocab_idx, N_train, vocab_size)    # (T2, )    or (T1, T2)
                    mark(ED, "per_T2_preparation", father="per_T2_block")
                    mark(ST, "per_T2_loom")
                    cur_loss      = self.loom(T1_block, T2_block, pos_T2, emb_T2, cur_loss_type, _all_loss)   # (T1, C, tp)
                    mark(ED, "per_T2_loom", father="per_T2_block")
                    
                    mark(ST, "per_T2_get_loss")
                    if   cur_T1_type == "train" and T2_block == vocab_T2_splits[-1]:
                        lth         = mask_fill_scalar_expand(self.choosing_mask[0], t_T2_splits4t.c if isinstance(t_T2_splits4t, BaseSample) else N_train, self.sample_k, 1, self.converge)
                        cur_loss    = cur_loss.to(all_loss.device, non_blocking=True)
                        loss_split_record["train_dyn_loss"]  += (self._get_best_loc(all_loss)[1] / lth).sum().item()        
                                                                                                               # (sum , mean, choose, mean)
                        loss_split_record["train_prob_loss"] += (self._get_best_loc(cur_loss)[1]).sum().item()              
                                                                                                               # (sum , sum , choose, mean) 
                        all_loss    = (self.loss_strategy['ratio_dyn_prob']  * all_loss / lth[:, None, None] + # (keep, mean, keep  , keep)
                                      (1 - self.loss_strategy['ratio_dyn_prob']) * cur_loss)                   # (keep, sum , keep  , keep)
                    elif cur_T1_type == "vocab" and T2_block == vocab_T2_splits[-1]:
                        lth         = mask_fill_scalar_expand(self.choosing_mask[0], N_train, self.sample_k, 1, self.converge)
                        cur_loss    = cur_loss.to(all_loss.device, non_blocking=True)
                        loss_split_record["vocab_dyn_loss"]  += (self._get_best_loc(all_loss)[1] / lth).sum().item()        
                                                                                                               # (sum , mean, choose, mean)
                        loss_split_record["vocab_sta_loss"]  += (self._get_best_loc(cur_loss)[1] / vocab_size).sum().item()              
                                                                                                               # (sum , mean, choose, mean) 
                        all_loss    = (self.loss_strategy['ratio_dyn_sta']  * all_loss / lth[:, None, None] +  # (keep, mean, keep  , keep)
                                      (1 - self.loss_strategy['ratio_dyn_sta']) * cur_loss / vocab_size)                 
                    else:
                        all_loss    += cur_loss.to(all_loss.device, non_blocking=True)
                        if  cur_T1_type == "valid" and T2_block == t_T2_splits4v[-1]:
                            lth = mask_fill_scalar_expand(self.choosing_mask[0], N_train, self.sample_k, 1, self.converge)
                            loss_split_record["valid_dyn_loss"] += (self._get_best_loc(all_loss)[1] / lth).sum().item()        
                                                                                                             # (sum , mean, choose, mean)
                            all_loss = all_loss / lth[:, None, None]                                         # (keep, mean, keep  , keep)
                    mark(ED, "per_T2_get_loss", father="per_T2_block")
                    mark(ED, "per_T2_block")
                mark(ED, "T14T2", father="per_T1_block")
                
                mark(ST, "T1_update")
                selected_locs, _ = self._get_best_loc(all_loss, dev_num=0) # (T1_block_size, tp)
                new_locations.index_copy_(
                    0,
                    sta_T1.view(-1),
                    selected_locs.view(-1, self.tp).cpu()
                )
                self.reset()
                mark(ED, "T1_update", father="per_T1_block")
                
                mark(ED, "per_T1_block")
            
            mark(ED, "epoch_train", father="per_epoch")
            
            ### step 3.2: 更新
            mark(ST, "epoch_update")
            self.main_locations.copy_(new_locations, non_blocking=True)
            for i, dev in enumerate(self.devices):
                self.locations[i].copy_(self.main_locations.to(dev, non_blocking=True), non_blocking=True)
            mark(ED, "epoch_update", father="per_epoch")
            
            ### step 3.3: 记录与打印
            mark(ST, "epoch_summary")
            print(f"epoch {epoch:3d} summary:", end=", ")
            for k, v in loss_split_record.items():
                if   k.startswith('train'):
                    v /= N_train
                elif k.startswith('valid'):
                    v /= N_valid
                elif k.startswith('vocab'):
                    v /= vocab_size
                
                print(f"{k:15s}: {v:.4f}", end=", ")
            
            ### step 3.4: 验证
            for split in ['train', 'valid']:
                splits  = t_T1_splits if split == 'train' else v_T1_splits
                dyn_idx = t_idx       if split == 'train' else v_idx
                dyn_emb = t_eu_emb    if split == 'train' else v_eu_emb
                targets = t_targets   if split == 'train' else v_targets
                N       = N_train     if split == 'train' else N_valid
                loss    = 0
                accuray = 0
                
                for block in splits:
                    Tl, Tr      = block if split == 'train' else (block[0] - N_train, block[1] - N_train)
                    dyn_loc     = self.locations[0][dyn_idx[Tl:Tr].to(self.devices[0], non_blocking=True)] # (T1, tp)
                    vocab_loc   = self.locations[0][vocab_idx.to(self.devices[0], non_blocking=True)] # (N_vocab, tp)
                    _, norm     = normalized_matmul(
                        dyn_emb[Tl:Tr]. to(self.devices[0], non_blocking=True), 
                        vocab_emb.to(self.devices[0], non_blocking=True).t()
                    ) # (T1, N_vocab)
                    ct_val      = self.distance(dyn_loc[:, None, :], vocab_loc[None, :, :], norm[..., None]).mean(dim=-1) # (T1, N_vocab)
                    loss       += F.cross_entropy(ct_val, targets[Tl:Tr].to(self.devices[0], non_blocking=True), reduction='sum').item()
                    accuray    += (ct_val.argmax(dim=-1) == targets[Tl:Tr].to(self.devices[0], non_blocking=True)).sum().item()
                
                loss = loss / N
                accuray = accuray / N
                print(f"{split:5s} loss: {loss:.4f}, accuracy: {accuray:.4f}", end=", ")

            print()
            
            mark(ED, "epoch_summary", father="per_epoch")
            mark(ED, "per_epoch")
            
            ### step 3.5: 可视化。仅可视化前 256 个
            if (epoch + 1) % 100 == 0 or epoch == 0:
                train_eu_emb = t_eu_emb[:256]                    # (256, dim)
                valid_eu_emb = v_eu_emb[:256]                    # (256, dim)
                S_tt_eu      = normalized_matmul(train_eu_emb, train_eu_emb.t())[0].cpu().numpy()
                S_vt_eu      = normalized_matmul(valid_eu_emb, train_eu_emb.t())[0].cpu().numpy()
                
                train_ct_emb = self.main_locations[t_idx[:256]]  # (256, tp)
                valid_ct_emb = self.main_locations[v_idx[:256]]  # (256, tp)
                S_tt_ct      = self.distance(train_ct_emb[:, None, :], train_ct_emb[None, :, :], torch.ones((256, 256, 1))).mean(dim=-1).cpu().numpy()
                S_vt_ct      = self.distance(valid_ct_emb[:, None, :], train_ct_emb[None, :, :], torch.ones((256, 256, 1))).mean(dim=-1).cpu().numpy()

                # visualize_similarity   (S_tt_eu, S_tt_ct, meta_name="{}" + "train_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0), loss_dyn_dyn=loss_split_record["train_dyn_loss"] / N_train)
                visualize_similarity   (S_tt_eu, S_tt_ct, meta_name="{}" + f"CT_Sorted_T{N_train}_C{sample_factor}" + ".png", save_eu=(epoch == 0), loss_dyn_dyn=loss_split_record["train_dyn_loss"] / N_train)
                visualize_pair_bihclust(S_vt_eu, S_vt_ct, meta_name="{}" + "valid_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))

        mark(ED, "all_epoch", father="total")
        mark(ED, "total")
        
    def reset(self):
        self.sta_T1        = [None for _ in range(len(self.devices))]
        self.sta_loc_T1    = [None for _ in range(len(self.devices))]
        self.cnc_loc_T1    = [None for _ in range(len(self.devices))]
        self.valid_samples = None
        self._valid_mask_cache = {}

    def _get_best_loc(self, loss: torch.Tensor, dev_num: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.devices[dev_num] if dev_num >= 0 else 'cpu'
        
        cnc_indices   = torch.argmin(loss, dim=1)                                        # (T1, tp)
        T1_indices    = torch.arange(self.sta_loc_T1[0].size(0), device=device)[:, None] # (T1, 1)
        dim_indices   = torch.arange(self.tp    ,                device=device)[None :]  # (1, tp)
        selected_locs = self.cnc_loc_T1[dev_num][T1_indices, cnc_indices, dim_indices]   # (T1, tp)
        real_loss     = loss[T1_indices, cnc_indices, dim_indices].mean(dim=-1)          # (T1, ) 
        return selected_locs, real_loss
    
if __name__ == "__main__":
    pass

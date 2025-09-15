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
        self.n : int = h
        self.emb_size : int = emb_size
        
        ### 2. CT Space Embeddings 初始化
        self.main_locations = torch.randint(
            1 - self.n, self.n, (self.emb_size, self.tp), 
            dtype=torch.int64, device=main_device, generator=generators[0]
        )

        ### 3. 额外参数
        self._pending_refs = {sid: [] for sid in range(self.num_devices)}        
        self.timer = CUDATimer()
    
    
    def generate_random_masks(self, sz, dev_num=0):
        device = self.devices[dev_num] if dev_num >= 0 else 'cpu'

        upper_bounds   = 2 ** torch.arange(self.h, dtype=torch.int64, device=device)
        random_numbers = torch.randint(
            0, self.n, 
            (self.h, sz, self.k, self.tp), 
            dtype=torch.int64, device=device, generator=generators[dev_num]
        ) # (H, B*T, K, D)
        masks = random_numbers & (upper_bounds.view(-1, 1, 1, 1) - 1)
        # masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3) # (B*T, H, K, D)
    
    def connection(self, ori_int: torch.Tensor, dev_num=0):
        device = self.devices[dev_num] if dev_num >= 0 else 'cpu'
        
        flip_masks = (1 << torch.arange(self.h, device=device, dtype=ori_int.dtype)).unsqueeze(0).unsqueeze(2)
        flipped_ints = ori_int.unsqueeze(1) ^ flip_masks # (B*T1, H, D)
        random_masks = self.generate_random_masks(flipped_ints.size(0), dev_num=dev_num)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h * N_K, self.tp)
        # (B*T1, H, 1, D) ^ (B*T1, H, K, D) -> (B*T1, H*K, D)
        loc = torch.cat((result, ori_int.unsqueeze(1), -result), dim=1) # (B*T1, H*K + 1 + H*K, D)
        return loc
              
    def generate_mask(self, size0: int, size1: int, dev_num=0):
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
                
        lth = mask.sum(dim=1) + 1e-12

        return mask, lth

    # @torch.compile()
    def get_best_loc(
        self, 
        cnc_loc : torch.Tensor, # (T, C, D)
        loss_cos: torch.Tensor, 
        loss_cro: torch.Tensor,
        loss_tot: torch.Tensor,
        sid     : int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        device = self.devices[sid]
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

    # @torch.compile()
    def cos_similarity(self, coord1: torch.Tensor, coord2: torch.Tensor) -> torch.Tensor:
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        xor_result = torch.abs(coord1) ^ torch.abs(coord2)
        _, exp = torch.frexp((xor_result + 1).to(torch.float32))
        s = exp.float() / self.h
        return sg * (1 - s)
    
    # @torch.compile()
    # @torch.compile(dynamic=True)
    def loom_train(
        self, 
        sta_loc: torch.Tensor,   # (T_train, dim_ct)
        cnc_loc: torch.Tensor,   # (T_train, N_C    , dim_ct)
        pos_loc: torch.Tensor,   # (T_train, N_trnbr, dim_ct)
        voc_loc: torch.Tensor,   # (T_train, K_vocab, dim_ct)

        sta_emb: torch.Tensor,   # (T_train, dim_eu)
        nei_emb: torch.Tensor,   # (T_train, N_trnbr, dim_eu)
        voc_emb: torch.Tensor,   # (T_train, K_vocab, dim_eu)
        
        mask   : torch.Tensor,   # (T_train, N_trnbr)
        lth    : torch.Tensor,   # (T_train)
        sid    : int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor,]:
        device = devices[sid]
        
        ### step 1: 计算欧氏空间的相似度
        eu_cos_val = F.normalize(sta_emb, dim=-1) @ F.normalize(nei_emb, dim=-1).transpose(-1, -2)  # (T_train, N_trnbr)
        if use_eu_norm == True:
            eu_cro_nrm = torch.norm(sta_emb, dim=-1, keepdim=True) @ torch.norm(voc_emb, dim=-1, keepdim=True).transpose(-1, -2)  
        else:
            eu_cro_nrm = torch.ones((T_train, K_vocab), device=device) * 20.
        
        ### step 2: 计算 CT 空间的相似度  
        pos_loc    = torch.cat([pos_loc, voc_loc], dim=1)  
                                                 # (T_train, N_trnbr + K_vocab, dim_ct)
        cos_sta_pos     = self.cos_similarity(
            sta_loc[:, None, :], pos_loc[:, :, :]      
        )                                        # (T_train, N_trnbr          , dim_ct)
        cos_sta_pos_sum = cos_sta_pos.sum(dim=-1) 
                                                 # (T_train, N_trnbr)
        cos_cnc_pos     = self.cos_similarity(
            cnc_loc[:, None, :, :], pos_loc[:, :, None,:]
        )                                        # (T_train, N_trnbr          , dim_ct)
        ct_val          = (
            cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + cos_cnc_pos
        ) / self.tp          
                
        '''
        # ct_val = ct_val_triton(
        #     cnc_loc.to(torch.int32).contiguous(),
        #     nei_loc.to(torch.int32).contiguous(),
        #     eu_cro_nrm.contiguous(),
        #     cos_sta_nei.contiguous(),
        #     cos_sta_pos_sum.contiguous(),
        #     tp=float(self.tp),
        #     h=float(self.h),
        #     out=None,                # 或传入你复用的 out 缓冲
        #     BLOCK_S=32,
        #     BLOCK_CD=32,
        #     NUM_WARPS=8,
        #     NUM_STAGES=2,
        # )
        '''
        
                            # (T_train, N_trnbr, N_C, dim_ct)
        ## 对于 T 个 starting point，向 S_tot 个 positive sample 连边。此时，我们把其中某个 positive sample 替换为 connected sample，共有 C 个；此时，D 个维度上的的距离是多少？

        ### step 3: 计算 loss
        ### step 3.1: 计算 cosine-similarity loss
        loss_cos   = ct_val[:, 0:N_trnbr]                             # (T_train, N_trnbr, N_C, dim_ct)
        eu_cos_val = eu_cos_val[..., None, None].expand(ct_val.shape) # (T_train, N_trnbr, N_C, dim_ct)
        loss_cos.sub_(eu_cos_val)                                     # (T_train, N_trnbr, N_C, dim_ct)
        loss_cos.pow_(2)                                              # (T_train, N_trnbr, N_C, dim_ct)
        loss_cos.mul_(torch.abs(eu_cos_val))                          # 让更相似的点权重大一些
        loss_cos.mul_(mask[..., None, None])                          # (T_train, N_trnbr, N_C, dim_ct)
        loss_cos   = loss_cos.sum(dim=1) / lth[:, None, None]         # (T_train, N_C, dim_ct)
        
        ### step 3.2: 计算 cross-entropy     loss
        ct_cro_val = ct_val[:, N_trnbr:]                              # (T_train, K_vocab, N_C, dim_ct)
        ct_cro_val = ct_cro_val * eu_cro_nrm[:, :, None, None]        # (T_train, K_vocab, N_C, dim_ct)
        loss_cro   = sampled_softmax_ce_uniform(ct_cro_val, voc_dim=1, V=N_vocab)
        loss_cro   = loss_cro.sum(dim=1)                              # (T_train, N_C, dim_ct)
        
        return loss_cos, loss_cro
                


    def loom_vocab(self, sid: int):
        ...
    
    
    
    def train_train_blocks(self, sid: int):
        for train_block_id in train4sid[sid]:
            train_block = train_blocks[train_block_id]
            train_slice = slice(train_block[0], train_block[1])
            
            ### step.1 准备数据
            with torch.cuda.device(0), torch.cuda.stream(data_streams[sid]):
                _cur_tar = self.train_tar[train_slice]                      # (T_train, )
                _nei_idx = self.train_sampler.get_connection(train_block)   # (T_train, N_trnbr)
                _voc_idx = self.vocab_sampler.get_connection(_cur_tar)      # (T_train, K_vocab)

                _sta_loc = self.main_locations[train_slice]                 # (T_train, dim_ct)
                _nei_loc = self.main_locations[_nei_idx]                    # (T_train, N_trnbr, dim_ct)
                _voc_loc = self.main_locations[_voc_idx]                    # (T_train, K_vocab, dim_ct)

                _sta_emb = self.train_emb[train_slice]                      # (T_train, dim_eu)
                _nei_emb = self.train_emb[_nei_idx]                         # (T_train, N_trnbr, dim_eu)
                _voc_emb = self.vocab_emb[_voc_idx]                         # (T_train, K_vocab, dim_eu)

                sta_loc, nei_loc, voc_loc, sta_emb, nei_emb, voc_emb = (
                    _sta_loc.to(device, non_blocking=True), 
                    _nei_loc.to(device, non_blocking=True), 
                    _voc_loc.to(device, non_blocking=True),
                    _sta_emb.to(device, non_blocking=True), 
                    _nei_emb.to(device, non_blocking=True), 
                    _voc_emb.to(device, non_blocking=True)
                )

            ### step.2 计算 loss 并选择最佳位置
            with torch.cuda.device(device), torch.cuda.stream(comp_streams[sid]):
                cnc_loc   = self.connection(sta_loc, dev_num=sid)
                mask, lth = self.generate_mask(T_train, N_trnbr, sid)

                loss_cos, loss_cro = self.loom_train(
                    sta_loc, cnc_loc,
                    nei_loc, voc_loc,
                    sta_emb, nei_emb, voc_emb,
                    mask, lth, sid
                )
                
                loss_tot = loss_strategy['ratio_cos'] * loss_cos + loss_strategy['ratio_cro'] * loss_cro
                loss_cos_T, loss_cro_T, loss_tot_T, selected_locs = self.get_best_loc(cnc_loc, loss_cos, loss_cro, loss_tot, sid) # (T_train, ) * 3, (T_train, dim_ct)
                
                self.main_locations[train_slice].copy_(selected_locs, non_blocking=True)
                self.loss_cos_buf[train_slice].copy_(loss_cos_T, non_blocking=True)
                self.loss_cro_buf[train_slice].copy_(loss_cro_T, non_blocking=True)
                self.loss_tot_buf[train_slice].copy_(loss_tot_T, non_blocking=True)

    def train_vocab_blocks(self, sid: int):
        cnc_loc = self.connection(self.main_locations[vocab_loc_slice], dev_num=sid)
        cnc_loc4sid = [cnc_loc.to(dev) for dev in devices]
        self._synchronize_all_streams()
        
        for vonbr_block_id in vonbr4sid[sid]:
            vonbr_block = vonbr_blocks[vonbr_block_id]          # vonbr, vocabulary neighbors, 其邻居始终是训练集的子集
            vonbr_slice = slice(vonbr_block[0], vonbr_block[1])
            
            ### step.1 准备数据
            with torch.cuda.device(0), torch.cuda.stream(data_streams[sid]):
                _cur_tar = self.train_tar[vonbr_slice]                      # (T_vonbr, )
                _voc_idx = self.vocab_sampler.get_connection(_cur_tar)      # (T_vonbr, K_vocab)

                _nei_loc = self.main_locations[vonbr_slice]                 # (T_vonbr, dim_ct)
                _voc_loc = self.main_locations[_voc_idx]                    # (T_vonbr, K_vocab, dim_ct)

                _nei_emb = self.train_emb[vonbr_slice]                      # (T_vonbr, dim_eu)
                _voc_emb = self.vocab_emb[_voc_idx]                         # (T_vonbr, K_vocab, dim_eu)

                nei_loc, voc_loc, nei_emb, voc_emb = (
                    _nei_loc.to(device, non_blocking=True), 
                    _voc_loc.to(device, non_blocking=True),
                    _nei_emb.to(device, non_blocking=True), 
                    _voc_emb.to(device, non_blocking=True)
                )

            ### step.2 计算 loss 并选择最佳位置
            with torch.cuda.device(device), torch.cuda.stream(comp_streams[sid]):
                loss_cos, loss_cro = self.loom_train(
                    voc_loc, cnc_loc4sid[sid], nei_loc,
                    nei_emb, voc_emb,
                    sid
                )
                
                loss_tot = loss_strategy['ratio_cos'] * loss
         
        
        
    def train_epoch(self, sid: int):
        try:
            stream = comp_streams[sid]
            device = devices[sid]
            for cur_epoch in range(train_epoch_num):
                self.epoch_barrier.wait()
                
                ### step 1: 考虑 train_blocks
                self.train_train_blocks(sid)
                ### step 2: 考虑 vocab_blocks

                
                self.epoch_barrier.wait()
        except Exception as e:
            print(f"Exception in thread {sid}: {e}")
            import traceback
            traceback.print_exc()
            os._exit(1)  # 立即终止所有线程和进程
                    

    @gettime(fmt='ms', pr=True)
    def train_all(
        self,        
        train_emb : torch.Tensor, # (N_train, dim), pinned memory
        vocab_emb : torch.Tensor, # (N_vocab, dim), pinned memory
        valid_emb : torch.Tensor, # (N_valid, dim), pinned memory

        train_tar : torch.Tensor, # (N_train, )   , pinned memory
        valid_tar : torch.Tensor, # (N_valid, )   , pinned memory
        
        train_sample_factor: float = 1.,
        ### 以上张量全部放在 cpu，避免占用 gpu 内存
    ):  
        """
        我们需要拟合的是一个:
        1. 对于 train: (N_train, N_train) 的 cos-similarity, (N_train, N_vocab) 的 cross-entropy
        2. 对于 vocab: (N_vocab, N_vocab) 的 cos-similarity, (N_vocab, N_train) 的 cross-entropy
        3. 对于 valid: (N_valid, N_train) 的 cos-similarity, (N_valid, N_vocab) 的 ... 想得美，valid 怎么能看到 groundtruth 呢？
        """
        mark(ST, "all")
        mark(ST, "all_preparation")
        
        ### step.2 准备数据分块与采样器
        mark(ST, "all_preparation_2")
        self.train_tar     = train_tar.long().to(main_device)
        self.valid_tar     = valid_tar.long().to(main_device)
        self.train_emb     = train_emb.to(main_device)
        self.vocab_emb     = vocab_emb.to(main_device)
        self.valid_emb     = valid_emb.to(main_device)
        
        self.locations     = [self.main_locations.to(dev) for dev in devices]
        
        self.train_sampler = TrainSampler()
        self.vocab_sampler = VocabSampler()
        self.valid_sampler = ValidSampler()
        
        self.loss_cos_buf  = torch.zeros(self.emb_size, device=main_device)
        self.loss_cro_buf  = torch.zeros(self.emb_size, device=main_device)
        self.loss_tot_buf  = torch.zeros(self.emb_size, device=main_device)
        self._synchronize_all_streams()
        mark(ED, "all_preparation_2", father="all_preparation")
        
        
        ### step 3: 开启多线程
        mark(ST, "all_preparation_3")
        self.epoch_barrier  = threading.Barrier(self.num_devices + 1) # num_devices 个生产线程，num_devices 个消费线程，1 个主线程
        threads: List[threading.Thread] = []
        for i, _ in enumerate(self.devices):
            thread = threading.Thread(target=self.train_epoch, args=(i,))
            thread.start()
            threads.append(thread)
        mark(ED, "all_preparation_3", father="all_preparation")
        mark(ED, "all_preparation", father="all")
        
        
        mark(ST, "all_epoch")
        ### step 3: 遍历所有 epoch
        for cur_epoch in range(train_epoch_num):
            mark(ST, "epoch")
            mark(ST, "epoch_preparation")
            self.cur_epoch = cur_epoch
            self.converge  = loss_strategy['converge'] is not None and (self.cur_epoch >= loss_strategy['converge'])
            loss_split_record = {
                "train_cos_loss":  0.,
                "train_cro_loss":  0.,
                "vocab_cos_loss":  0.,
                "vocab_cro_loss":  0.,
                "train_tot_loss":  0.,
            }
            
            if cur_epoch % 50 == 0 and cur_epoch != 0:
                self.train_sampler.reset_indices()
            

            self._synchronize_all_streams()
            mark(ED, "epoch_preparation", father="epoch")
            mark(ST, "epoch_train")
            
            ### step 3.1: 训练
            #### 主线程通过 wait 唤醒消费者线程
            self.epoch_barrier.wait()
            
            #### 主线程通过 wait 等待所有消费者线程完成
            self.epoch_barrier.wait()
            mark(ED, "epoch_train", father="epoch")
            mark(ST, "epoch_pos_train")
            # self.timer.finish_round() 
            
            ### 整合数据
            # self.main_locations = torch.cat([self.loc_splits[i].to(main_device) for i in range(len(self.splits))], dim=0)
            # loss_cos = torch.cat([self.loss_cos_buf[i].to(main_device) for i in range(len(self.splits))], dim=0)
            # loss_cro = torch.cat([self.loss_cro_buf[i].to(main_device) for i in range(len(self.splits))], dim=0)
            # loss_tot = torch.cat([self.loss_tot_buf[i].to(main_device) for i in range(len(self.splits))], dim=0)
            self._synchronize_all_streams()
            # self.loc_splits = [
            #     self.main_locations[block[0]:block[1]].to(self.devices[i % self.num_devices])  
            #     for i, block in enumerate(self.splits)
            # ]
            self.voc_loc = [self.main_locations[self.sta_slice].unsqueeze(0).expand(self.T_train, -1, -1).to(dev) for dev in self.devices] # 如果出现 CUDA BUG，可以尝试把这行代码放到上面 epoch 循环的开头
            self._pending_refs = {sid: [] for sid in range(self.num_devices)}
            
            self.sampler.update_locations(self.main_locations)
            self.prefetcher.new_epoch()

            mark(ED, "epoch_pos_train", father="epoch")
            ### step 3.2: 记录与打印
            
            for cur_type in ["train"]:
                if cur_type == "train":
                    cur_N = self.N_train
                    cur_slice = slice(0, self.N_train)
                else:
                    cur_N = self.N_valid
                    cur_slice = slice(self.N_train, self.N_train + self.N_valid)
                
                loss_split_record[f"{cur_type}_cos_loss"] = self.loss_cos_buf[cur_slice].sum().item() / cur_N
                loss_split_record[f"{cur_type}_cro_loss"] = self.loss_cro_buf[cur_slice].sum().item() / cur_N
                loss_split_record[f"{cur_type}_tot_loss"] = self.loss_tot_buf[cur_slice].sum().item() / cur_N
            
            print(f"epoch {cur_epoch:3d} summary:", end=" ")
            for k, v in loss_split_record.items():    
                if k.startswith("train_cos"): 
                    print(f"{k:15s}: {v:.6f}", end=", ")
                else:
                    print(f"{k:15s}: {v:.4f}", end=", ")
            ### step 3.3: 验证
            mark(ST, "epoch_valid")
            self.validate(cur_epoch)
            mark(ED, "epoch_valid", father="epoch")
            mark(ED, "epoch")
            
            ### step 3.4: 可视化
            self.visualize(cur_epoch)

        print("一切都已经结束了")
        
        self.prefetcher.stop()
        for thread in threads:
            thread.join()
        
        # print(kernel_ct_val_fused_cd.autotune_cache[(self.N_T, self.S_tot, 2 * self.h * self.k + 1, self.tp)]) 
        
        # self.timer.summary()
        mark(ED, "all_epoch", father="all")
        mark(ED, "all")
    
    def test_time_train_all(
        self,
        reset_locations: bool = False
    ):
        ...
        
    
    def validate(self, epoch):
        loss = {"train": 0., "valid": 0.}
        accuracy = {"train": 0., "valid": 0.}
        if epoch % 10 != 0 and epoch != 0:
            return
        
        for i, block in enumerate(self.splits):
            cur_type    = get_type(block, self.N_train, self.N_valid)
            targets     = self.tar_splits[i].to(main_device)      # (T, )
            dyn_loc     = self.main_locations[block[0]:block[1]]  # (T, D)
            sta_loc     = self.main_locations[self.sta_slice]     # (V, D)
            ct_val      = self.cos_similarity(
                dyn_loc[:, None, :], # (T, 1, D) 
                sta_loc[None, :, :], # (1, V, D)
                20. * torch.ones((1, 1, 1), device=main_device, dtype=torch.float32) # (1, 1, 1)
            ).mean(dim=-1) # (T, V)
            if block[0] < self.N_dyn and self.N_dyn <= block[1]:
                ct_val  = ct_val [:self.N_dyn - block[0]]
                targets = targets[:self.N_dyn - block[0]]
            loss[cur_type]        += F.cross_entropy(ct_val, targets, reduction='sum').item()
            accuracy[cur_type]    += (ct_val.argmax(dim=-1) == targets).sum().item()

        for cur_type in ["train", "valid"]:
            if cur_type == "train":
                cur_N = self.N_dyn
            else:
                cur_N = self.N_valid
            loss[cur_type]     /= cur_N
            accuracy[cur_type] /= cur_N
            
            print(f"{cur_type:5s} loss: {loss[cur_type]:.4f}, accuracy: {accuracy[cur_type]:.4f}", end=", ")

        print()
    
    def visualize(self, epoch: int):
        if (epoch ) % 50 == 0 or epoch == self.epoch_num - 1:
            train_eu_emb = self.sampler.emb_val[:256]                           # (256, dim)
            valid_eu_emb = self.sampler.emb_val[self.N_train:self.N_train+256]  # (256, dim)
            S_tt_eu      = normalized_matmul(train_eu_emb, train_eu_emb.t())[0].cpu().numpy()
            S_vt_eu      = normalized_matmul(valid_eu_emb, train_eu_emb.t())[0].cpu().numpy()
            
            train_ct_emb = self.main_locations[0:256]                              # (256, tp)
            valid_ct_emb = self.main_locations[self.N_train : self.N_train + 256]  # (256, tp)
            S_tt_ct      = self.cos_similarity(train_ct_emb[:, None, :], train_ct_emb[None, :, :], torch.ones((256, 256, 1), device=main_device)).mean(dim=-1).cpu().numpy()
            S_vt_ct      = self.cos_similarity(valid_ct_emb[:, None, :], train_ct_emb[None, :, :], torch.ones((256, 256, 1), device=main_device)).mean(dim=-1).cpu().numpy()

            visualize_similarity   (S_tt_eu, S_tt_ct, meta_name="{}" + "train_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))
            visualize_pair_bihclust(S_vt_eu, S_vt_ct, meta_name="{}" + "valid_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))

    def _synchronize_all_streams(self):
        torch.cuda.synchronize()
        torch.cuda.default_stream(main_device).synchronize()
        for sid in range(num_devices):
            comp_streams[sid].synchronize()
        

        
if __name__ == "__main__":
    pass

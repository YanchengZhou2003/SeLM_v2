import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
# torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
from matplotlib.colors import PowerNorm
from torch.nn import functional as F
from tqdm import tqdm

from src.gettime import CUDATimer, gettime, mark
# from src.loom_kernel import triton_loom_wrapper
from src.loss import compute_loss, compute_weighted_loss
from src.para import ED, N_T, ST, instant_writeback, n_embd, vocab_size
from src.sampler import BaseSample, Expander_Sampler
from src.utils import *
from src.vis import visualize_pair_bihclust, visualize_similarity

main_device = torch.device(0) 
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
        
        self.timer = CUDATimer()


    
    
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
                  S: int, S_: int, 
                  sum_dim  : int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_type   = (self.loss_strategy['cos_loss'], self.loss_strategy['cro_loss'])
        weight  = (self.loss_strategy['ratio_cos'], self.loss_strategy['ratio_cro'])
        loss_cos, loss_cro, loss_tot = compute_weighted_loss(
            loss_type, weight,
            ct_val, eu_val, 
            mask, lth,
            S, S_,
            sum_dim
        ) 
        return loss_cos, loss_cro, loss_tot
        
    def generate_mask(self, block: Tuple[int, int], N_dyn: int, N_sta: int):
        """
        我们期望的 mask 是这样的：
        1. 对于每一个 Tl, Tr 的块，我们需要生成 (Tr - Tl, S_) 的 mask，它的生成规则如下：
           (a) 如果 self.converge == False，则以 0.8 的概率，整行在 [0:S) 范围内全为 True
           (b) 如果 self.converge == False，则以 0.2 的概率，整行在 [0:S) 范围内全为 True，但随机选择 self.sample_k 个位置设为 True
           (c) 如果 self.converge == True ，则整行在 [0:S) 范围内全为 True
           (d) 如果 S_ > S，则从 max(N_dyn, Tl) 行开始，(Tr - Tl, S_:S_) 全为 False；其它行为 True
        2. 对于每一个 Tl, Tr 的块，我们需要生成 (Tr - Tl, ) 的 lth，它统计的是每一行在 [0:S) 范围内的 True 数量
        """
        self.masks = []
        self.lth = []
        for i, dev in enumerate(self.devices):
            s, e = block[0] + self.sub_splits[i].start, block[0] + self.sub_splits[i].stop
            T1_local = e - s
            with torch.cuda.device(dev), torch.cuda.stream(self.streams[i]):
                mask = torch.ones((T1_local, self.S), dtype=torch.bool, device=dev)

                if not self.converge:
                    choosing_mask = (torch.rand((T1_local,), device=dev) > 0.2)  # True=整行在[0:S)全True
                    sel_idx = (~choosing_mask).nonzero(as_tuple=False).squeeze(1)
                    if sel_idx.numel() > 0:
                        mask[sel_idx, :self.S - N_sta] = False
                        rows = sel_idx.repeat_interleave(self.sample_k)
                        cols = torch.randint(0, self.S - N_sta, (rows.numel(),), device=dev)
                        mask[rows, cols] = True

                local_start = max(N_dyn, s) - s
                if local_start < T1_local:
                    mask[local_start:, self.S - N_sta:self.S] = False

                lth_local = mask[:, :self.S - N_sta].sum(dim=1).to(torch.int32) + 1e-12
                self.masks.append(mask)
                self.lth.append(lth_local)

        for stream in self.streams:
            stream.synchronize()


    def distance(self, coord1: torch.Tensor, coord2: torch.Tensor, norm: torch.Tensor):
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        xor_result = torch.abs(coord1) ^ torch.abs(coord2)
        _, exp = torch.frexp((xor_result + 1).to(torch.float32))
        s = exp.float() / self.h
        return sg * (1 - s) * norm
    
    @torch.no_grad()
    def loom(
        self, 
        N_sta: int,
        _loss_cos: torch.Tensor, # (T, C, D)
        _loss_cro: torch.Tensor, # (T, C, D)
        _loss_tot: torch.Tensor, # (T, C, D)
    ):
            
        ### step 1: 获取基本信息
        T, S = N_T, self.S
        C, D = 2 * self.k * self.h + 1, self.tp
        
        ### step 2: 开始计算
        for i, (dev, stream, split) in enumerate(zip(self.devices, self.streams, self.sub_splits)):
            ### step 2.1: 基本数据
            Tl, Tr = split.start, split.stop
            subT = Tr - Tl
            dev_num = i
            
            ### step 2.2: 多 GPU 处理
            with torch.cuda.device(dev), torch.cuda.stream(stream): # type: ignore        
                ### step 2.2.1 : 获取基本信息，拼接张量
                self.timer.mark(f"dev{dev_num}_all", 0)
                self.timer.mark(f"dev{dev_num}_2.2.1", 0)
                pos_loc, pos_emb = self.pos_loc[i], self.pos_emb[i]
                mask, lth = self.masks[i], self.lth[i] 
                                                 # (subT, S), (subT)
                sta_emb = self.sta_emb[i]        # (subT, dim)
                sta_loc = self.sta_loc[i]        # (subT, D)
                cnc_loc = self.cnc_loc[i]        # (subT, C , D)
                self.timer.mark(f"dev{dev_num}_2.2.1", 1)
                
                ### step 2.2.2 : 计算欧式空间的值
                self.timer.mark(f"dev{dev_num}_2.2.2", 0)
                eu_val  = (sta_emb[:, None, :] @ pos_emb.transpose(1, 2)).squeeze(1)
                                                 # (subT, 1, dim ) @ (subT, dim, S) -> (subT, 1, S) -> (subT, S)
                # print(eu_val.shape, S, N_sta, self.targets[i].shape if self.targets[i] is not None else None)
                eu_val  [:, S - N_sta:]   = self.targets[i] if self.targets[i] is not None else 0
                                                 # (subT, S - N_sta:S), 的部分放为 groundtruth, (subT, :S - N_sta) 的部分维持
                eu_norm                   = torch.cat([
                    torch.ones((subT, S - N_sta), device=dev) ,
                    torch.ones((subT, N_sta), device=dev) * (20 if self.targets[i] is not None else 0)
                ], dim=1)                        # (subT, S)

                self.timer.mark(f"dev{dev_num}_2.2.2", 1)
                
                #### step 2.2.3 : 计算 CT 余弦相似度
                self.timer.mark(f"dev{dev_num}_2.2.3", 0)
                cos_sta_pos     = self.distance(
                    sta_loc[:, None, :]   , pos_loc[:, :, :]     , eu_norm[..., None]      
                )                                        # (subT, S, D)
                cos_sta_pos_sum = cos_sta_pos.sum(dim=-1) 
                                                         # (subT, S)
                self.timer.mark(f"dev{dev_num}_2.2.3", 1)
                
                ### step 2.2.4 : 计算 CT connected sample 的余弦相似度
                self.timer.mark(f"dev{dev_num}_2.2.4", 0)
                cos_cnc_pos     = self.distance(
                    cnc_loc[:, None, :, :], pos_loc[:, :, None,:], eu_norm[..., None, None]
                )                                        # (subT, S, C, D)
                ct_val          = (
                    cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + cos_cnc_pos
                ) / self.tp                              # (subT, S, C, D)
                # 对于 subT 个 starting point，向 S 个 positive sample 连边。此时，我们把其中某个 positive sample 替换为 connected sample，共有 C 个；此时，D 个维度上的的距离是多少？
                self.timer.mark(f"dev{dev_num}_2.2.4", 1)
                
                #### step 2.2.5: 计算 loss
                self.timer.mark(f"dev{dev_num}_2.2.5", 0)
                ct_val    = ct_val                                         # (subT, S, C, D)
                eu_val    = eu_val[..., None, None].expand(ct_val.shape)   # (subT, S, C, D)

                loss_cos, loss_cro, loss_tot = self.calc_loss(
                    ct_val, eu_val, 
                    mask[..., None, None], lth,
                    S - N_sta, S
                )                                                          # (subT, C, tp)      
                self.timer.mark(f"dev{dev_num}_2.2.5", 1)     
                ### 计算：计算结束 ###
                
                ### 通信2：数据传输开始 ###
                self.timer.mark(f"dev{dev_num}_2.2.6", 0)
                _loss_cos[Tl:Tr].copy_(loss_cos, non_blocking=True)
                _loss_cro[Tl:Tr].copy_(loss_cro, non_blocking=True)
                _loss_tot[Tl:Tr].copy_(loss_tot, non_blocking=True)
                ### 通信2：数据传输结束 ###
                self.timer.mark(f"dev{dev_num}_2.2.6", 1)
                self.timer.mark(f"dev{i}_all", 1)
                
        
        for i, stream in enumerate(self.streams):
            stream.synchronize()
            
        
        self.timer.finish_round()

    @gettime(fmt='ms', pr=True)
    def train_all(self,        
        train_emb : torch.Tensor, # (N_train, dim), pinned memory
        valid_emb : torch.Tensor, # (N_valid, dim), pinned memory

        train_tar : torch.Tensor, # (N_train, )   , pinned memory
        valid_tar : torch.Tensor, # (N_valid, )   , pinned memory
        
        sample_factor: float = 1.,
        ### 以上张量全部放在 cpu，避免占用 gpu 内存
    ):  
        N_train, N_valid = train_emb.size(0), valid_emb.size(0)
        train_emb = train_emb.to("cuda:0", non_blocking=True)
        valid_emb = valid_emb.to("cuda:0", non_blocking=True)
        train_idx = torch.arange(N_train).to("cuda:0", non_blocking=True)
        valid_idx = torch.arange(N_train, N_train + N_valid).to("cuda:0", non_blocking=True)

        """
        我们需要拟合的是一个：(N_train, N_train) 的矩阵，以及 (N_valid, N_train) 的矩阵
        并且构造 (N_dyn, N_sta) 的 groundtruth，向它对齐
        """
        mark(ST, "all")
        
        mark(ST, "all_preparation")
        
        mark(ST, "all_preparation_1")
        ### step 1: 获取基本信息
        N_dyn  , N_sta   = N_train - vocab_size, vocab_size
        dyn_slice        = slice(0, N_dyn)
        sta_slice        = slice(N_dyn, N_train)
        assert N_train % N_T == 0 and N_valid % N_T == 0, f"N_train 和 N_valid 必须是 N_T 的整数倍，但现在是 {N_train}, {N_valid}, {N_T}"
        mark(ED, "all_preparation_1", father="all_preparation")
        
        mark(ST, "all_preparation_2")
        ### step 2: 构造分块与采样
        train_splits = make_splits(0      , N_train          , N_T) 
        valid_splits = make_splits(N_train, N_train + N_valid, N_T)
        splits       = train_splits + valid_splits
        sampler      = Expander_Sampler(N_train, N_valid, N_dyn, N_sta, N_T, splits, train_emb, valid_emb, int(int(math.log2(N_train)) ** 2 * sample_factor), main_device, len(self.streams))
        sampler      .generate_graph(connect_to_sta=True)
        self.S       = sampler.S + N_sta
        # sampler      .generate_connections(expected_type="train")
        # sampler      .generate_connections(expected_type="valid")
        mark(ED, "all_preparation_2", father="all_preparation")
        
        
        mark(ST, "all_preparation_3")
        ### step 3: 固定全局不变信息
        voc_emb = train_emb[sta_slice] # (N_sta, dim)
        voc_idx = train_idx[sta_slice] # (N_sta, )
        _loss_cos = torch.empty((N_T, 2 * self.k * self.h + 1, self.tp), dtype=torch.float32, pin_memory=True) # (T, C, D)
        _loss_cro = torch.empty((N_T, 2 * self.k * self.h + 1, self.tp), dtype=torch.float32, pin_memory=True) # (T, C, D)
        _loss_tot = torch.empty((N_T, 2 * self.k * self.h + 1, self.tp), dtype=torch.float32, pin_memory=True) # (T, C, D)
        self.pos_loc = [torch.empty((N_T // len(self.devices), self.S, self.tp), dtype=torch.int64, device=dev) for dev in self.devices]
        self.pos_emb = [
            torch.cat([
                torch.empty((N_T // len(self.devices), self.S - N_sta, n_embd),  dtype=torch.int64, device=dev),
                voc_emb.unsqueeze(0).expand(N_T // len(self.devices), -1, -1).to(dev),
            ], dim=1)
            for dev in self.devices
        ]
        mark(ED, "all_preparation_3", father="all_preparation")
        
        mark(ED, "all_preparation", father="all")
        
        
        mark(ST, "all_epoch")
        ### step 3: 遍历所有 epoch
        for epoch in range(self.epoch_num):
            mark(ST, "epoch")
            mark(ST, "epoch_preparation")
            self.epoch    = epoch
            self.converge = self.loss_strategy['converge'] is not None and (self.epoch >= self.loss_strategy['converge'])
            new_locations = self.main_locations.clone().pin_memory() # (emb_size, tp)
            loss_split_record = {
                "train_cos_loss" :  0.,
                "train_cro_loss" :  0.,
                "train_tot_loss" :  0.,
                "valid_cos_loss" :  0.,
                "valid_cro_loss" :  0.,
                "valid_tot_loss" :  0.,
            }
            
            block2indices = None
            
            
            if epoch == 0:
                sampler.reset_indices("train")
            if epoch == 0:
                sampler.reset_indices("valid")
            

            mark(ED, "epoch_preparation", father="epoch")
            mark(ST, "epoch_train")
            ### step 3.1: 训练
            for block in splits:
                mark(ST, "block")
                mark(ST, "block_preparation")
                
                mark(ST, "block_preparation_3.1.0")
                ### step 3.1.0: 获取基本信息
                bs         = block[1] - block[0]
                assert bs == N_T, "每个 block 的大小必须等于 N_T"
                
                sub_splits = list(map(int, 
                    torch.linspace(0, block[1] - block[0], len(self.devices) + 1, dtype=torch.int64).tolist()            
                ))
                sub_splits = [slice(sub_splits[i], sub_splits[i+1]) for i in range(len(sub_splits) - 1)]
                self.sub_splits = sub_splits
                zip_s_d     = list(zip(sub_splits, self.devices))
                
                cur_type   = get_type (block, N_train, N_valid                      )  # 'train' / 'vocab' / 'valid'
                sta_emb    = get_emb  (block, N_train, N_valid, train_emb, valid_emb, block2indices)  # (N_T, dim)
                sta_idx    = get_idx  (block, N_train, N_valid, train_idx, valid_idx, block2indices)  # (N_T, )
                targets    = get_tar  (block, N_train         , train_tar           , block2indices)  # (N_T, )  
                targets    = F.one_hot(targets, num_classes=N_sta) * 1e2 if targets is not None else None     
                                                                                       # (N_T, N_sta), 可能为 None    
                mark(ED, "block_preparation_3.1.0", father="block_preparation")
                
                mark(ST, "block_preparation_3.1.1")
                ### step 3.1.1: 准备全局连接信息、可见邻居信息
                sta_loc   = self.locations[0][sta_idx]           # (N_T, tp)
                cnc_loc   = self.connection(sta_loc, dev_num=0).to("cpu", 
                                                                non_blocking=True)   # (N_T, C, tp)
                self      .generate_mask(
                    block,
                    N_dyn,
                    N_sta
                )                # 生成 mask
                mark(ED, "block_preparation_3.1.1", father="block_preparation")
                
                mark(ST, "block_preparation_3.1.2")
                ### step 3.1.2: 将数据分配到各个 GPU 上
                self.sta_emb = [sta_emb[s].to(dev, non_blocking=True) for s, dev in zip_s_d]
                self.targets = [targets[s].to(dev, non_blocking=True) for s, dev in zip_s_d] if targets is not None else [None for _ in self.devices]
                self.sta_idx = [sta_idx[s].to(dev, non_blocking=True) for s, dev in zip_s_d]
                self.sta_loc = [sta_loc[s].to(dev, non_blocking=True) for s, dev in zip_s_d]
                self.cnc_loc = [cnc_loc[s].to(dev, non_blocking=True) for s, dev in zip_s_d]
                mark(ED, "block_preparation_3.1.2", father="block_preparation")
                
                mark(ST, "block_preparation_3.1.4")
                ### step 3.1.4: 获取 pos 信息
                sample  = sampler[block]
                pos_idx = train_idx[sample]  # (T, S - N_sta, )   , cuda:0
                pos_emb = train_emb[pos_idx] # (T, S - N_sta, dim), cuda:0
                pos_loc = self.locations[0][pos_idx] 
                                             # (T, S - N_sta, dim), cuda:0
                voc_loc = self.locations[0][voc_idx].expand(N_T, -1, -1) 
                                             # (T, N_sta, dim),     cuda:0
                
                for i, dev in enumerate(self.devices):
                    self.pos_loc[i][:, :self.S - N_sta, :].copy_(pos_loc[sub_splits[i]], non_blocking=True)
                    self.pos_loc[i][:, self.S - N_sta:, :].copy_(voc_loc[sub_splits[i]], non_blocking=True)
                    self.pos_emb[i][:, :self.S - N_sta, :].copy_(pos_emb[sub_splits[i]], non_blocking=True)
                    
                mark(ED, "block_preparation_3.1.4", father="block_preparation")
                
                mark(ED, "block_preparation", father="block")
                
                
                mark(ST, "block_calc")
                ### step 3.1.5: 计算 loss
                self.loom(N_sta, _loss_cos, _loss_cro, _loss_tot)
                selected_locs, real_loss_cos, real_loss_cro, real_loss_tot = self._get_best_loc(
                    cnc_loc,
                    _loss_cos, _loss_cro, _loss_tot, 
                ) # (T), (T), (T), (T)
                mark(ED, "block_calc", father="block")
                
                mark(ST, "block_writeback")
                if instant_writeback:
                    self.main_locations.index_copy_(
                        0,
                        sta_idx.to('cpu', non_blocking=True),
                        selected_locs.view(-1, self.tp)
                    )
                    for i, dev in enumerate(self.devices):
                        self.locations[i].index_copy_(
                            0,
                            sta_idx.to(dev, non_blocking=True),
                            selected_locs.to(dev, non_blocking=True).view(-1, self.tp)
                        )
                loss_split_record[f"{cur_type}_cos_loss"] += float(real_loss_cos.sum().item())
                loss_split_record[f"{cur_type}_cro_loss"] += float(real_loss_cro.sum().item())
                loss_split_record[f"{cur_type}_tot_loss"] += float(real_loss_tot.sum().item())
                mark(ED, "block_writeback", father="block")
                mark(ED, "block")
                
            mark(ED, "epoch_train", father="epoch")
            
            ### step 3.2: 更新
            if not instant_writeback:
                mark(ST, "epoch_update")
                self.main_locations.copy_(new_locations, non_blocking=True)
                for i, dev in enumerate(self.devices):
                    self.locations[i].copy_(self.main_locations.to(dev, non_blocking=True), non_blocking=True)
                mark(ED, "epoch_update", father="epoch")


            ### step 3.3: 记录与打印
            print(f"epoch {epoch:3d} summary:", end=" ")
            for k, v in loss_split_record.items():
                if   k.startswith('train_cos'):
                    v /= N_train
                elif k.startswith('train_cro'):
                    v /= N_dyn
                elif k.startswith('train_tot'):
                    v /= N_train
                elif k.startswith('valid_cos'):
                    v /= N_valid
                else:
                    continue
                
                print(f"{k:15s}: {v:.4f}", end=", ")
            
            
            ### step 3.4: 验证
            mark(ST, "epoch_valid")
            for split in ['train', 'valid']:
                cur_splits  = train_splits         if split == 'train' else valid_splits
                dyn_idx = train_idx[dyn_slice] if split == 'train' else valid_idx
                sta_idx = train_idx[sta_slice] 
                sta_emb = train_emb[sta_slice]
                targets = train_tar[dyn_slice] if split == 'train' else valid_tar
                N       = N_train              if split == 'train' else N_valid
                loss    = 0
                accuray = 0

                for block in cur_splits:
                    Tl, Tr      = block if split == 'train' else (block[0] - N_train, block[1] - N_train)
                    if split == 'train':
                        if Tl >= N_dyn:
                            raise ValueError("忘记改了吧哥们")
                        if Tr > N_dyn:
                            Tr = N_dyn
            
                    dyn_loc     = self.locations[0][dyn_idx[Tl:Tr]] # (T, D)
                    sta_loc     = self.locations[0][sta_idx       ] # (V, D)
                    ct_val      = self.distance(
                        dyn_loc[:, None, :], # (T, 1, D) 
                        sta_loc[None, :, :], # (1, V, D)
                        20. * torch.ones((1, 1, 1), device=self.devices[0], dtype=torch.float32) # (1, 1, 1)
                    ).mean(dim=-1) # (T, V)
                    loss       += F.cross_entropy(ct_val, targets[Tl:Tr].to(self.devices[0], non_blocking=True), reduction='sum').item()
                    accuray    += (ct_val.argmax(dim=-1) == targets[Tl:Tr].to(self.devices[0], non_blocking=True)).sum().item()
                
                loss = loss / N
                accuray = accuray / N
                print(f"{split:5s} loss: {loss:.4f}, accuracy: {accuray:.4f}", end=", ")

            print()
            
            mark(ED, "epoch_valid", father="epoch")
            mark(ED, "epoch")
            
            ### step 3.5: 可视化。仅可视化前 256 个
            # if (epoch) % 10 == 0 or epoch == self.epoch_num - 1:
            #     train_eu_emb = train_emb[:256]                    # (256, dim)
            #     valid_eu_emb = valid_emb[:256]                    # (256, dim)
            #     S_tt_eu      = normalized_matmul(train_eu_emb, train_eu_emb.t())[0].cpu().numpy()
            #     S_vt_eu      = normalized_matmul(valid_eu_emb, train_eu_emb.t())[0].cpu().numpy()
                
            #     train_ct_emb = self.main_locations[train_idx[:256]]  # (256, tp)
            #     valid_ct_emb = self.main_locations[valid_idx[:256]]  # (256, tp)
            #     S_tt_ct      = self.distance(train_ct_emb[:, None, :], train_ct_emb[None, :, :], torch.ones((256, 256, 1))).mean(dim=-1).cpu().numpy()
            #     S_vt_ct      = self.distance(valid_ct_emb[:, None, :], train_ct_emb[None, :, :], torch.ones((256, 256, 1))).mean(dim=-1).cpu().numpy()

            #     visualize_similarity   (S_tt_eu, S_tt_ct, meta_name="{}" + "train_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0), loss_dyn_dyn=loss_split_record["train_cos_loss"] / N_train)
            #     # visualize_similarity   (S_tt_eu, S_tt_ct, meta_name="{}" + f"CT_Sorted_T{N_train}_C{sample_factor}" + ".png", save_eu=(epoch == 0), loss_dyn_dyn=loss_split_record["train_dyn_loss"] / N_train)
            #     visualize_pair_bihclust(S_vt_eu, S_vt_ct, meta_name="{}" + "valid_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))

        mark(ED, "all_epoch", father="all")
        mark(ED, "all")
        
        self.timer.summary()
        
    def reset_block(self):
        self.sta_idx    = [None for _ in range(len(self.devices))]
        self.sta_emb    = [None for _ in range(len(self.devices))]
        self.sta_loc    = [None for _ in range(len(self.devices))]
        self.cnc_loc    = [None for _ in range(len(self.devices))]
        self.targets    = [None for _ in range(len(self.devices))]
        self._valid_mask_cache = {}

    def _get_best_loc(self, 
                      cnc_loc : torch.Tensor, # (T, C, D)
                      loss_cos: torch.Tensor, 
                      loss_cro: torch.Tensor,
                      loss_tot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = 'cpu'
        
        cnc_indices   = torch.argmin(loss_tot, dim=1)                                    # (T, D)
        T_indices     = torch.arange(cnc_loc.size(0),    device=device)[:, None]         # (T, 1)
        dim_indices   = torch.arange(self.tp    ,        device=device)[None  :]         # (1, D)
        selected_locs = cnc_loc [T_indices, cnc_indices,  dim_indices]                   # (T, D)
        
        real_loss_cos = loss_cos[T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, )
        real_loss_cro = loss_cro[T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, )
        real_loss_tot = loss_tot[T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, ) 
        
        return selected_locs, real_loss_cos, real_loss_cro, real_loss_tot

    
if __name__ == "__main__":
    pass

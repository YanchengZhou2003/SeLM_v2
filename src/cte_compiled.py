import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch

torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
from matplotlib.colors import PowerNorm
from torch.nn import functional as F
from tqdm import tqdm

from src.gettime import gettime, mark
from src.loom_kernel import _loom_fused_no_branch  # triton_loom_wrapper
from src.loss import compute_loss, compute_weighted_loss
from src.para import ED, N_T, ST, vocab_size
from src.sample import BaseSample, Expander_Sample
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
        
    def generate_mask(self, block: Tuple[int, int], S: int, S_: int, N_dyn: int):
        """
        我们期望的 mask 是这样的：
        1. 对于每一个 Tl, Tr 的块，我们需要生成 (Tr - Tl, S_) 的 mask，它的生成规则如下：
           (a) 如果 self.converge == False，则以 0.8 的概率，整行在 [0:S) 范围内全为 True
           (b) 如果 self.converge == False，则以 0.2 的概率，整行在 [0:S) 范围内全为 True，但随机选择 self.sample_k 个位置设为 True
           (c) 如果 self.converge == True ，则整行在 [0:S) 范围内全为 True
           (d) 如果 S_ > S，则从 max(N_dyn, Tl) 行开始，(Tr - Tl, S_:S_) 全为 False；其它行为 True
        2. 对于每一个 Tl, Tr 的块，我们需要生成 (Tr - Tl, ) 的 lth，它统计的是每一行在 [0:S) 范围内的 True 数量
        """
        assert S > 0
        self.masks = []
        self.lth = []
        for i, dev in enumerate(self.devices):
            s, e = block[0] + self.sub_splits[i].start, block[0] + self.sub_splits[i].stop
            T1_local = e - s
            with torch.cuda.device(dev), torch.cuda.stream(self.streams[i]):
                mask = torch.ones((T1_local, S_), dtype=torch.bool, device=dev)

                if not self.converge:
                    choosing_mask = (torch.rand((T1_local,), device=dev) > 0.2)  # True=整行在[0:S)全True
                    sel_idx = (~choosing_mask).nonzero(as_tuple=False).squeeze(1)
                    if sel_idx.numel() > 0:
                        mask[sel_idx, :S] = False
                        rows = sel_idx.repeat_interleave(self.sample_k)
                        cols = torch.randint(0, S, (rows.numel(),), device=dev)
                        mask[rows, cols] = True

                if S_ > S:
                    local_start = max(N_dyn, s) - s
                    if local_start < T1_local:
                        mask[local_start:, S:S_] = False

                lth_local = mask[:, :S].sum(dim=1).to(torch.int32) + 1e-12
                self.masks.append(mask)
                self.lth.append(lth_local)

        for stream in self.streams:
            stream.synchronize()


    
    def loom(self, 
         pos_idx  : torch.Tensor, # (T, S_)
         pos_emb  : torch.Tensor, # (T, S_, dim)
         cur_type : str,
         _loss_cos: torch.Tensor, # (T, C, D)
         _loss_cro: torch.Tensor, # (T, C, D)
         _loss_tot: torch.Tensor  # (T, C, D)
    ):
        # --- 基本信息 ---
        T, S_ = pos_idx.size()
        C, D = 2 * self.k * self.h + 1, self.tp

        # 说明：推理阶段只需把 self.masks[i] 中不参与的列置 0；不需要 S_ 与任何拼接
        # mask/lth 由外部预先准备好，形状：
        #   masks[i] : (subT, S_)
        #   lth[i]   : (subT,)

        for i, (dev, stream, split) in enumerate(zip(self.devices, self.streams, self.sub_splits)):
            Tl, Tr = split.start, split.stop
            subT = Tr - Tl
            if subT <= 0:
                continue

            with torch.cuda.device(dev), torch.cuda.stream(stream):  # 仅调度/搬运
                # --- 切片搬运 ---
                sub_pos_idx, sub_pos_emb = to_dev(
                    pos_idx, pos_emb,
                    device=dev, s=Tl, e=Tr, dim=0
                )                                         # (subT, S), (subT, S, dim)

                # --- 位置张量 ---
                pos_loc = self.locations[i][sub_pos_idx]  # (subT, S, D)
                sta_emb = self.sta_emb[i]                 # (subT, dim)
                sta_loc = self.sta_loc[i]                 # (subT, D)
                cnc_loc = self.cnc_loc[i]                 # (subT, C, D)
                targets = self.targets[i] if self.targets[i] is not None else torch.ones((subT, vocab_size), dtype=torch.float32, device=dev) # (subT, N_sta)

                # --- mask/lth（推理阶段自行在外层把不参与列置 0） ---
                mask = self.masks[i]                      # (subT, S), bool
                lth  = self.lth[i]                        # (subT,)

                # --- 编译核：计算两路 loss ---
                # print(sub_pos_emb.shape, sta_emb.shape, sta_loc.shape, pos_loc.shape, cnc_loc.shape, targets.shape, mask.shape, lth.shape, S_ - vocab_size, S_)
                loss_cos, loss_cro = _loom_fused_no_branch(
                    sub_pos_emb, sta_emb, sta_loc, pos_loc, cnc_loc, targets,
                    mask, lth,
                    S_ - vocab_size, S_,
                    h=self.h, tp=self.tp, sum_dim=1
                )
                # --- 线性组合 ---
                loss_tot = (self.loss_strategy['ratio_cos'] * loss_cos +
                            self.loss_strategy['ratio_cro'] * loss_cro)   # (subT, C, D)

                # --- 回写（非阻塞）---
                _loss_cos[Tl:Tr].copy_(loss_cos, non_blocking=True)
                _loss_cro[Tl:Tr].copy_(loss_cro, non_blocking=True)
                _loss_tot[Tl:Tr].copy_(loss_tot, non_blocking=True)

        for stream in self.streams:
            stream.synchronize()
        


    @gettime(fmt='ms', pr=True)
    def forward(self,        
        train_emb : torch.Tensor, # (N_train, dim), pinned memory
        valid_emb : torch.Tensor, # (N_valid, dim), pinned memory
        
        train_idx : torch.Tensor, # (N_train, )   , pinned memory
        valid_idx : torch.Tensor, # (N_valid, )   , pinned memory

        train_tar : torch.Tensor, # (N_train, )   , pinned memory
        valid_tar : torch.Tensor, # (N_valid, )   , pinned memory
        
        sample_factor: float = 1.,
        ### 以上张量全部放在 cpu，避免占用 gpu 内存
    ):  
        """
        我们需要拟合的是一个：(N_train, N_train) 的矩阵，以及 (N_valid, N_train) 的矩阵
        并且构造 (N_dyn, N_sta) 的 groundtruth，向它对齐
        """
        mark(ST, "all")
        
        mark(ST, "all_preparation")
        
        mark(ST, "all_preparation_1")
        ### step 1: 获取基本信息
        N_train, N_valid = train_emb.size(0), valid_emb.size(0)
        N_dyn  , N_sta   = N_train - vocab_size, vocab_size
        dyn_slice        = slice(0, N_dyn)
        sta_slice        = slice(N_dyn, N_train)
        assert N_train % N_T == 0 and N_valid % N_T == 0, "N_train 和 N_valid 必须是 N_T 的整数倍"
        mark(ED, "all_preparation_1", father="all_preparation")
        
        mark(ST, "all_preparation_2")
        ### step 2: 构造分块与采样
        train_splits = make_splits(0      , N_train          , N_T) 
        valid_splits = make_splits(N_train, N_train + N_valid, N_T)
        splits       = train_splits + valid_splits
        sampler      = Expander_Sample(N_train, N_valid, 
                                       splits,
                                       train_idx, valid_idx,
                                       int(int(math.log2(N_train)) ** 2 * sample_factor)
                       )
        sampler      .generate_graph(connect_to_sta=True, N_dyn=N_dyn)
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
            
            if epoch % 20 == 0:
                sampler.reset_indices("train")
            if epoch % 5  == 0:
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
                
                cur_type   = get_type (block, N_train, N_valid)  # 'train' / 'vocab' / 'valid'
                sta_emb    = get_emb  (block, N_train, N_valid, train_emb, valid_emb)  # (N_T, dim)
                sta_idx    = get_idx  (block, N_train, N_valid, train_idx, valid_idx)  # (N_T, )
                targets    = get_tar  (block, N_train         , train_tar,          )  # (N_T, )  
                targets    = F.one_hot(targets, num_classes=N_sta) * 1e2 if targets is not None else None     
                                                                                       # (N_T, N_sta), 可能为 None    
                mark(ED, "block_preparation_3.1.0", father="block_preparation")
                
                mark(ST, "block_preparation_3.1.1")
                ### step 3.1.1: 准备全局连接信息、可见邻居信息
                sta_loc   = self.locations[0][sta_idx.to(self.devices[0])]           # (N_T, tp)
                cnc_loc   = self.connection(sta_loc, dev_num=0).to("cpu", 
                                                                non_blocking=True)   # (N_T, C, tp)
                self      .generate_mask(
                    block,
                    sampler.S,
                    sampler.S + N_sta,
                    N_dyn
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
                pos_idx = train_idx[sample] # (T, S, )
                pos_emb = train_emb[sample] # (T, S, dim)
                pos_idx = torch.cat([pos_idx, voc_idx.unsqueeze(0).expand(N_T, -1)], dim=1)     # (T, S + N_sta)
                pos_emb = torch.cat([pos_emb, voc_emb.unsqueeze(0).expand(N_T, -1, -1)], dim=1) # (T, S + N_sta, dim)
                S_      = pos_idx.size(1)
                
                mark(ED, "block_preparation_3.1.4", father="block_preparation")
                
                mark(ED, "block_preparation", father="block")
                
                
                mark(ST, "block_calc")
                ### step 3.1.5: 计算 loss
                self.loom(pos_idx, pos_emb, cur_type, _loss_cos, _loss_cro, _loss_tot)
                selected_locs, real_loss_cos, real_loss_cro, real_loss_tot = self._get_best_loc(
                    cnc_loc,
                    _loss_cos, _loss_cro, _loss_tot, 

                ) # (T), (T), (T), (T)
                new_locations.index_copy_(
                    0,
                    sta_idx,
                    selected_locs.view(-1, self.tp)
                )
                loss_split_record[f"{cur_type}_cos_loss"] += float(real_loss_cos.sum().item())
                loss_split_record[f"{cur_type}_cro_loss"] += float(real_loss_cro.sum().item())
                loss_split_record[f"{cur_type}_tot_loss"] += float(real_loss_tot.sum().item())
                mark(ED, "block_calc", father="block")
                mark(ED, "block")
                
            mark(ED, "epoch_train", father="epoch")
            
            ### step 3.2: 更新
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
                dyn_emb = train_emb[dyn_slice] if split == 'train' else valid_emb
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
            
                    dyn_loc     = self.locations[0][dyn_idx[Tl:Tr].to(self.devices[0], non_blocking=True)] # (T, D)
                    sta_loc     = self.locations[0][sta_idx       .to(self.devices[0], non_blocking=True)] # (V, D)
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
            if epoch % 10 == 0 or epoch == self.epoch_num - 1:
                train_eu_emb = train_emb[:256]                    # (256, dim)
                valid_eu_emb = valid_emb[:256]                    # (256, dim)
                S_tt_eu      = normalized_matmul(train_eu_emb, train_eu_emb.t())[0].cpu().numpy()
                S_vt_eu      = normalized_matmul(valid_eu_emb, train_eu_emb.t())[0].cpu().numpy()
                
                train_ct_emb = self.main_locations[train_idx[:256]]  # (256, tp)
                valid_ct_emb = self.main_locations[valid_idx[:256]]  # (256, tp)
                S_tt_ct      = self.distance(train_ct_emb[:, None, :], train_ct_emb[None, :, :], torch.ones((256, 256, 1))).mean(dim=-1).cpu().numpy()
                S_vt_ct      = self.distance(valid_ct_emb[:, None, :], train_ct_emb[None, :, :], torch.ones((256, 256, 1))).mean(dim=-1).cpu().numpy()

                visualize_similarity   (S_tt_eu, S_tt_ct, meta_name="{}" + "train_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0), loss_dyn_dyn=loss_split_record["train_cos_loss"] / N_train)
                # visualize_similarity   (S_tt_eu, S_tt_ct, meta_name="{}" + f"CT_Sorted_T{N_train}_C{sample_factor}" + ".png", save_eu=(epoch == 0), loss_dyn_dyn=loss_split_record["train_dyn_loss"] / N_train)
                visualize_pair_bihclust(S_vt_eu, S_vt_ct, meta_name="{}" + "valid_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))

        mark(ED, "all_epoch", father="all")
        mark(ED, "all")
        
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

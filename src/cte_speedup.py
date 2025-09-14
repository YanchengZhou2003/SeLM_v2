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
from src.loss import compute_loss, compute_weighted_loss
from src.para import (ED, ST, N_vocab, cur_portion, cur_tp,
                      generators, instant_writeback, n_embd, use_eu_norm,
                      T_train, N_trnbr, T_trnbr,
                      T_vonbr,
                      T_valid, N_vanbr, T_vanbr,)
from src.sampler import BaseSample, Expander_Sampler, Prefetcher
from src.utils import *
from src.vis import visualize_pair_bihclust, visualize_similarity

main_device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
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
        self.main_device = main_device
        self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.streams = [torch.cuda.default_stream(i) for i in range(torch.cuda.device_count())]
        self.num_devices = len(self.devices)        
    
        ### 2. 基本参数
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = f
        self.k = int(f*h / division_fact)
        self.loss_strategy = loss_strategy
        self.emb_size = emb_size
        self.epoch_num = epoch_num
        
        ### 3. CT Space Embeddings 初始化
        self.main_locations = torch.randint(
            1 - self.n, self.n, (self.emb_size, self.tp), 
            dtype=torch.int64, device=main_device, generator=generators[0]
        )
        self.locations = [self.main_locations.clone().to(dev) for dev in self.devices]

        ### 4. 训练时参数
        self.epoch = -1
        self.sample_k = sample_k
        self._pending_refs = {sid: [] for sid in range(self.num_devices)}
        self._valid_mask_cache = {}
        
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
    
    def connection(self, ori_int, dev_num=0):
        device = self.devices[dev_num] if dev_num >= 0 else 'cpu'
        
        flip_masks = (1 << torch.arange(self.h, device=device, dtype=ori_int.dtype)).unsqueeze(0).unsqueeze(2)
        flipped_ints = ori_int.unsqueeze(1) ^ flip_masks # (B*T1, H, D)
        random_masks = self.generate_random_masks(flipped_ints.size(0), dev_num=dev_num)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k, self.tp)
        # (B*T1, H, 1, D) ^ (B*T1, H, K, D) -> (B*T1, H*K, D)
        loc = torch.cat((result, ori_int.unsqueeze(1), -result), dim=1) # (B*T1, H*K + 1 + H*K, D)
        # indices = torch.randperm(loc.size(1), device=device) 
        # loc = loc[:, indices, :]
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
        
    def generate_mask(self, block: Tuple[int, int], cur_type: str, dev_num=0):
        """
        我们期望的 mask 是这样的：
        1. 我们需要生成 (T, S) 的 mask，它的生成规则如下：
        (a) 如果是 train:
            (1) 如果 self.converge 为 False，则随机；否则首先全 True
            (2) 如果 N_dyn < Tr, 则 [max(N_dyn - Tl, 0):] 的行中，[self.N_sta:] 全为 False
        (b) 如果是 valid:
            (1) 如果 self.converge 为 False，则随机；否则首先全 True
        """
        device  = self.devices[dev_num]
        tot_lth = self.S_tot if cur_type == "train" else self.T_val
        mask   = torch.ones((self.T_train, tot_lth), dtype=torch.bool, device=device)
        Tl, Tr = block
        
        if not self.converge:
            choosing_mask = (torch.rand((self.T_train,), device=device) > 0.2)  
            sel_idx = (~choosing_mask).nonzero(as_tuple=False).squeeze(1)
            if sel_idx.numel() > 0:
                cutoff = self.S_cos if cur_type == "train" else self.T_val
                mask[sel_idx, :cutoff] = False
                rows = sel_idx.repeat_interleave(self.sample_k)
                cols = torch.randint(0, cutoff, (rows.numel(),), device=device, generator=generators[dev_num])
                mask[rows, cols] = True
        if cur_type == "train":
            if Tr > self.N_dyn:
                local_start = max(0, self.N_dyn - Tl)
                mask[local_start:, self.N_sta] = False
            lth = mask[:, :self.S_cos].sum(dim=1) + 1e-12
        elif cur_type == "valid":
            lth = mask[:, :          ].sum(dim=1) + 1e-12

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
        
        # step1: 每个样本在 [0, tp) 之间随机生成一个排列（通过排序 trick）
        rand_vals     = torch.rand((self.T_train, self.tp), device=device, generator=generators[sid]) # (T, D)
        rand_cols     = rand_vals.argsort(dim=1)[:, :cur_tp]           # (T, cur_tp)
        # print(rand_cols[:4, :2])
        
        # step2: 每个 (t, dim) 的 argmin
        argmin_all    = torch.argmin(loss_tot, dim=1)                  # (T, D)

        # step3: 初始化为 "保持不更新"
        cnc_indices   = torch.full_like(argmin_all, self.k * self.h)   # (T, D)

        # step4: 高级索引直接填充
        # T_indices     = torch.arange(self.N_T,    device=device)[:, None]                # (T, 1)
        t_idx = torch.arange(self.T_train, device=loss_tot.device)[:, None].expand(self.T_train, cur_tp)  # (T, upd)
        t_mask = torch.rand(self.T_train, device=device, generator=generators[sid]) < cur_portion  # (T,)
        t_idx, rand_cols = t_idx[t_mask], rand_cols[t_mask]  # (T_upd, ), (T_upd, upd)
        
        cnc_indices[t_idx, rand_cols] = argmin_all[t_idx, rand_cols]
        # print(loss_tot.shape[1], cnc_indices.max().item(), cnc_indices.min().item())
        
        # cnc_indices   = torch.argmin(loss_tot, dim=1)                                  # (T, D)
        T_indices     = torch.arange(cnc_loc.size(0),    device=device)[:, None]       # (T, 1)
        dim_indices   = torch.arange(self.tp    ,        device=device)[None  :]         # (1, D)
        selected_locs = cnc_loc [T_indices, cnc_indices,  dim_indices]                   # (T, D)
        
        real_loss_cos = loss_cos[T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, )
        real_loss_cro = loss_cro[T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, )
        real_loss_tot = loss_tot[T_indices, cnc_indices, dim_indices].mean(dim=-1)       # (T, ) 
        
        return selected_locs, real_loss_cos, real_loss_cro, real_loss_tot

    # @torch.compile()
    def distance(self, coord1: torch.Tensor, coord2: torch.Tensor, norm: torch.Tensor):
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        xor_result = torch.abs(coord1) ^ torch.abs(coord2)
        _, exp = torch.frexp((xor_result + 1).to(torch.float32))
        s = exp.float() / self.h
        return sg * (1 - s) * norm
    
    # @torch.compile()
    # @torch.compile(dynamic=True)
    def loom(
        self, 
        sta_loc: torch.Tensor,   # (T, D)
        cnc_loc: torch.Tensor,   # (T, C, D)
        pos_loc: torch.Tensor,   # (T, S_tot, D)
        eu_val : torch.Tensor,   # (T, S_tot)
        targets: Optional[torch.Tensor],   # (T, S_tot - S_cos)
        mask   : torch.Tensor,   # (T, S_tot)
        lth    : torch.Tensor,   # (T)
        sid    : int = 0,
    ):
        device = self.devices[sid]
        C, D = 2 * self.k * self.h + 1, self.tp
        S_cos, S_tot = self.S_cos, self.S_tot
        
        
                
        # self.timer.mark(f"dev{sid}_EU_Calc", 0)
        eu_val  [:, S_cos:]   = targets if targets is not None else 0
                                                 # (T, S_cos:), 的部分放为 groundtruth, (T, :S_cos) 的部分维持
        eu_norm               = self.eu_norm[sid]                    
                                                 # (T, S_pot)


        # self.timer.mark(f"dev{sid}_EU_Calc", 1)
               
                
                
        # self.timer.mark(f"dev{sid}_CT_Preparation", 0)
        pos_loc         = torch.cat([pos_loc, self.voc_loc[sid]], dim=1)  # (T, S_tot, D)
        cos_sta_pos     = self.distance(
            sta_loc[:, None, :]   , pos_loc[:, :, :]     , eu_norm[..., None]      
        )                                        # (T, S_tot, D)
        cos_sta_pos_sum = cos_sta_pos.sum(dim=-1) 
                                                 # (T, S_tot)
        # self.timer.mark(f"dev{sid}_CT_Preparation", 1)
                
                
                
        # self.timer.mark(f"dev{sid}_COS_Calc", 0)
        
        
        ct_val = ct_val_triton(
            cnc_loc.to(torch.int32).contiguous(),
            pos_loc.to(torch.int32).contiguous(),
            eu_norm.contiguous(),
            cos_sta_pos.contiguous(),
            cos_sta_pos_sum.contiguous(),
            tp=float(self.tp),
            h=float(self.h),
            out=None,                # 或传入你复用的 out 缓冲
            BLOCK_S=32,
            BLOCK_CD=32,
            NUM_WARPS=8,
            NUM_STAGES=2,
        )
        
        # cos_cnc_pos     = self.distance(
        #     cnc_loc[:, None, :, :], pos_loc[:, :, None,:], eu_norm[..., None, None]
        # )                                        # (T, S_tot, C, D)
        # ct_val          = (
        #     cos_sta_pos_sum[:, :, None, None] - cos_sta_pos[:, :, None, :] + cos_cnc_pos
        # ) / self.tp                              # (T, S_tot, C, D)
        
        
        ## 对于 T 个 starting point，向 S_tot 个 positive sample 连边。此时，我们把其中某个 positive sample 替换为 connected sample，共有 C 个；此时，D 个维度上的的距离是多少？
        # self.timer.mark(f"dev{sid}_COS_Calc", 1)



        # self.timer.mark(f"dev{sid}_LOSS_Calc", 0)
        ct_val    = ct_val                                         # (T, S_tot, C, D)
        eu_val    = eu_val[..., None, None].expand(ct_val.shape)   # (T, S, C, D)

        loss_cos, loss_cro, loss_tot = self.calc_loss(
            ct_val, eu_val, 
            mask[..., None, None], lth,
            S_cos, S_tot
        )                                                          # (T, C, tp)      
        # self.timer.mark(f"dev{sid}_LOSS_Calc", 1)     
        # print(f"loss_cos: {loss_cos.mean().item():.10f}, loss_cro: {loss_cro.mean().item():.10f}, loss_tot: {loss_tot.mean().item():.10f}")
        selected_locs, loss_cos, loss_cro, loss_tot = self.get_best_loc(
            cnc_loc, loss_cos, loss_cro, loss_tot, sid
        ) # (T, D), (T), (T), (T)
        
        return selected_locs, loss_cos, loss_cro, loss_tot
                

                

        

    def train_epoch(self, sid: int):
        try:
            stream = self.streams[sid]
            device = self.devices[sid]
            for epoch in range(self.epoch_num):
                # 等待主线程的信号
                # print(f"[sid={sid}] epoch {epoch} reached barrier 1")
                self.epoch_barrier.wait()
                # print(f"[sid={sid}] epoch {epoch} passed barrier 1")
                
                while True:
                    block_id, block, sta_loc, pos_loc, eu_val, ready_evt = self.prefetcher.get_sample(sid)
                    if block is None:
                        break
                    # time.sleep(0.2)
                    with torch.cuda.device(device), torch.cuda.stream(stream):
                        stream.wait_event(ready_evt)  # 等待 copy 流完成
                        ### step 3.1.-1 处理引用持有
                        new_pending = []
                        for (refs, evt) in self._pending_refs[sid]:
                            if not evt.query():   # 如果事件还没完成
                                new_pending.append((refs, evt))
                        self._pending_refs[sid] = new_pending
                        # self.timer.mark(f"dev{sid}_ALL", 0)
                        # self.timer.mark(f"dev{sid}_Basic_Preparation", 0)
                        ### step 3.1.0: 准备基本信息
                        cur_type   = get_type(block, self.N_train, self.N_valid)
                        targets    = self.tar_splits[block_id] if cur_type == "train" else None    # (T, )                                                   
                        # self.timer.mark(f"dev{sid}_Basic_Preparation", 1)
                    
                    
                        # self.timer.mark(f"dev{sid}_Ad_Preparation", 0)
                        ### step 3.1.1: 准备全局连接信息、可见邻居信息
                        cnc_loc   = self.connection(
                            sta_loc, dev_num=sid
                        )                                                            # (T, C, D)
                        # print(cnc_loc[0, :5, 0])
                        mask, lth = self.generate_mask(block, self.N_dyn, sid)       # (T, S_tot), (T)
                        targets   = F.one_hot(targets, num_classes=self.N_sta) if targets is not None else None        # (T, )
                        # self.timer.mark(f"dev{sid}_Ad_Preparation", 1)
                    
                    
                        ### step 3.1.4: 获取 pos 信息
                        # self.timer.mark(f"dev{sid}_LOOM", 0)
                        cur_type   = "train" if block[0] < self.N_train else "valid"
                        selected_locs, loss_cos, loss_cro, loss_tot = self.loom(
                            sta_loc, cnc_loc, pos_loc, 
                            eu_val, targets,
                            mask, lth, sid
                        )
                        # self.timer.mark(f"dev{sid}_LOOM", 1)
                        
                        
                        # self.timer.mark(f"dev{sid}_WRITE", 0)
                        self.main_locations[block[0]:block[1]].copy_(selected_locs, non_blocking=True)
                        self.loss_cos_buf[block[0]:block[1]].copy_(loss_cos, non_blocking=True) # (T_sta, C, pos) 
                        self.loss_cro_buf[block[0]:block[1]].copy_(loss_cro, non_blocking=True)
                        self.loss_tot_buf[block[0]:block[1]].copy_(loss_tot, non_blocking=True)
                        
                        # 在当前 stream 上打一个事件
                        evt = torch.cuda.Event(enable_timing=False, blocking=False)
                        evt.record(stream)

                        # 保存引用（必须把三个张量都放进 tuple，不然会被回收）
                        self._pending_refs[sid].append(((selected_locs, loss_cos, loss_cro, loss_tot), evt))
    
                
                # print(f"[sid={sid}] epoch {epoch} reached barrier 2")
                self.epoch_barrier.wait()
                # print(f"[sid={sid}] epoch {epoch} passed barrier 2")
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
        
        ### step 1: 获取基本信息，构造分块与采样
        mark(ST, "all_preparation_1")
        ### step 1.1: 基本信息1，starting points 的大小以及分块大小
        self.N_train, self.N_vocab, self.N_valid = train_emb.size(0), vocab_emb.size(0), valid_emb.size(0)
        self.T_train, self.T_vocab, self.T_valid = T_train          , vocab_emb.size(0), T_valid
        
        ### step 1.2: 基本信息2：positive samples（可见邻居）的大小以及分块大小
        self.N_trnbr = int(int(math.log2(self.N_train)) ** 2 * train_sample_factor)
        self.N_vonbr = self.N_train
        self.N_vanbr = N_vanbr # number of valid neighbors
        self.T_trnbr = self.N_trnbr # 暂时不分块
        self.T_vonbr = T_vonbr 
        self.T_vanbr = T_vanbr
        
        ### step 1.3: 基本信息3：构建切片与分块
        train_splits = make_splits(0                          , self.N_train               , self.T_train) 
        vocab_splits = make_splits(self.N_train               , self.N_train + self.N_vocab, self.T_vocab)
        valid_splits = make_splits(self.N_train + self.N_vocab, self.N_train + self.N_valid, self.T_valid)
        
        train_slice  = slice(0, self.N_train)
        vocab_slice  = slice(self.N_train, self.N_train + self.N_vocab)
        valid_slice  = slice(self.N_train + self.N_vocab, self.N_train + self.N_valid)
        
        trnbr_splits = make_splits(0, self.N_trnbr, self.T_trnbr)
        vonbr_splits = make_splits(0, self.N_vonbr, self.T_vonbr)
        vanbr_splits = make_splits(0, self.N_vanbr, self.T_vanbr)
        
        self.splits  = train_splits + vocab_splits + valid_splits
        mark(ED, "all_preparation_1", father="all_preparation")
        
        
        ### step.2 准备数据分块与采样器
        mark(ST, "all_preparation_2")
        self.train_tar  = [train_tar.to(dev) for dev in self.devices]
        
        self.sampler    = Expander_Sampler(
            self.N_train, self.N_valid, self.N_dyn, self.N_sta, self.T_train,
            self.splits,
            train_emb, valid_emb,
            int(int(math.log2(self.N_train)) ** 2 * train_sample_factor), 
            self.N_vnbrs, self.T_val,
            main_device,
            len(self.streams),
            use_eu_norm=use_eu_norm
        )
        self.sampler.generate_graph()
        self.sampler.generate_connections("train")
        self.sampler.generate_connections("valid")
        self.sampler.update_locations(self.main_locations)
        self.prefetcher = Prefetcher(main_device, self.devices, self.streams, self.sampler, 4)
        self.S_tot = self.sampler.S_dyn + self.N_sta
        self.S_cos = self.sampler.S_dyn

        # self.loc_splits = [
        #     self.main_locations[block[0]:block[1]].to(self.devices[i % self.num_devices])  
        #     for i, block in enumerate(self.splits)
        # ]
        # self.loss_cos_buf = [None for _ in range(len(self.splits))] # torch.zeros(self.emb_size, device="cpu", pin_memory=True)
        # self.loss_cro_buf = [None for _ in range(len(self.splits))] # torch.zeros(self.emb_size, device="cpu", pin_memory=True)
        # self.loss_tot_buf = [None for _ in range(len(self.splits))] # torch.zeros(self.emb_size, device="cpu", pin_memory=True)
        self.loss_cos_buf = torch.zeros(self.emb_size, device=main_device)
        self.loss_cro_buf = torch.zeros(self.emb_size, device=main_device)
        self.loss_tot_buf = torch.zeros(self.emb_size, device=main_device)
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
        self.prefetcher.start()
        mark(ED, "all_preparation_3", father="all_preparation")
        mark(ED, "all_preparation", father="all")
        
        
        mark(ST, "all_epoch")
        ### step 3: 遍历所有 epoch
        
        for epoch in range(self.epoch_num):
            mark(ST, "epoch")
            mark(ST, "epoch_preparation")
            self.epoch    = epoch
            self.converge = self.loss_strategy['converge'] is not None and (self.epoch >= self.loss_strategy['converge'])
            loss_split_record = {
                "train_cos_loss" :  0.,
                "train_cro_loss" :  0.,
                "train_tot_loss" :  0.,
            }
            
            if epoch % 50 == 0 and epoch != 0:
                self.sampler.reset_indices("train")
            

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
            
            print(f"epoch {epoch:3d} summary:", end=" ")
            for k, v in loss_split_record.items():    
                if k.startswith("train_cos"): 
                    print(f"{k:15s}: {v:.6f}", end=", ")
                else:
                    print(f"{k:15s}: {v:.4f}", end=", ")
            ### step 3.3: 验证
            mark(ST, "epoch_valid")
            self.validate(epoch)
            mark(ED, "epoch_valid", father="epoch")
            mark(ED, "epoch")
            
            ### step 3.4: 可视化
            self.visualize(epoch)

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
            ct_val      = self.distance(
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
            S_tt_ct      = self.distance(train_ct_emb[:, None, :], train_ct_emb[None, :, :], torch.ones((256, 256, 1), device=main_device)).mean(dim=-1).cpu().numpy()
            S_vt_ct      = self.distance(valid_ct_emb[:, None, :], train_ct_emb[None, :, :], torch.ones((256, 256, 1), device=main_device)).mean(dim=-1).cpu().numpy()

            visualize_similarity   (S_tt_eu, S_tt_ct, meta_name="{}" + "train_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))
            visualize_pair_bihclust(S_vt_eu, S_vt_ct, meta_name="{}" + "valid_train_{}_" + f"epoch_{epoch:04d}" + ".png", save_eu=(epoch == 0))

    def _synchronize_all_streams(self):
        torch.cuda.synchronize()
        torch.cuda.default_stream(main_device).synchronize()
        for sid in range(self.num_devices):
            self.streams[sid].synchronize()
            self.prefetcher.copy_streams[sid].synchronize()
        self.prefetcher.prod_stream.synchronize()
        
if __name__ == "__main__":
    pass

import math
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch

from src.utils import get_idx, get_type


class BaseSample(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_connection(self, sta_id: torch.Tensor, dev_num=0) -> torch.Tensor:
        pass
    
# class CTE_Sample(BaseSample):
#     def __init__(self, h, f, division_fact, N):
#         super().__init__()  
#         assert N == 2 ** h, "N must be 2**h"
        
#         self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
#         self.h = h
#         self.n = int(2**h)
#         self.f = f
#         self.k = int(f*h // division_fact)
#         self.c = self.h * self.k + 1
        
#         self.loc2id = torch.randperm(self.n, dtype=torch.int64) # (N,), 每个位置上对应的是哪个真正节点的 id
#         self.id2loc = torch.argsort(self.loc2id) # (N,), 每个节点的 id 对应的位置
        
#         self._cnc_id_cache = {}
    
#     def _generate_random_masks(self, sz):
#         device = 'cpu'

#         upper_bounds   = 2 ** torch.arange(self.h, dtype=torch.int64, device=device)
#         random_numbers = torch.randint(0, self.n, (self.h, sz, self.k), dtype=torch.int64, device=device) # (H, n, K)
#         masks = random_numbers & (upper_bounds.view(-1, 1, 1) - 1)
#         return masks.permute(1, 0, 2) # (n, H, K)

#     def _connection(self, sta_loc):
#         # sta_loc: (n, )
#         device = 'cpu'
        
#         flip_masks = (1 << torch.arange(self.h, device=device, dtype=sta_loc.dtype)).unsqueeze(0) # (1, H)
#         flipped_ints = sta_loc.unsqueeze(1) ^ flip_masks # (n, H)
#         random_masks = self._generate_random_masks(flipped_ints.size(0))
#         result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k)
#         # (n, H, 1) ^ (n, H, K) -> (n, H*K)
#         loc = torch.cat((result, sta_loc.unsqueeze(1)), dim=1) # (n, H*K + 1)
#         return loc

#     def get_connection(self, sta_id: torch.Tensor):
#         # sta: (n, ), 表示节点 id
#         sta_loc = self.id2loc[sta_id]           # (n, ) , 表示节点位置
#         cnc_loc = self._connection(sta_loc)     # (n, c), 表示每个节点位置对应的连接节点位置
#         cnc_id  = self.loc2id[cnc_loc]          # (n, c), 表示每个节点 id 对应的连接节点 id
#         return cnc_id
    
#     def get_size(self) -> Tuple[int, int]:
#         return (0, self.c)
    
#     def reset_all(self):
#         self.loc2id = torch.randperm(self.n, dtype=torch.int64)
#         self.id2loc = torch.argsort(self.loc2id)
#         self._cnc_id_cache = {}
    
#     def reset_cnc(self):
#         self._cnc_id_cache = {}
    
#     def __setitem__(self, key, value):
#         self._cnc_id_cache[key] = value

#     def __getitem__(self, key):
#         return self._cnc_id_cache[key]


# class CTE_Sort_Sample(BaseSample):
#     def __init__(self, c: int):
#         super().__init__()  
#         self.c = c
#         self.connections = None
#         self._cnc_cache = {}
#         pass

#     def generate_connection(self, locations: torch.Tensor):
#         # locations: (n, c)
#         mean_locations = locations.float().mean(dim=-1)  # (n,)
#         sorted_values, sorted_indices = torch.sort(mean_locations)

#         # ---- 绘制直方图并保存 ----
#         # plt.figure()
#         # plt.hist(sorted_values.cpu().numpy(), bins=50)  # 直方图，50 个桶
#         # plt.xlabel("Mean location value (sorted)")
#         # plt.ylabel("Frequency")
#         # plt.title("Histogram of sorted_values")
#         # plt.tight_layout()
#         # plt.savefig("sorted_values_hist.png")
#         # plt.close()


#         n, c = locations.size(0), self.c
#         num_neighbors = int(0.8 * c)
#         num_random = c - num_neighbors

#         # ---- 向量化邻居 ----
#         idx = torch.arange(n, device=locations.device)  # (n,)
#         L = num_neighbors // 2
#         R = num_neighbors - L

#         # 初始窗口 [start, start+num_neighbors)
#         start = idx[:, None] - L
#         start = start.clamp(min=0)               # 左边不够 -> 往右移
#         start = torch.minimum(start, (n - num_neighbors) * torch.ones_like(start))  
#         # 右边不够 -> 往左移

#         neighbor_indices = start + torch.arange(num_neighbors, device=locations.device)
#         # (n, num_neighbors)，每行一个窗口
#         neighbor_indices = sorted_indices[neighbor_indices]

#         # ---- 随机部分 ----
#         random_indices = torch.randint(0, n, (n, num_random), device=locations.device)
#         random_indices = sorted_indices[random_indices]

#         # ---- 拼接 ----
#         connections = torch.cat((neighbor_indices, random_indices), dim=1)
#         self.connections = connections

#     def get_connection(self, sta_id: torch.Tensor):
#         return self.connections[sta_id]

#     def get_size(self) -> Tuple[int, int]:
#         return (0, self.c)
    
#     def __setitem__(self, key, value):
#         self._cnc_cache[key] = value
    
#     def __getitem__(self, key):
#         return self._cnc_cache[key]


import igraph as ig
import networkx as nx


class Expander_Sampler(BaseSample):
    def __init__(
        self, 
        N_train: int, N_vocab: int, N_valid: int,
        T_train: int, T_vocab: int, T_valid: int,
        N_trnbr: int, N_vonbr: int, N_vanbr: int,
        T_trnbr: int, T_vonbr: int, T_vanbr: int,
        train_slice : slice, vocab_slice: slice, valid_slice: slice,
        train_splits: List[Tuple[int, int]], vocab_splits: List[Tuple[int, int]], valid_splits: List[Tuple[int, int]], 
        train_emb: torch.Tensor, vocab_emb: torch.Tensor, valid_emb: torch.Tensor,
        main_device: torch.device,
        num_streams: int,
        use_eu_norm: bool = False
    ):
        """
        基于 d-正则随机图生成 Expander Graph.
        
        """
        super().__init__()
        ### step.1 读取基本信息 ###
        self.N_train, self.N_vocab, self.N_valid = N_train, N_vocab, N_valid
        self.T_train, self.T_vocab, self.T_valid = T_train, T_vocab, T_valid
        self.N_trnbr, self.N_vonbr, self.N_vanbr = N_trnbr, N_vonbr, N_vanbr
        self.T_trnbr, self.T_vonbr, self.T_vanbr = T_trnbr, T_vonbr, T_vanbr
        self.train_slice, self.vocab_slice, self.valid_slice = train_slice, vocab_slice, valid_slice
        self.train_splits, self.vocab_splits, self.valid_splits = train_splits, vocab_splits, valid_splits
        self.train_emb, self.vocab_emb, self.valid_emb = train_emb.to(main_device), vocab_emb.to(main_device), valid_emb.to(main_device)
        
        self.train_blk4stream = [list(range(sid, len(self.train_splits), num_streams)) for sid in range(num_streams)] 
        self.train_ptr        = [0 for _ in range(num_streams)]
        self.valid_blk4stream = [list(range(sid, len(self.valid_splits), num_streams)) for sid in range(num_streams)]
        self.valid_ptr        = [0 for _ in range(num_streams)]
        
        self.emb_val      = torch.cat([train_emb.to(main_device), valid_emb.to(main_device)], dim=0)
        self.emb_nrm      = self.emb_val.norm(p=2, dim=1, keepdim=True) if use_eu_norm else torch.full((self.emb_val.size(0), 1), 20.) # (N_train + N_valid, 1) 
        
        self.S_dyn = S_dyn
        self.S_val = S_val
        self.T_val = T_val
        self.N_val_blk = S_val // T_val
        self.neighbor_blk = [(T_val * i, T_val * (i + 1)) for i in range(self.N_val_blk)]
        self.neighbor_ptr = [0 for _ in range(num_streams)]
        assert S_val % T_val == 0, "S_val 必须是 T_val 的整数倍，也即，验证集见到的邻居数目必须是其分块大小的整数倍"
        
        self.main_device = main_device
        self.num_streams = num_streams  
        self.main_locations = torch.empty(0)
        self._cnc_cache = {}
        self._loc_cache = {}
        self._eu_val_cache = {}
        self._eu_nrm_cache = {}

    def update_locations(self, main_locations: torch.Tensor):
        self.main_locations = main_locations
    
    def new_epoch(self):
        self.train_ptr   = [0 for _ in range(self.num_streams)]
    
    def generate_graph(self):
        """
            1. 对于 (N_dyn, )，生成 N_dyn 个节点的 d-正则随机图，大小为 (N_dyn, S_dyn) 
            2. 对于 (N_sta, )，让它直接连向自身，剩下 S_dyn - N_sta 个邻居从 (N_dyn, ) 中选（但是之后其实不要它们）
            3. 对于 (N_valid, )，让它从 N_dyn 中选择 S_valid 个邻居
            4. 所以，我们得到两张图：
                - 训练图：大小为 (N_train, S_train)，为 (N_dyn, S_dyn) + (N_sta, S_dyn)
                - 验证图：大小为 (N_valid, S_valid)，
        """
        
        
        ### step 1: 生成训练-动态图 ###
        dyn_graph: List[List[int]] = []
        G = ig.Graph.K_Regular(n=self.N_dyn, k=self.S_dyn, directed=False, multiple=False)
        for node in range(self.N_dyn):
            neighbors = list(G.neighbors(node))
            assert len(neighbors) == self.S_dyn, f"节点 {node} 邻居数 != {self.S_dyn}"
            dyn_graph.append(neighbors)
        
        ### step 2: 生成训练-静态图 ###
        sta_graph: List[List[int]] = []
        for node in range(self.N_sta):
            neighbors = range(self.N_dyn, self.N_train)
            neighbors = list(neighbors)
            neighbors.extend(range(0, self.S_dyn - self.N_sta))
            sta_graph.append(neighbors) # 后面需要保证：mask 会把 self.N_sta 开始的邻居屏蔽
        
        ### step 3: 生成训练图 ###
        train_graph = dyn_graph + sta_graph  # (N_train, S_dyn)
        self.train_graph = torch.tensor(train_graph, dtype=torch.long, device=self.main_device) 
        
        ### step 4: 生成验证图 ###
        self.valid_graph = torch.randint(0, self.N_dyn, (self.N_valid, self.S_val), 
                                                     dtype=torch.long, device=self.main_device)
        
        
        
    def get_connection(self, cur_slice: slice, cur_type: str):
        """
        输入:
            sta_id: (m,) 的 LongTensor，表示节点 id
        输出:
            (m, c) 的 LongTensor，每行是对应节点的 c 个邻居 id
        """
        if cur_type == "train":
            return self.train_graph[cur_slice] # 切片索引
        elif cur_type == "valid":
            return self.valid_graph[cur_slice] # 切片索引
        
    def generate_connections(self, expected_type: Literal["train", "valid"]):
        for block in self.splits:
            cur_type = get_type(block, self.N_train, self.N_valid)
            if expected_type is not None and cur_type != expected_type:
                continue 
            
            self._cnc_cache[block]    = self.get_connection(
                slice(block[0], block[1]) if cur_type == "train" else 
                slice(block[0]-self.N_train, block[1]-self.N_train),
                cur_type=cur_type
            )
            
            self._eu_val_cache[block]     = (self.emb_val[block[0]:block[1]][:, None, :] @ torch.cat(
                [self.emb_val[self._cnc_cache[block]], self.voc_emb], 
            dim=1).permute(0, 2, 1)).squeeze()  # (T, 1, dim) @ (T, dim, S_tot) -> (T, 1, S_tot) -> (T, S_tot)
            
            if cur_type == "train":
                self._eu_nrm_cache[block] = (self.emb_nrm[block[0]:block[1]][:, None, :] @ self.voc_nrm).permute(0, 2, 1).squeeze()  
                                                # (T, 1, 1)   @ (T, 1, S_sta)   -> (T, 1, S_sta) -> (T, S_sta)
            
        
    def reset_indices(self, cur_type: str):
        if cur_type == "train":
            # 我们保证只打乱动态节点
            randperm = torch.randperm(self.N_dyn, device=self.main_device)
            randperm = torch.cat([randperm, torch.arange(self.N_dyn, self.N_train, device=self.main_device)], dim=0)
            self.train_graph = self.train_graph[randperm]
            self.generate_connections("train")
        else:
            self.valid_graph = torch.randint(0, self.N_dyn, (self.N_valid, self.S_val), 
                                                     dtype=torch.long, device=self.main_device)
            self.generate_connections("valid")





    def next_block(self, sid: int, cur_type: str) -> Tuple[
        Optional[int],              # block_id
        Optional[Tuple[int, int]],  # block
        Optional[int],              # st
        Optional[int],              # ed, 当 st == ed 时你就知道该停止了
        Optional[Tuple[int, int]]   # nbr_block, 仅在 cur_type == "valid" 时有效
    ]:
        if cur_type == "train":
            if self.train_ptr[sid] < len(self.train_blk4stream[sid]):
                block_id = self.train_blk4stream[sid][self.train_ptr[sid]]
                block    = self.splits[block_id]
                self.train_ptr[sid] += 1
                return block_id, block, 0, 0, None
            return None, None, None, None, None
        elif cur_type == "valid":
            if self.train_ptr[sid] < len(self.train_blk4stream[sid]):
                block_id = self.train_blk4stream[sid][self.train_ptr[sid]]
                block    = self.splits[block_id]
                nbr_blk_id = self.neighbor_ptr[sid]
                nbr_block  = self.neighbor_blk[nbr_blk_id]
                    
                self.neighbor_ptr[sid] += 1
                if self.neighbor_ptr[sid] >= self.N_val_blk:
                    self.neighbor_ptr[sid] = 0
                    self.train_ptr[sid]  += 1
            
                return block_id, block, nbr_blk_id, self.N_val_blk - 1, nbr_block
            return None, None, None, None, None
        else:
            raise ValueError(f"Unsupported cur_type: {cur_type}")
    
    def get_sta_loc(self, block: Tuple[int, int]) -> torch.Tensor:
        return self.main_locations[block[0]:block[1]] # (T, D)
    
    def get_pos_loc(self, block: Tuple[int, int], nbr_block: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if nbr_block is None:
            return self.main_locations[self.get_cnc(block)] # (T, S_cos, D)
        else:
            return self.main_locations[self.get_cnc(block)[:, nbr_block[0]:nbr_block[1], :]] # (T, T_val, D)
        
    def get_eu(self, block: Tuple[int, int],  nbr_block: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if nbr_block is None:
            return self._eu_val_cache[block], self._eu_nrm_cache[block]      # (T, S_tot)
        else:
            return (self._eu_val_cache[block][:, nbr_block[0]:nbr_block[1]], # (T, T_val)
                    self._eu_nrm_cache[block][:, nbr_block[0]:nbr_block[1]]) # (T, T_val)
    
    def get_cnc(self, block: Tuple[int, int]) -> torch.Tensor:
        return self._cnc_cache[block]  # (T, c)



import threading
import time
from collections import deque


class Prefetcher:
    """
    负责多设备/多流的数据异步管线管理：
    - 注册设备与其 copy-stream
    - H2D 侧按设备拷贝并生成 ready event
    - 计算侧按需获取并等待事件
    """
    def __init__(
        self, 
        main_device    : torch.device, 
        devices        : List[torch.device], 
        compute_streams: List[torch.cuda.Stream],
        sampler        : Expander_Sampler,
        num_prefetch   : int = 8
    ):
        assert len(devices) == len(compute_streams)
        self.main_device    = main_device
        self.devices        = devices
        self.num_devices    = len(devices)
        self.compute_stream = compute_streams
        self.sampler        = sampler
        self.num_prefetch   = num_prefetch
        self.cur_type       = "train"
        
        
        # 主生产流
        self.prod_stream    = torch.cuda.Stream(device=main_device)  # type: ignore
        
        # 每个 stream_id 一个 copy stream（在目标设备）
        self.copy_streams: Dict[int, torch.cuda.Stream] = { # type: ignore
            sid: torch.cuda.Stream(device=dev) for sid, dev in enumerate(self.devices)
        } 
        
        # 就绪缓存：sid -> deque[ (block_id, block, pos_loc_dev, pos_emb_dev, ready_event) ]
        self.ready_cache  : Dict[int, deque] = {sid: deque() for sid in range(len(self.devices))}
        # in-flight 源引用池：sid -> deque[ (pos_loc_src, pos_emb_src, ready_evt_for_copy_done) ]
        self.src_hold: Dict[int, deque] = {sid: deque() for sid in range(len(self.devices))}
        
        # 锁：唤醒各 sid 的预取线程
        self._cv  = [threading.Condition() for _ in self.devices]
        self._run = False
        
        # 后台线程 per sid
        self._threads: List[threading.Thread] = []
        self._new_epoch_flag: List[bool] = [False for _ in self.devices]  # 每个 sid 的新 epoch 标志

    # ------------ 生命周期控制 ------------ #
    def start(self):
        assert self.sampler is not None, "必须先设置 sampler"
        assert not self._run, "已经启动"
        
        self._run = True
        
        # 为每个 sid（stream_id）启动一个线程
        for sid in range(self.num_devices):
            thread = threading.Thread(target=self._worker, args=(sid,), daemon=True)
            thread.start()
            self._threads.append(thread)
    
    def new_epoch(self): 
        for sid in range(len(self.devices)):
            self.ready_cache[sid].clear()
            self.src_hold[sid].clear()
            self._new_epoch_flag[sid] = True
        
        self.sampler.new_epoch()
        
        for sid in range(len(self.devices)):
            with self._cv[sid]:
                self._cv[sid].notify()
    
    def stop(self):
        if not self._run:
            return
        self._run = False
        
        # 唤醒所有线程，让它们退出
        for cv in self._cv:
            with cv:
                cv.notify_all()
        
        # 等待所有线程结束
        for thread in self._threads:
            thread.join()
        self._threads.clear()   



    
    # ------------ 后台工作线程 ------------ #
    def _worker(self, sid: int):
        try:
            dev   = self.devices[sid]
            cstr  = self.copy_streams[sid]
            cv    = self._cv[sid]
            
            while self._run:
                # 尝试释放 src_hold，也即 main_device 的显存
                self._drain_src_hold(sid)  
                
                with cv: # 当前锁
                    while self._run and len(self.ready_cache[sid]) >= self.num_prefetch:
                        cv.wait(timeout=10000)  # 等待被唤醒，或超时
                    if not self._run:
                        break
                
                # 尝试拉取下一个 block
                block_id, block, st, ed, nbr_block = self.sampler.next_block(sid, self.cur_type)
                if block is None:
                    self.ready_cache[sid].append((None, None, None, None, None, None))
                    with cv:
                        cv.notify()
                        # 不 break，而是等主线程设置某个标志
                        while self._run and not self._new_epoch_flag[sid]:
                            cv.wait(timeout=10000)
                        self._new_epoch_flag[sid] = False
                        continue  # 回到 while self._run:，进入新一轮生产
            
                # (1) 生产阶段：队列未满，且有 block 可取，则开启生产线程
                with torch.cuda.device(self.main_device), torch.cuda.stream(self.prod_stream): # type: ignore
                    sta_loc_src = self.sampler.get_sta_loc(block)                   # (T, D)
                    pos_loc_src = self.sampler.get_pos_loc(block, nbr_block)        # (T, S_tot, D)
                    eu_val_src, eu_nrm_src = self.sampler.get_eu(block, nbr_block)  # (T, dim)
                    
                    ready_evt_for_copy_done = torch.cuda.Event(enable_timing=False, blocking=False)
                    ready_evt_for_copy_done.record(self.prod_stream)  # 在生产流上记录 ready
                
                # (2) 传输阶段：在 copy stream 上等待 ready，然后拷贝到目标设备
                with torch.cuda.device(dev), torch.cuda.stream(cstr): # type: ignore
                    cstr.wait_event(ready_evt_for_copy_done)  # 等待生产流完成
                    sta_loc_dev = sta_loc_src.to(dev, non_blocking=True)
                    pos_loc_dev = pos_loc_src.to(dev, non_blocking=True)
                    eu_val_dev = eu_val_src.to(dev, non_blocking=True)
                    eu_nrm_dev = eu_nrm_src.to(dev, non_blocking=True)
                    all_ready_evt = torch.cuda.Event(enable_timing=False, blocking=False)
                    all_ready_evt.record(cstr)  # 在 copy stream 上记录 ready 事件

                # (3) 保持引用：防止 pos_loc_src/pos_val_src 被释放；之所以可能被释放，是因为 with 之后就是线程的自主运行
                self.src_hold[sid].append((sta_loc_src, pos_loc_src, eu_val_src, eu_nrm_src, ready_evt_for_copy_done))
                
                # (4) 放入就绪缓存
                self.ready_cache[sid].append((block_id, block, st, ed, sta_loc_dev, pos_loc_dev, eu_val_dev, eu_nrm_dev, all_ready_evt))

                with cv:
                    cv.notify()  # 通知可能在等待的 get_sample
        except Exception as e:
            print(f"Exception in thread {sid}: {e}")
            import traceback
            traceback.print_exc()
            os._exit(1)  # 立即终止所有线程和进程
    
    def _drain_src_hold(self, sid: int, quota: int = 4):
        """尝试释放指定 sid 的 src_hold 中的部分内存"""
        dq  = self.src_hold[sid]
        cnt = 0
        
        while dq and cnt < quota:
            _, _, _, _, evt = dq[0]
            if evt.query():
                dq.popleft()
                cnt += 1
            else:
                break
            
    
    
    
    # ------------ 计算侧获取 ------------ #
    
    def get_sample(self, sid: int) -> Tuple[int, Tuple[int,int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  
        """
        仅消费：若队列为空，阻塞等待；不做调度。
        消费后通知后台线程以便补货。
        """
        
        device = self.devices[sid]
        cv     = self._cv[sid]
        
        with cv:
            while self._run and len(self.ready_cache[sid]) == 0:
                cv.wait(timeout=10000)  # 等待被唤醒，或超时
            if not self._run:
                raise RuntimeError("Prefetcher 已停止")
            
        # 如果队列非空，则取出
        block_id, block, st, ed, sta_loc_dev, pos_loc_dev, eu_val_dev, eu_nrm_dev, ready_evt = self.ready_cache[sid].popleft()
        # if block is None:
        #     return None, None, None, None
        
        # # 计算流需要等待 ready_evt
        # comp_stream = self.compute_stream[sid]
        # with torch.cuda.device(device): # type: ignore
        #     comp_stream.wait_event(ready_evt)  # 等待 copy 流完成
            
        with cv:
            cv.notify()  # 通知可能在等待的生产线程
        
        return block_id, block, st, ed, sta_loc_dev, pos_loc_dev, eu_val_dev, eu_nrm_dev, ready_evt

# from typing import Tuple

# import torch


# class Random_Sample(BaseSample):
#     def __init__(self, n: int, c: int, allow_self_loop: bool = False, unique: bool = True, device="cpu"):
#         """
#         完全随机生成 (n, c) 的邻接表。
        
#         参数:
#             n: 节点总数
#             c: 每个节点连接的邻居数
#             allow_self_loop: 是否允许自环 (自己连自己)
#             unique: 是否保证每行邻居不重复
#             device: 存储张量的设备
#         """
#         super().__init__()
#         self.n = n
#         self.c = c
#         self.allow_self_loop = allow_self_loop
#         self.unique = unique
#         self.device = device

#         self.connections = None
#         self.valid_indices = None
#         self._cnc_cache = {}

#     def generate_connection(self):
#         """生成 (n, c) 的邻接表"""
#         if self.unique:
#             # 每行不重复，使用 torch.multinomial
#             all_indices = torch.arange(self.n, device=self.device)
#             connections = []
#             for i in range(self.n):
#                 candidates = all_indices
#                 if not self.allow_self_loop:
#                     candidates = candidates[candidates != i]
#                 neighbors = torch.multinomial(
#                     torch.ones(len(candidates), device=self.device),
#                     self.c,
#                     replacement=False
#                 )
#                 connections.append(candidates[neighbors])
#             self.connections = torch.stack(connections, dim=0)
#         else:
#             # 允许重复采样
#             self.connections = torch.randint(
#                 0, self.n, (self.n, self.c), device=self.device
#             )
#             if not self.allow_self_loop:
#                 mask = self.connections == torch.arange(self.n, device=self.device).unsqueeze(1)
#                 while mask.any():
#                     self.connections[mask] = torch.randint(
#                         0, self.n, (mask.sum().item(),), device=self.device
#                     )
#                     mask = self.connections == torch.arange(self.n, device=self.device).unsqueeze(1)

#     def get_connection(self, sta_id: torch.Tensor):
#         return self.connections[sta_id]

#     def get_size(self) -> Tuple[int, int]:
#         return (self.n, self.c)

#     def __setitem__(self, key, value):
#         self._cnc_cache[key] = value

#     def __getitem__(self, key):
#         return self._cnc_cache[key]


if __name__ == "__main__":
    pass

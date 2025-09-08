import math
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch


class BaseSample(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_connection(self, sta_id: torch.Tensor, dev_num=0) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_size(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def __setitem__(self, key, value) -> torch.Tensor:
        pass
    
    @abstractmethod
    def __getitem__(self, key) -> torch.Tensor:
        pass

class CTE_Sample(BaseSample):
    def __init__(self, h, f, division_fact, N):
        super().__init__()  
        assert N == 2 ** h, "N must be 2**h"
        
        self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.h = h
        self.n = int(2**h)
        self.f = f
        self.k = int(f*h // division_fact)
        self.c = self.h * self.k + 1
        
        self.loc2id = torch.randperm(self.n, dtype=torch.int64) # (N,), 每个位置上对应的是哪个真正节点的 id
        self.id2loc = torch.argsort(self.loc2id) # (N,), 每个节点的 id 对应的位置
        
        self._cnc_id_cache = {}
    
    def _generate_random_masks(self, sz):
        device = 'cpu'

        upper_bounds   = 2 ** torch.arange(self.h, dtype=torch.int64, device=device)
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k), dtype=torch.int64, device=device) # (H, n, K)
        masks = random_numbers & (upper_bounds.view(-1, 1, 1) - 1)
        return masks.permute(1, 0, 2) # (n, H, K)

    def _connection(self, sta_loc):
        # sta_loc: (n, )
        device = 'cpu'
        
        flip_masks = (1 << torch.arange(self.h, device=device, dtype=sta_loc.dtype)).unsqueeze(0) # (1, H)
        flipped_ints = sta_loc.unsqueeze(1) ^ flip_masks # (n, H)
        random_masks = self._generate_random_masks(flipped_ints.size(0))
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k)
        # (n, H, 1) ^ (n, H, K) -> (n, H*K)
        loc = torch.cat((result, sta_loc.unsqueeze(1)), dim=1) # (n, H*K + 1)
        return loc

    def get_connection(self, sta_id: torch.Tensor):
        # sta: (n, ), 表示节点 id
        sta_loc = self.id2loc[sta_id]           # (n, ) , 表示节点位置
        cnc_loc = self._connection(sta_loc)     # (n, c), 表示每个节点位置对应的连接节点位置
        cnc_id  = self.loc2id[cnc_loc]          # (n, c), 表示每个节点 id 对应的连接节点 id
        return cnc_id
    
    def get_size(self) -> Tuple[int, int]:
        return (0, self.c)
    
    def reset_all(self):
        self.loc2id = torch.randperm(self.n, dtype=torch.int64)
        self.id2loc = torch.argsort(self.loc2id)
        self._cnc_id_cache = {}
    
    def reset_cnc(self):
        self._cnc_id_cache = {}
    
    def __setitem__(self, key, value):
        self._cnc_id_cache[key] = value

    def __getitem__(self, key):
        return self._cnc_id_cache[key]


class CTE_Sort_Sample(BaseSample):
    def __init__(self, c: int):
        super().__init__()  
        self.c = c
        self.connections = None
        self._cnc_cache = {}
        pass

    def generate_connection(self, locations: torch.Tensor):
        # locations: (n, c)
        mean_locations = locations.float().mean(dim=-1)  # (n,)
        sorted_values, sorted_indices = torch.sort(mean_locations)

        # ---- 绘制直方图并保存 ----
        # plt.figure()
        # plt.hist(sorted_values.cpu().numpy(), bins=50)  # 直方图，50 个桶
        # plt.xlabel("Mean location value (sorted)")
        # plt.ylabel("Frequency")
        # plt.title("Histogram of sorted_values")
        # plt.tight_layout()
        # plt.savefig("sorted_values_hist.png")
        # plt.close()


        n, c = locations.size(0), self.c
        num_neighbors = int(0.8 * c)
        num_random = c - num_neighbors

        # ---- 向量化邻居 ----
        idx = torch.arange(n, device=locations.device)  # (n,)
        L = num_neighbors // 2
        R = num_neighbors - L

        # 初始窗口 [start, start+num_neighbors)
        start = idx[:, None] - L
        start = start.clamp(min=0)               # 左边不够 -> 往右移
        start = torch.minimum(start, (n - num_neighbors) * torch.ones_like(start))  
        # 右边不够 -> 往左移

        neighbor_indices = start + torch.arange(num_neighbors, device=locations.device)
        # (n, num_neighbors)，每行一个窗口
        neighbor_indices = sorted_indices[neighbor_indices]

        # ---- 随机部分 ----
        random_indices = torch.randint(0, n, (n, num_random), device=locations.device)
        random_indices = sorted_indices[random_indices]

        # ---- 拼接 ----
        connections = torch.cat((neighbor_indices, random_indices), dim=1)
        self.connections = connections

    def get_connection(self, sta_id: torch.Tensor):
        return self.connections[sta_id]

    def get_size(self) -> Tuple[int, int]:
        return (0, self.c)
    
    def __setitem__(self, key, value):
        self._cnc_cache[key] = value
    
    def __getitem__(self, key):
        return self._cnc_cache[key]


import networkx as nx

class Expander_Sample(BaseSample):
    def __init__(self, n: int, c: int):
        """
        基于 d-正则随机图生成 Expander Graph.
        
        参数:
            n: 节点总数
            c: 每个节点的度 (d-regular)
        """
        super().__init__()
        assert c < n, "度 c 必须小于节点总数 n"
        if c & 1 != 0:
            c = c + 1  # 如果 c 是奇数，增加到下一个偶数
        self.n = n
        self.c = c
        self.connections = None
        self._cnc_cache = {}

    def generate_connection(self):
        """生成 (n, c) 的邻接表"""
        G = nx.random_regular_graph(self.c, self.n)

        connections = []
        for node in range(self.n):
            neighbors = list(G.neighbors(node))
            assert len(neighbors) == self.c, f"节点 {node} 邻居数 != {self.c}"
            connections.append(neighbors)

        self.connections = torch.tensor(connections, dtype=torch.long)

    def reset_indices(self):
        randperm = torch.randperm(self.n)
        self.connections = randperm[self.connections]
    
    def get_connection(self, sta_id: torch.Tensor):
        """
        输入:
            sta_id: (m,) 的 LongTensor，表示节点 id
        输出:
            (m, c) 的 LongTensor，每行是对应节点的 c 个邻居 id
        """
        return self.connections[sta_id]

    def get_size(self) -> Tuple[int, int]:
        return (0, self.c)

    def __setitem__(self, key, value):
        self._cnc_cache[key] = value

    def __getitem__(self, key):
        return self._cnc_cache[key]

import torch
from typing import Tuple

class Random_Sample(BaseSample):
    def __init__(self, n: int, c: int, allow_self_loop: bool = False, unique: bool = True, device="cpu"):
        """
        完全随机生成 (n, c) 的邻接表。
        
        参数:
            n: 节点总数
            c: 每个节点连接的邻居数
            allow_self_loop: 是否允许自环 (自己连自己)
            unique: 是否保证每行邻居不重复
            device: 存储张量的设备
        """
        super().__init__()
        self.n = n
        self.c = c
        self.allow_self_loop = allow_self_loop
        self.unique = unique
        self.device = device

        self.connections = None
        self._cnc_cache = {}

    def generate_connection(self):
        """生成 (n, c) 的邻接表"""
        if self.unique:
            # 每行不重复，使用 torch.multinomial
            all_indices = torch.arange(self.n, device=self.device)
            connections = []
            for i in range(self.n):
                candidates = all_indices
                if not self.allow_self_loop:
                    candidates = candidates[candidates != i]
                neighbors = torch.multinomial(
                    torch.ones(len(candidates), device=self.device),
                    self.c,
                    replacement=False
                )
                connections.append(candidates[neighbors])
            self.connections = torch.stack(connections, dim=0)
        else:
            # 允许重复采样
            self.connections = torch.randint(
                0, self.n, (self.n, self.c), device=self.device
            )
            if not self.allow_self_loop:
                mask = self.connections == torch.arange(self.n, device=self.device).unsqueeze(1)
                while mask.any():
                    self.connections[mask] = torch.randint(
                        0, self.n, (mask.sum().item(),), device=self.device
                    )
                    mask = self.connections == torch.arange(self.n, device=self.device).unsqueeze(1)

    def get_connection(self, sta_id: torch.Tensor):
        return self.connections[sta_id]

    def get_size(self) -> Tuple[int, int]:
        return (self.n, self.c)

    def __setitem__(self, key, value):
        self._cnc_cache[key] = value

    def __getitem__(self, key):
        return self._cnc_cache[key]


if __name__ == "__main__":
    pass

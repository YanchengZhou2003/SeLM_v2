import math
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch

from src.para import *
from src.utils import get_idx, get_type


class BaseSample(ABC):
    def __init__(self):
        pass
    

import igraph as ig
import networkx as nx


class TrainSampler(BaseSample):
    def __init__(
        self, 
    ):
        """
        基于 d-正则随机图生成 Expander Graph.
        
        """
        super().__init__()
        train_train_graph: List[List[int]] = []
        G = ig.Graph.K_Regular(n=N_train, k=N_ttnbr, directed=False, multiple=False)
        for node in range(N_train):
            neighbors = list(G.neighbors(node))
            assert len(neighbors) == N_ttnbr, f"节点 {node} 邻居数 != {N_ttnbr}"
            train_train_graph.append(neighbors)
        self.train_train_graph = torch.tensor(train_train_graph, dtype=torch.long, device=main_device) # (N_train, T_trnbr)
        self.train_vocab_graph = torch.randint(0, N_vocab, (N_train, N_tvnbr), device=main_device)     # (N_train, T_trnbr)
            
    def get_cos_connection(self, block: Tuple[int, int], cur_type="train_train") -> torch.Tensor:
        if cur_type == "train_train":
            return self.train_train_graph[block[0]:block[1]] 
        elif cur_type == "train_vocab":
            return self.train_vocab_graph[block[0]:block[1]]
        else:
            raise ValueError(f"Unsupported cur_type: {cur_type}")
    
    def get_cro_connection(self, cur_tar: torch.Tensor) -> torch.Tensor:
        voc_idx = torch.randint(0, N_vocab, (T_train, K_vocab - 1), device=main_device, dtype=torch.long)
        out     = torch.cat([cur_tar[:, None], voc_idx], dim=1) # (T_train, K_vocab)
        return out
        
    def reset_indices(self):
        randperm = torch.randperm(N_train, device=main_device)
        self.train_train_graph = self.train_train_graph[randperm]
        self.train_vocab_graph = torch.randint(0, N_vocab, (N_train, N_tvnbr), device=main_device)


class VocabSampler(BaseSample):
    def __init__(
        self, 
    ):
        """
        基于 Expander Graph 对 vocabulary 进行采样.
        """
        super().__init__()
        vocab_vocab_graph: List[List[int]] = []
        G = ig.Graph.K_Regular(n=N_vocab + int(N_vocab % 2 != 0), k=N_vvnbr, directed=False, multiple=False)
        for node in range(N_vocab):
            neighbors = list(G.neighbors(node))
            assert len(neighbors) == N_vvnbr, f"节点 {node} 邻居数 != {N_vvnbr}"
            vocab_vocab_graph.append(neighbors)
        self.vocab_vocab_graph = torch.tensor(vocab_vocab_graph, dtype=torch.long, device=main_device) # (N_vocab, T_vvnbr)
        if N_vocab % 2 != 0:
            self.vocab_vocab_graph[self.vocab_vocab_graph == N_vocab] = 0
        
        self.vocab_train_graph = torch.randint(0, N_train, (N_vocab, N_vtnbr), device=main_device)     # (N_vocab, T_vtnbr)
            
    def get_cos_connection(self, block: Tuple[int, int], cur_type="vocab_vocab") -> torch.Tensor:
        if cur_type == "vocab_vocab":
            return self.vocab_vocab_graph[block[0]:block[1]]
        elif cur_type == "vocab_train":
            return self.vocab_train_graph[block[0]:block[1]]
        else:
            raise ValueError(f"Unsupported cur_type: {cur_type}")
    
    def reset_indices(self):
        randperm = torch.randperm(N_vocab, device=main_device)
        self.vocab_vocab_graph = self.vocab_vocab_graph[randperm]
        self.vocab_train_graph = torch.randint(0, N_train, (N_vocab, N_vtnbr), device=main_device)


class ValidSampler(BaseSample):
    def __init__(
        self, 
    ):
        """
        """
        super().__init__()
        self.graph = torch.randint(0, N_train, (N_valid, N_vanbr), device=main_device) # (N_valid, N_vanbr)
            
    def get_cos_connection(self, block: Tuple[int, int]) -> torch.Tensor:
        return self.graph[block[0]:block[1]] 

    def reset_indices(self):
        self.graph = torch.randint(0, N_train, (N_valid, N_vanbr), device=main_device) # (N_valid, N_vanbr)


if __name__ == "__main__":
    pass

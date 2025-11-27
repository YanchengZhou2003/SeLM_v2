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
        super().__init__()
        
        if N_dynbr < N_train:
            expander_graph: List[List[int]] = []
            
            _G = ig.Graph.K_Regular(n=N_train, k=N_dynbr, directed=False, multiple=False)
            for node in range(N_train):
                neighbors = list(_G.neighbors(node))
                expander_graph.append(neighbors)
                
            self.dyn_graph = torch.tensor(expander_graph, dtype=torch.long, device=main_device)        # (N_train, N_dynbr)
        elif N_dynbr == N_train:
            self.dyn_graph = torch.arange(N_train, device=main_device).unsqueeze(0).repeat(N_train, 1) # (N_train, N_train)
        else:
            raise ValueError(f"N_dynbr ({N_dynbr}) cannot be larger than N_train ({N_train})")
        self.sta_graph = torch.arange(N_vocab, device=main_device).unsqueeze(0).repeat(N_train, 1)     # (N_train, N_stnbr)
        
    def get_connection(self, block: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dyn_graph[block[0]:block[1]], self.sta_graph[block[0]:block[1]]  # (T_train, N_dynbr), (T_train, N_stnbr) 
        
    def reset_indices(self):
        randperm = torch.randperm(N_train, device=main_device)
        self.dyn_graph = self.dyn_graph[randperm]

'''
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
        
        # self.vocab_train_graph = torch.randint(0, N_train, (N_vocab, N_vtnbr), device=main_device)     # (N_vocab, T_vtnbr)
            
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
        # self.vocab_train_graph = torch.randint(0, N_train, (N_vocab, N_vtnbr), device=main_device)
'''

class ValidSampler(BaseSample):
    def __init__(
        self, 
        train_dyn_graph: torch.Tensor
    ):
        """
        """
        random_indices  = torch.randint(0, N_train, (N_valid,), device=main_device)

        self.train_dyn_graph = train_dyn_graph.to(main_device)
        self.dyn_graph = train_dyn_graph[random_indices] # (N_valid, N_dynbr)
        self.sta_graph = torch.arange   (N_vocab, device=main_device).unsqueeze(0).repeat(N_valid, 1) # (N_valid, N_stnbr)

    def get_connection(self, block: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dyn_graph[block[0]:block[1]], self.sta_graph[block[0]:block[1]]

    def reset_indices(self):
        random_indices = torch.randint  (0, N_train, (N_valid,), device=main_device) # (N_valid,)
        self.dyn_graph = self.train_dyn_graph[random_indices] # (N_valid, N_dynbr)


if __name__ == "__main__":
    pass

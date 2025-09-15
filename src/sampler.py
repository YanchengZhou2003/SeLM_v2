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
        graph: List[List[int]] = []
        G = ig.Graph.K_Regular(n=N_train, k=N_trnbr, directed=False, multiple=False)
        for node in range(N_train):
            neighbors = list(G.neighbors(node))
            assert len(neighbors) == N_trnbr, f"节点 {node} 邻居数 != {N_trnbr}"
            graph.append(neighbors)
        self.graph = torch.tensor(graph, dtype=torch.long, device=main_device) # (N_train, T_trnbr)
            
    def get_connection(self, block: Tuple[int, int]) -> torch.Tensor:
        return self.graph[block[0]:block[1]] 

    def reset_indices(self):
        randperm = torch.randperm(N_train, device=main_device)
        self.graph = self.graph[randperm]

class VocabSampler(BaseSample):
    def __init__(
        self, 
    ):
        """
        基于 Sampled Softmax 对 vocabulary 进行采样.
        """
        super().__init__()
            
    def get_connection(self, cur_tar: torch.Tensor) -> torch.Tensor:
        B = cur_tar.size(0)
        
        # cnc_idx = torch.randint(0, N_vocab, (cur_tar.size(0), K_vocab), device=main_device, dtype=torch.long)
        # cnc_idx[:, 0] = cur_tar
        
        voc_idx = torch.arange(N_vocab, device=main_device).expand(B, N_vocab)  # (B, N_vocab)
        mask    = voc_idx != cur_tar[:, None]                                   # (B, N_vocab)
        out     = torch.cat(
            [cur_tar[:, None], voc_idx[mask].view(B, N_vocab - 1)], 
        dim=1)                                                                  # (B, N_vocab)
        
        return out



class ValidSampler(BaseSample):
    def __init__(
        self, 
    ):
        """
        基于 d-正则随机图生成 Expander Graph.
        
        """
        super().__init__()
        self.graph = torch.randint(0, N_train, (N_valid, N_vanbr), device=main_device) # (N_valid, N_vanbr)
            
    def get_connection(self, block: Tuple[int, int]) -> torch.Tensor:
        return self.graph[block[0]:block[1]] 

    def reset_indices(self):
        self.graph = torch.randint(0, N_train, (N_valid, N_vanbr), device=main_device) # (N_valid, N_vanbr)


if __name__ == "__main__":
    pass

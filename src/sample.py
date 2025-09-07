import os
import sys
import time
from datetime import datetime
import torch


class CTE_Sample:
    def __init__(self, h, f, division_fact,
                 N):
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
    
    def _generate_random_masks(self, sz, dev_num=0):
        device = self.devices[dev_num] if dev_num >= 0 else 'cpu'

        upper_bounds   = 2 ** torch.arange(self.h, dtype=torch.int64, device=device)
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k), dtype=torch.int64, device=device) # (H, n, K)
        masks = random_numbers & (upper_bounds.view(-1, 1, 1) - 1)
        return masks.permute(1, 0, 2) # (n, H, K)

    def _connection(self, sta_loc, dev_num=0):
        # sta_loc: (n, )
        device = self.devices[dev_num] if dev_num >= 0 else 'cpu'
        
        flip_masks = (1 << torch.arange(self.h, device=device, dtype=sta_loc.dtype)).unsqueeze(0) # (1, H)
        flipped_ints = sta_loc.unsqueeze(1) ^ flip_masks # (n, H)
        random_masks = self._generate_random_masks(flipped_ints.size(0), dev_num=dev_num)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k)
        # (n, H, 1) ^ (n, H, K) -> (n, H*K)
        loc = torch.cat((result, sta_loc.unsqueeze(1)), dim=1) # (n, H*K + 1)
        return loc

    def get_connection(self, sta_id: torch.Tensor, dev_num=0):
        # sta: (n, ), 表示节点 id
        sta_loc = self.id2loc[sta_id]                        # (n, ) , 表示节点位置
        cnc_loc = self._connection(sta_loc, dev_num=dev_num) # (n, c), 表示每个节点位置对应的连接节点位置 
        cnc_id  = self.loc2id[cnc_loc]                       # (n, c), 表示每个节点 id 对应的连接节点 id
        return cnc_id
        

      
    

    
if __name__ == "__main__":
    pass

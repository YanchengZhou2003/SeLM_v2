import argparse
import os
import signal
from typing import Dict, Optional

import torch

from src.utils import make_splits

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
main_device = torch.device('cuda:0')
devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
num_devices = len(devices)

comp_streams = [torch.cuda.default_stream(i) for i in range(torch.cuda.device_count())]
data_streams = [torch.cuda.Stream(0) for _ in range(torch.cuda.device_count())]

parser = argparse.ArgumentParser(description="Set hyperparameters for the model.")
### 1. GPT 训练相关参数
parser.add_argument("--cte_train_bs"      , type=int,   default=32,    help="")
parser.add_argument("--cte_train_iters"   , type=int,   default=1,     help="")
parser.add_argument("--cte_eval_bs"       , type=int,   default=32,    help="")
parser.add_argument("--cte_eval_iters"    , type=int,   default=1,     help="")

parser.add_argument("--N_train"           , type=int,   default=2048 , help="")
parser.add_argument("--T_train"           , type=int,   default=128,   help="")
parser.add_argument("--N_train_neighbors" , type=int,   default=256,   help="")

parser.add_argument("--T_vonbr"           , type=int,   default=512,   help="")

parser.add_argument("--N_valid"           , type=int,   default=2048,  help="")
parser.add_argument("--T_valid"           , type=int,   default=32,    help="")
parser.add_argument("--N_valid_neighbors" , type=int,   default=1024,  help="")

parser.add_argument("--h"                 , type=int,   default=18 ,   help="")
parser.add_argument("--tp"                , type=int,   default=16 ,   help="")
parser.add_argument("--cur_tp"            , type=int,   default=2  ,   help="")
parser.add_argument("--cur_portion"       , type=float, default=0.5 ,  help="")
parser.add_argument("--division_fact"     , type=float, default=1.0 ,  help="")

parser.add_argument("--train_epoch_num"   , type=int,   default=500,   help="")
parser.add_argument("--valid_epoch_num"   , type=int,   default=500,   help="")

parser.add_argument("--train_ratio_cos"   , type=float, default=0.99,  help="") 
parser.add_argument("--train_ratio_cro"   , type=float, default=0.01,  help="")
parser.add_argument("--vocab_ratio_cos"   , type=float, default=1.00,  help="")
parser.add_argument("--vocab_ratio_cro"   , type=float, default=0.00,  help="")

parser.add_argument("--train_converge"    , type=int,   default=50 ,   help="")
parser.add_argument("--valid_converge"    , type=int,   default=50 ,   help="")
parser.add_argument("--train_graph_reset" , type=int,   default=50,    help="")
parser.add_argument("--valid_graph_reset" , type=int,   default=50,    help="")

parser.add_argument("--train_only"        , type=int,   default=0 ,   help="")
parser.add_argument("--valid_only"        , type=int,   default=0 ,   help="")

parser.add_argument("--vis_path"          , type=str,   default='./vis2/tmp' , help="")
parser.add_argument("--use_eu_norm"       , type=int,   default=0    , help="")

args = parser.parse_args()

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
N_vocab = len(chars)

# 超参数：GPT
batch_size        = 64    
block_size        = 256
max_iters         = 4000
val_max_iters     = 1000
gpt_save_interval = 1000
learning_rate     = 3e-4
device            = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eval_iters        = 10
n_embd            = 384
n_head            = 6
n_layer           = 6
dropout           = 0.2

# 超参数：GPT 产生给 CTE 训练的数据
cte_train_bs      = args.cte_train_bs    # 64
cte_train_iters   = args.cte_train_iters # 5
cte_train_samples = cte_train_bs * cte_train_iters * block_size # 64 * 5 * 256 = 81,920
cte_eval_bs       = args.cte_eval_bs     # 32
cte_eval_iters    = args.cte_eval_iters  # 1
cte_eval_samples  = cte_eval_bs * cte_eval_iters * block_size   # 32 * 1 * 256 = 8,192
cte_save_interval = 1

# 超参数：CTE
h                 = args.h
n                 = 2 ** h
tp                = args.tp
factor            = 1 
eps               = 1e-5 

sample_k          = 1
cur_tp            = args.cur_tp            # 2
cur_portion       = args.cur_portion       # 0.5
use_eu_norm       = args.use_eu_norm       # 1

# 超参数：数据集，及其分块
N_train           = args.N_train           # 65536
T_train           = args.T_train           # 256
N_trnbr           = args.N_train_neighbors # 512

T_vonbr           = args.T_vonbr           # 1024

N_valid           = args.N_valid           # 8192
T_valid           = args.T_valid           # 256
N_vanbr           = args.N_valid_neighbors # 8192

emb_size          = N_train + N_vocab + N_valid

train_blocks      = make_splits(0, N_train, T_train) 
vonbr_blocks      = make_splits(0, N_train, T_vonbr)
valid_blocks      = make_splits(0, N_valid, T_valid)

train_loc_slice   = slice(0, N_train)
vocab_loc_slice   = slice(N_train, N_train + N_vocab)
valid_loc_slice   = slice(N_train + N_vocab, N_train + N_vocab + N_valid)    

num_train_blocks  = len(train_blocks)
num_vonbr_blocks  = len(vonbr_blocks)
num_valid_blocks  = len(valid_blocks)

train4sid         = [list(range(sid, num_train_blocks, num_devices)) for sid in range(num_devices)]
vonbr4sid         = [list(range(sid, num_vonbr_blocks, num_devices)) for sid in range(num_devices)]
valid4sid         = [list(range(sid, num_valid_blocks, num_devices)) for sid in range(num_devices)]




# 超参数：训练相关
loss_strategy: Dict = {
    'cos_loss'  : 'lap', # 
    'cro_loss'  : 'cro', # 
    'train_converge'  : args.train_converge,
    'valid_converge'  : args.valid_converge,
    'train_ratio_cos' : args.train_ratio_cos,  
    'train_ratio_cro' : args.train_ratio_cro,
    'vocab_ratio_cos' : args.vocab_ratio_cos,  
    'vocab_ratio_cro' : args.vocab_ratio_cro,    
}
train_epoch_num = args.train_epoch_num # 5
valid_epoch_num = args.valid_epoch_num # 5

train_graph_reset = args.train_graph_reset # 50
valid_graph_reset = args.valid_graph_reset # 50

division_fact   = args.division_fact   # 1.0
N_K             = int(h / division_fact)
N_C             = 2 * N_K * h + 1

train_only      = args.train_only        # 0
valid_only      = args.valid_only        # 0

generators = {}


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    统一设置 random、numpy、torch 的随机种子。
    
    Args:
        seed (int): 随机种子数值。
        deterministic (bool): 是否开启 PyTorch 确定性模式（可能略降速度）。
        benchmark (bool): 是否开启 cuDNN benchmark（会带来非确定性，通常与 deterministic=False 搭配）。
    """
    import os
    import random

    import numpy as np

    # 1) Python 和 NumPy
    random.seed(seed)
    np.random.seed(seed)

    # 2) 环境变量（影响某些哈希/多进程行为）
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        
        for dev_id in range(torch.cuda.device_count()):
            g = torch.Generator(device=f"cuda:{dev_id}")
            g.manual_seed(seed + dev_id + 1)  # 每个 GPU 使用不同种子
            generators[dev_id] = g
                # torch.cuda.manual_seed_all(seed + i + 1)  # 多 GPU

        # cuDNN / 后端设置
        ## torch.backends.cudnn.deterministic = deterministic
        # torch.backends.cudnn.benchmark = benchmark

    except ImportError:
        # 未安装 torch 时忽略
        pass

set_seed(42, deterministic=True, benchmark=False)


print(f"数据集总长度：{len(text)}")


# 额外内容
gpt_path         = './ckpt/gpt'
cte_path         = './ckpt/cte'
cache_path       = './data/'
train_cache_path = './ckpt/cte'
vis_path         = args.vis_path
os.makedirs(vis_path, exist_ok=True)

ST = False
ED = True
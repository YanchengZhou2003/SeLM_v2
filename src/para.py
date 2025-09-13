import argparse
import os
import signal
from typing import Dict, Optional

import torch

# def handler(sig, frame):
#     print("SIGINT received, force exit.")
#     os._exit(1)

# signal.signal(signal.SIGINT, handler)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Set hyperparameters for the model.")
parser.add_argument("--cte_train_bs"   , type=int,   default=32,   help="CTE training batch size")
parser.add_argument("--cte_train_iters", type=int,   default=1,    help="CTE training iterations")
parser.add_argument("--cte_eval_bs"    , type=int,   default=32,   help="CTE evaluation batch size")
parser.add_argument("--cte_eval_iters" , type=int,   default=1,    help="CTE evaluation iterations")
parser.add_argument("--ratio_cos"  , type=float, default=0.95, help="Ratio for") 
parser.add_argument("--ratio_cro"  , type=float, default=0.05,  help="Ratio for")
parser.add_argument("--train_length"   , type=int,   default=5120, help="Training sequence length")
parser.add_argument("--truncate_valid" , type=int,   default=-1 ,  help="Truncate validation set to this length; -1 means no truncation")
parser.add_argument("--sample_factor" ,  type=float, default=1.0 , help="Sample factor for Base_Sample")
parser.add_argument("--h" ,  type=int, default=27 , help="")
parser.add_argument("--tp" ,  type=int, default=2 , help="")
parser.add_argument("--instant_writeback" ,  type=int, default=0 , help="")
parser.add_argument("--N_T" ,  type=int, default=1024 , help="")
parser.add_argument("--epoch_num" ,  type=int, default=5 , help="")
parser.add_argument("--converge" ,   type=int, default=2 , help="")
parser.add_argument("--vis_path" ,   type=str, default='./vis2/tmp' , help="")
parser.add_argument("--cur_tp" ,   type=int, default=2 , help="")
parser.add_argument("--cur_portion" ,   type=float, default=0.5 , help="")

args = parser.parse_args()

# hyperparameters
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


h                 = args.h
tp                = args.tp
factor            = 1 
sample_factor     = args.sample_factor  # 1.0
eps               = 1e-5 
division_fact     = 1
sample_k          = 1
instant_writeback = args.instant_writeback  # 1
cur_tp            = args.cur_tp  # 2
cur_portion       = args.cur_portion  # 0.5

### 这里要实验至少 6 个量级, bs = 1, 2, 4, 8, 16, 32, 64
### 最少：1 * 10 * 256 = 2,560, 最多：64 * 10 * 256 = 163,840
cte_train_bs      = args.cte_train_bs    # 64
cte_train_iters   = args.cte_train_iters # 5
cte_train_samples = cte_train_bs * cte_train_iters * block_size # 64 * 5 * 256 = 81,920
cte_eval_bs       = args.cte_eval_bs     # 32
cte_eval_iters    = args.cte_eval_iters  # 1
cte_eval_samples  = cte_eval_bs * cte_eval_iters * block_size   # 32 * 1 * 256 = 8,192
cte_save_interval = 1

N_T               = args.N_T

train_length      = args.train_length  # 512

loss_strategy: Dict = {
    'cos_loss'  : 'square', # 
    'cro_loss'  : 'js', # 
    'converge'  : args.converge,
    'ratio_cos' : args.ratio_cos,  
    'ratio_cro' : args.ratio_cro    
}
epoch_num=args.epoch_num  # 50
generators = {}


'''
这个拟合效果比较好, loss 还低
loss_type: LossTypeDict = {
    'dyn_loss'  : 'lap', # dyn:  dynamic
    'sta_loss'  : 'square', # sta:  static
    'prob_loss' : 'js' ,    # prob: probability
    30: {
        'target'  : 'sta_only' , 
        'converge': 20
    }, # 50  < epoch <= 100 时仅优化 sta_loss，在第 80 个 epoch 开始 converge
    150: {
        'target'  : 'weighted_dyn_prob',
        'converge': 80,
        'ratio_dyn' : 0.99,
        'ratio_prob': 0.01   
    }
}
epoch_cte=150
'''


### dyn_loss / sta_loss 可选: 'abs', 'square'
### prob_loss 可选: 'kl' , 'js'
### method    可选: 
##### 'name': 'dyn_only'  , 表示当前仅优化 dyn_loss， 其它 loss 只计算、不优化
##### 'name': 'sta_only'  , 表示当前仅优化 sta_loss， 其它 loss 只计算、不优化
##### 'name': 'prob_only' , 表示当前仅优化 prob_loss，其它 loss 只计算、不优化
##### 'name': 'alternated', 表示交替优化, 一个 epoch 依次优化 dyn / sta / prob
##### 'name': 'weighted_dyn_prob'  , 表示加权优化. 需要额外指定 'ratio_dyn' 和 'ratio_prob', 且它们加和为 1


# ------------

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

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


print(f"数据集总长度：{len(text)}")

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 额外内容
gpt_path = './ckpt/gpt'
cte_path = './ckpt/cte'
cache_path = './data/'
train_cache_path = './ckpt/cte'
vis_path = args.vis_path
os.makedirs(vis_path, exist_ok=True)









ST = False
ED = True
import argparse
import os

import torch

from src.utils import LossTypeDict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Set hyperparameters for the model.")
args = parser.parse_args()

# hyperparameters
batch_size        = 64    
block_size        = 256
max_iters         = 1000
gpt_save_interval = 1000
cte_save_interval = 1
learning_rate     = 3e-4
device            = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eval_iters        = 10
n_embd            = 384
n_head            = 6
n_layer           = 6
dropout           = 0.2


h                 = 27
tp                = 2
c                 = 1 
eps               = 1e-5 
division_fact     = 1


### 这里要实验至少 6 个量级, bs = 1, 2, 4, 8, 16, 32, 64
### 最少：1 * 5 * 256 = 1,280, 最多：64 * 5 * 256 = 81,920 
cte_train_bs      = 64
cte_train_iters   = 5
cte_train_samples = cte_train_bs * cte_train_iters * block_size # 64 * 5 * 256 = 81,920
cte_eval_bs       = 32
cte_eval_iters    = 1
cte_eval_samples  = cte_eval_bs * cte_eval_iters * block_size # 32 * 1 * 256 = 8,192


T1_block_size     = 1024
T2_block_size     = 1024



loss_type: LossTypeDict = {
    'dyn_loss'  : 'square', # dyn:  dynamic
    'sta_loss'  : 'square', # sta:  static
    'prob_loss' : 'js' ,    # prob: probability
    30: {
        'target'  : 'sta_only' , 
        'converge': 20
    }, # 50  < epoch <= 100 时仅优化 sta_loss，在第 80 个 epoch 开始 converge
    150: {
        'target'  : 'weighted_dyn_prob',
        'converge': 80,
        'ratio_dyn' : 0.9950,
        'ratio_prob': 0.0050  
    }
}
epoch_cte=150



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

TTT_loss_type : LossTypeDict = {
    'dyn_loss'    : 'square'      , # dyn:  dynamic
    -1: {
        'target'  : 'TTT_only' , 
        'converge': 80
    }, 
}
epoch_cte_TTT=100
sample_k = 1

# ------------

torch.manual_seed(1337)

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
train_cache_path = './ckpt/cte'
sim_eu_path = './vis/' + f'b_{block_size}' + 'sim_eu_i{}.png'
sim_ct_path = './vis/' + f'b_{block_size}' + 'sim_ct_i{}.png'
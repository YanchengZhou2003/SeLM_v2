import json
import os
import random
import warnings
from datetime import datetime

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from tqdm import tqdm

# 仅屏蔽这类 FutureWarning（全局或在特定代码块前）
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch\.load` with `weights_only=False`"
)

from src.cte import *
from src.para import *
from src.utils import *
from src.vis import *

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long, pin_memory=True)
text_size = len(data)
n = int(0.9*text_size) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split, bs=None, ix=None, to_cuda=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = train_data if split == 'train' else val_data
    if bs is None:
        bs = batch_size
    if ix is None:
        ix = torch.randint(len(data) - block_size, (bs,))
    
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    
    if to_cuda:
        x, y, ix = x.to(device, non_blocking=True), y.to(device, non_blocking=True), ix.to(device, non_blocking=True),
    return x, y, ix


    

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)), persistent=False)
        self.rope = RotaryEmbedding(dim=head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)  
        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # head_size = n_embd // n_head
        head_size = 8
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(N_vocab, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_sta_emb(self):
        return F.normalize(self.token_embedding_table.weight, dim=-1)
    
    def forward(self, idx, targets, return_dyn_emb=False):
        B, T, V = batch_size, block_size, N_vocab
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,E)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,E)
        x = tok_emb # + pos_emb # (B,T,E)
        x = self.blocks(x) # (B,T,E)
        x: torch.Tensor = self.ln_f(x) # (B,T,E)
        if return_dyn_emb:
            return F.normalize(x, dim=-1)
        
        token_embeddings = self.token_embedding_table.weight  # (V, E)
        logits_eu = 20. * torch.matmul(F.normalize(x, dim=-1), F.normalize(token_embeddings, dim=-1).t()) # (B, T, V)
        loss_eu = F.cross_entropy(logits_eu.view(B * T, V), targets.view(B * T))
        
        return loss_eu




################ ------------- GPT 训练与评估 ------------- ################
def train_gpt(gpt_ckpt: str):
    model: GPTLanguageModel = GPTLanguageModel().to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    running_loss = []
    for iter in range(max_iters):
        
        xb, yb, _ = get_batch('train', to_cuda=True)
        loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if iter % gpt_save_interval == 0 or iter == max_iters - 1:
            print(f"current iter: {iter}, avg loss in last {gpt_save_interval} iters: {sum(running_loss) / len(running_loss)}")
            running_loss = []
            torch.save(model.state_dict(), os.path.join(gpt_path, gpt_ckpt.format(iter)))

def eval_gpt(gpt_ckpt: str):
    model: GPTLanguageModel = GPTLanguageModel()
    model_cktp = torch.load(os.path.join(gpt_path, gpt_ckpt), map_location='cpu')
    model.load_state_dict(model_cktp)
    model = model.to(device)
    model.eval()
    
    all_loss = []
    for iter in tqdm(range(val_max_iters)):
        xb, yb, _ = get_batch('val', to_cuda=True)
        loss = model(xb, yb)
        all_loss.append(loss.item())
    
    print(f"gpt eval loss: {sum(all_loss) / len(all_loss)}")

################ ------------- 从 GPT 到 CTE ------------- ################

@torch.no_grad()
def get_cte_train_and_test(gpt_ckpt: str, cache_save_path: str):
    ### step 0: 准备模型
    model: GPTLanguageModel = GPTLanguageModel()
    model_cktp = torch.load(os.path.join(gpt_path, gpt_ckpt), map_location='cpu')
    model.load_state_dict(model_cktp)
    model = model.to(device)
    model.eval()
    
    ### step 1: 准备基本数据
    train_y, train_emb = [], []
    eval_y, eval_emb = [], []
    cache_save_path = os.path.join(cache_path, cache_save_path)
    
    ### step 2: 迭代获取动态嵌入（dyn_emb）与数据元信息（ix）
    for _ in range(cte_train_iters):
        X_train, Y_train, _ = get_batch('train', bs=cte_train_bs, to_cuda=True)
        dyn_emb: torch.Tensor = model(X_train, targets=Y_train, return_dyn_emb=True)
        train_y.append(Y_train[:, -1].cpu())
        train_emb.append(dyn_emb[:, -1, :].cpu())
    
    for _ in range(cte_eval_iters):
        X_val, Y_val, _ = get_batch('val', bs=cte_eval_bs, to_cuda=True)
        dyn_emb: torch.Tensor = model(X_val, targets=Y_val, return_dyn_emb=True)
        eval_y.append(Y_val[:, -1].cpu())
        eval_emb.append(dyn_emb[:, -1, :].cpu())

    ### step 3: 拼接并保存
    train_cache = {
        'y': torch.stack(train_y, dim=0).reshape(-1),  # (cte_train_iters * cte_train_bs)
        'emb': torch.stack(train_emb, dim=0).reshape(-1, n_embd) # (cte_train_iters * cte_train_bs, n_embd)
    }
    eval_cache = {
        'y': torch.stack(eval_y, dim=0).reshape(-1),   # (cte_eval_iters  * cte_eval_bs)
        'emb': torch.stack(eval_emb, dim=0).reshape(-1, n_embd)  # (cte_eval_iters   * cte_eval_bs, n_embd)
    }
    train_cache_length = train_cache['emb'].shape[0] 
    eval_cache_length = eval_cache['emb'].shape[0]
    
    print(f"train cache length: {train_cache_length}, eval cache length: {eval_cache_length}")
    save_file(train_cache, cache_save_path.format(train_cache_length, 'train'))
    save_file(eval_cache, cache_save_path.format(eval_cache_length, 'val'))

################ ------------- 训练 CTE ------------- ################

@torch.no_grad()
def train_cte(cache_cktp: str, gpt_ckpt: str, train_length: int, val_length: int,):
    ### step 1: 读取缓存，并 pin 在 cpu。根据计算，10 ** 7 时也仅占用 15 GB
    assert math.log2(train_length) % 1 == 0, "train_length 必须是 2 的整数次幂"
    assert math.log2(val_length) % 1 == 0, "val_length 必须是 2 的整数次幂"
    
    
    train_cache = load_file(os.path.join(cache_path, cache_cktp.format(train_length, 'train')), device='cpu')
    valid_cache = load_file(os.path.join(cache_path, cache_cktp.format(val_length, 'val')), device='cpu')
    train_cache, valid_cache = pin_tensors_in_dict(train_cache), pin_tensors_in_dict(valid_cache)
    
    ''' debug '''
    # if args.truncate_valid > 0:
    #     valid_cache['emb'] = valid_cache['emb'][:args.truncate_valid]
    #     valid_cache['y'] = valid_cache['y'][:args.truncate_valid]
    #     val_length = valid_cache['emb'].shape[0]
    
    ### step 2: 取出数据
    gpt_weights = torch.load(os.path.join(gpt_path, gpt_ckpt), map_location='cpu') # (N_vocab, n_embd), not pinned
    vocab_emb   = F.normalize(gpt_weights['token_embedding_table.weight'], dim=-1) # (N_valid, n_embd), not pinned
    train_emb   = train_cache['emb']                                               # (N_train, n_embd), pinned memory
    train_y     = train_cache['y']                                                 # (N_train, ),       pinned memory
    valid_emb   = valid_cache['emb']                                               # (N_valid, n_embd), pinned memory
    valid_y     = valid_cache['y']                                                 # (N_valid, ),       pinned memory
    
    ### step 2: 初始化 GPT 和 CTE
    emb_size    = N_train + N_vocab + N_valid
    cte         = CritiGraph()
    cte         = torch.compile(cte)
    cte         .eval()
    
    ### step 3: 开始训练并同时测试
    #### step 3.1: 直接测 GPT 就好, 这里是在 cpu 上的，不要占 cuda 内存
    train_logits_eu = 20. * train_cache['emb'] @ vocab_emb.t() # (train_length, n_embd) @ (n_embd, vocab_size) -> (train_length, vocab_size)
    train_loss_eu   = F.cross_entropy(train_logits_eu.view(-1, train_logits_eu.size(-1)), train_cache['y'], reduction='mean')
    valid_logits_eu = 20. * valid_cache['emb'] @ vocab_emb.t() # (eval_length, n_embd) @ (n_embd, vocab_size) -> (eval_length, vocab_size)
    valid_loss_eu   = F.cross_entropy(valid_logits_eu.view(-1, valid_logits_eu.size(-1)), valid_cache['y'], reduction='mean')

    train_pred = train_logits_eu.argmax(dim=-1)            # (train_length,)
    train_acc = (train_pred == train_cache['y']).float().mean().item()
    valid_pred = valid_logits_eu.argmax(dim=-1)              # (eval_length,)
    valid_acc = (valid_pred == valid_cache['y']).float().mean().item()

    print(f"Before CTE Training: train eu loss: {train_loss_eu.item()}, eval eu loss: {valid_loss_eu.item()}")
    print(f"Before CTE Training: train eu acc: {train_acc}, eval eu acc: {valid_acc}")

    #### step 3.2: CTE 训练与测试
    if not valid_only:
        cte.train_all(
            train_emb, vocab_emb, train_y
        )
    if not train_only:
        cte.test_time_train_all(
            train_emb, valid_emb, vocab_emb, valid_y,
            f"./ckpt/cte/locations_h{h}_tp{tp}_N_train{N_train}_N_vocab{N_vocab}_N_valid{N_valid}_epoch{train_epoch_num}.pt"
        )


        



if __name__ == "__main__":
    gpt_ckpt = f"normed_b{block_size}_" + "iters_{}.pth".format(3000)
    # cte_ckpt = f"b_{block_size}" + "gpt_iters_2999_cte_iters_{}.pth"
    # cache_ckpt = gpt_ckpt.replace(".pth", "") + "_l{}_{}_cache_fixed256.pth"
    cache_ckpt = gpt_ckpt.replace(".pth", "") + "_l{}_{}_cache_onlylast.pth"


    # train_gpt(gpt_ckpt)
    # for iters in [0, 1000, 2000, 3000, 4000]:
    # eval_gpt(gpt_ckpt)
    
    # get_cte_train_and_test(gpt_ckpt, cache_ckpt)
    
    train_cte(cache_ckpt, gpt_ckpt, N_train, N_valid)
    
    
    # validate_cte(
    #     os.path.join(gpt_path, gpt_ckpt),
    #     os.path.join(cte_path, cte_ckpt.format(0)),
    #     os.path.join(train_cache_path, train_cache_ckpt.format(0))
    # )
                 

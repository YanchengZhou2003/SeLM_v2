import json
import os
import random
import warnings
from datetime import datetime

import matplotlib.ticker as ticker
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

from src.cte   import *
from src.para  import *
from src.utils import *
from src.vis   import *

## ========= 数据加载：更多 Tokens =========
if N_vocab > 65:
    from src.tokenizer import MultilingualBPETokenizer

    tokenizer = MultilingualBPETokenizer(vocab_size=N_vocab)
    tokenizer.load("./tokenizer")   # 已经训练好并保存过

    ## ====== 数据加载 ======
    ## 注意：这里 text 是完整语料字符串
    with open('./data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    data_ids = tokenizer.encoding(text)              # 列表[int]
    data = torch.tensor(data_ids, dtype=torch.long, pin_memory=True)

    text_size = len(data)
    n = int(0.9 * text_size)
    train_data = data[:n]
    val_data = data[n:]
elif N_vocab == 65:
    ## ====== 数据加载: 65 Tokens ======
    with open('./data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)    

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # ====== Train and test splits ======
    data = torch.tensor(encode(text), dtype=torch.long, pin_memory=True)
    text_size = len(data)
    datai = torch.tensor([i * block_size for i in range(text_size)], dtype=torch.long)
    n = int(0.9 * text_size) # first 90% will be train, rest val
    train_data = data[:n]
    val_data   = data[n:]
else:
    raise ValueError("N_vocab must be either 65 or greater than 65.")



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
        x, y, ix = x.to(main_device), y.to(main_device), ix.to(main_device),
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
    
    def forward(self, idx, targets, return_dyn_loss=False):
        B    = idx.size(0)
        T, V = block_size, N_vocab
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,E)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,E)
        x = tok_emb # + pos_emb # (B,T,E)
        x = self.blocks(x) # (B,T,E)
        x: torch.Tensor = self.ln_f(x) # (B,T,E)
        token_embeddings = self.token_embedding_table.weight  # (V, E)
        if use_eu_norm:
            # print("Using euclidean norm for logits calculation")
            logits_eu = torch.matmul(x, token_embeddings.t()) # (B, T, V)
        else:
            logits_eu = 20. * torch.matmul(F.normalize(x, dim=-1), F.normalize(token_embeddings, dim=-1).t()) # (B, T, V)
        
        if return_dyn_loss:
            if use_eu_norm:
                dyn_emb = x  # (B, T, E)
            else:
                dyn_emb = F.normalize(x, dim=-1)  # (B, T, E)
            loss    = F.cross_entropy(logits_eu.view(B * T, V), targets.view(B * T), reduction='none') # (B * T,)
            acc     = (logits_eu.argmax(dim=-1) == targets)                                            # (B, T)
            return loss.view(B, T), dyn_emb, acc, logits_eu.argmax(dim=-1)

        loss_eu = F.cross_entropy(logits_eu.view(B * T, V), targets.view(B * T))
        return loss_eu




################ ------------- GPT 训练与评估 ------------- ################
'''
def get_norm_distribution(model: GPTLanguageModel, iter: int):
    import matplotlib.ticker as ticker
    all_norms = []
    all_logits = []
    model.eval()
    for _ in tqdm(range(100)):
        xb, yb, _ = get_batch('train', to_cuda=True)
        loss, dyn_emb, acc, logits = model(xb, targets=yb, return_dyn_loss=True)  # (B, T), (B, T, E), scalar, (B, T, V)

        # 计算 dyn_emb 的 L2 norm
        norms = torch.norm(dyn_emb, dim=-1)   # (B, T)
        all_norms.append(norms.detach().cpu())

        # 收集 logits
        all_logits.append(logits.detach().cpu())

    # ===== dyn_emb norm 分布 =====
    all_norms = torch.cat([n.reshape(-1) for n in all_norms], dim=0)
    plt.figure(figsize=(6,4))
    plt.hist(all_norms.numpy(), bins=50, density=True, alpha=0.7)
    plt.xlabel("L2 Norm")
    plt.ylabel("Density")
    plt.title("Distribution of dyn_emb norms")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
    plt.savefig(f"gpt_dyn_emb_norm_dist_iter_{iter}.png")

    # ===== token embedding norm 分布 =====
    token_emb = model.token_embedding_table.weight.detach().cpu()  # (V, E)
    token_norms = torch.norm(token_emb, dim=-1)  # (V,)
    plt.figure(figsize=(6,4))
    plt.hist(token_norms.numpy(), bins=50, density=True, alpha=0.7, color="orange")
    plt.xlabel("L2 Norm")
    plt.ylabel("Density")
    plt.title("Distribution of token embedding norms")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
    plt.savefig(f"gpt_token_emb_norm_dist_iter_{iter}.png")

    # ===== logits 分布 =====
    all_logits = torch.cat([l.reshape(-1) for l in all_logits], dim=0)
    plt.figure(figsize=(6,4))
    plt.hist(all_logits.numpy(), bins=100, density=True, alpha=0.7, color="green")
    plt.xlabel("Logit value")
    plt.ylabel("Density")
    plt.title("Distribution of logits")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
    plt.savefig(f"gpt_logits_dist_iter_{iter}.png")

    model.train()
'''

def train_gpt(gpt_ckpt: str):
    model: GPTLanguageModel = GPTLanguageModel().to(main_device)
    # get_norm_distribution(model, -1)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
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
            # get_norm_distribution(model, iter)
            print(f"current iter: {iter}, avg loss in last {gpt_save_interval} iters: {sum(running_loss) / len(running_loss)}")
            running_loss = []
            torch.save(model.state_dict(), os.path.join(gpt_path, gpt_ckpt.format(iter)))

def eval_gpt(gpt_ckpt: str):
    model: GPTLanguageModel = GPTLanguageModel()
    model_cktp = torch.load(os.path.join(gpt_path, gpt_ckpt), map_location='cpu')
    model.load_state_dict(model_cktp)
    model = model.to(main_device)
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
    model = model.to(main_device)
    model.eval()
    
    ### step 1: 准备基本数据
    train_y, train_emb = [], []
    eval_y, eval_emb = [], []
    cache_save_path = os.path.join(cache_path, cache_save_path)
    # train_last_token_loss = []
    # valid_last_token_loss = []
    # train_last_token_acc  = []
    # valid_last_token_acc  = []
    
    ### step 2: 迭代获取动态嵌入（dyn_emb）与数据元信息（ix）
    for _ in range(cte_train_iters):
        X_train, Y_train, _ = get_batch('train', bs=cte_train_bs, to_cuda=True)
        loss, dyn_emb, acc, _ = model(X_train, targets=Y_train, return_dyn_loss=True) # (B, T), (B, T, E)
        train_y.append(Y_train[:, -1].cpu())
        train_emb.append(dyn_emb[:, -1, :].cpu())
        # train_last_token_loss.append(loss[:, -1].cpu()) # (B,)
        # train_last_token_acc.append(acc[:, -1].cpu())   # (B,)
    
    for _ in range(cte_eval_iters):
        X_val, Y_val, _ = get_batch('val', bs=cte_eval_bs, to_cuda=True)
        loss, dyn_emb, acc, _ = model(X_val, targets=Y_val, return_dyn_loss=True) # (B, T), (B, T, E)
        eval_y.append(Y_val[:, -1].cpu())
        eval_emb.append(dyn_emb[:, -1, :].cpu())
        # valid_last_token_loss.append(loss[:, -1].cpu()) # (B,)
        # valid_last_token_acc.append(acc[:, -1].cpu())   # (B,)
    
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
    
    ### step 4: 可视化 loss 的分布
    # train_last_token_loss = torch.cat(train_last_token_loss, dim=0).numpy()
    # valid_last_token_loss = torch.cat(valid_last_token_loss, dim=0).numpy()
    
    # fig, ax1 = plt.subplots()
    
    # # Train loss histogram on left y-axis
    # ax1.hist(train_last_token_loss, bins=50, alpha=0.5, color='blue', label='Train Last Token Loss')
    # ax1.set_xlim((0, 1.5))
    # ax1.set_xlabel('Loss')
    # ax1.set_ylabel('Train Frequency', color='blue')
    # ax1.tick_params(axis='y', labelcolor='blue')
    
    # # Valid loss histogram on right y-axis
    # ax2 = ax1.twinx()
    # ax2.hist(valid_last_token_loss, bins=50, alpha=0.5, color='red', label='Valid Last Token Loss')
    # ax2.set_ylabel('Valid Frequency', color='red')
    # ax2.tick_params(axis='y', labelcolor='red')
    
    # plt.title('Distribution of Last Token Loss')
    # fig.tight_layout()
    # plt.savefig(f'last_token_loss_distribution_ori{train_last_token_loss.shape[0]}_cur{train_cache_length}.png')
    # plt.close()
    
    print(f"train cache length: {train_cache_length}, eval cache length: {eval_cache_length}")
    save_file(train_cache, cache_save_path.format(train_cache_length, 'train'))
    save_file(eval_cache, cache_save_path.format(eval_cache_length, 'val'))

@torch.no_grad()
def get_cte_train_and_test_by_ratio(gpt_ckpt: str, train_cache_path: str, valid_cache_path: str):
    ### step 0: 准备模型
    model: GPTLanguageModel = GPTLanguageModel()
    model_cktp = torch.load(os.path.join(gpt_path, gpt_ckpt), map_location='cpu')
    model.load_state_dict(model_cktp)
    model = model.to(main_device)
    model.eval()
    
    ### step 1: 准备基本数据
    train_y_pos, train_emb_pos = [], []
    train_y_neg, train_emb_neg = [], []
    
    eval_y, eval_emb = [], []
    
    ### step 2: 迭代获取动态嵌入（dyn_emb）与数据元信息（ix）
    pos_num, neg_num = 0, 0
    pos_tar_num, neg_tar_num = int(round(N_train * pos_ratio)), int(round(N_train * (1 - pos_ratio)))
    
    while True:
        X_train, Y_train, _ = get_batch('train', to_cuda=True)
        loss, dyn_emb, acc, prediction = model(X_train, targets=Y_train, return_dyn_loss=True) # (B, T), (B, T, E)
        
        pos_indices = (prediction[:, -1] == Y_train[:, -1]) # (B,)
        neg_indices = (prediction[:, -1] != Y_train[:, -1]) # (B,)
        
        if pos_num < pos_tar_num:
            train_y_pos.append(Y_train[pos_indices, -1].cpu())
            train_emb_pos.append(dyn_emb[pos_indices, -1, :].cpu())
            pos_num += pos_indices.sum().item()
            print(f"pos_num: {pos_num} / {pos_tar_num}")
        if neg_num < neg_tar_num:
            train_y_neg.append(Y_train[neg_indices, -1].cpu())
            train_emb_neg.append(dyn_emb[neg_indices, -1, :].cpu())
            neg_num += neg_indices.sum().item()
            print(f"neg_num: {neg_num} / {neg_tar_num}")
        
        if pos_num >= pos_tar_num and neg_num >= neg_tar_num:
            break
        
    eval_num = 0
    eval_tar = N_valid
    
    while True:
        X_val, Y_val, _ = get_batch('val', to_cuda=True)
        loss, dyn_emb, acc, _ = model(X_val, targets=Y_val, return_dyn_loss=True) # (B, T), (B, T, E)
        eval_y.append(Y_val[:, -1].cpu())
        eval_emb.append(dyn_emb[:, -1, :].cpu())
        
        eval_num += Y_val.shape[0]
        print(f"eval_num: {eval_num} / {eval_tar}")
        if eval_num >= eval_tar:
            break
    
    ### step 3: 拼接并保存
    train_y = torch.cat(train_y_pos + train_y_neg, dim=0) 
    train_y = train_y[:N_train]
    train_emb = torch.cat(train_emb_pos + train_emb_neg, dim=0)
    train_emb = train_emb[:N_train]
    eval_y = torch.cat(eval_y, dim=0)[:N_valid]
    eval_emb = torch.cat(eval_emb, dim=0)[:N_valid]
    
    train_cache = {
        'y': train_y,            # (N_train)
        'emb': train_emb         # (N_train, n_embd)
    }
    eval_cache = {
        'y': eval_y,             # (N_valid)
        'emb': eval_emb          # (N_valid, n_embd)
    }
    train_cache_length = train_cache['emb'].shape[0] 
    eval_cache_length = eval_cache['emb'].shape[0]
    
    ###################################################################################
    
    
    ### step 4: 可视化输出层熵 + bin accuracy（train vs valid）

    import numpy as np
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        # logits: (B, V)
        prob = torch.softmax(logits, dim=-1)
        entropy = -(prob * prob.log()).sum(dim=-1)   # (B,)
        return entropy.cpu()

    # ======================================
    # 工具函数：分 bin 统计 accuracy，并绘制
    # ======================================
    def plot_entropy_with_accuracy(entropies, correctness, filename, title, color):
        """
        entropies:   (N,) numpy
        correctness: (N,) numpy，0/1
        """
        plt.figure(figsize=(8, 5))
        
        # 绘图并得到 bins
        counts, bin_edges, patches = plt.hist(
            entropies, bins=15, alpha=0.5, density=True, label=title, color=color
        )

        # 每个 bin 的 accuracy
        bin_acc = []
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (entropies >= lo) & (entropies < hi)
            if mask.sum() > 0:
                acc = correctness[mask].mean()
            else:
                acc = float('nan')
            bin_acc.append(acc)

        # 在柱子顶部标注 accuracy
        for i, patch in enumerate(patches):
            acc = bin_acc[i]
            if not np.isnan(acc):
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_height()
                plt.text(
                    x, y,
                    f"{acc:.2f}",
                    ha='center', va='bottom', fontsize=8
                )

        plt.xlabel("Entropy of output distribution")
        plt.ylabel("Density")
        plt.title(title)
        plt.tight_layout()
        plt.ylim(0.0, 0.85)
        plt.savefig(filename)
        plt.close()

    # ======================================
    # 重新迭代 train/valid，收集 entropy + correctness
    # ======================================

    train_entropy_list = []
    train_correct_list = []

    train_num = 0
    while train_num < train_cache_length:
        X_train, Y_train, _ = get_batch('train', to_cuda=True)
        with torch.no_grad():
            # dyn_emb
            _, dyn_emb, _, _ = model(X_train, targets=Y_train, return_dyn_loss=True)
            dyn_emb = dyn_emb[:, -1, :]     # (B, E)
            sta_emb = model.token_embedding_table.weight   # (V, E)

            # logits
            logits = 20. * F.normalize(dyn_emb, dim=-1) @ F.normalize(sta_emb, dim=-1).t()  # (B, V)

            # entropy
            H = compute_entropy(logits)                   # (B,)
            train_entropy_list.append(H)

            # correctness
            pred = logits.argmax(dim=-1).cpu()            # (B,)
            correct = (pred == Y_train[:, -1].cpu()).float()
            train_correct_list.append(correct)

            train_num += H.shape[0]


    valid_entropy_list = []
    valid_correct_list = []

    valid_num = 0
    while valid_num < eval_cache_length:
        X_val, Y_val, _ = get_batch('val', to_cuda=True)
        with torch.no_grad():
            _, dyn_emb, _, _ = model(X_val, targets=Y_val, return_dyn_loss=True)
            dyn_emb = dyn_emb[:, -1, :]
            sta_emb = model.token_embedding_table.weight

            logits = 20. * F.normalize(dyn_emb, dim=-1) @ F.normalize(sta_emb, dim=-1).t()

            H = compute_entropy(logits)
            valid_entropy_list.append(H)

            pred = logits.argmax(dim=-1).cpu()
            correct = (pred == Y_val[:, -1].cpu()).float()
            valid_correct_list.append(correct)

            valid_num += H.shape[0]


    # ======================================
    # 拼接为 numpy
    # ======================================
    train_entropy = torch.cat(train_entropy_list, dim=0)[:train_cache_length].numpy()
    valid_entropy = torch.cat(valid_entropy_list, dim=0)[:eval_cache_length].numpy()

    train_correctness = torch.cat(train_correct_list, dim=0)[:train_cache_length].numpy()
    valid_correctness = torch.cat(valid_correct_list, dim=0)[:eval_cache_length].numpy()

    # ======================================
    # 绘图：train + valid
    # ======================================
    plot_entropy_with_accuracy(
        train_entropy, train_correctness,
        filename=f'entropy_distribution_train{train_cache_length}.png',
        title="Output-layer entropy distribution (train)",
        color='blue'
    )

    plot_entropy_with_accuracy(
        valid_entropy, valid_correctness,
        filename=f'entropy_distribution_valid{eval_cache_length}.png',
        title="Output-layer entropy distribution (valid)",
        color='red'
    )

    
    
    
    #############################################################################
    
    print(f"train cache length: {train_cache_length}, eval cache length: {eval_cache_length}")
    save_file(train_cache, os.path.join(cache_path, train_cache_path))
    save_file(eval_cache, os.path.join(cache_path, valid_cache_path))


################ ------------- 训练 CTE ------------- ################

@torch.no_grad()
def main_cte(
    train_cache_cktp: str, 
    valid_cache_cktp: str,
    gpt_ckpt: str, 
    train_length: int, 
    val_length: int,
):
    ### step 1: 读取缓存，并 pin 在 cpu。根据计算，10 ** 7 时也仅占用 15 GB
    assert math.log2(train_length) % 1 == 0, "train_length 必须是 2 的整数次幂"
    assert math.log2(val_length) % 1 == 0, "val_length 必须是 2 的整数次幂"
    
    
    train_cache = load_file(os.path.join(cache_path, train_cache_cktp), device='cpu')
    valid_cache = load_file(os.path.join(cache_path, valid_cache_cktp), device='cpu')
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
    train_top   = train_cache['rk']
    valid_emb   = valid_cache['emb']                                               # (N_valid, n_embd), pinned memory
    valid_y     = valid_cache['y']                                                 # (N_valid, ),       pinned memory
    valid_top   = valid_cache['rk']
    
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

    # if not valid_only and not os.path.exists(train_save_path.replace(".pt", f"_epoch{train_epoch_num}.pt")):
    #     print("Not Found trained CTE model, start training...")
    cte.train_all(
        train_emb, train_top, vocab_emb, train_y
    )
    if not train_only:
        for i in range(train_epoch_num // save_interval):
            train_epoch = (i + 1) * save_interval
            print(f"CTE Training: Starting validation after {train_epoch} epochs...")
            cte.valid_all(
                train_emb, valid_emb, valid_top, vocab_emb, valid_y, train_epoch
            )


        
if __name__ == "__main__":
    print("Starting GPT training and evaluation...")
    print("Current Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    gpt_ckpt = f"voc{N_vocab}_normfixed20_b{block_size}" + "_iters_{}.pth" # .format(2999)
    # gpt_ckpt = f"tmp_" + "iters_{}.pth" # .format(2999)
    # train_gpt(gpt_ckpt)
    # for iters in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]:
    #     print(f"Evaluating GPT at iters={iters}")
    #     eval_gpt(gpt_ckpt.format(iters))
    
    best_valid_epochs = {
        65 :  9999,
        384:  3000,
        512:  2000,
        768:  2000,
        1024: 2000
    }
    best_valid_epoch = best_valid_epochs.get(N_vocab, None)
    if best_valid_epoch is None:
        raise ValueError(f"No best valid epoch found for N_vocab={N_vocab}")
    gpt_ckpt = gpt_ckpt.format(best_valid_epoch)
    
    # train_cache_ckpt = gpt_ckpt.replace(".pth", "") + f"_ps{pos_ratio}_train{N_train}_cache_last.pth"
    # valid_cache_ckpt = gpt_ckpt.replace(".pth", "") + f"_valid{N_valid}_cache_last.pth"
    # get_cte_train_and_test_by_ratio(gpt_ckpt, train_cache_ckpt, valid_cache_ckpt)
    
    train_cache_ckpt = f"rk{N_top}_" + gpt_ckpt.replace(".pth", "") + f"_train{N_train}_cache_last.pth"
    valid_cache_ckpt = f"rk{N_top_v}_" + f"q=train{N_train}_" + gpt_ckpt.replace(".pth", "") + f"_valid{N_valid}_cache_last.pth"
    try:
        main_cte(train_cache_ckpt, valid_cache_ckpt, gpt_ckpt, N_train, N_valid)
    except Exception as e:
        print("An error occurred during CTE training/evaluation:", str(e))
        traceback.print_exc()
        os._exit(1)
    
    print("Finished GPT training and evaluation.")
    print("Current Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                 

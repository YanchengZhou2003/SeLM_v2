import json
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from torch.nn import functional as F

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
datai = torch.tensor([i * block_size for i in range(text_size)], dtype=torch.long, pin_memory=True)
n = int(0.9*text_size) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
train_datai = datai[:n]
val_datai = datai[n:]

da = (torch.arange(vocab_size) + text_size * block_size).unsqueeze(0).expand(batch_size, -1) # (B, vocab_size)

# Train Cache
_train_cache = []

# data loading
def get_batch(split, ix=None):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    datai = train_datai if split == 'train' else val_datai
    if ix is None:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    xi = torch.stack([datai[i:i+block_size] for i in ix])
    xi_pad = torch.arange(0, block_size).unsqueeze(0)
    xi = xi + xi_pad
    
    xi = torch.cat((xi, da[0:1].repeat(xi.size(0), 1)), dim=1)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, xi, y = x.to(device), xi.to(device), y.to(device)
    return x, xi, y, ix

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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
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

    def forward(self, idx, targets, return_dyn_emb=False):
        B, T, V = batch_size, block_size, vocab_size
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,E)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,E)
        x = tok_emb # + pos_emb # (B,T,E)
        x = self.blocks(x) # (B,T,E)
        x: torch.Tensor = self.ln_f(x) # (B,T,E)
        if return_dyn_emb:
            return x
        
        token_embeddings = self.token_embedding_table.weight  # (V, E)
        logits_eu = torch.matmul(x, token_embeddings.t()) # (B, T, V)
        loss_eu = F.cross_entropy(logits_eu.view(B * T, V), targets.view(B * T))
        
        return loss_eu


@torch.no_grad()
def evaluate(model: GPTLanguageModel):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses_ori = torch.zeros(eval_iters)
        losses_cts = torch.zeros(eval_iters)
        losses_gap = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, XI, Y, _ = get_batch(split)
            _, loss_ori, loss_cts, loss_gap = model(X, XI, targets=Y, evaluation=1)
            losses_ori[k] = loss_ori.item()
            losses_cts[k] = loss_cts.item()
            losses_gap[k] = loss_gap.item()
        out[split] = [losses_ori.mean(), losses_cts.mean(), losses_gap.mean()]
    model.train()
    return out


def train_gpt(gpt_ckpt: str):
    model: GPTLanguageModel = GPTLanguageModel().to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    running_loss = []
    for iter in range(max_iters):
        
        xb, xi, yb, _ = get_batch('train')
        loss = model(xb, xi, targets=yb)
        optimizer.zero_grad(set_to_none=True)
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if iter % gpt_save_interval == 0 or iter == max_iters - 1:
            print(f"current iter: {iter}, avg loss in last {gpt_save_interval} iters: {sum(running_loss) / len(running_loss)}")
            running_loss = []
            torch.save(model.state_dict(), os.path.join(gpt_path, gpt_ckpt.format(iter)))

@torch.no_grad()
def train_cte(gpt_cktp: str, cte_cktp: str, train_cache_cktp: str):
    model: GPTLanguageModel = GPTLanguageModel()
    model_cktp = torch.load(gpt_cktp, map_location='cpu')
    
    load_partial_state_dict(model, model_cktp, skip_substrings=["cte"], strict=False)
    del model.cte
    
    torch.cuda.empty_cache()
    model.cte = CritiGraph(h, tp, c, eps, epoch_cte, batch_size_cte, convergence, text_size * block_size + vocab_size, block_size + vocab_size, division_fact, loss_type)
    
    model = model.to(device)
    
    model.eval()
    for iter in range(max_iters):
        if iter % cte_save_interval == 0 or iter == max_iters - 1: 
            visualization = True
        else:
            visualization = False
        
        xb, xi, yb, ix = get_batch('train')
        _train_cache.append(ix)
        
        _, loss_eu, loss_ct, _, var = model(xb, xi, targets=yb, train_cte=1, visualization=visualization)
        print(f"current train iter: {iter}, loss_eu: {fmt6w(loss_eu.item())}, loss_ct: {loss_ct.item()}")
        
        if visualization:    
            visualize_similarity(model, var, iter)
            torch.save(model.cte.state_dict(), cte_cktp.format(iter))
            torch.save(_train_cache, train_cache_cktp.format(iter))

            


def visualize_similarity(model, var, iter):
    emb_eu = var['emb_normed'][0]
    emb_ct = model.cte.main_locations[var['idi'][0].cpu()]
    print(emb_eu.shape)
    
    # --- 相似度矩阵 (eu) ---
    S = torch.matmul(emb_eu, emb_eu.T).cpu().numpy()
    np.fill_diagonal(S, 1.0)

    # 转为“距离”做聚类
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    dvec = squareform(D, checks=False)

    Z = linkage(dvec, method='average')
    order = leaves_list(Z)
    S_re = S[order][:, order]

    # --- 一张图：左树 + 热图 + 右色条 (eu) ---
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[2.5, 14, 0.5],
        height_ratios=[1.0],
        wspace=0.0, hspace=0.0
    )

    # 左侧行树
    ax_row = fig.add_subplot(gs[0, 0])
    dendrogram(Z, ax=ax_row, orientation="right", no_labels=True, color_threshold=None)
    ax_row.invert_yaxis()
    ax_row.set_xticks([]); ax_row.set_yticks([])

    # 中间热图 (eu)
    ax = fig.add_subplot(gs[0, 1])
    vmin, vmax = S_re.min(), S_re.max()
    norm = PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)
    im = ax.imshow(S_re, cmap="inferno", norm=norm,
                   interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Hierarchical Cosine Similarity (eu)")

    # 右侧颜色条
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("eu-cosine similarity", rotation=270, labelpad=25)

    fig.savefig(sim_eu_path.format(iter), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Similarity + tree visualization saved to {sim_eu_path.format(iter)}")

    # --- 相似度矩阵 (ct) ---
    distance_ct = model.cte.main_distance(
        emb_ct.unsqueeze(1), emb_ct.unsqueeze(0),
        torch.ones((block_size + vocab_size, block_size + vocab_size, 1))
    ).mean(dim=-1).cpu().numpy()
    np.fill_diagonal(distance_ct, 1.0)
    S_ct = distance_ct[order][:, order]

    # --- 一张图：左树 + 热图 + 右色条 (ct) ---
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[2.5, 14, 0.5],
        height_ratios=[1.0],
        wspace=0.0, hspace=0.0
    )

    # 左侧行树 (沿用同一个 Z 顺序)
    ax_row = fig.add_subplot(gs[0, 0])
    dendrogram(Z, ax=ax_row, orientation="right", no_labels=True, color_threshold=None)
    ax_row.invert_yaxis()
    ax_row.set_xticks([]); ax_row.set_yticks([])

    # 中间热图 (ct)
    ax = fig.add_subplot(gs[0, 1])
    vmin, vmax = S_ct.min(), S_ct.max()
    norm = PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)
    im = ax.imshow(S_ct, cmap="inferno", norm=norm,
                   interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Hierarchical Cosine Similarity (ct)")

    # 右侧颜色条
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ct-cosine similarity", rotation=270, labelpad=25)

    fig.savefig(sim_ct_path.format(iter), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Similarity + tree visualization saved to {sim_ct_path.format(iter)}")




if __name__ == "__main__":
    gpt_ckpt = f"b_{block_size}" + "iters_{}.pth"
    # cte_ckpt = f"b_{block_size}" + "gpt_iters_2999_cte_iters_{}.pth"
    # train_cache_ckpt = f"b_{block_size}" + "gpt_iters_2999_cte_iters_{}_train_cache.pth"



    # train_gpt(gpt_ckpt)
    
    # train_cte(os.path.join(gpt_path, gpt_ckpt),
    #           os.path.join(cte_path, cte_ckpt),
    #           os.path.join(train_cache_path, train_cache_ckpt))
    
    
    validate_cte(
        os.path.join(gpt_path, gpt_ckpt),
        os.path.join(cte_path, cte_ckpt.format(0)),
        os.path.join(train_cache_path, train_cache_ckpt.format(0))
    )
                 

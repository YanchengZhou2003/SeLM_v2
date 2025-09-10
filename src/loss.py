import math
from typing import Literal, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


def js_div(
    input,                 # P：log-prob 或 prob
    target,                # Q：log-prob 或 prob
    log_input: bool = True,
    log_target: bool = False,
    reduction: str = 'none',  # 与 F.kl_div 一致的默认
    eps: float = 1e-12
):
    """
    返回与 input/target 同形的逐元素 JS 的“密度项”，
    需要调用方沿分布维 sum（与 F.kl_div 的使用一致）。
    说明：这里的“逐元素”指把 KL 的 integrand 展开到分布维上。
    """
    # 统一为 log 概率
    if log_input:
        logP = input
    else:
        P = input.clamp_min(eps)
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(eps)
        logP = P.log()

    if log_target:
        logQ = target
    else:
        Q = target.clamp_min(eps)
        Q = Q / Q.sum(dim=-1, keepdim=True).clamp_min(eps)
        logQ = Q.log()

    # logM = log(0.5*P + 0.5*Q)
    logM = torch.logaddexp(logP, logQ) - math.log(2.0)

    # 逐元素 KL integrand：P*(logP-logM) 与 Q*(logQ-logM)
    P = torch.exp(logP)
    Q = torch.exp(logQ)
    elem_kl_p_m = P * (logP - logM)
    elem_kl_q_m = Q * (logQ - logM)

    # 逐元素 JS integrand 的和（还没沿分布维 sum）
    elem_js = 0.5 * (elem_kl_p_m + elem_kl_q_m)  # 形状与 input 相同

    if reduction == 'none':
        return elem_js
    elif reduction == 'sum':
        return elem_js.sum()
    elif reduction == 'mean':
        return elem_js.mean()
    elif reduction == 'batchmean':
        # 约定 batch 在 0 维；与 F.kl_div 相同语义：先总和，再除以 batch 大小
        return elem_js.sum() / max(elem_js.size(0), 1)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
    

def compute_loss(
    kind: str,
    ct_val: torch.Tensor,
    eu_val: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sum_dim: int = 1,
) -> torch.Tensor:
    """
    计算 loss：
    - 'square' / 'abs': 回归型，先按 T2 维度求和后再 / lth
    - 'kl' / 'js': 概率型，对 log_softmax 后的分布做散度，按 T2 维度求和
    返回：(B, T1, C, tp)
    """
    if mask is None:
        mask = torch.ones_like(ct_val)

    if kind == 'square':
        loss = torch.square(ct_val - eu_val)          # (T1, T2, C, tp)
        loss = loss * mask
        loss = loss.sum(dim=sum_dim)                  # (T1, C, tp)
        return loss
    
    if kind == 'lap':
        loss = torch.square(ct_val - eu_val) * torch.abs(eu_val)          # (T1, T2, C, tp)
        loss = loss * mask
        loss = loss.sum(dim=sum_dim)                  # (T1, C, tp)
        return loss        

    if kind == 'abs':
        loss = torch.abs(ct_val - eu_val)             # (T1, T2, C, tp)
        loss = loss * mask
        loss = loss.sum(dim=sum_dim)                  # (T1, C, tp)
        return loss

    # 概率型：对类别维做 softmax，再计算散度
    log_ct = F.log_softmax(ct_val, dim=sum_dim)
    log_eu = F.log_softmax(eu_val, dim=sum_dim)

    if kind == 'kl':
        loss = F.kl_div(
            log_ct, log_eu,
            log_target=True,
            reduction='none'
        )                                             # (T1, T2, C, tp)
    elif kind == 'js':
        loss = js_div(                                # type: ignore[name-defined]
            log_ct, log_eu,
            log_target=True,
            reduction='none'
        )                                             # (T1, T2, C, tp)
    else:
        raise ValueError(f'Unsupported loss kind: {kind}')

    loss = loss * mask
    loss = loss.sum(dim=sum_dim)                      # (T1, C, tp)
    return loss



def compute_weighted_loss(
    loss_type : Tuple[str, str],
    weight: Tuple[float, float],
    ct_val: torch.Tensor,
    eu_val: torch.Tensor,
    mask: torch.Tensor,
    lth:  torch.Tensor,
    S: int, S_: int,
    sum_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 loss：
    loss_type[0]: 为回归型，所以我们要对 sum_dim 求 mean; 这里会用到 lth
    loss_type[1]: 为概率型，所以我们要对 sum_dim 求 sum
    """
    ct_val_cos = ct_val[:, :S, ...]           # (subT, S, C, D)
    eu_val_cos = eu_val[:, :S, ...]           # (subT, S, C, D)
    mask_cos   = mask  [:, :S, ...]           # (subT, S, C, D)
    lth_cos    = lth   [:, None, None]        # (subT, 1, 1)
    if   loss_type[0] == 'square':
        loss_cos = torch.square(ct_val_cos - eu_val_cos)     
    elif loss_type[0] == 'lap':
        loss_cos = torch.square(ct_val_cos - eu_val_cos) * torch.abs(eu_val_cos)            
    elif loss_type[0] == 'abs':
        loss_cos = torch.abs(ct_val_cos - eu_val_cos)            
    else:
        raise ValueError(f'Unsupported loss kind: {loss_type[0]}')
    loss_cos = loss_cos * mask_cos           # (subT, S, C, D)
    loss_cos = loss_cos.sum(dim=sum_dim)     # (subT, C, D)
    loss_cos = loss_cos / lth_cos            # (subT, C, D) 

    if S_ > S:
        ct_val_cro = ct_val[:, S:, ...]      # (subT, S_ - S, C, D)
        eu_val_cro = eu_val[:, S:, ...]      # (subT, S_ - S, C, D)
        mask_cro   = mask  [:, S:, ...]      # (subT, S_ - S, C, D)
    
        # 概率型：对类别维做 softmax，再计算散度
        log_ct_cro = F.log_softmax(ct_val_cro, dim=sum_dim)
        log_eu_cro = F.log_softmax(eu_val_cro, dim=sum_dim)

        if loss_type[1] == 'kl':
            loss_cro = F.kl_div(
                log_ct_cro, log_eu_cro,
                log_target=True,
                reduction='none'
            )                                # (subT, S_ - S, C, D)
        elif loss_type[1] == 'js':
            loss_cro = js_div(                              
                log_ct_cro, log_eu_cro,
                log_target=True,
                reduction='none'
            )                                # (subT, S_ - S, C, D)
        else:
            raise ValueError(f'Unsupported loss kind: {loss_type[1]}')

        loss_cro = loss_cro * mask_cro
        loss_cro = loss_cro.sum(dim=sum_dim) # (subT, C, D)
    else:
        loss_cro = torch.zeros_like(loss_cos)
    
    loss_tot     = weight[0] * loss_cos + weight[1] * loss_cro  # (subT, C, D)
    
    return loss_cos, loss_cro, loss_tot



def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1,)):
    # logits: [B, C] 或 [*, C]; targets: [B] 或 [*]
    assert logits.shape[:-1] == targets.shape
    maxk = max(ks)
    topk_idx = logits.topk(maxk, dim=-1).indices            # [..., maxk]
    t = targets.unsqueeze(-1).expand_as(topk_idx)           # [..., maxk]
    correct = (topk_idx == t)                               # [..., maxk]
    res = {}
    for k in ks:
        hits = correct[..., :k].any(dim=-1).float().sum().item()
        total = targets.numel()
        res[k] = hits / total
    return res
from typing import Literal, Mapping, Optional
import torch
import torch.nn.functional as F
import math


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
    lth: torch.Tensor | float | int,
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
        loss = loss.sum(dim=sum_dim) / lth            # (T1, C, tp)
        return loss
    
    if kind == 'lap':
        loss = torch.square(ct_val - eu_val) * torch.abs(eu_val)          # (T1, T2, C, tp)
        loss = loss * mask
        loss = loss.sum(dim=sum_dim) / lth            # (T1, C, tp)
        return loss        

    if kind == 'abs':
        loss = torch.abs(ct_val - eu_val)             # (T1, T2, C, tp)
        loss = loss * mask
        loss = loss.sum(dim=sum_dim) / lth            # (T1, C, tp)
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
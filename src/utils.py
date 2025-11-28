import functools
import inspect
import math
import time
from collections import abc
from contextlib import ContextDecorator
from typing import (Any, Callable, Dict, Iterable, List, Literal, Mapping,
                    MutableMapping, Optional, Sequence, Tuple, TypedDict,
                    Union)

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")  # 使用无界面后端，避免在服务器/笔记本上弹窗  
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import is_valid_y, squareform

Mark = Callable[[str], None]
_cache = {}
# 单个阶段配置

from typing import Literal, Mapping, TypedDict, Union


# 仅目标 + 收敛
class PhaseConfigBase(TypedDict):
    target: Literal['dyn_only', 'sta_only', 'alternated', 'prob_only', 'TTT_only']
    converge: int

# 带权重的阶段（用于 weighted_dyn_prob）
class PhaseConfigWeighted(TypedDict):
    target: Literal['weighted_dyn_prob']
    converge: int
    ratio_dyn: float
    ratio_prob: float

PhaseConfig = Union[PhaseConfigBase, PhaseConfigWeighted]

# 顶层字典的值类型：三类字符串 loss 的值 + 阶段配置
TopValue = Union[
    Literal['square'],  # dyn_loss, sta_loss 取 'square'
    Literal['abs'],  # dyn_loss, sta_loss 取 'square'
    Literal['kl'],      # prob_loss 取 'kl'
    Literal['js'],      # prob_loss 取 'js'
    PhaseConfig         # 整数键对应的阶段配置
]

# 顶层键类型：字符串键 或 整数键
TopKey = Union[
    Literal['dyn_loss', 'sta_loss', 'prob_loss'],
    int  # 阶段边界 epoch
]

LossTypeDict = Mapping[TopKey, TopValue]


def timeit(
    name: Optional[str] = None,
    *,
    unit: Literal["s", "ms", "us"] = "ms",
    clock: Literal["perf", "process"] = "perf",
    printer: Callable[[str], None] = print,
    show_deltas: bool = True,    # 显示相邻标记的间隔
    show_total: bool = True,     # 显示总耗时
):
    """
    装饰器：被装饰函数内部可调用 mark(label) 记录时间点，返回时统一输出时间线。
    使用方法：
        @timeline("fit-epoch")
        def train(..., mark=None):
            ...; mark("load"); ...; mark("forward"); ...; mark("backward")
    """
    get_time = time.perf_counter if clock == "perf" else time.process_time

    def _fmt(sec: float) -> str:
        if unit == "s":
            return f"{sec:.6f}s"
        if unit == "ms":
            return f"{sec*1e3:.3f}ms"
        if unit == "us":
            return f"{sec*1e6:.0f}µs"
        return f"{sec:.6f}s"

    def decorator(func):
        disp = name or f"{func.__module__}.{func.__qualname__}"

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                marks: List[Tuple[str, float]] = []
                t0 = get_time()

                def mark(label: str):
                    marks.append((label, get_time()))

                # 将 mark 注入到函数参数里（若用户没显式传入）
                if "mark" not in kwargs:
                    kwargs["mark"] = mark

                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    t_end = get_time()
                    # 组装输出
                    lines = [f"[timeline] {disp}"]
                    prev = t0
                    for lbl, tt in marks:
                        delta = tt - prev
                        since0 = tt - t0
                        if show_deltas:
                            lines.append(f"  - {lbl}: +{_fmt(delta)} (t={_fmt(since0)})")
                        else:
                            lines.append(f"  - {lbl}: t={_fmt(since0)}")
                        prev = tt
                    if show_total:
                        lines.append(f"  = total: {_fmt(t_end - t0)}")
                    printer("\n".join(lines))
            return awrapper
        else:
            @functools.wraps(func)
            def swrapper(*args, **kwargs):
                marks: List[Tuple[str, float]] = []
                t0 = get_time()

                def mark(label: str):
                    marks.append((label, get_time()))

                if "mark" not in kwargs:
                    kwargs["mark"] = mark

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    t_end = get_time()
                    lines = [f"[timeline] {disp}"]
                    prev = t0
                    for lbl, tt in marks:
                        delta = tt - prev
                        since0 = tt - t0
                        if show_deltas:
                            lines.append(f"  - {lbl}: +{_fmt(delta)} (t={_fmt(since0)})")
                        else:
                            lines.append(f"  - {lbl}: t={_fmt(since0)}")
                        prev = tt
                    if show_total:
                        lines.append(f"  = total: {_fmt(t_end - t0)}")
                    printer("\n".join(lines))
            return swrapper

    return decorator


def named(**tensors: torch.Tensor) -> Dict[str, torch.Tensor]:
    return tensors

def pinned_copy_by_name(named_tensors: dict[str, torch.Tensor], _cache: dict[str, torch.Tensor] = {}):
    out: dict[str, torch.Tensor] = {}
    for name, t in named_tensors.items():
        # 若规格改变则重建
        if (name not in _cache or
            _cache[name].shape != t.shape or
            _cache[name].dtype != t.dtype):
            _cache[name] = torch.empty_like(t, device="cpu", pin_memory=True)
        dst = _cache[name]
        dst.copy_(t, non_blocking=True)
        out[name] = dst
    return out



def batch_concat(
    base: torch.Tensor,          # (bs1, A, B, ...)
    small: torch.Tensor,         # (bs2, a, b, ...)
    pad_value: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 base 与 small 合并为 (bs1+bs2, A, B, ...)，其中 small 的每个样本
    左对齐放入 (A,B,...) 的画布，右/下/后侧用 pad_value 填充。
    返回 (y, mask)，mask 为 bool，True 表示有效原数据位置（方案 B）。

    约束：
      - base.ndim == small.ndim
      - base.shape[1:] = (A,B,...)，small.shape[1:] = (a,b,...)
      - 且每维 A>=a, B>=b, ...
      - dtype 与 device 以 base 为准（small 会被拷贝到相同 device/dtype）
    """
    if base.ndim != small.ndim:
        raise ValueError(f"rank mismatch: base.ndim={base.ndim}, small.ndim={small.ndim}")
    if base.ndim < 2:
        raise ValueError("expect at least 2 dims: (batch, ...)")
    bs1, *big_shape = base.shape
    bs2, *small_shape = small.shape

    if any(B < S for B, S in zip(big_shape, small_shape)):
        raise ValueError(f"target dims must be >= small dims, got {big_shape} vs {small_shape}")

    # 统一 dtype/device 到 base
    if small.dtype != base.dtype or small.device != base.device:
        small = small.to(dtype=base.dtype, device=base.device)

    # 分配输出与 mask（方案B：有效为 True）
    out_shape = (bs1 + bs2, *big_shape)
    y = torch.full(out_shape, pad_value, dtype=base.dtype, device=base.device)
    mask = torch.zeros(out_shape, dtype=torch.bool, device=base.device)

    # 1) 复制 base 批
    y[:bs1] = base
    mask[:bs1] = True  # base 全部为有效

    # 2) 将 small 批逐样本放入左对齐画布
    # 写入区域：[:bs2, :a, :b, ...]
    write_slices = [slice(bs1, bs1 + bs2)]
    valid_slices = [slice(0, bs2)]
    for S in small_shape:
        write_slices.append(slice(0, S))
        valid_slices.append(slice(0, S))
    write_slices = tuple(write_slices)
    valid_slices = tuple(valid_slices)

    # 放置数据
    y[write_slices] = small
    mask[write_slices] = True  # 仅 small 的有效子区为 True；填充区域保持 False

    return y, mask

def to_dev(
    *tensors: torch.Tensor,
    device: torch.device | str = 'cpu',
    s: Optional[int] = None,
    e: Optional[int] = None,
    dim: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    将若干张量在指定维度 dim 上做切片 [s:e]，并以非阻塞方式传输到指定 device。
    - 若 s、e 为 None，则等价于整段 [:]
    - 要求所有输入是 torch.Tensor
    - 通过构造 slice 元组避免触发高级索引
    - 若仅有一个张量输入，则返回该张量本身；否则返回包含各张量的元组
    """
    dev = torch.device(device)

    def _slice_along_dim(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"to_dev expects torch.Tensor, got {type(t)}")

        # 规范化 dim，支持负维度
        d = dim if dim >= 0 else dim + t.dim()
        if d < 0 or d >= t.dim():
            raise IndexError(
                f"dim out of range (got {dim} for tensor with {t.dim()} dims)"
            )

        # 构造所有维度为 ":" 的索引列表，再在 dim 位置替换为 slice(s, e)
        index = [slice(None)] * t.dim()
        index[d] = slice(s, e)
        return t.__getitem__(tuple(index))

    sliced = tuple(_slice_along_dim(t) for t in tensors)
    moved = tuple(t.to(dev, non_blocking=True) for t in sliced)

    # 仅一个张量时，直接返回该张量
    if len(moved) == 1:
        return moved[0]
    return moved

def get_strategy(loss_type: LossTypeDict, epoch: int) -> PhaseConfig:
    # 找到所有小于等于 epoch 的阈值
    valid_keys = [k for k in loss_type.keys() if isinstance(k, int) and epoch <= k]
    if not valid_keys:
        # 若没有上界覆盖，则可选择返回最大键对应配置或抛错
        raise ValueError(f"No strategy configured for epoch={epoch}")
    # 取最小的上界（即最早满足条件的区间）
    key = min(valid_keys)
    return loss_type[key]


def fmt6(x):
    sgn = 1 if x < 0 else 0
    absx = abs(x)
    int_len = len(str(int(absx)))
    # 留出小数点和可能的负号
    decimals = 6 - int_len - 1 - sgn
    if decimals < 0:
        # 放不下，返回溢出标记或科学计数法（任选其一）
        # return "######"  # 固定 6 个井号
        # 或者用科学计数法，尽量贴近 6 宽：
        return f"{x:.1e}"[:6]  # 简单截断
    return f"{x:.{decimals}f}"

def fmt6w(x):
    s = fmt6(x)
    # 若 s 长度不足 6，用空格左填充至 6
    return f"{s:>6}"[:6]


    
def is_all_ones(
    x: torch.Tensor,
    *,
    atol: float = 1e-8,
    rtol: float = 0.0,
    empty_is_one: bool = False
) -> Tuple[bool, torch.dtype]:
    """
    检查张量是否全为 1（根据其 dtype 语义），并返回 (是否全为1, dtype)。

    规则：
      - bool: 全为 True
      - 整数: 全等于 1
      - 浮点: 全接近 1（使用 rtol/atol）
      - 复数: 实部接近 1 且虚部接近 0（使用 rtol/atol）
      - 空张量: 返回 empty_is_one（默认 False）

    参数:
      x: 待检查的张量
      atol: 绝对误差容差（针对浮点/复数）
      rtol: 相对误差容差（针对浮点/复数）
      empty_is_one: 空张量是否视为全为 1

    返回:
      (is_all_ones: bool, dtype: torch.dtype)
    """
    dtype = x.dtype

    # 空张量处理
    if x.numel() == 0:
        return (bool(empty_is_one), dtype)

    if dtype == torch.bool:
        # 全为 True
        return (bool(torch.all(x)), dtype)

    if dtype.is_floating_point:
        # 使用 torch.isclose 与 1.0 比较
        one = torch.ones((), dtype=dtype, device=x.device)
        is_one = torch.isclose(x, one, rtol=rtol, atol=atol)
        return (bool(torch.all(is_one)), dtype)

    if dtype in (torch.complex64, torch.complex128, torch.complex32) if hasattr(torch, "complex32") else \
       (dtype == torch.complex64 or dtype == torch.complex128):
        # 复数：实部 ~ 1，虚部 ~ 0
        real_close = torch.isclose(x.real, torch.ones((), dtype=x.real.dtype, device=x.device), rtol=rtol, atol=atol)
        imag_close = torch.isclose(x.imag, torch.zeros((), dtype=x.imag.dtype, device=x.device), rtol=rtol, atol=atol)
        return (bool(torch.all(real_close & imag_close)), dtype)

    if dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        # 整数精确比较
        one = torch.ones((), dtype=dtype, device=x.device)
        return (bool(torch.all(x == one)), dtype)

    # 其他非常见 dtype（如 bfloat16、float16 等）会被 is_floating_point 捕获；
    # 如果出现未覆盖 dtype，这里保守返回 False。
    return (False, dtype)


from fnmatch import fnmatch
from typing import Dict, Iterable, Tuple

import torch


def load_partial_state_dict(model, state_dict, skip_substrings=None, strict=False):
    """
    加载 state_dict 时跳过包含指定子串的参数。
    
    参数:
        model          : torch.nn.Module
        state_dict     : dict, 通常由 torch.load 得到
        skip_substrings: List[str], 比如 ["cte", "bias"]，只要键名里含有这些子串就跳过
        strict         : bool, 传给 load_state_dict
    """
    skip_substrings = skip_substrings or []

    filtered_state = {}
    for k, v in state_dict.items():
        if any(substr in k for substr in skip_substrings):
            continue
        filtered_state[k] = v

    missing, unexpected = model.load_state_dict(filtered_state, strict=strict)
    print(missing, unexpected)
    return missing, unexpected

def truncate_embedding_state_dict(state_dict, key: str, b: int, mode: str = "truncate"):
    """
    修改 state_dict 中某个 nn.Embedding 的权重。

    参数:
        state_dict : dict，通常由 torch.load 得到
        key        : str，Embedding 的键名，比如 "embedding.weight"
        b          : int，只保留/覆盖前 b 行
        mode       : str，可选:
                       - "truncate": 把张量裁剪成 (b, E)，state_dict[key] 的形状直接变小
                       - "partial" : 保持原始大小，只更新前 b 行，其余行保持不变
    返回:
        new_state_dict : dict，修改后的 state_dict
    """
    if key not in state_dict:
        raise KeyError(f"{key} not found in state_dict")

    W = state_dict[key]
    if W.ndim != 2:
        raise ValueError(f"{key} is not a 2D tensor, got shape {tuple(W.shape)}")
    if b > W.size(0):
        raise ValueError(f"b={b} > {W.size(0)} rows in {key}")

    new_state_dict = state_dict.copy()

    if mode == "truncate":
        new_state_dict[key] = W[:b].clone()
    elif mode == "partial":
        # 保持大小不变，只更新前 b 行
        W_new = W.clone()
        W_new[b:] = torch.zeros_like(W_new[b:])  # 也可以保持原样
        new_state_dict[key] = W_new
    else:
        raise ValueError(f"Unknown mode={mode}, expected 'truncate' or 'partial'")

    return new_state_dict

def fetch_locals(*names: str) -> Dict[str, Any]:
    """
    从调用者的局部作用域（local namespace）抓取指定变量名，并以字典返回。
    若变量不存在，将不会出现在结果中。
    """
    frame = inspect.currentframe()
    try:
        caller = frame.f_back  # type: ignore
        locs = caller.f_locals if caller is not None else {}
        return {name: locs[name] for name in names if name in locs}
    finally:
        # 避免循环引用导致的内存泄漏
        del frame
        if 'caller' in locals():
            del caller


def to_one_hot_logits(
    targets: torch.Tensor,  # (B, T) 的类别索引
    vocab_size: int,        # V
    *,
    pos_val: float = 1.0,   # 目标位置的值
    neg_val: float = -1e9,   # 非目标位置的值
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    将 (B, T) 的类别索引转换为 (B, T, V) 的 one-hot 风格“logits”。

    约定：
      - targets 中的取值应为 [0, V-1] 的整数（可为任意整型/长整型张量）。
      - 返回张量在 (b, t, targets[b,t]) 位置为 pos_val，其余为 neg_val。
      - 如果需要真正的 logits（例如对交叉熵无穷大间隔），可以设 pos_val=0, neg_val=-inf，
        然后在 softmax 前加上这些分数。

    参数:
      targets: (B, T) 整数索引
      vocab_size: 词表大小 V
      pos_val: 目标位置的值（默认 1.0）
      neg_val: 非目标位置的值（默认 0.0）
      dtype: 输出 dtype（默认跟随 pos_val/neg_val 推断或使用浮点）

    返回:
      (B, T, V) 的张量
    """
    if targets.dim() != 2:
        raise ValueError(f"targets must be 2D (B, T), got shape {tuple(targets.shape)}")

    B, T = targets.shape
    device = targets.device

    # 选择 dtype：优先使用用户指定；否则用浮点（与 pos/neg 值兼容）
    if dtype is None:
        # 若 pos/neg 是浮点，则用 float32；否则默认 float32
        dtype = torch.float32

    # 初始化为 neg_val
    out = torch.full((B, T, vocab_size), fill_value=neg_val, dtype=dtype, device=device)

    # 安全性检查：索引范围
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.long):
        raise TypeError(f"targets must be integer dtype, got {targets.dtype}")

    if torch.any(targets < 0) or torch.any(targets >= vocab_size):
        bad_idx = torch.stack([(targets < 0), (targets >= vocab_size)]).any(0)
        # 定位第一个错误位置，给出更明确的报错
        first = (bad_idx.view(-1).nonzero(as_tuple=False)[0].item()
                 if bad_idx.any() else None)
        raise IndexError(f"targets contain out-of-range indices for V={vocab_size}. Example bad index at flat pos {first}")

    # 用高级索引写入 pos_val
    # 展平成 (B*T,) 方便构造坐标
    flat_targets = targets.reshape(-1)
    bt = torch.arange(B * T, device=device)
    b = bt // T
    t = bt % T
    out[b, t, flat_targets] = pos_val

    return out


def get_cached_tensor(shape, dtype=torch.float32, device:Union[str, torch.device]='cpu', fill_value=None):
    """
    获取缓存张量，避免重复分配内存。
    Args:
        shape (tuple): 张量形状
        dtype (torch.dtype): 数据类型
        device (str or torch.device): 设备
        fill_value (optional, float/bool/int): 若指定，则填充值
    Returns:
        torch.Tensor
    """
    key = (shape, dtype, torch.device(device).type)

    if key not in _cache:
        _cache[key] = torch.empty(shape, dtype=dtype, device=device)

    t = _cache[key]

    if fill_value is not None:
        t.fill_(fill_value)

    return t


def save_heatmap(
    x: torch.Tensor,
    path: str = "tmp_heatmap.png",
    *,
    vmin: Optional[float] = None, # type: ignore
    vmax: Optional[float] = None, # type: ignore
    figsize: Tuple[float, float] = (4.0, 3.6),
    dpi: int = 200,
    add_colorbar: bool = True,
    cmap: str = "viridis",
    interpolation: str = "nearest",
    tight: bool = True,
    nan_color: Optional[Tuple[float, float, float, float]] = None,  # e.g., (1,1,1,0) for transparent
) -> str:
    """
    将 (N, N) 的张量保存为热图图片（不显示）。
    
    参数:
      x: (N, N) 的 torch.Tensor（可在任意设备/类型；内部会转为 CPU float）
      path: 保存文件名（如 'tmp.png'、'out.jpg' 等）
      vmin/vmax: 颜色范围；默认自动根据数据取 min/max
      figsize: 画布尺寸（英寸）
      dpi: 输出分辨率
      add_colorbar: 是否添加色条
      cmap: 颜色映射（默认 viridis，与示例相同风格）
      interpolation: 图像插值方式（'nearest' 常用于热图）
      tight: 是否使用紧凑布局
      nan_color: NaN 的颜色 RGBA；None 表示使用 cmap 默认 NaN 颜色

    返回:
      保存路径（便于链式使用）
    """
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"Expected (N, N) tensor, got shape {tuple(x.shape)}")

    # 移到 CPU 并转 float，用于 matplotlib
    x_np = x.detach().to(torch.float32).cpu().numpy()

    # 设置 colormap（可选自定义 NaN 颜色）
    cmap_obj = plt.get_cmap(cmap).copy()
    if nan_color is not None:
        cmap_obj.set_bad(nan_color)

    # 计算显示范围
    vmin: float = x_np.min() if vmin is None else float(vmin)
    vmax: float = x_np.max() if vmax is None else float(vmax)
    if vmin == vmax:
        # 避免全常数导致的警告/空白：人为扩一点范围
        eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6 + 1e-6
        vmin -= eps
        vmax += eps

    # 绘制
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(x_np, cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation=interpolation, origin="upper")

    # 可选：添加 colorbar（与示例类似的连续色条）
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # 去掉坐标轴刻度线与边框（如需栅格可自行调整）
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if tight:
        plt.tight_layout()

    # 保存并清理
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return path


def save_hierarchy_heatmap(
    mat: torch.Tensor,
    path: str = "tmp_hierarchy.png",
    *,
    is_distance: bool = False,
    linkage_method: Literal[
        "single", "complete", "average", "weighted",
        "centroid", "median", "ward"
    ] = "ward",
    title: str = "Hierarchical Clustered Matrix",
    xlabel: str = "Index",
    ylabel: str = "Index",
    cmap: str = "viridis",
    interpolation: str = "nearest",
    dpi: int = 300,
    figsize=(8, 8),
    colorbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    reorder_only: bool = True,
) -> str:
    """
    对 (N, N) 张量执行层次化聚类重排并保存热图。

    参数:
      mat: (N, N) 的相似度或距离矩阵（torch.Tensor，设备/类型不限）
      path: 保存路径（默认 'tmp_hierarchy.png'）
      is_distance: True 则输入为距离矩阵；False 则视为相似度并转换成距离
      linkage_method: scipy.cluster.hierarchy.linkage 的 method，默认 'ward'
      title/xlabel/ylabel: 图标题与轴标签
      cmap/interpolation: imshow 的配色与插值
      dpi/figsize: 图片分辨率与画布大小
      colorbar_label: 色条标签；None 则自动根据 is_distance 选择
      vmin/vmax: 颜色范围；默认使用数据 min/max
      reorder_only: True 表示仅重排并显示原始量（相似度或距离）；
                    False 表示将相似度转换为距离后也用距离值显示

    返回:
      保存路径
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected (N, N) tensor, got shape {tuple(mat.shape)}")

    N = mat.shape[0]
    # 转 CPU numpy
    mat_np = mat.detach().to(torch.float32).cpu().numpy()

    # 将相似度转换为距离以用于 linkage
    if is_distance:
        D = mat_np.copy()
        # 对角归零，强制对称
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)
        # 检查是否非负
        if (D < -1e-7).any():
            warnings.warn("Distance matrix has negative entries; clipping to 0.")
            D = np.maximum(D, 0.0)
    else:
        S = mat_np
        # 归一处理可选：很多相似度在 [-1, 1] 或 [0, 1]
        # 转成距离 d = 1 - s，并裁剪以避免异常
        D = 1.0 - S
        D = np.clip(D, 0.0, 2.0)
        # 对角归零，强制对称
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)

    # linkage 需要 condensed 形式（N*(N-1)/2,)
    # 若已是 condensed，可跳过 squareform 转换，这里做鲁棒处理
    if is_valid_y(D):
        y = D  # 已经是 condensed
    else:
        y = squareform(D, checks=False)

    # 执行层次化聚类
    Z = linkage(y, method=linkage_method)
    order = leaves_list(Z)

    # 重排矩阵供展示
    if reorder_only:
        show_mat = mat_np[order][:, order]
    else:
        show_mat = D[order][:, order]

    # 颜色范围
    if vmin is None:
        vmin = float(np.nanmin(show_mat))
    if vmax is None:
        vmax = float(np.nanmax(show_mat))
    if vmin == vmax:
        eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6 + 1e-6
        vmin -= eps
        vmax += eps

    # 绘制
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(show_mat, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)

    # 色条
    if colorbar_label is None:
        colorbar_label = "distance" if is_distance or not reorder_only else "similarity"
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, rotation=270, labelpad=20)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 隐藏刻度以获得更干净的矩阵视图
    # ax.set_xticks([])
    # ax.set_yticks([])
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    return path

def compute_indices_and_visual_all(
    loss_full: torch.Tensor,      # (B, T1, T2, C, tp)
    loss_reduced: torch.Tensor,   # (B, T1, C, tp) —— loss_full 在 T2 上经 reduce 得到
    *,
    reduce: Literal["sum", "mean"] = "mean",  # 仅用于一致性记录（不会重新聚合）
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 argmin 索引以及基于该索引的可视化矩阵（一次性为所有 b, p 生成）。

    返回:
      indices: (B, T1, tp)
      visual:  (B, tp, T1, T2)  其中 visual[b, p] 即为 (T1, T2) 可视化矩阵
    """
    if loss_full.ndim != 5:
        raise ValueError(f"loss_full must be (B, T1, T2, C, tp), got {tuple(loss_full.shape)}")
    if loss_reduced.ndim != 4:
        raise ValueError(f"loss_reduced must be (B, T1, C, tp), got {tuple(loss_reduced.shape)}")

    B, T1, T2, C, TP = loss_full.shape
    Br, T1r, Cr, TPr = loss_reduced.shape
    if not (B == Br and T1 == T1r and C == Cr and TP == TPr):
        raise ValueError(f"Shape mismatch: full={tuple(loss_full.shape)} reduced={tuple(loss_reduced.shape)}")

    # 1) 在 C 维做 argmin
    indices = torch.argmin(loss_reduced, dim=2)  # (B, T1, tp)

    # 2) 基于 indices 从 loss_full 中抽取可视化矩阵 V[b,p] = loss_full[b, :, :, indices[b,:,p], p]
    # 展平 (B, T1, tp) -> (B*tp, T1)，便于一次性 batch gather
    idx_bp_t1 = indices.permute(0, 2, 1).reshape(B * TP, T1)  # (B*tp, T1)

    # 重排 loss_full 为 (B, tp, T1, T2, C)
    L = loss_full.permute(0, 4, 1, 2, 3).contiguous()  # (B, tp, T1, T2, C)
    L = L.reshape(B * TP, T1, T2, C)                   # (B*tp, T1, T2, C)

    # 为每个 (b,p,t1) 提供要选取的 c 索引，扩展到 T2 维度以便 gather
    gather_idx = idx_bp_t1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T2, 1)  # (B*tp, T1, T2, 1)

    # 在最后一维 C 上 gather，得到 (B*tp, T1, T2, 1) -> squeeze -> (B*tp, T1, T2)
    visual_bp = torch.gather(L, dim=-1, index=gather_idx).squeeze(-1)

    # 还原形状为 (B, tp, T1, T2)
    visual = visual_bp.view(B, TP, T1, T2)

    return indices, visual

def save_paired_hierarchy_heatmaps(
    A: torch.Tensor,
    B: torch.Tensor,
    path: str = "tmp_pair_hierarchy.png",
    *,
    is_distance: bool = False,  # False 表示 A/B 是相似度矩阵，将被转换为距离做聚类
    linkage_method: Literal[
        "single", "complete", "average", "weighted",
        "centroid", "median", "ward"
    ] = "ward",
    titles: Tuple[str, str] = ("A (clustered by A)", "B (reordered by A)"),
    cmap: str = "viridis",
    interpolation: str = "nearest",
    figsize: Tuple[float, float] = (10.0, 5.0),
    dpi: int = 300,
    vmin_A: Optional[float] = None,
    vmax_A: Optional[float] = None,
    vmin_B: Optional[float] = None,
    vmax_B: Optional[float] = None,
    colorbar_labels: Tuple[Optional[str], Optional[str]] = (None, None),
    show_axes: bool = False,
) -> str:
    """
    对 A 做层次化聚类，获得顺序 order；将 A、B 都按该顺序重排后，并排绘制两张热图。

    参数:
      A, B: (N, N) 张量；B 会用 A 的聚类顺序重排
      is_distance: 若为 True，则 A、B 被视为距离矩阵；否则视为相似度并转换为距离做聚类
      linkage_method: linkage 的聚类方法
      titles: 左右子图标题
      cmap / interpolation: 颜色与插值
      figsize / dpi: 图像大小与分辨率
      vmin_A/vmax_A, vmin_B/vmax_B: 各自的颜色范围；None 表示自动取 min/max
      colorbar_labels: 左右色条标签；None 则自动推断
      show_axes: 是否显示坐标轴刻度/边框

    返回:
      保存路径
    """
    # 基本校验
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (N,N), got {tuple(A.shape)}")
    if B.shape != A.shape:
        raise ValueError(f"B must have same shape as A, got {tuple(B.shape)} vs {tuple(A.shape)}")
    N = A.shape[0]

    # 转 numpy
    A_np = A.detach().to(torch.float32).cpu().numpy()
    B_np = B.detach().to(torch.float32).cpu().numpy()

    # 构造用于聚类的距离矩阵 D_A
    if is_distance:
        D_A = A_np.copy()
        # 对角为 0，强制对称与非负
        np.fill_diagonal(D_A, 0.0)
        D_A = 0.5 * (D_A + D_A.T)
        D_A = np.clip(D_A, 0.0, None)
    else:
        # 相似度 -> 距离
        D_A = 1.0 - A_np
        D_A = np.clip(D_A, 0.0, 2.0)
        np.fill_diagonal(D_A, 0.0)
        D_A = 0.5 * (D_A + D_A.T)

    # linkage 需要 condensed 距离
    if is_valid_y(D_A):
        y = D_A
    else:
        y = squareform(D_A, checks=False)

    Z = linkage(y, method=linkage_method)
    order = leaves_list(Z)

    # 重排 A, B 用于显示（注意：显示时仍显示原量，不强制显示距离）
    A_show = A_np[order][:, order]
    B_show = B_np[order][:, order]

    # 颜色范围
    def _range(mat, vmin, vmax):
        if vmin is None:
            vmin = float(np.nanmin(mat))
        if vmax is None:
            vmax = float(np.nanmax(mat))
        if vmin == vmax:
            eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6 + 1e-6
            vmin -= eps; vmax += eps
        return vmin, vmax

    vmin_A, vmax_A = _range(A_show, vmin_A, vmax_A)
    vmin_B, vmax_B = _range(B_show, vmin_B, vmax_B)

    # 色条标签自动推断
    label_A = colorbar_labels[0]
    label_B = colorbar_labels[1]
    if label_A is None:
        label_A = "distance" if is_distance else "similarity"
    if label_B is None:
        label_B = "distance" if is_distance else "similarity"

    # 绘制左右子图
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    axL, axR = axes

    imA = axL.imshow(A_show, cmap=cmap, interpolation=interpolation, vmin=vmin_A, vmax=vmax_A)
    imB = axR.imshow(B_show, cmap=cmap, interpolation=interpolation, vmin=vmin_B, vmax=vmax_B)

    axL.set_title(titles[0])
    axR.set_title(titles[1])

    # 色条
    cbarA = plt.colorbar(imA, ax=axL, fraction=0.046, pad=0.04)
    cbarB = plt.colorbar(imB, ax=axR, fraction=0.046, pad=0.04)
    cbarA.set_label(label_A, rotation=270, labelpad=16)
    cbarB.set_label(label_B, rotation=270, labelpad=16)

    # 坐标轴外观
    if not show_axes:
        for ax in (axL, axR):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    else:
        axL.set_xlabel("Index (A-order)")
        axL.set_ylabel("Index (A-order)")
        axR.set_xlabel("Index (A-order)")
        axR.set_ylabel("Index (A-order)")

    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return path

def make_soft_targets(
    target: torch.LongTensor,
    V: int,
    eps: float = 0.1,
    return_logits: bool = False,
    ignore_index: int | None = None,
    dtype: torch.dtype = torch.float32,
    delta: float = 1e-6,
):
    """
    将 (B, T) 的整数标签转换为“合理”的 (B, T, V) 目标：
      - 默认返回 label smoothing 后的概率分布（soft targets）
      - 可选返回与该分布一致的一组 logits（softmax 后还原该分布）

    参数:
      target: (B, T) LongTensor，值域 [0, V-1] 或为 ignore_index
      V: 类别/词表大小
      eps: label smoothing 系数 in [0, 1)
      return_logits: 若为 True，返回 logits；否则返回概率分布
      ignore_index: 可选，用于掩码的填充值（如 pad）。这些位置将被置为均匀分布或零向量（见下）
      dtype: 输出张量的数据类型
      delta: 构造 logits 时用于避免 log(0) 的下限

    返回:
      out: (B, T, V) 张量
           - 当 return_logits=False: 每个位置为概率分布，和为 1
           - 当 return_logits=True : 一组 logits，使 softmax(out) ≈ 上述分布

    约定:
      - 若提供 ignore_index，则对应位置的行为：
        * return_logits=False: 返回均匀分布（对损失进行掩码时通常无影响）
        * return_logits=True : 返回零向量 logits（softmax 为均匀分布）
    """
    assert target.dim() == 2, "target should be (B, T)"
    assert V > 1, "V must be > 1"
    assert 0 <= eps < 1, "eps must be in [0, 1)"

    B, T = target.shape
    device = target.device

    # 1) 构造平滑分布
    probs = torch.full((B, T, V), eps / (V - 1), device=device, dtype=dtype)
    # 掩码：有效位置
    if ignore_index is None:
        idx = target.unsqueeze(-1)  # (B, T, 1)
        probs.scatter_(dim=-1, index=idx, value=1.0 - eps)
    else:
        valid = (target != ignore_index)
        # 先全部填充均匀分布（对 ignore_index 位置即为最终分布）
        # 对有效位置再 scatter 为平滑分布
        if valid.any():
            idx = target.clamp_min(0).unsqueeze(-1)  # 避免 scatter 索引为负
            probs[valid] = eps / (V - 1)
            probs.scatter_(dim=-1, index=idx, value=1.0 - eps)

    if not return_logits:
        return probs

    # 2) 从分布构造一组 logits（非唯一；softmax 后还原 probs）
    safe = probs.clamp_min(delta)
    logits = safe.log()
    # 去掉每个位置的常数自由度（使均值为 0，不影响 softmax）
    logits = logits - logits.mean(dim=-1, keepdim=True)

    # 对 ignore_index 位置，返回零向量 logits（softmax 为均匀分布）
    if ignore_index is not None:
        invalid = (target == ignore_index)
        if invalid.any():
            logits[invalid] = 0.0

    return logits


@torch.no_grad()
def directed_label_smoothing(
    targets: torch.LongTensor,          # (B, T)
    teacher_logits: torch.Tensor,       # (B, T, V)
    mode: Literal["mul", "add"] = "mul",
    alpha: float = 1.2,                 # 正类增强因子 (>1)
    beta: float = 0.9,                  # 负类衰减因子 (<1, >0)
    add_gamma: float = 0.05,            # add 模式时正类加成
    min_prob: float = 1e-8,             # 数值稳定下限
    return_logits: bool = False,
    ignore_index: int | None = None,
    dtype: torch.dtype | None = None,
):
    """
    基于教师 logits 的“定向 Label Smoothing”：
      - 提升正确类概率，压低错误类概率，再归一化
      - 支持乘法缩放("mul")或加法拉伸("add")

    参数:
      targets: (B, T) LongTensor, 类别索引，取值 [0, V-1] 或 ignore_index
      teacher_logits: (B, T, V) 浮点张量
      mode:
        - "mul": 正类乘 alpha (>1)，负类乘 beta (0<beta<1)，后归一化（推荐）
        - "add": 正类加 add_gamma (>0)，负类按原比例分配剩余质量
      alpha, beta: 乘法模式的增强/衰减系数
      add_gamma: 加法模式对正类的加成量
      min_prob: 防止出现 0 概率导致后续 log 的数值问题
      return_logits: 若 True，返回一组 logits（softmax 后还原该分布）
      ignore_index: 若提供，忽略位置返回均匀分布（或零 logits）
      dtype: 输出 dtype，默认跟 teacher_logits 一致

    返回:
      probs 或 logits: (B, T, V)
    """
    assert targets.dim() == 2, "targets should be (B, T)"
    assert teacher_logits.dim() == 3, "teacher_logits should be (B, T, V)"
    B, T = targets.shape
    Bt, Tt, V = teacher_logits.shape
    assert (B, T) == (Bt, Tt), "shapes of targets and teacher_logits must match in (B,T)"
    if dtype is None:
        dtype = teacher_logits.dtype

    device = teacher_logits.device
    targets = targets.to(device)

    # 1) 教师概率
    p = F.softmax(teacher_logits, dim=-1)  # (B, T, V)

    # 2) 构建 mask
    idx = targets.unsqueeze(-1)  # (B, T, 1)
    if ignore_index is not None:
        valid = (targets != ignore_index)
        invalid = ~valid
    else:
        valid = torch.ones((B, T), dtype=torch.bool, device=device)
        invalid = torch.zeros((B, T), dtype=torch.bool, device=device)

    # 3) 定向 smoothing
    probs = p.clone()

    if mode == "mul":
        # 乘法缩放：正类 * alpha, 负类 * beta，然后归一化
        # 先对全部乘以 beta，再把正类乘以 (alpha/beta)
        if not (alpha > 1.0 and 0.0 < beta < 1.0):
            raise ValueError("For mode='mul', require alpha>1 and 0<beta<1.")
        probs = probs * beta
        # scatter 增强正类
        scale = torch.full_like(probs[..., :1], fill_value=alpha / beta)
        probs.scatter_(-1, idx, probs.gather(-1, idx) * (alpha / beta))

        # 仅对有效位置归一化
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = probs / denom

    elif mode == "add":
        # 加法拉伸：正类加 add_gamma，负类按比例缩放剩余概率
        if not (add_gamma > 0):
            raise ValueError("For mode='add', require add_gamma>0.")
        # 取出正类概率
        p_y = probs.gather(-1, idx)  # (B, T, 1)
        # 负类总质量
        neg_mass = (1.0 - p_y).clamp_min(1e-12)
        # 新的正类概率
        new_p_y = (p_y + add_gamma).clamp(max=1.0 - 1e-12)
        # 剩余给负类的质量
        remain = (1.0 - new_p_y)
        # 负类按原比例分配
        probs = probs * (remain / neg_mass)
        probs.scatter_(-1, idx, new_p_y)
    else:
        raise ValueError("mode must be 'mul' or 'add'.")

    # 4) ignore_index 位置设为均匀分布
    if invalid.any():
        uniform = torch.full((V,), 1.0 / V, device=device, dtype=probs.dtype)
        probs[invalid] = uniform

    # 5) 数值稳定与 dtype
    probs = probs.clamp_min(min_prob)
    probs = probs / probs.sum(dim=-1, keepdim=True)  # 再归一，确保严格为分布
    probs = probs.to(dtype)

    if not return_logits:
        return probs

    # 6) 构造对应 logits（可选）
    logits = probs.clamp_min(min_prob).log()
    logits = logits - logits.mean(dim=-1, keepdim=True)
    if invalid.any():
        logits[invalid] = 0.0  # 对应均匀分布
    return logits


def pin_tensors_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    仅遍历给定字典的第一层键值：
    - 若值是位于 CPU 的 torch.Tensor，则替换为值.pin_memory()
    - 其他类型保持不变
    - 不做递归，不做设备迁移，不做任何附加处理
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.device.type == "cpu":
            out[k] = v.pin_memory()
        else:
            out[k] = v
    return out


def make_splits(start: int, end: int, block_size: int) -> List[Tuple[int, int]]:
    """
    生成 [start, end) 区间内的切分，步长为 block_size。
    最后一段不足 block_size 时以 end 结尾。
    
    例如：
    make_splits(0, 10, 4) -> [(0, 4), (4, 8), (8, 10)]
    make_splits(3, 11, 5) -> [(3, 8), (8, 11)]
    """
    if block_size <= 0:
        raise ValueError("block_size 必须为正整数")
    if end < start:
        raise ValueError("end 必须 >= start")

    splits: List[Tuple[int, int]] = []
    cur = start
    while cur < end:
        nxt = cur + block_size
        if nxt > end:
            nxt = end
        splits.append((cur, nxt))
        cur = nxt
    return splits


def get_type(block: Tuple[int, int], N_train: int, N_valid: int) -> str:
    """
    根据 block 的起止位置，判断其类型：
      - 'train' : 完全在训练集范围内 [0, N_train)
      - 'valid' : 完全在验证集范围内 [N_train, N_train + N_valid)
    """
    start, end = block
    if start < 0 or end <= start:
        raise ValueError(f"Invalid block {block}")
    if end <= N_train:
        return 'train'
    elif start >= N_train and end <= N_train + N_valid:
        return 'valid'
    else:
        raise ValueError(f"Block {block} out of range for train+valid sizes {N_train + N_valid}")
    
    
    
def get_emb(block: Tuple[int, int], 
            N_train  : int,          N_valid  : int,
            train_emb: torch.Tensor, valid_emb: torch.Tensor, 
            block2indices: Optional[Dict[Tuple[int, int], Tuple[torch.Tensor, str]]]
    ) -> torch.Tensor:
    """
    根据 block 的起止位置，获取对应的嵌入表示。
    """
    if block2indices is not None:
        val = block2indices.get(block, None)
        if val is not None:
            return train_emb[val[0]] if val[1] == 'train' else valid_emb[val[0]]
        else:
            raise ValueError(f"Block {block} not found in block2indices")
    start, end = block
    if start < 0 or end <= start:
        raise ValueError(f"Invalid block {block}")
    if end <= N_train:
        return train_emb[start:end]
    elif start >= N_train and end <= N_train + N_valid:
        return valid_emb[start - N_train:end - N_train]
    else:
        raise ValueError(f"Block {block} out of range for train+valid sizes {N_train + N_valid}")
    
def get_idx(block: Tuple[int, int], 
            N_train  : int,          N_valid  : int,
            train_idx: torch.Tensor, valid_idx: torch.Tensor,
            block2indices: Optional[Dict[Tuple[int, int], Tuple[torch.Tensor, str]]]
    ) -> torch.Tensor:
    """
    根据 block 的起止位置，获取对应的索引。
    """
    if block2indices is not None:
        val = block2indices.get(block, None)
        if val is not None:
            return train_idx[val[0]] if val[1] == 'train' else valid_idx[val[0]]
        else:
            raise ValueError(f"Block {block} not found in block2indices")
    
    start, end = block
    if start < 0 or end <= start:
        raise ValueError(f"Invalid block {block}")
    if end <= N_train:
        return train_idx[start:end]
    elif start >= N_train and end <= N_train + N_valid:
        return valid_idx[start - N_train:end - N_train]
    else:
        raise ValueError(f"Block {block} out of range for train+valid sizes {N_train + N_valid}")
    
def get_tar(block: Tuple[int, int], N_train: int, train_tar: torch.Tensor,
            block2indices: Optional[Dict[Tuple[int, int], Tuple[torch.Tensor, str]]]
    ) -> Optional[torch.Tensor]:
    """
    根据 block 的起止位置，获取对应的目标标签。
    """
    if block2indices is not None:
        val = block2indices.get(block, None)
        if val is not None:
            return train_tar[val[0]] if val[1] == 'train' else None
        else:
            raise ValueError(f"Block {block} not found in block2indices")
    
    start, end = block
    if start < 0 or end <= start:
        raise ValueError(f"Invalid block {block}")
    if end <= N_train:
        return train_tar[start:end]
    else:
        return None
    
    
    
def get_loss_type(T1_type: str, T2_type: str, loss_strategy: Dict) -> str:
    """
    根据 T1 和 T2 的类型，确定损失计算的类型：
      - 'tt' : T1 和 T2 都是训练集
      - 'tv' : T1 是训练集，T2 是词表
      - 'vt' : T1 是验证集，T2 是训练集
      - 'vv' : T1 是验证集，T2 是词表
      - 'vocab' : T1 是词表，T2 是词表
      - 'other' : 其他组合，不计算损失
    """
    if T1_type == 'train' and T2_type == 'train':
        return loss_strategy['dyn_loss']
    elif T1_type == 'train' and T2_type == 'vocab':
        return loss_strategy['prob_loss']
    elif T1_type == 'valid' and T2_type == 'train':
        return loss_strategy['dyn_loss']
    elif T1_type == 'valid' and T2_type == 'vocab':
        return loss_strategy['prob_loss']
    elif T1_type == 'vocab' and T2_type == 'vocab':
        return loss_strategy['sta_loss']
    elif T1_type == 'vocab' and T2_type == 'train':
        return loss_strategy['dyn_loss']
    else:
        raise ValueError(f"Invalid combination of T1_type '{T1_type}' and T2_type '{T2_type}'")


def normalized_matmul(A: torch.Tensor, B: torch.Tensor, 
                      eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入:
        A, B: 形状相同的张量，且其最后两维满足矩阵乘法要求：
              A[..., m, k] 与 B[..., k, n]
        eps: 归一化时防止除零的微小常数
    流程:
        1) 对 dim = -1 / -2 做 L2 归一化 (keepdim=True)
        2) 进行矩阵乘法
        3) 返回 (乘积, A 的范数 * B 的范数)，范数为最后一维的 L2 范数 (keepdim=True)
    返回:
        prod: torch.Tensor，矩阵乘积，形状为 A[..., m, k] @ B[..., k, n] -> [..., m, n]
        norm: torch.Tensor，形状为 A[..., m, 1] * B[..., 1, n]
    """
    # 计算最后一维的 L2 范数，保留维度
    normA = torch.norm(A, p=2, dim=-1, keepdim=True).clamp_min(eps)
    normB = torch.norm(B, p=2, dim=-2, keepdim=True).clamp_min(eps)

    # 归一化到单位范数
    A_normed = A / normA
    B_normed = B / normB

    # 矩阵乘法（针对最后两维），其余维度按批对齐
    prod = A_normed @ B_normed
    norm = normA * normB

    return prod, norm


def mask_fill_scalar_expand(mask: torch.Tensor,
                            true_value,
                            false_value,
                            dim: int,
                            all_true=False) -> torch.Tensor:
    """
    Args:
        mask: Bool tensor of shape (N,)
        true_value: scalar (number or 0-dim tensor)
        false_value: scalar (number or 0-dim tensor)
        dim: total number of dims for the output (>=1)

    Returns:
        Tensor of shape (N, 1, 1, ..., 1) with total dims == dim.
    """
    assert mask.dtype == torch.bool and mask.dim() == 1, "mask must be (N,) bool"
    assert dim >= 1, "dim must be >= 1"

    N = mask.shape[0]
    out_shape = (N,) + (1,) * (dim - 1)
    # Create base tensor filled with false_value
    out = torch.full(out_shape, true_value, dtype=None, device=mask.device)
    if all_true:
        return out
    
    # Expand mask to target shape for where()
    mask_expanded = mask.view(N, *([1] * (dim - 1)))
    out = torch.where(mask_expanded, out, torch.as_tensor(false_value, device=mask.device))
    return out


def gather_idx_and_emb(train_idx: torch.Tensor, train_emb: torch.Tensor, idx2d: torch.Tensor):
    # idx2d: (T, S), long, device 与 train_* 一致
    T, S = idx2d.shape
    flat = idx2d.reshape(-1)                       # (T*S,)
    pos_idx = train_idx.index_select(0, flat)      # (T*S,)
    pos_idx = pos_idx.view(T, S)

    flat_emb = train_emb.index_select(0, flat)     # (T*S, dim)
    pos_emb = flat_emb.view(T, S, train_emb.size(1))
    return pos_idx, pos_emb


import os
import traceback
import functools

import sys

def in_debug_mode() -> bool:
    # 环境变量优先，其次检查 debugpy
    if os.environ.get("DEBUG_MODE", "0") == "1":
        return True
    return "debugpy" in sys.modules


import traceback, functools, os, sys

def thread_guard(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if in_debug_mode():
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Exception in {func.__name__}: {e}")
            traceback.print_exc()
            os._exit(1)
    return wrapper


import torch
import torch.nn.functional as F

def sampled_softmax_loss(eu_val, ct_val, N_sampled, N_topk, temperature, N_total):


    # 1. 计算 HT 权重 w_i
    # ----------------------------------------------------------
    # 剩余部分原本大小为 (N_total - N_top)，你从其中随机采样 (N_dynbr - N_top)
    n_random = N_sampled - N_topk
    total_random_pool = N_total - N_topk

    # inclusion probability
    pi = n_random / total_random_pool

    # w: (N_dynbr,)
    w = torch.ones(N_sampled, device=eu_val.device, dtype=eu_val.dtype)
    if n_random > 0:
        w[N_topk:N_sampled] = 1.0 / pi

    # log w: reshape 成 (1, N_dynbr, 1, 1)
    log_w = torch.log(w).view(1, N_sampled, 1, 1)

    # 2. 构造 HT 修正后的 logits
    # ----------------------------------------------------------
    scale = 20.0 / temperature

    eu_adj = eu_val * scale + log_w
    ct_adj = ct_val * scale + log_w

    # 3. 使用 HT 加权后的 logits 计算 softmax / log_softmax
    # ----------------------------------------------------------
    p_x      = torch.softmax(eu_adj, dim=1)       # (B, N_dynbr, N_C, D)
    log_p_x  = torch.log_softmax(eu_adj, dim=1)
    log_p_y  = torch.log_softmax(ct_adj, dim=1)

    # 4. 最终 loss
    # ----------------------------------------------------------
    # sum over softmax-dimension (=1)
    loss = ((log_p_x - log_p_y) * p_x).sum(dim=1)
    return loss

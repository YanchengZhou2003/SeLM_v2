# timing_utils.py
import time
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
from typing import Dict, List, Optional, Tuple


# —— 计时器 —— 
class _Recorder:
    def __init__(self, unit: str):
        assert unit in ("ms", "s")
        self.unit = unit
        # name -> [t0, t0, ...]；允许循环/并发多次 start
        self._starts: Dict[str, List[float]] = defaultdict(list)
        # name -> (total_seconds, count)
        self._acc: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))
        # name -> [(t0, t1), ...]
        self._intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        # 父子直接关系的累计：father -> { child -> seconds }
        self._children_direct: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._warned: set[str] = set()

    def start(self, name: str) -> None:
        self._starts[name].append(time.perf_counter())

    def end(self, name: str, father: Optional[str]) -> None:
        stk = self._starts.get(name)
        if not stk:
            if name not in self._warned:
                self._warned.add(name)
            return
        t0 = stk.pop()
        t1 = time.perf_counter()
        dt = t1 - t0

        # 自身累计
        total, cnt = self._acc[name]
        self._acc[name] = (total + dt, cnt + 1)
        self._intervals[name].append((t0, t1))

        # 显式父段归属：只在 end 时用 father 参数决定
        if father:
            self._children_direct[father][name] += dt

    # —— 汇总 —— 
    def avg_seconds(self) -> Dict[str, float]:
        return {n: total / cnt for n, (total, cnt) in self._acc.items() if cnt > 0}

    def totals_seconds(self) -> Dict[str, float]:
        return {n: sum(e - s for (s, e) in ivs) for n, ivs in self._intervals.items()}

    def composition(self) -> Dict[str, Tuple[float, Dict[str, float]]]:
        """
        返回：{ parent_name: (parent_total_seconds, { child_name: child_total_seconds, ... }), ... }
        分母为父段的“总时长”（所有出现加总），分子为通过 father 指定的直接子段累计时长。
        """
        result: Dict[str, Tuple[float, Dict[str, float]]] = {}
        totals = self.totals_seconds()
        for pname, child_map in self._children_direct.items():
            parent_total = totals.get(pname, 0.0)
            result[pname] = (parent_total, dict(child_map))
        return result

_current_session: ContextVar[Optional[_Recorder]] = ContextVar("_current_session", default=None)

# —— 格式化（定宽）——
_NAME_W = 20  # name 列宽
_VAL_W  = 20  # 数值列宽（含单位或百分号）

def _fmt_time_exact(seconds: float, unit: str) -> str:
    if unit == "ms":
        val = seconds * 1e3
        s = f"{val:>7.3f}ms"
        return s[-_VAL_W:].rjust(_VAL_W)
    else:
        val = seconds
        s = f"{val:>8.3f}s"
        return s.rjust(_VAL_W)

def _fmt_percent_exact(ratio: float) -> str:
    pct = ratio * 100.0
    return f"{pct:>9.2f}%"

def _print_table(avg_seconds: Dict[str, float], unit: str, comp: Dict[str, Tuple[float, Dict[str, float]]]) -> None:
    # 平均用时
    for name in sorted(avg_seconds.keys()):
        t = _fmt_time_exact(avg_seconds[name], unit)
        print(f"{name:<{_NAME_W}}{t}")
    # 组成比例
    if comp:
        print("-" * (_NAME_W + _VAL_W))
        for pname in sorted(comp.keys()):
            parent_total, child_map = comp[pname]
            t = _fmt_time_exact(parent_total, unit)
            print(f"{pname:<{_NAME_W}}{t}")  # 父段总时长
            if parent_total <= 0 or not child_map:
                continue
            items = sorted(child_map.items(), key=lambda kv: kv[1], reverse=True)
            for cname, child_total in items:
                ratio = child_total / parent_total if parent_total > 0 else 0.0
                p10 = _fmt_percent_exact(ratio)
                print(f"  {cname:<{_NAME_W-2}}{p10}")

def gettime(fmt: str = "ms", pr=True):
    """
    装饰器：统计平均用时 + 显式父子组成比例。
    - 平均用时：同名多次起止求平均
    - 组成比例：需在 end 时通过 father 指定父段，才会计入该父段的组成
    """
    assert fmt in ("ms", "s"), "fmt 必须为 'ms' 或 's'"
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            rec = _Recorder(fmt)
            token = _current_session.set(rec)
            try:
                return fn(*args, **kwargs)
            finally:
                avgs = rec.avg_seconds()
                comp = rec.composition()
                if pr:
                    _print_table(avgs, rec.unit, comp)
                _current_session.reset(token)
        return wrapper
    return deco

def mark(type: bool, name: str, father: Optional[str] = None) -> None:
    """
    标记：
      type = 0/False -> 起始；
      type = 1/True  -> 结束；此时可指定 father，用于把本次 name 的用时计入 father 的组成。
      name  = 段名（用于聚合平均）
      father= 父段名（可选；仅在结束时有效）
    """
    rec = _current_session.get()
    if rec is None or not isinstance(name, str) or not name:
        return
    if type:
        rec.end(name, father)
    else:
        rec.start(name)


import torch


class CUDATimer:
    def __init__(self, warmup: int = 3):
        self.warmup = warmup
        self.events = {}     # name -> (start_event, end_event)
        self.records = {}    # name -> [elapsed per round]
        self.rounds = 0      # 已完成的轮数

    def mark(self, name: str, type: int):
        """记录开始/结束事件"""
        if type == 0:  # start
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            self.events[name] = (start_ev, end_ev)
            start_ev.record()
        elif type == 1:  # end
            if name not in self.events:
                raise RuntimeError(f"No start for {name}")
            _, end_ev = self.events[name]
            end_ev.record()
        else:
            raise ValueError("type must be 0 or 1")

    def finish_round(self):
        """在每次 loom 结束时调用，统一同步并统计"""
        torch.cuda.synchronize()
        for name, (start_ev, end_ev) in self.events.items():
            elapsed = start_ev.elapsed_time(end_ev)
            self.records.setdefault(name, []).append(elapsed)
        self.rounds += 1

    def summary(self, merge_devices=True, width=25):
        """
        merge_devices=True: 除了 per-device，还会输出合并后的全局平均
        width: 每列最小宽度，用于对齐输出
        """
        # -------- 先输出 per-device --------
        print("\nPer-device results:")
        for name, vals in self.records.items():
            if len(vals) <= self.warmup:
                print(f"{name.ljust(width)} not enough records")
                continue
            t = torch.tensor(vals[self.warmup:])
            avg, std, n = t.mean().item(), t.std().item(), len(t)
            print(f"{name.ljust(width)} avg={avg:8.3f} ms  std={std:8.3f} ms  n={n:5d}")

        # -------- 再输出合并结果 --------
        if merge_devices:
            print("\nMerged across devices:")
            merged = {}
            for name, vals in self.records.items():
                short_name = name.split("_", 1)[-1]  # 去掉 devX_
                merged.setdefault(short_name, []).extend(vals)

            for name, vals in merged.items():
                if len(vals) <= self.warmup:
                    print(f"{name.ljust(width)} not enough records")
                    continue
                t = torch.tensor(vals[self.warmup:])
                avg, std, n = t.mean().item(), t.std().item(), len(t)
                print(f"{name.ljust(width)} avg={avg:8.3f} ms  std={std:8.3f} ms  n={n:5d}")


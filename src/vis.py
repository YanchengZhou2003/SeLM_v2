import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from scipy.cluster.hierarchy import (dendrogram, leaves_list, linkage,
                                     optimal_leaf_ordering)
from scipy.spatial.distance import pdist, squareform

from src.para import *


def visualize_similarity(S_eu: np.ndarray, S_ct: np.ndarray, meta_name="", save_eu=True, loss_dyn_dyn=0.):
    sim_ct_path = os.path.join(vis_path, meta_name.format("", "ct"))
    sim_eu_path = os.path.join(vis_path, meta_name.format("_", "eu"))
    
    # --- 相似度矩阵 (eu) ---
    np.fill_diagonal(S_eu, 1.0)

    # 转为“距离”做聚类
    D = 1.0 - S_eu
    np.fill_diagonal(D, 0.0)
    dvec = squareform(D, checks=False)

    Z = linkage(dvec, method='average')
    order = leaves_list(Z)
    S_re = S_eu[order][:, order]

    # --- 一张图：左树 + 热图 + 右色条 (eu) ---
    if save_eu:
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

        fig.savefig(sim_eu_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    # print(f"Similarity + tree visualization saved to {sim_eu_path}")

    # --- 相似度矩阵 (ct) ---
    S_ct = S_ct[order][:, order]

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
    ax.set_title(f"Hierarchical Cosine Similarity (ct), loss={loss_dyn_dyn:.4f}")

    # 右侧颜色条
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ct-cosine similarity", rotation=270, labelpad=25)

    fig.savefig(sim_ct_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # print(f"Similarity + tree visualization saved to {sim_ct_path}")



# ---- 基于行/列余弦距离的层次聚类（Hierarchical clustering, 层次聚类）----
def _order_by_hclust_rows(X, method="average", metric="cosine", optimal=True):
    if len(X) <= 1:
        return np.arange(len(X)), None
    dvec = pdist(X, metric=metric)
    Z = linkage(dvec, method=method)
    if optimal and len(X) > 2:
        Z = optimal_leaf_ordering(Z, dvec)
    return leaves_list(Z), Z

def _safe_l2_norm_rows(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

# ---- 统一绘图风格 ----
def _plot_panel(S_re, Zr, Zc, title, cmap, use_power_norm, gamma, vmin, vmax, save_path):
    from matplotlib.colors import PowerNorm

    N, M = S_re.shape
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[2.5, 14, 0.7],
        height_ratios=[2.5, 14],
        wspace=0.0, hspace=0.0
    )

    # 左：行树
    ax_row = fig.add_subplot(gs[1, 0])
    if Zr is not None and N > 1:
        dendrogram(Zr, ax=ax_row, orientation="right", no_labels=True, color_threshold=None)
    ax_row.invert_yaxis(); ax_row.set_xticks([]); ax_row.set_yticks([])

    # 上：列树
    ax_col = fig.add_subplot(gs[0, 1])
    if Zc is not None and M > 1:
        dendrogram(Zc, ax=ax_col, orientation="top", no_labels=True, color_threshold=None)
    ax_col.set_xticks([]); ax_col.set_yticks([])

    # 中：热图
    ax = fig.add_subplot(gs[1, 1])
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax) if use_power_norm else None
    im = ax.imshow(S_re, cmap=cmap, norm=norm, vmin=None if use_power_norm else vmin,
                   vmax=None if use_power_norm else vmax, interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)

    # 右：色条（保持一致）
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("similarity", rotation=270, labelpad=25)

    # 左上占位
    ax_empty = fig.add_subplot(gs[0, 0]); ax_empty.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---- 主入口：用 A 聚类，B 复用同序重排并绘制，风格一致 ----
def visualize_pair_bihclust(A, B, meta_name: str, save_eu,
                            method="average",
                            metric_rows="cosine", metric_cols="cosine",
                            use_power_norm=True, gamma=2.0,
                            cmap="inferno",
                            share_color_scale=False,
                            title_A="Hierarchical similarity (A)",
                            title_B="Hierarchical similarity (B)"):
    """
    A: (N, M) 相似度矩阵，用于确定行/列顺序
    B: (N, M) 相似度矩阵，按 A 的顺序重排后绘制
    share_color_scale: 若为 True，则 A/B 共享同一 (vmin, vmax)，确保色彩对比可比
    返回：row_order, col_order, A_re, B_re
    """
    sim_ct_path = os.path.join(vis_path, meta_name.format("", "ct"))
    sim_eu_path = os.path.join(vis_path, meta_name.format("_", "eu"))
    
    A = np.asarray(A)
    B = np.asarray(B)
    assert A.shape == B.shape, "A 与 B 形状必须一致以便同序对齐"

    N, M = A.shape

    # 归一化到单位范数，稳健计算余弦距离
    Ar = _safe_l2_norm_rows(A)     # 行向量
    Ac = _safe_l2_norm_rows(A.T)   # 列向量（对 A^T 的行做聚类）

    row_order, Zr = _order_by_hclust_rows(Ar, method=method, metric=metric_rows, optimal=True)
    col_order, Zc = _order_by_hclust_rows(Ac, method=method, metric=metric_cols, optimal=True)

    A_re = A[row_order][:, col_order]
    B_re = B[row_order][:, col_order]

    if share_color_scale:
        vmin = min(A_re.min(), B_re.min())
        vmax = max(A_re.max(), B_re.max())
        vA = vB = (vmin, vmax)
    else:
        vA = (A_re.min(), A_re.max())
        vB = (B_re.min(), B_re.max())

    # 绘制两张图，风格参数一致
    _plot_panel(A_re, Zr, Zc, title_A, cmap, use_power_norm, gamma, *vA, sim_eu_path if save_eu else None)
    _plot_panel(B_re, Zr, Zc, title_B, cmap, use_power_norm, gamma, *vB, sim_ct_path)

    return row_order, col_order, A_re, B_re


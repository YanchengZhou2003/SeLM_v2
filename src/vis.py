import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering, dendrogram
from scipy.spatial.distance import pdist

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
    fig = plt.figure(figsize=(18, 14))
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
    else:
        plt.show()

# ---- 主入口：用 A 聚类，B 复用同序重排并绘制，风格一致 ----
def visualize_pair_bihclust(A, B,
                            method="average",
                            metric_rows="cosine", metric_cols="cosine",
                            use_power_norm=True, gamma=2.0,
                            cmap="inferno",
                            share_color_scale=False,
                            save_path_A=None, save_path_B=None,
                            title_A="Hierarchical similarity (A)",
                            title_B="Hierarchical similarity (B)"):
    """
    A: (N, M) 相似度矩阵，用于确定行/列顺序
    B: (N, M) 相似度矩阵，按 A 的顺序重排后绘制
    share_color_scale: 若为 True，则 A/B 共享同一 (vmin, vmax)，确保色彩对比可比
    返回：row_order, col_order, A_re, B_re
    """
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
    _plot_panel(A_re, Zr, Zc, title_A, cmap, use_power_norm, gamma, *vA, save_path_A)
    _plot_panel(B_re, Zr, Zc, title_B, cmap, use_power_norm, gamma, *vB, save_path_B)

    return row_order, col_order, A_re, B_re


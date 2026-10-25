#!/usr/bin/env python3
"""E6 §5.4 — 「基的选择」这条免费结构项到底值多少。

lambda_h2 = 1/4 * sum_(rho c) ( |w_B| + sum_b |w_b| )^2

注意 lambda_h2 只吃 (w, W)：`sum_b |w_b|` 是系数在**当前两体基**下的 L1 范数。
而物理只要求 X = sum_b w_b u_b u_b^T。若把同一个 X 改写到它的**特征基**下，
系数就变成特征值，L1 变成核范数 —— 那是「正交表示的 L1」的下界。

所以 (核范数 / 现在的 L1) 就是**换基能白拿多少**的定量上界。

用法: python src/e6/e6_basis_probe.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMP / "data" / "verifiers"))

from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors, reconstruct_x, split_u_vectors  # noqa: E402
from dfthcsos.verify import load_inputs                                                          # noqa: E402

SUB = COMP / "data" / "sample_submission" / "dfthcsos" / "submission.npz"
LAM_H1 = 12.880166407
E_GAP = 3.965477244
LE_NOW = 49.301369251


def main() -> None:
    p = print
    sol = load_inputs(SUB)
    z = np.load(SUB, allow_pickle=False)
    words, edges = z["angle_words"], z["tree_edges"]

    u_all = reconstruct_u_vectors(np.asarray(words, dtype=np.float64) / (1 << 11), edges)
    _, ub = split_u_vectors(u_all, sol.w)
    w = np.asarray(sol.w, dtype=np.float64)
    W = np.asarray(sol.W, dtype=np.float64)

    gram = ub @ np.swapaxes(ub, -1, -2)
    off = np.abs(gram - np.eye(ub.shape[1])[None]).max()
    p("=" * 92)
    p("[0] 提交的两体向量是不是正交的？")
    p("=" * 92)
    p("    max |u_i·u_j − delta_ij| = %.6f   -> %s"
      % (off, "**不正交**" if off > 1e-8 else "正交"))
    p("    契约没有强制两体行正交（只查行长非零），所以 (w,u)->X 是**多对一**映射。")
    p("    这既给了空间，也说明「核范数」只是**正交表示**的界，未必是全局下界。")

    X = reconstruct_x(ub, w)                      # (R,C,N,N)
    l1 = np.abs(w).sum(axis=1)                    # (R,C) 当前 sum_b|w_b|
    tr = w.sum(axis=1)                            # = Tr X
    ev = np.linalg.eigvalsh(X)
    nuc = np.abs(ev).sum(axis=-1)                 # 特征基下的 |w|_1 = 核范数
    rank = (np.abs(ev) > 1e-9).sum(axis=-1)

    p("")
    p("=" * 92)
    p("[1] 一致性检查")
    p("=" * 92)
    p("    rank(X) 范围 = [%d, %d]（<= B = 60，因为 X 是 60 个秩一项之和）"
      % (rank.min(), rank.max()))
    p("    max |Tr X − sum_b w_b| = %.3e   (必须为 0：换基不改变迹)" % np.abs(
        np.trace(X, axis1=-1, axis2=-2) - tr).max())

    p("")
    p("=" * 92)
    p("[2] L1 范数：现在 vs 特征基（核范数）")
    p("=" * 92)
    p("    sum_(rho c) sum_b |w_b|  = %12.6f   （现在）" % l1.sum())
    p("    sum_(rho c) ||X||_*     = %12.6f   （核范数，正交表示的下界）" % nuc.sum())
    p("    比值 = %.4f  =>  换基最多把 sum|w| 降 %.2f%%"
      % (l1.sum() / nuc.sum(), 100 * (1 - nuc.sum() / l1.sum())))

    wB = W + tr
    cur = 0.25 * np.sum((np.abs(wB) + l1) ** 2)
    new = 0.25 * np.sum((np.abs(wB) + nuc) ** 2)
    p("")
    p("    lambda_h2 现在        = %12.6f" % cur)
    p("    lambda_h2 核范数替换  = %12.6f   （省 %.2f%%）" % (new, 100 * (1 - new / cur)))
    p("    Lambda    现在        = %12.6f" % (LAM_H1 + cur))
    p("    Lambda    乐观        = %12.6f" % (LAM_H1 + new))

    def le(L: float) -> float:
        return math.sqrt(E_GAP * (2 * L - E_GAP))

    g = le(LAM_H1 + cur) / le(LAM_H1 + new)
    p("    lambda_eff 现在       = %12.6f" % le(LAM_H1 + cur))
    p("    lambda_eff 乐观       = %12.6f" % le(LAM_H1 + new))
    p("")
    p("    ==> dfthcsos 单项 gain = %.4f  ->  总分 x%.4f"
      % (g, g ** (1 / 3)))
    p("    （注意：换基会改变词表的量化误差，这里假设 X 能被精确复现 —— 这是**上界**。）")

    p("")
    p("=" * 92)
    p("[3] 差最大的 8 个 (rho,c)")
    p("=" * 92)
    d = (np.abs(wB) + l1) - (np.abs(wB) + nuc)
    p("    %-10s %10s %10s %10s %12s" % ("(rho,c)", "|w|_1", "核范数", "差", "w_B"))
    for r, c in np.dstack(np.unravel_index(np.argsort(-d, axis=None)[:8], d.shape))[0]:
        p("    (%2d,%2d)  %10.4f %10.4f %10.4f %12.6f"
          % (r, c, l1[r, c], nuc[r, c], d[r, c], wB[r, c]))
    p("")
    p("    |w|_1 − 核范数 分布: mean %.4f / max %.4f / |w|_1 > 3*核范数 的对数 %d/%d"
      % ((l1 - nuc).mean(), (l1 - nuc).max(), int((l1 > 3 * nuc).sum()), l1.size))
    p("    最大那几对 w_B ≈ 0 —— 是「正负系数互相抵消」的受挫对，")
    p("    它们贡献 1/4*(0 + sum|w|)^2，是 lambda_h2 里最贵的形状。")


if __name__ == "__main__":
    main()

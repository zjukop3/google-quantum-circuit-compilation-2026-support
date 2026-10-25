#!/usr/bin/env python3
"""E6 §3 — lambda_eff 到底还有多少空间：直接测 epsilon <-> Lambda 的 Pareto。

lambda_eff = sqrt(E_gap*(2*Lambda - E_gap))，而
    Lambda = sum_j |lambda_j|  +  1/4 sum_(rho c) ( |w_B^(rho c)| + sum_b |w_b^(rho c)| )^2
误差侧只有两个闸门：error_1 <= 20 mHa（用了 15.9%）与 error_2 <= 20 mHa（用了 93.2%）。
所以问题就一句：**在 error_2 不超 20 mHa 的前提下，Lambda 还能砍掉多少？**

三条探测方向（都在 (R,B,C)=(65,60,60) 形状不变的前提下，纯 numpy、不重训）：
  A. 整体缩放   w -> t*w, W -> t*W          （最粗的一维，用来看 Pareto 的陡度）
  B. 按贡献丢秩 把 1/4(|w_B|+sum|w|)^2 最小的一批 (rho,c) 置零  （最有物理意义的压缩方向）
  C. 符号相干性 诊断：sum_b|w_b| 与 |sum_b w_b| 的差（Lambda 里唯一可以「不花钱」的结构项）

用法: python src/e6/e6_lambda_probe.py [--points 12]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMP / "data" / "verifiers"))

from dfthcsos.scoreinfra.dfthc_math import (                 # noqa: E402
    evaluate_dfthc_vectors, lambda_h2, reconstruct_u_vectors,
)
from dfthcsos.verify import load_inputs                      # noqa: E402

SAMPLE_DIR = COMP / "data" / "sample_submission" / "dfthcsos"
SAMPLE_NPZ = SAMPLE_DIR / "submission.npz"
ERROR_GATE = 0.020
E_GAP_FLOOR = 3.914898


def lam_eff(E_gap: float, Lam: float) -> float:
    return math.sqrt(E_gap * (2.0 * Lam - E_gap)) if 0 < E_gap < 2 * Lam else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=10)
    ap.add_argument("--skip-a", action="store_true", help="跳过整体缩放扫描")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    sol = load_inputs(SAMPLE_NPZ)
    z = np.load(SAMPLE_NPZ, allow_pickle=False)
    words, edges = z["angle_words"], z["tree_edges"]
    R, B, C = (int(x) for x in sol.w.shape)

    u_all = reconstruct_u_vectors(np.asarray(words, dtype=np.float64) / (1 << 11), edges)

    def ev(w, W, tag):
        t0 = time.time()
        m = evaluate_dfthc_vectors(u_all, w, W, sol.S, sol.beta_1, sol.ref,
                                   energy_upper=sol.energy_upper)
        m["tag"] = tag
        m["seconds"] = time.time() - t0
        return m

    w0 = np.asarray(sol.w, dtype=np.float64)
    W0 = np.asarray(sol.W, dtype=np.float64)
    base = ev(w0, W0, "baseline")

    print("=" * 100)
    print("baseline:  Lambda=%.6f  E_gap=%.6f  lambda_eff=%.6f  err1=%.4f mHa  err2=%.4f mHa"
          % (base["Lambda"], base["E_gap"], base["lambda_eff"],
             1e3 * base["error_1"], 1e3 * base["error_2"]))
    print("           预算余量 err2: %.4f mHa (=预算的 %.2f%%)"
          % (1e3 * (ERROR_GATE - base["error_2"]), 100 * (1 - base["error_2"] / ERROR_GATE)))
    print("=" * 100)

    # ---------------------------------------------------------------- A 缩放
    rows_a = []
    if not a.skip_a:
        print("\n[A] 整体缩放 w->t*w, W->t*W  （看基线是不是误差的驻点）")
        print("    %-6s %12s %12s %12s %10s %10s %s"
              % ("t", "Lambda", "E_gap", "lambda_eff", "err1 mHa", "err2 mHa", "判定"))
        for t in np.linspace(0.97, 1.03, 7):
            m = ev(t * w0, t * W0, f"scale {t:.3f}")
            rows_a.append({"t": float(t), **{k: float(m[k]) for k in
                                             ("Lambda", "E_gap", "lambda_eff", "error_1", "error_2")}})
            print("    %-6.3f %12.6f %12.6f %12.6f %10.4f %10.4f %s"
                  % (t, m["Lambda"], m["E_gap"], m["lambda_eff"], 1e3 * m["error_1"],
                     1e3 * m["error_2"],
                     "OK" if m["error_2"] <= ERROR_GATE else "REJECT"))
        lo, hi = rows_a[2], rows_a[-1]        # 0.99 -> 1.03（避开 t<1 的爆炸区）
        dlam = hi["Lambda"] / lo["Lambda"] - 1
        derr = hi["error_2"] - lo["error_2"]
        slope_mha = 1e3 * derr / 4            # mHa of err2 per +1% of t
        print("    -> 斜率: t 每 +1%%  =>  err2 %+.1f mHa ,  Lambda %+.3f%%"
              % (slope_mha, 100 * (dlam / 4)))
        budget_mha = 1e3 * (ERROR_GATE - base["error_2"])
        dt = 0.01 * budget_mha / slope_mha    # t 的相对位移上限
        print("       预算只剩 %.4f mHa  =>  t 只能动 %.3e（%.4f%%）"
              % (budget_mha, dt, 100 * dt))
        print("       对应 Lambda 只能动 %.3e（%.2e%%）== 等于没用"
              % (abs(hi["Lambda"] - lo["Lambda"]) / 4 * dt,
                 100 * abs(dlam / 4) * dt))
        print("       注：t=1 是 err2 的尖锐极小点，两侧都炸 —— 基线坐在 eps 的驻点上。")

    # ------------------------------------------------------- B 按贡献丢秩
    contrib = 0.25 * (np.abs(W0 + np.sum(w0, axis=1)) + np.linalg.norm(w0, ord=1, axis=1)) ** 2
    flat = contrib.ravel()
    order = np.argsort(flat)                        # 贡献最小的 (rho,c) 排前面
    total = float(np.sum(flat))
    cum = np.cumsum(flat[order])
    print("\n[B] 按 Lambda 贡献丢秩（把贡献最小的 (rho,c) 整对置零）")
    print("    总 Lambda 贡献 %.6f；单对贡献 最大 %.4f / 中位 %.3e / 最小 %.3e"
          % (total, flat.max(), float(np.median(flat)), flat.min()))
    print("    要丢掉 Lambda 的百分之X 需要丢多少对：", end="")
    for f in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
        k = int(np.searchsorted(cum, f * total)) + 1
        print(" %.0f%%->%d对" % (100 * f, min(k, flat.size)), end="")
    print()
    print("    %-10s %10s %12s %12s %12s %10s %10s %s"
          % ("丢Lamb占", "丢对数", "Lambda", "E_gap", "lambda_eff", "err1 mHa", "err2 mHa", "判定"))
    rows_b = [{"frac_lambda": 0.0, "k": 0,
               **{k: float(base[k]) for k in
                  ("Lambda", "E_gap", "lambda_eff", "error_1", "error_2")}}]
    for frac in np.linspace(0.0, 1.0, a.points)[1:]:
        if frac <= 0:
            continue
        k = int(np.searchsorted(cum, frac * total)) + 1
        k = min(k, flat.size)
        idx = np.unravel_index(order[:k], contrib.shape)
        w, W = w0.copy(), W0.copy()
        w[idx[0], :, idx[1]] = 0.0
        W[idx[0], idx[1]] = 0.0
        m = ev(w, W, f"drop-lambda {frac:.2f}")
        rows_b.append({"frac_lambda": float(frac), "k": k,
                       "Lambda_dropped_frac": float(np.sum(flat[order[:k]]) / total),
                       **{kk: float(m[kk]) for kk in
                          ("Lambda", "E_gap", "lambda_eff", "error_1", "error_2")}})
        print("    %-10s %10d %12.6f %12.6f %12.6f %10.4f %10.4f %s"
              % ("%.0f%%" % (100 * frac), k, m["Lambda"], m["E_gap"], m["lambda_eff"],
                 1e3 * m["error_1"], 1e3 * m["error_2"],
                 "OK" if m["error_2"] <= ERROR_GATE else "REJECT"))

    # ------------------------------------------------------- C 符号相干性
    a_rc = np.linalg.norm(w0, ord=1, axis=1)
    s_rc = np.sum(w0, axis=1)
    wB = W0 + s_rc
    incoherent = a_rc - np.abs(s_rc)
    lam_coherent = 0.25 * float(np.sum((np.abs(wB) + np.abs(s_rc)) ** 2))
    lam_h1_base = float(base["Lambda"]) - lambda_h2(w0, W0)
    print("\n[C] 符号相干性诊断（Lambda 里唯一「不花钱」的结构项）")
    print("    sum_b|w_b| 与 |sum_b w_b| 的差： 总 %.6f / mean %.6f / max %.6f"
          % (float(incoherent.sum()), float(incoherent.mean()), float(incoherent.max())))
    print("    若所有 w_b 同号（完全相干）: lam_h2 -> %.6f  (现在 %.6f, 只省 %.2f%%)"
          % (lam_coherent, lambda_h2(w0, W0), 100 * (1 - lam_coherent / lambda_h2(w0, W0))))
    print("    -> 相干上界 + lam_h1 %.4f => Lambda >= %.4f  (floor 49.0)" %
          (lam_h1_base, lam_coherent + lam_h1_base))

    out = {"baseline": {k: float(base[k]) for k in
                        ("Lambda", "E_gap", "lambda_eff", "error_1", "error_2", "E_SOS")},
           "A_scale": rows_a, "B_drop": rows_b,
           "C_sign": {"incoherent_sum": float(incoherent.sum()),
                      "lam_h2_if_coherent": lam_coherent,
                      "lam_h1": lam_h1_base},
           "budget_left_mHa": 1e3 * (ERROR_GATE - base["error_2"])}
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()

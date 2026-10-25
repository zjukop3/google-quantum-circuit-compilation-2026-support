#!/usr/bin/env python3
"""E6 §2 — 三项相乘到底意味着什么：交换率的定量边界 + dfthcsos 的参数曲面。

score = lambda_eff * T_total * Q_total
  =>  dS/S = dlambda/lambda + dT/T + dQ/Q            （**没有平方项，也没有立方项**）

于是「+1 个 qubit 值多少 Toffoli」变成一个可以直接写出来的数：

    1 qubit  <->  T/Q  个 Toffoli      (dfthcsos)
    1 qubit  <->  C/Q  个 Toffoli      (invsqrt, score = Q*C)
    1 qubit  <->  2T/Q 个 Toffoli      (dqi,      score = R*Q^2*T)

本脚本
  (a) 把三个 benchmark 的交换率算出来做对比（含 E5 的 invsqrt 优化后数值）；
  (b) 用官方 verifier 的成本模型扫 (R,B,C) / J / brot，给出 T-Q 成本曲面；
  (c) 把「更大 (R,B,C)」翻译成**必须的 lambda_eff 降幅**（break-even 阈值）。

用法: python src/e6/e6_exchange.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMP / "data" / "verifiers"))

from dfthcsos.verify import QUBIT_CAP, heuristic_costs            # noqa: E402

N = 76
BROT = 12
CHI = 10

# 三个 benchmark 的现状
BASELINES = {
    "invsqrt(reference)": dict(Q=260, C=7987.593206597238, note="Q*C"),
    "invsqrt(E5 已提交)": dict(Q=231, C=7108.127186961416, note="Q*C"),
    "dqi(reference)": dict(Q=1893, T=5725108.38, note="R*Q^2*T"),
    "dfthcsos(reference)": dict(Q=730, T=93237, note="lambda*T*Q"),
}


def exchange_rates() -> list[dict]:
    rows = []
    for name, v in BASELINES.items():
        if name.startswith("invsqrt"):
            rate = v["C"] / v["Q"]
            formula = "C/Q"
        elif name.startswith("dqi"):
            rate = 2.0 * v["T"] / v["Q"]
            formula = "2T/Q"
        else:
            rate = v["T"] / v["Q"]
            formula = "T/Q"
        rows.append({"benchmark": name, "formula": formula, "rate": rate,
                     "Q": v["Q"], "dT_for_1q": rate})
    return rows


def cost_model(R: int, B: int, C: int, brot: int = BROT, J: int | None = None,
               tree_toffoli: int | None = None, N: int = N) -> dict:
    """T_total / Q_total，逐字复刻 dfthcsos.verify.resource_costs 的算术。"""
    A = N + R * B
    sel = (A - 1).bit_length()
    if J is None:
        J = (QUBIT_CAP - sel - (sel - 1)) // brot
    K = math.ceil((N - 1) / J)
    h = heuristic_costs(N, R, B, C, brot)
    tree = (A - 2) if tree_toffoli is None else tree_toffoli
    ang_q = sel + J * brot + (sel - 1)
    T = 2 * K * tree + h["toffoli"]
    Q = h["persistent_qubits"] + J * brot + max(h["temporary_qubits"], sel - 1)
    return {"R": R, "B": B, "C": C, "brot": brot, "J": J, "K": K, "A": A, "sel": sel,
            "ang_qubits": ang_q, "within_cap": ang_q <= QUBIT_CAP, "tree": tree,
            "angles_T": 2 * K * tree, "heuristic_T": h["toffoli"],
            "outer_T": h["outer_state_prep_toffoli"], "inner_T": h["inner_state_prep_toffoli"],
            "givens_T": h["givens_rotations_toffoli"],
            "persistent": h["persistent_qubits"] + J * brot,
            "temporary": max(h["temporary_qubits"], sel - 1),
            "h_persistent": h["persistent_qubits"], "h_temporary": h["temporary_qubits"],
            "T": T, "Q": Q, "TQ": T * Q}


def main() -> None:
    p = print
    base = cost_model(65, 60, 60)
    p("=" * 96)
    p("[A] 交换率 —— 「+1 个 qubit 值多少 Toffoli」")
    p("=" * 96)
    p("     score = X * T * Q 类的公式里，dS/S = dT/T + dQ/Q，所以 1 个 qubit 的代价是 T/Q 个 Toffoli。")
    p("     %-22s %-8s %10s %14s %14s" % ("benchmark", "公式", "rate", "dT per +1 q", "dQ per -1 T"))
    for r in exchange_rates():
        p("     %-22s %-8s %10.4f %14.2f %14.6f"
          % (r["benchmark"], r["formula"], r["rate"], r["dT_for_1q"], 1.0 / r["rate"]))
    p("")
    p("     排序（rate 越小 = qubit 越便宜 = 越愿意「拿 T 换 Q」）：")
    rr = sorted(exchange_rates(), key=lambda x: x["rate"])
    for i, r in enumerate(rr):
        p("       %d. %-22s rate %10.4f" % (i + 1, r["benchmark"], r["rate"]))
    dq = [x for x in rr if x["benchmark"].startswith("dqi")][0]["rate"]
    df = [x for x in rr if x["benchmark"].startswith("dfthcsos")][0]["rate"]
    iv = [x for x in rr if x["benchmark"].startswith("invsqrt(E5")][0]["rate"]
    p("")
    p("     ==> dfthcsos = %.1f Toffoli/qubit：比 dqi(%.0f) 温和 %.0fx，比 invsqrt E5(%.2f) 严苛 %.2fx"
      % (df, dq, dq / df, iv, df / iv))
    p("     即：**dfthcsos 上「拿 T 换 Q」是 1:1 的，但 1 个 qubit 要 128 个 Toffoli 才回本** ——")
    p("     这在三项里是中间态：既不像 dqi 那样几乎可以随便加 qubit，也不像 invsqrt 那样 1 个 qubit 只值 31 个 Toffoli。")

    p("")
    p("=" * 96)
    p("[B] 两个 floor 各自值多少 —— 别把它们和 lambda_eff 混为一谈")
    p("=" * 96)
    lam, eg = 308.4559397330944, 3.965477244291833
    le = math.sqrt(eg * (2 * lam - eg))
    p("     Lambda    = %12.6f  (floor 49.0)   -> 单独贴 floor: lambda_eff %.6f   gain %.4fx"
      % (lam, math.sqrt(eg * (2 * 49.0 - eg)), le / math.sqrt(eg * (2 * 49.0 - eg))))
    p("     E_gap     = %12.6f  (floor 3.914898) -> 单独贴 floor: lambda_eff %.6f   gain %.4fx"
      % (eg, math.sqrt(3.914898 * (2 * lam - 3.914898)),
         le / math.sqrt(3.914898 * (2 * lam - 3.914898))))
    p("     两个同时贴                                        -> lambda_eff %.6f   gain %.4fx"
      % (math.sqrt(3.914898 * (2 * 49.0 - 3.914898)),
         le / math.sqrt(3.914898 * (2 * 49.0 - 3.914898))))
    p("     ★ 49.0 是 Lambda（原始 LCU 1-范数）的下限，现在 Lambda=308.46 是它的 6.30 倍；")
    p("       lambda_eff=49.30137 与 49.0 只是**数字巧合** —— verifier 明确写 lambda_eff 没有独立 floor。")

    p("")
    p("=" * 96)
    p("[C] J / brot 扫描：ANGLE 家族的成本是 2*ceil(75/J)*(A-2)")
    p("=" * 96)
    p("     %-4s %-5s %-4s %-4s %-6s %10s %10s %9s %7s %s"
      % ("J", "K", "sel", "anc", "宽度", "angles_T", "heuristic", "T", "Q", "cap"))
    for J in range(6, 13):
        m = cost_model(65, 60, 60, J=J)
        p("     %-4d %-5d %-4d %-4d %-6d %10d %10d %9d %7d %s"
          % (J, m["K"], m["sel"], m["sel"] - 1, m["ang_qubits"], m["angles_T"],
             m["heuristic_T"], m["T"], m["Q"],
             "OK" if m["within_cap"] else "REJECT(>128)"))
    p("")
    p("     树 ancilla = sel-1 = %d 是**必需**的：12 bit 地址的二叉树深度 12，DFS 时祖先控制位必须同时存活。"
      % (base["sel"] - 1))
    p("     宽度 = sel + J*brot + (sel-1)；brot=12 固定 => J=8 是上限（119 < 128），J=9 要 131 > 128。")
    m9 = cost_model(65, 60, 60, J=9)
    p("     假设 J=9 可行（131 qubit，被 cap 挡）：T %d -> %d (%.2f%%), Q %d -> %d (+%.2f%%)"
      % (base["T"], m9["T"], 100 * (1 - m9["T"] / base["T"]), base["Q"], m9["Q"],
         100 * (m9["Q"] / base["Q"] - 1)))
    p("       -> 净 gain %.4fx（T 与 Q 同为 1 次方：gain = (T0/T1)*(Q0/Q1)）"
      % ((base["T"] / m9["T"]) * (base["Q"] / m9["Q"])))
    p("       代价是 131>128 的硬墙 —— 除非把 11 个树 ancilla 压到 8 个，而那是不可能的（见 e6_cap.py）。")

    p("")
    p("=" * 96)
    p("[D] (R,B,C) 曲面：更大模型要付多少成本，必须换来多少 lambda_eff")
    p("=" * 96)
    p("     baseline (R,B,C)=(65,60,60):  A=%d  T=%d  Q=%d  T*Q=%.6g"
      % (base["A"], base["T"], base["Q"], base["TQ"]))
    p("")
    p("     %-16s %5s %10s %6s %10s %8s %13s %14s %s"
      % ("(R,B,C)", "A", "T", "Q", "T*Q", "TQx", "需要 lambda 降到", "Λ 需降到*", ""))
    grid = [(65, 60, 60), (66, 60, 60), (70, 60, 60), (80, 60, 60), (100, 60, 60),
            (65, 61, 60), (65, 70, 60), (65, 80, 60),
            (65, 60, 61), (65, 60, 70), (65, 60, 80),
            (70, 70, 70), (60, 60, 60), (65, 55, 60), (65, 60, 55)]
    for (R, B, C) in grid:
        try:
            m = cost_model(R, B, C)
        except ValueError as exc:
            p("     %-16s  -- %s" % (f"({R},{B},{C})", exc))
            continue
        ratio = m["TQ"] / base["TQ"]
        # break-even: lambda must drop by exactly `ratio` (1/output)
        lam_needed = le / ratio
        # invert lambda_eff = sqrt(Eg*(2*Lam-Eg)) at E_gap = 3.914898 (floor)
        lam_raw = (lam_needed ** 2 / 3.914898 + 3.914898) / 2.0
        p("     %-16s %5d %10d %6d %10.4g %7.3fx %13.4f %14.2f   %s"
          % (f"({R},{B},{C})", m["A"], m["T"], m["Q"], m["TQ"], ratio, lam_needed, lam_raw,
             "" if m["within_cap"] else "cap!"))
    p("     * 最后一列是在 E_gap 贴 floor(3.914898) 的前提下反解的 Λ；用它统一各点的口径。")
    p("")
    p("     读法：**TQx 那一列就是 break-even 的 lambda_eff 倍率**。成本涨 x 倍，lambda_eff 就得跟着降 x 倍。")
    p("     · (66,60,60)：成本只涨 1.4%，lambda_eff 只要降 1.4% 就回本 —— 这是唯一「看着还行」的一格；")
    p("     · (80,60,60)：成本涨 22.0% -> lambda_eff 得从 49.30 掉到 40.43（Lambda 308->211）;")
    p("     · (60,60,60) 反而是**赚的**（成本 0.927x），即 lambda_eff 可以恶化 7.9% 仍然回本 ——")
    p("       但 E2 §2.3 实测 (56,60,60) 的 error_2 = 0.295（超闸 15 倍），这条路被残差封死；")
    p("     · 同一 A（=R*B 固定）下挪 R/B/C 的形状只值千分之几，没有可利用的杠杆。")

    p("")
    p("=" * 96)
    p("[E] 「要达到同样总分，dfthcsos 需要多少」")
    p("=" * 96)
    p("     总分 = (prod_b ref_b/cost_b)^(1/3)，所以 dfthcsos 单项 gain g 只贡献 g^(1/3)。")
    inv_ratio = 2076774.233715282 / 1641977.380188087
    p("     E5 已拿到的 invsqrt 比例 = %.6f -> 总分 %.5f" % (inv_ratio, inv_ratio ** (1 / 3)))
    for g in (1.05, 1.10, 1.157, 1.20, 1.2648, 1.5, 2.0, 2.5688):
        lam_t = le / g
        lam_raw = (lam_t ** 2 / 3.914898 + 3.914898) / 2.0
        p("     dfthcsos gain %.4fx (lambda_eff %.2f, 需 Lambda %.1f) -> 总分 x%.5f"
          % (g, lam_t, lam_raw, g ** (1 / 3)))
    p("     ==> 想在总分上拿到 +8.14%%（= E5 那次的分量），dfthcsos 必须单项 gain >= %.4f"
      % (inv_ratio,))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E6 §1 — dfthcsos 三块成本（lambda_eff / T_total / Q_total）的逐项解剖。

全部数字来自**官方 verifier 自己的代码**（`dfthcsos.verify` + `scoreinfra/*`），
这里不重新推导公式，只把每一项单独取出来，并用随包发布的样例提交交叉核对。

用法:
    python src/e6/e6_anatomy.py
    python src/e6/e6_anatomy.py --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMP / "data" / "verifiers"))

from dfthcsos.scoreinfra.circuits import circuit_costs, read_circuit      # noqa: E402
from dfthcsos.scoreinfra.dfthc_math import (                             # noqa: E402
    evaluate_dfthc_angle_words, lambda_h2,
)
from dfthcsos.verify import (                                            # noqa: E402
    KEEP_BITS, LAMBDA_FLOOR, E_GAP_FLOOR, QUBIT_CAP,
    heuristic_costs, load_inputs,
)

SAMPLE_DIR = COMP / "data" / "sample_submission" / "dfthcsos"
SAMPLE_NPZ = SAMPLE_DIR / "submission.npz"

# 官方 SCORING_README / E1_REPRODUCE 的记录值（回归靶子）
TARGET = {"Lambda": 308.4559397330944, "E_gap": 3.965477244291833,
          "lambda_eff": 49.30136925057438, "T_total": 93237, "Q_total": 730,
          "score": 3355599588.3155365}
# E1_REPRODUCE §2 的逐项分解（用来确认我们复现的是同一条账）
E2_SPLIT = {"angles": 79480, "inner": 9461, "givens": 3450, "outer": 846}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    sol = load_inputs(SAMPLE_NPZ)
    z = np.load(SAMPLE_NPZ, allow_pickle=False)
    words, edges = z["angle_words"], z["tree_edges"]
    R, B, C = (int(x) for x in sol.w.shape)
    N = sol.num_orbitals
    A = N + R * B
    brot, J = 12, 8
    K = math.ceil((N - 1) / J)

    m = evaluate_dfthc_angle_words(words, sol.w, sol.W, sol.S, sol.beta_1, sol.ref,
                                   brot=brot, tree_edges=edges,
                                   energy_upper=sol.energy_upper)
    circles = [read_circuit(SAMPLE_DIR / f"angle_k_{k}.kmb", format="kmb") for k in range(10)]
    cc = circuit_costs(circles)
    per_circ = int(cc[0]["max_magic"])
    angles_toffoli = 2 * K * per_circ
    h = heuristic_costs(N, R, B, C, brot)

    # ---- lambda 块：三个可分辨的组成 ---------------------------------------
    lam_h2 = lambda_h2(sol.w, sol.W)
    lam_h1 = m["Lambda"] - lam_h2
    dW2 = 0.5 * float(np.sum(np.asarray(sol.W) ** 2))
    # E_SOS = ecore - eta*beta_1 - (lambda_h1 - sum lam_j) - 0.5*sum(W**2)
    # => -(lambda_h1 - sum lam_j) = E_SOS - ecore + eta*beta_1 + 0.5*sum(W**2)
    eta = float(sol.ref["electrons"])
    ecore = float(sol.ref["ecore"])
    e_gap_terms = {
        "energy_upper": float(sol.energy_upper),
        "ecore": ecore,
        "-eta*beta_1": -eta * float(sol.beta_1),
        "-0.5*sum(W**2)": -dW2,
        "-(lam_h1 - sum lam_j)": m["E_SOS"] - ecore + eta * float(sol.beta_1) + dW2,
    }

    T_total = angles_toffoli + h["toffoli"]
    Q_total = h["persistent_qubits"] + J * brot + max(h["temporary_qubits"], 11)
    score = m["lambda_eff"] * T_total * Q_total

    out = {
        "shapes": {"N": N, "R": R, "B": B, "C": C, "A": A, "brot": brot, "J": J, "K": K,
                   "chi": KEEP_BITS, "eta": int(sol.ref["electrons"]), "A_minus_2": A - 2},
        "lambda_block": {
            "Lambda": m["Lambda"], "lam_h1": lam_h1, "lam_h2": lam_h2,
            "lam_h1_share": lam_h1 / m["Lambda"], "lam_h2_share": lam_h2 / m["Lambda"],
            "E_SOS": m["E_SOS"], "E_gap": m["E_gap"], "E_upper": float(sol.energy_upper),
            "lambda_eff": m["lambda_eff"],
            "error_1": m["error_1"], "error_2": m["error_2"],
            "gap_in_domain": bool(m["gap_in_domain"]),
            "floor_limited_lambda_eff": math.sqrt(
                E_GAP_FLOOR * (2 * LAMBDA_FLOOR - E_GAP_FLOOR)),
            "Lambda_floor_ratio": m["Lambda"] / LAMBDA_FLOOR,
            "E_gap_floor_ratio_minus1": m["E_gap"] / E_GAP_FLOOR - 1.0,
        },
        "e_gap_terms": e_gap_terms,
        "T_block": {
            "per_circuit_max_magic": per_circ, "K": K,
            "angles_2K(A-2)": angles_toffoli,
            "outer": h["outer_state_prep_toffoli"],
            "inner": h["inner_state_prep_toffoli"],
            "givens": h["givens_rotations_toffoli"],
            "heuristic": h["toffoli"], "T_total": T_total,
            "angles_share": angles_toffoli / T_total,
        },
        "Q_block": {
            "angle_circuit_max_qubits": cc[0]["num_qubits"], "cap": QUBIT_CAP,
            "angles_selection": 12, "angles_words": J * brot, "angles_ancilla": 11,
            "outer_persistent": h["outer_state_prep_persistent_qubits"],
            "outer_temporary": h["outer_state_prep_temporary_qubits"],
            "inner_persistent": h["inner_state_prep_persistent_qubits"],
            "inner_temporary": h["inner_state_prep_temporary_qubits"],
            "persistent_total": h["persistent_qubits"] + J * brot,
            "temporary_total": max(h["temporary_qubits"], 11),
            "Q_total": Q_total,
        },
        "per_circuit": [{"k": c["k"], "num_qubits": c["num_qubits"], "max_magic": c["max_magic"],
                         "reaction_depth": c["reaction_depth"],
                         "num_instructions": c["num_instructions"]} for c in cc],
        "score": score,
        "regression": {
            "Lambda": [m["Lambda"], TARGET["Lambda"]],
            "E_gap": [m["E_gap"], TARGET["E_gap"]],
            "lambda_eff": [m["lambda_eff"], TARGET["lambda_eff"]],
            "T_total": [T_total, TARGET["T_total"]],
            "Q_total": [Q_total, TARGET["Q_total"]],
            "score": [score, TARGET["score"]],
            "E2_split(angles/inner/givens/outer)": [
                [angles_toffoli, h["inner_state_prep_toffoli"], h["givens_rotations_toffoli"],
                 h["outer_state_prep_toffoli"]],
                [E2_SPLIT["angles"], E2_SPLIT["inner"], E2_SPLIT["givens"], E2_SPLIT["outer"]]],
        },
    }

    if a.json:
        print(json.dumps(out, indent=2, default=float))
        return

    p = print
    s = out["shapes"]
    p("=" * 88)
    p("dfthcsos 解剖  |  (N,R,B,C)=(%d,%d,%d,%d)  A=N+R*B=%d  brot=%d  J=%d  K=%d  chi=%d  eta=%d"
      % (s["N"], s["R"], s["B"], s["C"], s["A"], s["brot"], s["J"], s["K"], s["chi"], s["eta"]))
    p("=" * 88)

    lb = out["lambda_block"]
    p("\n[1] lambda_eff = sqrt( E_gap * (2*Lambda - E_gap) )")
    p("     Lambda              = %16.9f    (verifier floor 49.0 -> 现在高出 %.2fx)"
      % (lb["Lambda"], lb["Lambda_floor_ratio"]))
    p("       ├ lam_h1 = S|eig h1'|  = %16.9f   %5.2f%%   单体 LCU 1-范数"
      % (lb["lam_h1"], 100 * lb["lam_h1_share"]))
    p("       └ lam_h2 = 1/4 S(W_B+Sw)^2 = %12.9f   %5.2f%%   二体 LCU 1-范数 <- 全部杠杆"
      % (lb["lam_h2"], 100 * lb["lam_h2_share"]))
    p("     E_gap = E_upper - E_SOS = %12.9f   (floor %.6f -> 现在高出 %.2f%%)"
      % (lb["E_gap"], E_GAP_FLOOR, 100 * lb["E_gap_floor_ratio_minus1"]))
    p("       E_SOS 分解: ecore %.6f  -eta*beta_1 %.6f  -(S|l|-Sl) %.6f  -1/2 S W^2 %.6f"
      % (out["e_gap_terms"]["ecore"], out["e_gap_terms"]["-eta*beta_1"],
         out["e_gap_terms"]["-(lam_h1 - sum lam_j)"], out["e_gap_terms"]["-0.5*sum(W**2)"]))
    p("     error_1 / error_2   = %7.4f / %7.4f mHa   (gate 20 mHa; 预算占用 %5.1f%% / %5.1f%%)"
      % (1e3 * lb["error_1"], 1e3 * lb["error_2"],
         100 * lb["error_1"] / .02, 100 * lb["error_2"] / .02))
    p("     ==> lambda_eff      = %16.9f" % lb["lambda_eff"])
    p("         两个 floor 同时贴的理论下限 = %12.9f   (倍率 %.4f)"
      % (lb["floor_limited_lambda_eff"], lb["lambda_eff"] / lb["floor_limited_lambda_eff"]))
    p("         ★ verifier 源码明写: 'lambda_eff remains the score factor and has NO"
      " separate acceptance floor'")

    tb = out["T_block"]
    p("\n[2] T_total = %d 的构成" % tb["T_total"])
    p("     ANGLE 家族  2*K*(A-2) = 2*%d*%d = %9d   %5.2f%%   <- 唯一由「我们写的电路」决定的项"
      % (tb["K"], tb["per_circuit_max_magic"], tb["angles_2K(A-2)"], 100 * tb["angles_share"]))
    p("     outer_state_prep                 = %9d   %5.2f%%"
      % (tb["outer"], 100 * tb["outer"] / tb["T_total"]))
    p("     inner_state_prep                 = %9d   %5.2f%%"
      % (tb["inner"], 100 * tb["inner"] / tb["T_total"]))
    p("     givens_rotations 2*(N-1)*(2b-1)  = %9d   %5.2f%%   = 2*75*23"
      % (tb["givens"], 100 * tb["givens"] / tb["T_total"]))
    p("     heuristic 小计                   = %9d   %5.2f%%   <- 闭式模型，只吃 (R,B,C,N,chi,beta)"
      % (tb["heuristic"], 100 * (1 - tb["angles_share"])))

    qb = out["Q_block"]
    p("\n[3] Q_total = %d 的构成" % qb["Q_total"])
    p("     常驻 %3d = outer_state_prep %2d + inner_state_prep %2d + ANGLE 词寄存器 J*brot %2d"
      % (qb["persistent_total"], qb["outer_persistent"], qb["inner_persistent"], qb["angles_words"]))
    p("     临时 %3d = max( outer %d , inner %d , ANGLE ancilla %d )   <- 完全由 inner QROAM 决定"
      % (qb["temporary_total"], qb["outer_temporary"], qb["inner_temporary"], qb["angles_ancilla"]))
    p("     ANGLE 电路自身宽度 %d < cap %d   (12 selection + %d words + %d 树 ancilla)"
      % (qb["angle_circuit_max_qubits"], qb["cap"], qb["angles_words"], qb["angles_ancilla"]))
    p("     ⚠ ANGLE 只贡献 %d 常驻 + %d 临时；它的「119 qubit」不是 730 的一部分"
      % (qb["angles_words"], qb["angles_ancilla"]))

    p("\n[4] 10 条 ANGLE 电路逐条（样例提交 angle_k_*.kmb）")
    p("     %-3s %-10s %-10s %-12s %s" % ("k", "qubits", "max_magic", "react_depth", "num_ops"))
    for c in out["per_circuit"]:
        p("     %-3d %-10d %-10d %-12d %d"
          % (c["k"], c["num_qubits"], c["max_magic"], c["reaction_depth"], c["num_instructions"]))

    p("\n[5] 回归核对（对标官方 SCORING_README / E1 / E2）")
    for k, (got, want) in out["regression"].items():
        gf = np.atleast_1d(got).astype(float)
        wf = np.atleast_1d(want).astype(float)
        ok = bool(np.allclose(gf, wf, rtol=1e-11, atol=1e-9))
        p("     %-38s got %-34s want %-30s %s"
          % (k, np.array2string(gf, precision=9), np.array2string(wf, precision=9),
             "OK " if ok else "MISMATCH"))
    p("=" * 88)


if __name__ == "__main__":
    main()

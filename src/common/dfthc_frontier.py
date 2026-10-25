#!/usr/bin/env python3
"""Map the (R, B, C) feasibility frontier for dfthcsos by truncating the baseline.

A truncated (R', B', C') factorization *is* a legal submission (the verifier only
reads array shapes and checks behaviour), so this needs no training: it is a hard
feasibility probe.  Truncation is strictly worse than a re-trained factorization
of the same size, so anything feasible here is feasible for real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMP / "src"))
sys.path.insert(0, str(COMP / "data" / "verifiers"))

from dfthc_fastscore import cost                                    # noqa: E402
from dfthcsos.scoreinfra.dfthc_math import evaluate_dfthc_angle_words as ev  # noqa: E402
from dfthcsos.verify import load_inputs                             # noqa: E402

SAMPLE = COMP / "data" / "sample_submission" / "dfthcsos" / "submission.npz"
N = 76


def main():
    z = np.load(SAMPLE, allow_pickle=False)
    words, edges = z["angle_words"], z["tree_edges"]
    sol = load_inputs(SAMPLE)
    R0, B0, C0 = sol.w.shape
    print("baseline (R,B,C) = %d,%d,%d   A=%d" % (R0, B0, C0, N + R0 * B0))
    print("%-14s %-6s %-9s %-9s %-8s %-10s %-4s %-9s %-8s %-11s %s" % (
        "(R',B',C')", "A", "err_2", "Lambda", "E_gap", "lambda_eff",
        "J/K", "T_total", "Q_total", "score", "x"))

    def probe(Rp, Bp, Cp):
        keep = np.empty(N + Rp * Bp, dtype=np.int64)
        keep[:N] = np.arange(N)
        for b in range(Rp):
            keep[N + b * Bp: N + (b + 1) * Bp] = N + b * B0 + np.arange(Bp)
        m = ev(words[keep],
               np.ascontiguousarray(sol.w[:Rp, :Bp, :Cp]),
               np.ascontiguousarray(sol.W[:Rp, :Cp]),
               sol.S, sol.beta_1, sol.ref, brot=12, tree_edges=edges,
               energy_upper=sol.energy_upper)
        A = N + Rp * Bp
        c = cost(A, N, 8, 12, Rp, Bp, Cp, A - 2)
        ok = (m["error_1"] <= 0.020 and m["error_2"] <= 0.020 and m["Lambda"] > 49.0
              and m["E_gap"] > 3.914898 and m["gap_in_domain"] and c["within_cap"])
        return m, c, (m["lambda_eff"] * c["toffoli"] * c["qubits"] if ok else float("inf")), ok

    grid = ([(R0, B0, C0)]
            + [(r, B0, C0) for r in (56, 48, 40, 32, 24)]
            + [(R0, b, C0) for b in (56, 50, 44, 36, 28)]
            + [(R0, B0, c) for c in (56, 48, 40, 32)]
            + [(48, 48, 48), (40, 40, 40), (32, 48, 48), (48, 32, 48)])
    base = None
    for (Rp, Bp, Cp) in grid:
        try:
            m, c, s, ok = probe(Rp, Bp, Cp)
        except Exception as exc:
            print("%-14s ERROR %s" % ((Rp, Bp, Cp), str(exc)[:60]))
            continue
        if base is None:
            base = s
        x = base / s if (ok and s) else float("nan")
        print("%-14s %-6d %-9.5f %-9.3f %-8.5f %-10.5f %-4s %-9.0f %-8.0f %-11.4g %.3f%s" % (
            (Rp, Bp, Cp), N + Rp * Bp, m["error_2"], m["Lambda"], m["E_gap"],
            m["lambda_eff"], "%d/%d" % (c["J"], c["K"]), c["toffoli"], c["qubits"],
            s, x, "" if ok else "  REJECT"))


if __name__ == "__main__":
    main()

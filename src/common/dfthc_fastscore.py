#!/usr/bin/env python3
"""Fast dfthcsos score model: exact verifier metrics WITHOUT the kickmix simulation.

The shipped verifier couples two independent halves:
  * metrics (error_1, error_2, Lambda, E_SOS, E_gap, lambda_eff) <- words + weights
  * resources (T_total, Q_total)                                <- brot, J, R, B, C, A

`recover_words_from_circuits` is only there to prove the circuits *do* implement
the words; both halves are pure numpy.  So a candidate submission can be priced
in seconds here, and only the finalist needs the ~2 min kickmix run.

Usage:
    python src/dfthc_fastscore.py                 # brot sweep on the sample
    python src/dfthc_fastscore.py --words FILE --brot 12 --J 8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMP / "data" / "verifiers"))

from dfthcsos.scoreinfra.dfthc_math import evaluate_dfthc_angle_words   # noqa: E402
from dfthcsos.verify import (                                           # noqa: E402
    QUBIT_CAP, heuristic_costs, load_inputs,
)

SAMPLE = COMP / "data" / "sample_submission" / "dfthcsos" / "submission.npz"


def angles_cost(num_vectors: int, J: int, brot: int, tree_toffoli: float | None = None):
    """Per-pass ANGLES Toffoli count and qubit use of a (A, J) one-hot QROM family.

    The reference family is a full binary segment tree: one `init_and` (1 CCX)
    per internal node, `del_and` uncomputed by measurement (free).  With A
    incompressible random rows that is exactly A - 2 nodes per circuit.
    """
    if tree_toffoli is None:
        tree_toffoli = num_vectors - 2
    K = math.ceil((num_vectors - (num_vectors - (76 + 65 * 60))) / 0)  # unused guard
    return tree_toffoli


def cost(num_vectors: int, N: int, J: int, brot: int, R: int, B: int, C: int,
         tree_toffoli: float) -> dict:
    """T_total / Q_total for a family whose per-circuit tree costs `tree_toffoli`."""
    K = math.ceil((N - 1) / J)
    h = heuristic_costs(N, R, B, C, brot)
    sel = (num_vectors - 1).bit_length()
    ang_persistent = J * brot
    ang_temporary = sel - 1                      # one ancilla per tree level
    ang_qubits = sel + ang_persistent + ang_temporary
    toffoli = 2.0 * K * tree_toffoli + h["toffoli"]
    persistent = h["persistent_qubits"] + ang_persistent
    temporary = max(h["temporary_qubits"], ang_temporary)
    return {
        "K": K, "brot": brot, "J": J,
        "angles_toffoli": 2.0 * K * tree_toffoli,
        "heuristic_toffoli": h["toffoli"],
        "toffoli": toffoli,
        "angles_qubits": ang_qubits,
        "within_cap": ang_qubits <= QUBIT_CAP,
        "persistent_qubits": persistent,
        "temporary_qubits": temporary,
        "qubits": persistent + temporary,
    }


def evaluate(words, edges, brot, sol, *, J=None, R=None, B=None, C=None,
             tree_toffoli=None):
    R = R if R is not None else sol.w.shape[0]
    B = B if B is not None else sol.w.shape[1]
    C = C if C is not None else sol.w.shape[2]
    N = sol.num_orbitals
    A = N + R * B
    if J is None:
        sel = (A - 1).bit_length()
        J = (QUBIT_CAP - sel - (sel - 1)) // brot
    m = evaluate_dfthc_angle_words(words, sol.w, sol.W, sol.S, sol.beta_1, sol.ref,
                                   brot=brot, tree_edges=edges,
                                   energy_upper=sol.energy_upper)
    c = cost(A, N, J, brot, R, B, C,
             tree_toffoli if tree_toffoli is not None else A - 2)
    accepted = (m["error_2"] <= 0.020 and m["error_1"] <= 0.020
                and m["Lambda"] > 49.0 and m["E_gap"] > 3.914898
                and m["gap_in_domain"] and c["within_cap"])
    return {**{k: m[k] for k in ("error_1", "error_2", "Lambda", "E_SOS", "E_gap",
                                 "lambda_eff", "gap_in_domain")},
            **c, "accepted": accepted,
            "score": (m["lambda_eff"] * c["toffoli"] * c["qubits"]) if accepted else float("inf")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--words", type=Path, default=SAMPLE)
    ap.add_argument("--brot", type=int, default=None)
    ap.add_argument("--J", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    z = np.load(a.words, allow_pickle=False)
    words, edges = z["angle_words"], z["tree_edges"]
    sol = load_inputs(SAMPLE)
    print("A=%d  L=%d  N=%d  R,B,C=%s" % (words.shape[0], words.shape[1],
                                          sol.num_orbitals, sol.w.shape))
    if a.brot is not None:
        w = (words >> (12 - a.brot)).astype(np.int64) if a.brot <= 12 else words
        r = evaluate(w, edges, a.brot, sol, J=a.J)
        print(json.dumps(r, indent=2, default=float) if a.json else
              "brot=%d J=%d score=%.6g T=%.0f Q=%.0f lambda_eff=%.5f err2=%.5f ok=%s"
              % (a.brot, r["J"], r["score"], r["toffoli"], r["qubits"],
                 r["lambda_eff"], r["error_2"], r["accepted"]))
        return
    hdr = ("brot", "err_2", "lambda_eff", "J", "K", "T_total", "Q_total", "score", "x")
    print("%-5s %-9s %-10s %-4s %-4s %-9s %-8s %-11s %s" % hdr)
    base = None
    for brot in (12, 11, 10, 9, 8, 7, 6):
        w = (words >> (12 - brot)).astype(np.int64)
        r = evaluate(w, edges, brot, sol)
        if brot == 12:
            base = r["score"]
        x = (base / r["score"]) if (base and r["accepted"]) else float("nan")
        print("%-5d %-9.5f %-10.5f %-4d %-4d %-9.0f %-8.0f %-11.4g %.3f%s" % (
            brot, r["error_2"], r["lambda_eff"], r["J"], r["K"], r["toffoli"],
            r["qubits"], r["score"], x, "" if r["accepted"] else "  REJECT"))


if __name__ == "__main__":
    main()

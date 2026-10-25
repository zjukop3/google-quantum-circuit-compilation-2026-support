#!/usr/bin/env python3
"""E5 diagnostic: where does `Lambda` come from, and how much error budget is left?

Read-only.  Uses the *official* verifier's own modules (`dfthcsos.verify.load_inputs`,
`scoreinfra.dfthc_math`, `scoreinfra.costs_prepare`) on a factorization `.npz` --
the continuous factorization, i.e. before the 12-bit angle-word quantization.

Reports the split of `Lambda = lambda_h1 + lambda_h2`, the error budget consumed,
and the `lambda_eff` implied by pushing each quantity to its acceptance floor.

    PY=.../2609-xingzhi-drug/.venv/bin/python
    $PY src/analyze_dfthcsos_budget.py data/sample_submission/dfthcsos/submission.npz
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "data" / "verifiers"))

from dfthcsos.scoreinfra import dfthc_math as M  # noqa: E402
from dfthcsos.scoreinfra.costs_prepare import SpinFreeData_V1  # noqa: E402
from dfthcsos.verify import (  # noqa: E402
    E_GAP_FLOOR,
    ERROR_2_THRESHOLD,
    LAMBDA_FLOOR,
    UPPER_ENERGY,
    load_inputs,
)


def lambda_eff(Lam: float, gap: float) -> float:
    return math.sqrt(gap * (2 * Lam - gap))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path, nargs="?", default=ROOT / "data/sample_submission/dfthcsos/submission.npz")
    ap.add_argument("--brot", type=int, default=12)
    ap.add_argument("--J", type=int, default=8)
    args = ap.parse_args()

    inp = load_inputs(args.npz)
    w, W, S, beta_1, ref = inp.w, inp.W, inp.S, inp.beta_1, inp.ref
    N = ref["oei"].shape[0]
    R, B, C = w.shape

    # Reproduce the verifier's continuous-factorization path.
    u = np.asarray(inp.params["U"], dtype=np.float64)
    u = u.reshape(-1, N) if u.ndim != 3 else u
    one_body, u_b_rho = M.split_u_vectors(u / np.linalg.norm(u, axis=1, keepdims=True), w)
    X = M.reconstruct_x(u_b_rho, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1_thc = ref["oei"] - 0.5 * np.einsum("prrq->pq", ref["tei"])
    h1_prime = M.build_h1_prime(h1_thc, S, beta_1, ref["electrons"], W, X)
    ev = np.linalg.eigvalsh(h1_prime)

    lam_h1 = float(np.sum(np.abs(ev)))
    lam_h2 = M.lambda_h2(w, W)
    Lam = lam_h1 + lam_h2
    e_sos = float(ref["ecore"] - ref["electrons"] * beta_1 - (lam_h1 - ev.sum()) - 0.5 * np.sum(W**2))
    gap = UPPER_ENERGY - e_sos
    err1 = M.h1_thc_reconstruction_error(one_body, h1_thc, S, beta_1, ref["electrons"], W, X)
    h2d = M.reconstruct_h2(X) - ref["tei"]
    d = np.arange(N)
    h2d[:, :, d, d] -= 0.5 * S[:, :, None]
    h2d[d, d, :, :] -= 0.5 * S[None, :, :]
    err2 = float(np.linalg.norm(h2d.ravel()))

    print(f"factorization       {args.npz}")
    print(f"R, B, C, N          {R}, {B}, {C}, {N}")
    print(f"ecore               {ref['ecore']:.9f}   eta={ref['electrons']}  beta_1={beta_1:.9f}")
    print()
    print("--- continuous factorization (no angle quantization) ---")
    print(f"error_1             {1e3*err1:.6f} mHa   (budget {1e3*ERROR_2_THRESHOLD:.1f} mHa)")
    print(f"error_2             {1e3*err2:.6f} mHa   = {100*err2/ERROR_2_THRESHOLD:5.1f}% of budget")
    print(f"lambda_h1 (sum|eig h1'|)  {lam_h1:12.6f}   {100*lam_h1/Lam:5.1f}% of Lambda")
    print(f"lambda_h2 (1/4 sum w^2)   {lam_h2:12.6f}   {100*lam_h2/Lam:5.1f}% of Lambda")
    print(f"Lambda              {Lam:.6f}   (floor {LAMBDA_FLOOR}, ratio {Lam/LAMBDA_FLOOR:.3f}x)")
    print(f"E_SOS               {e_sos:.6f}")
    print(f"E_gap               {gap:.6f}   (floor {E_GAP_FLOOR}, margin {gap-E_GAP_FLOOR:.6f})")
    print(f"lambda_eff          {lambda_eff(Lam, gap):.6f}")
    print(f"  trace h1'         {ev.sum():.6f}   sum|ev| - trace = {lam_h1-ev.sum():.6f}")

    print()
    print("--- what each floor alone would buy (hypothetical) ---")
    rows = [
        ("baseline", Lam, gap),
        ("Lambda -> 49.0+ (gap fixed)", 49.0 + 1e-9, gap),
        ("E_gap  -> 3.914898+ (Lambda fixed)", Lam, E_GAP_FLOOR + 1e-9),
        ("both floors", 49.0 + 1e-9, E_GAP_FLOOR + 1e-9),
    ]
    base = lambda_eff(Lam, gap)
    for name, L, g in rows:
        v = lambda_eff(L, g) if 0 < g <= 2 * L else float("nan")
        print(f"  {name:38} Lambda={L:9.4f} E_gap={g:9.6f}  lambda_eff={v:9.6f}  ratio={base/v:6.3f}x")

    print()
    print("--- circuit-side factors at this shape (independent of Lambda) ---")
    for brot in (10, 11, 12, 13, 16):
        co = SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=10, beta=brot)
        heur = co.outer_state_prep() + co.inner_state_prep() + co.givens_rotations()
        hp = co.outer_state_prep_qubit_cost()["persistent"] + co.inner_state_prep_qubit_cost()["persistent"]
        ht = max(co.outer_state_prep_qubit_cost()["temporary"],
                 co.inner_state_prep_qubit_cost()["temporary"])
        print(f"  brot={brot:2}  heuristic_toffoli={heur:8,}  persistent_q={hp:5}  temp_q={ht:5}  "
              f"gives_T_and_Q_a_factor_{1.0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

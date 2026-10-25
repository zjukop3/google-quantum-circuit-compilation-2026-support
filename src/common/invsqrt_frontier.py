#!/usr/bin/env python3
"""Exact invsqrt precision frontier: (Q, Toffolis, worst error, verifier min-margin, Q*C).

The shipped verifier's `worst_error_over_tolerance` equals
`max_x |y1/2**F - 1/sqrt(x)| sqrt(x) / 2**-24` exactly -- confirmed to 8 significant
figures against `notes/runs/invsqrt_sample_full.json` (0.4769116097939472).  So the
analytic fixed-point model is the ground truth and a candidate can be priced in ~1 s
instead of the 4 m 42 s full verification.

One `fixed_point_model` pass per candidate yields BOTH the accuracy ratio and the exact
`min_x (tol(x) - |error(x)|)` the verifier uses to size `rotation_eps`, hence `T` and the
final `C = Toffolis + T/4`.

Also reports the optimized-circuit qubit count (`Q_opt = Q_ref - coef_bits`), since
`src/invsqrt_opt_circuit.py` keeps one fewer coefficient register at identical Toffolis
and identical arithmetic.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data" / "verifiers" / "invsqrt"))

from nr_reference_circuit import Precision, build_nr_circuit, fixed_point_model  # noqa: E402

N_P = 10
MAX_X = 3 * (2**N_P - 2)**2
TOL_REL = 2.0**-24
PRECISION_BITS = 24  # the verifier default: tol(x) = 2^-24 beta / sqrt(x)
BASELINE_SCORE = 2076774.234


def tables():
    """`target` and `allowed` in units of 2^-64 turns, exactly as the verifier builds them."""
    rs = [math.isqrt((1 << 130) // x) for x in range(1, MAX_X + 1)]
    target = np.array([((r + 1) >> 1) & ((1 << 64) - 1) for r in rs], dtype=np.uint64)
    allowed = np.array([min(r >> (PRECISION_BITS + 1), 1 << 63) for r in rs], dtype=np.uint64)
    return target, allowed


def price(prec: Precision, target, allowed, xs, exact) -> dict:
    out = {"delta_bits": prec.delta_bits, "coef_bits": prec.coef_bits,
           "square_bits": prec.square_bits, "product_bits": prec.product_bits,
           "result_bits": prec.result_bits}
    try:
        y1 = fixed_point_model(N_P, prec, xs)
    except (OverflowError, ValueError) as ex:
        out["reject"] = type(ex).__name__
        return out
    F = prec.output_bits(N_P)
    approx = y1 / 2.0**F
    rel = np.abs(approx - exact) / exact
    ratio = float(rel.max()) / TOL_REL
    out["ratio"] = ratio
    if ratio >= 1.0:
        out["reject"] = "over_tolerance"
        return out
    theta = np.left_shift(y1.astype(np.uint64), np.uint64(64 - F))
    used = np.abs(theta.astype(np.int64) - target.astype(np.int64))
    mm = int((allowed.astype(np.int64) - used).min())
    out["min_margin_units"] = mm
    if mm <= 0:
        out["reject"] = "no_margin"
        return out
    c = build_nr_circuit(N_P, prec=prec)
    st = c.operation_statistics()
    Q, Tf = c.num_qubits, st.num_toffoli_like
    nrot, tbase = st.num_z_pow(min_clifford_hierarchy_level=4), st.num_t_like
    eps = math.pi * (mm / 2.0**64) / nrot if nrot else None
    rc = 0.56 * math.log2(1 / eps) + 5.3 if eps else 0.0
    C = Tf + (tbase + nrot * rc) / 4.0
    Q_opt = Q - prec.coef_bits
    out.update(Q_ref=Q, Q_opt=Q_opt, toffolis=Tf, n_rot=nrot, t_base=tbase,
               rotation_eps=eps, C=C, score_ref=Q * C, score_opt=Q_opt * C,
               gain_ref=BASELINE_SCORE / (Q * C), gain_opt=BASELINE_SCORE / (Q_opt * C))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-bits", type=int, nargs="+", default=[12, 13, 14, 15])
    ap.add_argument("--coef-bits", type=int, nargs="+", default=[16, 17, 18, 19])
    ap.add_argument("--square-bits", type=int, nargs="+", default=[27, 28, 29, 30])
    ap.add_argument("--product-bits", type=int, nargs="+", default=[27, 28, 29, 30])
    ap.add_argument("--result-bits", type=int, nargs="+", default=[28, 29, 30, 31, 32])
    ap.add_argument("--out", type=Path, default=HERE.parent / "notes" / "runs" / "invsqrt_frontier.json")
    args = ap.parse_args()

    t0 = time.time()
    target, allowed = tables()
    xs = np.arange(1, MAX_X + 1, dtype=np.int64)
    exact = 1.0 / np.sqrt(xs)
    print(f"[prep] tables + grid ready in {time.time()-t0:.1f}s", flush=True)

    grid = [dict(delta_bits=d, coef_bits=c, square_bits=s, product_bits=p, result_bits=r)
            for d, c, s, p, r in itertools.product(args.delta_bits, args.coef_bits,
                                                   args.square_bits, args.product_bits,
                                                   args.result_bits)]
    print(f"[grid] {len(grid)} candidates", flush=True)
    rows = []
    for i, g in enumerate(grid):
        row = price(Precision(**g), target, allowed, xs, exact)
        rows.append(row)
        if "reject" not in row:
            print(f"  OK  {g['delta_bits']}/{g['coef_bits']}/{g['square_bits']}/"
                  f"{g['product_bits']}/{g['result_bits']}  ratio={row['ratio']:.4f} "
                  f"Q_ref={row['Q_ref']} Q_opt={row['Q_opt']} T={row['toffolis']} "
                  f"n_rot={row['n_rot']} C={row['C']:.2f} "
                  f"score={row['score_opt']:.0f} gain={row['gain_opt']:.4f}x", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(grid)}] {time.time()-t0:.0f}s", flush=True)

    ok = sorted((r for r in rows if "reject" not in r), key=lambda r: r["score_opt"])
    print(f"\nfeasible: {len(ok)}/{len(rows)}  ({time.time()-t0:.0f}s)")
    print(f"\n{'cfg':24s} {'ratio':>7s} {'Q_ref':>6s} {'Q_opt':>6s} {'T':>6s} {'nrot':>5s} "
          f"{'C':>9s} {'score_opt':>10s}  gain")
    for r in ok[:25]:
        cfg = f"{r['delta_bits']}/{r['coef_bits']}/{r['square_bits']}/{r['product_bits']}/{r['result_bits']}"
        print(f"{cfg:24s} {r['ratio']:7.4f} {r['Q_ref']:6d} {r['Q_opt']:6d} {r['toffolis']:6d} "
              f"{r['n_rot']:5d} {r['C']:9.2f} {r['score_opt']:10.0f}  {r['gain_opt']:.4f}x")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

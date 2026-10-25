"""E6 experiment I: freeze the sweep's per-iteration multiplier liveness.

Measures, in the shipped module and on official shots, the per-iteration "does any
multiplier site see `mul == 1`" bitmap, then rebuilds the decoder with the
multipliers of every iteration the majority bitmap marks dead removed, and checks
the official verifier on the same shots.

Relies on `e6_semantics` for the non-invasive probe (`probe='sitexor'`) and for the
freeze (`freeze='iters'`).

Usage
-----
    python src/e6/e6_iteration_freeze.py --n 70 --degree 12 --m 4095 --shots 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import e6_semantics as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--m', type=int, default=4095)
    ap.add_argument('--shots', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-workers', type=int, default=2)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    m, n, degree = args.m, args.n, args.degree
    n_iters = 3 * (n + 1) + 9

    V = S.load_module('vdqi', ROOT / 'data' / 'verifiers' / 'dqi'
                      / 'verify_submission_dqi.py')
    ref = S.load_module('refdqi', ROOT / 'data' / 'verifiers' / 'dqi'
                        / 'dqi_reference_circuit.py')

    spec = V.ProblemSpecDQI.benchmark_instance(m=m, n=n, degree=degree,
                                               num_shots=args.shots)
    tables = V.FieldTables(km.GF2Field(degree))
    rng = np.random.default_rng(20261025)
    wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    shots = V.sample_shots(spec, tables, rng, args.shots, wp)
    bits = V.to_bits(shots[0], degree)

    t0 = time.time()
    F, A, pcirc = S.measure_sitexor_liveness(ref, m, n, degree, shots=args.shots,
                                             bits=bits)
    cf = Counter(tuple(F[:, s].tolist()) for s in range(args.shots))
    ca = Counter(tuple(A[:, s].tolist()) for s in range(args.shots))
    Fm = np.array(cf.most_common(1)[0][0], bool)
    Am = np.array(ca.most_common(1)[0][0], bool)
    majF, majA = tuple(Fm.tolist()), tuple(Am.tolist())
    n_fwd_diff = sum(1 for s in range(args.shots)
                     if tuple(F[:, s].tolist()) != majF)
    n_adj_diff = sum(1 for s in range(args.shots)
                     if tuple(A[:, s].tolist()) != majA)
    print(f'[liveness] forward any-live {int(Fm.sum())}/{n_iters} '
          f'({len(cf)} distinct patterns); adjoint {int(Am.sum())}/{n_iters} '
          f'({len(ca)} distinct); shots differing from majority '
          f'{n_fwd_diff}/{args.shots} = {n_fwd_diff / args.shots:.5f} '
          f'({time.time() - t0:.1f}s)', flush=True)

    t0 = time.time()
    cf_ = S.build_circuit(ref, m, n, degree, frozen_iters=(~Fm, ~Am))
    T_f = cf_.operation_statistics().num_toffoli_like
    T_r = pcirc.operation_statistics().num_toffoli_like
    print(f'[frozen] T {T_r:,} -> {T_f:,} '
          f'({(T_f - T_r) / T_r * 100:+.3f}%) ({time.time() - t0:.1f}s)',
          flush=True)
    ve, pe, T, ok = S.run_verify(V, cf_, spec, shots,
                                 batch_size=args.batch_size,
                                 max_workers=args.max_workers)
    print(f'  value_errors {ve} ({ve / args.shots:.5f}) phase_errors {pe} '
          f'-> {(ve + pe) / args.shots:.5f} wrong', flush=True)

    if args.json:
        Path(args.json).write_text(json.dumps({
            'n': n, 'degree': degree, 'm': m, 'shots': args.shots,
            'n_iters': n_iters,
            'forward_any_live': int(Fm.sum()),
            'adjoint_any_live': int(Am.sum()),
            'distinct_forward_patterns': len(cf),
            'distinct_adjoint_patterns': len(ca),
            'shots_differing_forward': n_fwd_diff,
            'shot_diff_fraction': n_fwd_diff / args.shots,
            'T_ref': T_r, 'T_frozen': T_f,
            'delta_T_fraction': (T_f - T_r) / T_r,
            'value_errors': ve, 'phase_errors': pe,
            'error_fraction': (ve + pe) / args.shots,
        }, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

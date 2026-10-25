"""E4: verify the reset-based invsqrt circuit against the reference, exhaustively for small n_p.

Three checks per `x` and per `n_p`:
  * `y1` (read from a `compute_only` build) equals the shipped `fixed_point_model`;
  * the tracked phase (turns, exact `Fraction`) equals the reference's;
  * `x` is restored and every scratch qubit is back to |0>.

Usage
-----
    ../2609-xingzhi-drug/.venv/bin/python src/e4/invsqrt_reset_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'invsqrt'))

from nr_reference_circuit import Precision, build_nr_circuit, fixed_point_model  # noqa: E402
from invsqrt_equiv import PASS_PREC, run_phases, run_y1  # noqa: E402
from invsqrt_reset_circuit import build_reset_circuit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-p', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    ap.add_argument('--narrow-z', action='store_true', help='also test the one-bit-narrower z (known broken)')
    ap.add_argument('--out', type=Path, default=ROOT / 'notes' / 'runs' / 'e4_reset_check.json')
    args = ap.parse_args()

    report, all_ok = {}, True
    print(f'{"n_p":>4s} {"Q":>5s} {"T":>6s} {"y1==model":>10s} {"phase==ref":>10s} '
          f'{"x ok":>5s} {"clean":>6s}')
    for n_p in args.n_p:
        prec = Precision(**PASS_PREC)
        xs = np.arange(1, 2 ** (2 * n_p + 2), dtype=np.uint64)

        reset = build_reset_circuit(n_p, prec=prec, narrow_z=args.narrow_z)
        model = fixed_point_model(n_p, prec, xs.astype(np.int64))
        y1_ok = bool((run_y1(build_reset_circuit(n_p, prec=prec, compute_only=True, narrow_z=args.narrow_z), n_p, xs)
                      == model.astype(np.uint64)).all())

        ref = run_phases(build_nr_circuit(n_p, prec=prec), n_p, xs)
        got = run_phases(reset, n_p, xs)
        # `read_shot_phase` reports HALF turns, and the verifier compares turns mod 1, so only an
        # EVEN difference in half-turn units is a harmless global phase.  (Comparing the raw
        # half-turn difference as an integer was the bug that let 0.5-turn errors pass here while
        # the real verifier rejected them: see notes/E4_INVSQRT_QUBIT.md section 5.)
        phase_ok = all(((Fraction(g) - Fraction(r)) / 2).denominator == 1
                       for g, r in zip(got['phases'], ref['phases']))
        ok = y1_ok and phase_ok and got['x_restored'] and got['dirty_count'] == 0
        all_ok &= ok
        report[n_p] = {'Q': reset.num_qubits,
                       'T': reset.operation_statistics().num_toffoli_like,
                       'y1_matches_model': y1_ok, 'phase_matches_reference': phase_ok,
                       'x_restored': got['x_restored'], 'dirty_count': got['dirty_count']}
        print(f'{n_p:>4d} {reset.num_qubits:>5d} '
              f'{reset.operation_statistics().num_toffoli_like:>6,d} '
              f'{str(y1_ok):>10s} {str(phase_ok):>10s} {str(got["x_restored"]):>5s} '
              f'{str(got["dirty_count"] == 0):>6s}{"  OK" if ok else "  <-- FAIL"}', flush=True)

    report['all_ok'] = all_ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f'\n{"PASS" if all_ok else "FAIL"} -> {args.out}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

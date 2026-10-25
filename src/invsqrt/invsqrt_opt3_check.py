"""Exhaustive small-`n_p` equivalence check for E4's `invsqrt_opt3_circuit`.

For every `x` in the legal range it asserts, against the shipped reference and the accuracy
model:

* `y1` (the numerator that carries the phase) is bit-identical to `fixed_point_model`;
* the shot phase equals the reference's modulo one full turn (a global phase is free);
* `x` is restored and every other qubit is back in |0>.

Phase comparison units matter: `read_shot_phase` reports HALF turns while the verifier
compares turns mod 1, so only an EVEN half-turn difference is a harmless global phase.

Usage
-----
    ../2609-xingzhi-drug/.venv/bin/python src/e4/invsqrt_opt3_check.py [--n-p 2 3 4 5 6]
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
from invsqrt_opt3_circuit import build_opt3_circuit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-p', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    ap.add_argument('--out', type=Path, default=ROOT / 'notes' / 'runs' / 'e4_opt3_check.json')
    args = ap.parse_args()

    report, all_ok = {}, True
    print(f'{"n_p":>4s} {"Q":>5s} {"T":>6s} {"y1==model":>10s} {"phase==ref":>10s} '
          f'{"x ok":>5s} {"clean":>6s}')
    for n_p in args.n_p:
        prec = Precision(**PASS_PREC)
        xs = np.arange(1, 2 ** (2 * n_p + 2), dtype=np.uint64)

        circuit = build_opt3_circuit(n_p, prec=prec)
        model = fixed_point_model(n_p, prec, xs.astype(np.int64))
        y1 = run_y1(build_opt3_circuit(n_p, prec=prec, compute_only=True), n_p, xs)
        y1_ok = bool((y1 == model.astype(np.uint64)).all())

        ref = run_phases(build_nr_circuit(n_p, prec=prec), n_p, xs)
        got = run_phases(circuit, n_p, xs)
        phase_ok = all(((Fraction(g) - Fraction(r)) / 2).denominator == 1
                       for g, r in zip(got['phases'], ref['phases']))

        ok = y1_ok and phase_ok and got['x_restored'] and got['dirty_count'] == 0
        all_ok &= ok
        toffoli = circuit.operation_statistics().num_toffoli_like
        report[str(n_p)] = {'n_p': n_p, 'Q': circuit.num_qubits, 'T': toffoli,
                            'y1_matches_model': y1_ok, 'phase_matches_reference': phase_ok,
                            'x_restored': got['x_restored'], 'dirty_count': got['dirty_count']}
        print(f'{n_p:>4d} {circuit.num_qubits:>5d} {toffoli:>6,d} {str(y1_ok):>10s} '
              f'{str(phase_ok):>10s} {str(bool(got["x_restored"])):>5s} '
              f'{str(got["dirty_count"] == 0):>6s}{"  OK" if ok else "  <-- FAIL"}', flush=True)

    report['all_ok'] = all_ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f'\n{"PASS" if all_ok else "FAIL"} -> {args.out}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

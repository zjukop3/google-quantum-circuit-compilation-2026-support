"""E5: exhaustive small-`n_p` equivalence check for the binary-flags circuit.

For every legal `x` and every `n_p` in the sweep, asserts against the shipped model and
reference:

  * `y1` is bit-identical to `fixed_point_model` (the accuracy ground truth);
  * the shot phase equals the reference's modulo one full turn (a global phase is free);
  * `x` is restored and every scratch qubit (thermometer included) is back in |0>.

`read_shot_phase` reports HALF turns while the verifier compares turns mod 1, so only an
EVEN half-turn difference is a harmless global phase.

Usage
-----
    ../2609-xingzhi-drug/.venv/bin/python src/e5/e5_flags_check.py [--n-p 2 3 4 5 6]
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
for p in (ROOT / 'src', HERE, ROOT / 'src' / 'e4', ROOT / 'data' / 'verifiers' / 'invsqrt'):
    sys.path.insert(0, str(p))

from nr_reference_circuit import Precision, build_nr_circuit, fixed_point_model  # noqa: E402
from invsqrt_equiv import PASS_PREC, run_phases, run_y1  # noqa: E402
from invsqrt_opt3_circuit import build_opt3_circuit  # noqa: E402
from invsqrt_flags_circuit import build_flags_circuit, thermometer_to_binary_terms  # noqa: E402


def check_thermometer_terms() -> bool:
    """The CX-only thermometer->binary decode must be exact for every n_p in the sweep."""
    ok = True
    for n_p in range(2, 33):
        sb = max(1, (n_p + 1 - 1).bit_length())
        terms = thermometer_to_binary_terms(n_p, sb)
        for s in range(n_p + 1):
            flags = [1 if s >= k + 1 else 0 for k in range(n_p)]  # thermometer of s
            got = sum(sum(flags[i] for i in idx) % 2 << j for j, idx in enumerate(terms))
            if got != s:
                print(f'  decode mismatch n_p={n_p} s={s}: got {got}')
                ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-p', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    ap.add_argument('--out', type=Path, default=ROOT / 'notes' / 'runs' / 'e5_flags_check.json')
    args = ap.parse_args()

    decode_ok = check_thermometer_terms()
    print(f'thermometer->binary decode exact for n_p=2..32: {decode_ok}\n')

    report, all_ok = {'decode_exact': decode_ok}, decode_ok
    print(f'{"n_p":>4s} {"Q":>5s} {"T":>7s} {"Q_e4":>5s} {"T_e4":>7s} {"y1==model":>10s} '
          f'{"phase==ref":>10s} {"x ok":>5s} {"clean":>6s}')
    for n_p in args.n_p:
        prec = Precision(**PASS_PREC)
        width = 2 * n_p + 2
        xs = np.arange(1, 2**width, dtype=np.uint64)

        circuit = build_flags_circuit(n_p, prec=prec)
        e4 = build_opt3_circuit(n_p, prec=prec)
        model = fixed_point_model(n_p, prec, xs.astype(np.int64))
        y1 = run_y1(build_flags_circuit(n_p, prec=prec, compute_only=True), n_p, xs)
        y1_ok = bool((y1 == model.astype(np.uint64)).all())

        ref = run_phases(build_nr_circuit(n_p, prec=prec), n_p, xs)
        got = run_phases(circuit, n_p, xs)
        phase_ok = all(((Fraction(g) - Fraction(r)) / 2).denominator == 1
                       for g, r in zip(got['phases'], ref['phases']))

        ok = y1_ok and phase_ok and got['x_restored'] and got['dirty_count'] == 0
        all_ok &= ok
        st, st4 = circuit.operation_statistics(), e4.operation_statistics()
        report[str(n_p)] = {'n_p': n_p, 'Q': circuit.num_qubits, 'T': st.num_toffoli_like,
                            'Q_e4': e4.num_qubits, 'T_e4': st4.num_toffoli_like,
                            'y1_matches_model': y1_ok, 'phase_matches_reference': phase_ok,
                            'x_restored': got['x_restored'], 'dirty_count': got['dirty_count']}
        print(f'{n_p:>4d} {circuit.num_qubits:>5d} {st.num_toffoli_like:>7,d} '
              f'{e4.num_qubits:>5d} {st4.num_toffoli_like:>7,d} '
              f'{str(y1_ok):>10s} {str(phase_ok):>10s} {str(bool(got["x_restored"])):>5s} '
              f'{str(got["dirty_count"] == 0):>6s}{"  OK" if ok else "  <-- FAIL"}', flush=True)

    report['all_ok'] = all_ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f'\n{"PASS" if all_ok else "FAIL"} -> {args.out}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

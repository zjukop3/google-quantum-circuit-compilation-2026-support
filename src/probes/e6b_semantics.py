"""E6b: official small-scale semantic check of the controlled `div_ixor_into_q` variant.

Reuses the E4 harness (`src/e4/semantic.py`), which in turn reuses the *shipped* verifier's
sampler and register validation, so a PASS here means exactly what the shipped `verify()` means
by it: on every shot the syndrome registers come back unchanged, every non-syndrome qubit
returns to |0>, and the accumulated phase equals `(-1) ** sum_j popcount(e_j & M_j)` for the
planted error word.

This is the check E4's diagnostic variant failed 512/512: dropping the sub-step control breaks
the syndrome restoration.  The controlled rebuild must pass it.

Usage
-----
    PYTHONPATH=src/e6/patch python src/e6/e6b_semantics.py --variant div_ixor_q_controlled --shots 512
    PYTHONPATH=src/e6/patch python src/e6/e6b_semantics.py --variant ref --variant div_ixor_q_controlled
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src' / 'e4'))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'dqi'))

import verify_submission_dqi as V  # noqa: E402  the shipped verifier, imported as-is
from semantic import check_circuit, make_shots  # noqa: E402  the E4 harness
from e6b_variants import load_variant  # noqa: E402


def run(variant: str, m: int, n: int, degree: int, num_shots: int,
        shot_seed: int, batch_size: int, mode: str) -> dict:
    ns, note = load_variant(variant)
    spec = V.ProblemSpecDQI.benchmark_instance(m=m, n=n, degree=degree, num_shots=num_shots)
    tables = V.FieldTables(km.GF2Field(degree))
    shots = make_shots(spec, tables, shot_seed, num_shots, mode)

    t0 = time.time()
    circuit = ns['build_reference_circuit'](m, n, degree)
    t_build = time.time() - t0
    t1 = time.time()
    res = check_circuit(circuit, spec, shots, batch_size=batch_size)
    t_sim = time.time() - t1
    return {'variant': variant, 'note': note, 'mode': mode,
            'num_qubits': circuit.num_qubits,
            'static_toffoli': circuit.operation_statistics().num_toffoli_like,
            'build_seconds': round(t_build, 2), 'sim_seconds': round(t_sim, 2), **res}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', action='append', default=None)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--shot-seed', type=int, default=20261025)
    ap.add_argument('--mode', choices=('official', 'maxweight'), default='official')
    args = ap.parse_args()

    print(f'kickmix : {km.__file__}')
    variants = args.variant or ['ref', 'div_ixor_q_controlled']
    for name in variants:
        r = run(name, args.m, args.n, args.degree, args.shots,
                args.shot_seed, args.batch_size, args.mode)
        print('=' * 78)
        print(f"variant    : {r['variant']}")
        print(f"note       : {r['note']}")
        print(f"Q / T      : {r['num_qubits']} / {r['static_toffoli']:,}")
        print(f"build/sim  : {r['build_seconds']} s / {r['sim_seconds']} s")
        print(f"shots      : {r['num_shots']} ({r['mode']})")
        print(f"errors     : value={r['value_errors']} phase={r['phase_errors']} "
              f"total={r['total_errors']}")
        print(f"VERDICT    : {r['verdict']}")


if __name__ == '__main__':
    main()

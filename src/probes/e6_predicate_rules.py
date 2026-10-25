"""E6 experiment K: can the sub-step-1 termination predicate be replaced by a
structurally constant equivalent?

The deviating shots are triggered by

    aux = c1 AND NOT(n_B == 1)
    flag ^= aux AND (reg_b[m-2] == 0)          # dqi_reference_circuit.py:268-272

`reg_b[m-2]` is a fresh element of GF(2^b), so the event has per-opportunity
probability 2^-b -- the "2^-12" of the E6 brief.  This script tests, with the
official verifier on official shots, whether that term can be replaced by a rule
that is structurally constant on the `aux` support:

    r1  n_B == 1        -- identically 0 there (deviation impossible)
    r2  deleted         -- same as r1, but also removes the gate
    r3  cx(aux, flag)   -- identically 1
    r4  n_A == n_B      -- the *legitimate* sub-step-0 rule

Usage
-----
    python src/e6/e6_predicate_rules.py --n 70 --degree 12 --m 280 --shots 1024
    python src/e6/e6_predicate_rules.py --n 6 --degree 6 --m 63 --shots 4096
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import e6_semantics as S  # noqa: E402

PREDICATE = 'builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)'

RULES = {
    'ref': None,
    'r1_nB_eq_1': 'builder.flip_if_equal(n_B, 1, target=flag, control=aux)',
    'r2_delete': 'pass  # term deleted',
    'r3_always': 'builder.cx(aux, flag)  # term identically 1',
    'r4_nA_eq_nB': 'builder.flip_if_equal(n_A, n_B, target=flag, control=aux)',
}


def install_rule(ref, rule_src):
    src = inspect.getsource(ref.ProosAndZalkaDecoder._update_flag)
    assert src.count(PREDICATE) == 1
    src = src.replace(PREDICATE, rule_src, 1)
    src = '\n'.join(l[4:] if l.startswith('    ') else l for l in src.split('\n'))
    ns = {}
    exec(compile(src, '<rule>', 'exec'), vars(ref), ns)
    return ns['_update_flag']


def build_with_rule(ref, m, n, degree, rule_src):
    decoder = ref.ProosAndZalkaDecoder(n, km.GF2Field(degree))
    fn = install_rule(ref, rule_src)
    decoder._update_flag = fn.__get__(decoder, type(decoder))
    b = km.CircuitBuilder()
    syn = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    err = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    decoder.build_syndrome_decoding_mbuc(b, syndromes=syn, error_mbuc=err,
                                         n_iters=3 * (n + 1) + 9)
    return b.finish_circuit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--shots', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-workers', type=int, default=2)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    m, n, degree = args.m, args.n, args.degree

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

    results = {}
    for name, rule in RULES.items():
        t0 = time.time()
        if rule is None:
            circ = ref.build_reference_circuit(m, n, degree)
        else:
            circ = build_with_rule(ref, m, n, degree, rule)
        ve, pe, T, ok = S.run_verify(V, circ, spec, shots,
                                     batch_size=args.batch_size,
                                     max_workers=args.max_workers)
        results[name] = dict(value_errors=ve, phase_errors=pe,
                             toffoli_count=T, eps_ok=ok)
        print(f'rule={name:12s} value_errors={ve:6d} ({ve / args.shots:.4e}) '
              f'phase_errors={pe:6d} T={T:,.0f} ({time.time() - t0:.0f}s)',
              flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {'n': n, 'degree': degree, 'm': m, 'shots': args.shots,
             **{f'{k}_{kk}': vv for k, v in results.items() for kk, vv in v.items()}},
            indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

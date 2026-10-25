"""E7-G — small-scale semantic equivalence of the frozen circuit (m <= 280).

Imports the *official* verifier helpers (`sample_shots`, `optimal_dqi_weights`,
`validate_registers`, `to_bits`, `verify`) rather than copying them, and runs the
shipped reference circuit and the frozen circuit on **the same shots**
(`seed=` is passed explicitly, otherwise `verify` derives the seed from the
circuit hash and the two circuits would see different samples).

Three regimes:
  * official distribution, 512 shots
  * maximal-weight stress, 1024 shots  (error weight forced to l = n//2)
  * official distribution, 2048 shots

Usage
-----
    $P src/e7/e7_verify_small.py --model nA070_69_K2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'e6'))
sys.path.insert(0, str(HERE))

import e6_semantics as ES          # noqa: E402

import kickmix as km               # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', default=str(ROOT / 'notes/runs/e7/freeze_sites.json'))
    ap.add_argument('--model', default='nA070_69_K2')
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--max-workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    spec_mod = importlib.util.spec_from_file_location(
        'vdqi', ROOT / 'data' / 'verifiers' / 'dqi' / 'verify_submission_dqi.py')
    V = importlib.util.module_from_spec(spec_mod)
    sys.modules['vdqi'] = V
    spec_mod.loader.exec_module(V)
    ref = load_module('e7ref_small', ROOT / 'data' / 'verifiers' / 'dqi'
                      / 'dqi_reference_circuit.py')

    FS = None
    n_frozen = 0
    if args.model != 'none':
        js = json.loads(Path(args.freeze).read_text())[args.model]
        n_frozen = len(js['sites'])
        FS = np.zeros((2, 222, 73), dtype=bool)
        for it, i in js['sites']:
            FS[0, int(it), int(i)] = True
            FS[1, int(it), int(i)] = True
    circ_ref = ES.build_circuit(ref, args.m, args.n, args.degree, frozen_sites=None)
    circ_frz = (ES.build_circuit(ref, args.m, args.n, args.degree, frozen_sites=FS)
                if FS is not None else circ_ref)
    print(f'[build] m={args.m} frozen_sites={n_frozen} '
          f'T(ref)={circ_ref.operation_statistics().num_toffoli_like:,} '
          f'T(frozen)={circ_frz.operation_statistics().num_toffoli_like:,}')

    spec = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=args.n, degree=args.degree,
                                               num_shots=256)
    tables = V.FieldTables(km.GF2Field(args.degree))
    max_err = spec.max_errors
    regimes = [('official-512', 512, None),
               ('maxweight-1024', 1024, 'max'),
               ('official-2048', 2048, None)]
    out = {'model': args.model, 'm': args.m, 'results': {}}
    for name, shots, mode in regimes:
        rng = np.random.default_rng(args.seed)
        if mode == 'max':
            wp = np.zeros(max_err + 1)
            wp[max_err] = 1.0
        else:
            wp = V.optimal_dqi_weights(spec.m, max_err, spec.degree, spec.r) ** 2
        sy, mb, ph = V.sample_shots(spec, tables, rng, shots, wp)
        specn = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=args.n,
                                                    degree=args.degree, num_shots=shots)
        rec = {}
        for tag, cc in (('ref', circ_ref), ('frozen', circ_frz)):
            t0 = time.time()
            _ = None
            ve, pe, tof, passed = ES.run_verify(V, cc, specn, (sy, mb, ph),
                                                batch_size=args.batch_size,
                                                max_workers=args.max_workers)
            rec[tag] = {'value_errors': ve, 'phase_errors': pe,
                        'toffoli_count': tof, 'eps_assert_ok': bool(passed),
                        'sec': round(time.time() - t0, 1)}
        rec['identical'] = (rec['ref']['value_errors'] == rec['frozen']['value_errors']
                            and rec['ref']['phase_errors'] == rec['frozen']['phase_errors'])
        out['results'][name] = rec
        print(f'[{name}] ref  value/phase = {rec["ref"]["value_errors"]}/'
              f'{rec["ref"]["phase_errors"]}  T={rec["ref"]["toffoli_count"]:,.0f}')
        print(f'[{name}] frz  value/phase = {rec["frozen"]["value_errors"]}/'
              f'{rec["frozen"]["phase_errors"]}  T={rec["frozen"]["toffoli_count"]:,.0f}'
              f'   identical={rec["identical"]}', flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)
    ok = all(v['identical'] for v in out['results'].values())
    print(f'[summary] all regimes identical: {ok}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

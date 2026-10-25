"""E8-G — small-scale semantic equivalence of the `cswap`-frozen circuit (m <= 280).

Imports the *official* verifier helpers (`sample_shots`, `optimal_dqi_weights`,
`validate_registers`, `to_bits`, `verify`) rather than copying them, and runs the
shipped reference circuit and the frozen circuit on **the same shots**
(`seed=` is passed explicitly, otherwise `verify` derives the seed from the
circuit hash and the two circuits would see different samples).

Three regimes:
  * official distribution,  512 shots
  * maximal-weight stress, 1024 shots  (error weight forced to l = n//2)
  * official distribution, 2048 shots

Usage
-----
    $P src/e8/e8_verify_small.py --freeze build/e8/cswap_freeze.npz
    $P src/e8/e8_verify_small.py --freeze build/e8/stacked_freeze.npz --mul-freeze notes/runs/e7/freeze_sites.json
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
import e8_src as E8                # noqa: E402

import kickmix as km               # noqa: E402

FAMILIES = ('swap', 'shift', 'sweep')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_cswap_freeze(path: Path, n: int, fams=FAMILIES):
    z = np.load(path)
    U = {}
    for f in fams:
        if f in z:
            U[f] = z[f]
    return U, E8.full_freeze(U, n)


def load_mul_freeze(path: Path, n: int, model: str):
    js = json.loads(path.read_text())
    spec = js[model] if model in js else js
    FS = np.zeros(E8.grid_shapes(n)['mul'], dtype=bool)
    for it, i in spec['sites']:
        FS[0, int(it), int(i)] = True
        FS[1, int(it), int(i)] = True
    return FS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', default=None, help='npz of per-pass cswap freeze masks')
    ap.add_argument('--mul-freeze', default=None, help='E7-style freeze_sites.json')
    ap.add_argument('--mul-model', default='nA070_69_K2')
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--max-workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    spec_mod = importlib.util.spec_from_file_location(
        'vdqi', ROOT / 'data' / 'verifiers' / 'dqi' / 'verify_submission_dqi.py')
    V = importlib.util.module_from_spec(spec_mod)
    sys.modules['vdqi'] = V
    spec_mod.loader.exec_module(V)

    ref = ES.load_module('e8ref_small', ROOT / 'data' / 'verifiers' / 'dqi'
                         / 'dqi_reference_circuit.py')

    frozen, n_sites = {}, {}
    if args.freeze:
        z = np.load(args.freeze)
        for f in FAMILIES + ('mul',):
            if f in z:
                frozen[f] = z[f]
                n_sites[f] = int(z[f].sum())
    if args.mul_freeze:
        frozen['mul'] = load_mul_freeze(Path(args.mul_freeze), n, args.mul_model)[0]
        n_sites['mul'] = int(frozen['mul'].sum())
    fams = tuple(frozen)

    # reference (shipped) + frozen (e8-patched source)
    ns_ref = {'__name__': 'e8sm_ref'}
    exec(compile(E8.REF.read_text(), str(E8.REF), 'exec'), ns_ref)
    c_ref, _d, _s = E8.build_circuit(ns_ref, args.m, n, degree)

    ns_frz = E8.load_module('small', freeze=True, families=fams or None)
    c_frz, _d, _s = E8.build_circuit(ns_frz, args.m, n, degree,
                                     freeze=E8.full_freeze(frozen, n) if frozen else None)

    print(f'[build] m={args.m} families={fams} '
          f'frozen(per-pass sites)={n_sites}')
    print(f'[build] T(ref)={c_ref.operation_statistics().num_toffoli_like:,} '
          f'T(frozen)={c_frz.operation_statistics().num_toffoli_like:,} '
          f'Q(ref)={c_ref.num_qubits} Q(frozen)={c_frz.num_qubits}')

    spec = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=n, degree=degree,
                                              num_shots=256)
    tables = V.FieldTables(km.GF2Field(degree))
    max_err = spec.max_errors
    regimes = [('official-512', 512, None),
               ('maxweight-1024', 1024, 'max'),
               ('official-2048', 2048, None)]
    out = {'m': args.m, 'n': n, 'families': list(fams), 'frozen_sites': n_sites,
           'T_ref': int(c_ref.operation_statistics().num_toffoli_like),
           'T_frozen': int(c_frz.operation_statistics().num_toffoli_like),
           'results': {}}
    for name, shots, mode in regimes:
        rng = np.random.default_rng(args.seed)
        if mode == 'max':
            wp = np.zeros(max_err + 1)
            wp[max_err] = 1.0
        else:
            wp = V.optimal_dqi_weights(spec.m, max_err, degree, spec.r) ** 2
        sy, mb, ph = V.sample_shots(spec, tables, rng, shots, wp)
        specn = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=n, degree=degree,
                                                    num_shots=shots)
        rec = {}
        for tag, cc in (('ref', c_ref), ('frozen', c_frz)):
            t0 = time.time()
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
    raise SystemExit(main())

"""E8-F — exact T audit of a `cswap` freeze mask.

`e8_src.census()` records every `cswap` call the unfrozen circuit actually emits
(some `sweep_positions` sites carry no `cswap` at all, because the Python-level
`swaps` list is empty there -- those sites can appear "freezable" in a union but
save nothing).  This script intersects a freeze mask with the census and prints
the implied delta-T, then optionally rebuilds the circuit and checks the number.

Usage
-----
    $P src/e8/e8_audit.py --freeze build/e8/cswap_freeze.npz --m 4095
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import e8_src as E8  # noqa: E402

FAMILIES = ('swap', 'shift', 'sweep')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', required=True)
    ap.add_argument('--m', type=int, default=4095)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--no-build', action='store_true')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    sh = E8.grid_shapes(n)
    m, ni = sh['m'], sh['n_iters']
    swap_t = 2 * (m.bit_length() + 1) + m * degree

    z = np.load(args.freeze)
    F = {f: z[f] for f in FAMILIES if f in z}
    print('[freeze] ' + ' '.join(f'{f}={int(F[f].sum())}' for f in F))

    t0 = time.time()
    cen = E8.census(n, degree, args.n)['census']
    print(f'[census] {time.time() - t0:.1f}s  '
          + ' '.join(f'{k}={len(v)}' for k, v in cen.items()))

    # emitted sets (per pass, from the single-pass record; the mask covers both)
    emit = {}
    emit['swap'] = np.zeros((ni,), dtype=bool)
    for it, i, _k in cen['swap'][:ni]:
        emit['swap'][it] = True
    emit['shift'] = np.ones((ni, m - 1), dtype=bool)
    emit['sweep'] = np.zeros((ni, m), dtype=bool)
    for it, i, k in cen['sweep'][:ni] if False else cen['sweep']:
        if k and k > 0:
            emit['sweep'][it, i] = True
    print('[emitted/pass] ' + ' '.join(f'{f}={int(emit[f].sum())}' for f in FAMILIES))

    rep = {}
    total = 0
    for f in FAMILIES:
        if f not in F:
            continue
        eff = F[f] & emit[f]           # frozen AND actually emitted
        per = swap_t if f == 'swap' else degree
        t = int(eff.sum()) * per * 2   # both passes
        rep[f] = {'freeze_sites': int(F[f].sum()), 'emitted_sites': int(emit[f].sum()),
                  'effective_sites': int(eff.sum()), 'T_per_site': per,
                  'delta_T': t}
        total += t
        print(f'[{f:6s}] frozen={int(F[f].sum())} emitted={int(emit[f].sum())} '
              f'effective={int(eff.sum())} x {per} T x 2 = -{t:,} T')
    rep['delta_T_total'] = total
    print(f'[total] expected delta_T = -{total:,}')

    if not args.no_build:
        t0 = time.time()
        ns = E8.load_module('audit', freeze=True)
        fs = E8.full_freeze(F, n)
        m_ch = max(1, args.m)
        try:
            c, _d, _s = E8.build_circuit(ns, m_ch, n, degree, freeze=fs)
            st = c.operation_statistics()
            rep['built_T'] = int(st.num_toffoli_like)
            rep['built_Q'] = c.num_qubits
            print(f'[build] m={args.m} T={st.num_toffoli_like:,} Q={c.num_qubits} '
                  f'({time.time() - t0:.1f}s)')
        except Exception as exc:  # noqa: BLE001
            rep['build_error'] = f'{type(exc).__name__}: {exc}'
            print('[build] failed:', rep['build_error'])
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

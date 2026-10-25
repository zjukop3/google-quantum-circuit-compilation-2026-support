"""E8-I — how big *is* the support of the `cswap` control bitmaps?

The fix-point's freeze set is `complement(union of every sample seen so far)`, so
its converged size is `grid - support`, where the *support* is the set of sites
that are lit on at least one shot of the grading distribution.  A single
102,400-shot sample under-estimates the support, so the fix-point erodes a little
every round.  This script measures the growth curve directly: draw K independent
official-distribution samples and accumulate the union, printing after each one.

That gives
  * a lower bound on the support (and how fast it saturates), and
  * a much better seed union for the fix-point loop.

Usage
-----
    $P src/e8/e8_union_trend.py --samples 16 --shots 25600 \
        --save notes/runs/e8/union_seed_big.npz --json notes/runs/e8/union_trend.json
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
sys.path.insert(0, str(ROOT / 'src' / 'e6'))
sys.path.insert(0, str(ROOT / 'src' / 'e7'))
sys.path.insert(0, str(HERE))

import e8_probe as P             # noqa: E402
import e8_src as E8              # noqa: E402
import e7_diag_official as D     # noqa: E402

FAMILIES = ('swap', 'shift', 'sweep')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--shots', type=int, default=25600)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--chunk', type=int, default=2560)
    ap.add_argument('--base-seed', type=int, default=90210)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--save', default=None)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    circ, dec, syn = P.build_probe(n, degree)
    U = {f: np.zeros(E8.grid_shapes(n)[f][1:], dtype=bool) for f in FAMILIES}
    trend = []
    t0 = time.time()
    for k in range(args.samples):
        base = args.base_seed + 7919 * k
        sy, _mb, _ph = D.official_sample(base, args.shots, args.batch_size,
                                         n=n, degree=degree)
        res = P.probe_sample(circ, dec, syn, sy, degree, n=n, chunk=args.chunk,
                             families=FAMILIES, verbose=False)
        add = {f: int((res['union'][f] & ~U[f]).sum()) for f in FAMILIES}
        for f in FAMILIES:
            U[f] |= res['union'][f]
        trend.append({'k': k, 'base_seed': base,
                      'sample_union': {f: int(res['union'][f].sum()) for f in FAMILIES},
                      'union': {f: int(U[f].sum()) for f in FAMILIES},
                      'added': add})
        freezable = {f: int((~U[f]).sum()) for f in FAMILIES}
        print(f'[sample {k:2d}] own=' + ' '.join(
            f'{f}={int(res["union"][f].sum())}' for f in FAMILIES)
            + '  U=' + ' '.join(f'{f}={int(U[f].sum())}' for f in FAMILIES)
            + '  +' + ' '.join(f'{f}={add[f]}' for f in FAMILIES)
            + '  freezable=' + ' '.join(f'{f}={freezable[f]}' for f in FAMILIES)
            + f'  ({time.time() - t0:.0f}s)', flush=True)

    T = 0
    sh = E8.grid_shapes(n)
    swap_t = 2 * (sh['m'].bit_length() + 1) + sh['m'] * degree
    T = int((~U['swap']).sum()) * swap_t * 2 \
        + (int((~U['shift']).sum()) + int((~U['sweep']).sum())) * degree * 2
    print(f'[result] after {args.samples * args.shots:,} shots: '
          + ' '.join(f'{f}_lit={int(U[f].sum())}' for f in FAMILIES)
          + f'  freezable T={T:,}')
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save, **U)
        print('wrote', args.save)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {'samples': args.samples, 'shots_each': args.shots, 'trend': trend,
             'freezable_T': T}, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""E6 experiment E: the exact shape of one deviation.

Find deviating shots, then dump

  * the `ctrl` (sub-step decode) trajectory, reference vs deviating, iteration by
    iteration, around the first and the last difference;
  * the target iteration set `c0 XOR c3` (the iterations on which
    `_multiply_and_add_or_sub` actually touches `reg_b`);
  * the *slot-level* symmetric difference: which `(iteration, slot)` positions
    would be mis-frozen if Stage A froze the query set instead of the liveness
    bitmap;
  * the `reg_b` word values, to see whether the deviation is a *window shift*.

Usage
-----
    python src/e6/e6_shape.py --n 70 --shots 4096 --batches 8 --show 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src' / 'e3'))

import e6_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--batches', type=int, default=8,
                    help='total simulated shots = shots x batches')
    ap.add_argument('--per-batch', type=int, default=512)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--regime', default='natural', choices=('generic', 'natural'))
    ap.add_argument('--ref-shots', type=int, default=512)
    ap.add_argument('--show', type=int, default=2)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    circ, dec, n_iters, m_e, syn = C.build(n, degree)
    rng = np.random.default_rng(999)
    ref_data = C.gen_data('generic', n, degree, args.ref_shots, rng)
    ref = C.run_batch(circ, dec, syn, ref_data, degree, seed=77, n_iters=n_iters)
    refmat = ref['ctrl'].transpose(2, 0, 1)
    refsig = refmat.reshape(args.ref_shots, -1)
    cnt = Counter(refsig[i].tobytes() for i in range(args.ref_shots))
    ref_key, n_maj = cnt.most_common(1)[0]
    ri = refsig.tobytes().find(ref_key) // refsig.shape[1]
    ref_ctrl = refmat[ri]
    ref_tgt = ref_ctrl[:, 0] ^ ref_ctrl[:, 3]
    ref_all = ref_ctrl.sum(axis=1)
    print(f'[build] Q={circ.num_qubits} n_iters={n_iters} m_e={m_e}')
    print(f'[ref] majority {n_maj}/{args.ref_shots}; target iters={int(ref_tgt.sum())}, '
          f'all-active iters={int(ref_all.sum())}', flush=True)

    shown = 0
    n_dev = 0
    pos_sym = []
    tgt_sym = []
    tot = 0
    for bi in range(args.batches):
        rng = np.random.default_rng(1000 * (bi + 1) + args.seed)
        data = C.gen_data(args.regime, n, degree, args.per_batch, rng)
        tr = C.run_batch(circ, dec, syn, data, degree, seed=7 + bi, n_iters=n_iters)
        mat = tr['ctrl'].transpose(2, 0, 1)
        eq = np.all(mat == ref_ctrl[None, :, :], axis=2)
        for s in range(args.per_batch):
            if eq[s].all():
                continue
            n_dev += 1
            tgt = mat[s][:, 0] ^ mat[s][:, 3]
            tgt_sym.append(int((tgt != ref_tgt).sum()))
            # slot-level: the sweep touches each of the m slots on a target iteration
            a = np.repeat(tgt, m_e)
            b = np.repeat(ref_tgt, m_e)
            pos_sym.append(int((a != b).sum()))
            if shown < args.show and args.json is None:
                shown += 1
                d = np.nonzero(~eq[s])[0]
                print(f'\n--- deviating shot #{n_dev} (batch {bi} idx {s}) ---')
                print(f'  events at c1 iters: '
                      f'{np.nonzero(tr["rb"][:, s] & tr["c1"][:, s])[0].tolist()}')
                print(f'  first ctrl divergence {int(d[0])}, '
                      f'last {int(d[-1])}, n diverging {len(d)}')
                print(f'  target-bitmap symdiff {tgt_sym[-1]} of {n_iters} iterations')
                print(f'  target iters ref: {np.nonzero(ref_tgt)[0][:12].tolist()} ...')
                print(f'  target iters dev: {np.nonzero(tgt)[0][:12].tolist()} ...')
                print(f'  ref target iters (last 8): {np.nonzero(ref_tgt)[0][-8:].tolist()}')
                print(f'  dev target iters (last 8): {np.nonzero(tgt)[0][-8:].tolist()}')
                lo = max(0, int(d[0]) - 2)
                hi = min(n_iters, int(d[0]) + 8)
                print('  iter | ref ctrl | dev ctrl | refT devT  (around first divergence)')
                for it in range(lo, hi):
                    print(f'  {it:4d} | {ref_ctrl[it].tolist()} | {mat[s][it].tolist()}'
                          f' |  {int(ref_tgt[it])}    {int(tgt[it])}')
                lo2 = max(0, int(d[-1]) - 5)
                print('  iter | ref ctrl | dev ctrl | refT devT  (around last divergence)')
                for it in range(lo2, n_iters):
                    print(f'  {it:4d} | {ref_ctrl[it].tolist()} | {mat[s][it].tolist()}'
                          f' |  {int(ref_tgt[it])}    {int(tgt[it])}')
        tot += args.per_batch
        print(f'[batch {bi}] dev={n_dev} / {tot}', flush=True)

    tgt_sym = np.asarray(tgt_sym)
    pos_sym = np.asarray(pos_sym)
    print('\n== E. summary ============================================')
    print(f'shots {tot}, deviating {n_dev}, p = {n_dev / tot:.5f}')
    if n_dev:
        print(f'  target-bitmap symdiff : min {tgt_sym.min()} median '
              f'{int(np.median(tgt_sym))} max {tgt_sym.max()} of {n_iters}')
        print(f'  slot-level symdiff    : min {pos_sym.min()} median '
              f'{int(np.median(pos_sym))} max {pos_sym.max()} of {n_iters * m_e}')
        print(f'  slot-level as fraction: median '
              f'{np.median(pos_sym) / (n_iters * m_e):.4f}')
    if args.json:
        Path(args.json).write_text(json.dumps({
            'n': n, 'shots': tot, 'n_dev': n_dev, 'p': n_dev / tot,
            'tgt_symdiff': {'min': int(tgt_sym.min()), 'median': int(np.median(tgt_sym)),
                            'max': int(tgt_sym.max())} if n_dev else None,
            'slot_symdiff': {'min': int(pos_sym.min()), 'median': int(np.median(pos_sym)),
                             'max': int(pos_sym.max())} if n_dev else None,
            'n_iters': n_iters, 'm_e': m_e,
        }, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

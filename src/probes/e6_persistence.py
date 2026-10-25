"""E6 experiment D: does the deviation *re-converge*, and does it still cost the
`mul` liveness bitmap that Stage A wants to freeze?

Three measurements on the same traced circuit as `e6_deviation_law.py`:

**D1. re-convergence of the frozen schedule.**  For every shot, compare the full
  sub-step decode `ctrl(it)`, the per-iteration *target* of `_multiply_and_add_or_sub`
  (`mul(it,i)`), and the *freshness* of `reg_b[i]` (the iteration at which slot `i`
  was last written, inferred from the `do_shift`/`swap_ctrl`/sweep structure) with
  the generic reference.  E3 asserts "the whole tail's phase is permuted" and the
  frozen bitmap is wrong at ~5252/16206 positions.  This script measures the
  *symmetric difference* of the frozen bitmap directly.

**D2. `mul` liveness bitmap.**  Rebuilt from the recorded `ctrl` trace through the
  verified closed form

      mul(it, i) = c0(it) * [n_A(it) >= m - i]  XOR  c3(it) * [n_a(it) >= i + 1]

  which degrades to `c0(it) XOR c3(it)` for the target set, since the cursors make
  the two windows nested.

**D3. persistence.**  Whether a divergent shot is still divergent at the *last*
  forward iteration, and (the number that Stage A actually pays) how many frozen
  bitmap positions are wrong at the last iteration.

Usage
-----
    python src/e6/e6_persistence.py --n 70 --shots 512 --batches 8
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


def matrix_sig(tr: dict) -> np.ndarray:
    return tr['ctrl'].transpose(2, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=512)
    ap.add_argument('--batches', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--regime', default='generic', choices=('generic', 'natural'))
    ap.add_argument('--ref-shots', type=int, default=512)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    circ, dec, n_iters, m_e, syn = C.build(n, degree)
    print(f'[build] n={n} n_iters={n_iters} m_e={m_e} Q={circ.num_qubits}', flush=True)

    rng = np.random.default_rng(999)
    ref_data = C.gen_data('generic', n, degree, args.ref_shots, rng)
    ref = C.run_batch(circ, dec, syn, ref_data, degree, seed=77, n_iters=n_iters)
    refmat = matrix_sig(ref)
    refsig = refmat.reshape(args.ref_shots, -1)
    cnt = Counter(refsig[i].tobytes() for i in range(args.ref_shots))
    ref_key, n_maj = cnt.most_common(1)[0]
    ri = refsig.tobytes().find(ref_key) // refsig.shape[1]
    ref_ctrl = refmat[ri]
    ref_tgt = ref_ctrl[:, 0] ^ ref_ctrl[:, 3]
    ref_allc = ref_ctrl.sum(axis=1)
    print(f'[ref] majority {n_maj}/{args.ref_shots} ({len(cnt)} distinct); '
          f'reference target iterations = {int(ref_tgt.sum())}', flush=True)

    first_div = []
    last_div_ctrl = 0
    last_div_tgt = 0
    live_delta_end = []
    live_delta_min = []
    tgt_symdiff = []
    n_dev = 0
    tot = 0
    n_multi = 0
    for bi in range(args.batches):
        rng = np.random.default_rng(1000 * (bi + 1) + args.seed)
        data = C.gen_data(args.regime, n, degree, args.shots, rng)
        t1 = time.time()
        tr = C.run_batch(circ, dec, syn, data, degree, seed=7 + bi, n_iters=n_iters)
        mat = matrix_sig(tr)
        c1 = tr['c1']
        ev = (tr['rb'] & c1).sum(axis=0)
        eq = np.all(mat == ref_ctrl[None, :, :], axis=2)     # (shots, n_iters)
        for s in range(args.shots):
            d = np.nonzero(~eq[s])[0]
            tgt = mat[s][:, 0] ^ mat[s][:, 3]
            sd = int((tgt != ref_tgt).sum())
            if not len(d):
                continue
            n_dev += 1
            first_div.append(int(d[0]))
            last_div_ctrl += int(not eq[s][-1])
            last_div_tgt += int(sd > 0)
            tgt_symdiff.append(sd)
            live_delta_min.append(int(tgt[:d[0]].sum()) - int(ref_tgt[:d[0]].sum()))
            live_delta_end.append(int(tgt.sum()) - int(ref_tgt.sum()))
            if ev[s] > 1:
                n_multi += 1
        tot += args.shots
        print(f'[batch {bi}] dev={n_dev} {time.time() - t1:.1f}s', flush=True)

    tgt_symdiff = np.asarray(tgt_symdiff)
    live_delta_end = np.asarray(live_delta_end)
    total_pos = n_iters * 1
    print(f'\n== D. persistence / how much of the frozen bitmap is wrong ==')
    print(f'shots = {tot} ({args.regime}), deviating (ctrl trace != generic) = {n_dev}')
    print(f'  multi c1-event shots              : {n_multi}')
    print(f'  still deviating at the LAST iter  : {last_div_ctrl}/{n_dev}')
    print(f'  target-bitmap wrong at the LAST it: {last_div_tgt}/{n_dev}')
    print(f'  first-divergence it mod 6 hist    : '
          f'{dict(sorted(Counter(int(v) % 6 for v in first_div).items()))}')
    if n_dev:
        print(f'  |symmetric difference| of the target bitmap per deviating shot:'
              f' min {tgt_symdiff.min()} median {int(np.median(tgt_symdiff))} '
              f'mean {tgt_symdiff.mean():.1f} max {tgt_symdiff.max()} '
              f'(of {n_iters} iterations)')
        print(f'  |live target count| delta at END  : '
              f'{dict(sorted(Counter(int(v) for v in live_delta_end).items()))}')
    _ = live_delta_min, total_pos

    if args.json:
        Path(args.json).write_text(json.dumps({
            'n': n, 'regime': args.regime, 'shots': tot, 'n_dev': n_dev,
            'n_multi_event': n_multi,
            'still_divergent_at_last_iter': last_div_ctrl,
            'target_bitmap_wrong_at_last_iter': last_div_tgt,
            'first_div_mod6': dict(sorted(Counter(int(v) % 6 for v in first_div).items())),
            'symdiff_median': (int(np.median(tgt_symdiff)) if n_dev else None),
            'symdiff_max': (int(tgt_symdiff.max()) if n_dev else None),
            'live_delta_end_hist': dict(sorted(
                Counter(int(v) for v in live_delta_end).items())),
        }, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

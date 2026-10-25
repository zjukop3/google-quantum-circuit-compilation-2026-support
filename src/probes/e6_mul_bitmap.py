"""E6 experiment G: the *true* sweep liveness bitmap at n=70.

The `c0 XOR c3` target bitmap is only the outer gate of the sweep.  The actual
decision is the quantum qubit `mul` (\(= c0_hc * [n_a>=1] + c3_hc * [n_A>=1]
over GF(2)\)) plus per-position conditions.  This script measures, per shot, the
real per-iteration liveness of `mul` and compares:

  * the **generic** shot's liveness bitmap against the `c0 XOR c3` bitmap,
  * a **deviating** shot's liveness bitmap against the generic one,

so we know exactly how much the freeze would have to give up.

Usage
-----
    python src/e6/e6_mul_bitmap.py --n 70 --shots 512
"""

from __future__ import annotations

import argparse
import json
import sys
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
    ap.add_argument('--shots', type=int, default=512)
    ap.add_argument('--ref-shots', type=int, default=256)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    n, degree = args.n, args.degree
    n_iters = 3 * (n + 1) + 9
    ns = C.load_probe_module(full=False)

    circ, dec, _ni, _me, syn = C.build(n, degree, ns=ns, probe=True)

    rng = np.random.default_rng(999)
    ref_data = C.gen_data('generic', n, degree, args.ref_shots, rng)
    ref = C.run_batch(circ, dec, syn, ref_data, degree, seed=77, n_iters=n_iters)
    mat = ref['ctrl'].transpose(2, 0, 1)
    sig = mat.reshape(args.ref_shots, -1)
    cnt = Counter(sig[i].tobytes() for i in range(args.ref_shots))
    key, n_maj = cnt.most_common(1)[0]
    ri = sig.tobytes().find(key) // sig.shape[1]
    ref_ctrl = mat[ri]
    ref_tgt = (ref_ctrl[:, 0] ^ ref_ctrl[:, 3]).astype(bool)
    ref_mul = ref['muls'][:, ri].astype(bool)
    print(f'[ref] majority {n_maj}/{args.ref_shots} ({len(cnt)} distinct)')
    print(f'  c0^c3 target live : {int(ref_tgt.sum())}/{n_iters}')
    print(f'  real mul   live   : {int(ref_mul.sum())}/{n_iters}')
    print(f'  target-only       : {int((ref_tgt & ~ref_mul).sum())}')
    print(f'  mul-only          : {int((ref_mul & ~ref_tgt).sum())}')

    rng = np.random.default_rng(1234)
    data = C.gen_data('natural', n, degree, args.shots, rng)
    tr = C.run_batch(circ, dec, syn, data, degree, seed=9, n_iters=n_iters)
    m = tr['ctrl'].transpose(2, 0, 1)
    dev = np.any(m != ref_ctrl[None, :, :], axis=(1, 2))
    mul = tr['muls'].astype(bool)
    print(f'\n[measure] shots {args.shots}, deviating {int(dev.sum())}'
          f' ({dev.mean():.5f})')
    if dev.any():
        sd = np.abs(mul[:, dev].sum(axis=0) - ref_mul.sum())
        diff = (mul[:, dev] != ref_mul[:, None]).sum(axis=0)
        print(f'  |live-count delta| vs generic: min {sd.min()} med '
              f'{np.median(sd):.1f} max {sd.max()}')
        print(f'  bitmap Hamming diff           : min {diff.min()} med '
              f'{np.median(diff):.1f} max {diff.max()}')
    sig = Counter(tuple(mul[:, s].tolist()) for s in range(args.shots))
    print(f'  distinct mul bitmaps: {len(sig)}')
    print(f'  live counts over shots: {sorted(Counter(int(mul[:, s].sum()) for s in range(args.shots)).items())}')
    if args.json:
        Path(args.json).write_text(json.dumps({
            'n': n, 'shots': args.shots, 'n_iters': n_iters,
            'ref_target_live': int(ref_tgt.sum()),
            'ref_mul_live': int(ref_mul.sum()),
            'n_deviating': int(dev.sum()),
            'distinct_mul_bitmaps': len(sig),
        }, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

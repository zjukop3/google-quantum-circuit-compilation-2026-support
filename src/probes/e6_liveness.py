"""E6 experiment H: is the sweep liveness bitmap *shot-independent*?

This is the question Stage A actually depends on.  The sweep's `mul` qubit is
\(c0_hc\cdot[n_a \ge 1] + c3_hc\cdot[n_A \ge 1]\) over GF(2), and the multiplier
is live exactly on the iterations where a live position exists.  If the resulting
per-iteration bitmap is the same on every shot -- including the ~1.3% of shots
whose EEA control trace deviates -- then freezing it into a compile-time constant
is *exactly* equivalent, not merely usually correct, and Stage A's 2,430,600 T
(42.5% of dqi's T) becomes claimable.

The probe records:
  * `muls`  -- liveness during the **forward** EEA pass,
  * `mulsA` -- liveness during the **adjoint** EEA pass.

Usage
-----
    python src/e6/e6_liveness.py --n 70 --shots 4096
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src' / 'e3'))

import e6_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=2048)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--regime', default='natural', choices=('generic', 'natural'))
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    n, degree = args.n, args.degree
    n_iters = 3 * (n + 1) + 9
    # the adjoint liveness only exists in the full build, and the two passes need
    # separate bitmaps, so build with full=True.
    ns = C.load_probe_module(full=True)

    circ, dec, _ni, _me, syn = C.build(n, degree, ns=ns, probe=True, full=True)

    Fs, As, dev_flags = [], [], []
    rng = np.random.default_rng(4242)
    ref_F = ref_A = ref_ctrl = None
    for start in range(0, args.shots, args.batch_size):
        b = min(args.batch_size, args.shots - start)
        data = C.gen_data(args.regime, n, degree, b, rng)
        tr = C.run_batch(circ, dec, syn, data, degree, seed=7 + start, n_iters=n_iters)
        F = tr['muls'].astype(bool)          # forward EEA pass
        A = tr['mulsA'].astype(bool)         # adjoint EEA pass
        mat = tr['ctrl'].transpose(2, 0, 1)
        if ref_ctrl is None:
            sig = mat.reshape(b, -1)
            key = Counter(sig[i].tobytes() for i in range(b)).most_common(1)[0][0]
            ri = sig.tobytes().find(key) // sig.shape[1]
            ref_ctrl = mat[ri]
            ref_F = F[:, ri]
        Fs.append(F)
        As.append(A)
        dev_flags.append(np.any(mat != ref_ctrl[None, :, :], axis=(1, 2)))

    F = np.concatenate(Fs, axis=1)
    A = np.concatenate(As, axis=1)
    dev = np.concatenate(dev_flags)
    shots = F.shape[1]

    sigF = Counter(tuple(F[:, s].tolist()) for s in range(shots))
    print(f'n={n} shots={shots} ({args.regime}); deviating {int(dev.sum())}'
          f' ({dev.mean():.5f})')
    print(f'  distinct FORWARD liveness bitmaps : {len(sigF)}')
    top = sigF.most_common(5)
    for k, c in top:
        print(f'    {c:6d}/{shots}  live={sum(k)}')
    majF = np.array(sigF.most_common(1)[0][0], dtype=bool)
    n_match = int((F == majF[:, None]).all(axis=0).sum())
    print(f'  shots whose forward bitmap == majority : {n_match}/{shots}'
          f' = {n_match / shots:.6f}')
    if dev.any():
        dm = int((F[:, dev] == majF[:, None]).all(axis=0).sum())
        print(f'    among deviating shots                : {dm}/{int(dev.sum())}')
        nd = int((F[:, ~dev] == majF[:, None]).all(axis=0).sum())
        print(f'    among generic shots                  : {nd}/{int((~dev).sum())}')
    nA = int(A.any())
    if nA:
        sigA = Counter(tuple(A[:, s].tolist()) for s in range(shots))
        majA = np.array(sigA.most_common(1)[0][0], dtype=bool)
        nA_match = int((A == majA[:, None]).all(axis=0).sum())
        print(f'  distinct ADJOINT liveness bitmaps  : {len(sigA)}'
              f'; majority matches {nA_match}/{shots} = {nA_match / shots:.6f}')
        print(f'  F | A equals the next-level bitmap : '
              f'{bool(np.array_equal(F | A, majF | majA)) if nA else "n/a"}')
    if args.json:
        Path(args.json).write_text(json.dumps({
            'n': n, 'shots': shots, 'regime': args.regime,
            'n_deviating': int(dev.sum()),
            'distinct_forward_bitmaps': len(sigF),
            'forward_majority_matches': n_match,
            'forward_majority_fraction': n_match / shots,
            'forward_majority_live': int(majF.sum()),
            'n_iters': n_iters,
        }, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

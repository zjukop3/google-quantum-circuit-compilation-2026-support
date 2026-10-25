"""E7-I — turn measured per-shot liveness into a freeze set, and fix-point it.

Why not the model closure?  `e7_enum.py`'s union (K <= 2 stalls x initial-shift
class) is a *schedule*-level closure: 15,920 sites (98.24%), because it unions
over stall *pairs* whose rate is 2^-24 per shot and which therefore never occur.
The *sample*-level union saturates instead:

    32,768 shots -> 14,775 sites (91.17%)      [E6]
   102,400 shots -> 14,791 sites (91.27%)      [E7, official distribution]

so the freezable complement is ~1,415 sites and essentially sample-independent.
The freeze set is then the complement of the union over **all** samples seen so
far, and it is *validated* (not assumed) by the full 102,400-shot official run
of the final artifact.

Usage
-----
    $P src/e7/e7_freeze_from_union.py --live notes/runs/e7/live_a.npy \
        --live notes/runs/e7/live_b.npy --out notes/runs/e7/freeze_obs.json \
        --model obs102k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'e6'))
sys.path.insert(0, str(HERE))

import e6d_schedule as S          # noqa: E402

N_ITERS, M_E = 222, 73
BIT_T = 75


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='obs102k')
    ap.add_argument('--check', action='append', default=[],
                    help='live .npy file(s) whose union must be covered')
    ap.add_argument('--merge', default=None,
                    help='existing freeze json whose sites are kept frozen too')
    args = ap.parse_args()

    union = np.zeros((N_ITERS, M_E), dtype=bool)
    tot = 0
    for p in args.live:
        a = np.load(p)
        u = a.any(axis=0)
        union |= u
        tot += a.shape[0]
        print(f'[live] {p}: {a.shape[0]} shots -> {int(u.sum())} sites; '
              f'running union {int(union.sum())}')
    freeze = ~union
    print(f'[union] {tot} shots total -> {int(union.sum())} live '
          f'({union.sum() / union.size:.5f})  freeze={int(freeze.sum())} '
          f'({freeze.sum() / freeze.size:.5f})  T_saved={int(freeze.sum()) * BIT_T * 2:,}')

    for p in args.check:
        a = np.load(p)
        u = a.any(axis=0)
        miss = u & freeze
        print(f'[check] {p}: {a.shape[0]} shots -> new union {int(u.sum())}; '
              f'frozen-but-live sites = {int(miss.sum())}  '
              f'{"CONVERGED" if miss.sum() == 0 else "MUST EXTEND"}')
        if miss.sum():
            worst = np.argwhere(miss)
            n_shots = int((a & miss[None]).any(axis=(1, 2)).sum())
            print(f'         sites {worst[:6].tolist()} …  '
                  f'affected shots in this sample: {n_shots}')
    sites = [[int(a), int(b)] for a, b in zip(*np.nonzero(freeze))]
    if args.merge and Path(args.merge).exists():
        extra = json.loads(Path(args.merge).read_text())[args.model]['sites']
        have = {(a, b) for a, b in sites}
        for it, i in extra:
            if (int(it), int(i)) not in have:
                sites.append([int(it), int(i)])
        sites = [v for v in sites if v not in [[int(a), int(b)] for a, b in
                                               zip(*np.nonzero(union))]]
        print(f'[merge] kept {len(extra)} previously-known sites')
    out = {args.model: {'sites': sites, 'union': int(union.sum()),
                        'shots': tot}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f'[out] {args.out}: {len(sites)} frozen sites')
    return 0


if __name__ == '__main__':
    sys.exit(main())

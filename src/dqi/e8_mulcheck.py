"""E8-K — cross-check the direct `mul` probe against E7's closed-form bitmap.

E7 froze the sweep multipliers from a *closed form* (`e6d_schedule.mul_bitmap`)
rather than from a direct probe.  E8's `mul` family is read directly from the
`mul` qubit at the exact point the multipliers consume it.  Both must agree on
the same official sample, or the stacked build would be mixing two instruments
that disagree about the same sites.

Usage
-----
    $P src/e8/e8_mulcheck.py --seed-file build/e7/dqi_fp2.kmb --shots 16384
"""
from __future__ import annotations

import argparse
import hashlib
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
import e6d_schedule as S         # noqa: E402
import e7_diag_official as D     # noqa: E402

import kickmix as km             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-file', default=str(ROOT / 'build/e7/dqi_fp2.kmb'))
    ap.add_argument('--shots', type=int, default=16384)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--chunk', type=int, default=2048)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    n, degree = args.n, args.degree

    h = hashlib.sha256(Path(args.seed_file).read_bytes()).hexdigest()
    print(f'[seed] sha256={h[:16]}…')
    sy, _mb, _ph = D.official_sample(int(h, 16), args.shots, args.batch_size,
                                     n=n, degree=degree)

    # (a) direct probe on the e8 forward-only probe circuit
    circ, dec, syn = P.build_probe(n, degree)
    pr = dec._e8_probe
    fl = km.array([pr['mul'][0][it][i] for it in range(222) for i in range(73)])
    A = np.zeros((222, 73), dtype=bool)
    t0 = time.time()
    for start in range(0, args.shots, args.chunk):
        stop = min(start + args.chunk, args.shots)
        bits = S.bits_of(sy[start:stop], degree)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True,
                           seed=7 + start)
        sim.use_same_registers_as(circ)
        sim.write_across_shots(km.array([q for r in syn for q in r]), bits)
        sim.do(circ)
        a = np.asarray(sim.read_across_shots(fl)).astype(bool).reshape(222, 73, -1)
        A |= a.any(axis=-1)
    print(f'[probe] direct mul union={int(A.sum())} ({time.time() - t0:.0f}s)')

    # (b) E7's closed form on the same sample
    ns = S.load_probe_module()
    c2, d2, ni, m_e, s2 = S.build(n, degree, n, ns)
    B = np.zeros((222, 73), dtype=bool)
    t0 = time.time()
    for start in range(0, args.shots, args.chunk):
        stop = min(start + args.chunk, args.shots)
        T = S.run_batch(c2, d2, s2, sy[start:stop], degree, seed=7, n_iters=ni)
        B |= S.mul_bitmap(T, m_e).any(axis=0)
    print(f'[closed form] mul union={int(B.sum())} ({time.time() - t0:.0f}s)')

    print(f'[compare] identical={bool((A == B).all())} '
          f'direct-only={int((A & ~B).sum())} closedform-only={int((B & ~A).sum())} '
          f'intersection={int((A & B).sum())}')
    res = {'seed_file': args.seed_file, 'sha256': h, 'shots': args.shots,
           'direct_union': int(A.sum()), 'closed_form_union': int(B.sum()),
           'identical': bool((A == B).all()),
           'direct_only': int((A & ~B).sum()), 'closed_form_only': int((B & ~A).sum())}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

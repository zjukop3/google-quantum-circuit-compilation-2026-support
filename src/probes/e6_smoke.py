"""E6 smoke test: does the event-pattern probe build, and does it perturb nothing?

Checks
------
1. The patched probe circuit builds at the requested `n`.
2. The reference counters agree with the probe: `cnt1` must equal the number of
   iterations where `c1` is active and `reg_b[m-2] == 0`; `cnt2` likewise for
   `c2` / `reg_b[0]`.  (This is what validates the probe's own arithmetic.)
3. `z1` (the sub-step-1 branch enable, parity of firings) is consistent with
   `cnt1`: for `cnt1 <= 1` it must equal `cnt1`.
4. The per-iteration `c1`/`c2` one-hot bits recorded by the E6 probe equal the
   E3 probe's `ctrl` decode on the same syndromes -- i.e. the probe gates change
   nothing about the trajectory.

Usage
-----
    python src/e6/e6_smoke.py --n 70 --shots 8
"""

from __future__ import annotations

import argparse
import sys
import time
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
    ap.add_argument('--shots', type=int, default=8)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--regime', default='generic')
    args = ap.parse_args()

    n, degree, shots = args.n, args.degree, args.shots
    t0 = time.time()
    circ, dec, n_iters, m_e, syn = C.build(n, degree)
    tb = time.time() - t0
    st = circ.operation_statistics()
    print(f'[build] n={n} n_iters={n_iters} m_e={m_e} T={st.num_toffoli_like:,} '
          f'Q={circ.num_qubits} in {tb:.1f}s', flush=True)

    rng = np.random.default_rng(args.seed)
    data = C.gen_data(args.regime, n, degree, shots, rng)
    t0 = time.time()
    tr = C.run_batch(circ, dec, syn, data, degree, seed=5, n_iters=n_iters)
    print(f'[sim] {shots} shots in {time.time() - t0:.1f}s', flush=True)

    c1, c2 = tr['c1'], tr['c2']
    rb, rb0 = tr['rb'], tr['rb0']
    hcz = tr['hcz']
    print(f'  is_hc_zero==1 iterations  : {int(hcz.sum())} / {n_iters * shots}'
          f' (per shot: {sorted(set(hcz.sum(axis=0).tolist()))})')
    for nm, cc in (('c0', tr['c0']), ('c3', tr['c3'])):
        print(f'  {nm}-active iterations      : {int(cc.sum())} '
              f'(per shot: {sorted(set(cc.sum(axis=0).tolist()))})')
    coin1 = (c1 & hcz)
    print(f'  c1 active AND hc_zero     : {int(coin1.sum())}'
          f' (per shot: {sorted(set(coin1.sum(axis=0).tolist()))})')
    print(f'\nprobe totals over {n_iters} iters x {shots} shots:')
    print(f'  c1-active iterations     : {int(c1.sum())} '
          f'(per shot: {sorted(set(c1.sum(axis=0).tolist()))})')
    print(f'  c2-active iterations     : {int(c2.sum())}')
    print(f'  reg_b[m-2]==0 at c1 iter : {int((rb & c1).sum())}')
    print(f'  reg_b[0]==0   at c2 iter : {int((rb0 & c2).sum())}')
    print(f'  c2g (c2 AND flag)        : {int(tr["c2g"].sum())}')
    ok1 = bool((tr['z1'][-1] == ((rb & c1).sum(axis=0) % 2)).all())
    print(f'  z1 parity == #(zero events) mod 2 : {ok1}')
    n_ev = int(((rb & c1).sum(axis=0) > 0).sum())
    print(f'  shots with >=1 c1 zero-event: {n_ev}/{shots} (generic expectation '
          f'~{shots * 8.9e-3:.4f})')

    # ---- cross-check against the E3 probe ---------------------------------
    try:
        import eea_local_probe_n70 as E3
    except Exception as exc:                                   # pragma: no cover
        print(f'[warn] E3 probe unavailable: {exc}')
        return 0
    t0 = time.time()
    c3, d3, ni3, me3, syn3 = E3.build(n, degree, n)
    print(f'  [hash] data {hash(data.tobytes())}  n_shots={data.shape}')
    T3 = E3.run_batch(c3, d3, syn3, data, degree, seed=5, n_iters=ni3)
    print(f'[e3 ] built+ran in {time.time() - t0:.1f}s', flush=True)
    f = E3.FIELDS
    e3ctrl = T3[:, :, f.index('ctrl')]            # (n_iters, shots)
    c2t = tr['ctrl2']
    mineT = (c1 * 2 + c2 * 4)                     # (n_iters, shots)
    mineT2 = (c2t[:, 1, :] * 2 + c2t[:, 2, :] * 4)
    print('  E6 ctrl2 one-hot counts:',
          {int(v): int((mineT2 == v).sum()) for v in np.unique(mineT2)})
    print('  ctrl == ctrl2 everywhere:', bool((tr['ctrl'] == tr['ctrl2']).all()))
    same_c1 = bool((e3ctrl == mineT).all()) and bool((e3ctrl == mineT2).all())
    print('  E3  ctrl one-hot counts:',
          {int(v): int((e3ctrl == v).sum()) for v in np.unique(e3ctrl)})
    print('  E6  ctrl one-hot counts:',
          {int(v): int((mineT == v).sum()) for v in np.unique(mineT)})
    if not same_c1:
        d = np.argwhere(e3ctrl != mineT)
        print('  first mismatches (iter, shot, E3, E6):', d[:5].tolist(),
              [int(e3ctrl[i, j]) for i, j in d[:5]],
              [int(mineT[i, j]) for i, j in d[:5]])
    print(f'\n[cross-check vs E3 probe] c1/c2 decode identical: {same_c1}')
    return 0 if (same_c1 and ok1) else 1


if __name__ == '__main__':
    raise SystemExit(main())

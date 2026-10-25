"""E6 experiment A+B+C: what *is* the dqi deviation, and what is the 2^-12?

Design
------
One probe build (`e6_common`), one run per batch, three questions answered at once:

**A. Where does the 2^-12 live?**
  For every iteration the probe records, *unconditionally*,
        `rb[it]  = [reg_b[m-2] == 0]`   `rb0[it] = [reg_b[0] == 0]`
        `ctrl[it] = one-hot sub-step decode`
  so the *opportunity* set is exactly `{it : c1}` and the *event* set is
  `{it : c1 and rb}`.  Their sizes per shot give the deviation law directly.
  Validated against the E3 probe's independent `ctrl` trace on the same syndromes.

**B. Is the deviation a function of the event *pattern* alone?**
  Group the shots by the bit-pattern of their event set and check that all shots
  in a group share one full sub-step-phase trajectory.

**C. Minimal sufficient description of one deviating shot.**
  Dump the metadata trace of a deviating shot next to the generic reference,
  locate the first differing iteration, and report which fields differ and by how
  much (the "extra sub-step 1 iteration + permanent lag" claim, made exact).

Usage
-----
    python src/e6/e6_deviation_law.py --n 70 --shots 256 --batches 4
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


def ctrl_sig(tr: dict) -> np.ndarray:
    """(shots, n_iters*4) uint8 signature of the full sub-step trajectory."""
    return tr['ctrl'].transpose(2, 0, 1).reshape(tr['ctrl'].shape[2], -1)


def matrix_sig(tr: dict) -> np.ndarray:
    """(shots, n_iters, 4) sub-step trajectory, shot-major."""
    return tr['ctrl'].transpose(2, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=256)
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--regime', default='natural', choices=('generic', 'natural'))
    ap.add_argument('--ref-shots', type=int, default=512)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    t0 = time.time()
    circ, dec, n_iters, m_e, syn = C.build(n, degree)
    print(f'[build] n={n} n_iters={n_iters} m_e={m_e} '
          f'T={circ.operation_statistics().num_toffoli_like:,} Q={circ.num_qubits} '
          f'in {time.time() - t0:.1f}s', flush=True)

    # ---- generic reference (majority vote over forced-generic shots) --------
    rng = np.random.default_rng(999)
    ref_data = C.gen_data('generic', n, degree, args.ref_shots, rng)
    ref = C.run_batch(circ, dec, syn, ref_data, degree, seed=77, n_iters=n_iters)
    refmat = matrix_sig(ref)
    refsig = refmat.reshape(args.ref_shots, -1)
    cnt = Counter(refsig[i].tobytes() for i in range(args.ref_shots))
    ref_key, n_maj = cnt.most_common(1)[0]
    ri = refsig.tobytes().find(ref_key) // refsig.shape[1]
    ref_ctrl = refmat[ri]                              # (n_iters, 4)
    ref_opp = int(ref['c1'][:, ri].sum())
    ref_ev = int((ref['rb'][:, ri] & ref['c1'][:, ri]).sum())
    print(f'[ref] majority ctrl signature: {n_maj}/{args.ref_shots} agree, '
          f'{len(cnt)} distinct; reference shot has {ref_opp} c1 opportunities, '
          f'{ref_ev} zero-events', flush=True)

    c1_opp, ev1, ev2, pats, sigs, first_div, persist = [], [], [], [], [], [], 0
    tot = 0
    for bi in range(args.batches):
        rng = np.random.default_rng(1000 * (bi + 1) + args.seed)
        data = C.gen_data(args.regime, n, degree, args.shots, rng)
        t1 = time.time()
        tr = C.run_batch(circ, dec, syn, data, degree, seed=7 + bi, n_iters=n_iters)
        c1, c2 = tr['c1'], tr['c2']
        mat = matrix_sig(tr)
        eq = np.all(mat == ref_ctrl[None, :, :], axis=2)       # (shots, n_iters)
        for s in range(args.shots):
            c1_opp.append(int(c1[:, s].sum()))
            e1 = int((tr['rb'][:, s] & c1[:, s]).sum())
            e2 = int((tr['rb0'][:, s] & c2[:, s]).sum())
            ev1.append(e1)
            ev2.append(e2)
            pats.append(np.nonzero(tr['rb'][:, s] & c1[:, s])[0].tolist())
            sigs.append(mat[s].tobytes())
            d = np.nonzero(~eq[s])[0]
            first_div.append(int(d[0]) if len(d) else -1)
            if len(d) and not eq[s][-1]:
                persist += 1
        tot += args.shots
        print(f'[batch {bi}] {args.shots} shots: opp={sorted(set(c1_opp[-args.shots:]))} '
              f'events={sorted(set(ev1[-args.shots:]))} ev2={sorted(set(ev2[-args.shots:]))} '
              f'{time.time() - t1:.1f}s', flush=True)

    ev1 = np.asarray(ev1)
    ev2 = np.asarray(ev2)
    c1_opp = np.asarray(c1_opp)
    first_div = np.asarray(first_div, dtype=int)
    n_dev = int((ev1 > 0).sum())
    p1 = n_dev / tot
    lo, hi = C.clopper_pearson(n_dev, tot)

    print('\n== A. the 2^-12 ==========================================')
    print(f'shots = {tot}, regime = {args.regime}')
    print(f'c1 opportunities per shot : {sorted(set(c1_opp.tolist()))}')
    print(f'c1 zero-events per shot   : '
          f'{ {int(v): int((ev1 == v).sum()) for v in np.unique(ev1)} }')
    print(f'c2 zero-events per shot   : '
          f'{ {int(v): int((ev2 == v).sum()) for v in np.unique(ev2)} }')
    print(f'P(>=1 c1 zero-event)      = {p1:.6f}  95% CI [{lo:.6f}, {hi:.6f}]')
    print(f'predicted #c1 x 2^-12     = {c1_opp.mean() * 2 ** -12:.6f}')

    print('\n== B. deviation as a function of the event pattern ========')
    groups: dict[bytes, set] = {}
    for i in range(tot):
        groups.setdefault(bytes(str(pats[i]), 'utf-8'), set()).add(sigs[i])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f'distinct event patterns         : {len(groups)}')
    print(f'distinct ctrl trajectories      : {len(set(sigs))}')
    print(f'shots with any ctrl divergence  : {int((first_div >= 0).sum())}')
    print(f'shots still diverging at the end: {persist}')
    print(f'patterns mapping to >1 traj     : {len(multi)}')
    for k in list(multi)[:3]:
        print(f'  pattern size {len(eval(k))} -> {len(multi[k])} trajectories')
    fd = Counter(int(v) for v in first_div if v >= 0)
    print(f'first-divergence iteration hist (top 8): {fd.most_common(8)}')
    if len(fd):
        mods = Counter(int(v) % 6 for v in fd)
        print(f'first-divergence it mod 6 hist: {dict(sorted(mods.items()))}')

    print('\n== C. minimal sufficient description ======================')
    dev = np.nonzero(ev1 > 0)[0]
    if not len(dev):
        print('no deviating shot in this sample -- increase --shots/--batches')
    else:
        s = int(dev[0])
        mm = np.frombuffer(sigs[s], dtype=np.uint8).reshape(n_iters, 4)
        d = np.nonzero((mm != ref_ctrl).any(axis=1))[0]
        print(f'shot {s}: {len(pats[s])} c1 zero-event(s) at iterations {pats[s]}')
        print(f'  first ctrl divergence at iteration {int(d[0])}')
        print(f'  diverging iterations {len(d)}/{n_iters}, last {int(d[-1])}')
        print(f'  still diverging at the last iteration: {not bool((mm[-1] == ref_ctrl[-1]).all())}')
        it = int(d[0])
        print(f'  ref ctrl around {it}: '
              f'{[ref_ctrl[k].tolist() for k in range(max(0, it - 1), it + 3)]}')
        print(f'  dev ctrl around {it}: '
              f'{[mm[k].tolist() for k in range(max(0, it - 1), it + 3)]}')

    if args.json:
        Path(args.json).write_text(json.dumps({
            'n': n, 'regime': args.regime, 'shots': tot,
            'c1_opp_per_shot': sorted(set(c1_opp.tolist())),
            'ev1_hist': {int(v): int((ev1 == v).sum()) for v in np.unique(ev1)},
            'ev2_hist': {int(v): int((ev2 == v).sum()) for v in np.unique(ev2)},
            'p_ev1': p1, 'p_ev1_ci95': [lo, hi],
            'predicted_p': float(c1_opp.mean() * 2 ** -12),
            'n_distinct_patterns': len(groups),
            'n_distinct_ctrl_traces': len(set(sigs)),
            'patterns_with_multi_traj': len(multi),
            'n_shots_with_ctrl_divergence': int((first_div >= 0).sum()),
            'persist_at_end': persist,
        }, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

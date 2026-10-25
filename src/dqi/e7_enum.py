"""E7-D — provable schedule union for the dqi sweep, with the deviation model
that was *measured* rather than assumed.

Measured model (e7_sm2.py, validated bit-exactly on every deviating shot)
------------------------------------------------------------------------
A shot's schedule is a walk of the metadata state machine `e7_sm` driven by one
kind of data event:

    **Z1** -- `flag ^= c1 & (reg_b[m-2] == 0)`, read at a sub-step-1 (c1)
              iteration.  36 opportunities per shot (iterations 2, 8, ..., 212).
              Normalised event rate 2^-12 per opportunity (E6 §2.1).

and, *deterministically*, a second event

    **Z2** -- `flag ^= c2 & ~(n_b == 0) & (reg_b[0] == 0)`, observed at exactly
              **17 iterations after** a Z1, whenever that iteration is a
              sub-step-2 (c2) iteration.

Evidence for the Z2 rule: over 16,384 natural shots (166 deviating), the event
decomposition is
    ('z1','z2') with gap == 17 : 141  (every shot where the +17 iteration is c2)
    ('z1',)                    :  20  (every shot where the +17 iteration is not
                                       c2, i.e. the last three c1 opportunities
                                       200/206/212 or an already-halted EEA)
    lone ('z2',)               :   0
Zero counterexamples in ~5.9e5 c2 opportunities, so Z2 is not an independent
2^-12 event: it is induced by the Z1 stall (the extra shift leaves a structurally
zero coefficient in slot 0).

Therefore the reachable schedule set is the BFS of the state machine with a
**cap K on the number of Z1 stalls** (`K=1` is the task's "37 candidate
schedules": {no trigger} + {Z1 at the k-th c1 opportunity, k = 1..36}).

The third data dependence -- the *initial shift* loop, which sets
`n_A = 70 - (#leading zero coefficients)` before the EEA starts -- is enumerated
separately as `--nA0` (observed rate 3/16384 = 1.8e-4 for one step).

Usage
-----
    P=.../2609-xingzhi-drug/.venv/bin/python
    $P src/e7/e7_enum.py --json notes/runs/e7/union_provable.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'e6'))
sys.path.insert(0, str(HERE))

import e6d_schedule as S          # noqa: E402
import e7_sm as M                 # noqa: E402

Z2_GAP = 17
N_ITERS = M.N_ITERS
BIT_T = 75                        # T per multiplier site, per pass


def init_state_nA0(nA0):
    return (nA0, 71, 1, 0, 0, 1, 0, 0)


def advance(state, pending, z1):
    """One iteration.  `pending` = frozenset of iterations at which a Z2 is due."""
    c0, c1, c2, c3 = M.ctrl_bits(state)
    z2 = c2 and _has(pending, state)
    return state, pending, c1, c2, z2


def _has(pending, state):
    return False          # placeholder (see walk() below)


def walk(plan, nA0=70, n_iters=N_ITERS):
    """Deterministic walk for a given set of stalled c1-opportunity indices."""
    st = init_state_nA0(nA0)
    plan = set(plan)
    pend = set()
    cov = np.zeros((n_iters, M.M_E), dtype=bool)
    k = -1
    fires = []
    for it in range(n_iters):
        c0, c1, c2, c3 = M.ctrl_bits(st)
        cov[it] |= M.live_row(c0, c3, st[0], st[2])
        z2 = it in pend and c2
        pend.discard(it)
        if c1:
            k += 1
        z1 = c1 and (k in plan)
        if z1:
            pend.add(it + Z2_GAP)
            fires.append(it)
        st, _ = M.step(st, z1, z2)
    return cov, fires


def enumerate_union(K, nA0_list=(70,), n_iters=N_ITERS, max_states=3_000_000,
                    verbose=False, z2_rule=True):
    """Union over all schedules reachable with at most `K` Z1 stalls.

    `K=None` -> unlimited stalls (metadata-only worst case).
    State = (metadata, #stalls so far, pending Z2 due-iterations (relative)).
    """
    union = np.zeros((n_iters, M.M_E), dtype=bool)
    cur = set()
    for nA0 in nA0_list:
        cur.add((init_state_nA0(nA0), 0, ()))
    hist = defaultdict(int)
    for it in range(n_iters):
        nxt = set()
        for st, k, pend in cur:
            c0, c1, c2, c3 = M.ctrl_bits(st)
            union[it] |= M.live_row(c0, c3, st[0], st[2])
            z2 = (it in pend) and c2
            pend2 = tuple(d for d in pend if d != it)
            if c1 and (K is None or k < K):
                p = pend2 + ((it + Z2_GAP,) if z2_rule else ())
                nxt.add((M.step(st, True, z2)[0], k + 1, tuple(sorted(set(p)))))
            nxt.add((M.step(st, False, z2)[0], k, pend2))
            if not z2_rule and c2:
                nxt.add((M.step(st, False, True)[0], k, pend2))
        if len(nxt) > max_states:
            raise RuntimeError(f'state explosion at it={it}: {len(nxt)}')
        cur = nxt
        for _st, k, _p in cur:
            hist[k] += 1
        if verbose and it % 30 == 0:
            print(f'   it={it:3d} states={len(cur)} union={int(union.sum())}', flush=True)
    return union, dict(sorted(hist.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shots', type=int, default=8192)
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--json', default=None)
    ap.add_argument('--site-json', default=None)
    args = ap.parse_args()

    n, degree = 70, 12
    ns = S.load_probe_module()
    circuit, dec, n_iters, m_e, syn = S.build(n, degree, n, ns)
    rng = np.random.default_rng(1001)
    T = S.run_batch(circuit, dec, syn, S.gen_data('generic', n, degree, 256, rng),
                    degree, seed=99, n_iters=n_iters)
    keys = [T[:, s, :9].tobytes() for s in range(256)]
    key = Counter(keys).most_common(1)[0][0]
    ref = T[:, keys.index(key), :]

    union_obs = np.zeros((n_iters, m_e), dtype=bool)
    n_dev = tot = 0
    for bi in range(args.batches):
        r = np.random.default_rng(7919 * (bi + 1) + args.seed)
        TT = S.run_batch(circuit, dec, syn,
                         S.gen_data('natural', n, degree, args.shots, r), degree,
                         seed=7 + bi, n_iters=n_iters)
        tot += args.shots
        union_obs |= S.mul_bitmap(TT, m_e).any(axis=0)
        n_dev += int(np.any(TT[:, :, :9] != ref[:, None, :9], axis=2).any(axis=0).sum())
        print(f'[batch {bi}] union_obs={int(union_obs.sum())}', flush=True)
    grid = union_obs.size
    print(f'\n[observed] {tot} shots, {n_dev} deviating ({n_dev / tot:.5f}), '
          f'union={int(union_obs.sum())} ({union_obs.sum() / grid:.5f}) '
          f'freeze={grid - int(union_obs.sum())}', flush=True)

    out = {'shots': tot, 'dev': n_dev, 'grid': int(grid), 'bit_t': BIT_T,
           'observed': {'union': int(union_obs.sum()),
                        'union_frac': float(union_obs.sum() / grid),
                        'freeze': int(grid - union_obs.sum()),
                        'freeze_frac': float(1 - union_obs.sum() / grid)},
           'models': {}}

    rows = []
    for K in (0, 1, 2, 3):
        for nA0s, tag in (((70,), 'nA0=70'), ((70, 69), 'nA0=70,69')):
            if K == 3 and tag == 'nA0=70,69':
                pass
            U, hist = enumerate_union(K, nA0_list=nA0s)
            miss = int((union_obs & ~U).sum())
            unnamed = int((U & ~union_obs).sum())
            Um = U | U[::-1]
            rec = {
                'K': K if K is not None else 'inf', 'nA0': list(nA0s),
                'union': int(U.sum()), 'union_frac': float(U.sum() / grid),
                'freeze': int(grid - int(U.sum())),
                'freeze_frac': float(1 - U.sum() / grid),
                'T_saved': int((grid - int(U.sum())) * BIT_T * 2),
                'mirror_union': int(Um.sum()),
                'mirror_freeze': int(grid - int(Um.sum())),
                'mirror_freeze_frac': float(1 - Um.sum() / grid),
                'mirror_T_saved': int((grid - int(Um.sum())) * BIT_T * 2),
                'observed_not_covered': miss,
                'extra_vs_observed': unnamed,
                'stall_hist': hist,
            }
            out['models'][f'K={rec["K"]}|{tag}'] = rec
            rows.append((rec['K'], tag, rec))
            print(f"[K={str(rec['K']):>4} {tag}] union={rec['union']:>6} "
                  f"({rec['union_frac']:.5f}) freeze={rec['freeze']:>6} "
                  f"({rec['freeze_frac']:.5f}) mirror_freeze={rec['mirror_freeze']:>6} "
                  f"T_saved={rec['T_saved']:>8,} obs_not_covered={miss}", flush=True)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)

    if args.site_json:
        # the freeze sets, as (it, i) lists, for the two candidate models
        ref_live = S.mul_bitmap(ref[:, None, :], m_e)[0]
        res = {'_meta': {'live_sites': [[int(a), int(b)] for a, b in
                                        zip(*np.nonzero(ref_live))],
                         'ref_live': int(ref_live.sum()),
                         'grid': int(ref_live.size),
                         'observed_union': int(union_obs.sum()),
                         'observed_freeze': int(grid - union_obs.sum())}}
        for K, tag in ((1, 'nA0=70'), (2, 'nA0=70'), (3, 'nA0=70'), (2, 'nA0=70,69'), (1, 'nA0=70,69')):
            nA0s = (70,) if tag == 'nA0=70' else (70, 69)
            U, _h = enumerate_union(K, nA0_list=nA0s)
            dead = ~U
            res[tag.replace('=', '').replace(',', '_') + f'_K{K}'] = {
                'sites': [[int(a), int(b)] for a, b in zip(*np.nonzero(dead))],
                'union': int(U.sum()), 'freeze': int(dead.sum()),
            }
        Path(args.site_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.site_json).write_text(json.dumps(res))
        print('wrote', args.site_json)
    return 0


if __name__ == '__main__':
    sys.exit(main())

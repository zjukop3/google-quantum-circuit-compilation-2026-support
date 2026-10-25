"""E7-A — turn the E6 empirical schedule union into a *provable* union.

E6 (`notes/E6_DQI_DEVIATION.md` §3b) established, empirically, that the sweep
liveness union over 32,768 shots is 14,775 / 16,206 sites (91.17%), leaving
1,431 "safe freeze" sites (8.83%).  That union is a *lower* bound of the true
union (it is what 32,768 shots happened to visit), so 8.83% is an *upper* bound
of the freezable fraction.  This script replaces it with a closed-form
enumeration.

Model (measured here, not assumed)
----------------------------------
The disputed term `flag ^= c1 & (reg_b[m-2] == 0)` makes sub-step 1 run one
extra iteration.  Because ``n_iters`` is a compile-time constant, that extra
iteration is *inserted*: everything after it is delayed by one iteration and the
last iteration is dropped.  In terms of the per-iteration state vector
``(ctrl, n_A, n_a, ...)`` this is

    s(it) = it - lag(it),     lag non-decreasing, increments by 1 exactly at a
                              triggering iteration,

so the schedule of a shot is a **walk** through the reference state sequence
``R``: at every iteration either advance (``s += 1``) or -- if the current state
``R[s]`` has ``c1 == 1`` -- stall (``s`` repeats, ``lag += 1``).

Given the state sequence, the live multiplier sites of iteration ``it`` are the
closed form ``mul_bitmap`` of `src/e6/e6d_schedule.py`, which depends only on
``(c0, c3, n_A, n_a)`` of the *state*, never on the global iteration index:

    live_sites(s) = { i : c0(s) & (i >= m_e - n_A(s))  XOR  c3(s) & (i <= n_a(s)-1) }

Hence

    union(K) = { (it, i) : exists a walk s(.) with total lag <= K and i in
                          live_sites(s(it)) }.

``K=1`` is exactly the task's "37 candidate schedules" ({no trigger} + {trigger
at the i-th c1 opportunity, i = 1..36}): with one stall the state at iteration
``it`` is either ``R[it]`` or ``R[it-1]``, and the stall can be placed at any of
the 36 opportunities.

``K -> inf`` is the metadata-only closure (unlimited consecutive stalls).

The DP below is exact and costs nothing (222 x (K+1) booleans).

Usage
-----
    P=.../2609-xingzhi-drug/.venv/bin/python
    $P src/e7/e7_union.py --shots 8192 --batches 4 \
        --json notes/runs/e7/union_n70.json --ref-json notes/runs/e7/ref_n70.json
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
sys.path.insert(0, str(ROOT / 'src' / 'e6'))

import e6d_schedule as S  # noqa: E402

FIELDS = S.FIELDS
NSCHED = 9          # schedule fields (n_A..ctrl); rb2/rb0 are data-side


# --------------------------------------------------------------------------- #
# reference state sequence
# --------------------------------------------------------------------------- #

def reference_trace(circuit, dec, syn, n, degree, n_iters, ref_shots=256, seed=1):
    rng = np.random.default_rng(seed + 1000 * n)
    T = S.run_batch(circuit, dec, syn,
                    S.gen_data('generic', n, degree, ref_shots, rng),
                    degree, seed=99, n_iters=n_iters)
    keys = [T[:, s, :NSCHED].tobytes() for s in range(ref_shots)]
    cnt = Counter(keys)
    key, votes = cnt.most_common(1)[0]
    return T[:, keys.index(key), :], dict(votes=votes, distinct=len(cnt),
                                          shots=ref_shots)


def walk_dp(c1_ref, n_iters, K):
    """Reachable (it, lag) pairs for a walk with total lag <= K."""
    reach = np.zeros((n_iters, K + 1), dtype=bool)
    reach[0, 0] = True
    for it in range(n_iters):
        row = reach[it]
        for lag in np.nonzero(row)[0]:
            s = it - lag                      # current state index
            if s > n_iters - 1:
                continue
            if it + 1 < n_iters:              # advance: state s+1
                reach[it + 1, lag] = True
            if lag < K and c1_ref[s]:          # stall: state s repeats
                reach[it + 1, lag + 1] = True
    return reach


def union_from_walks(reach, live_ref):
    n_iters, m_e = live_ref.shape
    U = np.zeros((n_iters, m_e), dtype=bool)
    for it in range(n_iters):
        for lag in np.nonzero(reach[it])[0]:
            U[it] |= live_ref[it - lag]
    return U


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=8192)
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--ref-shots', type=int, default=256)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--regime', default='natural')
    ap.add_argument('--grid-sites', type=int, default=16206)
    ap.add_argument('--bit-t', type=int, default=75)
    ap.add_argument('--json', default=None)
    ap.add_argument('--ref-json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    n_iters = 3 * (n + 1) + 9
    m_e = n + 3

    ns = S.load_probe_module()
    t0 = time.time()
    circuit, dec, n_iters, m_e, syn = S.build(n, degree, n, ns)
    stats = circuit.operation_statistics()
    print(f'[build] n={n} m_e={m_e} n_iters={n_iters} probe_T={stats.num_toffoli_like:,} '
          f'Q={circuit.num_qubits:,} ({time.time() - t0:.1f}s)', flush=True)

    ref, rinfo = reference_trace(circuit, dec, syn, n, degree, n_iters,
                                 args.ref_shots, args.seed)
    idx = {f: i for i, f in enumerate(FIELDS)}
    ctrl = ref[:, idx['ctrl']]
    c0_ref = (ctrl & 1).astype(bool)
    c1_ref = ((ctrl >> 1) & 1).astype(bool)
    c2_ref = ((ctrl >> 2) & 1).astype(bool)
    c3_ref = ((ctrl >> 3) & 1).astype(bool)
    live_ref = S.mul_bitmap(ref[:, None, :], m_e)[0]        # (n_iters, m_e)
    print(f'[ref] votes={rinfo["votes"]}/{rinfo["shots"]} distinct={rinfo["distinct"]}  '
          f'live={int(live_ref.sum())}  c1={int(c1_ref.sum())}', flush=True)

    c1_pos = [int(x) for x in np.nonzero(c1_ref)[0]]
    c0_pos = [int(x) for x in np.nonzero(c0_ref)[0]]
    c3_pos = [int(x) for x in np.nonzero(c3_ref)[0]]
    print(f'[ref] c1 opportunities ({len(c1_pos)}): {c1_pos}', flush=True)
    print(f'[ref] c0 iters={len(c0_pos)} head={c0_pos[:12]}  c3 iters={len(c3_pos)} '
          f'head={c3_pos[:12]}', flush=True)

    # ---------------------------------------------------------------- observed
    union_obs = np.zeros((n_iters, m_e), dtype=bool)
    dev_shots = 0
    tot = 0
    shift_ok = Counter()
    shift_bad = []
    fire_hist = Counter()
    fire_pos_shift = Counter()     # lag of the trigger state relative to the shot it
    trigger_state_hist = Counter()
    double_pair = Counter()
    ref_sig = ref[:, :NSCHED]
    sigs = {}
    per_batch = []
    for bi in range(args.batches):
        r = np.random.default_rng(7919 * (bi + 1) + args.seed)
        data = S.gen_data(args.regime, n, degree, args.shots, r)
        t1 = time.time()
        T = S.run_batch(circuit, dec, syn, data, degree, seed=7 + bi, n_iters=n_iters)
        bm = S.mul_bitmap(T, m_e)
        union_obs |= bm.any(axis=0)
        diff = np.any(T[:, :, :NSCHED] != ref_sig[:, None, :], axis=2)   # (it, shots)
        dev = diff.any(axis=0)
        ndev = int(dev.sum())
        dev_shots += ndev
        tot += args.shots

        c1s, auxs, preds, fires, _, _ = S.derived(T)
        for s in np.nonzero(dev)[0]:
            d = np.nonzero(diff[:, s])[0]
            first = int(d[0])
            fire_it = [int(x) for x in np.nonzero(fires[:, s])[0]]
            fire_hist[len(fire_it)] += 1
            if len(fire_it) >= 2:
                double_pair[tuple(fire_it[:2])] += 1
            # shift model: state(it) = it - lag(it), lag = number of triggers < it
            #            => trace_dev[it] == trace_ref[it - #fires_before(it)]
            farr = np.zeros(n_iters, dtype=int)
            for f in fire_it:
                farr[f:] += 0  # placeholder; handled below
            f_incl = np.zeros(n_iters, dtype=float)
            for f in fire_it:
                f_incl[f:] += 1
            f_before = np.zeros(n_iters, dtype=float)
            for f in fire_it:
                f_before[f + 1:] += 1
            # after a trigger at f the extra iteration is *at* f+1, so the state
            # index at `it` is it - (#triggers at or before it-1)
            lag = f_before
            src = np.round(np.arange(n_iters) - lag).astype(int)
            ok = (src >= 0) & (src < n_iters)
            src_c = np.clip(src, 0, n_iters - 1)
            match = np.all(T[:, s, :NSCHED] == ref[src_c, :NSCHED], axis=1)
            good = int((match | ~ok).sum())
            shift_ok[f'{good}/{n_iters}'] += 1
            if good < n_iters - 1:
                bad = [int(x) for x in np.nonzero(~match)[0][:8]]
                if len(shift_bad) < 8:
                    shift_bad.append({'shot': int(s), 'batch': bi, 'fires': fire_it,
                                      'bad_iters': bad,
                                      'first_div': first})
            for f in fire_it:
                # which reference state did the trigger happen in?
                # forward pass: trigger at shot-iteration f reads state s(f)
                trigger_state_hist[('rel', int(f - (f - len([x for x in fire_it if x < f]))))
                                   ] += 0
            fire_pos_shift[len(fire_it)] += 0
        per_batch.append({'batch': bi, 'shots': args.shots, 'dev': ndev,
                          'sec': round(time.time() - t1, 1)})
        print(f'[batch {bi}] shots={args.shots} dev={ndev} '
              f'union={int(union_obs.sum())} {time.time() - t1:.1f}s', flush=True)

    p_dev = dev_shots / tot
    obs_live = int(union_obs.sum())
    print(f'[obs] shots={tot} dev={dev_shots} ({p_dev:.5f}) union={obs_live} '
          f'({obs_live / union_obs.size:.5f}) freeze={union_obs.size - obs_live}', flush=True)

    # ------------------------------------------------------------- provable
    models = {}
    for K in (0, 1, 2, 3, 4, 6, 8, 12, 16, n_iters):
        reach = walk_dp(c1_ref, n_iters, K)
        U = union_from_walks(reach, live_ref)
        miss = int((union_obs & ~U).sum())
        models[str(K)] = {
            'K': K,
            'union': int(U.sum()),
            'union_frac': float(U.sum() / U.size),
            'freeze': int(U.size - U.sum()),
            'freeze_frac': float(1 - U.sum() / U.size),
            'T_saved_both_passes': int((U.size - U.sum()) * args.bit_t * 2),
            'obs_union_not_covered': miss,
            'max_lag_reached': int(np.nonzero(reach.any(axis=0))[0].max()) if reach.any() else 0,
        }
        print(f'[K={K:>3}] union={int(U.sum()):>6} ({U.sum() / U.size:.5f}) '
              f'freeze={int(U.size - U.sum()):>6} ({1 - U.sum() / U.size:.5f}) '
              f'obs_not_covered={miss}', flush=True)

    # mirror closure: site (it, i) may also be live in the *adjoint* pass, which
    # replays the same EEA state sequence with `it` running backwards.  A site is
    # freeze-safe for both passes iff it is dead in U and its mirror
    # (n_iters-1-it, i) is dead in U too -- conservative, and exact if the
    # adjoint bitmap is the iteration-mirror of the forward one.
    for k, v in models.items():
        U = union_from_walks(walk_dp(c1_ref, n_iters, int(k)), live_ref)
        Um = U | np.flip(U, axis=0)
        v['mirror_union'] = int(Um.sum())
        v['mirror_freeze'] = int(Um.size - Um.sum())
        v['mirror_freeze_frac'] = float(1 - Um.sum() / Um.size)
        v['mirror_T_saved_both_passes'] = int((Um.size - Um.sum()) * args.bit_t * 2)
        print(f'[K={k:>3}] mirror-freeze={v["mirror_freeze"]:>6} '
              f'({v["mirror_freeze_frac"]:.5f})  T_saved={v["mirror_T_saved_both_passes"]:,}',
              flush=True)

    out = {
        'n': n, 'degree': degree, 'm_e': m_e, 'n_iters': n_iters,
        'ref': rinfo, 'ref_live': int(live_ref.sum()),
        'grid': int(live_ref.size), 'bit_t': args.bit_t,
        'c1_positions': c1_pos, 'n_c1': len(c1_pos),
        'c0_positions': c0_pos, 'c3_positions': c3_pos,
        'shots': tot, 'dev_shots': dev_shots, 'p_dev': p_dev,
        'observed_union': obs_live,
        'observed_union_frac': obs_live / union_obs.size,
        'observed_freeze': int(union_obs.size - obs_live),
        'observed_freeze_frac': float(1 - obs_live / union_obs.size),
        'shift_model_ok': dict(shift_ok),
        'shift_model_bad': shift_bad,
        'fire_count_hist': dict(sorted(fire_hist.items())),
        'double_fire_pairs': {str(k): v for k, v in double_pair.most_common(20)},
        'models': models,
        'per_batch': per_batch,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)

    if args.ref_json:
        refd = {
            'fields': list(FIELDS),
            'ctrl': ctrl.tolist(),
            'n_A': ref[:, idx['n_A']].tolist(),
            'n_B': ref[:, idx['n_B']].tolist(),
            'n_a': ref[:, idx['n_a']].tolist(),
            'n_b': ref[:, idx['n_b']].tolist(),
            'n_q': ref[:, idx['n_q']].tolist(),
            'flag': ref[:, idx['flag']].tolist(),
            'counter': ref[:, idx['counter']].tolist(),
            'hc': ref[:, idx['hc']].tolist(),
            'live_per_iter': [int(x) for x in live_ref.sum(axis=1)],
            'live_positions': [[int(it), int(i)] for it, i in zip(*np.nonzero(live_ref))],
        }
        Path(args.ref_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.ref_json).write_text(json.dumps(refd))
        print('wrote', args.ref_json)
    return 0


if __name__ == '__main__':
    sys.exit(main())

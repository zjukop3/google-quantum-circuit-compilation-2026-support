"""E7-C — infer the *exact* deviation event sequence of real shots.

The metadata state machine (`e7_sm.py`) reproduces the reference trace bit-exactly,
but a single forced Z1 trigger only reproduces a real deviating shot up to ~17
iterations after the trigger.  So the state machine needs a *second* data-dependent
event (a `Z2` = `reg_b[0] == 0` stall inside sub-step 2).

This script infers, for every deviating shot produced by the probe, the exact
sequence of stall events, by walking the state machine and picking, at each
iteration, the branch that matches the shot's own probe trace.  Then it reports
the statistics of that event structure.

Usage
-----
    P=.../2609-xingzhi-drug/.venv/bin/python
    $P src/e7/e7_sm2.py --shots 32768 --batches 4 --json notes/runs/e7/events.json
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

N_ITERS = M.N_ITERS


def next_state_choices(state):
    """All (kind, next_state) branches at this iteration."""
    c0, c1, c2, c3 = M.ctrl_bits(state)
    out = [('none', M.step(state, False, False)[0])]
    if c1:
        out.append(('z1', M.step(state, True, False)[0]))
    if c2:
        out.append(('z2', M.step(state, False, True)[0]))
    return out


def infer_events(trace):
    """Walk the machine against a probe trace; return (events, ok, first_fail)."""
    st = M.init_state()
    events = []
    for it in range(N_ITERS):
        obs = tuple(int(x) for x in trace[it, :9])
        if M.fields(st) != obs:
            return events, False, it
        choices = next_state_choices(st)
        nxt_obs = (tuple(int(x) for x in trace[it + 1, :9])
                   if it + 1 < N_ITERS else None)
        pick = None
        for kind, ns in choices:
            if kind == 'none':
                pick = (kind, ns)
                break
        if nxt_obs is not None and M.fields(pick[1]) != nxt_obs:
            for kind, ns in choices:
                if kind != 'none' and M.fields(ns) == nxt_obs:
                    pick = (kind, ns)
                    break
        kind, st = pick
        if kind != 'none':
            events.append((it, kind))
    return events, True, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shots', type=int, default=8192)
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max-dev', type=int, default=400)
    ap.add_argument('--json', default=None)
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

    stats = defaultdict(Counter)
    patterns = Counter()
    first_fail_hist = Counter()
    okc = badc = 0
    n_dev = 0
    shots_total = 0
    union_obs = np.zeros((n_iters, m_e), dtype=bool)
    union_inferred = np.zeros((n_iters, m_e), dtype=bool)
    examples = []
    store = []

    for bi in range(args.batches):
        r = np.random.default_rng(7919 * (bi + 1) + args.seed)
        data = S.gen_data('natural', n, degree, args.shots, r)
        TT = S.run_batch(circuit, dec, syn, data, degree, seed=7 + bi, n_iters=n_iters)
        shots_total += args.shots
        union_obs |= S.mul_bitmap(TT, m_e).any(axis=0)
        diff = np.any(TT[:, :, :9] != ref[:, None, :9], axis=2)
        dev = np.nonzero(diff.any(axis=0))[0]
        n_dev += len(dev)
        for s in dev:
            tr = TT[:, s, :]
            ev, ok, f = infer_events(tr)
            if ok:
                okc += 1
            else:
                badc += 1
                first_fail_hist[f] += 1
                if len(examples) < 5:
                    examples.append({'shot': int(s), 'batch': bi,
                                     'events': [[int(a), b] for a, b in ev],
                                     'fail_it': f})
                continue
            store.append(ev)
            kinds = tuple(k for _i, k in ev)
            its = tuple(i for i, _k in ev)
            stats['n_events'][len(ev)] += 1
            stats['kinds'][kinds] += 1
            gaps = [its[j + 1] - its[j] for j in range(len(its) - 1)]
            for g in gaps:
                stats['gap'][g] += 1
            # reproduce the trace from the inferred events -> bitmap
            st = M.init_state()
            evd = dict(its_to_kind(ev))
            for it in range(n_iters):
                c0, c1, c2, c3 = M.ctrl_bits(st)
                union_inferred[it] |= M.live_row(c0, c3, st[0], st[2])
                st, _ = M.step(st, evd.get(it) == 'z1', evd.get(it) == 'z2')
        print(f'[batch {bi}] shots={args.shots} dev={len(dev)} ok={okc} bad={badc}', flush=True)

    print(f'\ndecomposed {okc}/{okc + badc} deviating shots exactly '
          f'(dev rate {n_dev / shots_total:.5f} on {shots_total} shots)')
    print('n_events histogram:', dict(sorted(stats['n_events'].items())))
    print('event-kind patterns:', dict(stats['kinds'].most_common(12)))
    print('gaps between consecutive events:', dict(sorted(stats['gap'].items())[:20]))
    print('inference failures at iteration:', dict(sorted(first_fail_hist.items())))
    print('observed union  :', int(union_obs.sum()))
    print('inferred union  :', int(union_inferred.sum()),
          ' subset of observed:', bool(np.all(~union_inferred | union_obs)),
          ' equal:', bool(np.array_equal(union_inferred, union_obs)))

    out = {
        'shots': shots_total, 'dev': n_dev, 'decomposed': okc, 'failed': badc,
        'n_events': dict(sorted(stats['n_events'].items())),
        'kinds': {str(k): v for k, v in stats['kinds'].most_common(30)},
        'gaps': dict(sorted(stats['gap'].items())),
        'fail_iters': dict(sorted(first_fail_hist.items())),
        'examples': examples,
        'union_observed': int(union_obs.sum()),
        'union_inferred': int(union_inferred.sum()),
        'union_grid': int(union_obs.size),
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)
    return 0


def its_to_kind(ev):
    return [(i, k) for i, k in ev]


if __name__ == '__main__':
    sys.exit(main())

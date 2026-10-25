"""E7-B — abstract metadata state machine for the dqi EEA, and the *provable*
schedule union.

Why not a shift model
---------------------
E6 describes the deviation as "sub-step 1 runs one extra iteration".  That is
true of the *trigger*, but `e7_diag.py` shows the consequence is not a clean
1-iteration shift: after the transient the deviating run reaches the end of the
EEA state sequence two iterations *earlier* than the reference, and the transient
itself mixes sub-step phases.  A shift model therefore badly under-predicts the
live-site union (`e7_union.py`, K=1: 7,598 vs observed 13,699).

What is exact
-------------
Everything the sweep liveness bitmap depends on -- `c0`, `c3`, `n_A`, `n_a` --
is *metadata*, and the metadata evolves by a finite deterministic state machine:

    state = (n_A, n_B, n_a, n_b, n_q, flag, counter, halting_counter)

whose only data-dependent inputs are the two "coefficient is zero" predicates of
`_update_flag`:

    Z1 = (reg_b[m-2] == 0)   read at a sub-step-1 (c1) iteration   <- the disputed term
    Z2 = (reg_b[0]   == 0)   read at a sub-step-2 (c2) iteration

Every other guard is a comparison of metadata registers.  Hence the set of all
schedules the algorithm can produce is *exactly* the reachable set of this state
machine over the free binary inputs `(Z1, Z2)` -- no quantum simulation needed,
and no reliance on the measured 2^-12 sampling rate.

Enumeration
-----------
`enumerate_union(K)` BFSes the state machine, branching on `Z1` (a *trigger*) and
capping the total number of triggers at `K`.  `K=1` is the task's "37 candidate
schedules"; `K=2/3/...` extend it; `K=None` is the metadata-only worst case
(arbitrarily many stalls).

Usage
-----
    P=.../2609-xingzhi-drug/.venv/bin/python
    $P src/e7/e7_sm.py --validate
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

N_ITERS = 222          # 3 * (70 + 1) + 9
M_E = 73               # n + 3
MASK = 255             # 8-bit metadata registers


# --------------------------------------------------------------------------- #
# the state machine
# --------------------------------------------------------------------------- #

def init_state(n: int = 70):
    """Metadata at the start of iteration 0 (right after `_counter_controls`)."""
    return (n, n + 1, 1, 0, 0, 1, 0, 0)   # n_A, n_B, n_a, n_b, n_q, flag, counter, hc


def ctrl_bits(state):
    n_A, n_B, n_a, n_b, n_q, flag, counter, hc = state
    zero = (hc == 0)
    return (zero and counter == 0, zero and counter == 1,
            zero and counter == 2, zero and counter == 3)


def fields(state):
    """The 9 trace fields at the *start* of an iteration, in `e6d_schedule.FIELDS` order."""
    n_A, n_B, n_a, n_b, n_q, flag, counter, hc = state
    c0, c1, c2, c3 = ctrl_bits(state)
    code = int(c0) | (int(c1) << 1) | (int(c2) << 2) | (int(c3) << 3)
    return (n_A, n_B, n_a, n_b, n_q, flag, counter, hc, code)


def step(state, z1=False, z2=False):
    """One forward EEA iteration.  Returns (next_state, fired)."""
    n_A, n_B, n_a, n_b, n_q, flag, counter, hc = state
    c0, c1, c2, c3 = ctrl_bits(state)
    is_c3 = (counter == 3)

    f = flag
    if c0:
        f ^= int(n_A == n_B) ^ int(n_A < n_B and n_q == 0)
    if c1:
        f ^= int(n_A == n_B) ^ 1
        f ^= int(n_B != 1 and z1)
    if c2:
        f ^= 1
        f ^= int(n_b != 0 and z2)
        f ^= int(n_a == (n_b + 1) & MASK)
    if c3:
        f ^= int(n_a == n_b) ^ int(n_q == 1)
    f &= 1

    c0_nf = c0 and not f
    if c0:
        n_q = (n_q + 1) & MASK
    if c0_nf:
        n_B = (n_B - 1) & MASK
    if c1:
        n_B = (n_B - 1) & MASK
    if c2:
        n_b = (n_b + 1) & MASK
    if c3 and not f:
        n_b = (n_b + 1) & MASK
    if c3:
        n_q = (n_q - 1) & MASK
    if c3 and f:
        n_A, n_B = n_B, n_A
        n_a, n_b = n_b, n_a
    if is_c3 and f:
        busy = n_b < n_B
        running = busy and n_A != 0
        if not running:
            hc = (hc + 1) & MASK

    counter = (counter + f) & 3
    fired = bool(c1 and n_B != 1 and z1)
    return (n_A, n_B, n_a, n_b, n_q, f, counter, hc), fired


def live_row(c0, c3, n_A, n_a):
    row = np.zeros(M_E, dtype=bool)
    if c0:
        row[M_E - n_A:] = True
    if c3:
        row[:n_a] = True
    return row


def simulate(trigger_plan=(), n_iters=N_ITERS, n=70):
    """Run with `Z1` forced true at the given c1-*opportunity* indices (0-based)."""
    st = init_state(n)
    plan = set(trigger_plan)
    trace, fires = [], []
    k = -1
    for it in range(n_iters):
        trace.append(fields(st))
        zero = (st[7] == 0)
        c1 = zero and st[6] == 1
        if c1:
            k += 1
        z1 = c1 and (k in plan)
        st, fired = step(st, z1, False)
        if fired:
            fires.append(it)
    return trace, fires


# --------------------------------------------------------------------------- #
# enumeration of the provable union
# --------------------------------------------------------------------------- #

def enumerate_union(K=None, n_iters=N_ITERS, n=70, max_states=2_000_000,
                    verbose=False, allow_z2=True):
    """Union over every schedule reachable with at most `K` Z1-triggers."""
    union = np.zeros((n_iters, M_E), dtype=bool)
    cur = {(init_state(n), 0)}
    counts, trigger_hist = [], defaultdict(int)
    for it in range(n_iters):
        nxt = set()
        for state, k in cur:
            c0, c1, c2, c3 = ctrl_bits(state)
            union[it] |= live_row(c0, c3, state[0], state[2])
            if c1 and (K is None or k < K):
                ns, _ = step(state, True, False)
                nxt.add((ns, k + 1))
            ns, _ = step(state, False, False)
            nxt.add((ns, k))
            if allow_z2 and c2:
                # Z2 is a second, independent stall of the same kind; enumerated
                # only as a robustness control (it does not count as a "trigger")
                ns2, _ = step(state, False, True)
                nxt.add((ns2, k))
        if len(nxt) > max_states:
            raise RuntimeError(f'state explosion at it={it}: {len(nxt)}')
        cur = nxt
        counts.append(len(cur))
        for _s, k in cur:
            trigger_hist[k] += 1
        if verbose and it % 40 == 0:
            print(f'  it={it:3d} states={len(cur)} union={int(union.sum())}', flush=True)
    return union, counts, dict(sorted(trigger_hist.items()))


# --------------------------------------------------------------------------- #
# validation against the real probe
# --------------------------------------------------------------------------- #

def load_ref_and_shots(shots=8192, seed2=4242):
    import e6d_schedule as S
    n, degree = 70, 12
    n_iters = 3 * (n + 1) + 9
    ns = S.load_probe_module()
    circuit, dec, _ni, _me, syn = S.build(n, degree, n, ns)
    rng = np.random.default_rng(1001)
    T = S.run_batch(circuit, dec, syn, S.gen_data('generic', n, degree, 256, rng),
                    degree, seed=99, n_iters=n_iters)
    keys = [T[:, s, :9].tobytes() for s in range(256)]
    key = Counter(keys).most_common(1)[0][0]
    ref = T[:, keys.index(key), :]
    rng2 = np.random.default_rng(seed2)
    T2 = S.run_batch(circuit, dec, syn, S.gen_data('natural', n, degree, shots, rng2),
                     degree, seed=5, n_iters=n_iters)
    return S, ref, T2, n_iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--union', action='store_true')
    ap.add_argument('--shots', type=int, default=8192)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    S, ref, T2, n_iters = load_ref_and_shots(args.shots)
    MASK9 = 9

    # --- 1. no-trigger simulation must reproduce the probe reference trace ---
    sim, _ = simulate(())
    sim = np.array(sim, dtype=np.int64)
    same = np.all(sim == ref[:, :9])
    bad = [int(x) for x in np.nonzero(~np.all(sim == ref[:, :9], axis=1))[0][:10]]
    print(f'[valid 1] no-trigger state machine == probe reference trace on all 9 '
          f'fields: {same}  (mismatching iterations: {bad})')
    if not same:
        for it in bad[:3]:
            print(f'   it={it} sim={sim[it].tolist()} ref={ref[it, :9].tolist()}')

    # --- 2. forced triggers must reproduce real deviating shots --------------
    diff = np.any(T2[:, :, :9] != ref[:, None, :9], axis=2)
    dev = np.nonzero(diff.any(axis=0))[0]
    print(f'[valid 2] {T2.shape[1]} natural shots, {len(dev)} deviating '
          f'({len(dev) / T2.shape[1]:.5f})')
    matched = tried = 0
    detail = []
    FIDX = {f: i for i, f in enumerate(S.FIELDS)}
    for s in dev:
        tr = T2[:, s, :]
        fires = [it for it in range(n_iters)
                 if bool(tr[it, FIDX['ctrl']] & 2)
                 and int(tr[it, FIDX['rb2']]) == 0
                 and int(tr[it, FIDX['n_B']]) != 1]      # aux & rb2==0
        if not fires:
            continue
        plan, occ = [], -1
        for it in range(n_iters):
            zero = (int(tr[it, 7]) == 0)
            if zero and int(tr[it, 6]) == 1:
                occ += 1
                if it in fires:
                    plan.append(occ)
        sim2, _f = simulate(plan)
        sim2 = np.array(sim2, dtype=np.int64)
        good = bool(np.all(sim2 == tr[:, :9]))
        tried += 1
        matched += int(good)
        if not good and len(detail) < 5:
            d = [int(x) for x in np.nonzero(~np.all(sim2 == tr[:, :9], axis=1))[0][:6]]
            detail.append({'shot': int(s), 'fires': fires, 'plan': plan, 'mismatch_it': d})
        if tried >= 40:
            break
    print(f'[valid 2] forced-trigger simulation reproduces {matched}/{tried} real '
          f'deviating shots')
    for d in detail:
        print('   ', d)

    if args.union:
        out = {'valid_no_trigger': bool(same), 'valid_forced': [matched, tried]}
        for K in (1, 2, 3, 4, 6, 10, None):
            U, counts, th = enumerate_union(K, verbose=False)
            tot = U.size
            rec = {
                'K': K if K is not None else 'inf',
                'union': int(U.sum()), 'union_frac': float(U.sum() / tot),
                'freeze': int(tot - U.sum()), 'freeze_frac': float(1 - U.sum() / tot),
                'max_states': int(max(counts)), 'trigger_hist': th,
            }
            out[f'K={K}'] = rec
            print(f"[K={str(K):>4}] union={rec['union']:>6} ({rec['union_frac']:.5f}) "
                  f"freeze={rec['freeze']:>6} ({rec['freeze_frac']:.5f}) "
                  f"max_states={rec['max_states']}", flush=True)
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(out, indent=2))
            print('wrote', args.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())

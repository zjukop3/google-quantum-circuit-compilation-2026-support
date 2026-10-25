"""E5: what do `_shift_b_array_right` and `_swap_a_and_b` actually *do* per iteration?

Both primitives are pure data *moves*, and the only reason they cost anything in
`num_toffoli_like` is that their controls are quantum:

    _swap_a_and_b          cswap(swap_ctrl, reg_a+meta, reg_b+meta)   892 T / call
    _shift_b_array_right   cswap(do_shift_i, reg_b[i-1], reg_b[i])  72x12 T / call

An *unconditional* swap / rotate compiles to `swap`s and costs **0 T** (measured,
`e5_ablate.py::primitive_costs`).  So the whole 779,664 T is the price of the
*conditioning*, not of the movement.  This script asks whether the conditioning
can be hoisted out of the element loop, which needs the runtime shape of

    swap_ctrl = c3_hc AND flag
    do_shift(i) = ctrl_nB*[n_B >= m-i+1]  XOR  c2_hc*[n_b >= i]  XOR  c3_nf*[n_B < m-i]

Reuses E3's forward-only metadata trace probe (`eea_local_probe_n70.py`), which
records `(n_A, n_B, n_a, n_b, n_q, flag, counter, hc, ctrl)` at the start of every
forward EEA iteration.

Usage
-----
    python src/e5/e5_schedule.py --n 70 --shots 512
    python src/e5/e5_schedule.py --n 70 --shots 512 --json notes/runs/e5_schedule.json
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
sys.path.insert(0, str(ROOT / 'src' / 'e3'))

import eea_local_probe_n70 as LP  # noqa: E402

F = {name: i for i, name in enumerate(LP.FIELDS)}


def shift_control(T: np.ndarray, m_e: int) -> np.ndarray:
    """(n_iters, shots, m_e) boolean `do_shift(i)` bitmap, from the closed form.

    Mirrors `_shift_b_array_right` exactly:

        nf   = (c0_hc & ~flag, c3_hc & ~flag)
        ctrl_nB = c0_nf | c1_hc
        do_shift(i) = ctrl_nB*[n_B >= m-i+1] ^ c2_hc*[n_b >= i] ^ c3_nf*[n_B < m-i]

    with `_xor_threshold`'s out-of-range rules folded in (lim_B = m-2, lim_b =
    min(m-2, it//2+1); a threshold whose value exceeds its limit becomes a plain
    copy of the control, or nothing).
    """
    n_iters, shots, _ = T.shape
    ctrl = T[:, :, F['ctrl']]
    c0 = (ctrl & 1).astype(bool)
    c1 = ((ctrl >> 1) & 1).astype(bool)
    c2 = ((ctrl >> 2) & 1).astype(bool)
    c3 = ((ctrl >> 3) & 1).astype(bool)
    flag = T[:, :, F['flag']].astype(bool)
    n_B = T[:, :, F['n_B']].astype(np.int32)
    n_b = T[:, :, F['n_b']].astype(np.int32)

    c0_nf = c0 & ~flag
    c3_nf = c3 & ~flag
    ctrl_nB = c0_nf | c1
    lim_B = m_e - 2

    out = np.zeros((n_iters, shots, m_e), dtype=bool)
    for it in range(n_iters):
        lim_b = min(m_e - 2, it // 2 + 1)
        i = np.arange(m_e)
        # term 1: _xor_threshold(cur_B_ge, lim_B, do_shift, ctrl_nB, m-i+1, at_least=True)
        val = m_e - i + 1
        ge = np.zeros((shots, m_e), dtype=bool)
        for j in range(m_e):
            v = int(val[j])
            if v > lim_B:
                pass                      # at_least + out of range -> no-op
            else:
                ge[:, j] = ctrl_nB[it] & (n_B[it] >= v)
        # term 2: _xor_threshold(cur_b_ge, lim_b, do_shift, c2_hc, i, at_least=True)
        lt = np.zeros((shots, m_e), dtype=bool)
        for j in range(m_e):
            v = int(i[j])
            if v > lim_b:
                pass
            else:
                lt[:, j] = c2[it] & (n_b[it] >= v)
        # term 3: _xor_threshold(cur_B_lt, lim_B, do_shift, c3_nf, m-i, at_least=False)
        g = np.zeros((shots, m_e), dtype=bool)
        for j in range(m_e):
            v = int(m_e - i[j])
            if v > lim_B:
                g[:, j] = c3_nf[it]          # at_least=False, out of range -> copy ctrl
            else:
                g[:, j] = c3_nf[it] | (n_B[it] < v)
        out[it] = ge ^ lt ^ g
    return out


def window_of(bits: np.ndarray) -> tuple[int, int, int]:
    """(lo, hi, count) of the True run of a 1-D boolean row; (-1,-1,0) if empty."""
    idx = np.flatnonzero(bits)
    if idx.size == 0:
        return -1, -1, 0
    return int(idx[0]), int(idx[-1]), int(idx.size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=512)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--regime', default='generic',
                    choices=('generic', 'natural', 'lead1', 'lead2'))
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    ns = LP.load_probe_module()
    circuit, dec, n_iters, m_e, syn = LP.build(n, degree, n)
    print(f'[build] n={n} m_e={m_e} n_iters={n_iters} '
          f'Q={circuit.num_qubits} T={circuit.operation_statistics().num_toffoli_like:,}')

    rng = np.random.default_rng(1234 + args.seed)
    data = LP.gen_data(args.regime, n, degree, args.shots, rng)
    T = LP.run_batch(circuit, dec, syn, data, degree, seed=7, n_iters=n_iters)
    print(f'[trace] {args.shots} shots, regime={args.regime}')

    # reference trace = majority over forced-generic shots
    ref, ndist, nmaj, nref = LP.reference_trace(circuit, dec, syn, n, degree, n_iters,
                                                shots=min(256, args.shots))
    print(f'[ref] majority trace {nmaj}/{nref} agree, {ndist} distinct')

    sb = shift_control(T, m_e)
    ref_sb = shift_control(ref[:, None, :], m_e)[:, 0, :]
    swap_ctrl = ((T[:, :, F['ctrl']] >> 3) & 1).astype(bool) & T[:, :, F['flag']].astype(bool)
    ref_swap = ((ref[:, F['ctrl']] >> 3) & 1).astype(bool) & ref[:, F['flag']].astype(bool)

    out: dict = {'n': n, 'degree': degree, 'm_e': m_e, 'n_iters': n_iters,
                 'shots': args.shots, 'regime': args.regime}

    print('=' * 78)
    print('A. swap_ctrl = c3_hc AND flag   (the control of the 892-qubit cswap)')
    print(f'   iterations where swap_ctrl is 1 in the generic path: '
          f'{int(ref_swap.sum())} / {n_iters}')
    per_it = swap_ctrl.sum(axis=1)
    varying = [it for it in range(n_iters) if per_it[it] not in (0, args.shots)]
    print(f'   iterations where swap_ctrl depends on the shot: {len(varying)} / {n_iters}')
    out['swap'] = {'generic_active_iters': int(ref_swap.sum()),
                   'shot_dependent_iters': len(varying)}
    if varying:
        ex = varying[:12]
        print('   first shot-dependent iterations (active shots / total):  '
              + ', '.join(f'{it}:{per_it[it]}/{args.shots}' for it in ex))

    print('=' * 78)
    print('B. do_shift(i) window shape (the control chain of the shift)')
    generic_w = window_of(ref_sb[0]) if n_iters else (-1, -1, 0)
    # accumulate statistics over iterations and shots
    sizes = []
    los = []
    his = []
    rows_with_true = []
    all_true = 0
    contiguous = 0
    for it in range(n_iters):
        for s in range(0, min(args.shots, 64)):
            lo, hi, cnt = window_of(sb[it, s])
            sizes.append(cnt)
            if cnt:
                los.append(lo)
                his.append(hi)
                if hi - lo + 1 == cnt:
                    contiguous += 1
            else:
                rows_with_true.append(0)
    sizes = np.array(sizes)
    los = np.array(los)
    his = np.array(his)
    nonempty = sizes[sizes > 0]
    print(f'   non-empty windows: {nonempty.size} / {sizes.size}  '
          f'({100.0*nonempty.size/max(sizes.size,1):.1f}% of (it,shot) slots)')
    if nonempty.size:
        print(f'   window size: min={nonempty.min()} median={int(np.median(nonempty))} '
              f'mean={nonempty.mean():.1f} max={nonempty.max()}   '
              f'(m_e-1 = {m_e-1})')
        print(f'   window lo : min={los.min()} median={int(np.median(los))} max={los.max()}')
        print(f'   window hi : min={his.min()} median={int(np.median(his))} max={his.max()}')
        print(f'   contiguous (no holes in the True run): {contiguous}/{los.size}')
        print(f'   windows equal to the full array [0, m_e-1]: '
              f'{int(((los==0)&(his==m_e-1)).sum())}/{los.size}')
        out['shift'] = {
            'nonempty_frac': float(nonempty.size / sizes.size),
            'size_min': int(nonempty.min()), 'size_median': int(np.median(nonempty)),
            'size_mean': float(nonempty.mean()), 'size_max': int(nonempty.max()),
            'lo_min': int(los.min()), 'lo_max': int(los.max()),
            'hi_min': int(his.min()), 'hi_max': int(his.max()),
            'contiguous_frac': float(contiguous / los.size),
            'full_array_frac': float(((los == 0) & (his == m_e - 1)).sum() / los.size),
        }

    print('=' * 78)
    print('C. how many distinct do_shift bitmaps are there overall?')
    keys = [sb[it, s].tobytes() for it in range(n_iters) for s in range(args.shots)]
    cnt_kv = Counter(keys)
    print(f'   {len(cnt_kv)} distinct (it,shot) bitmaps over {len(keys)} draws')
    print(f'   most common bitmap covers {cnt_kv.most_common(1)[0][1]}/'
          f'{len(keys)} = {100.0*cnt_kv.most_common(1)[0][1]/len(keys):.1f}%')
    # per-iteration determinism: is the bitmap the same for every shot at fixed it?
    det_iters = sum(1 for it in range(n_iters)
                    if len({sb[it, s].tobytes() for s in range(args.shots)}) == 1)
    print(f'   iterations whose bitmap is shot-independent: {det_iters} / {n_iters}')
    out['shift']['distinct_bitmaps'] = len(cnt_kv)
    out['shift']['shot_independent_iters'] = det_iters

    print('=' * 78)
    print('D. "free global rotation + small correction" feasibility')
    print('   cost model: an unconditional rotate costs 0 T; a window rotate costs')
    print('   (window size) conditional word-swaps.  A whole-array rotation costs 0,')
    print('   so the win would be (m_e-1 - #corrections)*degree*444 T.')
    if nonempty.size:
        w_mean = float(nonempty.mean())
        full = (m_e - 1)
        print(f'   emitted today            : {full} word-swaps x {degree} bits '
              f'x 444 calls = {full*degree*444:,} T')
        print(f'   emitted with a window walk: {full} word-swaps (unchanged: the window')
        print(f'                               bounds are quantum, so all {full} gates')
        print(f'                               must still be emitted)')
        print(f'   *active* fraction today   : {100.0*w_mean/full:.1f}% of the emitted chain')

    print('=' * 78)
    print('E. if the *transport* controls were frozen into a static schedule...')
    print('   (the only route that can actually capture the 13.6%, since an')
    print('    unconditional move costs 0 T -- but it needs a compile-time control)')
    same = np.all(swap_ctrl == ref_swap[:, None], axis=0)
    sb_same = np.all(sb == ref_sb[:, None, :], axis=(0, 2))
    for label, ok in (('swap_ctrl bitmap', same), ('do_shift bitmap', sb_same)):
        frac = float(ok.mean())
        print(f'   {label:18s} matches the generic path on {int(ok.sum())}/{args.shots} shots '
              f'= {100.0*frac:.3f}%  (deviation rate {1.0-frac:.3e})')
        out.setdefault('static_freeze', {})[label] = {
            'match_shots': int(ok.sum()), 'match_frac': frac,
            'deviation_rate': 1.0 - frac}
    print('   a 102,400-shot contract needs a zero-failure rate, so a static freeze')
    print('   is only usable if (1 - match_frac) is exactly 0 -- not "small".')

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, default=float))
        print(f'\nwrote {p}')


if __name__ == '__main__':
    main()

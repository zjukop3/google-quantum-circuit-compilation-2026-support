"""E6 experiment I: freeze the sweep liveness bitmap and check the answer.

Stage A (the 2,430,600 T / 42.5%-of-T prize) needs the sweep's per-iteration
liveness bitmap to be a compile-time constant.  `e6_liveness.py` measures that the
bitmap is constant on generic shots but *different* on the ~0.8-1.3% of shots
whose EEA control trace deviates.  The only question that matters is whether that
difference changes the computed answer.

Method
------
1. Build the full decoder once (forward EEA + adjoint EEA + uncompute; Chien
   search omitted -- the only consumer of `error_mbuc`, and phase-only).
2. From a batch of generic shots, take the majority forward/adjoint liveness
   bitmaps.
3. Rebuild with `frozen_target=(F_maj, A_maj)`: the sweep's quantum `mul` control
   is replaced by a compile-time constant, so the core multiplications are elided
   on non-live iterations.
4. On one shared set of `natural` shots, check the algorithmic contract for both
   circuits: `S_i -> S_i` and every other qubit back to |0>.  Classify each shot
   as generic or deviating via the forward-only probe's control trace.

If the frozen circuit restores every syndrome, the freeze is semantically exact
and the 2,430,600 T is claimable; if not, the number of failing shots is the
concrete price of the 8.887e-3 deviation rate.

Usage
-----
    python src/e6/e6_frozen_schedule.py --n 70 --per-batch 256 --batches 8
    python src/e6/e6_frozen_schedule.py --n 20 --per-batch 256 --batches 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src' / 'e3'))

import e6_common as C  # noqa: E402


def majority_bitmaps(ns_full, n, degree, ref_shots, n_iters, seed=999):
    """Majority (forward, adjoint) liveness bitmaps and the majority ctrl trace."""
    circ, dec, _ni, _me, syn = C.build(n, degree, ns=ns_full, probe=True, full=True)
    rng = np.random.default_rng(seed)
    data = C.gen_data('generic', n, degree, ref_shots, rng)
    tr = C.run_batch(circ, dec, syn, data, degree, seed=77, n_iters=n_iters)
    mat = tr['ctrl'].transpose(2, 0, 1)
    sig = mat.reshape(ref_shots, -1)
    from collections import Counter
    key, n_maj = Counter(sig[i].tobytes() for i in range(ref_shots)).most_common(1)[0]
    ri = sig.tobytes().find(key) // sig.shape[1]
    F = np.asarray(tr['muls'][:, ri], dtype=bool)
    A = np.asarray(tr['mulsA'][:, ri], dtype=bool)
    return F, A, mat[ri], n_maj


def contract(circuit, dec, syn, data, degree, batch_size, skip_probe):
    """(syndrome_restored_failures, all-other-qubits-zero_failures)."""
    inp = C.bits_of(data, degree)
    syn_set = set(q for r in syn for q in r)
    skip = set(dec.probe_set) if skip_probe else set()
    others = [q for q in range(circuit.num_qubits) if q not in syn_set and q not in skip]
    n = len(data)
    bad_syn = np.zeros(n, dtype=bool)
    bad_anc = np.zeros(n, dtype=bool)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True,
                           seed=7 + start)
        sim.use_same_registers_as(circuit)
        sim.write_across_shots(km.array([q for r in syn for q in r]), inp[:, start:stop])
        sim.do(circuit)
        got = np.array([np.asarray(sim.read_across_shots(q)).astype(bool)
                        for r in syn for q in r])
        bad_syn[start:stop] = (got ^ inp[:, start:stop]).any(axis=0)
        for q in others:
            bad_anc[start:stop] |= np.asarray(
                sim.read_across_shots(km.q(q))).astype(bool)
    return bad_syn, bad_anc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--per-batch', type=int, default=256)
    ap.add_argument('--batches', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--regime', default='natural', choices=('generic', 'natural'))
    ap.add_argument('--ref-shots', type=int, default=256)
    ap.add_argument('--check-batch', type=int, default=64)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    n_iters = 3 * (n + 1) + 9
    nsf = C.load_probe_module(full=False)
    nsfull = C.load_probe_module(full=True)

    t0 = time.time()
    F_maj, A_maj, ref_ctrl, n_maj = majority_bitmaps(
        nsfull, n, degree, args.ref_shots, n_iters)
    print(f'[ref] generic majority {n_maj}/{args.ref_shots}; liveness '
          f'forward={int(F_maj.sum())} adjoint={int(A_maj.sum())} of {n_iters}'
          f'  ({time.time() - t0:.1f}s)', flush=True)

    t0 = time.time()
    rcirc, rdec, _, _, rsyn = C.build(n, degree, ns=nsfull, probe=True, full=True)
    T_r = rcirc.operation_statistics().num_toffoli_like
    print(f'[ref   ] Q={rcirc.num_qubits} T={T_r:,}  ({time.time() - t0:.1f}s)',
          flush=True)
    t0 = time.time()
    fcirc, fdec, _, _, fsyn = C.build(n, degree, ns=nsfull, probe=True, full=True,
                                      frozen_target=(F_maj, A_maj))
    T_f = fcirc.operation_statistics().num_toffoli_like
    print(f'[frozen] Q={fcirc.num_qubits} T={T_f:,}  ({time.time() - t0:.1f}s)',
          flush=True)

    pcirc, pdec, _, _, psyn = C.build(n, degree, ns=nsf, probe=True)
    all_data = []
    for bi in range(args.batches):
        rng = np.random.default_rng(1000 * (bi + 1) + args.seed)
        all_data.append(C.gen_data(args.regime, n, degree, args.per_batch, rng))
    data = np.concatenate(all_data)

    dev = np.zeros(len(data), dtype=bool)
    for bi in range(args.batches):
        d = data[bi * args.per_batch:(bi + 1) * args.per_batch]
        tr = C.run_batch(pcirc, pdec, psyn, d, degree, seed=7 + bi, n_iters=n_iters)
        m = tr['ctrl'].transpose(2, 0, 1)
        dev[bi * args.per_batch:(bi + 1) * args.per_batch] = np.any(
            m != ref_ctrl[None, :, :], axis=(1, 2))

    t0 = time.time()
    sr, ar = contract(rcirc, rdec, rsyn, data, degree, args.check_batch, True)
    print(f'[sim] ref {time.time() - t0:.1f}s', flush=True)
    t0 = time.time()
    sf, af = contract(fcirc, fdec, fsyn, data, degree, args.check_batch, True)
    print(f'[sim] frozen {time.time() - t0:.1f}s', flush=True)

    print('\n== I. frozen sweep liveness ==============================')
    print(f'shots {len(data)} ({args.regime}); deviating {int(dev.sum())}'
          f' ({dev.mean():.5f})')
    print(f'  T: ref {T_r:,} -> frozen {T_f:,}  ({(T_f - T_r) / T_r * 100:+.3f}%)')
    print(f'  syndrome failures  ref / frozen : {int(sr.sum())} / {int(sf.sum())}')
    print(f'  ancilla failures   ref / frozen : {int(ar.sum())} / {int(af.sum())}')
    if dev.any():
        print(f'    frozen syndrome failures on DEVIATING shots : '
              f'{int(sf[dev].sum())} / {int(dev.sum())}')
        print(f'    frozen syndrome failures on generic  shots : '
              f'{int(sf[~dev].sum())} / {int((~dev).sum())}')
    if args.json:
        Path(args.json).write_text(json.dumps({
            'n': n, 'shots': len(data), 'regime': args.regime,
            'n_deviating': int(dev.sum()), 'p_dev': float(dev.mean()),
            'T_ref': T_r, 'T_frozen': T_f,
            'forward_live': int(F_maj.sum()), 'adjoint_live': int(A_maj.sum()),
            'syndrome_failures_ref': int(sr.sum()),
            'syndrome_failures_frozen': int(sf.sum()),
            'ancilla_failures_ref': int(ar.sum()),
            'ancilla_failures_frozen': int(af.sum()),
            'frozen_failures_deviating': int(sf[dev].sum()) if dev.any() else 0,
            'frozen_failures_generic': int(sf[~dev].sum()),
            'n_iters': n_iters,
        }, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

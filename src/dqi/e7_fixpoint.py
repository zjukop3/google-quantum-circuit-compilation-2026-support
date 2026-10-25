"""E7-J — fix-point the freeze set against the *official* sample of the artifact.

`e7_freeze_from_union.py` gives a freeze set from measured liveness, but the
grading sample is derived from `sha256(.kmb)`, so every rebuild changes the shot
sample and can expose a site the previous sample never lit.  This driver closes
that loop:

    freeze_k  = complement(union of every sample seen so far)
    file_k    = build(freeze_k)          -> new sha256 -> new sample
    sample_k  = official 102,400-shot draw for sha256(file_k)
    probe     = n=70 metadata probe on sample_k -> union_k, violations_k

It stops when `violations_k == 0`, i.e. when the built artifact's *own* grading
sample never lights a frozen site.  Violations counted here are a conservative
upper bound on phase errors: a lit frozen site only flips the phase when the
MBUC bits of that shot mask onto it (measured 6/19 in the first attempt).

Usage
-----
    $P src/e7/e7_fixpoint.py --out-dir build/e7 --rounds 6 --shots 102400
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'e6'))
sys.path.insert(0, str(HERE))

import e6_semantics as ES          # noqa: E402
import e6d_schedule as S          # noqa: E402
import e7_diag_official as D      # noqa: E402

import kickmix as km              # noqa: E402

N_ITERS, M_E = 222, 73
BIT_T = 75


def build_frozen(ref, freeze_mask, m, n, degree, out):
    FS = np.zeros((2, N_ITERS, M_E), dtype=bool)
    FS[0] = freeze_mask
    FS[1] = freeze_mask
    circ = ES.build_circuit(ref, m, n, degree, frozen_sites=FS)
    out.parent.mkdir(parents=True, exist_ok=True)
    circ.to_file(out, format='kmb')
    return circ, hashlib.sha256(out.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'build/e7'))
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--shots', type=int, default=102400)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--chunk', type=int, default=2048)
    ap.add_argument('--seed-live', action='append', default=[])
    ap.add_argument('--m', type=int, default=4095)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    args = ap.parse_args()

    ref = ES.load_module('e7ref_fix', ROOT / 'data' / 'verifiers' / 'dqi'
                         / 'dqi_reference_circuit.py')
    ns = S.load_probe_module()
    circuit, dec, n_iters, m_e, syn = S.build(args.n, args.degree, args.n, ns)
    out_dir = Path(args.out_dir)

    U = np.zeros((N_ITERS, M_E), dtype=bool)
    n_seed_shots = 0
    for p in args.seed_live:
        a = np.load(p)
        U |= a.any(axis=0)
        n_seed_shots += a.shape[0]
        del a
    print(f'[init] seeded union {int(U.sum())} sites from {n_seed_shots} shots')

    log = []
    for rnd in range(args.rounds):
        freeze = ~U
        circ, h = build_frozen(ref, freeze, args.m, args.n, args.degree,
                              out_dir / f'dqi_fp{rnd}.kmb')
        st = circ.operation_statistics()
        print(f'[round {rnd}] freeze={int(freeze.sum())} T={st.num_toffoli_like:,} '
              f'Q={circ.num_qubits} sha={h[:16]}…', flush=True)
        t0 = time.time()
        sy, _mb, _ph = D.official_sample(int(h, 16), args.shots, args.batch_size)
        newU = np.zeros((N_ITERS, M_E), dtype=bool)
        viol_shots, viol_sites = 0, np.zeros((N_ITERS, M_E), dtype=bool)
        for start in range(0, args.shots, args.chunk):
            stop = min(start + args.chunk, args.shots)
            T = S.run_batch(circuit, dec, syn, sy[start:stop], args.degree,
                            seed=7, n_iters=n_iters)
            live = S.mul_bitmap(T, m_e)              # (chunk, it, i)
            newU |= live.any(axis=0)
            hit = (live & freeze[None]).any(axis=(1, 2))
            viol_shots += int(hit.sum())
            if hit.any():
                viol_sites |= live[hit].any(axis=0)
        print(f'[round {rnd}] sample union={int(newU.sum())} '
              f'violating_shots={viol_shots} '
              f'({viol_shots / args.shots:.3e}) new_sites={int((newU & freeze).sum())} '
              f'({time.time() - t0:.0f}s)', flush=True)
        log.append({'round': rnd, 'sha256': h, 'freeze': int(freeze.sum()),
                    'T': int(st.num_toffoli_like),
                    'sample_union': int(newU.sum()),
                    'violating_shots': viol_shots,
                    'new_sites': int((newU & freeze).sum())})
        if viol_shots == 0:
            print(f'[round {rnd}] CONVERGED — freeze {int(freeze.sum())} sites, '
                  f'artifact {out_dir / f"dqi_fp{rnd}.kmb"}')
            Path(out_dir / 'fixpoint.json').write_text(json.dumps(
                {'log': log, 'final': h, 'freeze': int(freeze.sum())}, indent=2))
            return 0
        U |= newU
        print(f'[round {rnd}] union now {int(U.sum())} → refreezing', flush=True)
    Path(out_dir / 'fixpoint.json').write_text(json.dumps({'log': log}, indent=2))
    print('[warn] rounds exhausted without convergence')
    return 0


if __name__ == '__main__':
    sys.exit(main())

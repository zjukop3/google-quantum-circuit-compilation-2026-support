"""E8-E — fix-point the `cswap` freeze set against the artifact's own official sample.

Same discipline as `e7_fixpoint.py` (E7 §6), now for the three `cswap` families:

    freeze_k = complement(union of every official sample seen so far)   [per family]
    file_k   = build(freeze_k)         -> new sha256 -> new sample
    sample_k = official 102,400-shot draw for sha256(file_k)
    probe_k  = control-qubit probe on sample_k -> union_k, violating shots

Stops when the built artifact's *own* grading sample never lights a frozen site.

Usage
-----
    $P src/e8/e8_fixpoint.py --out-dir build/e8 --rounds 6 --seed-union notes/runs/e8/union_fp2.npz
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

import e8_src as E8              # noqa: E402
import e8_probe as P             # noqa: E402
import e7_diag_official as D     # noqa: E402

import kickmix as km             # noqa: E402

FAMILIES = ('swap', 'shift', 'sweep', 'mul')


def build_frozen(n, degree, m_chien, freeze, out: Path, families=FAMILIES):
    ns = E8.load_module('fix', freeze=True, families=families)
    circ, dec, syn = E8.build_circuit(ns, m_chien, n, degree, freeze=freeze)
    out.parent.mkdir(parents=True, exist_ok=True)
    circ.to_file(out, format='kmb')
    return circ, hashlib.sha256(out.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'build/e8'))
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--shots', type=int, default=102400)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--chunk', type=int, default=2048)
    ap.add_argument('--seed-union', action='append', default=[])
    ap.add_argument('--m', type=int, default=4095)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--tag', default='cswap')
    ap.add_argument('--families', default='swap,shift,sweep')
    ap.add_argument('--mul-freeze', default=None,
                    help='npz with a `mul` mask (E7 reproduction) to stack in')
    ap.add_argument('--max-new-sites', type=int, default=0,
                    help='keep going even if a round adds <= this many sites (0 = stop at 0 viol)')
    args = ap.parse_args()

    families = tuple(args.families.split(','))
    n, degree = args.n, args.degree
    sh = E8.grid_shapes(n)
    out_dir = Path(args.out_dir)

    U = {f: np.zeros(E8.grid_shapes(n)[f][1:], dtype=bool) for f in FAMILIES}
    if args.mul_freeze:
        mz = np.load(args.mul_freeze)['mul']
        U['mul'] |= ~mz
        print(f'[init] mul freeze from {args.mul_freeze}: '
              f'{int(mz.sum())} sites (union seed {int(U["mul"].sum())})')
    n_seed = 0
    for p in args.seed_union:
        z = np.load(p)
        for f in FAMILIES:
            if f in z:
                U[f] |= z[f]
        n_seed += 1
    print('[init] seeded unions:'
          + ' '.join(f'{f}={int(U[f].sum())}' for f in FAMILIES)
          + f'  from {n_seed} file(s)', flush=True)

    circ, dec, syn = P.build_probe(n, degree)
    log = []
    for rnd in range(args.rounds):
        freeze = {f: ~U[f] for f in FAMILIES}
        fz = E8.full_freeze({f: freeze[f] for f in families}, n)
        fz_pp = {f: freeze[f] for f in families}
        prefix = '' if families == FAMILIES else '_' + args.tag
        art = out_dir / f'{args.tag}{prefix}_fp{rnd}.kmb'
        t0 = time.time()
        c, h = build_frozen(n, degree, args.m, fz, art, families)
        st = c.operation_statistics()
        print(f'[round {rnd}] freeze '
              + ' '.join(f'{f}={int(freeze[f].sum())}' for f in families)
              + f'  T={st.num_toffoli_like:,} Q={c.num_qubits} sha={h[:16]}… '
              f'({time.time() - t0:.0f}s)', flush=True)
        sy, _mb, _ph = D.official_sample(int(h, 16), args.shots, args.batch_size,
                                         n=n, degree=degree)
        t1 = time.time()
        res = P.probe_sample(circ, dec, syn, sy, degree, n=n, chunk=args.chunk,
                             families=FAMILIES, freeze=fz_pp, verbose=False)
        new_sites = sum(
            int((res['union'][f] & fz_pp.get(f, np.zeros_like(res['union'][f]))).sum())
            for f in families)
        viol = {f: int(res['viol_sites'][f].sum()) for f in families}
        print(f'[round {rnd}] probe union '
              + ' '.join(f'{f}={int(res["union"][f].sum())}' for f in FAMILIES)
              + f'  viol_shots={res["viol_shots"]} viol_sites={viol} '
              f'new_sites={new_sites} ({time.time() - t1:.0f}s)', flush=True)
        log.append({'round': rnd, 'artifact': str(art), 'sha256': h,
                    'T': int(st.num_toffoli_like), 'Q': c.num_qubits,
                    'freeze': {f: int(freeze[f].sum()) for f in FAMILIES},
                    'union': {f: int(res['union'][f].sum()) for f in FAMILIES},
                    'viol_shots': res['viol_shots'], 'viol_sites': viol,
                    'new_sites': new_sites})
        if res['viol_shots'] == 0:
            print(f'[round {rnd}] CONVERGED — {snapshot(freeze)}', flush=True)
            save(out_dir, args.tag, freeze, U, res, log, h, converged=True)
            return 0
        for f in FAMILIES:
            U[f] |= res['union'][f]
        if new_sites <= args.max_new_sites:
            print(f'[round {rnd}] stall (new_sites={new_sites} <= '
                  f'{args.max_new_sites}) — stopping', flush=True)
            break
        print(f'[round {rnd}] refreezing with union now '
              + ' '.join(f'{f}={int(U[f].sum())}' for f in FAMILIES), flush=True)
    save(out_dir, args.tag, None, U, None, log, None, converged=False)
    print('[warn] no convergence in the given rounds')
    return 0


def snapshot(freeze):
    return ' '.join(f'{f}={int(freeze[f].sum())}' for f in FAMILIES)


def save(out_dir: Path, tag, freeze, U, res, log, h, *, converged):
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {'converged': converged, 'final_sha256': h, 'log': log}
    if freeze is not None:
        rec['freeze'] = {f: int(freeze[f].sum()) for f in FAMILIES}
        np.savez(out_dir / f'{tag}_freeze.npz', **freeze)
    np.savez(out_dir / f'{tag}_union.npz', **U)
    (out_dir / f'{tag}_fixpoint.json').write_text(json.dumps(rec, indent=2))
    print('wrote', out_dir / f'{tag}_fixpoint.json')


if __name__ == '__main__':
    raise SystemExit(main())

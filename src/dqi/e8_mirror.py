"""E8-D — is the `cswap` control bitmap the same in both EEA passes?

The freeze decision for `cswap` is *mirror critical* the same way it was for
`mul` (E7 §5.3): the sample union is cheap to measure on the forward pass only
(the forward EEA is a prefix of the decoder), yet the freeze mask is applied to
**both** the forward and the adjoint pass.  If the adjoint bitmap were the
iteration mirror of the forward one, the mirror-closed freezable set would be
empty and this whole line of work would collapse.

Method: build the *complete* decoder (forward EEA + Chien/Forney + reverse EEA)
with one probe ancilla per (pass, family, site), run official-distribution shots,
and compare the two passes site by site.  Under the freeze patch a site that is
live in a pass is a semantic error there, so this is the exact question the
freeze hinges on.

Usage
-----
    $P src/e8/e8_mirror.py --shots 512 --json notes/runs/e8/mirror.json
"""
from __future__ import annotations

import argparse
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

import e8_src as E8            # noqa: E402
import e6_common as C          # noqa: E402
import e6d_schedule as S       # noqa: E402

import kickmix as km           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shots', type=int, default=512)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--regime', choices=('official', 'natural'), default='official')
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    sh = E8.grid_shapes(n)
    n_iters, m = sh['n_iters'], sh['m']

    t0 = time.time()
    ns = E8.load_module('mirror', probe=True)
    circ, dec, syn = E8.build_circuit(ns, args.m, n, degree, probe=True)
    st = circ.operation_statistics()
    print(f'[build] {time.time() - t0:.1f}s Q={circ.num_qubits} '
          f'T={st.num_toffoli_like:,} instr={st.num_instructions:,}')
    pr = dec._e8_probe
    flats = {(f, p): km.array(
        [pr[f][p][it] for it in range(n_iters)] if f == 'swap'
        else [pr[f][p][it][i] for it in range(n_iters)
              for i in range(m - 1 if f == 'shift' else m)])
        for f in ('swap', 'shift', 'sweep') for p in (0, 1)}

    if args.regime == 'official':
        import e7_diag_official as D
        sy, _mb, _ph = D.official_sample(args.seed, args.shots, args.chunk,
                                         n=n, degree=degree)
    else:
        rng = np.random.default_rng(args.seed)
        sy = S.gen_data('natural', n, degree, args.shots, rng)

    unions = {}
    for f in ('swap', 'shift', 'sweep'):
        shape = E8.grid_shapes(n)[f][1:]
        unions[f] = [np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)]

    t0 = time.time()
    for start in range(0, args.shots, args.chunk):
        stop = min(start + args.chunk, args.shots)
        bits = S.bits_of(sy[start:stop], degree)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True,
                           seed=11 + start)
        sim.use_same_registers_as(circ)
        sim.write_across_shots(km.array([q for r in syn for q in r]), bits)
        sim.do(circ)
        for f in ('swap', 'shift', 'sweep'):
            shape = unions[f][0].shape + (stop - start,)
            for p in (0, 1):
                a = np.asarray(sim.read_across_shots(flats[(f, p)])).astype(bool)
                unions[f][p] |= a.reshape(shape).any(axis=-1)
    print(f'[sim] {args.shots} shots in {time.time() - t0:.1f}s')

    out = {'shots': args.shots, 'families': {}}
    for f in ('swap', 'shift', 'sweep'):
        u0, u1 = unions[f]
        same = int((u0 == u1).sum())
        tot = int(u0.size)
        out['families'][f] = {
            'sites': tot, 'forward_lit': int(u0.sum()), 'adjoint_lit': int(u1.sum()),
            'identical': bool((u0 == u1).all()), 'agree': same,
            'lit_only_forward': int((u0 & ~u1).sum()),
            'lit_only_adjoint': int((u1 & ~u0).sum()),
        }
        print(f"[{f:6s}] sites={tot} fwd_lit={int(u0.sum())} adj_lit={int(u1.sum())} "
              f"identical={bool((u0 == u1).all())} "
              f"(fwd-only {int((u0 & ~u1).sum())}, adj-only {int((u1 & ~u0).sum())})")
    allsame = all(v['identical'] for v in out['families'].values())
    print(f'[result] both passes have the identical control bitmap: {allsame}')
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

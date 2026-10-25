"""E8-C — control-qubit probe for the three `cswap` families.

Reads, per shot, the *actual* control qubit of every `cswap` call in the forward
EEA pass, by CNOT-ing that qubit into a fresh per-site ancilla (read-only
fan-out: the whole circuit is a reversible classical computation per shot, so a
CNOT out of a basis-state qubit is an exact copy).

Site grids
----------
    swap   (n_iters,)              one `cswap(swap_ctrl, 892 pairs)` per iteration
    shift  (n_iters, m-1)          `cswap(do_shift, reg_b[i-1], reg_b[i])`, i=1..m-1
    sweep  (n_iters, m)            `cswap(swaps[0]|both, q_i, reg_a[i])`

`probe_union()` returns, per family, the **union over shots** of lit sites, which
is exactly the complement of the largest statically freezable set for that
sample (a site may be frozen only if its control is |0> on *every* shot).

Usage
-----
    $P src/e8/e8_probe.py --shots 4096 --seed-from-file build/e7/dqi_fp2.kmb
    $P src/e8/e8_probe.py --shots 102400 --regime official \
        --seed-file build/e7/dqi_fp2.kmb --json notes/runs/e8/probe_first.json
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

import e8_src as E8            # noqa: E402
import e6_common as C          # noqa: E402
import e6d_schedule as S       # noqa: E402

import kickmix as km           # noqa: E402

FAMILIES = ('swap', 'shift', 'sweep', 'mul')


def build_probe(n: int = 70, degree: int = 12):
    ns = E8.load_module('fwdprobe', probe=True, forward_only=True)
    circ, dec, syn = E8.build_circuit(ns, n, n, degree, probe=True)
    return circ, dec, syn


def flat_sites(pr, fam: str, p: int, n_iters: int, m: int):
    """Flat list of probe ancillas in canonical site order for one family/pass."""
    if fam == 'swap':
        return [pr['swap'][p][it] for it in range(n_iters)]
    if fam == 'shift':
        return [pr['shift'][p][it][i] for it in range(n_iters) for i in range(m - 1)]
    if fam == 'sweep':
        return [pr['sweep'][p][it][i] for it in range(n_iters) for i in range(m)]
    return [pr['mul'][p][it][i] for it in range(n_iters) for i in range(m)]


def union_shape(fam: str, n_iters: int, m: int):
    if fam == 'swap':
        return (n_iters,)
    if fam == 'shift':
        return (n_iters, m - 1)
    return (n_iters, m)


def probe_sample(circ, dec, syn, syndromes, degree, *, n: int = 70,
                 chunk: int = 2048, families=FAMILIES, freeze=None,
                 p: int = 0, seed: int = 7, verbose: bool = True):
    """Union over shots of lit control sites + per-shot violation stats.

    Returns dict with `union` per family (bool arrays, forward pass only) and,
    if `freeze` is given, `viol_shots` (shots touching a frozen site) plus the
    per-site violation counts.
    """
    sh = E8.grid_shapes(n)
    n_iters, m = sh['n_iters'], sh['m']
    pr = dec._e8_probe
    flats = {f: km.array(flat_sites(pr, f, p, n_iters, m)) for f in families}
    union = {f: np.zeros(union_shape(f, n_iters, m), dtype=bool) for f in families}
    viol = {f: np.zeros_like(union[f]) for f in families} if freeze else None
    n_shots = syndromes.shape[0]
    viol_shots = 0
    t0 = time.time()
    for start in range(0, n_shots, chunk):
        stop = min(start + chunk, n_shots)
        data = syndromes[start:stop]
        bits = S.bits_of(data, degree)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True,
                           seed=seed + start)
        sim.use_same_registers_as(circ)
        sim.write_across_shots(km.array([q for r in syn for q in r]), bits)
        sim.do(circ)
        hit_any = np.zeros(stop - start, dtype=bool)
        for f in families:
            a = np.asarray(sim.read_across_shots(flats[f])).astype(bool).reshape(
                union_shape(f, n_iters, m) + (stop - start,))
            u = a.any(axis=-1)
            union[f] |= u
            if freeze is not None:
                fz = freeze.get(f)
                if fz is not None and fz.ndim > a.ndim - 1:
                    fz = fz[0]
                if fz is not None and fz.any():
                    lit = a & fz[..., None]
                    viol[f] |= lit.any(axis=-1)
                    hit_any |= lit.reshape(-1, lit.shape[-1]).any(axis=0)
        viol_shots += int(hit_any.sum())
        if verbose and (start // chunk) % 5 == 0:
            done = stop / n_shots
            print(f'  [probe] {stop}/{n_shots} ({done:5.1%}) '
                  f'union swap={int(union["swap"].sum())} '
                  f'shift={int(union["shift"].sum())} '
                  f'sweep={int(union["sweep"].sum())} '
                  f'viol={viol_shots} ({time.time() - t0:.0f}s)', flush=True)
    out = {'union': union, 'viol_shots': viol_shots, 'viol_sites': viol,
           'shots': n_shots, 'sec': round(time.time() - t0, 1)}
    return out


def summarise(union, n: int, degree: int = 12):  # noqa: D401
    """Freeze sizes / saved T implied by a union."""
    sh = E8.grid_shapes(n)
    m, ni = sh['m'], sh['n_iters']
    swap_t = 2 * (m.bit_length() + 1) + m * degree
    dec = {k: v for k, v in union.items()}
    sw_f = int((~dec['swap']).sum())
    sh_f = int((~dec['shift']).sum())
    sp_f = int((~dec['sweep']).sum())
    return {
        'swap_union': int(dec['swap'].sum()), 'swap_sites': int(dec['swap'].size),
        'swap_freeze': sw_f, 'swap_T': sw_f * swap_t,
        'shift_union': int(dec['shift'].sum()), 'shift_sites': int(dec['shift'].size),
        'shift_freeze': sh_f, 'shift_T': sh_f * degree,
        'sweep_union': int(dec['sweep'].sum()), 'sweep_sites': int(dec['sweep'].size),
        'sweep_freeze': sp_f, 'sweep_T': sp_f * degree,
        'freeze_T_total': sw_f * swap_t + (sh_f + sp_f) * degree,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--chunk', type=int, default=2048)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--regime', choices=('official', 'natural', 'generic'),
                    default='generic')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--seed-file', default=None,
                    help='derive the sample seed from sha256(this file) (official)')
    ap.add_argument('--save-union', default=None)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    circ, dec, syn = build_probe(args.n, args.degree)
    st = circ.operation_statistics()
    print(f'[probe-circuit] Q={circ.num_qubits} T={st.num_toffoli_like:,} '
          f'instr={st.num_instructions:,}')

    if args.regime == 'official':
        import e7_diag_official as D
        if args.seed_file:
            h = hashlib.sha256(Path(args.seed_file).read_bytes()).hexdigest()
            base = int(h, 16)
            print(f'[seed] sha256({Path(args.seed_file).name})={h[:16]}…')
        else:
            base = args.seed if args.seed is not None else 20261025
        sy, _mb, _ph = D.official_sample(base, args.shots, args.batch_size,
                                         n=args.n, degree=args.degree)
    else:
        rng = np.random.default_rng(args.seed if args.seed is not None else 1)
        sy = S.gen_data(args.regime, args.n, args.degree, args.shots, rng)

    res = probe_sample(circ, dec, syn, sy, args.degree, n=args.n,
                       chunk=args.chunk)
    s = summarise(res['union'], args.n, args.degree)
    print('[union]', json.dumps(s, indent=2))
    print(f'[time] {res["sec"]}s for {res["shots"]} shots')
    if args.save_union:
        Path(args.save_union).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_union, **res['union'])
        print('wrote', args.save_union)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {'shots': res['shots'], 'sec': res['sec'], 'summary': s}, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

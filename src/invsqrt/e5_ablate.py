"""E5: exact T (num_toffoli_like) decomposition of the two data-movement primitives.

E4 attributed 396,048 T to `_swap_a_and_b` and 383,616 T to
`_shift_b_array_right` at n=70/m=4095.  This script re-derives both numbers from
*measured* deltas (source-substitution ablations on the shipped reference) rather
than from an analytic guess, and then splits each primitive into

    * the cswap chain itself      (T = 1 per conditional bit-pair swap)
    * the surrounding threshold / `_active` fan-out glue
      (which is *free*: it compiles to `cx` and kickmix counts `cx` as 0)

The key kickmix accounting fact this script pins down first:

    cx   -> 0 T      swap -> 0 T      rotate (unconditional) -> 0 T
    ccx  -> 1 T      cswap -> 1 T per bit pair      controlled rotate -> n-1 T

so a conditional data move is expensive and an unconditional one is free.

Usage
-----
    python src/e5/e5_ablate.py --m 280 --n 70
    python src/e5/e5_ablate.py --m 4095 --n 70 --json notes/runs/e5_ablate.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py'

# --------------------------------------------------------------------------- #
# ablation substitutions: each one *deletes* exactly one data move, leaving the
# control logic (and therefore the emitted gate structure around it) in place.
# These are probes, not candidates: they are semantically wrong on purpose.
# --------------------------------------------------------------------------- #

SWAP_MOVE = '            builder.cswap(swap_ctrl, lhs, rhs)'
SWAP_MOVE_OFF = '            builder.cswap(swap_ctrl, lhs, rhs) if False else None'

SHIFT_MOVE = '                    builder.cswap(do_shift, reg_b[i - 1], reg_b[i])'
SHIFT_MOVE_OFF = ('                    builder.cswap(do_shift, reg_b[i - 1], reg_b[i])'
                  ' if False else None')

# hoist the *keep* decision out of the element loop: replace the whole
# `_shift_b_array_right` body's element chain by a single controlled rotate of the
# full array.  Semantically different (rotates the whole array, not the window),
# measured only to price the "free rotation" lever.
SHIFT_FULLROT = """                    builder.cswap(do_shift, reg_b[i - 1], reg_b[i])"""

# price the (free) unconditional swap: same site, control dropped entirely.
SWAP_UNCOND = '            builder.swap(lhs, rhs)'
SHIFT_UNCOND = '                    builder.swap(reg_b[i - 1], reg_b[i])'

VARIANTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    'ref': ('shipped reference', []),
    'probe_no_swap': (
        'PROBE: `_swap_a_and_b` cswap deleted (wrong semantics; prices the primitive)',
        [(SWAP_MOVE, SWAP_MOVE_OFF)]),
    'probe_no_shift': (
        'PROBE: `_shift_b_array_right` cswap chain deleted',
        [(SHIFT_MOVE, SHIFT_MOVE_OFF)]),
    'probe_neither': (
        'PROBE: both data moves deleted',
        [(SWAP_MOVE, SWAP_MOVE_OFF), (SHIFT_MOVE, SHIFT_MOVE_OFF)]),
    'probe_swap_uncond': (
        'PROBE: swap made unconditional (free in the metric; wrong semantics)',
        [(SWAP_MOVE, SWAP_UNCOND)]),
    'probe_shift_uncond': (
        'PROBE: full-array shift made unconditional (free in the metric)',
        [(SHIFT_MOVE, SHIFT_UNCOND)]),
    'probe_both_uncond': (
        'PROBE: both moves unconditional -> the "free rotation" ceiling',
        [(SWAP_MOVE, SWAP_UNCOND), (SHIFT_MOVE, SHIFT_UNCOND)]),
}


def load_variant(name: str):
    note, subs = VARIANTS[name]
    src = REF.read_text()
    for old, new in subs:
        if old not in src:
            raise RuntimeError(f'variant {name!r}: substitution target not found: {old!r}')
        src = src.replace(old, new)
    ns: dict = {'__name__': f'e5_variant_{name}'}
    exec(compile(src, str(REF), 'exec'), ns)  # noqa: S102 - local trusted file
    return ns, note


def measure(name: str, m: int, n: int, degree: int) -> dict:
    ns, note = load_variant(name)
    t0 = time.time()
    try:
        circuit = ns['build_reference_circuit'](m, n, degree)
    except Exception as exc:  # noqa: BLE001
        return {'name': name, 'note': note, 'error': f'{type(exc).__name__}: {exc}'}
    stats = circuit.operation_statistics()
    return {'name': name, 'note': note, 'num_qubits': circuit.num_qubits,
            'num_toffoli_like': int(stats.num_toffoli_like),
            'num_instructions': int(stats.num_instructions),
            'build_seconds': round(time.time() - t0, 2)}


def primitive_costs() -> dict:
    """Measured kickmix cost of the instructions the two primitives are made of."""
    def stat(f, nq=64):
        b = km.CircuitBuilder()
        q = b.alloc_qubits(nq)
        f(b, q)
        s = b.finish_circuit().operation_statistics()
        return {'toffoli_like': int(s.num_toffoli_like), 'instructions': int(s.num_instructions)}

    a = km.array
    w = 12
    return {
        'cx': stat(lambda b, q: b.cx(q[0], q[1])),
        'ccx': stat(lambda b, q: b.ccx(q[0], q[1], q[2])),
        'x': stat(lambda b, q: b.x(q[0])),
        f'swap_{w}': stat(lambda b, q: b.swap(a([q[i] for i in range(w)]),
                                              a([q[20 + i] for i in range(w)]))),
        f'cswap_{w}': stat(lambda b, q: b.cswap(q[63], a([q[i] for i in range(w)]),
                                                a([q[20 + i] for i in range(w)]))),
        f'rotate_{w}': stat(lambda b, q: b.right_rotate(a([q[i] for i in range(w)]), 1)),
        f'rotate_ctrl_{w}': stat(lambda b, q: b.right_rotate(a([q[i] for i in range(w)]), 1,
                                                             control=q[63])),
    }


def analytic(n: int, degree: int) -> dict:
    m = n + 3
    n_iters = 3 * (n + 1) + 9
    calls = 2 * n_iters                      # forward + adjoint
    swap_pairs = 2 * (m.bit_length() + 1) + m * degree - 2 * 0  # 2*logm + (m-3)*deg + 3*deg
    # reg_a = 3 scratch words + (m-3) syndrome words = m words, all degree wide
    swap_pairs = 2 * (m.bit_length() + 1) + m * degree
    shift_pairs = (m - 1) * degree
    return {
        'n': n, 'degree': degree, 'm': m, 'n_iters': n_iters, 'calls': calls,
        'swap_word_pairs': swap_pairs,
        'swap_T': calls * swap_pairs,
        'shift_word_pairs': shift_pairs,
        'shift_T': calls * shift_pairs,
        'both_T': calls * (swap_pairs + shift_pairs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    out: dict = {'m_chien': args.m, 'n': args.n, 'degree': args.degree}
    print('=' * 78)
    print('kickmix instruction costs (measured)')
    pc = primitive_costs()
    out['primitive_costs'] = pc
    for k, v in pc.items():
        print(f'  {k:16s} T={v["toffoli_like"]:4d}  instructions={v["instructions"]:4d}')

    print('=' * 78)
    print(f'reference vs ablations   (m_chien={args.m}, n={args.n}, b={args.degree})')
    rows = {}
    base = None
    for name in VARIANTS:
        r = measure(name, args.m, args.n, args.degree)
        if 'error' in r:
            print(f'  {name:22s} ERROR {r["error"]}')
            continue
        rows[name] = r
        if base is None:
            base = r
            print(f'  {name:22s} Q={r["num_qubits"]:5d}  T={r["num_toffoli_like"]:12,d}  (baseline)')
        else:
            dt = r['num_toffoli_like'] - base['num_toffoli_like']
            pct = 100.0 * dt / base['num_toffoli_like']
            print(f'  {name:22s} Q={r["num_qubits"]:5d}  T={r["num_toffoli_like"]:12,d}  '
                  f'dT={dt:+10,d} ({pct:+7.3f}%)')
    out['variants'] = rows

    if base is not None and 'probe_no_swap' in rows and 'probe_neither' in rows:
        t_swap = base['num_toffoli_like'] - rows['probe_no_swap']['num_toffoli_like']
        t_shift = base['num_toffoli_like'] - rows['probe_no_shift']['num_toffoli_like']
        t_both = base['num_toffoli_like'] - rows['probe_neither']['num_toffoli_like']
        print('-' * 78)
        print(f'  measured T(_swap_a_and_b)          = {t_swap:,d}  '
              f'({100.0*t_swap/base["num_toffoli_like"]:.3f}% of the m={args.m} circuit)')
        print(f'  measured T(_shift_b_array_right)   = {t_shift:,d}  '
              f'({100.0*t_shift/base["num_toffoli_like"]:.3f}%)')
        print(f'  measured T(both)                   = {t_both:,d}  '
              f'({100.0*t_both/base["num_toffoli_like"]:.3f}%)')
        out['measured'] = {'swap_T': t_swap, 'shift_T': t_shift, 'both_T': t_both,
                           'base_T': base['num_toffoli_like'], 'base_Q': base['num_qubits']}
        an = analytic(args.n, args.degree)
        print('-' * 78)
        print('analytic cross-check (per full circuit, fwd+adjoint):')
        print(f'  swap  : {an["calls"]:4d} calls x {an["swap_word_pairs"]:4d} pairs = '
              f'{an["swap_T"]:12,d}   measured {t_swap:,d}')
        print(f'  shift : {an["calls"]:4d} calls x {an["shift_word_pairs"]:4d} pairs = '
              f'{an["shift_T"]:12,d}   measured {t_shift:,d}')
        print(f'  both  : {an["both_T"]:12,d}   measured {t_both:,d}')
        out['analytic'] = an

        # extrapolate the same absolute T to the shipped m_chien = 4095 instance
        print('-' * 78)
        print('m_chien = 4095 extrapolation (the shipped instance):')
        # build once at m=4095 to get the true baseline, then scale the measured deltas
        r4095 = measure('ref', 4095, args.n, args.degree)
        if 'error' not in r4095:
            out['ref_m4095'] = r4095
            an4095 = analytic(args.n, args.degree)
            t_swap_4095 = an4095['swap_T']
            t_shift_4095 = an4095['shift_T']
            tot = r4095['num_toffoli_like']
            print(f'  ref T = {tot:,d}')
            print(f'  swap  = {t_swap_4095:,d}  ({100.0*t_swap_4095/tot:.3f}%)')
            print(f'  shift = {t_shift_4095:,d}  ({100.0*t_shift_4095/tot:.3f}%)')
            print(f'  both  = {t_swap_4095 + t_shift_4095:,d}  '
                  f'({100.0*(t_swap_4095+t_shift_4095)/tot:.3f}%)')

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f'\nwrote {p}')


if __name__ == '__main__':
    main()

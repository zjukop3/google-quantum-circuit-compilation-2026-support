"""E8-A — exact T (num_toffoli_like) ledger of the three `cswap` families.

E5 (`notes/E5_EEA_POINTER.md`) measured `_swap_a_and_b` = 396,048 T and
`_shift_b_array_right` = 383,616 T by source ablation.  This script re-derives
those numbers *and* prices the third, so-far-unpriced family: the
position-controlled quotient swap inside `sweep_positions`
(`builder.cswap(swaps[0], q_i, reg_a[i])`).

Each ablation deletes exactly one `cswap` call site and nothing else, so the
delta is the price of that call site.  Alongside it the script prints the
*analytic* site grid E8 models:

    swap_a_and_b      2 passes x 222 iters                      x 892 pairs
    shift_b_right     2 passes x 222 iters x 72 positions       x  12 pairs
    sweep quotient    2 passes x 222 iters x 73 positions       x  12 pairs

Usage
-----
    $P src/e8/e8_price.py --m 4095 --n 70
    $P src/e8/e8_price.py --m 280 --n 70 --json notes/runs/e8/price_m280.json
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

SWAP_MOVE = '            builder.cswap(swap_ctrl, lhs, rhs)'
SHIFT_MOVE = '                    builder.cswap(do_shift, reg_b[i - 1], reg_b[i])'
SWEEP_ONE = '                            builder.cswap(swaps[0], q_i, reg_a[i])'
SWEEP_BOTH = '                    builder.cswap(both, q_i, reg_a[i])'

OFF = ' if False else None'
IND = ' ' * 4

VARIANTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    'ref': ('shipped reference', []),
    'no_swap': ('delete `_swap_a_and_b` cswap (892 pairs/call)',
                [(SWAP_MOVE, SWAP_MOVE + OFF)]),
    'no_shift': ('delete `_shift_b_array_right` cswap chain (72 x 12 pairs/call)',
                 [(SHIFT_MOVE, SHIFT_MOVE + OFF)]),
    'no_sweep': ('delete the `sweep_positions` quotient cswap (12 pairs/position)',
                 [(SWEEP_ONE, SWEEP_ONE + OFF)]),
    'no_sweep_both': ('delete the `sweep_positions` quotient cswap incl. 2-swap branch',
                      [(SWEEP_ONE, SWEEP_ONE + OFF),
                       (SWEEP_BOTH, SWEEP_BOTH + OFF)]),
    'no_all': ('delete all three cswap families',
               [(SWAP_MOVE, SWAP_MOVE + OFF), (SHIFT_MOVE, SHIFT_MOVE + OFF),
                (SWEEP_ONE, SWEEP_ONE + OFF)]),
}


def load_variant(name: str):
    note, subs = VARIANTS[name]
    src = REF.read_text()
    for old, new in subs:
        if old not in src:
            raise RuntimeError(f'{name}: substitution target missing: {old!r}')
        src = src.replace(old, new, 1)
    ns: dict = {'__name__': f'e8_variant_{name}'}
    exec(compile(src, str(REF), 'exec'), ns)
    return ns, note


def primitive_costs() -> dict:
    def stat(f, nq=64):
        b = km.CircuitBuilder()
        q = b.alloc_qubits(nq)
        f(b, q)
        s = b.finish_circuit().operation_statistics()
        return int(s.num_toffoli_like)

    a = km.array
    w = 12
    return {
        'cx': stat(lambda b, q: b.cx(q[0], q[1])),
        f'cswap_{w}': stat(lambda b, q: b.cswap(q[63], a([q[i] for i in range(w)]),
                                                a([q[20 + i] for i in range(w)]))),
        f'cswap_{892}': None,
        'cswap_pairs': None,
    }


def cswap_scaling() -> dict:
    """T of one `cswap` as a function of the pair count (should be 1 T / pair)."""
    out = {}
    for w in (1, 2, 4, 8, 12, 16, 64, 892):
        b = km.CircuitBuilder()
        q = b.alloc_qubits(2 * w + 1)
        b.cswap(q[0], km.array([q[1 + i] for i in range(w)]),
                km.array([q[1 + w + i] for i in range(w)]))
        out[str(w)] = int(b.finish_circuit().operation_statistics().num_toffoli_like)
    return out


def analytic(n: int, degree: int) -> dict:
    m = n + 3
    n_iters = 3 * (n + 1) + 9
    calls = 2 * n_iters
    swap_pairs = 2 * (m.bit_length() + 1) + m * degree
    shift_pairs = (m - 1) * degree
    return {
        'n': n, 'degree': degree, 'm': m, 'n_iters': n_iters, 'calls': calls,
        'swap_sites_per_pass': n_iters, 'swap_T_per_site': swap_pairs,
        'swap_T': calls * swap_pairs,
        'shift_sites_per_pass': n_iters * (m - 1),
        'shift_T_per_site': degree,
        'shift_T': calls * shift_pairs,
        'both_T': calls * (swap_pairs + shift_pairs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=4095)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    print('[primitives]', json.dumps(primitive_costs()))
    print('[cswap scaling w -> T]', json.dumps(cswap_scaling()))

    res = {}
    for name in VARIANTS:
        ns, note = load_variant(name)
        t0 = time.time()
        circ = ns['build_reference_circuit'](args.m, args.n, args.degree)
        st = circ.operation_statistics()
        res[name] = {'note': note, 'num_qubits': circ.num_qubits,
                     'T': int(st.num_toffoli_like),
                     'instructions': int(st.num_instructions),
                     'sec': round(time.time() - t0, 1)}
        print(f"[{name:14s}] T={res[name]['T']:>12,}  Q={res[name]['num_qubits']:>6}  "
              f"({res[name]['sec']}s)  {note}", flush=True)

    base = res['ref']['T']
    for name in VARIANTS:
        if name != 'ref':
            res[name]['delta_T'] = res[name]['T'] - base
            print(f"[delta {name:14s}] {res[name]['delta_T']:>+12,}")

    ana = analytic(args.n, args.degree)
    print('[analytic grid]', json.dumps(ana, indent=2))
    out = {'m': args.m, 'n': args.n, 'degree': args.degree, 'variants': res,
           'analytic': ana, 'cswap_scaling': cswap_scaling()}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

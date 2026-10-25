"""E4: exact qubit (Q) anatomy of the dqi reference decoder.

Two independent measurements, both on the shipped reference circuit:

1. **Allocation ledger** -- wrap `CircuitBuilder.alloc_qubits` and record every
   allocation against the enclosing `builder.mark(...)` stack.  Gives the
   subsystem breakdown of the *persistent* width.

2. **Peak attribution** -- build partial circuits (`initial_shift`,
   `+ forward_eea`, `+ chien`, `+ reverse_eea`) and read `circuit.num_qubits`
   for each.  `num_qubits` is the peak of `num_allocated_qubits` over the whole
   build, so differencing the prefixes says which stage *sets* Q.

Usage
-----
    python src/e4/qbreak.py --m 280 --n 70
    python src/e4/qbreak.py --m 280 --n 70 --json notes/runs/e4_qbreak.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py'


def load(ns_name: str = 'ref'):
    src = REF.read_text()
    ns: dict = {'__name__': ns_name}
    exec(compile(src, str(REF), 'exec'), ns)  # noqa: S102 - local trusted file
    return ns


# --------------------------------------------------------------------------- #
# 1. allocation ledger under the mark() stack
# --------------------------------------------------------------------------- #

def ledger(m: int, n: int, degree: int) -> dict:
    ns = load()
    B = km.CircuitBuilder
    orig_alloc = B.alloc_qubits
    orig_mark = B.mark
    stack: list[str] = []
    totals: collections.Counter = collections.Counter()

    class MarkCtx:
        def __init__(self, label, inner):
            self.label, self.inner = label, inner

        def __enter__(self):
            stack.append(self.label)
            return self.inner.__enter__()

        def __exit__(self, *a):
            r = self.inner.__exit__(*a)
            stack.pop()
            return r

    def mark(self, label):
        return MarkCtx(label, orig_mark(self, label))

    def alloc(self, k, *a, **kw):
        totals[tuple(stack) if stack else ('<top>',)] += k
        return orig_alloc(self, k, *a, **kw)

    B.alloc_qubits = alloc
    B.mark = mark
    try:
        circuit = ns['build_reference_circuit'](m, n, degree)
    finally:
        B.alloc_qubits = orig_alloc
        B.mark = orig_mark

    return {
        'num_qubits': circuit.num_qubits,
        'num_bits': circuit.num_bits,
        'num_toffoli_like': circuit.operation_statistics().num_toffoli_like,
        'alloc_ledger': {' > '.join(k): v for k, v in totals.most_common()},
        'alloc_total': sum(totals.values()),
    }


# --------------------------------------------------------------------------- #
# 2. peak attribution by building prefixes of the reference pipeline
# --------------------------------------------------------------------------- #

SHIFT = """        with builder.mark('initial_shift'):
            self._initial_shift_loop(builder, reg_a, n_A, is_adjoint=False)
"""
FWD = """        with builder.mark('forward_eea'):
            self._synchronized_eea(builder, *eea, n_iters=n_iters, is_adjoint=False)
"""
CHIEN = """        with builder.mark('chien_forney_mbuc'):
            self._chien_search_and_forney_mbuc(builder, reg_b, error_mbuc, n_b, n_B)
"""
REV = """        with builder.mark('reverse_eea'):
            self._synchronized_eea(builder, *eea, n_iters=n_iters, is_adjoint=True)
"""

STAGES = [
    ('initial_shift only', [FWD, CHIEN, REV]),
    ('+ forward_eea', [CHIEN, REV]),
    ('+ chien_forney', [REV]),
    ('+ reverse_eea (full)', []),
]


def peak_attribution(m: int, n: int, degree: int) -> dict:
    out: dict[str, int] = {}
    for name, drop in STAGES:
        src = REF.read_text()
        for block in drop:
            if block not in src:
                raise RuntimeError(f'stage block not found: {block!r}')
            src = src.replace(block, '', 1)
        ns: dict = {'__name__': f'peak_{name}'}
        exec(compile(src, str(REF), 'exec'), ns)  # noqa: S102
        circuit = ns['build_reference_circuit'](m, n, degree)
        out[name] = circuit.num_qubits
    return out


# --------------------------------------------------------------------------- #
# 3. the analytic ledger (what the 1893 is made of, by hand)
# --------------------------------------------------------------------------- #

def analytic(n: int, degree: int) -> dict:
    m_e = n + 3
    logm = m_e.bit_length() + 1
    n_iters = 3 * (n + 1) + 9
    persistent = {
        'syndromes S0..S{n-1} (contract input, reused as reg_a[3:])':
            n * degree,
        'reg_a[0:3] low-degree headroom words': 3 * degree,
        'reg_b[0:m_e-1] auxiliary EEA polynomial': m_e * degree,
        'n_A, n_B, n_a, n_b, n_q (log2(m_e)+1 each)': 5 * logm,
        'halting_counter (log2(n_iters))': n_iters.bit_length(),
        'counter': 2,
        'flag': 1,
    }
    transient_eea = {
        'is_c3 (sub-step-3 predicate)': 1,
        'is_hc_zero (halting counter == 0)': 1,
        'ctrl (4-bit one-hot sub-step decode)': 4,
        'q_i (quotient coefficient, live across the sweep)': degree,
        'ratio (division result, must survive init..del)': degree,
        'division scaffold (divisor inverse + 4 Itoh-Tsujii chain words)':
            degree + 4 * degree,
    }
    transient_chien = {
        'sigma_prime_inv': degree,
        'located': degree,
        'is_sigma_zero': 1,
        'Itoh-Tsujii chain kept for del_gf2_inverse_with_scaffold': 4 * degree,
    }
    return {
        'm_e': m_e, 'logm': logm, 'n_iters': n_iters,
        'persistent': persistent, 'persistent_total': sum(persistent.values()),
        'transient_eea': transient_eea, 'transient_eea_total': sum(transient_eea.values()),
        'transient_chien': transient_chien, 'transient_chien_total': sum(transient_chien.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--json', type=str, default=None)
    args = ap.parse_args()

    led = ledger(args.m, args.n, args.degree)
    peaks = peak_attribution(args.m, args.n, args.degree)
    an = analytic(args.n, args.degree)

    print(f"=== Q = circuit.num_qubits (m_chien={args.m}, n={args.n}, degree={args.degree}) ===")
    print(f"Q = {led['num_qubits']}   T = {led['num_toffoli_like']:,}   bits = {led['num_bits']:,}")
    print()
    print("--- allocation ledger (sum of every alloc_qubits, grouped by mark) ---")
    for k, v in led['alloc_ledger'].items():
        print(f"  {v:6d}  {k}")
    print()
    print("--- peak attribution (num_qubits of each pipeline prefix) ---")
    prev = 0
    for k, v in peaks.items():
        print(f"  {k:16s} Q={v:5d}   (delta from previous prefix {v - prev:+d})")
        prev = v
    print()
    print("--- analytic anatomy ---")
    print(f"  m_e = n+3 = {an['m_e']}, logm = {an['logm']}, n_iters = {an['n_iters']}")
    for label, tbl in (('persistent', an['persistent']),
                       ('transient @ forward-EEA quotient_div', an['transient_eea']),
                       ('transient @ Chien Forney loop', an['transient_chien'])):
        print(f"  [{label}]")
        for k, v in tbl.items():
            print(f"     {v:5d}  {k}")
        print(f"     {sum(tbl.values()):5d}  subtotal")
    print()
    print(f"  persistent {an['persistent_total']} + EEA transient {an['transient_eea_total']}"
          f" = {an['persistent_total'] + an['transient_eea_total']}")
    print(f"  persistent {an['persistent_total']} + Chien transient {an['transient_chien_total']}"
          f" = {an['persistent_total'] + an['transient_chien_total']}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {'ledger': led, 'peaks': peaks, 'analytic': an}, indent=2))
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

"""Optimized invsqrt (Benchmark I) circuit: reference algorithm, leaner Horner registers.

Drop-in replacement for `data/verifiers/invsqrt/nr_reference_circuit.build_nr_circuit`.
The *arithmetic is bit-identical* to the reference -- `fixed_point_model` and
`relative_error` from the shipped module describe this circuit exactly -- so the
accuracy argument carries over unchanged.  Only the register allocation differs.

The saving
----------
The reference allocates four full-width coefficient registers `c0..c3` and keeps all
of them live for the whole of Horner's rule:

    step 1:  c2 -= T(c3 * delta)      c3 dead afterwards
    step 2:  c1 -= T(r1 * delta)      uses r1, which lives in c2
    step 3:  c0 -= T(r2 * delta)      uses r2, which lives in c1

Measured lifetimes (forward pass):

    register   first use        last use
    c3         step 1 (mult)    end of step 1
    c2         step 1 (acc)     step 2 (mult)
    c1         step 2 (acc)     step 3 (mult)
    c0         step 3 (acc)     -> becomes y0

`c3` is dead the moment step 1 ends and `c0` is not touched until step 3, so three
registers suffice:

    A <- c0            (clean at start; idle until step 3)
    C <- c2            (acc of step 1, then holds r1)
    B <- c3            (mult of step 1)
        step 1:  C -= T(B * delta)
        B ^= c3        (classical clear, Clifford/free);  B <- c1
        step 2:  B -= T(C * delta)
        step 3:  A -= T(B * delta)          A = y0

`c2`'s slot (`C`) is *not* reusable for `c0`: after step 1 it holds the quantum state
`r1`, which cannot be cleared by a classical XOR.  Hence 3 registers, not 2.  Every
load/clear is `gf2_iadd` (Clifford, zero Toffoli), so the Toffoli count is unchanged
and `num_qubits` drops by exactly `coef_bits`.

The reverse pass is the literal inverse of the forward op sequence (same operations,
reversed, each self-inverse except `_add_row`, which flips `inverse`).

Cost model (measured, `Precision()` defaults + this change):
    reference : Q = 260, Toffolis = 7735, score = 260 * 7987.59 = 2,076,774
    this file : Q = 243, Toffolis = 7735  ->  ~1.070x
"""

from __future__ import annotations

import sys
from pathlib import Path

import kickmix as km

# Reuse the shipped reference verbatim for everything except the Horner register layout.
_VERIFIER_DIR = Path(__file__).resolve().parent.parent / "data" / "verifiers" / "invsqrt"
if str(_VERIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(_VERIFIER_DIR))

from nr_reference_circuit import (  # noqa: E402
    Precision,
    _and,
    _add_row,
    _correct,
    _denormalize,
    _multiply_m,
    _multiply_y0,
    _normalize,
    _phase_by_bits,
    _select_interval,
    _square,
    fixed_point_model,
    relative_error,
)

__all__ = ["Precision", "build_opt_circuit", "fixed_point_model", "relative_error", "COEF_SLOTS"]

# Coefficient slots in the lean layout: (acc/mult role per Horner step).
COEF_SLOTS = 3


def _load_coef(builder: km.CircuitBuilder, prec: Precision, x: km.array,
               selectors: km.array, reg: km.array, j: int) -> None:
    """XOR coefficient `j` of the selected interval into `reg` (Clifford, its own inverse).

    Identical to the reference `_load_coefficients`, restricted to a single coefficient,
    so it can be issued lazily at the point the coefficient is first needed.
    """
    top, t_a, t_b = x[len(x) - 1], selectors[0], selectors[1]
    table = prec.coefficient_table()
    for control, value in ((True, table[0][j]), (top, table[0][j] ^ table[2][j]),
                           (t_a, table[0][j] ^ table[1][j]), (t_b, table[2][j] ^ table[3][j])):
        builder.gf2_iadd(value, target=reg, control=control)


def _horner3(builder: km.CircuitBuilder, inverse: bool, prec: Precision, x: km.array,
             selectors: km.array, a: km.array, b: km.array, c: km.array,
             scratch: km.array) -> None:
    """`y0 = c0 - (c1 - (c2 - c3 delta) delta) delta` on three coefficient registers.

    Forward leaves `y0` in `a`; `inverse` restores `a, b, c` to `c0, c3, c2`.
    """
    width, wc = len(x), len(a)
    delta = [(selectors[2], 1)]  # (qubit, m) with weight 2^-m: the half bit, then bits of M.
    delta += [(x[width - 2 - m], m) for m in range(2, min(prec.delta_bits, width - 2) + 1)]

    def step(acc: km.array, mult: km.array) -> None:
        for control, m in delta:
            _add_row(builder, inverse, control, [(mult[i], i - m) for i in range(m, wc)],
                     acc, scratch, subtract=True)

    def load(reg: km.array, j: int) -> None:
        _load_coef(builder, prec, x, selectors, reg, j)

    if inverse:
        # Literal inverse of the forward op sequence below, in reverse order.
        # Forward ops on b are  load(b,3), load(b,3), load(b,1)  ->  three inverses here.
        step(a, b)          # undo step 3  (a -= T(b d))
        step(b, c)          # undo step 2  (b -= T(c d))
        load(b, 1)          # undo load(b,1)  -> b clean
        load(b, 3)          # undo the second load(b,3)  -> b = c3
        step(c, b)          # undo step 1  (c -= T(b d))
        load(b, 3)          # undo the first load(b,3)   -> b clean
        load(c, 2)          # undo load(c,2)  -> c clean
        load(a, 0)          # undo load(a,0)  -> a clean
    else:
        load(a, 0)          # a := c0            (idle until step 3)
        load(c, 2)          # c := c2
        load(b, 3)          # b := c3
        step(c, b)          # c := c2 - T(c3 delta)  = r1
        load(b, 3)          # b ^= c3 -> 0        (c3 is dead)
        load(b, 1)          # b := c1
        step(b, c)          # b := c1 - T(r1 delta) = r2
        step(a, b)          # a := c0 - T(r2 delta) = y0


def _compute_y1_opt(builder: km.CircuitBuilder, inverse: bool, prec: Precision, x: km.array,
                    flags: km.array, selectors: km.array, coefs: list[km.array], square: km.array,
                    z: km.array, y1: km.array, scratch: km.array) -> None:
    """The reference `_compute_y1`, with the three-register Horner."""
    a, b, c = coefs  # a=c0 slot, b=c3->c1->r2, c=c2->r1
    y0, p = a, y1[:len(y1) - len(flags)]
    with builder.mark('uncompute_y1' if inverse else 'compute_y1'):
        if inverse:
            with builder.mark('denormalize'):
                _denormalize(builder, inverse, flags, y1)
            with builder.mark('multiply_y0'):
                _multiply_y0(builder, inverse, y0, z, p, scratch)
            with builder.mark('correct'):
                _correct(builder, inverse, z)
            with builder.mark('multiply_m'):
                _multiply_m(builder, inverse, x, square, z, scratch)
            with builder.mark('square'):
                _square(builder, inverse, y0, square, scratch)
            with builder.mark('horner'):
                _horner3(builder, inverse, prec, x, selectors, a, b, c, scratch)
            with builder.mark('select_interval'):
                _select_interval(builder, inverse, x, selectors)
            with builder.mark('normalize'):
                _normalize(builder, inverse, x, flags)
        else:
            with builder.mark('normalize'):
                _normalize(builder, inverse, x, flags)
            with builder.mark('select_interval'):
                _select_interval(builder, inverse, x, selectors)
            with builder.mark('horner'):
                _horner3(builder, inverse, prec, x, selectors, a, b, c, scratch)
            with builder.mark('square'):
                _square(builder, inverse, y0, square, scratch)
            with builder.mark('multiply_m'):
                _multiply_m(builder, inverse, x, square, z, scratch)
            with builder.mark('correct'):
                _correct(builder, inverse, z)
            with builder.mark('multiply_y0'):
                _multiply_y0(builder, inverse, y0, z, p, scratch)
            with builder.mark('denormalize'):
                _denormalize(builder, inverse, flags, y1)


def build_opt_circuit(n_p: int, *, prec: Precision = Precision(),
                      compute_only: bool = False) -> km.Circuit:
    """The reference circuit with three coefficient registers instead of four."""
    if n_p < 1:
        raise ValueError(f'n_p must be positive, got {n_p}.')
    width = 2 * n_p + 2
    out_bits = prec.output_bits(n_p)
    builder = km.CircuitBuilder()

    def alloc(length: int, name: str) -> km.array:
        if compute_only:
            return builder.create_quantum_register(length, name=name)
        return builder.alloc_qubits(length)

    x = builder.create_quantum_register(width, name='x')
    y1, flags, selectors = alloc(out_bits, 'y1'), alloc(n_p, 'flags'), alloc(3, 'selectors')
    coefs = [alloc(prec.coef_bits, name) for name in ('a', 'b', 'c')]
    square, z = alloc(prec.square_bits, 'square'), alloc(prec.product_bits + 2, 'z')
    scratch = alloc(max(prec.square_bits, prec.product_bits + 2, prec.result_bits + 1), 'scratch')
    registers = (x, flags, selectors, coefs, square, z, y1, scratch)
    _compute_y1_opt(builder, False, prec, *registers)
    if not compute_only:
        _phase_by_bits(builder, y1)
        _compute_y1_opt(builder, True, prec, *registers)
        for reg in (y1, flags, selectors, *coefs, square, z, scratch):
            builder.free(reg)
    return builder.finish_circuit()


def main() -> None:
    import argparse
    import math

    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', help='Path of the .kmb (or .kmx) file to write.')
    parser.add_argument('--n-p', type=int, default=10)
    parser.add_argument('--format', choices=('kmb', 'kmx'), default='kmb')
    parser.add_argument('--delta-bits', type=int, default=None)
    parser.add_argument('--coef-bits', type=int, default=None)
    parser.add_argument('--square-bits', type=int, default=None)
    parser.add_argument('--product-bits', type=int, default=None)
    parser.add_argument('--result-bits', type=int, default=None)
    parser.add_argument('--skip-error', action='store_true',
                        help='skip the full-domain relative-error scan (slow)')
    args = parser.parse_args()

    prec = Precision(**{k: v for k, v in {
        'delta_bits': args.delta_bits, 'coef_bits': args.coef_bits,
        'square_bits': args.square_bits, 'product_bits': args.product_bits,
        'result_bits': args.result_bits}.items() if v is not None} or {})
    if not args.skip_error:
        max_x = 3 * (2**args.n_p - 2)**2
        err = relative_error(args.n_p, prec, np.arange(1, max_x + 1))
        print(f'Model: max relative error 2^{math.log2(err.max()):.2f} (tolerance 2^-24) '
              f'ratio={err.max() / 2.0**-24:.4f}')
    circuit = build_opt_circuit(args.n_p, prec=prec)
    circuit.to_file(args.output, format=args.format)
    stats = circuit.operation_statistics()
    print(f'Wrote {args.output}: {stats.num_toffoli_like:,} Toffolis, {circuit.num_qubits} qubits.')


if __name__ == '__main__':
    main()

"""E4: the last two profitable qubit/Toffoli trades on top of E3's optimal invsqrt circuit.

E3 landed on `Q = 239, T = 7327` (`12/17/28/27/28`).  E4's survey of the remaining levers
(`notes/E4_INVSQRT_QUBIT.md`) left exactly two moves whose Toffoli price is below the
`C / (Q - 1) ~= 32` exchange rate:

1. **Two selector qubits instead of three.**  `selectors[2] == half == top & third` is only
   ever consumed as the weight-1 control of the first `delta` row of each Horner step, and the
   bits it is built from (`x[-1]`, `x[-3]`) are livelier than it is.  Computing it into the
   spare top of the product scratch right before the step and clearing it with `del_and`
   afterwards costs one `init_and` per step and no ancilla at all: 6 Toffolis over the whole
   circuit for one persistent qubit.

2. **The `correct` constant through the scratch.**  `iadd` charges `L - 1` ancillas for a
   classical constant offset but only `L - 2` for an `L`-bit quantum slice, and both cost
   `L - 1` Toffolis.  Feeding `3 2^27 + 1` in as a `bit_store`-populated scratch slice
   therefore removes the widest single transient in the circuit for one extra Toffoli per
   pass.  Only the offset's *representation* changes, so this is exactly the reference's
   `correct` op for op.

Both changes are bit-identical arithmetic, so the accuracy model is untouched.

Usage
-----
    ../2609-xingzhi-drug/.venv/bin/python src/e4/invsqrt_opt3_circuit.py build/e4/invsqrt_opt3.kmb
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'invsqrt'))

import kickmix as km  # noqa: E402

from nr_reference_circuit import (  # noqa: E402
    Precision,
    _add_row,
    _and,
    _denormalize,
    _multiply_m,
    _multiply_y0,
    _normalize,
    _phase_by_bits,
    _square,
    relative_error,
)
import invsqrt_opt_circuit as O  # noqa: E402

BEST = dict(delta_bits=12, coef_bits=17, square_bits=28, product_bits=27, result_bits=28)


def _select_interval2(builder: km.CircuitBuilder, inverse: bool, x: km.array,
                      selectors: km.array) -> None:
    """`_select_interval` without the `half` selector, which is formed per Horner step."""
    width = len(x)
    top, second, third = x[width - 1], x[width - 2], x[width - 3]
    builder.x(top)
    _and(builder, inverse, top, third, selectors[0])
    builder.x(top)
    _and(builder, inverse, top, second, selectors[1])


def _horner3_half(builder: km.CircuitBuilder, inverse: bool, prec: Precision, x: km.array,
                  selectors: km.array, a: km.array, b: km.array, c: km.array,
                  scratch: km.array) -> None:
    """E3's three-register Horner with the `(top & third)` selector formed on the fly.

    `scratch[coef_bits:]` is idle throughout Horner (`_add_row` only ever touches
    `scratch[:len(acc)]` and the product slots below that), so the top scratch bit carries
    the extra control instead of a third persistent selector qubit.
    """
    width, wc = len(x), len(a)
    half = scratch[-1]
    top, third = x[width - 1], x[width - 3]
    delta = [(half, 1)]
    delta += [(x[width - 2 - m], m) for m in range(2, min(prec.delta_bits, width - 2) + 1)]

    def step(acc: km.array, mult: km.array) -> None:
        builder.init_and(top, third, half)
        for control, m in delta:
            _add_row(builder, inverse, control, [(mult[i], i - m) for i in range(m, wc)],
                     acc, scratch, subtract=True)
        builder.del_and(top, third, half)

    def load(reg: km.array, j: int) -> None:
        O._load_coef(builder, prec, x, selectors, reg, j)

    if inverse:
        step(a, b)
        step(b, c)
        load(b, 1)
        load(b, 3)
        step(c, b)
        load(b, 3)
        load(c, 2)
        load(a, 0)
    else:
        load(a, 0)
        load(c, 2)
        load(b, 3)
        step(c, b)          # c := c2 - T(c3 delta)  = r1
        load(b, 3)          # b ^= c3 -> 0
        load(b, 1)          # b := c1
        step(b, c)          # b := c1 - T(r1 delta) = r2
        step(a, b)          # a := c0 - T(r2 delta) = y0


def _correct_slice(builder: km.CircuitBuilder, inverse: bool, z: km.array,
                   scratch: km.array) -> None:
    """`z := 3 - z` exactly as the reference, but with the constant staging through `scratch`.

    The reference form `iadd((3 << (len(z) - 2)) + 1, target=z)` is a classical-constant
    offset, which the adder services with `len(z) - 1` ancillas.  The same value supplied as
    an `len(z)`-bit quantum slice needs `len(z) - 2`.  Same Toffolis, one ancilla fewer.
    """
    constant = (3 << (len(z) - 2)) + 1
    carrier = scratch[:len(z)]
    bits = [i for i in range(len(z)) if (constant >> i) & 1]
    if inverse:
        # Literal inverse of the forward sequence below.
        builder.x(km.array([carrier[i] for i in bits]))
        builder.isub(carrier, target=z)
        builder.x(km.array([carrier[i] for i in bits]))
        builder.x(z)
    else:
        builder.x(z)
        builder.x(km.array([carrier[i] for i in bits]))
        builder.iadd(carrier, target=z)
        builder.x(km.array([carrier[i] for i in bits]))


def build_opt3_circuit(n_p: int, *, prec: Precision = Precision(),
                       compute_only: bool = False) -> km.Circuit:
    """E3's `build_opt_circuit` with two selectors and the `correct` scratch carrier."""
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
    y1, flags, selectors = alloc(out_bits, 'y1'), alloc(n_p, 'flags'), alloc(2, 'selectors')
    coefs = [alloc(prec.coef_bits, name) for name in ('a', 'b', 'c')]
    square, z = alloc(prec.square_bits, 'square'), alloc(prec.product_bits + 2, 'z')
    scratch = alloc(max(prec.square_bits, prec.product_bits + 2, prec.result_bits + 1), 'scratch')

    def half_of_y1(inverse: bool) -> None:
        y0, p = coefs[0], y1[:len(y1) - len(flags)]
        if inverse:
            with builder.mark('denormalize'):
                _denormalize(builder, inverse, flags, y1)
            with builder.mark('multiply_y0'):
                _multiply_y0(builder, inverse, y0, z, p, scratch)
            with builder.mark('correct'):
                _correct_slice(builder, inverse, z, scratch)
            with builder.mark('multiply_m'):
                _multiply_m(builder, inverse, x, square, z, scratch)
            with builder.mark('square'):
                _square(builder, inverse, y0, square, scratch)
            with builder.mark('horner'):
                _horner3_half(builder, inverse, prec, x, selectors, *coefs, scratch)
            with builder.mark('select_interval'):
                _select_interval2(builder, inverse, x, selectors)
            with builder.mark('normalize'):
                _normalize(builder, inverse, x, flags)
        else:
            with builder.mark('normalize'):
                _normalize(builder, inverse, x, flags)
            with builder.mark('select_interval'):
                _select_interval2(builder, inverse, x, selectors)
            with builder.mark('horner'):
                _horner3_half(builder, inverse, prec, x, selectors, *coefs, scratch)
            with builder.mark('square'):
                _square(builder, inverse, y0, square, scratch)
            with builder.mark('multiply_m'):
                _multiply_m(builder, inverse, x, square, z, scratch)
            with builder.mark('correct'):
                _correct_slice(builder, inverse, z, scratch)
            with builder.mark('multiply_y0'):
                _multiply_y0(builder, inverse, y0, z, p, scratch)
            with builder.mark('denormalize'):
                _denormalize(builder, inverse, flags, y1)

    half_of_y1(False)
    if not compute_only:
        _phase_by_bits(builder, y1)
        half_of_y1(True)
        for reg in (y1, flags, selectors, *coefs, square, z, scratch):
            builder.free(reg)
    return builder.finish_circuit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output')
    ap.add_argument('--n-p', type=int, default=10)
    ap.add_argument('--format', choices=('kmb', 'kmx'), default='kmb')
    for name in ('delta_bits', 'coef_bits', 'square_bits', 'product_bits', 'result_bits'):
        ap.add_argument(f'--{name.replace("_", "-")}', type=int, default=None)
    ap.add_argument('--skip-error', action='store_true')
    args = ap.parse_args()

    prec = Precision(**{k: v for k, v in {
        'delta_bits': args.delta_bits, 'coef_bits': args.coef_bits,
        'square_bits': args.square_bits, 'product_bits': args.product_bits,
        'result_bits': args.result_bits}.items() if v is not None} or {})
    if not args.skip_error:
        max_x = 3 * (2**args.n_p - 2)**2
        err = relative_error(args.n_p, prec, np.arange(1, max_x + 1))
        print(f'Model: max relative error 2^{math.log2(err.max()):.2f} (tolerance 2^-24) '
              f'ratio={err.max() / 2.0**-24:.4f}')
    circuit = build_opt3_circuit(args.n_p, prec=prec)
    circuit.to_file(args.output, format=args.format)
    stats = circuit.operation_statistics()
    print(f'Wrote {args.output}: Q={circuit.num_qubits} T={stats.num_toffoli_like:,} '
          f'C={stats.num_toffoli_like + stats.num_t_like / 4:.1f} '
          f'score={circuit.num_qubits * (stats.num_toffoli_like + stats.num_t_like / 4):,.0f}')


if __name__ == '__main__':
    main()

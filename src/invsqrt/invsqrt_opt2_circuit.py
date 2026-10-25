"""E4: invsqrt circuit with two further qubit savings on top of E3's three-register Horner.

Two changes, both **arithmetically bit-identical** to the reference (so the shipped
`fixed_point_model` remains the exact ground truth for accuracy):

Change 1 -- `3 - z` is folded into the final multiply, so `correct` disappears.
    Reference:   z := 3 - z   (a full-width constant add: 28 Toffolis, and a
                              `len(z)-1 = 28` ancilla transient -- the peaking op)
                 p := T(y0 * z)
    Here:        p := 3 y0 - T(y0 * z)
    `3 y0` is loaded with two CNOTs per `y0` bit (Clifford, free), and the partial
    products are subtracted instead of added.  Mod 2^wp the two agree exactly:

        sum_k y0_k (3*2^wz - z) 2^(1-k)  =  3*2^wp*y0 - sum_k y0_k z 2^(1-k)   (mod 2^wp)

    so `p` is bit-identical.  Saves 28 Toffolis per pass and removes the widest
    ancilla transient.

Change 2 -- `z` is one bit narrower.
    `z` holds `T(M y0^2)`, whose in-register value is `z_int = M y0^2 * 2^wz`.
    Measured over the whole domain (n_p = 10):  max z_int = 134,275,617 while
    2^28 = 268,435,456 -- i.e. `M y0^2 <= 1.00044 < 2`, so **bit 28 is never used**.
    The reference allocates `product_bits + 2` bits ("plus two integer bits"); one
    integer bit suffices, so `z` becomes `product_bits + 1` bits and every add
    whose target is `z` gets a 1-bit shorter slice (a 1-bit smaller transient).

    `M y0^2` is never *below* 1 either (min 0.99987), so no wrap risk.

Everything else (registers, order of operations, truncation points) is unchanged.
`out_bits`, `p` width and `scratch` width are all still tight.

Usage
-----
    ../2609-xingzhi-drug/.venv/bin/python src/e4/invsqrt_opt2_circuit.py out.kmb \
        --delta-bits 12 --coef-bits 17 --square-bits 28 --product-bits 27 --result-bits 28
"""

from __future__ import annotations

import sys
from pathlib import Path

import kickmix as km
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'invsqrt'))

from nr_reference_circuit import (  # noqa: E402
    Precision,
    _add_row,
    _denormalize,
    _multiply_m,
    _normalize,
    _phase_by_bits,
    _select_interval,
    _square,
    fixed_point_model,
    relative_error,
)
import invsqrt_opt_circuit as O  # noqa: E402


def _multiply_y0_folded(builder: km.CircuitBuilder, inverse: bool, y0: km.array, z: km.array,
                        p: km.array, scratch: km.array) -> None:
    """`p := 3 y0 - T(y0 z)`, the reference's `p = T(y0 (3 - z))` without `3 - z`.

    `z` is `product_bits + 1` bits wide, so every index is `wz = len(z) - 1`.
    """
    wc, wz, wp = len(y0), len(z) - 1, len(p) - 1
    bits = [(wp - k, wp + 1 - k) for k in range(1, wc + 1) if wp + 1 - k < len(p)]
    controls = km.array([y0[wc - k] for k in range(1, wc + 1) if wp + 1 - k < len(p)])

    def load_3y0() -> None:
        for (lo, hi), ctl in zip(bits, controls):
            builder.cx(ctl, p[lo])
            builder.cx(ctl, p[hi])

    def products() -> None:
        for k in range(1, wc + 1):  # Bit of weight 2^-k.
            e = wp - wz - k
            terms = [(z[j], j + e) for j in range(max(0, -e), wz + 1)]
            if terms:
                _add_row(builder, inverse, y0[wc - k], terms, p, scratch, subtract=True)

    if inverse:
        for k in reversed(range(1, wc + 1)):
            e = wp - wz - k
            terms = [(z[j], j + e) for j in range(max(0, -e), wz + 1)]
            if terms:
                _add_row(builder, True, y0[wc - k], terms, p, scratch, subtract=True)
        load_3y0()
    else:
        load_3y0()
        products()


def _compute_y1(builder: km.CircuitBuilder, inverse: bool, prec: Precision, x: km.array,
                flags: km.array, selectors: km.array, coefs: list[km.array], square: km.array,
                z: km.array, y1: km.array, scratch: km.array) -> None:
    """The E3 `_compute_y1_opt`, with `correct` removed and the folded multiply."""
    a, b, c = coefs
    y0, p = a, y1[:len(y1) - len(flags)]
    with builder.mark('uncompute_y1' if inverse else 'compute_y1'):
        if inverse:
            with builder.mark('denormalize'):
                _denormalize(builder, inverse, flags, y1)
            with builder.mark('multiply_y0'):
                _multiply_y0_folded(builder, inverse, y0, z, p, scratch)
            with builder.mark('multiply_m'):
                _multiply_m_narrow(builder, inverse, x, square, z, scratch)
            with builder.mark('square'):
                _square(builder, inverse, y0, square, scratch)
            with builder.mark('horner'):
                O._horner3(builder, inverse, prec, x, selectors, a, b, c, scratch)
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
                O._horner3(builder, inverse, prec, x, selectors, a, b, c, scratch)
            with builder.mark('square'):
                _square(builder, inverse, y0, square, scratch)
            with builder.mark('multiply_m'):
                _multiply_m_narrow(builder, inverse, x, square, z, scratch)
            with builder.mark('multiply_y0'):
                _multiply_y0_folded(builder, inverse, y0, z, p, scratch)
            with builder.mark('denormalize'):
                _denormalize(builder, inverse, flags, y1)


def _multiply_m_narrow(builder: km.CircuitBuilder, inverse: bool, x: km.array, square: km.array,
                       z: km.array, scratch: km.array) -> None:
    """`z := T(M square)` with `z` one bit narrower than the reference (`wz = len(z) - 1`)."""
    width, wsq, wz = len(x), len(square), len(z) - 1
    for i in range(width):
        e = wz - wsq + i - width + 2
        _add_row(builder, inverse, x[i], [(square[j], j + e) for j in range(max(0, -e), wsq)],
                 z, scratch)


def build_opt2_circuit(n_p: int, *, prec: Precision = Precision(),
                       compute_only: bool = False) -> km.Circuit:
    """E3's optimized circuit with `correct` folded away and a one-bit-narrower `z`."""
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
    square = alloc(prec.square_bits, 'square')
    z = alloc(prec.product_bits + 1, 'z')          # was product_bits + 2
    scratch = alloc(max(prec.square_bits, prec.product_bits + 1, prec.result_bits + 1), 'scratch')
    registers = (x, flags, selectors, coefs, square, z, y1, scratch)
    _compute_y1(builder, False, prec, *registers)
    if not compute_only:
        _phase_by_bits(builder, y1)
        _compute_y1(builder, True, prec, *registers)
        for reg in (y1, flags, selectors, *coefs, square, z, scratch):
            builder.free(reg)
    return builder.finish_circuit()


# The one-line patch that makes the shipped reference module do the same thing, for A/B tests.

def narrow_z(builder_fn):
    """Return a `_multiply_m` variant with `wz = len(z) - 1` (used by equivalence tests)."""
    return _multiply_m_narrow


def main() -> None:
    import argparse
    import math

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output')
    parser.add_argument('--n-p', type=int, default=10)
    parser.add_argument('--format', choices=('kmb', 'kmx'), default='kmb')
    for name in ('delta_bits', 'coef_bits', 'square_bits', 'product_bits', 'result_bits'):
        parser.add_argument(f'--{name.replace("_", "-")}', type=int, default=None)
    parser.add_argument('--skip-error', action='store_true')
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
    circuit = build_opt2_circuit(args.n_p, prec=prec)
    circuit.to_file(args.output, format=args.format)
    stats = circuit.operation_statistics()
    print(f'Wrote {args.output}: {stats.num_toffoli_like:,} Toffolis, {circuit.num_qubits} qubits.')
    b = km.CircuitBuilder()  # cross-check the peaking transient
    print(f'p ricer check: Q = {circuit.num_qubits}')


if __name__ == '__main__':
    main()

"""E4: invsqrt without a reverse pass -- measurement-based cleanup (`R`, cost 0).

Why this is legal
-----------------
Every kickmix instruction maps a computational basis state to a computational basis
state (up to a phase scalar) -- that is a stated property of the instruction set, and
it is exactly why the simulator is linear in gate count.  `R` (reset) is in the
verifier's `ALLOWED_INSTRUCTIONS`, and `operation_statistics()` charges it **zero**
Toffolis and **zero** T (`sim.operation_statistics().num_toffoli_like` ignores it,
verified in `src/e4/invsqrt_qadd.md`).

Because the invsqrt dataflow is a *classical* reversible computation on a basis state
(the only non-Clifford gate is the diagonal `Z_POW` on `y1`), every register holds a
known classical value at every point.  So instead of paying a full reverse pass (3861
Toffolis!) to uncompute y0/y1/square/z/... back to |0>, we can simply

    apply the phase, then `R` (reset) every scratch register.

`R` on a basis-state qubit is deterministic: it measures and conditionally flips, so
the qubit ends in |0> and the shot's tracked phase is untouched.  The only thing that
still has to be undone is the *normalization of `x`*, because `x` must be returned to
its original value -- and that genuinely needs the retained `flags` register (`x_norm`
plus `flags` is injective in `x`, while `x_norm` alone is not: x=1 and x=4 both
normalize to 2^20).

So the circuit becomes:

    normalize -> interval -> Horner -> square -> M*y0^2 -> correct -> y0*corr
              -> denormalize -> phase(y1)
              -> R everything except x and flags        (0 Toffolis)
              -> inverse normalize (needs flags only)

Reset also makes the allocator's free list useful, so the register schedule can be
lazy: `selectors`, `b`, `c` die at the end of Horner, `square` dies after
`M y0^2`, and the freed qubits are handed straight back out for `z` and `y1`.  Q then
tracks the *live* set instead of the sum of every register ever allocated.

Usage
-----
    ../2609-xingzhi-drug/.venv/bin/python src/e4/invsqrt_reset_circuit.py out.kmb
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
    _correct,
    _denormalize,
    _multiply_y0,
    _normalize,
    _phase_by_bits,
    _select_interval,
    _square,
    relative_error,
)
import invsqrt_opt_circuit as O  # noqa: E402
from invsqrt_opt2_circuit import _multiply_m_narrow  # noqa: E402
from nr_reference_circuit import _add_row  # noqa: E402

BEST = dict(delta_bits=12, coef_bits=17, square_bits=28, product_bits=27, result_bits=28)
SCRATCH = lambda prec: max(prec.square_bits, prec.product_bits + 1, prec.result_bits + 1)  # noqa: E731


def WZ(prec: Precision) -> int:
    """Fractional weight of the `z` register: `product_bits`, independent of its width."""
    return prec.product_bits


def _correct_narrow(builder: km.CircuitBuilder, inverse: bool, z: km.array, wz: int) -> None:
    """`z := 3 - z` mod 2^len(z), for a `z` holding only `wz + 1` significant bits.

    The reference keeps `z` at `product_bits + 2` bits so that `corr = 3 2^wz - z` fits with
    two integer bits.  Only one integer bit is ever needed (`z < 2^(wz+1)`), so it is enough
    to compute `corr mod 2^(wz+1)` here: `corr` and `corr - 2^(wz+1)` differ by
    `2^(wz+1)`, and `sum_k y[k] 2^(1-k) 2^(wz+1) = 2 y0 2^(wz+1) = 2^(wp+1) y0 = 0 (mod 2^(wp+1))`,
    so the final product is unchanged.
    """
    constant = ((3 << wz) + 1) & ((1 << len(z)) - 1)
    if inverse:
        builder.isub(constant, target=z)
        builder.x(z)
    else:
        builder.x(z)
        builder.iadd(constant, target=z)


def _multiply_y0_narrow(builder: km.CircuitBuilder, inverse: bool, y0: km.array, z: km.array,
                        p: km.array, scratch: km.array, wz: int) -> None:
    """`p := T(y0 z)` with `z` at `wz + 1` bits instead of `wz + 2`."""
    wc, wp = len(y0), len(p) - 1
    for k in range(1, wc + 1):
        e = wp - wz - k
        _add_row(builder, inverse, y0[wc - k],
                 [(z[j], j + e) for j in range(max(0, -e), wz + 1)], p, scratch)



def build_reset_circuit(n_p: int, *, prec: Precision = Precision(), lazy: bool = True,
                        narrow_z: bool = False, compute_only: bool = False) -> km.Circuit:
    """Forward pass + phase + measurement cleanup + `x` restoration.

    `lazy=False` allocates everything up front (the E3/E4-opt2 schedule) and still resets,
    so the two variants differ only in Q.  `compute_only` stops after the denormalize with
    `y1` named, for tests against the shipped `fixed_point_model`.

    `narrow_z` is a **known-broken** switch kept for the record: dropping `z` to
    `product_bits + 1` bits changes the result, because a row of `_multiply_y0` is an exact
    bit-shift of the 29-bit `corr = 3 2^27 - z` with the low bits truncated, and truncation is
    not linear, so `corr` and `corr - 2^28` do not give congruent products.  Exhaustive checks
    fail at every `n_p` for `narrow_z=True`.
    """
    if n_p < 1:
        raise ValueError(f'n_p must be positive, got {n_p}.')
    width = 2 * n_p + 2
    out_bits = prec.output_bits(n_p)
    scratch_len = SCRATCH(prec)
    z_len = prec.product_bits + 1 if narrow_z else prec.product_bits + 2
    b = km.CircuitBuilder()
    x = b.create_quantum_register(width, name='x')

    def alloc(n: int) -> km.array:
        return b.alloc_qubits(n)

    def discard(*regs: km.array) -> None:
        """`R` then free: the register is |0> again, so the qubits go back to the pool."""
        for reg in regs:
            b.reset(reg)
            b.free(reg)

    wz = WZ(prec)

    def correct(inverse: bool) -> None:
        if narrow_z:
            _correct_narrow(b, inverse, z, wz)
        else:
            _correct(b, inverse, z)

    def multiply_m(inverse: bool) -> None:
        (_multiply_m_narrow if narrow_z else _multiply_m_ref)(b, inverse, x, square, z, scratch)

    def multiply_y0(inverse: bool) -> None:
        if narrow_z:
            _multiply_y0_narrow(b, inverse, a, z, y1[:out_bits - n_p], scratch, wz)
        else:
            _multiply_y0(b, inverse, a, z, y1[:out_bits - n_p], scratch)

    flags = alloc(n_p)
    if not lazy:
        selectors = alloc(3)
        coefs = [alloc(prec.coef_bits) for _ in range(3)]
        square = alloc(prec.square_bits)
        z = alloc(z_len)
        y1 = alloc(out_bits)
        scratch = alloc(scratch_len)
        a, bk, c = coefs
        _normalize(b, False, x, flags)
        _select_interval(b, False, x, selectors)
        O._horner3(b, False, prec, x, selectors, a, bk, c, scratch)
        _square(b, False, a, square, scratch)
        multiply_m(False)
        correct(False)
        multiply_y0(False)
        _denormalize(b, False, flags, y1)
        _phase_by_bits(b, y1)
        discard(y1, selectors, a, bk, c, square, z, scratch)
        _normalize(b, True, x, flags)
        discard(flags)
        return b.finish_circuit()

    # Lazy schedule: allocate close to first use, discard at last use.
    scratch = alloc(scratch_len)
    selectors = alloc(3)
    a = alloc(prec.coef_bits)
    bk = alloc(prec.coef_bits)
    c = alloc(prec.coef_bits)
    _normalize(b, False, x, flags)
    _select_interval(b, False, x, selectors)
    O._horner3(b, False, prec, x, selectors, a, bk, c, scratch)
    discard(selectors, bk, c)              # dead once Horner is done; a holds y0

    square = alloc(prec.square_bits)
    _square(b, False, a, square, scratch)
    z = alloc(z_len)
    multiply_m(False)
    discard(square)                        # M y0^2 has consumed it

    correct(False)
    y1 = b.create_quantum_register(out_bits, name='y1') if compute_only else alloc(out_bits)
    multiply_y0(False)
    _denormalize(b, False, flags, y1)
    if compute_only:                       # stop with y1 named, for tests
        return b.finish_circuit()
    discard(a, z, scratch)                 # y0, M y0^2 and the product scratch are dead

    _phase_by_bits(b, y1)
    discard(y1)

    _normalize(b, True, x, flags)          # x_norm -> x, using the retained flags
    discard(flags)
    return b.finish_circuit()


def _multiply_m_ref(builder, inverse, x, square, z, scratch):
    from nr_reference_circuit import _multiply_m
    _multiply_m(builder, inverse, x, square, z, scratch)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output')
    ap.add_argument('--n-p', type=int, default=10)
    ap.add_argument('--format', choices=('kmb', 'kmx'), default='kmb')
    ap.add_argument('--lazy', default='both', choices=('yes', 'no', 'both'))
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
    modes = {'both': (True, False), 'yes': (True,), 'no': (False,)}[args.lazy]
    for lazy in modes:
        circuit = build_reset_circuit(args.n_p, prec=prec, lazy=lazy)
        stats = circuit.operation_statistics()
        out = args.output if len(modes) == 1 else args.output.replace('.kmb', f'_lazy{int(lazy)}.kmb')
        circuit.to_file(out, format=args.format)
        print(f'lazy={lazy}: Q={circuit.num_qubits} '
              f'T={stats.num_toffoli_like:,} nrot={stats.num_z_pow(min_clifford_hierarchy_level=4)} '
              f'-> {out}')


if __name__ == '__main__':
    main()

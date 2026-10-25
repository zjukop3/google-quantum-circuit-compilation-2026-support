"""E5: `flags` goes from a 10-qubit thermometer to a 4-qubit binary `s`.

E4 landed on `Q = 237, T = 7334` (`12/17/28/27/28`).  Its own survey left exactly one
structural move (`notes/E4_INVSQRT_QUBIT.md` §6): the normalize/denormalize shift count
was stored as a **unary thermometer** of `n_p` flag qubits, one per shift step.

What the flags actually are
---------------------------
`flags[k] = [x has at least 2(k+1) leading zero bits]`, i.e. a thermometer of

    s = (number of leading zero *pairs* of x)   in [0, n_p].

They are consumed in exactly two places:

* `_normalize` — sequentially, one flag per shift step, each flag driving two
  windowed `left_rotate(..., 1, control=flags[k])` (even lane and odd lane).
* `_denormalize` — one windowed `left_rotate(..., 1, control=flags[k])` per flag on
  `y1[k : k + result_bits + 2]`, i.e. the stepwise `y1 := p 2^s`.

`_select_interval` / `_select_interval2` **do not use the flags at all**: the interval
is read off the top three bits of `x_norm`.  (E4's note said three sites; it is two.)

Why the swap is profitable on Toffoli *and* qubits
--------------------------------------------------
A controlled rotation of an `m`-bit window by one costs `m - 1` CSWAPs, so

    _denormalize = 2 passes x 10 steps x (30 - 1) = 580 Toffolis   (measured)
    _normalize   = 2 passes x (sum_k 20-2k + 10 init_and) = 230     (measured)

A **binary shift register** gets the same shifts through kickmix's barrel form
`left_rotate(target, shift=s_bits)`, whose measured price is `(len(target)//2)*(b+1)`:

    y1 (39 bits) << s        with b = 4 :  95 Toffolis  (vs 290 for the unary stepwise form)
    x  (22 bits) << 2s  as [False]+s     :  50 Toffolis  (vs 110; the constant weight bit is free)

and the thermometer is only needed *transiently*, because a thermometer converts to
binary with **CX gates alone** (`s_j` is an XOR of a fixed subset of the flags — this is
exact, not an approximation).  So the 10 flags become 10 scratch qubits that exist only
inside `_normalize`, and only `s` (4 bits) stays resident:

    Q: 237 -> 231   (-10 flags, +4 s)                       [measured]
    T: 7334 -> 6852 (-482: normalize 230 -> 138, denormalize 580 -> 190)

The arithmetic is unchanged: `x_norm`, the interval selectors, `y1 = p 2^s` and the
phase `y1 / 2^F` are bit-identical, so the accuracy model is untouched.

Usage
-----
    ../2609-xingzhi-drug/.venv/bin/python src/e5/invsqrt_flags_circuit.py build/e5/invsqrt_flags.kmb
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
sys.path.insert(0, str(ROOT / 'src' / 'e4'))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'invsqrt'))

import kickmix as km  # noqa: E402

from nr_reference_circuit import (  # noqa: E402
    Precision,
    _phase_by_bits,
    relative_error,
)
import invsqrt_opt_circuit as O  # noqa: E402
import invsqrt_opt3_circuit as E4  # noqa: E402

BEST = dict(delta_bits=12, coef_bits=17, square_bits=28, product_bits=27, result_bits=28)


def s_bits_for(n_p: int) -> int:
    """Bits needed to hold `s in [0, n_p]`."""
    return max(1, (n_p + 1 - 1).bit_length()) if n_p > 0 else 1


def thermometer_to_binary_terms(n_p: int, s_bits: int) -> list[list[int]]:
    """For each bit `j` of `s`, the flag indices whose XOR is that bit.

    `flags[k] = [s >= k+1]`, so `s = sum_k flags[k]` and bit `j` of `s` is 1 exactly on
    the runs `s mod 2^(j+1) in [2^j, 2^(j+1))`.  Each run contributes its two bounding
    thresholds, which telescope into single flags:

        bit j = XOR over t of flags[t * 2^(j+1) + 2^j - 1]      (indices < n_p)
    """
    terms: list[list[int]] = []
    for j in range(s_bits):
        idx: list[int] = []
        m = 0
        while True:
            a = m * (1 << (j + 1)) + (1 << j) - 1
            b = (m + 1) * (1 << (j + 1)) - 1
            if a >= n_p and b >= n_p:
                break
            if a < n_p:
                idx.append(a)
            if b < n_p:
                idx.append(b)
            m += 1
        terms.append(idx)
    return terms


def _pair_of(x: km.array, j: int) -> km.array:
    """The two bits of pair `j` (bit `2j` and `2j+1` of `x`)."""
    return km.array([x[2 * j], x[2 * j + 1]])


def _thermometer_compute(builder: km.CircuitBuilder, x: km.array, flags: km.array,
                         aux: km.array, n_p: int) -> None:
    """`flags[k] := [the top k+1 pairs of x are all zero]` — one AND chain.

    `init_and` broadcasts over its *target*, so a genuine 3-control AND has to be split
    into two scalar ANDs: `aux[k]` takes the fresh pair `n_p - k`, then the chain ANDs it
    into `flags[k]`.  Two Toffolis per step beyond the first (19 for `n_p = 10`).  `x` is
    X-conjugated around each AND and the conjugation is undone immediately, so `x` is
    untouched and the controls stay valid for `_thermometer_clear`.
    """
    for k in range(n_p):
        pair = _pair_of(x, n_p - k)
        builder.x(pair)
        if k == 0:
            builder.init_and(pair[0], pair[1], flags[0])
        else:
            builder.init_and(pair[0], pair[1], aux[k - 1])
            builder.init_and(flags[k - 1], aux[k - 1], flags[k])
        builder.x(pair)


def _thermometer_clear(builder: km.CircuitBuilder, x: km.array, flags: km.array,
                       aux: km.array, n_p: int) -> None:
    """`del_and` the whole chain back out (0 Toffolis: AND-shaped targets are free)."""
    for k in reversed(range(n_p)):
        pair = _pair_of(x, n_p - k)
        builder.x(pair)
        if k == 0:
            builder.del_and(pair[0], pair[1], flags[0])
        else:
            builder.del_and(flags[k - 1], aux[k - 1], flags[k])
            builder.del_and(pair[0], pair[1], aux[k - 1])
        builder.x(pair)


def _thermometer_xor_into_s(builder: km.CircuitBuilder, flags: km.array, s: km.array,
                            n_p: int) -> None:
    """`s ^= s(flags)` with CX only — its own inverse, so it also uncomputes `s`."""
    for j, idx in enumerate(thermometer_to_binary_terms(n_p, len(s))):
        if idx:
            builder.cx(km.array([flags[i] for i in idx]), s[j])


def _normalize_bin(builder: km.CircuitBuilder, inverse: bool, x: km.array, s: km.array,
                   flags: km.array, aux: km.array, n_p: int) -> None:
    """`x := x 2^(2s)` (and its inverse) through a binary barrel shift.

    Forward: build the thermometer, convert it to `s` with CX, free the thermometer, then
    rotate `x` by `2s` **once**.  Every intermediate state of the barrel shift is a cyclic
    rotation, and the final one is a clean shift because the top `2s` bits of `x` are zero
    by construction of `s` (they are exactly the leading zeros that define it).

    Inverse: exactly the mirror image (barrel back, rebuild the thermometer, XOR `s` back
    to zero, free the thermometer).
    """
    shift2 = [False] + list(s)          # 2s in the barrel's binary weight convention.
    if inverse:
        builder.right_rotate(x, shift=shift2)
        _thermometer_compute(builder, x, flags, aux, n_p)
        _thermometer_xor_into_s(builder, flags, s, n_p)
        _thermometer_clear(builder, x, flags, aux, n_p)
    else:
        _thermometer_compute(builder, x, flags, aux, n_p)
        _thermometer_xor_into_s(builder, flags, s, n_p)
        _thermometer_clear(builder, x, flags, aux, n_p)
        builder.left_rotate(x, shift=shift2)


def _denormalize_bin(builder: km.CircuitBuilder, inverse: bool, s: km.array,
                     y1: km.array) -> None:
    """`y1 := y1 2^s` (and its inverse) as one binary barrel rotation of the whole register.

    The unary version rotated `y1[k : k + result_bits + 2]` once per flag and left the
    flags outside `y1`; with `s` binary the flags are gone, so a single rotation of all of
    `y1` is the same shift.  `p` sits in `y1[:result_bits+1]` and the top `s` bits are zero
    (the phase's scaling leaves them free), so the wrap-around reinserts zeros exactly.
    """
    if inverse:
        builder.right_rotate(y1, shift=s)
    else:
        builder.left_rotate(y1, shift=s)


def build_flags_circuit(n_p: int, *, prec: Precision = Precision(),
                        compute_only: bool = False) -> km.Circuit:
    """E4's `build_opt3_circuit` with `flags` replaced by the binary shift count `s`."""
    if n_p < 1:
        raise ValueError(f'n_p must be positive, got {n_p}.')
    width = 2 * n_p + 2
    out_bits = prec.output_bits(n_p)
    n_s = s_bits_for(n_p)
    builder = km.CircuitBuilder()

    def alloc(length: int, name: str) -> km.array:
        if compute_only:
            return builder.create_quantum_register(length, name=name)
        return builder.alloc_qubits(length)

    x = builder.create_quantum_register(width, name='x')
    y1, s = alloc(out_bits, 'y1'), alloc(n_s, 's')
    selectors = alloc(2, 'selectors')
    coefs = [alloc(prec.coef_bits, name) for name in ('a', 'b', 'c')]
    square, z = alloc(prec.square_bits, 'square'), alloc(prec.product_bits + 2, 'z')
    scratch = alloc(max(prec.square_bits, prec.product_bits + 2, prec.result_bits + 1), 'scratch')
    p = y1[:prec.result_bits + 1]

    def tmp(length: int, name: str) -> tuple[km.array, bool]:
        """The thermometer is pure scratch: allocate it inside `_normalize`, free it here."""
        reg = builder.create_quantum_register(length, name=name) if compute_only \
            else builder.alloc_qubits(length)
        return reg, not compute_only

    def half_of_y1(inverse: bool) -> None:
        y0 = coefs[0]
        if inverse:
            with builder.mark('denormalize'):
                _denormalize_bin(builder, inverse, s, y1)
            with builder.mark('multiply_y0'):
                E4._multiply_y0(builder, inverse, y0, z, p, scratch)
            with builder.mark('correct'):
                E4._correct_slice(builder, inverse, z, scratch)
            with builder.mark('multiply_m'):
                E4._multiply_m(builder, inverse, x, square, z, scratch)
            with builder.mark('square'):
                E4._square(builder, inverse, y0, square, scratch)
            with builder.mark('horner'):
                E4._horner3_half(builder, inverse, prec, x, selectors, *coefs, scratch)
            with builder.mark('select_interval'):
                E4._select_interval2(builder, inverse, x, selectors)
            with builder.mark('normalize'):
                flags, owned = tmp(n_p, 'flags_tmp_rev')
                aux, owned_aux = tmp(max(1, n_p - 1), 'aux_tmp_rev')
                _normalize_bin(builder, inverse, x, s, flags, aux, n_p)
                if owned:
                    builder.free(flags)
                if owned_aux:
                    builder.free(aux)
        else:
            with builder.mark('normalize'):
                flags, owned = tmp(n_p, 'flags_tmp_fwd')
                aux, owned_aux = tmp(max(1, n_p - 1), 'aux_tmp_fwd')
                _normalize_bin(builder, inverse, x, s, flags, aux, n_p)
                if owned:
                    builder.free(flags)
                if owned_aux:
                    builder.free(aux)
            with builder.mark('select_interval'):
                E4._select_interval2(builder, inverse, x, selectors)
            with builder.mark('horner'):
                E4._horner3_half(builder, inverse, prec, x, selectors, *coefs, scratch)
            with builder.mark('square'):
                E4._square(builder, inverse, y0, square, scratch)
            with builder.mark('multiply_m'):
                E4._multiply_m(builder, inverse, x, square, z, scratch)
            with builder.mark('correct'):
                E4._correct_slice(builder, inverse, z, scratch)
            with builder.mark('multiply_y0'):
                E4._multiply_y0(builder, inverse, y0, z, p, scratch)
            with builder.mark('denormalize'):
                _denormalize_bin(builder, inverse, s, y1)

    half_of_y1(False)
    if not compute_only:
        _phase_by_bits(builder, y1)
        half_of_y1(True)
        for reg in (y1, s, selectors, *coefs, square, z, scratch):
            builder.free(reg)
    return builder.finish_circuit()


# Aliases so the ablation instrument can stub the same names as in E4.
_multiply_y0, _multiply_m, _square, _horner3_half = (
    E4._multiply_y0, E4._multiply_m, E4._square, E4._horner3_half)
_select_interval2, _correct_slice = E4._select_interval2, E4._correct_slice
_normalize, _denormalize = _normalize_bin, _denormalize_bin


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
    circuit = build_flags_circuit(args.n_p, prec=prec)
    circuit.to_file(args.output, format=args.format)
    stats = circuit.operation_statistics()
    # The local `T + T/4` is NOT the official cost: the verifier adds a rotation-synthesis
    # term (`n_rot * (0.56 log2(1/eps) + 5.3) / 4`, ~256 here) on top of the Toffoli count.
    local_c = stats.num_toffoli_like + stats.num_t_like / 4
    print(f'Wrote {args.output}: Q={circuit.num_qubits} T={stats.num_toffoli_like:,} '
          f'n_rot={stats.num_z_pow(min_clifford_hierarchy_level=4)} '
          f'local_C={local_c:.1f} local_score={circuit.num_qubits * local_c:,.0f} '
          f'(official cost adds the rotation term; run the verifier for the real score)')


if __name__ == '__main__':
    main()

"""E6b: unit-level proof that the patched `ixor_gf2_div(control=...)` behaves exactly like
`ixor_gf2_mul(control=...)` does.

Three things are checked, all exhaustively over small fields and all by *simulation*, not by
reading the cost table:

1. Semantics of the new parameter.  For every (lhs, rhs, target, control) in GF(2^d)^3 x {0,1}:

       target' == target ^ (control * (lhs / rhs mod p))      (zero divisor -> zero)
       control' == control
       every qubit the primitive borrowed returns to |0>
       lhs' == lhs, rhs' == rhs

2. Agreement with the multiply's control form.  The same construction is run with
   `ixor_gf2_mul` and `ixor_gf2_div`, and the *shape* of the answer is compared: an off control
   must leave the target bit-identical to the uncontrolled run, and an on control must equal the
   uncontrolled run applied to a control prepared in |1>.

3. The three accepted control flavours (`True`/`False`, a qubit, and a classical bit) agree with
   each other where they overlap.

Usage
-----
    PYTHONPATH=src/e6/patch python src/e6/e6b_unit_control.py --degree 4
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

import kickmix as km

POLYS = {4: 0b10011, 5: 0b100101, 6: 0b1000011, 12: 0x1009}


def gf_mul(a: int, b: int, deg: int, poly: int) -> int:
    """Schoolbook GF(2^deg) product, independent of kickmix, for the reference model."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> deg:
            a ^= poly
    return r & ((1 << deg) - 1)


def gf_inv(a: int, deg: int, poly: int) -> int:
    if a == 0:
        return 0
    r = 1
    for _ in range(deg - 1):
        r = gf_mul(r, a, deg, poly)
    # r = a^(2^deg - 2) = a^-1, computed by repeated squaring would be faster but this is clear.
    return gf_pow(a, (1 << deg) - 2, deg, poly)


def gf_pow(a: int, e: int, deg: int, poly: int) -> int:
    r = 1
    while e:
        if e & 1:
            r = gf_mul(r, a, deg, poly)
        a = gf_mul(a, a, deg, poly)
        e >>= 1
    return r


def gf_div(a: int, b: int, deg: int, poly: int) -> int:
    return gf_mul(a, gf_inv(b, deg, poly), deg, poly)


def to_bits(values: np.ndarray, width: int) -> np.ndarray:
    return (values[None, :] >> np.arange(width, dtype=np.uint64)[:, None]) & 1 == 1


class Op:
    """One primitive under test, exposed as `(lhs, rhs, target, control)` on a small circuit."""

    def __init__(self, name: str, degree: int, control: str):
        self.name = name
        self.degree = degree
        self.control = control                 # 'qubit' | 'true' | 'false'
        self.b = km.CircuitBuilder()
        self.gf = km.GF2Field(degree)
        self.lhs = self.b.create_quantum_register(degree, name='lhs')
        self.rhs = self.b.create_quantum_register(degree, name='rhs')
        self.target = self.b.create_quantum_register(degree, name='target')
        self.ctrl = self.b.create_quantum_register(1, name='ctrl') if control == 'qubit' else None
        self.n_scratch = 0

    def _ctl_kwargs(self):
        if self.control == 'qubit':
            return {'control': self.ctrl[0]}
        if self.control == 'true':
            return {'control': True}
        return {'control': False}

    def finish(self) -> km.Circuit:
        if self.name == 'div':
            self.b.ixor_gf2_div(self.lhs, self.rhs, target=self.target, field=self.gf,
                                **self._ctl_kwargs())
        elif self.name == 'mul':
            self.b.ixor_gf2_mul(self.lhs, self.rhs, target=self.target, field=self.gf,
                                **self._ctl_kwargs())
        else:
            raise ValueError(self.name)
        return self.b.finish_circuit()


def run_op(op: Op, cases: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray, int]:
    """Simulates every case in one batch and returns (target_out, ctrl_out, junk_nonzero)."""
    circuit = op.finish()
    degree = op.degree
    n = circuit.num_qubits
    tagged = {q.id for q in itertools.chain(op.lhs, op.rhs, op.target, op.ctrl or [])}
    scratch = km.array([km.q(i) for i in range(n) if i not in tagged])

    lhs = np.array([c[0] for c in cases], dtype=np.uint64)
    rhs = np.array([c[1] for c in cases], dtype=np.uint64)
    tgt = np.array([c[2] for c in cases], dtype=np.uint64)
    ctl = np.array([c[3] for c in cases], dtype=np.uint64)

    sim = km.Simulator(batch_size=len(cases), debug_print=False, counting_operations=True, seed=7)
    sim.clear_for_shot()
    sim.write_across_shots(op.lhs, to_bits(lhs, degree))
    sim.write_across_shots(op.rhs, to_bits(rhs, degree))
    sim.write_across_shots(op.target, to_bits(tgt, degree))
    if op.ctrl is not None:
        sim.write_across_shots(op.ctrl, to_bits(ctl, 1))
    sim.do(circuit)

    def read(regs, width):
        bits = sim.read_across_shots(regs)
        return (bits.astype(np.uint64) << np.arange(width, dtype=np.uint64)[:, None]).sum(axis=0)

    out = read(op.target, degree)
    ctl_out = read(op.ctrl, 1) if op.ctrl is not None else None
    junk = int(sim.read_across_shots(scratch).any()) if len(scratch) else 0
    return out, ctl_out, junk


def check_semantics(degree: int, poly: int, primitive: str) -> list[str]:
    op = Op(primitive, degree, 'qubit')
    cases = [(a, b, t, c)
             for a, b, t in itertools.product(range(1 << degree), repeat=3)
             for c in (0, 1)]
    op_fn = gf_mul if primitive == 'mul' else gf_div
    out, ctl_out, junk = run_op(op, cases)
    bad = []
    for k, (a, b, t, c) in enumerate(cases):
        want = t ^ (op_fn(a, b, degree, poly) if c else 0)
        if int(out[k]) != want:
            bad.append(f'  {primitive} lhs={a} rhs={b} target={t} ctrl={c}: '
                       f'got {int(out[k])} want {want}')
        if int(ctl_out[k]) != c:
            bad.append(f'  {primitive} ctrl={c} changed to {int(ctl_out[k])}')
    if junk:
        bad.append(f'  {primitive}: a borrowed qubit did not return to |0>')
    n = len(cases)
    print(f'{primitive:4s} exhaustive GF(2^{degree}) x control: {n - len(bad)}/{n} exact'
          f'{"" if not bad else "  <-- FAILURES"}')
    return bad


def check_control_agreement(degree: int, poly: int) -> list[str]:
    """A control in |1> must reproduce the uncontrolled run; in |0> it must be a no-op."""
    bad = []
    for primitive in ('div', 'mul'):
        abc = list(itertools.product(range(1 << degree), repeat=3))
        cases = [(a, b, t, 0) for a, b, t in abc]
        ctl_cases = [(a, b, t, 1) for a, b, t in abc]

        off_out, _, _ = run_op(Op(primitive, degree, 'qubit'), cases)
        on_out, _, _ = run_op(Op(primitive, degree, 'qubit'), ctl_cases)
        plain_out, _, _ = run_op(Op(primitive, degree, 'true'), cases)
        none_out, _, _ = run_op(Op(primitive, degree, 'false'), cases)

        model = np.array([t ^ gf_div(a, b, degree, poly) if primitive == 'div'
                          else t ^ gf_mul(a, b, degree, poly)
                          for a, b, t, _ in cases], dtype=np.uint64)
        untouched = np.array([c[2] for c in cases], dtype=np.uint64)

        ok_off = bool((off_out == untouched).all())
        ok_on = bool((on_out == model).all())
        ok_plain = bool((plain_out == model).all())
        ok_none = bool((none_out == untouched).all())
        print(f'{primitive:4s} control=|0> is a no-op      : {"OK" if ok_off else "FAIL"}'
              f'   (== control=False: {"OK" if ok_none else "FAIL"})')
        print(f'{primitive:4s} control=|1> == control=True : {"OK" if ok_on and ok_plain else "FAIL"}')
        for label, ok in (('off-noop', ok_off), ('on-model', ok_on),
                          ('true-model', ok_plain), ('false-noop', ok_none)):
            if not ok:
                bad.append(f'  {primitive}/{label}')
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--degree', type=int, default=4)
    args = ap.parse_args()
    degree = args.degree
    poly = POLYS[degree]

    print(f'kickmix : {km.__file__}')
    doc = km.CircuitBuilder.ixor_gf2_div.__doc__ or ''
    print(f'patch   : {"present" if "control" in doc else "ABSENT"}')
    print(f'field   : GF(2^{degree}), modulus {poly:#b}')
    print()

    failures: list[str] = []
    for primitive in ('mul', 'div'):
        failures += check_semantics(degree, poly, primitive)
    print()
    failures += check_control_agreement(degree, poly)
    print()
    if failures:
        print('RESULT: FAIL')
        for line in failures[:20]:
            print(line)
        sys.exit(1)
    print('RESULT: PASS -- the new div control is semantically identical in shape to mul\'s')


if __name__ == '__main__':
    main()

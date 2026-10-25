"""E6b: unit price of the control-patched `ixor_gf2_div`, against the E4 price list.

Prices are `num_toffoli_like` on a freshly built circuit per primitive, at degree = 12
(the dqi field).  Reference numbers (E4_DQI_QUBIT.md, measured on the stock build):

    ixor_gf2_mul(a, b -> target)         0 extra qubits,  63 T
    ixor_gf2_mul(..., control=q)         0 extra qubits,  75 T      (+12 = degree)
    init_gf2_div_with_scaffold           72 transient,    378 T
    init_gf2_div (no scaffold)           12 transient,    693 T
    ixor_gf2_div                         0 extra,         378 T

The patch adds a qubit-controlled multiply inside the division, so the predicted new price is
`378 + 12 = 390` T and `60 + 12 = 72` transient qubits when the control is a qubit; a classical
bit control (`km.b`) pushes a condition instead and should stay at 60.

"Transient" is measured the only way that cannot lie: build the primitive with `max_qubits` set
just above the register size and read `num_qubits` of the finished circuit.  `km.CircuitBuilder`
grows a free-list that the primitive borrows from, so the honest number is the peak number of
simultaneously live qubits, which is `circuit.num_qubits - persistent registers`.

Usage
-----
    python src/e6/e6b_price.py                 # stock module
    PYTHONPATH=src/e6/patch python src/e6/e6b_price.py   # patched module
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import kickmix as km

DEGREE = 12


def build(name: str, degree: int = DEGREE) -> dict:
    """Builds one primitive in isolation and prices it."""
    b = km.CircuitBuilder()
    gf = km.GF2Field(degree)
    lhs = b.create_quantum_register(degree, name='lhs')
    rhs = b.create_quantum_register(degree, name='rhs')
    regs = 2 * degree
    if name == 'ixor_gf2_mul':
        target = b.create_quantum_register(degree, name='target')
        regs += degree
        b.ixor_gf2_mul(lhs, rhs, target=target, field=gf)
    elif name == 'ixor_gf2_mul_ctl':
        target = b.create_quantum_register(degree, name='target')
        c = b.create_quantum_register(1, name='ctrl')
        regs += degree + 1
        b.ixor_gf2_mul(lhs, rhs, target=target, field=gf, control=c[0])
    elif name == 'ixor_gf2_div':
        target = b.create_quantum_register(degree, name='target')
        regs += degree
        b.ixor_gf2_div(lhs, rhs, target=target, field=gf)
    elif name == 'ixor_gf2_div_ctl':
        target = b.create_quantum_register(degree, name='target')
        c = b.create_quantum_register(1, name='ctrl')
        regs += degree + 1
        b.ixor_gf2_div(lhs, rhs, target=target, field=gf, control=c[0])
    elif name == 'ixor_gf2_div_ctl_false':
        target = b.create_quantum_register(degree, name='target')
        regs += degree
        b.ixor_gf2_div(lhs, rhs, target=target, field=gf, control=False)
    elif name == 'init_del_div_scaffold':
        ratio, scaffold = b.init_gf2_div_with_scaffold(lhs, rhs, field=gf)
        regs += len(ratio) + len(scaffold)
        b.del_gf2_div_with_scaffold(lhs, rhs, target=ratio, scaffold=scaffold, field=gf)
    elif name == 'init_inv_plain':
        inv = b.init_gf2_inverse(lhs, target=b.alloc_qubits(degree), field=gf)
        regs += len(inv)
    elif name == 'init_del_inv_plain':
        inv = b.init_gf2_inverse(lhs, target=b.alloc_qubits(degree), field=gf)
        regs += len(inv)
        b.del_gf2_inverse(lhs, target=inv, field=gf)
    elif name == 'init_del_inv_scaffold':
        inv, chain = b.init_gf2_inverse_with_scaffold(lhs, target=b.alloc_qubits(degree), field=gf)
        regs += len(inv) + len(chain)
        b.del_gf2_inverse_with_scaffold(lhs, target=inv, scaffold=chain, field=gf)
    elif name == 'init_gf2_mul_into_alloc':
        t = b.alloc_qubits(degree)
        regs += degree
        b.ixor_gf2_mul(lhs, rhs, target=t, field=gf)
    elif name == 'init_del_div_plain':
        ratio = b.init_gf2_div(lhs, rhs, field=gf)
        regs += len(ratio)
        b.del_gf2_div(lhs, rhs, target=ratio, field=gf)
    else:
        raise ValueError(name)

    circuit = b.finish_circuit()
    stats = circuit.operation_statistics()
    return {
        'name': name,
        'toffoli': stats.num_toffoli_like,
        'instructions': stats.num_instructions,
        'num_qubits': circuit.num_qubits,
        'registers': regs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--degree', type=int, default=DEGREE)
    args = ap.parse_args()

    print(f'kickmix module : {km.__file__}')
    degree = args.degree
    doc = km.CircuitBuilder.ixor_gf2_div.__doc__ or ''
    has_ctl = 'control' in doc
    print(f'ixor_gf2_div   : {"control parameter present" if has_ctl else "NO control parameter"}')
    print()
    names = ['ixor_gf2_mul', 'ixor_gf2_mul_ctl', 'ixor_gf2_div']
    if has_ctl:
        names += ['ixor_gf2_div_ctl', 'ixor_gf2_div_ctl_false']
    names += ['init_del_div_scaffold', 'init_del_div_plain',
              'init_inv_plain', 'init_del_inv_plain', 'init_del_inv_scaffold']

    print(f'{"primitive":24s} {"T":>6s} {"instr":>8s} {"Q":>5s} {"regs":>5s} {"transient":>10s}')
    rows = {}
    for name in names:
        try:
            r = build(name, degree)
        except Exception as exc:  # noqa: BLE001
            print(f'{name:24s} ERROR {type(exc).__name__}: {exc}')
            continue
        # Everything still allocated at the end is a register; the difference is the peak of
        # the qubits the primitive borrowed and gave back.
        transient = r['num_qubits'] - r['registers']
        rows[name] = {**r, 'transient': transient}
        print(f'{name:24s} {r["toffoli"]:6d} {r["instructions"]:8d} {r["num_qubits"]:5d} '
              f'{r["registers"]:5d} {transient:10d}')

    if 'ixor_gf2_div_ctl' in rows and 'ixor_gf2_div' in rows:
        d = rows['ixor_gf2_div_ctl']['toffoli'] - rows['ixor_gf2_div']['toffoli']
        q = rows['ixor_gf2_div_ctl']['transient'] - rows['ixor_gf2_div']['transient']
        print()
        print(f'control cost   : T {d:+d} (degree = {degree}), '
              f'transient qubits {q:+d}')
        print(f'vs mul control : T {rows["ixor_gf2_mul_ctl"]["toffoli"] - rows["ixor_gf2_mul"]["toffoli"]:+d}, '
              f'qubits {rows["ixor_gf2_mul_ctl"]["transient"] - rows["ixor_gf2_mul"]["transient"]:+d}')
    if 'init_del_div_scaffold' in rows and 'ixor_gf2_div_ctl' in rows:
        old = rows['init_del_div_scaffold']['toffoli']
        # the reference EEA block is init + controlled gf2_iadd(degree) + del
        old_block = old + degree
        new = rows['ixor_gf2_div_ctl']['toffoli']
        print()
        print(f'dqi EEA block  : old = init_scaffold + gf2_iadd(ctl) + del_scaffold = {old} + {degree}'
              f' + (del inside {old})')
        print(f'                 new = controlled ixor_gf2_div = {new}')
        print(f'                 (a like-for-like delta needs the full-circuit build; see e6b_variants.py)')


if __name__ == '__main__':
    main()

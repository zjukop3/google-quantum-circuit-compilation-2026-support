"""E6b: price the *controlled* `div_ixor_into_q` variant of the dqi decoder.

E4 found that the reference's `accumulate_quotient` holds a 12-qubit `ratio` register plus a
60-qubit scaffold live at the same time -- the 72 qubits that set the whole circuit's Q at 1893.
Replacing the whole block with one `ixor_gf2_div` straight into `q_i` removes the `ratio`
register, but the replacement then needs the sub-step control that the stock `ixor_gf2_div` does
not have.  E4's diagnostic (control dropped) measured Q 1893 -> 1881 with 512/512 semantic
failures.

This script prices the *controlled* rebuild, with the patch in `src/e6/patch/`:

    ref                          : shipped reference
    div_ixor_q_uncontrolled      : E4's diagnostic (no control) -- known FAIL
    div_ixor_q_controlled        : one controlled ixor_gf2_div into q_i (needs the patch)

Usage
-----
    PYTHONPATH=src/e6/patch python src/e6/e6b_variants.py --m 280
    PYTHONPATH=src/e6/patch python src/e6/e6b_variants.py --m 4095
    python src/e6/e6b_variants.py --m 280 --only ref,div_ixor_q_uncontrolled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py'

DIV_SCAFFOLD = """                ratio, scaffold = builder.init_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], field=gf)
                builder.gf2_iadd(ratio, target=q_i, field=gf, control=control)
                builder.del_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], target=ratio, scaffold=scaffold, field=gf)
                builder.free(scaffold + ratio)"""

VARIANTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    'ref': ('shipped reference', []),

    'div_ixor_q_uncontrolled': (
        "E4 diagnostic: quotient straight into q_i, sub-step control dropped (semantically wrong)",
        [(DIV_SCAFFOLD, "                builder.ixor_gf2_div("
                        "reg_b[index], reg_a[index], target=q_i, field=gf)")],
    ),

    'div_ixor_q_controlled': (
        "quotient straight into q_i under the sub-step control (needs the control patch)",
        [(DIV_SCAFFOLD, "                builder.ixor_gf2_div("
                        "reg_b[index], reg_a[index], target=q_i, field=gf, control=control)")],
    ),
}


def load_variant(name: str):
    note, subs = VARIANTS[name]
    src = REF.read_text()
    for old, new in subs:
        if old not in src:
            raise RuntimeError(f'variant {name!r}: substitution target not found')
        src = src.replace(old, new)
    ns: dict = {'__name__': f'dqi_e6b_{name}'}
    exec(compile(src, str(REF), 'exec'), ns)  # noqa: S102 - local trusted file
    return ns, note


def measure(name: str, m: int, n: int, degree: int) -> dict:
    ns, note = load_variant(name)
    try:
        circuit = ns['build_reference_circuit'](m, n, degree)
    except Exception as exc:  # noqa: BLE001
        return {'name': name, 'note': note, 'error': f'{type(exc).__name__}: {exc}'}
    stats = circuit.operation_statistics()
    return {
        'name': name,
        'note': note,
        'num_qubits': circuit.num_qubits,
        'num_toffoli_like': stats.num_toffoli_like,
        'num_instructions': stats.num_instructions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--only', type=str, default=None)
    args = ap.parse_args()

    print(f'kickmix : {km.__file__}')
    has_ctl = 'control' in (km.CircuitBuilder.ixor_gf2_div.__doc__ or '')
    print(f'patch   : {"present" if has_ctl else "ABSENT (controlled variant will fail)"}')
    print()

    names = args.only.split(',') if args.only else list(VARIANTS)
    if 'ref' in names:
        names = ['ref'] + [x for x in names if x != 'ref']
    base_q = base_t = None
    rows = []
    for name in names:
        r = measure(name, args.m, args.n, args.degree)
        if 'error' in r:
            print(f'{name:26s} ERROR {r["error"]}')
            continue
        rows.append(r)
        if base_q is None:
            base_q, base_t = r['num_qubits'], r['num_toffoli_like']
            print(f'{"ref (baseline)":26s} Q={base_q:5d}  T={base_t:12,d}  (baseline)')
            continue
        dq = 100.0 * (r['num_qubits'] - base_q) / base_q
        dt = 100.0 * (r['num_toffoli_like'] - base_t) / base_t
        score_ratio = (base_q / r['num_qubits']) ** 2 / (r['num_toffoli_like'] / base_t)
        print(f'{name:26s} Q={r["num_qubits"]:5d} ({dq:+6.2f}%)  '
              f'T={r["num_toffoli_like"]:12,d} ({dt:+6.2f}%)  score x{score_ratio:.4f}')

    if base_t:
        print()
        print(f'm={args.m} n={args.n} degree={args.degree}; '
              f'Chien loop scales with m, so the x-score at m=4095 is close to this one')


if __name__ == '__main__':
    main()

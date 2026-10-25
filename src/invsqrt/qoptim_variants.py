"""E4: scouting harness for dqi qubit-diet variants.

Builds `build_reference_circuit(m, n, degree)` with a small set of *semantics-
preserving* source-level substitutions and reports `num_qubits` (Q) and
`num_toffoli_like` (T) for each, so candidates can be ranked before any
simulation is spent.

Every variant is a literal string substitution on the shipped reference source
(`data/verifiers/dqi/dqi_reference_circuit.py`), so the diff is auditable and no
variant can silently change the algorithm.

Usage
-----
    python src/e4/qoptim_variants.py --m 280 --n 70
    python src/e4/qoptim_variants.py --m 280 --n 70 --only ref,div_noscaffold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py'


# --------------------------------------------------------------------------- #
# Variant definitions: (name, human note, list of (old, new) substitutions)
# --------------------------------------------------------------------------- #

DIV_SCAFFOLD = """                ratio, scaffold = builder.init_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], field=gf)
                builder.gf2_iadd(ratio, target=q_i, field=gf, control=control)
                builder.del_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], target=ratio, scaffold=scaffold, field=gf)
                builder.free(scaffold + ratio)"""

DIV_PLAIN = """                ratio = builder.init_gf2_div(reg_b[index], reg_a[index], field=gf)
                builder.gf2_iadd(ratio, target=q_i, field=gf, control=control)
                builder.del_gf2_div(reg_b[index], reg_a[index], target=ratio, field=gf)
                builder.free(ratio)"""

DIV_INTO_Q = """                ratio = builder.init_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], field=gf, target=q_i)[0]
                raise AssertionError('unused')"""

INV_CHIEN_SCAFFOLD = """                _, inverse_chain = builder.init_gf2_inverse_with_scaffold(
                    sigma_prime, target=sigma_prime_inv, field=gf)"""

INV_CHIEN_PLAIN = """                inverse_chain = None
                builder.init_gf2_inverse(sigma_prime, target=sigma_prime_inv, field=gf)"""

INV_CHIEN_DEL_SETUP = """                builder.del_gf2_inverse_with_scaffold(
                    sigma_prime, target=sigma_prime_inv, scaffold=inverse_chain, field=gf)
                builder.free(inverse_chain)"""

INV_CHIEN_DEL_PLAIN = """                builder.del_gf2_inverse(
                    sigma_prime, target=sigma_prime_inv, field=gf)"""

VARIANTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    'div_ixor_into_q': (
        'DIAGNOSTIC ONLY (drops the sub-step control): quotient computed straight '
        'into q_i with ixor_gf2_div, to expose the internal peak of the shipped '
        'division primitive',
        [(DIV_SCAFFOLD, "                builder.ixor_gf2_div(reg_b[index], reg_a[index], "
                        "target=q_i, field=gf)")],
    ),

    'ref': ('shipped reference', []),

    'div_noscaffold': (
        'EEA quotient division without the retained scaffold '
        '(del_gf2_div recomputes the inverse; T up, transient Q down?)',
        [(DIV_SCAFFOLD, DIV_PLAIN)],
    ),

    'chien_inv_noscaffold': (
        'Chien Forney inversion recomputed instead of retained '
        '(T up, transient Q down?)',
        [(INV_CHIEN_SCAFFOLD, INV_CHIEN_PLAIN),
         (INV_CHIEN_DEL_SETUP, INV_CHIEN_DEL_PLAIN)],
    ),

    'both_noscaffold': (
        'both scaffolds dropped',
        [(DIV_SCAFFOLD, DIV_PLAIN),
         (INV_CHIEN_SCAFFOLD, INV_CHIEN_PLAIN),
         (INV_CHIEN_DEL_SETUP, INV_CHIEN_DEL_PLAIN)],
    ),

    'logm_minus1': (
        'degree metadata registers one bit narrower '
        '(m_e.bit_length() instead of m_e.bit_length()+1)',
        [('        logm = m.bit_length() + 1', '        logm = m.bit_length()')],
    ),

}


def load_variant(name: str):
    """Loads the reference module with the variant's substitutions applied."""
    note, subs = VARIANTS[name]
    src = REF.read_text()
    for old, new in subs:
        if old not in src:
            raise RuntimeError(f'variant {name!r}: substitution target not found')
        src = src.replace(old, new)
    ns: dict = {'__name__': f'dqi_variant_{name}'}
    exec(compile(src, str(REF), 'exec'), ns)  # noqa: S102 - local trusted file
    return ns, note


def measure(name: str, m: int, n: int, degree: int) -> dict:
    ns, note = load_variant(name)
    builder = ns['build_reference_circuit']
    try:
        circuit = builder(m, n, degree)
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

    names = args.only.split(',') if args.only else list(VARIANTS)
    if 'ref' in names:
        names = ['ref'] + [x for x in names if x != 'ref']
    base_q = base_t = None
    for name in names:
        r = measure(name, args.m, args.n, args.degree)
        if 'error' in r:
            print(f'{name:24s} ERROR {r["error"]}')
            continue
        if base_q is None:
            base_q, base_t = r['num_qubits'], r['num_toffoli_like']
            print(f'{"ref (baseline)":24s} Q={base_q:5d}  T={base_t:9,d}  '
                  f'(baseline for deltas)')
            continue
        dq = 100.0 * (r['num_qubits'] - base_q) / base_q
        dt = 100.0 * (r['num_toffoli_like'] - base_t) / base_t
        score_ratio = (base_q / r['num_qubits']) ** 2 / max(
            r['num_toffoli_like'] / base_t, 1e-12)
        print(f'{name:24s} Q={r["num_qubits"]:5d} ({dq:+6.2f}%)  '
              f'T={r["num_toffoli_like"]:9,d} ({dt:+6.2f}%)  '
              f'score x{score_ratio:.4f}')


if __name__ == '__main__':
    main()

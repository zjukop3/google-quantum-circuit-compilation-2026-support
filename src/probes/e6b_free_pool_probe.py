"""E6b: does the dqi `quotient_div` site have enough *reusable* qubits for a
controlled `ixor_gf2_div`?

`num_free_qubits` is the headroom to `max_qubits`, not the free list, so it cannot answer
this.  What matters is whether `builder.alloc_qubits(n)` hands back already-used-then-freed
ids (no growth in `num_qubits`) or fresh ids (growth).

This probe replaces the reference's scaffolded division block with

    _ids = builder.alloc_qubits(NEED)
    record max(_ids)
    builder.free(_ids)

which is exactly the transient a controlled `ixor_gf2_div` creates from the primitive's own
workspace: `gf_div_workspace_size` = 60 (inverse 12 + Itoh-Tsujii chain 48) plus `degree` = 12
for the controlled multiply's temporary = 72.  `q_i` and `ctrl` stay live either way.

Semantics are *not* preserved (the division is dropped) -- this is a pure Q-peak probe.

Usage
-----
    python src/e6/e6b_free_pool_probe.py --m 280 --n 70
    python src/e6/e6b_free_pool_probe.py --m 280 --n 70 --need 72
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'dqi'))

REF = ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py'

DIV_ANCHOR = """            with builder.mark('quotient_div'):
                ratio, scaffold = builder.init_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], field=gf)
                builder.gf2_iadd(ratio, target=q_i, field=gf, control=control)
                builder.del_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], target=ratio, scaffold=scaffold, field=gf)
                builder.free(scaffold + ratio)"""


def make_probe(need: int, keep_dir: str, keep_q: str) -> str:
    return f"""            with builder.mark('quotient_div'):
                _ids = builder.alloc_qubits({need})
                _hi = max(x.id for x in _ids)
                {keep_dir}.append(_hi)
                {keep_q}.append(int(builder.num_allocated_qubits))
                builder.free(_ids)"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--need', type=int, default=72,
                    help='workspace qubits the patched primitive would ask for')
    args = ap.parse_args()

    src = REF.read_text()
    if DIV_ANCHOR not in src:
        raise RuntimeError('probe anchor not found in the reference circuit')

    ns: dict = {'__name__': 'dqi_e6b_probe'}
    ns['_ids_hi'] = []
    ns['_q_at_site'] = []
    ns['_e6b_probe'] = make_probe(args.need, '_ids_hi', '_q_at_site')
    src = src.replace(DIV_ANCHOR, ns['_e6b_probe'])
    exec(compile(src, str(REF), 'exec'), ns)  # noqa: S102 - local trusted file

    circuit = ns['build_reference_circuit'](args.m, args.n, args.degree)
    stats = circuit.operation_statistics()
    hi = ns['_ids_hi']
    q = ns['_q_at_site']
    print(f'probe: {args.need} qubits requested at every quotient_div site '
          f'(m={args.m}, n={args.n}, degree={args.degree})')
    print(f'sites probed            : {len(hi)}')
    print(f'allocated max qubit id  : min {min(hi)} / med {sorted(hi)[len(hi) // 2]} / max {max(hi)}')
    print(f'num_qubits at the site  : min {min(q)} / max {max(q)}')
    print(f'final circuit Q         : {circuit.num_qubits}')
    print(f'final circuit T (static): {stats.num_toffoli_like:,}')
    print()
    print('Reference (unpatched) at these flags: Q=1893, T=5,725,461  [m=4095]')
    print('Interpretation: if max allocated id stayed below the reference peak,')
    print('the workspace comes out of the recycled pool and the patch is Q-neutral.')


if __name__ == '__main__':
    main()

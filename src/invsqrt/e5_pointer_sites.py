"""E5: site-by-site T decomposition of the pointer prototype.

`e5_pointer.py` shows the whole role-pointer costs +19.4% T at n=70.  This script
turns each substitution group on/off in isolation so the note can say exactly
where the money goes:

    swap_ptr   `_swap_a_and_b`: 892-qubit cswap -> 16-qubit metadata cswap + p ^= ctrl
    sweep      per-position role wrap around the multiply-add body
    shift      `_shift_b_array_right`: chain duplicated for both arrays
    div        `accumulate_quotient`: conditional operand read
    flag       `_update_flag`: conditional reads of reg_b[m-2] / reg_b[0]
    normalize  the two whole-array cswaps bracketing Chien/Forney

Usage
-----
    python src/e5/e5_pointer_sites.py --m 280 --n 70
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import kickmix as km

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import e5_pointer as P  # noqa: E402

REF = P.REF

GROUPS: dict[str, int] = {
    # index into P.SUBS
    'alloc': 0, 'alloc_tuple': 1, 'free': 2, 'normalize': 3, 'sig': 4,
    'flag_call': 5, 'mul_call': 6, 'shift_call': 7, 'swap_call': 8,
    'flag_sig': 9, 'flag_hi': 10, 'flag_lo': 11,
    'mul_sig': 12, 'div_in': 13, 'div_out': 14, 'sweep': 15,
    'shift_sig': 16, 'shift_move': 17,
    'swap_sig': 18, 'swap_body': 19,
}

SITE = {
    'plumbing': ['alloc', 'alloc_tuple', 'free', 'sig', 'flag_call', 'mul_call',
                 'shift_call', 'swap_call', 'flag_sig', 'mul_sig', 'shift_sig',
                 'swap_sig'],
    'swap_ptr': ['swap_body'],
    'sweep': ['sweep'],
    'shift': ['shift_move'],
    'div': ['div_in', 'div_out'],
    'flag': ['flag_hi', 'flag_lo'],
    'normalize': ['normalize'],
}


def build_src(enabled: set[str]) -> str:
    """Reference source with exactly the named substitution groups applied."""
    wanted = set()
    for site in enabled:
        wanted.update(SITE[site])
    src = REF.read_text()
    applied = 0
    for idx, (old, new) in enumerate(P.SUBS):
        name = [k for k, v in GROUPS.items() if v == idx][0]
        if name in wanted:
            if old not in src:
                raise RuntimeError(f'anchor missing for {name}')
            src = src.replace(old, new, 1)
            applied += 1
    # a partial build is only valid when the plumbing is present
    if applied and 'plumbing' not in enabled:
        raise RuntimeError('plumbing group required')
    return src


def measure(enabled: set[str], m: int, n: int, degree: int) -> dict:
    if not enabled:
        src = REF.read_text()
    else:
        src = build_src(enabled)
    ns: dict = {'__name__': 'e5_site'}
    exec(compile(src, str(REF), 'exec'), ns)  # noqa: S102
    try:
        c = ns['build_reference_circuit'](m, n, degree)
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}
    st = c.operation_statistics()
    return {'num_qubits': c.num_qubits, 'static_toffoli': int(st.num_toffoli_like)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    base = measure(set(), args.m, args.n, args.degree)
    print(f'{"ref":34s} Q={base["num_qubits"]:5d}  T={base["static_toffoli"]:12,d}')
    out = {'m': args.m, 'n': args.n, 'ref': base, 'sites': {}}

    # one site at a time (plumbing always on, because it carries the saving)
    order = ['swap_ptr', 'sweep', 'shift', 'div', 'flag', 'normalize']
    for site in order:
        r = measure({'plumbing', site}, args.m, args.n, args.degree)
        if 'error' in r:
            print(f'{site:34s} ERROR {r["error"]}')
            continue
        d = r['static_toffoli'] - base['static_toffoli']
        dq = r['num_qubits'] - base['num_qubits']
        print(f'plumbing + {site:23s} Q={r["num_qubits"]:5d} ({dq:+d})  '
              f'T={r["static_toffoli"]:12,d}  dT={d:+12,d} ({100.0*d/base["static_toffoli"]:+7.3f}%)')
        out['sites'][site] = {**r, 'dT': d, 'dQ': dq}

    full = measure(set(SITE), args.m, args.n, args.degree)
    out['full'] = full
    if 'static_toffoli' in full:
        d = full['static_toffoli'] - base['static_toffoli']
        print(f'{"full pointer prototype":34s} Q={full["num_qubits"]:5d} '
              f'({full["num_qubits"]-base["num_qubits"]:+d})  '
              f'T={full["static_toffoli"]:12,d}  dT={d:+12,d} '
              f'({100.0*d/base["static_toffoli"]:+.3f}%)')
        # sum of the per-site deltas (sanity: should reconstruct the full delta)
        s = sum(v['dT'] for v in out['sites'].values())
        print(f'{"sum of isolated site deltas":34s} {"":21s} dT={s:+12,d} '
              f'({100.0*s/base["static_toffoli"]:+.3f}%)')
        out['sum_site_deltas'] = s

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, default=float))
        print(f'\nwrote {p}')


if __name__ == '__main__':
    main()

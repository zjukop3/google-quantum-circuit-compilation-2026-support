"""E8-J — build a production artifact from a `cswap` (and optionally `mul`) freeze.

    $P src/e8/e8_build.py --freeze build/e8b/cswap_freeze.npz \
        --out build/e8/dqi_cswap.kmb
    # stacked: E7's converged mul freeze + the E8 cswap freeze in one file
    $P src/e8/e8_build.py --freeze build/e8b/cswap_freeze.npz \
        --mul-freeze build/e8/mul_freeze_e7.npz --stacked \
        --out build/e8/dqi_stacked.kmb

Prints the T / Q ledger, the sha256 and a one-shot quick price
(`num_toffoli_like`) so the delta against the reference is verifiable before any
full verification run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'e6'))
sys.path.insert(0, str(HERE))

import e8_src as E8  # noqa: E402

FAMILIES = ('swap', 'shift', 'sweep')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', default=None)
    ap.add_argument('--mul-freeze', default=None,
                    help='npz with a `mul` key, or E7 freeze_sites.json')
    ap.add_argument('--mul-model', default='nA070_69_K2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--m', type=int, default=4095)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    freeze, families = {}, []
    if args.freeze:
        z = np.load(args.freeze)
        for f in FAMILIES + ('mul',):
            if f in z:
                freeze[f] = z[f]
                families.append(f)
    if args.mul_freeze:
        p = Path(args.mul_freeze)
        if p.suffix == '.json':
            freeze['mul'] = E8.load_mul_mask(p, args.mul_model, n)[0]
        else:
            freeze['mul'] = np.load(p)['mul']
        families.append('mul')
    if not families:
        print('nothing to freeze')
        return 1

    fams = tuple(families)
    ns = E8.load_module('prod', freeze=True, families=fams)
    fs = E8.full_freeze(freeze, n)
    t0 = time.time()
    circ, _d, _s = E8.build_circuit(ns, args.m, n, degree, freeze=fs)
    st = circ.operation_statistics()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    circ.to_file(out, format='kmb')
    h = hashlib.sha256(out.read_bytes()).hexdigest()

    ref_ns = {'__name__': 'e8ref_prod'}
    exec(compile(E8.REF.read_text(), str(E8.REF), 'exec'), ref_ns)
    rq, _d2, _s2 = E8.build_circuit(ref_ns, args.m, n, degree)
    rst = rq.operation_statistics()

    print(f'[families] {fams}')
    print('[freeze sites/pass] ' + ' '.join(
        f'{f}={int(freeze[f].sum())}' for f in fams))
    print(f'[ref]    T={rst.num_toffoli_like:,} Q={rq.num_qubits}')
    print(f'[frozen] T={st.num_toffoli_like:,} Q={circ.num_qubits}')
    dT = st.num_toffoli_like - rst.num_toffoli_like
    print(f'[delta]  dT={dT:,} ({dT / rst.num_toffoli_like:+.4%})  '
          f'ratio={rst.num_toffoli_like / st.num_toffoli_like:.6f}  '
          f'Q delta={circ.num_qubits - rq.num_qubits}')
    print(f'[write]  {out} {out.stat().st_size:,} B  sha256={h}')
    print(f'[time]   {time.time() - t0:.1f}s')
    rec = {'families': list(fams), 'm': args.m, 'n': n, 'degree': degree,
           'T_ref': int(rst.num_toffoli_like), 'T_frozen': int(st.num_toffoli_like),
           'Q_ref': rq.num_qubits, 'Q_frozen': circ.num_qubits,
           'delta_T': int(dT), 'ratio': rst.num_toffoli_like / st.num_toffoli_like,
           'sha256': h, 'out': str(out),
           'freeze_sites_per_pass': {f: int(freeze[f].sum()) for f in fams}}
    if args.json:
        Path(args.json).write_text(json.dumps(rec, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""E7-F — build the statically frozen dqi circuit (`dqi.kmb`).

Freeze set: the `(it, i)` sweep sites that are dead in the **provable union**
`e7_enum.py` computes for the validated deviation model (Z1 stalls <= K plus the
initial-shift class), applied to both EEA passes.  On those sites the
`ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], control=mul)` call is deleted
(75 T-like gates each, times two passes).

Everything else -- register layout, the `mul` qubit and its fan-in, the
position-controlled `cswap`, the shift gates -- is left bit-identical, so the
only difference between the reference and the frozen circuit is the missing
multipliers.

Usage
-----
    $P src/e7/e7_build.py --model nA070_69_K2 --out build/e7/dqi_frozen.kmb
    $P src/e7/e7_build.py --model none --out /tmp/dqi_ref_rebuild.kmb   # baseline
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

import e6_semantics as ES          # noqa: E402

N_ITERS = 222
M_E = 73


def freeze_array(model: str, path: Path):
    """(2, n_iters, m_e) bool: True = the multiplier is deleted at that site."""
    if model == 'none':
        return None
    spec = json.loads(path.read_text())[model]
    FS = np.zeros((2, N_ITERS, M_E), dtype=bool)
    for it, i in spec['sites']:
        FS[0, int(it), int(i)] = True
        FS[1, int(it), int(i)] = True
    return FS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', default=str(ROOT / 'notes/runs/e7/freeze_sites.json'))
    ap.add_argument('--model', default='nA070_69_K2')
    ap.add_argument('--m', type=int, default=4095)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--out', default=None)
    ap.add_argument('--stats-only', action='store_true')
    args = ap.parse_args()

    ref = ES.load_module('e7ref_build', ROOT / 'data' / 'verifiers' / 'dqi'
                         / 'dqi_reference_circuit.py')
    FS = freeze_array(args.model, Path(args.freeze))
    n_frozen = 0 if FS is None else int(FS[0].sum())
    t0 = time.time()
    circ = ES.build_circuit(ref, args.m, args.n, args.degree, frozen_sites=FS)
    st = circ.operation_statistics()
    print(f'[build] model={args.model} m={args.m} n={args.n} b={args.degree} '
          f'({time.time() - t0:.1f}s)')
    print(f'[build] frozen sites (per pass) = {n_frozen}')
    print(f'[build] Q = {circ.num_qubits:,}   T = {st.num_toffoli_like:,}')
    if not args.stats_only:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        circ.to_file(out, format='kmb')
        h = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f'[build] wrote {out}  {out.stat().st_size:,} B  sha256={h[:16]}… '
              f'({time.time() - t1:.1f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

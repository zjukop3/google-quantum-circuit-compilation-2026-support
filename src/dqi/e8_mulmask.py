"""E8-H — recover E7's converged mul freeze mask, and prove the E8 patch route is
byte-identical to E7's.

E7's fixpoint never saved its final mask, but the sample of every artifact it
built is reproducible (`seed = int(sha256(.kmb), 16)`), and `build/e7/dqi_fp0.kmb`
/ `dqi_fp1.kmb` are still on disk.  So

    U_e7 = union_acc.npy                                     (307,200 seed shots)
         | mul_bitmap(sample_k)  for k = 0, 1                (the two rounds)

and the frozen mask is `~U_e7`, which must have exactly 1,391 sites.

The stronger check: rebuild the mul-only freeze through the **E8** patcher and
compare the resulting sha256 with E7's verified artifact `dqi_fp2.kmb`
(`3dfcad7bea9be65bc60c2fb224acf417a3eab477348f07e5cef9cea44a245ca2`).  Equality
means the two independent patch routes emit gate-identical circuits, so the
stacked artifact's mul half is literally the artifact that already PASSED the
official 102,400-shot verification.

Usage
-----
    $P src/e8/e8_mulmask.py --json notes/runs/e8/mulmask.json
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
sys.path.insert(0, str(ROOT / 'src' / 'e7'))
sys.path.insert(0, str(HERE))

import e8_src as E8                # noqa: E402
import e6d_schedule as S           # noqa: E402
import e7_diag_official as D       # noqa: E402

import kickmix as km               # noqa: E402

TARGET_SHA = '3dfcad7bea9be65bc60c2fb224acf417a3eab477348f07e5cef9cea44a245ca2'


def mul_union_of_file(kfile: Path, circ, dec, syn, n, degree, shots, batch):
    h = hashlib.sha256(kfile.read_bytes()).hexdigest()
    sy, _mb, _ph = D.official_sample(int(h, 16), shots, batch, n=n, degree=degree)
    U = np.zeros((222, 73), dtype=bool)
    chunk = 2048
    for start in range(0, shots, chunk):
        stop = min(start + chunk, shots)
        T = S.run_batch(circ, dec, syn, sy[start:stop], degree, seed=7, n_iters=222)
        U |= S.mul_bitmap(T, 73).any(axis=0)
    return h, U


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shots', type=int, default=102400)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--m', type=int, default=4095)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    n, degree = args.n, args.degree

    U = np.load(ROOT / 'notes/runs/e7/union_acc.npy').astype(bool)
    print(f'[seed] union_acc  lit={int(U.sum())} freeze={int((~U).sum())}')
    ns = S.load_probe_module()
    circ, dec, n_iters, m_e, syn = S.build(n, degree, n, ns)
    for k in (0, 1):
        f = ROOT / f'build/e7/dqi_fp{k}.kmb'
        if not f.exists():
            print('[skip]', f, 'missing')
            continue
        t0 = time.time()
        h, Uk = mul_union_of_file(f, circ, dec, syn, n, degree, args.shots,
                                  args.batch_size)
        add = int((Uk & ~U).sum())
        U |= Uk
        print(f'[round {k}] {f.name} sha={h[:16]}… sample_union={int(Uk.sum())} '
              f'new={add}  union now {int(U.sum())} ({time.time() - t0:.0f}s)',
              flush=True)

    freeze = ~U
    print(f'[result] union={int(U.sum())} freeze={int(freeze.sum())} '
          f'(E7 log says freeze=1391)')

    ns_b = E8.load_module('mulonly', freeze=True, families=('mul',))
    FS = np.zeros(E8.grid_shapes(n)['mul'], dtype=bool)
    FS[0] = freeze
    FS[1] = freeze
    out = ROOT / 'build/e8/dqi_mulonly_repro.kmb'
    out.parent.mkdir(parents=True, exist_ok=True)
    c, _d, _s = E8.build_circuit(ns_b, args.m, n, degree, freeze={'mul': FS})
    c.to_file(out, format='kmb')
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    st = c.operation_statistics()
    print(f'[repro] T={st.num_toffoli_like:,} Q={c.num_qubits} sha={h[:16]}…')
    print(f'[repro] matches E7 dqi_fp2.kmb: {h == TARGET_SHA}')
    res = {'union': int(U.sum()), 'freeze': int(freeze.sum()),
           'repro_sha': h, 'target_sha': TARGET_SHA, 'sha_match': h == TARGET_SHA,
           'T': int(st.num_toffoli_like)}
    np.savez(ROOT / 'build/e8/mul_freeze_e7.npz', mul=freeze)
    print('wrote', ROOT / 'build/e8/mul_freeze_e7.npz')
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

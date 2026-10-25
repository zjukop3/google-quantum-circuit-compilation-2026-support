"""E4: is a narrower EEA metadata register an equivalent optimisation?  No.

`dqi_reference_circuit.py` allocates the five EEA metadata registers
(`n_A, n_B, n_a, n_b, n_q`) with `logm = m_e.bit_length() + 1 = 8` bits.  Dropping
the `+ 1` builds a circuit with Q = 1888 instead of 1893 (score x1.0091) and it
*passes* 2048 shots of the official distribution plus 1024 max-weight shots.

This script shows why that pass is not evidence of correctness: the reference's
metadata registers **overflow their own modulus by design**.  Measured at
`n = 70`, the 8-bit residues reach `n_A = 253`, `n_a = 148` -- i.e. the schedule
consumes `n_A mod 2^8`, and changing the modulus changes the schedule.  Tracing
both widths on the same syndromes, `n_B` already differs at forward iteration 0
and `n_a`/`n_B` differ at all 222 of them.

So `logm_minus1` is a *different algorithm*, not an equivalent one, and the
reference's zero-failure evidence (102,400 shots, eps_plus = 8.66e-4) does not
transfer to it.  It is rejected.  See `notes/E4_DQI_QUBIT.md` section 4.4.

Usage
-----
    python src/e4/metadata_width.py --n 70 --shots 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from regb_liveness import run_trace, word_values  # noqa: E402

FIELDS = ['n_A', 'n_B', 'n_a', 'n_b', 'n_q']


def trace_metadata(n: int, degree: int, shots: int, seed: int, logm_extra: int):
    """Per-iteration `(5, iters, shots)` metadata words at the given register width."""
    _a, _b, meta, _nd = run_trace(n, degree, shots, seed, None, 'uniform', 'start',
                                  logm_extra)
    logm = (n + 3).bit_length() + logm_extra
    return [word_values(meta[:, k * logm:(k + 1) * logm, :], logm)[:, 0, :]
            for k in range(5)]


def compare(n: int = 70, degree: int = 12, shots: int = 4, seed: int = 20261025) -> dict:
    shipped = trace_metadata(n, degree, shots, seed, 1)   # logm = m_e.bit_length()+1
    narrow = trace_metadata(n, degree, shots, seed, 0)    # logm = m_e.bit_length()
    iters = shipped[0].shape[0]
    out = {}
    for name, s, k in zip(FIELDS, shipped, narrow):
        differing = [it for it in range(iters) if not np.array_equal(s[it], k[it])]
        over = [it for it in range(iters) if (s[it] > 127).any()]
        out[name] = {
            'max_shipped': int(s.max()), 'max_narrow': int(k.max()),
            'iters_differing': len(differing), 'iters': iters,
            'first_differing_iter': differing[0] if differing else None,
            'iters_shipped_gt_127': len(over),
            'first_iter_gt_127': over[0] if over else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261025)
    args = ap.parse_args()

    res = compare(args.n, args.degree, args.shots, args.seed)
    print(f'EEA metadata: shipped logm = m_e.bit_length()+1 = 8 bits   '
          f'vs narrow logm = 7 bits   (n={args.n}, shots={args.shots})')
    print()
    hdr = (f"{'field':6s} {'max(shipped)':>13s} {'max(narrow)':>12s} "
           f"{'iters differing':>17s} {'first diff':>11s} "
           f"{'iters >127':>11s} {'first >127':>11s}")
    print(hdr)
    print('-' * len(hdr))
    for name, d in res.items():
        print(f"{name:6s} {d['max_shipped']:13d} {d['max_narrow']:12d} "
              f"{d['iters_differing']:>10d}/{d['iters']:<5d} "
              f"{str(d['first_differing_iter']):>11s} "
              f"{d['iters_shipped_gt_127']:11d} {str(d['first_iter_gt_127']):>11s}")
    print()
    print('Verdict: the shipped metadata registers exceed 2**7 (residues reach '
          f"{max(d['max_shipped'] for d in res.values())}), and the schedule is a "
          'function of the residue.')
    print('         logm_minus1 therefore changes the algorithm from iteration 0 -- '
          'NOT an equivalent optimisation.')


if __name__ == '__main__':
    main()

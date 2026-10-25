"""E5: close the "just make the move unconditional" loophole empirically.

`e5_ablate.py` established that an unconditional swap / rotate costs **0 T** in
`num_toffoli_like`, so the entire 779,664 T of `_swap_a_and_b` +
`_shift_b_array_right` is the price of their *quantum controls*.  The obvious
counter-proposal is then "drop the controls and pay 0 T".  This script prices and
*checks* that proposal:

    probe_swap_uncond    A<->B swapped every iteration, unconditionally
    probe_shift_uncond   the shift chain executed with no control at all
    probe_both_uncond    both

Every one of these is a hard 779,664 T win (-13.62% at m=4095, score x1.16) --
exactly E4's headline number.  They are also wrong, and this script shows *how*
wrong, so the E4 upper bound is not left standing as an actionable target.

Usage
-----
    python src/e5/e5_freeceil.py --n 10 --shots 512
    python src/e5/e5_freeceil.py --n 70 --shots 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src' / 'e4'))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'dqi'))

import e5_ablate as AB  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=512)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    import semantic as SEM  # noqa: E402
    import verify_submission_dqi as V  # noqa: E402
    import kickmix as km  # noqa: E402

    spec = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=args.n, degree=args.degree,
                                               num_shots=args.shots)
    tables = V.FieldTables(km.GF2Field(args.degree))
    shots = SEM.make_shots(spec, tables, 20261025, args.shots, 'official')

    names = ['ref', 'probe_no_swap', 'probe_no_shift', 'probe_neither',
             'probe_swap_uncond', 'probe_shift_uncond', 'probe_both_uncond']
    out: dict = {'m': args.m, 'n': args.n, 'shots': args.shots, 'variants': {}}
    base = None
    print(f'{args.m=} {args.n=} {args.shots=} (official weight distribution)')
    for name in names:
        ns, note = AB.load_variant(name)
        t0 = time.time()
        try:
            circuit = ns['build_reference_circuit'](args.m, args.n, args.degree)
        except Exception as exc:  # noqa: BLE001
            print(f'{name:20s} BUILD ERROR {type(exc).__name__}: {exc}')
            continue
        q = circuit.num_qubits
        t = int(circuit.operation_statistics().num_toffoli_like)
        res = SEM.check_circuit(circuit, spec, shots, batch_size=64)
        rec = {'num_qubits': q, 'static_toffoli': t, 'semantic': res}
        if base is None:
            base = rec
            dbest = ''
        else:
            dt = 100.0 * (t - base['static_toffoli']) / base['static_toffoli']
            mult = (base['num_qubits'] / q) ** 2 / (t / base['static_toffoli'])
            rec['dT_pct'] = dt
            rec['score_multiple'] = mult
            dbest = f' dT={dt:+8.3f}%  score x{mult:.4f}'
        print(f'{name:20s} Q={q:5d} T={t:12,d}{dbest}  '
              f'value={res["value_errors"]:5d} phase={res["phase_errors"]:5d} '
              f'-> {res["verdict"]}  ({time.time()-t0:.1f}s)')
        out['variants'][name] = rec

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, default=float))
        print(f'\nwrote {p}')


if __name__ == '__main__':
    main()

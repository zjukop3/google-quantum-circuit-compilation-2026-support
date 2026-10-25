"""E4: bit-exact equivalence tests for the qubit-dieted invsqrt circuits.

Three circuits are compared **phase for phase** over the whole input space for small
`n_p`, using the shipped sampler (`km.Simulator`) and exact `Fraction` phases:

    reference    data/verifiers/invsqrt/nr_reference_circuit.build_nr_circuit
    E3 opt       src/invsqrt_opt_circuit.build_opt_circuit     (3-register Horner)
    E4 opt2      src/e4/invsqrt_opt2_circuit.build_opt2_circuit (3-register Horner
                 + `3 - z` folded into the final multiply + one-bit-narrower `z`)

Checks per circuit and per `x`:
  * `x` is restored exactly;
  * every scratch qubit returns to |0>;
  * the tracked phase (turns) equals the reference's, as an exact `Fraction`.

Also compares each circuit's `y1` value against the shipped `fixed_point_model`
(which is the analytic ground truth the verifier's tolerance is measured against),
by running the circuit in `compute_only` mode where it stops with `y1` named.

Usage
-----
    ../2609-xingzhi-drug/.venv/bin/python src/e4/invsqrt_equiv.py
    ../2609-xingzhi-drug/.venv/bin/python src/e4/invsqrt_equiv.py --n-p 5 --out notes/runs/e4_equiv.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'invsqrt'))

import kickmix as km  # noqa: E402

from nr_reference_circuit import Precision, build_nr_circuit, fixed_point_model  # noqa: E402
import invsqrt_opt_circuit as O  # noqa: E402
import invsqrt_opt2_circuit as N  # noqa: E402

PASS_PREC = dict(delta_bits=12, coef_bits=17, square_bits=28, product_bits=27, result_bits=28)
BUILDERS = {'reference': build_nr_circuit, 'e3_opt': O.build_opt_circuit, 'e4_opt2': N.build_opt2_circuit}


def to_bits(values: np.ndarray, width: int) -> np.ndarray:
    return (values[None, :] >> np.arange(width, dtype=np.uint64)[:, None]) & 1 == 1


def parts(circuit: km.Circuit, n_p: int):
    width = 2 * n_p + 2
    x_reg = list(circuit.register_data['x'])
    io = set(x_reg)
    scratch = km.array([km.q(i) for i in range(circuit.num_qubits) if km.q(i) not in io])
    return width, km.array(x_reg), scratch


def run_phases(circuit: km.Circuit, n_p: int, xs: np.ndarray) -> dict:
    width, x_reg, scratch = parts(circuit, n_p)
    sim = km.Simulator(batch_size=len(xs), debug_print=False, counting_operations=True, seed=11)
    sim.clear_for_shot()
    sim.write_across_shots(x_reg, to_bits(xs, width))
    sim.do(circuit)
    got = (sim.read_across_shots(x_reg).astype(np.uint64)
           << np.arange(width, dtype=np.uint64)[:, None]).sum(axis=0)
    dirty = sim.read_across_shots(scratch).any(axis=1) if len(scratch) else np.zeros(len(xs), bool)
    return {'phases': [sim.read_shot_phase(i) for i in range(len(xs))],
            'x_restored': bool((got == xs).all()),
            'dirty_count': int(dirty.sum())}


def run_y1(circuit: km.Circuit, n_p: int, xs: np.ndarray) -> np.ndarray:
    """Read the named `y1` register of a `compute_only` circuit."""
    width, _, _ = parts(circuit, n_p)
    y1 = km.array(list(circuit.register_data['y1']))
    sim = km.Simulator(batch_size=len(xs), debug_print=False, counting_operations=True, seed=12)
    sim.clear_for_shot()
    sim.write_across_shots(km.array(list(circuit.register_data['x'])), to_bits(xs, width))
    sim.do(circuit)
    bits = sim.read_across_shots(y1)
    return (bits.astype(np.uint64) << np.arange(len(y1), dtype=np.uint64)[:, None]).sum(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-p', type=int, nargs='+', default=[2, 3, 4, 5])
    ap.add_argument('--out', type=Path, default=ROOT / 'notes' / 'runs' / 'e4_equiv.json')
    args = ap.parse_args()

    report = {}
    all_ok = True
    print(f'{"n_p":>4s} {"Q_ref":>6s} {"Q_e3":>6s} {"Q_e4":>6s} {"T_ref":>6s} {"T_e3":>6s} {"T_e4":>6s} '
          f'{"phase==ref":>11s} {"y1==model":>10s} {"x ok":>5s} {"clean":>6s}')
    for n_p in args.n_p:
        prec = Precision(**PASS_PREC)
        width = 2 * n_p + 2
        xs = np.arange(1, 2**width, dtype=np.uint64)
        circ = {}
        qs, ts = {}, {}
        for name, fn in BUILDERS.items():
            c = fn(n_p, prec=prec)
            circ[name] = c
            qs[name] = c.num_qubits
            ts[name] = c.operation_statistics().num_toffoli_like

        ref = run_phases(circ['reference'], n_p, xs)
        row = {'n_p': n_p, 'Q': qs, 'T': ts, 'x_restored': {}, 'dirty': {}}
        phase_ok = {}
        for name in BUILDERS:
            if name == 'reference':
                phase_ok[name] = True
                row['x_restored'][name] = ref['x_restored']
                row['dirty'][name] = ref['dirty_count']
                continue
            r = run_phases(circ[name], n_p, xs)
            phase_ok[name] = (r['phases'] == ref['phases'])
            row['x_restored'][name] = r['x_restored']
            row['dirty'][name] = r['dirty_count']

        # y1 vs the shipped model, in compute_only mode.
        model = fixed_point_model(n_p, prec, xs.astype(np.int64))
        y1_ok = {}
        for name, fn in BUILDERS.items():
            co = fn(n_p, prec=prec, compute_only=True)
            got = run_y1(co, n_p, xs)
            y1_ok[name] = bool((got == model.astype(np.uint64)).all())

        row['phase_matches_reference'] = phase_ok
        row['y1_matches_model'] = y1_ok
        report[n_p] = row
        ok = (all(phase_ok.values()) and all(y1_ok.values())
              and all(row['x_restored'].values()) and not any(row['dirty'].values()))
        all_ok &= ok
        print(f'{n_p:>4d} {qs["reference"]:>6d} {qs["e3_opt"]:>6d} {qs["e4_opt2"]:>6d} '
              f'{ts["reference"]:>6d} {ts["e3_opt"]:>6d} {ts["e4_opt2"]:>6d} '
              f'{str(phase_ok):>11s} {str(all(y1_ok.values())):>10s} '
              f'{str(all(row["x_restored"].values())):>5s} {str(not any(row["dirty"].values())):>6s}'
              f'{"  <-- OK" if ok else "  <-- MISMATCH"}', flush=True)

    report['all_ok'] = all_ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f'\n{"ALL EQUIVALENT" if all_ok else "MISMATCH FOUND"} -> {args.out}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

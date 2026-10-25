"""E4: fast, deterministic semantic check for dqi decoder candidates.

Reuses the *official* verifier's sampling and register validation
(`sample_shots`, `optimal_dqi_weights`, `FieldTables`, `validate_registers`,
`to_bits`) so a local PASS means exactly what the shipped verifier means by it:
on every shot the syndrome registers come back unchanged, every non-syndrome
qubit returns to |0>, and the accumulated phase equals
`(-1) ** sum_j popcount(e_j & M_j)` for the *planted* error word.

Difference from `verify()`: shots are generated once and shared by every
candidate, and the `eps_plus < 1/16` assertion is not raised -- raw
`value_errors` / `phase_errors` are returned instead, because a 1k-shot local
run cannot reach the 102,400-shot statistical bound anyway.

Usage
-----
    python src/e4/semantic.py --variant ref --m 280 --shots 1024
    python src/e4/semantic.py --variant ref --variant logm_minus1 --m 280
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'dqi'))

import verify_submission_dqi as V  # noqa: E402  (the shipped verifier, imported as-is)
from qoptim_variants import load_variant  # noqa: E402


def make_shots(spec: V.ProblemSpecDQI, tables: V.FieldTables, seed: int,
               num_shots: int, mode: str = 'official') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates `num_shots` DQI test cases with the official sampler.

    `mode='official'` uses the leaderboard weight distribution.  `mode='maxweight'`
    pins the error weight to the largest correctable value `k = n // 2`, which is
    the regime that exercises the EEA hardest (largest degrees, longest schedule)
    and is the right place to look for a schedule-dependent defect.
    """
    rng = np.random.default_rng(seed)
    weight_probs = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    if mode == 'official':
        return V.sample_shots(spec, tables, rng, num_shots, weight_probs)
    if mode != 'maxweight':
        raise ValueError(mode)
    k = spec.max_errors
    one_hot = np.zeros(len(weight_probs))
    one_hot[k] = 1.0
    return V.sample_shots(spec, tables, rng, num_shots, one_hot)


def check_circuit(circuit: km.Circuit, spec: V.ProblemSpecDQI,
                  shots: tuple[np.ndarray, np.ndarray, np.ndarray],
                  batch_size: int = 64, seed: int = 99) -> dict:
    """Runs the three-condition contract check on every shot."""
    syndromes, mbuc, phases = shots
    num_shots = len(syndromes)
    syndrome_qubits, mbuc_bits, ancilla_qubits = V.validate_registers(circuit, spec)
    syndrome_array = km.array(syndrome_qubits)
    mbuc_array = km.array(mbuc_bits)

    value_errors = 0
    phase_errors = 0
    toffoli_total = 0
    for start in range(0, num_shots, batch_size):
        stop = start + batch_size
        sim = km.Simulator(batch_size=batch_size, debug_print=False,
                           counting_operations=True, seed=seed + start)
        sim.use_same_registers_as(circuit)
        input_syndromes = V.to_bits(syndromes[start:stop], spec.degree)
        sim.write_across_shots(syndrome_array, input_syndromes)
        sim.write_across_shots(mbuc_array, V.to_bits(mbuc[start:stop], spec.degree))
        sim.do(circuit)
        phase_flipped = sim.read_phase_flipped_across_shots()
        reg_diff = sim.read_across_shots(syndrome_array) ^ input_syndromes
        junk = (sim.read_across_shots(ancilla_qubits) if len(ancilla_qubits)
                else np.zeros((0, batch_size), dtype=bool))
        val_err = reg_diff.any(axis=0) | junk.any(axis=0)
        ph_err = (phase_flipped ^ phases[start:stop]) & ~val_err
        value_errors += int(val_err.sum())
        phase_errors += int(ph_err.sum())
        toffoli_total += int(sim.operation_statistics().num_toffoli_like)

    total = value_errors + phase_errors
    return {
        'num_shots': num_shots,
        'value_errors': value_errors,
        'phase_errors': phase_errors,
        'total_errors': total,
        'verdict': 'PASS' if total == 0 else 'FAIL',
        'observed_error_rate': total / num_shots,
        'mean_toffoli_per_shot': toffoli_total / num_shots,
    }


def run(variant: str, m: int, n: int, degree: int, num_shots: int,
        shot_seed: int, batch_size: int = 64, mode: str = 'official') -> dict:
    ns, note = load_variant(variant)
    spec = V.ProblemSpecDQI.benchmark_instance(m=m, n=n, degree=degree,
                                               num_shots=num_shots)
    tables = V.FieldTables(km.GF2Field(degree))
    shots = make_shots(spec, tables, shot_seed, num_shots, mode)

    t0 = time.time()
    try:
        circuit = ns['build_reference_circuit'](m, n, degree)
    except Exception as exc:  # noqa: BLE001
        return {'variant': variant, 'note': note, 'build': 'ERROR',
                'error': f'{type(exc).__name__}: {exc}'}
    t_build = time.time() - t0
    q = circuit.num_qubits
    t_static = circuit.operation_statistics().num_toffoli_like

    t1 = time.time()
    res = check_circuit(circuit, spec, shots, batch_size=batch_size)
    t_sim = time.time() - t1
    return {'variant': variant, 'note': note, 'mode': mode, 'build': 'OK', 'num_qubits': q,
            'static_toffoli': t_static, 'build_seconds': round(t_build, 2),
            'sim_seconds': round(t_sim, 2), **res}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', action='append', default=None)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--shot-seed', type=int, default=20261025)
    ap.add_argument('--mode', choices=('official', 'maxweight'), default='official')
    args = ap.parse_args()

    variants = args.variant or ['ref']
    for name in variants:
        r = run(name, args.m, args.n, args.degree, args.shots,
                args.shot_seed, args.batch_size, args.mode)
        print('=' * 78)
        print(f"variant   : {r['variant']}")
        print(f"note      : {r['note']}")
        if 'error' in r:
            print(f"BUILD     : {r['error']}")
            continue
        print(f"Q         : {r['num_qubits']}")
        print(f"T (static): {r['static_toffoli']:,}")
        print(f"build     : {r['build_seconds']} s   sim: {r['sim_seconds']} s")
        print(f"mode      : {r['mode']}")
        print(f"shots     : {r['num_shots']}")
        print(f"errors    : value={r['value_errors']} phase={r['phase_errors']} "
              f"total={r['total_errors']}")
        print(f"VERDICT   : {r['verdict']}")


if __name__ == '__main__':
    main()

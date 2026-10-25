"""E6-D, path (a): can the disputed predicate be replaced by a structurally
constant decision?

Three variants of the **shipped** decoder (only line 270 changes; the uncompute
pair around it is kept symmetric, so `aux` is still cleanly restored):

    orig    : flag ^= aux AND (reg_b[m-2] == 0)          (shipped)
    delete  : the term is removed            -> predicate identically FALSE
    force   : flag ^= aux                    -> predicate identically TRUE

Both replacements are *structurally constant*: no data-dependent comparison is
left, so a compile-time schedule becomes exact by construction.  The question is
what they cost.  Everything is measured with the official verifier on the same
shots (`verify_submission_dqi.verify(..., shots=...)`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'dqi'))

import kickmix as km                                              # noqa: E402
import verify_submission_dqi as V                                 # noqa: E402

SRC = (ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py').read_text()
TERM = '            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)'
REPL = {
    'orig': TERM,
    'delete': '            pass  # [e6d path a] predicate identically FALSE',
    'force': '            builder.cx(aux, flag)  # [e6d path a] predicate identically TRUE',
}


def build(kind: str, m: int, n: int, degree: int) -> km.Circuit:
    assert TERM in SRC, 'term anchor moved'
    src = SRC.replace(TERM, REPL[kind], 1)
    ns = {'__name__': f'e6dpa_{kind}'}
    exec(compile(src, f'<e6dpa:{kind}>', 'exec'), ns)
    return ns['build_reference_circuit'](m, n, degree)


def detailed_check(circuit, spec, shots, batch_size=256, sim_seed=12345):
    """Split the official `value_errors` into syndrome-restore vs junk-ancilla."""
    syn_q, mbuc_b, anc = V.validate_registers(circuit, spec)
    syn_a, mbuc_a = km.array(syn_q), km.array(mbuc_b)
    syndromes, mbuc, phases = shots
    n = len(syndromes)
    tot = {'syndrome': 0, 'junk': 0, 'phase': 0}
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True, seed=sim_seed)
        sim.use_same_registers_as(circuit)
        ib = V.to_bits(syndromes[start:stop], spec.degree)
        sim.write_across_shots(syn_a, ib)
        sim.write_across_shots(mbuc_a, V.to_bits(mbuc[start:stop], spec.degree))
        sim.do(circuit)
        pf = sim.read_phase_flipped_across_shots()
        rd = sim.read_across_shots(syn_a) ^ ib
        jk = sim.read_across_shots(anc) if len(anc) else np.zeros((0, stop - start), dtype=bool)
        s_err = rd.any(axis=0); j_err = jk.any(axis=0)
        tot['syndrome'] += int(s_err.sum())
        tot['junk'] += int(j_err.sum())
        tot['phase'] += int(((pf ^ phases[start:stop]) & ~(s_err | j_err)).sum())
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=63)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=32768)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    spec = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=args.n, degree=args.degree,
                                               num_shots=args.shots)
    tables = V.FieldTables(km.GF2Field(args.degree))
    rng = np.random.default_rng(args.seed)
    wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    shots = V.sample_shots(spec, tables, rng, args.shots, wp)
    lead_zero = float((shots[0][:, -1] == 0).mean())
    print(f'spec m={spec.m} n={spec.n} degree={args.degree} shots={args.shots}; '
          f'P(S[n-1]==0)={lead_zero:.5f}', flush=True)

    out = {'spec': {'m': spec.m, 'n': spec.n, 'degree': args.degree},
           'shots': args.shots, 'P_lead_zero': lead_zero, 'variants': {}}
    for kind in ('orig', 'delete', 'force'):
        circ = build(kind, args.m, args.n, args.degree)
        T = circ.operation_statistics().num_toffoli_like
        det = detailed_check(circ, spec, shots, args.batch_size)
        print(f'[{kind:6s}] detail {det}', flush=True)
        try:
            res = V.verify(circ, spec, shots=shots, batch_size=args.batch_size, max_workers=1)
        except AssertionError as err:
            res = {'value_errors': det['syndrome'] + det['junk'] + det['phase'] or -1,
                   'phase_errors': det['phase'], 'empirical_error_rate': float('nan'),
                   'effective_error_rate': 1.0, 'score': float('nan'),
                   'rejected': str(err)[:160]}
        out['variants'][kind] = {
            'T': T, 'Q': circ.num_qubits,
            'value_errors': res['value_errors'], 'phase_errors': res['phase_errors'],
            'empirical_error_rate': res['empirical_error_rate'],
            'effective_error_rate': res['effective_error_rate'],
            'score': res['score'],
        }
        out['variants'][kind]['detail'] = det
        print(f'[{kind:6s}] T={T:,} Q={circ.num_qubits} '
              f'value_errors={res["value_errors"]} phase_errors={res["phase_errors"]} '
              f'({res["empirical_error_rate"]:.5f}) eps+={res["effective_error_rate"]:.5e}',
              flush=True)

    o = out['variants']['orig']
    for kind in ('delete', 'force'):
        v = out['variants'][kind]
        tot = v['value_errors'] + v['phase_errors']
        out['variants'][kind]['broken_fraction'] = tot / args.shots
        out['variants'][kind]['T_delta'] = v['T'] - o['T']
    print('\nsummary:')
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)


if __name__ == '__main__':
    main()

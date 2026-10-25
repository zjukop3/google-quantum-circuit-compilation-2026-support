"""E6-D: diagnose the 70% -- which qubits actually end up dirty?

`self_cancel` inserts the *shipped* term twice in a row (a logical no-op) and
still "breaks" 70.76% of shots, exactly like deleting it.  That can only be a
bookkeeping/compile-time effect, so look at the raw dirty-qubit footprints.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'dqi'))
sys.path.insert(0, str(HERE))

import kickmix as km                                              # noqa: E402
import verify_submission_dqi as V                                 # noqa: E402
import e6d_dummy_term as D                                        # noqa: E402


def run_detailed(circuit, spec, shots, batch_size=256, sim_seed=99):
    syn_q, mbuc_b, anc = V.validate_registers(circuit, spec)
    syn_a, mbuc_a = km.array(syn_q), km.array(mbuc_b)
    syndromes, mbuc, phases = shots
    n = len(syndromes)
    out = {}
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True, seed=sim_seed)
        sim.use_same_registers_as(circuit)
        ib = V.to_bits(syndromes[start:stop], spec.degree)
        sim.write_across_shots(syn_a, ib)
        sim.write_across_shots(mbuc_a, V.to_bits(mbuc[start:stop], spec.degree))
        sim.do(circuit)
        pf = np.asarray(sim.read_phase_flipped_across_shots()).astype(bool)
        rd = np.asarray(sim.read_across_shots(syn_a)).astype(bool) ^ np.asarray(ib)
        jk = np.asarray(sim.read_across_shots(anc)).astype(bool) if len(anc) else np.zeros((0, stop - start), dtype=bool)
        out[start] = {'phase_flipped': pf, 'syn_diff': rd, 'junk': jk,
                      'phases': phases[start:stop], 'anc_map': list(anc)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=63)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=256)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    spec = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=args.n, degree=args.degree,
                                               num_shots=args.shots)
    tables = V.FieldTables(km.GF2Field(args.degree))
    rng = np.random.default_rng(20261025)
    wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    shots = V.sample_shots(spec, tables, rng, args.shots)

    res = {'spec': {'m': args.m, 'n': args.n}, 'shots': args.shots, 'variants': {}}
    for kind in ('orig', 'delete', 'self_cancel'):
        circ = D.build(kind, args.n, args.degree, args.m)
        det = run_detailed(circ, spec, shots)
        pf = np.concatenate([v['phase_flipped'] for v in det.values()])
        rd = np.concatenate([v['syn_diff'] for v in det.values()], axis=1)
        jk = np.concatenate([v['junk'] for v in det.values()], axis=1)
        ph = np.concatenate([v['phases'] for v in det.values()])
        anc = [q.id if hasattr(q, 'id') else int(q) for q in det[0]['anc_map']]
        syn_bad = rd.any(axis=0)
        junk_bad = jk.any(axis=0)
        dirty_idx = Counter()
        for s in np.nonzero(junk_bad)[0][:200]:
            for k in np.nonzero(jk[:, s])[0]:
                dirty_idx[int(anc[k])] += 1
        res['variants'][kind] = {
            'syndrome_bad': int(syn_bad.sum()),
            'junk_bad': int(junk_bad.sum()),
            'phase_bad': int(((pf ^ ph) & ~(syn_bad | junk_bad)).sum()),
            'n_dirty_qubits_per_shot_mean': float(jk.sum(axis=0)[junk_bad].mean()) if junk_bad.any() else 0.0,
            'dirty_qubit_index_hist': dict(sorted(dirty_idx.items())[:15]),
            'n_distinct_dirty_qubits': len(dirty_idx),
        }
        print(kind, json.dumps(res['variants'][kind]), flush=True)
        if kind == 'orig':
            res['example_orig'] = {'pf_head': [int(x) for x in pf[:16]],
                                   'ph_head': [int(x) for x in ph[:16]]}
        else:
            res[f'example_{kind}'] = {
                'pf_head': [int(x) for x in pf[:16]],
                'ph_head': [int(x) for x in ph[:16]],
                'syn_bad_mask_head': [int(x) for x in syn_bad[:16]],
                'junk_bad_mask_head': [int(x) for x in junk_bad[:16]],
            }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2))
        print('wrote', args.json)


if __name__ == '__main__':
    main()

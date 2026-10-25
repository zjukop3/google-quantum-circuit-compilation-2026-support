"""E6-D: is the term load-bearing, or is deleting it a compile-time artifact?

Facts so far (n=6, full shipped decoder, official shot distribution):
  * shipped decoder: 0 errors / 16384 shots
  * the disputed term fires on 3/2048 shots in the forward pass and on the *same*
    3 shots in the adjoint pass (identical fire statistics)
  * deleting the term in one pass, the other, or both -> 70.76% of shots break

A branch that fires on 0.15% of shots cannot logically break 70% of them.  So we
compare three *semantically equivalent-to-nothing* edits that differ in how much
they perturb the generated gate list:

  delete      : the instruction is removed (gate list shorter, allocation moves)
  dummy       : same predicate, same control, target moved to a fresh scratch
                qubit that is allocated and freed on the spot (gate list intact)
  retarget    : predicate/control kept, target = a fresh qubit that stays alive
                until the end of the method (allocation pattern kept)

If `dummy`/`retarget` break ~0.15% while `delete` breaks 70%, the 70% is a
compile-time artifact of the edit, not a property of the branch.
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

NOOP = '            builder.x(flag)\n            builder.x(flag)'
MARK = "        with builder.mark('update_flag'):"
TERM2 = '            builder.flip_if_equal(reg_b[0], 0, target=flag, control=aux)'

VARIANTS = {
    'orig': TERM,
    'delete': '            pass  # removed',
    'dummy': ('            _scratch = builder.alloc_qubits(1)[0]\n'
              '            builder.flip_if_equal(reg_b[m - 2], 0, target=_scratch, '
              'control=aux)\n'
              '            builder.free(_scratch)'),
    'self_cancel': ('            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, '
                    'control=aux)\n'
                    '            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, '
                    'control=aux)'),
    'noop_x': ('            builder.x(flag)\n'
               '            builder.x(flag)'),
    'noop_scratch': ('            _np = builder.alloc_qubits(1)[0]\n'
                     '            builder.x(_np)\n'
                     '            builder.x(_np)\n'
                     '            builder.free(_np)'),
    'dup_after': (TERM + '\n' + TERM),
    'noop_early': 'x',
    'noop_after_term': 'x',
    'delete_c2': '            pass',
    'delete_both': '            pass',
    'noop_at_c2': '            x',
    'xcontrol': ('            _xor = builder.alloc_qubits(1)[0]\n'
                 '            builder.cx(aux, _xor)\n'
                 '            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, '
                 'control=_xor)\n'
                 '            builder.free(_xor)'),
}


REMOVED = '            pass  # term removed'
def build(kind, n, degree, m):
    src = SRC
    if kind == 'delete_c2':                      # keep c1, remove c2
        src = src.replace(TERM2, REMOVED, 1)
    elif kind == 'delete_both':
        src = src.replace(TERM, REMOVED, 1).replace(TERM2, REMOVED, 1)
    elif kind == 'noop_early':
        src = src.replace(MARK, MARK + '\n' + NOOP, 1)
    elif kind == 'noop_after_term':
        src = src.replace(TERM, TERM + '\n' + NOOP, 1)
    elif kind == 'noop_at_c2':
        src = src.replace(TERM2, NOOP, 1)
    else:
        src = src.replace(TERM, VARIANTS[kind], 1)
    ns = {'__name__': f'e6ddum_{kind}'}
    exec(compile(src, f'<e6ddum:{kind}>', 'exec'), ns)
    return ns['build_reference_circuit'](m, n, degree)


def check(circuit, spec, shots, batch_size=512, sim_seed=99):
    syn_q, mbuc_b, anc = V.validate_registers(circuit, spec)
    syn_a, mbuc_a = km.array(syn_q), km.array(mbuc_b)
    syndromes, mbuc, phases = shots
    n = len(syndromes)
    tot = {'syndrome': 0, 'junk': 0, 'phase': 0}
    broken = np.zeros(n, dtype=bool)
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
        s_err = rd.any(axis=0)
        j_err = jk.any(axis=0)
        tot['syndrome'] += int(s_err.sum())
        tot['junk'] += int(j_err.sum())
        tot['phase'] += int(((pf ^ phases[start:stop]) & ~(s_err | j_err)).sum())
        broken[start:stop] = s_err | j_err
    return tot, broken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=63)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=8192)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    spec = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=args.n, degree=args.degree,
                                               num_shots=args.shots)
    tables = V.FieldTables(km.GF2Field(args.degree))
    rng = np.random.default_rng(20261025)
    wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    shots = V.sample_shots(spec, tables, rng, args.shots)
    P_lead = float((shots[0][:, -1] == 0).mean())

    out = {'spec': {'m': args.m, 'n': args.n, 'degree': args.degree},
           'shots': args.shots, 'P_lead_zero': P_lead, 'variants': {}}
    broken_sets = {}
    for kind in VARIANTS:
        circ = build(kind, args.n, args.degree, args.m)
        T = circ.operation_statistics().num_toffoli_like
        tot, broken = check(circ, spec, shots)
        T_ref = out['variants'].get('orig', {}).get('T')
        out['variants'][kind] = {'T': T, 'Q': circ.num_qubits, 'detail': tot,
                                 'broken_shots': int(broken.sum()),
                                 'broken_fraction': float(broken.mean()),
                                 'T_delta': None if T_ref is None else T - T_ref}
        broken_sets[kind] = broken
        print(f'[{kind:12s}] T={T:,} dT={"" if T_ref is None else T - T_ref:>7} '
              f'{tot} broken={int(broken.sum())}/{args.shots} ({broken.mean():.5f})',
              flush=True)
    # overlaps
    o = broken_sets['orig']
    for kind in ('delete', 'delete_c2', 'delete_both', 'noop_at_c2', 'dummy', 'self_cancel', 'xcontrol'):
        if kind in broken_sets:
            b = broken_sets[kind]
            out['variants'][kind]['overlap_with_orig'] = int((b & o).sum())
            out['variants'][kind]['broken_but_orig_ok'] = int((b & ~o).sum())
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'detail'}
                      for k, v in out['variants'].items()}, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)


if __name__ == '__main__':
    main()

"""E6-D: which PASS does the disputed sub-step-1 term matter in?

At n=6 the shipped decoder is perfect (0 errors / 16384 shots) but deleting the
disputed term breaks 70% of them -- two orders of magnitude more than the
forward-pass branch rate measured with the metadata probe (1.0e-3).  The
forward EEA is a prefix of the decoder, so the extra damage has to come from the
reverse (adjoint) EEA pass, the Chien/Forney segment in between, or from the
fact that deleting a *conditional flip* is not the same as never firing it.

This script splits the question by deleting the term in one pass at a time.
`_update_flag` is called once per iteration per pass, so the build-time call
index identifies the pass: [0, n_iters) = forward, [n_iters, 2*n_iters) = adjoint.
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
sys.path.insert(0, str(HERE))

import kickmix as km                                              # noqa: E402
import verify_submission_dqi as V                                 # noqa: E402

SRC = (ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py').read_text()
TERM = '            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)'

GUARDED = '''            _k = self._ufi
            self._ufi += 1
            if not (self._drop_forward and _k < self._n_iters) and not (
                    self._drop_adjoint and _k >= self._n_iters):
                builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)
            if _k < self._n_iters:
                self._pass_of[_k] = 0
            else:
                self._pass_of[_k] = 1
'''

INIT = '''    _ufi = 0
    _drop_forward = False
    _drop_adjoint = False
    _pass_of = {}
'''


def build(n, degree, m, drop_forward=False, drop_adjoint=False):
    src = SRC.replace(TERM, GUARDED.strip('\n'), 1)
    src = src.replace('class ProosAndZalkaDecoder:', INIT + '\n\nclass ProosAndZalkaDecoder:', 1)
    ns = {'__name__': 'e6dpass'}
    exec(compile(src, '<e6dpass>', 'exec'), ns)
    cls = ns['ProosAndZalkaDecoder']
    cls._n_iters = 3 * (n + 1) + 9
    cls._drop_forward = drop_forward
    cls._drop_adjoint = drop_adjoint
    cls._ufi = 0
    cls._pass_of = {}
    b = km.CircuitBuilder()
    syn = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    err = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    cls(n, km.GF2Field(degree)).build_syndrome_decoding_mbuc(
        b, syndromes=syn, error_mbuc=err, n_iters=cls._n_iters)
    circ = b.finish_circuit()
    calls = dict(cls._pass_of)
    return circ, calls


def check(circuit, spec, shots, batch_size=512, sim_seed=99):
    syn_q, mbuc_b, anc = V.validate_registers(circuit, spec)
    syn_a, mbuc_a = km.array(syn_q), km.array(mbuc_b)
    syndromes, mbuc, phases = shots
    n = len(syndromes)
    tot = {'syndrome': 0, 'junk': 0, 'phase': 0}
    per_shot = np.zeros(n, dtype=bool)
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
        per_shot[start:stop] = s_err | j_err
    return tot, per_shot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=63)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=16384)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    spec = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=args.n, degree=args.degree,
                                               num_shots=args.shots)
    tables = V.FieldTables(km.GF2Field(args.degree))
    rng = np.random.default_rng(20261025)
    wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    shots = V.sample_shots(spec, tables, rng, args.shots)

    out = {'spec': {'m': args.m, 'n': args.n, 'degree': args.degree},
           'shots': args.shots, 'variants': {}}
    for label, kw in (('orig', {}),
                      ('drop_forward', {'drop_forward': True}),
                      ('drop_adjoint', {'drop_adjoint': True}),
                      ('drop_both', {'drop_forward': True, 'drop_adjoint': True})):
        circ, calls = build(args.n, args.degree, args.m, **kw)
        nf = sum(1 for v in calls.values() if v == 0)
        na = len(calls) - nf
        tot, per_shot = check(circ, spec, shots)
        T = circ.operation_statistics().num_toffoli_like
        out['variants'][label] = {'T': T, 'Q': circ.num_qubits, 'detail': tot,
                                  'uf_calls_forward': nf, 'uf_calls_adjoint': na,
                                  'broken_shots': int(per_shot.sum()),
                                  'broken_fraction': float(per_shot.mean())}
        print(f'[{label:13s}] T={T:,} update_flag calls fwd={nf} adj={na} {tot} '
              f'broken={int(per_shot.sum())}/{args.shots} '
              f'({per_shot.mean():.5f})', flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)


if __name__ == '__main__':
    main()

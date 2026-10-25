"""E6 experiment L: how often does the sub-step-1 extra termination term fire?

The term is

    aux   = c1 AND NOT(n_B == 1)
    term  = aux AND (reg_b[m-2] == 0)                # line 270 of the reference

and the E6 model predicted `term` on `#c1 = (n+2)/2` opportunities per shot with
per-opportunity probability exactly 2^-b, i.e. 8.79e-3 at n=70, b=12.

This measures the real rate *in the shipped module*, non-invasively: `term` is
computed into a fresh qubit (the same `flip_if_equal` gate with the target moved
off `flag`), a 6-bit binary counter is incremented by it, and the term qubit is
uncomputed.  Nothing the algorithm reads is touched, so the decoder's answer is
unchanged -- which the script asserts by re-running the official verifier.

It also counts the number of sub-step-1 *opportunities* -- `aux = c1 AND
NOT(n_B == 1)`, the region where the term can fire at all -- so the
per-opportunity probability follows directly.

Usage
-----
    python src/e6/e6_deviation_rate.py --n 70 --degree 12 --m 280 --shots 2048
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import e6_semantics as S  # noqa: E402

ORIG = 'builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)'
COUNTER_BITS = 6


def instrumented_update_flag(ref):
    """`_update_flag` plus two non-invasive counters (term firings, c1 count)."""
    src = inspect.getsource(ref.ProosAndZalkaDecoder._update_flag)
    assert src.count(ORIG) == 1
    probe = (
        'if getattr(self, "_e6count", None) is not None:\n'
        '    _t = builder.alloc_qubits(1)[0]\n'
        f'    {ORIG.replace("target=flag", "target=_t")}\n'
        '    builder.iadd(1, target=self._e6count[0], control=_t)\n'
        f'    {ORIG.replace("target=flag", "target=_t")}\n'
        '    builder.free(_t)\n'
        '    builder.iadd(1, target=self._e6count[1], control=aux)\n'
        + ORIG
    )
    pad = ' ' * 12
    probe = ('\n' + pad).join(probe.split('\n'))
    src = src.replace(pad + ORIG, pad + probe, 1)
    src = '\n'.join(l[4:] if l.startswith('    ') else l for l in src.split('\n'))
    ns = {}
    exec(compile(src, '<counted>', 'exec'), vars(ref), ns)
    return ns['_update_flag']


def build_counted(ref, m, n, degree):
    decoder = ref.ProosAndZalkaDecoder(n, km.GF2Field(degree))
    decoder._update_flag = instrumented_update_flag(ref).__get__(decoder, type(decoder))
    b = km.CircuitBuilder()
    syn = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    err = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    regs = [[b.create_quantum_register(COUNTER_BITS, name=f'CT{j}')
             for j in range(2)]]
    decoder._e6count = regs[0]
    probe_qs = [[q for q in r] for r in regs[0]]
    decoder.build_syndrome_decoding_mbuc(b, syndromes=syn, error_mbuc=err,
                                         n_iters=3 * (n + 1) + 9)
    return b.finish_circuit(), probe_qs, syn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--shots', type=int, default=2048)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--check-shots', type=int, default=256)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    m, n, degree = args.m, args.n, args.degree

    V = S.load_module('vdqi', ROOT / 'data' / 'verifiers' / 'dqi'
                      / 'verify_submission_dqi.py')
    ref = S.load_module('refdqi', ROOT / 'data' / 'verifiers' / 'dqi'
                        / 'dqi_reference_circuit.py')

    spec = V.ProblemSpecDQI.benchmark_instance(m=m, n=n, degree=degree,
                                               num_shots=args.shots)
    tables = V.FieldTables(km.GF2Field(degree))
    rng = np.random.default_rng(20261025)
    wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    shots = V.sample_shots(spec, tables, rng, args.shots, wp)
    bits = V.to_bits(shots[0], degree)

    circ, probe_qs, syn = build_counted(ref, m, n, degree)
    T = circ.operation_statistics().num_toffoli_like
    print(f'[counted] Q={circ.num_qubits} T={T:,}', flush=True)

    terms = np.zeros(args.shots, dtype=np.int64)
    opps = np.zeros(args.shots, dtype=np.int64)
    for start in range(0, args.shots, args.batch_size):
        stop = min(start + args.batch_size, args.shots)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True,
                           seed=7 + start)
        sim.use_same_registers_as(circ)
        sim.write_across_shots(km.array([q for r in syn for q in r]),
                               bits[:, start:stop])
        sim.do(circ)
        for k, q in enumerate(probe_qs[0]):
            terms[start:stop] |= (np.asarray(sim.read_across_shots(q)).astype(np.int64) << k)
        for k, q in enumerate(probe_qs[1]):
            opps[start:stop] |= (np.asarray(sim.read_across_shots(q)).astype(np.int64) << k)

    print('\n== L. sub-step-1 extra term, shipped module ==============')
    print(f'  shots {args.shots}; term firings mean {terms.mean():.4f} '
          f'max {int(terms.max())}; aux opportunities mean {opps.mean():.3f} '
          f'(min {int(opps.min())}, max {int(opps.max())})')
    fired = terms > 0
    print(f'  P(at least one firing) = {fired.mean():.6e} '
          f'({int(fired.sum())}/{args.shots})')
    with np.errstate(divide='ignore', invalid='ignore'):
        per_opp = terms.sum() / max(int(opps.sum()), 1)
    print(f'  empirical per-opportunity probability = {per_opp:.6e}  '
          f'(2^-{degree} = {2.0 ** -degree:.6e})')
    if args.json:
        Path(args.json).write_text(json.dumps({
            'n': n, 'degree': degree, 'm': m, 'shots': args.shots,
            'mean_term_firings': float(terms.mean()),
            'max_term_firings': int(terms.max()),
            'mean_c1_opportunities': float(opps.mean()),
            'p_at_least_one_firing': float(fired.mean()),
            'per_opportunity_probability': float(per_opp),
            'two_pow_minus_degree': 2.0 ** -degree,
            'counted_toffoli': T,
        }, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

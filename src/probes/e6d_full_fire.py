"""E6-D: ground truth on the disputed branch -- fires per call, per pass.

`_update_flag` is called once per iteration per pass, so the build-time call
index identifies (pass, iteration).  For every call we fan out, immediately
before and after line 270,

    aux, flag_before, flag_after, reg_b[m-2]

onto fresh ancillas, and afterwards reconstruct per-call firing statistics.
This is the only measurement that settles "what is the 2^-12 a 2^-12 of".
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

import kickmix as km                                              # noqa: E402

SRC = (ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py').read_text()
TERM = '            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)'

INSTR = '''            _k = self._ufi
            self._ufi += 1
            if _k < self._pN:
                builder.cx(km.array([aux]), self._pr['aux'][_k])
                builder.cx(km.array([flag]), self._pr['f0'][_k])
                builder.cx(reg_b[m - 2], self._pr['rb2'][_k])
            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)
            if _k < self._pN:
                builder.cx(km.array([flag]), self._pr['f1'][_k])
'''

INIT = """    _ufi = 0
    _pN = 0
    _pr = None
"""

METHOD = """    def _alloc_call_probe(self, builder, n_calls, degree):
        def q(width, name, i):
            return builder.create_quantum_register(width, name=f'{name}{i}')
        self._pN = n_calls
        self._pr = {'aux': [q(1, 'Paux', i) for i in range(n_calls)],
                    'f0': [q(1, 'Pf0', i) for i in range(n_calls)],
                    'f1': [q(1, 'Pf1', i) for i in range(n_calls)],
                    'rb2': [q(degree, 'Prb', i) for i in range(n_calls)]}
"""


def build(n, degree, m, num_calls):
    src = SRC.replace(TERM, INSTR.strip('\n'), 1)
    src = src.replace('class ProosAndZalkaDecoder:', INIT + '\n\nclass ProosAndZalkaDecoder:', 1)
    src = src.replace('    def __init__(self, n_syndromes: int', METHOD + '\n    def __init__(self, n_syndromes: int', 1)
    ns = {'__name__': 'e6dfire'}
    exec(compile(src, '<e6dfire>', 'exec'), ns)
    cls = ns['ProosAndZalkaDecoder']
    cls._ufi = 0
    b = km.CircuitBuilder()
    syn = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    err = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    dec = cls(n, km.GF2Field(degree))
    dec._alloc_call_probe(b, num_calls, degree)
    dec.build_syndrome_decoding_mbuc(b, syndromes=syn, error_mbuc=err,
                                     n_iters=3 * (n + 1) + 9)
    return b.finish_circuit(), dec, syn


def run(circuit, dec, syn, data, degree, n_calls, seed=5, n_iters=None):
    shots = data.shape[0]
    sim = km.Simulator(batch_size=shots, counting_operations=True, seed=seed)
    sim.use_same_registers_as(circuit)
    rows = []
    for reg in range(data.shape[1]):
        for b in range(degree):
            rows.append(((data[:, reg] >> b) & 1).astype(bool))
    sim.write_across_shots(km.array([q for r in syn for q in r]), np.array(rows, dtype=bool))
    sim.do(circuit)
    out = {}
    for key in ('aux', 'f0', 'f1', 'rb2'):
        w = len(dec._pr[key][0])
        flat = km.array([q for i in range(n_calls) for q in dec._pr[key][i]])
        bits = sim.read_across_shots(flat)
        wide = np.zeros((n_calls * w, shots), dtype=np.uint16)
        for k in range(min(len(bits), n_calls * w)):
            wide[k] = np.asarray(bits[k]).astype(np.uint16)
        wide = wide.reshape(n_calls, w, shots)
        v = np.zeros((n_calls, shots), dtype=np.uint16)
        for bit in range(w):
            v |= wide[:, bit, :] << bit
        out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=63)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    n_iters = 3 * (args.n + 1) + 9
    n_calls = 2 * n_iters
    circ, dec, syn = build(args.n, args.degree, args.m, n_calls)
    used = dec._ufi
    print(f'[build] T={circ.operation_statistics().num_toffoli_like:,} '
          f'Q={circ.num_qubits:,} update_flag calls={used} (n_iters={n_iters})', flush=True)
    cut = n_iters
    n_calls = used

    rng = np.random.default_rng(11)
    full = (1 << args.degree) - 1
    data = rng.integers(0, full + 1, size=(args.shots, args.n)).astype(np.int64)
    P = run(circ, dec, syn, data, args.degree, n_calls, seed=5)
    flip = P['f0'] != P['f1']
    aux = P['aux'].astype(bool)
    rb2 = P['rb2']

    res = {'n': args.n, 'degree': args.degree, 'shots': args.shots,
           'n_iters': n_iters, 'calls': n_calls}
    for label, sl in (('forward', slice(0, cut)), ('adjoint', slice(cut, n_calls))):
        f = flip[sl]
        a = aux[sl]
        z = (rb2[sl] == 0)
        n_slot = f.size
        res[label] = {
            'calls': int(f.shape[0]),
            'opps': int(n_slot),
            'aux_high': int(a.sum()),
            'P_aux': float(a.mean()),
            'fires': int(f.sum()),
            'P_fire_per_opp': float(f.mean()),
            'P_rb2_zero_given_aux': float(z[a].mean()) if a.any() else None,
            'shots_with_fire': int(f.any(axis=0).sum()),
            'P_shot_with_fire': float(f.any(axis=0).mean()),
            'flip_without_aux': int((f & ~a).sum()),
            'aux_high_no_flip': int((a & ~f).sum()),
            'two_pow_minus_degree': 2.0 ** -args.degree,
        }
        print(label, json.dumps(res[label], indent=2), flush=True)
    # where (mod 6) do fires happen?
    for label, sl in (('forward', slice(0, cut)), ('adjoint', slice(cut, n_calls))):
        idx = np.nonzero(flip[sl].any(axis=1))[0]
        res[label]['fire_call_mod6'] = dict(sorted(Counter((idx % 6).tolist()).items()))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2))
        print('wrote', args.json)


if __name__ == '__main__':
    main()

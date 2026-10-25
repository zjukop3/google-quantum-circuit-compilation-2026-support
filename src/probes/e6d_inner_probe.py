"""E6-D: inner probe -- what does the disputed term see and do, gate by gate?

`e6d_schedule.py` says the predicate
    c1 & (n_B != 1) & (reg_b[m-2] == 0)
is true on 1.1% of forced-generic shots (n=6), while the *effective* rate of the
disputed term (measured by building orig vs delete) is 0.073% -- a factor 15.
So the two disagree about which shots fire.

This probe fan-outs, immediately around line 270 of the shipped module,

    aux, flag_before, reg_b[m-2] (at the check)  and  flag_after

one fresh ancilla per iteration.  Everything else in the circuit is untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e6d_schedule as E                                          # noqa: E402

import kickmix as km                                              # noqa: E402

TERM = '            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)'

INNER = '''            _i = self._uf_i
            self._uf_i += 1
            if _i < self._ip_len:
                builder.cx(km.array([aux]), self._ip['aux'][_i])
                builder.cx(km.array([flag]), self._ip['f0'][_i])
                builder.cx(reg_b[m - 2], self._ip['rb2'][_i])
            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)
            if _i < self._ip_len:
                builder.cx(km.array([flag]), self._ip['f1'][_i])
'''

ALLOC = '''    def _alloc_inner_probe(self, builder, n_iters, deg):
        def q(width, name, it):
            return builder.create_quantum_register(width, name=f'{name}{it}')
        self._ip_len = n_iters
        self._uf_i = 0
        self._ip = {
            'aux': [q(1, 'Iaux', it) for it in range(n_iters)],
            'f0': [q(1, 'If0', it) for it in range(n_iters)],
            'f1': [q(1, 'If1', it) for it in range(n_iters)],
            'rb2': [q(deg, 'Irb2', it) for it in range(n_iters)],
        }

'''


def build_source() -> str:
    src = E.build_probe_source()
    assert TERM in src, 'term anchor moved'
    src = src.replace(TERM, INNER.strip('\n'), 1)
    src = src.replace(E.WRAP_ANCHOR, ALLOC + E.WRAP_ANCHOR, 1)
    # allocate the inner probe inside the (already patched) wrapper
    src = src.replace(
        "        if getattr(self, 'trace_probe_enabled', True):\n"
        "            self._alloc_trace_probe(builder, n_iters, m_, logm, n_iters.bit_length())",
        "        if getattr(self, 'trace_probe_enabled', True):\n"
        "            self._alloc_trace_probe(builder, n_iters, m_, logm, n_iters.bit_length())\n"
        "        self._alloc_inner_probe(builder, n_iters, self.field.degree)", 1)
    return src


def load():
    ns = {'__name__': 'e6dinner'}
    exec(compile(build_source(), '<e6dinner>', 'exec'), ns)
    return ns


IFIELDS = ('aux', 'f0', 'f1', 'rb2')
IWIDTHS = (1, 1, 1, 12)


def run(circuit, dec, syn, data, degree, seed, n_iters):
    shots = data.shape[0]
    sim = km.Simulator(batch_size=shots, counting_operations=True, seed=seed)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(km.array([q for r in syn for q in r]), E.bits_of(data, degree))
    sim.do(circuit)
    ip = dec._ip
    out = np.zeros((n_iters, shots, len(IFIELDS)), dtype=np.uint16)
    for j, name in enumerate(IFIELDS):
        w = len(ip[name][0])
        flat = km.array([q for it in range(n_iters) for q in ip[name][it]])
        bits = sim.read_across_shots(flat)
        wide = np.zeros((n_iters * w, shots), dtype=np.uint16)
        for k in range(min(len(bits), n_iters * w)):
            wide[k] = np.asarray(bits[k]).astype(np.uint16)
        wide = wide.reshape(n_iters, w, shots)
        for bit in range(w):
            out[:, :, j] |= wide[:, bit, :] << bit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    n, deg = args.n, args.degree

    rng = np.random.default_rng(args.seed)
    data = E.gen_data('generic', n, deg, args.shots, rng)

    ns = load()
    circ, dec, ni, me, syn = E.build(n, deg, n, ns)
    print(f'[build] T={circ.operation_statistics().num_toffoli_like:,} Q={circ.num_qubits:,} '
          f'n_iters={ni}', flush=True)
    I = run(circ, dec, syn, data, deg, 5, ni)
    T = E.run_batch(circ, dec, syn, data, deg, seed=5, n_iters=ni)

    idx = {f: i for i, f in enumerate(E.FIELDS)}
    c1, aux_m, pred_m, fire_m, nB, rb2_m = E.derived(T)
    aux_c = I[:, :, 0].astype(bool)          # true aux at the check
    f0 = I[:, :, 1].astype(bool)
    f1 = I[:, :, 2].astype(bool)
    rb2_c = I[:, :, 3]

    active = c1 & aux_c
    print(f'shots             {args.shots}')
    print(f'c1-iterations     {int(c1.sum())}  (per shot {c1.sum()/args.shots:.3f})')
    print(f'aux_c TRUE at c1  {int(active.sum())}  '
          f'(P(aux_c|per c1 opp) {active.sum()/c1.sum():.5f})')
    print(f'P(rb2_at_check==0 | aux active) = '
          f'{(rb2_c[active] == 0).sum()}/{int(active.sum())} = '
          f'{(rb2_c[active] == 0).mean():.6f}   (2^-12={2.0**-deg:.3e})')
    print(f'flag flipped (f0!=f1) at active  {int((active & (f0 != f1)).sum())}')
    print(f'flag flipped anywhere            {int((f0 != f1).sum())}')
    print(f'P(rb2_at_start==0 | aux active)  {(rb2_m[active] == 0).mean():.6f}')
    # how often do start and check values of reg_b[m-2] differ?
    print(f'P(rb2_start != rb2_check)        {(rb2_m != rb2_c).mean():.6f}')

    pat = Counter()
    for s in range(args.shots):
        it = np.nonzero(active[:, s])[0]
        if len(it):
            for i in it:
                pat[(int(rb2_c[i, s] == 0), int(f0[i, s] != f1[i, s]))] += 1
    print('(rb2_check_zero, flag_flipped) counts:', dict(pat))

    out = {
        'n': n, 'degree': deg, 'shots': args.shots,
        'c1_iterations': int(c1.sum()),
        'aux_true_at_check': int(active.sum()),
        'P_aux_given_c1': float(active.sum() / c1.sum()),
        'P_rb2_check_zero_given_aux': float((rb2_c[active] == 0).mean()),
        'P_rb2_start_zero_given_aux': float((rb2_m[active] == 0).mean()),
        'P_rb2_start_neq_check': float((rb2_m != rb2_c).mean()),
        'flag_flips_at_active': int((active & (f0 != f1)).sum()),
        'two_pow_minus_degree': 2.0 ** -deg,
        'joint_hist': {str(k): int(v) for k, v in pat.items()},
        'rb2_check_support_at_aux': int(len(np.unique(rb2_c[active]))),
        'rb2_start_support_at_aux': int(len(np.unique(rb2_m[active]))),
    }
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)


if __name__ == '__main__':
    main()

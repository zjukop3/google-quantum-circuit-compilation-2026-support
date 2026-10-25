"""E6-D: where in the iteration does the disputed check actually run?

The metadata probe (start of iteration) and the inner probe (line 270) disagree
about `reg_b[m-2]` on 90% of the iterations, so one of the two probe positions
does not mean what it looks like.  This probe reads `reg_b[m-2]` at five
program points of every iteration:

    p0  right after `_counter_controls`            ("start of iteration")
    p1  right after the disputed term (inside `_update_flag`)
    p2  right after `_multiply_and_add_or_sub`
    p3  right after `_shift_b_array_right`
    p4  at the very end of the iteration (after `_update_counter`)

If p4[it] == p0[it+1] everywhere, the probe positions are consistent and the
mapping program-point -> iteration index is what it looks like; then any p0/p1
difference is a real ordering fact about the executed circuit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e6d_schedule as E                                          # noqa: E402
import kickmix as km                                              # noqa: E402

WRAP = '''
    def _alloc_order_probe(self, builder, n_iters, deg):
        self._p_len = n_iters
        self._p = {k: [builder.create_quantum_register(deg, name=f'{k}{it}')
                       for it in range(n_iters)] for k in ('p0', 'p1', 'p2', 'p3', 'p4')}

    def _probe_rb2(self, builder, reg_b, it, key):
        if it < self._p_len:
            builder.cx(reg_b[len(reg_b) - 2], self._p[key][it])

'''

MULT = """                    lambda: self._multiply_and_add_or_sub(builder, reg_a, reg_b, n_A, n_a,
                                                          n_q, ctrl, it,
                                                          is_adjoint=is_adjoint),"""
MULT_NEW = """                    lambda: (self._multiply_and_add_or_sub(builder, reg_a, reg_b, n_A, n_a,
                                                           n_q, ctrl, it,
                                                           is_adjoint=is_adjoint),
                             self._probe_rb2(builder, reg_b, it, 'p2')),"""

SHIFT = """                    lambda: self._shift_b_array_right(builder, reg_b, n_B, n_b, flag, ctrl,
                                                      it, is_adjoint=is_adjoint),"""
SHIFT_NEW = """                    lambda: (self._shift_b_array_right(builder, reg_b, n_B, n_b, flag, ctrl,
                                                       it, is_adjoint=is_adjoint),
                             self._probe_rb2(builder, reg_b, it, 'p3')),"""

END_ANCHOR = "                    self._update_counter(builder, counter, flag, is_adjoint=False)"
END_NEW = ("                    self._update_counter(builder, counter, flag, is_adjoint=False)\n"
           "                    self._probe_rb2(builder, reg_b, it, 'p4')")

TERM = '            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)'
TERM_NEW = ('            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)\n'
            "            self._probe_rb2(builder, reg_b, self._uf_i, 'p1')\n"
            '            self._uf_i += 1')


def build_source() -> str:
    src = E.build_probe_source()
    for a, b in ((MULT, MULT_NEW), (SHIFT, SHIFT_NEW), (END_ANCHOR, END_NEW),
                 (TERM, TERM_NEW)):
        assert a in src, a[:60]
        src = src.replace(a, b, 1)
    # start-of-iteration probe: reuse E6-D's trace-probe insertion point
    src = src.replace('                    builder.cx(reg_b[len(reg_b) - 2], _tk[\'rb2\'][it])',
                      '                    self._probe_rb2(builder, reg_b, it, \'p0\')', 1)
    src = src.replace(E.WRAP_ANCHOR, WRAP + E.WRAP_ANCHOR, 1)
    src = src.replace(
        "        self._alloc_trace_probe(builder, n_iters, m_, logm, n_iters.bit_length())",
        "        self._alloc_trace_probe(builder, n_iters, m_, logm, n_iters.bit_length())\n"
        "        self._alloc_order_probe(builder, n_iters, self.field.degree)\n"
        "        self._uf_i = 0", 1)
    return src


def load():
    ns = {'__name__': 'e6dorder'}
    exec(compile(build_source(), '<e6dorder>', 'exec'), ns)
    return ns


def run(circuit, dec, syn, data, degree, seed, n_iters):
    shots = data.shape[0]
    sim = km.Simulator(batch_size=shots, counting_operations=True, seed=seed)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(km.array([q for r in syn for q in r]), E.bits_of(data, degree))
    sim.do(circuit)
    out = {}
    for key in ('p0', 'p1', 'p2', 'p3', 'p4'):
        w = degree
        flat = km.array([q for it in range(n_iters) for q in dec._p[key][it]])
        bits = sim.read_across_shots(flat)
        wide = np.zeros((n_iters * w, shots), dtype=np.uint16)
        for k in range(min(len(bits), n_iters * w)):
            wide[k] = np.asarray(bits[k]).astype(np.uint16)
        wide = wide.reshape(n_iters, w, shots)
        v = np.zeros((n_iters, shots), dtype=np.uint16)
        for bit in range(w):
            v |= wide[:, bit, :] << bit
        out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=512)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    n, deg = args.n, args.degree
    rng = np.random.default_rng(args.seed)
    data = E.gen_data('generic', n, deg, args.shots, rng)
    ns = load()
    circ, dec, ni, me, syn = E.build(n, deg, n, ns)
    print(f'[build] T={circ.operation_statistics().num_toffoli_like:,} '
          f'Q={circ.num_qubits:,} n_iters={ni}', flush=True)
    P = run(circ, dec, syn, data, deg, 5, ni)
    T = E.run_batch(circ, dec, syn, data, deg, seed=5, n_iters=ni)
    idx = {f: i for i, f in enumerate(E.FIELDS)}

    stats = {}
    stats['p0_eq_trace_rb2'] = float((P['p0'] == T[:, :, idx['rb2']]).mean())
    stats['p4_it_eq_p0_it1'] = float((P['p4'][:-1] == P['p0'][1:]).mean())
    stats['p1_eq_p0'] = float((P['p1'] == P['p0']).mean())
    stats['p1_eq_p2'] = float((P['p1'] == P['p2']).mean())
    stats['p1_eq_p3'] = float((P['p1'] == P['p3']).mean())
    stats['p1_eq_p4'] = float((P['p1'] == P['p4']).mean())
    stats['p2_eq_p0'] = float((P['p2'] == P['p0']).mean())
    stats['p3_eq_p0'] = float((P['p3'] == P['p0']).mean())
    # zero rates (the whole point)
    for key in ('p0', 'p1', 'p2', 'p3', 'p4'):
        stats[f'zero_rate_{key}'] = float((P[key] == 0).mean())
        stats[f'support_{key}'] = int(len(np.unique(P[key])))
    c1 = ((T[:, :, idx['ctrl']] >> 1) & 1).astype(bool)
    for key in ('p1',):
        z = P[key][c1]
        stats['zero_rate_p1_at_c1'] = float((z == 0).mean())
        stats['n_c1'] = int(z.size)
    # conditional at the check: does p1==0 predict the flag flip at p1's term?
    print(json.dumps(stats, indent=2))
    print('\nper-iteration (shot 0):  it  p0   p1   p2   p3   p4   ctrl')
    for it in range(min(ni, 18)):
        print(f'   {it:2d}  {P["p0"][it,0]:5d} {P["p1"][it,0]:5d} {P["p2"][it,0]:5d} '
              f'{P["p3"][it,0]:5d} {P["p4"][it,0]:5d}   {int(T[it,0,idx["ctrl"]])}')
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(stats, indent=2))
        print('wrote', args.json)


if __name__ == '__main__':
    main()

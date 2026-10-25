"""What does left_rotate(x, shift=[False] + list(s)) actually compute, and what does it cost?"""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE))
import numpy as np
import kickmix as km


def bits_of(vs, w):
    return (vs[None, :] >> np.arange(w, dtype=np.uint64)[:, None]) & 1 == 1


def rot_left(v, k, n, cyc=True):
    m = (1 << n) - 1
    if cyc:
        return ((v << k) | (v >> (n - k))) & m
    return (v << k) & m


def run(n, s_bits, mk_shift, vals, svals):
    b = km.CircuitBuilder()
    x = b.create_quantum_register(n, 'x')
    s = b.create_quantum_register(s_bits, 's')
    mk_shift(b, x, s)
    c = b.finish_circuit()
    sim = km.Simulator(batch_size=len(vals), debug_print=False, seed=1)
    sim.clear_for_shot()
    sim.write_across_shots(x, bits_of(np.array(vals, dtype=np.uint64), n))
    sim.write_across_shots(s, bits_of(np.array(svals, dtype=np.uint64), s_bits))
    sim.do(c)
    return c, (sim.read_across_shots(x).astype(np.uint64) << np.arange(n, dtype=np.uint64)[:, None]).sum(axis=0)


n = 22
vals = np.array([0b1000000000000000000001, 0b1011000000000000000000, 0b111, 0b10000000000], dtype=np.uint64)
svals = np.array([1, 2, 3, 4], dtype=np.uint64)
m = (1 << n) - 1

for label, mk, T_exp in [
    ('shift=s (4b)', lambda b, x, s: b.left_rotate(x, shift=s), 51),
    ('shift=[False]+s (list)', lambda b, x, s: b.left_rotate(x, shift=[False] + list(s)), None),
    ('shift=s, control=true (explicit)', lambda b, x, s: b.left_rotate(x, shift=s, control=True), None),
]:
    c, out = run(n, 4, mk, vals, svals)
    T = c.operation_statistics().num_toffoli_like
    print(f'{label:<34s} T={T:>4d}')
    for name, fn in [('<<s', lambda v, k: (v << k) & m), ('<<2s', lambda v, k: (v << (2 * k)) & m),
                     ('rot<<s', lambda v, k: rot_left(v, k, n)), ('rot<<2s', lambda v, k: rot_left(v, 2 * k, n)),
                     ('rot>>s', lambda v, k: rot_left(v, (-k) % n, n))]:
        ok = all(int(o) == fn(int(v), int(k)) for o, v, k in zip(out, vals, svals))
        if ok:
            print(f'      MATCHES {name}')
    print('      got ', [hex(int(o)) for o in out])
    print('      <<2s', [hex(((int(v) << (2 * int(k))) & m)) for v, k in zip(vals, svals)])

# what does a bool in the middle do?
c, out = run(n, 4, lambda b, x, s: b.left_rotate(x, shift=list(s) + [False]), vals, svals)
print(f'shift=list(s)+[False]              T={c.operation_statistics().num_toffoli_like}')
print('      got ', [hex(int(o)) for o in out])

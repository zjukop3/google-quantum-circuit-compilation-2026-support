"""E5 probe: measured kickmix prices for the primitives a binary-`s` normalize/denormalize needs."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'invsqrt'))
import kickmix as km


def cost(fn):
    b = km.CircuitBuilder()
    fn(b)
    c = b.finish_circuit()
    st = c.operation_statistics()
    return dict(Q=c.num_qubits, T=st.num_toffoli_like, noncliff=st.num_non_clifford,
                tlike=st.num_t_like, cs=st['CSWAP'], ccx=st['CCX'], instr=st.num_instructions)


def show(name, fn):
    r = cost(fn)
    print(f'{name:<58s} Q={r["Q"]:>4d} T={r["T"]:>6,d} CSWAP={r["cs"]:>5d} CCX={r["ccx"]:>5d} instr={r["instr"]:>6d}')


# --- init_and variants
show('init_and(a,b,t)',
     lambda b: (lambda t: b.init_and(t[0], t[1], t[2]))(b.create_quantum_register(3, 't')))
def three_ctl(b):
    t = b.create_quantum_register(4, 't')
    b.init_and(t[0], km.array([t[1], t[2]]), t[3])
show('init_and([a,b,c],t) 3 controls', three_ctl)
def four_ctl(b):
    t = b.create_quantum_register(5, 't')
    b.init_and(t[0], km.array([t[1], t[2], t[3]]), t[4])
show('init_and([a,b,c,d],t) 4 controls', four_ctl)
def two_arr(b):
    t = b.create_quantum_register(4, 't')
    b.init_and(t[0], km.array([t[1], t[2]]), t[3])
show('init_and([a,b],[c],t) 3 controls (split)', two_arr)

# --- iadd
for L in (3, 4, 5, 6, 33):
    show(f'iadd(1, target={L}bit) classical const',
         lambda L=L: (lambda t: b_iadd(t, 1, None))(None) if False else None)
def mk_iadd(L, ctrl=None):
    def f(b):
        t = b.create_quantum_register(L, 't')
        if ctrl is None:
            b.iadd(1, target=t)
        else:
            c = b.create_quantum_register(1, 'c')[0]
            b.iadd(1, target=t, control=c)
    return f
for L in (3, 4, 5, 6):
    show(f'iadd(1,target={L}b)', mk_iadd(L))
    show(f'iadd(1,target={L}b,control=q)', mk_iadd(L, 1))

# --- left_rotate with shift register
def mk_rot(n, bits, ctrl=False):
    def f(b):
        t = b.create_quantum_register(n, 't')
        s = b.create_quantum_register(bits, 's') if bits else None
        if bits:
            if ctrl:
                c = b.create_quantum_register(1, 'c')[0]
                b.left_rotate(t, shift=s, control=c)
            else:
                b.left_rotate(t, shift=s)
        else:
            if ctrl:
                c = b.create_quantum_register(1, 'c')[0]
                b.left_rotate(t, control=c)
            else:
                b.left_rotate(t)
    return f
for n in (22, 39, 30):
    for bits in (4, 5, 6):
        show(f'left_rotate(target={n}, shift={bits}b)', mk_rot(n, bits))
show('left_rotate(target=39, shift=[0,s0,s1,s2,s3])',
     lambda b: b.left_rotate(b.create_quantum_register(39, 't'),
                             shift=[False] + list(b.create_quantum_register(4, 's'))))
show('left_rotate(target=22, shift=5b)', mk_rot(22, 5))
show('left_rotate(target=39, shift=5b)', mk_rot(39, 5))
for n in (30, 39, 22):
    show(f'left_rotate(target={n}, 1, control=q)', mk_rot(n, 0, True))

"""Debug hook: compare x_norm / s / y1 of the E5 circuit against the model."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT/'src', HERE, ROOT/'src'/'e4', ROOT/'data'/'verifiers'/'invsqrt'):
    sys.path.insert(0, str(p))
import numpy as np
import kickmix as km
from nr_reference_circuit import Precision, fixed_point_model
from invsqrt_equiv import PASS_PREC, to_bits
import invsqrt_opt3_circuit as E4
import invsqrt_flags_circuit as F


def read_reg(sim, circuit, name, n):
    reg = km.array(list(circuit.register_data[name]))
    b = sim.read_across_shots(reg)
    return (b.astype(np.uint64) << np.arange(n, dtype=np.uint64)[:, None]).sum(axis=0)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--n-p', type=int, default=2)
    ap.add_argument('--limit', type=int, default=16)
    a = ap.parse_args()
    n_p, prec = a.n_p, Precision(**PASS_PREC)
    width, n_s = 2 * n_p + 2, F.s_bits_for(a.n_p)
    xs = np.arange(1, min(2**width, a.limit + 1), dtype=np.uint64)
    circ = F.build_flags_circuit(n_p, prec=prec, compute_only=True)
    sim = km.Simulator(batch_size=len(xs), debug_print=False, seed=5)
    sim.clear_for_shot()
    sim.write_across_shots(km.array(list(circ.register_data['x'])), to_bits(xs, width))
    sim.do(circ)
    xnorm = read_reg(sim, circ, 'x', width)
    s = read_reg(sim, circ, 's', n_s)
    y1 = read_reg(sim, circ, 'y1', prec.output_bits(n_p))
    ms = ((width - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2)
    ms = ms.astype(np.uint64)
    mx = xs << (2 * ms)
    my1 = fixed_point_model(n_p, prec, xs.astype(np.int64)).astype(np.uint64)
    print('registers:', sorted(circ.register_data))
    print(f'{"x":>6s} {"s":>3s} {"s*":>3s} {"xn":>8s} {"xn*":>8s} {"y1":>10s} {"y1*":>10s}')
    for i, x in enumerate(xs):
        print(f'{int(x):>6d} {int(s[i]):>3d} {int(ms[i]):>3d} {int(xnorm[i]):>8d} {int(mx[i]):>8d} '
              f'{int(y1[i]):>10d} {int(my1[i]):>10d}'
              + ('' if (s[i] == ms[i] and xnorm[i] == mx[i] and y1[i] == my1[i]) else '   <<<'))


if __name__ == '__main__':
    main()

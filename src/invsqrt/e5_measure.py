"""E5 stage-cost ablation: where the Toffolis of the E4 circuit actually live."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'invsqrt'))
sys.path.insert(0, str(ROOT / 'src' / 'e4'))
import kickmix as km
import nr_reference_circuit as R
import invsqrt_opt3_circuit as E4
import invsqrt_opt_circuit as O
from nr_reference_circuit import Precision

PASS_PREC = dict(delta_bits=12, coef_bits=17, square_bits=28, product_bits=27, result_bits=28)


def stats(n_p, prec):
    c = E4.build_opt3_circuit(n_p, prec=prec)
    st = c.operation_statistics()
    return c.num_qubits, st.num_toffoli_like


def stage_cost(name, n_p, prec, patch, restore):
    base = stats(n_p, prec)[1]
    patch()
    try:
        q, t = stats(n_p, prec)
    finally:
        restore()
    return name, base, t, base - t, q


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-p', type=int, default=10)
    a = ap.parse_args()
    prec = Precision(**PASS_PREC)
    q, t = stats(a.n_p, prec)
    print(f'n_p={a.n_p} E4 baseline: Q={q} T={t:,}')
    for fn in ('_normalize', '_denormalize', '_select_interval2', '_square', '_multiply_m',
               '_multiply_y0', '_correct_slice', '_horner3_half', '_phase_by_bits'):
        if not hasattr(E4, fn):
            continue
        orig = getattr(E4, fn)
        setattr(E4, fn, lambda *aa, **kk: None)
        try:
            q2, t2 = stats(a.n_p, prec)
        finally:
            setattr(E4, fn, orig)
        print(f'  stub E4.{fn:<18s} -> T={t2:>6,d}  stage saves {t - t2:>6,d} '
              f'({(t - t2) / t * 100:5.1f}%)  Q={q2}')


if __name__ == '__main__':
    main()

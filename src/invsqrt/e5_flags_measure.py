"""Stage ablation + Q ledger for the E5 binary-flags circuit vs the E4 baseline."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT/'src', HERE, ROOT/'src'/'e4', ROOT/'data'/'verifiers'/'invsqrt'):
    sys.path.insert(0, str(p))
import invsqrt_opt3_circuit as E4
import invsqrt_flags_circuit as F
from nr_reference_circuit import Precision

PASS_PREC = dict(delta_bits=12, coef_bits=17, square_bits=28, product_bits=27, result_bits=28)


def stats(mod, n_p, prec):
    c = mod.build_opt3_circuit(n_p, prec=prec) if mod is E4 else mod.build_flags_circuit(n_p, prec=prec)
    st = c.operation_statistics()
    return c.num_qubits, st.num_toffoli_like


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--n-p', type=int, default=10)
    a = ap.parse_args()
    prec = Precision(**PASS_PREC)
    q4, t4 = stats(E4, a.n_p, prec)
    q5, t5 = stats(F, a.n_p, prec)
    print(f'n_p={a.n_p}')
    print(f'E4 : Q={q4}  T={t4:,}')
    print(f'E5 : Q={q5}  T={t5:,}   dQ={q5-q4:+d} dT={t5-t4:+,}')

    dead = set()
    print('\nE5 stage ablation (stub = replace by no-op):')
    for fn in ('_normalize_bin', '_denormalize_bin', '_select_interval2', '_correct_slice',
               '_phase_by_bits', '_thermometer_compute', '_thermometer_xor_into_s',
               '_thermometer_clear', '_horner3_half'):
        if not hasattr(F, fn):
            continue
        orig = getattr(F, fn)
        setattr(F, fn, lambda *aa, **kk: None)
        try:
            q2, t2 = stats(F, a.n_p, prec)
        finally:
            setattr(F, fn, orig)
        print(f'  {fn:<18s} saves {t5 - t2:>6,d} T   ({t5 - t2:>5d})  Q={q2}')

    print('\nE4 stage ablation for reference:')
    for fn in ('_normalize', '_denormalize', '_horner3_half'):
        orig = getattr(E4, fn)
        setattr(E4, fn, lambda *aa, **kk: None)
        try:
            q2, t2 = stats(E4, a.n_p, prec)
        finally:
            setattr(E4, fn, orig)
        print(f'  {fn:<18s} saves {t4 - t2:>6,d} T   Q={q2}')


if __name__ == '__main__':
    main()

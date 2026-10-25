"""E4: the dqi qubit frontier -- what is actually reachable, and what is not.

Answers the two questions the E2/APPROACH notes left open:

1. What is the **hard lower bound** on Q for the shipped contract?
2. What Q does a given score multiple require, and which of those are reachable?

Contract facts read off the shipped verifier (`verifiers/dqi/verify_submission_dqi.py`):
`S0..S{n-1}` are quantum registers of width `b` that must be restored every shot,
`E0..E{m-1}` are classical, and every other qubit must return to |0>.  So the
syndrome registers are 840 qubits that can never be removed.

Key-equation facts: the decoder must produce the error locator
`sigma` (deg <= t = n//2 -> t+1 = 36 field elements) and the error evaluator
`omega` (deg <= t-1 -> t = 35 elements).  A decoder that materialises both, while
the syndromes are still live, needs `(36 + 35) * 12 = 852` further qubits.

Usage
-----
    python src/e4/qfrontier.py --n 70 --degree 12
"""

from __future__ import annotations

import argparse


def frontier(n: int = 70, degree: int = 12, q_now: int = 1893,
             t_now: float = 5_725_108.38, r_factor: float = 1.1912604648115832) -> dict:
    b = degree
    t = n // 2
    syndromes = n * b
    sigma = (t + 1) * b
    omega = t * b
    core = syndromes + sigma + omega
    practical = syndromes + (n + 3) * b          # one n+3-word working array (the reference layout)
    without_omega = syndromes + sigma            # never materialise omega: Forney evaluated online

    def gain(q: int, t_rel: float = 1.0) -> float:
        return (q_now / q) ** 2 / t_rel

    return {
        'n': n, 'degree': b, 't': t,
        'syndrome_floor': syndromes,
        'sigma_words': t + 1, 'omega_words': t,
        'floor_sigma_plus_omega': core,
        'floor_no_omega': without_omega,
        'practical_reference_layout': practical,
        'q_now': q_now, 'r_factor': r_factor,
        'gain_if_core': gain(core),
        'gain_if_practical': gain(practical),
        'gain_if_no_omega': gain(without_omega),
        'extra_over_core_pct': 100.0 * (q_now - core) / core,
        'target_1325_working_set_words': (1325 - syndromes) / b,
        'target_1325_reachable': 1325 >= without_omega,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--q-now', type=int, default=1893)
    args = ap.parse_args()

    f = frontier(args.n, args.degree, args.q_now)
    b = f['degree']
    print(f"dqi Q frontier  (n={f['n']}, b={b}, t=n//2={f['t']}, Q_now={f['q_now']})")
    print()
    rows = [
        ('syndrome registers S0..S69 (contract input, unrestorable)',
         f['syndrome_floor'], 'HARD FLOOR of the contract'),
        (f"sigma, {f['sigma_words']} words  (=t+1)", f['sigma_words'] * b, 'must be materialised'),
        (f"omega, {f['omega_words']} words  (=t)", f['omega_words'] * b, 'must be materialised'),
        ('working array, one n+3-word block', (f['n'] + 3) * b, 'the reference layout'),
        ('reference Q', f['q_now'], 'measured'),
    ]
    for label, q, note in rows:
        print(f'  {q:6d}   {label:<52s} {note}')
    print()
    print(f"  reference is {f['extra_over_core_pct']:.1f}% above the "
          f"sigma+omega floor ({f['floor_sigma_plus_omega']} qubits)")
    print()
    print('  Reachability of the score multiple:')
    for label, q in (('materialising sigma AND omega', f['floor_sigma_plus_omega']),
                     ('one n+3 working array (practical)', f['practical_reference_layout']),
                     ('never materialising omega (online Forney)', f['floor_no_omega'])):
        print(f"    Q >= {q:5d}  ->  score x{(f['q_now'] / q) ** 2:.3f}   ({label})")
    print()
    print(f"  the requested Q = 1325 needs a working set of "
          f"{f['target_1325_working_set_words']:.1f} words beyond the syndromes "
          f"(sigma alone is {f['sigma_words']}).")
    print(f"  reachable only via the 'no omega' floor "
          f"({f['floor_no_omega']} qubits): {f['target_1325_reachable']}")


if __name__ == '__main__':
    main()

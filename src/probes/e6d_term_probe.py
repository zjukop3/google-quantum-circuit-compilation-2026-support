"""E6-D: what does the disputed sub-step-1 term actually DO?

The E3 story is
    flag ^= c1 AND NOT(n_B == 1) AND (reg_b[m-2] == 0)        (line 270)
and the model says it fires with probability 2^-12 per c1 opportunity.

A trajectory probe says otherwise: on forced-generic n=6 data the predicate
"c1 & n_B!=1 & reg_b[m-2]==0" is true on ~1.1% of shots, but only ~0.073% of
shots actually leave the generic schedule.  So one of the two is wrong.

This script settles it by building three circuit variants and running the SAME
data through all three:

    orig    : shipped term
    delete  : term removed (flag never gets the extra flip)
    force   : term replaced by `cx(aux, flag)` (flag ^= c1 & NOT(n_B==1))

If `force` changes the schedule on a shot where `orig` == `delete`, then the
term is *inert* in the shipped circuit and the "2^-12 predicate" story is wrong.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import e6d_schedule as E                                          # noqa: E402

TERM_ORIG = '            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)'
TERM_DELETE = '            pass  # [e6d] disputed term removed'
TERM_FORCE = '            builder.cx(aux, flag)  # [e6d] term forced on'
TERM_NEVER = '            pass  # placeholder'


def build_variant_source(kind: str) -> str:
    src = E.build_probe_source()
    assert TERM_ORIG in src, 'term anchor moved'
    rep = {'orig': TERM_ORIG, 'delete': TERM_DELETE, 'force': TERM_FORCE}[kind]
    return src.replace(TERM_ORIG, rep, 1)


def load(kind):
    ns = {'__name__': 'e6dvar'}
    exec(compile(build_variant_source(kind), f'<e6dvar:{kind}>', 'exec'), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    n, d = args.n, args.degree

    rng = np.random.default_rng(args.seed)
    data = E.gen_data('generic', n, d, args.shots, rng)

    traces = {}
    for kind in ('orig', 'delete', 'force'):
        ns = load(kind)
        circ, dec, ni, me, syn = E.build(n, d, n, ns)
        T = E.run_batch(circ, dec, syn, data, d, seed=5, n_iters=ni)
        traces[kind] = T
        print(f'[{kind}] T={circ.operation_statistics().num_toffoli_like:,} '
              f'Q={circ.num_qubits:,}', flush=True)

    TO = traces['orig']
    base = traces['delete']
    forc = traces['force']

    # majority trace of `delete` = "no-extra-flip" reference schedule
    keys = [base[:, s, :9].tobytes() for s in range(args.shots)]
    c = Counter(keys)
    kdel, ndel = c.most_common(1)[0]
    ref_del = base[:, keys.index(kdel), :]

    keys_o = [TO[:, s, :9].tobytes() for s in range(args.shots)]
    co = Counter(keys_o)
    korig, norig = co.most_common(1)[0]
    ref_orig = TO[:, keys_o.index(korig), :]

    d_od = np.any(TO[:, :, :9] != base[:, :, :9], axis=2)     # orig vs delete
    d_of = np.any(forc[:, :, :9] != TO[:, :, :9], axis=2)     # force vs orig
    d_fd = np.any(forc[:, :, :9] != base[:, :, :9], axis=2)   # force vs delete
    dev_from_generic = np.any(TO[:, :, :9] != ref_orig[:, None, :9], axis=2)

    n_od = int(d_od.any(axis=0).sum())
    n_of = int(d_of.any(axis=0).sum())
    n_fd = int(d_fd.any(axis=0).sum())
    n_dev = int(dev_from_generic.any(axis=0).sum())
    print(f'\nshots={args.shots}  delete-majority {ndel}  orig-majority {norig}')
    print(f'orig vs delete  : {n_od} shots differ  ({n_od/args.shots:.5f})')
    print(f'force vs orig   : {n_of} shots differ  ({n_of/args.shots:.5f})')
    print(f'force vs delete : {n_fd} shots differ  ({n_fd/args.shots:.5f})')
    print(f'orig != orig-majority (deviation): {n_dev} ({n_dev/args.shots:.5f})')

    # predicate statistics
    c1, aux, pred, fire, nB, rb2 = E.derived(TO)
    fireshots = set(np.nonzero(fire.any(axis=0))[0].tolist())
    devshots = set(np.nonzero(dev_from_generic.any(axis=0))[0].tolist())
    od_shots = set(np.nonzero(d_od.any(axis=0))[0].tolist())
    of_shots = set(np.nonzero(d_of.any(axis=0))[0].tolist())
    print(f'predicate-true shots {len(fireshots)}   schedule-deviating {len(devshots)}')
    print(f'predicate-true but orig==delete: {len(fireshots - od_shots)}')
    print(f'orig!=delete but predicate false: {len(od_shots - fireshots)}')
    print(f'force!=orig but predicate false : {len(of_shots - fireshots)}')

    # a few worked examples
    ex = []
    for s in sorted(fireshots - od_shots)[:3]:
        its = [int(x) for x in np.nonzero(fire[:, s])[0]]
        ex.append({'shot': int(s), 'kind': 'predicate-true but term inert',
                   'fire_iters': its,
                   'rb2_at_fire': [int(TO[i, s, E.FIELDS.index('rb2')]) for i in its],
                   'nA_at_fire': [int(TO[i, s, E.FIELDS.index('n_A')]) for i in its],
                   'nB_at_fire': [int(TO[i, s, E.FIELDS.index('n_B')]) for i in its]})
    for s in sorted(od_shots)[:3]:
        d = np.nonzero(np.any(TO[:, s, :9] != base[:, s, :9], axis=1))[0]
        ex.append({'shot': int(s), 'kind': 'term changes the schedule',
                   'first_diff_iter': int(d[0]),
                   'rb2_at_first_diff': int(TO[d[0], s, E.FIELDS.index('rb2')])})

    out = {'n': n, 'degree': d, 'shots': args.shots, 'seed': args.seed,
           'n_orig_vs_delete': n_od, 'n_force_vs_orig': n_of, 'n_force_vs_delete': n_fd,
           'n_deviation_orig': n_dev,
           'p_orig_vs_delete': n_od / args.shots,
           'p_force_vs_orig': n_of / args.shots,
           'n_predicate_true_shots': len(fireshots),
           'n_predicate_true_but_inert': len(fireshots - od_shots),
           'n_terms_change_without_predicate': len(od_shots - fireshots),
           'examples': ex}
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

"""E7-H — why did 6 shots fail?  Replays the *official* grading sample.

The full 102,400-shot verification of `build/e7/dqi_frozen.kmb` produced
`value_errors=0, phase_errors=6` (R = 1.2166 instead of 1.1913), while the
unfrozen reference run on the *same* sample produced 0 errors.  So six shots
have a sweep live on a site the provable union called dead.

This script rebuilds the exact official sample for a given circuit hash
(`verify()` derives its seed from `sha256(.kmb)`), feeds those syndromes into the
cheap n=70 metadata probe, and reports, per shot:
  * whether the freeze set is touched (=> that shot fails)
  * the real metadata trace on all 9 fields
  * the event pattern produced by the state-machine walk (`e7_sm2.infer_events`)

Usage
-----
    $P src/e7/e7_diag_official.py --hash-file build/e7/dqi_frozen.kmb \
        --model nA070_69_K2 --json notes/runs/e7/diag_official.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'e6'))
sys.path.insert(0, str(HERE))

import e6d_schedule as S          # noqa: E402
import e7_sm as M                 # noqa: E402
import e7_sm2 as M2               # noqa: E402

import kickmix as km              # noqa: E402


def official_sample(base_seed, num_shots, batch_size, m=4095, n=70, degree=12,
                    r=2016):
    spec_mod = importlib.util.spec_from_file_location(
        'vdqi2', ROOT / 'data' / 'verifiers' / 'dqi' / 'verify_submission_dqi.py')
    V = importlib.util.module_from_spec(spec_mod)
    sys.modules['vdqi2'] = V
    spec_mod.loader.exec_module(V)
    spec = V.ProblemSpecDQI.benchmark_instance(m=m, n=n, degree=degree,
                                               num_shots=batch_size)
    tables = V.FieldTables(km.GF2Field(degree))
    wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    n_batches = num_shots // batch_size
    seeds = np.random.SeedSequence(base_seed, pool_size=8).generate_state(
        8 * n_batches, dtype=np.uint64).reshape(-1, 2, 4)
    sy = np.zeros((num_shots, n), dtype=np.int64)
    mb = np.zeros((num_shots, m), dtype=np.int64)
    ph = np.zeros(num_shots, dtype=bool)
    for i in range(n_batches):
        rng = np.random.default_rng(int.from_bytes(seeds[i, 0], 'little'))
        a, b, c = V.sample_shots(spec, tables, rng, batch_size, wp)
        sl = slice(i * batch_size, (i + 1) * batch_size)
        sy[sl], mb[sl], ph[sl] = a, b, c
    return sy, mb, ph


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hash-file', default=str(ROOT / 'build/e7/dqi_frozen.kmb'))
    ap.add_argument('--base-seed', default=None)
    ap.add_argument('--freeze', default=str(ROOT / 'notes/runs/e7/freeze_sites.json'))
    ap.add_argument('--model', default='nA070_69_K2')
    ap.add_argument('--shots', type=int, default=102400)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--probe-batch', type=int, default=2048)
    ap.add_argument('--max-report', type=int, default=3)
    ap.add_argument('--json', default=None)
    ap.add_argument('--probe-live', default=None)
    args = ap.parse_args()

    if args.base_seed is not None:
        base_seed = int(args.base_seed)
    else:
        h = hashlib.sha256(Path(args.hash_file).read_bytes()).hexdigest()
        base_seed = int(h, 16)
        print(f'[seed] sha256={h[:16]}…')
    sy, mb, ph = official_sample(base_seed, args.shots, args.batch_size)
    print(f'[sample] {sy.shape[0]} shots, mean weight '
          f'{float((sy != 0).sum(1).mean()):.2f}')

    fz_sites = json.loads(Path(args.freeze).read_text())[args.model]['sites']
    fz = np.zeros((M.N_ITERS, M.M_E), dtype=bool)
    for it, i in fz_sites:
        fz[int(it), int(i)] = True

    n, degree = 70, 12
    ns = S.load_probe_module()
    circuit, dec, n_iters, m_e, syn = S.build(n, degree, n, ns)
    live = np.zeros((args.shots, n_iters, m_e), dtype=bool)
    trace = np.zeros((args.shots, n_iters, len(S.FIELDS)), dtype=np.uint16)
    for start in range(0, args.shots, args.probe_batch):
        stop = min(start + args.probe_batch, args.shots)
        T = S.run_batch(circuit, dec, syn, sy[start:stop], degree, seed=7,
                        n_iters=n_iters)
        live[start:stop] = S.mul_bitmap(T, m_e)
        trace[start:stop] = T.transpose(1, 0, 2)
    pl = Path(args.probe_live) if args.probe_live else None
    if pl is not None and pl.exists():
        live = np.load(pl)
        print(f'[probe] loaded live bitmap {live.shape} from {pl}')
    elif pl is not None:
        np.save(pl, live)
        print(f'[probe] saved live bitmap {live.shape} to {pl}')
    union = live.any(axis=0)
    print(f'[union] empirical union of this sample: {int(union.sum())} / {union.size} '
          f'({union.sum() / union.size:.5f})  freeze={union.size - int(union.sum())} '
          f'({1 - union.sum() / union.size:.5f})')
    viol = (live & fz[None]).any(axis=(1, 2))
    idx = np.flatnonzero(viol)
    print(f'[result] shots whose sweep touches a frozen site: {len(idx)} '
          f'({len(idx) / args.shots:.3e})  -> {idx[:20].tolist()}')

    # how many *non*-frozen sites are live on those shots (for context)
    details = []
    for s in idx[:args.max_report]:
        tr = trace[s]
        sites = np.argwhere(live[s] & fz)
        ev = M2.infer_events(tr) if hasattr(M2, 'infer_events') else None
        d = {'shot': int(s), 'n_frozen_sites_live': int(len(sites)),
             'frozen_sites_live': [[int(a), int(b)] for a, b in sites[:12]],
             'syndrome_weight': int((sy[s] != 0).sum()),
             'first_rows': [[int(v) for v in tr[t]] for t in range(6)],
             'infer': None if ev is None else {
                 'ok': bool(ev[1]), 'first_fail': ev[2],
                 'events': [[int(a), str(b)] for a, b in ev[0]]}}
        details.append(d)
        print(f'[shot {s}] weight={d["syndrome_weight"]} '
              f'frozen-live={d["n_frozen_sites_live"]} {d["frozen_sites_live"][:4]}')
        print(f'   rows0-3: {d["first_rows"][:4]}')
        print(f'   infer: {d["infer"]}')

    out = {'hash_file': args.hash_file, 'shots': args.shots,
           'model': args.model, 'violating_shots': idx.tolist(),
           'violating_count': int(len(idx)), 'details': details}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""E7 diag: what exactly does a deviating shot's schedule look like vs the reference?

Prints, for one deviating shot, the per-iteration
    (c0,c1,c2,c3, n_A, n_a)  of reference and of the shot, side by side, plus the
best-matching reference index per iteration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'e6'))
import e6d_schedule as S  # noqa: E402


def main() -> int:
    n, degree = 70, 12
    n_iters = 3 * (n + 1) + 9
    m_e = n + 3
    ns = S.load_probe_module()
    circuit, dec, n_iters, m_e, syn = S.build(n, degree, n, ns)
    idx = {f: i for i, f in enumerate(S.FIELDS)}

    rng = np.random.default_rng(1001)
    T = S.run_batch(circuit, dec, syn, S.gen_data('generic', n, degree, 256, rng),
                    degree, seed=99, n_iters=n_iters)
    from collections import Counter
    keys = [T[:, s, :9].tobytes() for s in range(256)]
    key = Counter(keys).most_common(1)[0][0]
    ref = T[:, keys.index(key), :]

    def sig(tr):
        c = tr[:, idx['ctrl']]
        return np.stack([(c >> k) & 1 for k in range(4)], axis=1)

    refsig = sig(ref)
    rs = (refsig[:, 0] * 1 + refsig[:, 1] * 2 + refsig[:, 2] * 4 + refsig[:, 3] * 8)
    print('ref ctrl code per iteration:', ''.join('%x' % v for v in rs))
    print('ref n_A:', ref[:, idx['n_A']].tolist())
    print('ref n_a:', ref[:, idx['n_a']].tolist())
    print('ref counter:', ref[:, idx['counter']].tolist())
    print('ref hc:', ref[:, idx['hc']].tolist())

    r2 = np.random.default_rng(4242)
    shots = 8192
    T2 = S.run_batch(circuit, dec, syn, S.gen_data('natural', n, degree, shots, r2),
                     degree, seed=5, n_iters=n_iters)
    diff = np.any(T2[:, :, :9] != ref[:, None, :9], axis=2)
    dev = np.nonzero(diff.any(axis=0))[0]
    print(f'shots={shots} dev={len(dev)}')
    for s in dev[:3]:
        d = np.nonzero(diff[:, s])[0]
        tr = T2[:, s, :]
        c = tr[:, idx['ctrl']]
        ds = np.stack([(c >> k) & 1 for k in range(4)], axis=1)
        ds_code = (ds[:, 0] * 1 + ds[:, 1] * 2 + ds[:, 2] * 4 + ds[:, 3] * 8)
        print(f'--- shot {s}: first_div={d[0]} last_div={d[-1]} ndiff={len(d)}')
        print('  dev ctrl code:', ''.join('%x' % v for v in ds_code))
        print('  dev n_A :', tr[:, idx['n_A']].tolist())
        print('  dev n_a :', tr[:, idx['n_a']].tolist())
        print('  dev counter:', tr[:, idx['counter']].tolist())
        print('  dev hc :', tr[:, idx['hc']].tolist())
        # best-matching reference index for each iteration, using the schedule fields
        rows = tr[:, :9]
        best = []
        for it in range(n_iters):
            m = np.all(ref[:, :9] == rows[it], axis=1)
            cand = np.nonzero(m)[0]
            best.append(int(cand[0]) if len(cand) else -1)
        print('  best-matching ref index:', best)
        print('  it-lag:', [it - b if b >= 0 else None for it, b in enumerate(best)])
        live_dev = S.mul_bitmap(tr[:, None, :], m_e)[0]
        live_ref = S.mul_bitmap(ref[:, None, :], m_e)[0]
        print('  live/dev counts:', live_dev.sum(axis=1).tolist())
        print('  live/ref counts:', live_ref.sum(axis=1).tolist())
    return 0


if __name__ == '__main__':
    sys.exit(main())

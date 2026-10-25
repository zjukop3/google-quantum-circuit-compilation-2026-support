"""E6-D — independent re-measurement of the dqi EEA "schedule deviation".

What this measures, in one shot (no Chien / no reverse EEA needed: the forward EEA is
a prefix of the shipped decoder, and the Chien segment cannot influence it).

Per *iteration* (`it = 0..n_iters-1`, value read at the start of the iteration,
right after `_counter_controls`, i.e. exactly where `_multiply_and_add_or_sub`
reads its controls) we fan out, with plain CNOTs onto fresh |0> ancillas:

    n_A, n_B, n_a, n_b, n_q, flag, counter, halting_counter, ctrl[0..3]   (the schedule)
    reg_b[m-2]  (12 bits, the coefficient inside `flip_if_equal(reg_b[m-2], 0, ...)`)
    reg_b[0]    (12 bits, the sub-step-2 coefficient)

CNOT fan-out onto a fresh ancilla is a *read-only* copy whenever the source is in a
computational-basis state -- which it is here: the verifier drives this circuit with
classical syndrome data, so the whole circuit is a reversible classical computation
per shot.  `e6d_probe_equivalence.py` checks that the probe circuit's syndrome output
is bit-identical to the shipped circuit's on the same shots.

Derived quantities
------------------
c1(it)   = ctrl bit 1                       (the sub-step-1 control)
aux(it)  = c1 AND NOT(n_B == 1)             (the control of the disputed term)
pred(it) = (reg_b[m-2] == 0)                (the disputed predicate)
fire(it) = aux AND pred                     -> `flag ^= 1` inside sub-step 1

Usage
-----
    python src/e6/e6d_schedule.py --n 70 --shots 8192 --batches 2 --regime generic
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py'
SRC = REF.read_text()

CTRL_ANCHOR = ('                ctrl = self._counter_controls('
               'builder, counter, is_c3, is_hc_zero)')

FIELDS = ('n_A', 'n_B', 'n_a', 'n_b', 'n_q', 'flag', 'counter', 'hc', 'ctrl', 'rb2', 'rb0')
WIDTHS = (8, 8, 8, 8, 8, 1, 2, 8, 4, 12, 12)

TRACE_PROBE = '''                if not is_adjoint and it < self._tp_len:
                    _tk = self._trace_probe
                    builder.cx(n_A, _tk['n_A'][it])
                    builder.cx(n_B, _tk['n_B'][it])
                    builder.cx(n_a, _tk['n_a'][it])
                    builder.cx(n_b, _tk['n_b'][it])
                    builder.cx(n_q, _tk['n_q'][it])
                    builder.cx(km.array([flag]), _tk['flag'][it])
                    builder.cx(counter, _tk['counter'][it])
                    builder.cx(halting_counter, _tk['hc'][it])
                    builder.cx(ctrl, _tk['ctrl'][it])
                    builder.cx(reg_b[len(reg_b) - 2], _tk['rb2'][it])
                    builder.cx(reg_b[0], _tk['rb0'][it])
'''

WRAP_ANCHOR = '    def _initial_shift_loop('

WRAP = '''
    def _alloc_trace_probe(self, builder, n_iters, m_, logm, hcw):
        def q(width, name, it):
            return builder.create_quantum_register(width, name=f'{name}{it}')
        self._tp_len = n_iters
        self._trace_probe = {
            'n_A': [q(logm, 'TnA', it) for it in range(n_iters)],
            'n_B': [q(logm, 'TnB', it) for it in range(n_iters)],
            'n_a': [q(logm, 'Tna', it) for it in range(n_iters)],
            'n_b': [q(logm, 'Tnb', it) for it in range(n_iters)],
            'n_q': [q(logm, 'Tnq', it) for it in range(n_iters)],
            'flag': [q(1, 'Tfl', it) for it in range(n_iters)],
            'counter': [q(2, 'Tct', it) for it in range(n_iters)],
            'hc': [q(hcw, 'Thc', it) for it in range(n_iters)],
            'ctrl': [q(4, 'Tcx', it) for it in range(n_iters)],
            'rb2': [q(self.field.degree, 'Trb2', it) for it in range(n_iters)],
            'rb0': [q(self.field.degree, 'Trb0', it) for it in range(n_iters)],
        }

    def build_syndrome_decoding_mbuc(self, builder, *, syndromes, error_mbuc, n_iters):
        m_ = len(syndromes) + 3
        logm = m_.bit_length() + 1
        self._trace_probe = None
        self._tp_len = n_iters
        if getattr(self, 'trace_probe_enabled', True):
            self._alloc_trace_probe(builder, n_iters, m_, logm, n_iters.bit_length())
        return self._orig_build_syndrome_decoding_mbuc(
            builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters)

'''

TAIL_START = "        with builder.mark('chien_forney_mbuc'):"
TAIL_END = "            builder.free(reg)\n"


def build_probe_source() -> str:
    src = SRC
    assert CTRL_ANCHOR in src, 'ctrl anchor moved'
    src = src.replace(CTRL_ANCHOR, CTRL_ANCHOR + '\n' + TRACE_PROBE.rstrip('\n'), 1)
    src = src.replace(WRAP_ANCHOR, WRAP.strip('\n') + '\n\n' + WRAP_ANCHOR, 1)
    src = src.replace(
        '    def build_syndrome_decoding_mbuc(self, builder: km.CircuitBuilder, *,',
        '    def _orig_build_syndrome_decoding_mbuc(self, builder: km.CircuitBuilder, *,', 1)
    start = src.index(TAIL_START)
    end = src.index(TAIL_END, start) + len(TAIL_END)
    src = src[:start] + '        # [e6d probe] Chien / reverse EEA / unshift omitted\n' + src[end:]
    return src


def load_probe_module():
    ns = {'__name__': 'e6dprobe'}
    exec(compile(build_probe_source(), '<e6dprobe>', 'exec'), ns)
    return ns


def build(n, degree, m_chien, ns=None):
    if ns is None:
        ns = load_probe_module()
    m_e = n + 3
    n_iters = 3 * (n + 1) + 9
    dec = ns['ProosAndZalkaDecoder'](n, km.GF2Field(degree))
    b = km.CircuitBuilder()
    syn = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    err = [b.create_classical_register(degree, name=f'E{j}') for j in range(m_chien)]
    dec.build_syndrome_decoding_mbuc(b, syndromes=syn, error_mbuc=err, n_iters=n_iters)
    return b.finish_circuit(), dec, n_iters, m_e, syn


def bits_of(data, degree):
    mat = np.asarray(data, dtype=np.int64)
    rows = []
    for reg in range(mat.shape[1]):
        for b in range(degree):
            rows.append(((mat[:, reg] >> b) & 1).astype(bool))
    return np.array(rows, dtype=bool)


def run_batch(circuit, dec, syn, data, degree, seed=7, n_iters=None):
    """(n_iters, shots, 11) uint8 trace."""
    shots = data.shape[0]
    sim = km.Simulator(batch_size=shots, counting_operations=True, seed=seed)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(km.array([q for r in syn for q in r]), bits_of(data, degree))
    sim.do(circuit)

    T = np.zeros((n_iters, shots, len(FIELDS)), dtype=np.uint16)   # rb2/rb0 are 12-bit
    tp = dec._trace_probe
    for j, name in enumerate(FIELDS):
        w = len(tp[name][0])          # actual register width (varies with n)
        assert w <= WIDTHS[j], (name, w, WIDTHS[j])
        flat = km.array([q for it in range(n_iters) for q in tp[name][it]])
        bits = sim.read_across_shots(flat)
        wide = np.zeros((n_iters * w, shots), dtype=np.uint16)
        for k in range(min(len(bits), n_iters * w)):
            wide[k] = np.asarray(bits[k]).astype(np.uint16)
        wide = wide.reshape(n_iters, w, shots)
        for bit in range(w):
            T[:, :, j] |= wide[:, bit, :] << bit
    return T


# --------------------------------------------------------------------------- #

def gen_data(kind, n, degree, shots, rng):
    full = (1 << degree) - 1
    if kind == 'generic':
        d = rng.integers(1, full + 1, size=(shots, n)).astype(np.int64)
        d[:, n - 1] = rng.integers(1, full + 1, size=shots)
        return d
    if kind == 'natural':
        return rng.integers(0, full + 1, size=(shots, n)).astype(np.int64)
    if kind == 'lead1':
        d = rng.integers(1, full + 1, size=(shots, n)).astype(np.int64)
        d[:, n - 1] = 0
        return d
    if kind == 'zero':
        return np.zeros((shots, n), dtype=np.int64)
    raise ValueError(kind)


def derived(T):
    """(n_iters, shots) boolean arrays c1, aux, pred, fire + int nB."""
    idx = {f: i for i, f in enumerate(FIELDS)}
    c1 = ((T[:, :, idx['ctrl']] >> 1) & 1).astype(bool)
    nB = T[:, :, idx['n_B']]
    rb2 = T[:, :, idx['rb2']]
    aux = c1 & (nB != 1)
    pred = rb2 == 0
    fire = aux & pred
    return c1, aux, pred, fire, nB, rb2


def mul_bitmap(T, m_e):
    idx = {f: i for i, f in enumerate(FIELDS)}
    c0 = (T[:, :, idx['ctrl']] & 1).astype(bool)
    c3 = ((T[:, :, idx['ctrl']] >> 3) & 1).astype(bool)
    n_A = T[:, :, idx['n_A']]
    n_a = T[:, :, idx['n_a']]
    i = np.arange(m_e)[None, :]
    left = c0[:, :, None] & (i >= (m_e - n_A)[:, :, None])
    right = c3[:, :, None] & (i <= (n_a - 1)[:, :, None])
    return (left ^ right).transpose(1, 0, 2)      # (shots, n_iters, m_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=8192)
    ap.add_argument('--batches', type=int, default=2)
    ap.add_argument('--ref-shots', type=int, default=256)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--regime', default='generic')
    ap.add_argument('--json', default=None)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    n, degree = args.n, args.degree
    m_chien = n
    ns = load_probe_module()
    t0 = time.time()
    circuit, dec, n_iters, m_e, syn = build(n, degree, m_chien, ns)
    t_build = time.time() - t0
    print(f'[build] n={n} m_e={m_e} n_iters={n_iters} '
          f'T={circuit.operation_statistics().num_toffoli_like:,} Q={circuit.num_qubits:,} '
          f'({t_build:.1f}s)', flush=True)

    # ---- majority reference trace on forced-generic shots -----------------
    rng = np.random.default_rng(args.seed + 1000 * n)
    Tref = run_batch(circuit, dec, syn,
                     gen_data('generic', n, degree, args.ref_shots, rng),
                     degree, seed=99, n_iters=n_iters)
    NSCHED = 9                                     # schedule fields only (excl. rb2/rb0)
    keys = [Tref[:, s, :NSCHED].tobytes() for s in range(args.ref_shots)]
    cnt = Counter(keys)
    key, kmaj = cnt.most_common(1)[0]
    ref = Tref[:, keys.index(key), :]
    print(f'[ref] majority {kmaj}/{args.ref_shots}, {len(cnt)} distinct', flush=True)
    ref_bm = mul_bitmap(ref[:, None, :], m_e)[0]
    ref_live_per_it = ref_bm.reshape(n_iters, m_e).sum(axis=1)
    ref_live = int(ref_bm.sum())
    # reference fire pattern
    rc1, raux, rpred, rfire, _, _ = derived(ref[:, None, :])
    rfire = rfire[:, 0]

    tot = 0
    trace_mis = 0
    bitmap_mis = 0
    fire_count = Counter()
    fire_pattern_dev = []
    first_div_mod6 = Counter()
    first_div_mod3 = Counter()
    persist_last = 0
    ndiff_hist = Counter()
    naux_tot = 0
    npred_given_aux = 0
    n_c1_tot = 0
    n_nB_one = 0
    rb2_zero_all = 0
    rb2_tot_all = 0
    nB_one_at_c1 = 0
    maxfire = 0
    live_delta_hist = Counter()
    dev_shots_examples = []
    per_batch = []
    rb2_val_at_aux = {}
    rb2_val_at_c1 = {}
    union_all = np.zeros((n_iters, m_e), dtype=bool)
    union_dev = np.zeros((n_iters, m_e), dtype=bool)

    for bi in range(args.batches):
        r = np.random.default_rng(7919 * (bi + 1) + args.seed)
        data = gen_data(args.regime, n, degree, args.shots, r)
        t1 = time.time()
        T = run_batch(circuit, dec, syn, data, degree, seed=7 + bi, n_iters=n_iters)
        dt = time.time() - t1

        c1, aux, pred, fire, nB, rb2 = derived(T)
        diffmask = np.any(T[:, :, :9] != ref[:, None, :9], axis=2)  # (n_iters, shots)
        row_same = ~diffmask.any(axis=0)
        tmis = int((~row_same).sum())
        bm = mul_bitmap(T, m_e)
        union_all |= bm.any(axis=0)
        union_dev |= bm[~row_same].any(axis=0) if (~row_same).any() else union_dev
        bmis = int((~np.all(bm == ref_bm[None, :, :], axis=(1, 2))).sum())

        n_c1_tot += int(c1.sum())
        n_nB_one += int(((nB == 1)).sum())
        m = c1
        naux_tot += int(aux.sum())
        npred_given_aux += int((aux & pred).sum())
        nB_one_at_c1 += int((c1 & (nB == 1)).sum())
        rb2_zero_all += int((rb2 == 0).sum())
        rb2_tot_all += int(rb2.size)
        for v in np.unique(rb2[aux], return_counts=True)[0][:0]:
            pass
        _av = rb2[aux]
        for v, c in zip(*np.unique(_av, return_counts=True)):
            rb2_val_at_aux[int(v)] = rb2_val_at_aux.get(int(v), 0) + int(c)
        _c1v = rb2[c1]
        for v, c in zip(*np.unique(_c1v, return_counts=True)):
            rb2_val_at_c1[int(v)] = rb2_val_at_c1.get(int(v), 0) + int(c)

        f = fire.sum(axis=0)
        for v in f:
            fire_count[int(v)] += 1
        maxfire = max(maxfire, int(f.max()))
        dev = ~row_same
        if dev.any():
            # per-shot difference of the fire pattern vs reference
            fd = (fire != rfire[:, None]).sum(axis=0)
            for s in np.nonzero(dev)[0]:
                fire_pattern_dev.append(int(fd[s]))
                d = np.nonzero(diffmask[:, s])[0]
                first = int(d[0])
                first_div_mod6[first % 6] += 1
                first_div_mod3[first % 3] += 1
                ndiff_hist[int(len(d))] += 1
                if diffmask[-1, s]:
                    persist_last += 1
                live_delta_hist[int(bm[s].sum()) - ref_live] += 1
            for s in np.nonzero(dev)[0][:3]:
                if len(dev_shots_examples) < 6:
                    d = np.nonzero(diffmask[:, s])[0]
                    dev_shots_examples.append({
                        'shot': int(s) if bi == 0 else -1,
                        'batch': bi,
                        'first_div': int(d[0]),
                        'last_div': int(d[-1]),
                        'n_diff_iters': int(len(d)),
                        'fire_times': [int(x) for x in np.nonzero(fire[:, s])[0]],
                        'ref_fire_times': [int(x) for x in np.nonzero(rfire)[0]],
                        'diff_iters_head': [int(x) for x in d[:12]],
                    })
        tot += args.shots
        trace_mis += tmis
        bitmap_mis += bmis
        per_batch.append({'batch': bi, 'shots': args.shots, 'trace_mismatch': tmis,
                          'bitmap_mismatch': bmis, 'sec': round(dt, 1),
                          'max_fire': int(f.max())})
        print(f'[batch {bi}] shots={args.shots} dev={tmis} ({tmis/args.shots:.5f}) '
              f'bitmap_dev={bmis} live_ref={ref_live} {dt:.1f}s', flush=True)

    p_dev = trace_mis / tot
    out = {
        'tag': args.tag, 'n': n, 'degree': degree, 'regime': args.regime, 'shots': tot,
        'n_iters': n_iters, 'm_e': m_e,
        'build_sec': round(t_build, 2),
        'probe_T': circuit.operation_statistics().num_toffoli_like,
        'probe_Q': circuit.num_qubits,
        'ref_shots': args.ref_shots, 'ref_majority': kmaj, 'ref_distinct': len(cnt),
        'dev_shots': trace_mis, 'p_dev': p_dev,
        'bitmap_dev': bitmap_mis, 'p_bitmap': bitmap_mis / tot,
        # ---- predicate statistics ------------------------------------------
        'c1_high_total': n_c1_tot, 'c1_per_shot': n_c1_tot / tot,
        'nB_eq_1_total': n_nB_one,
        'nB_eq_1_at_c1_total': nB_one_at_c1,
        'aux_high_total': naux_tot, 'aux_per_shot': naux_tot / tot,
        'fire_total': int(sum(k * v for k, v in fire_count.items())),
        'fire_per_shot': sum(k * v for k, v in fire_count.items()) / tot,
        'fire_count_hist': dict(sorted(fire_count.items())),
        'max_fire_in_shot': maxfire,
        'P_fire_given_aux': (npred_given_aux / naux_tot) if naux_tot else None,
        'P_rb2_zero_all_shot_iters': rb2_zero_all / rb2_tot_all,
        'P_rb2_zero_given_aux': (npred_given_aux / naux_tot) if naux_tot else None,
        'rb2_val_at_aux_top': dict(sorted(rb2_val_at_aux.items(),
                                         key=lambda kv: -kv[1])[:12]),
        'rb2_val_at_aux_support': len(rb2_val_at_aux),
        'rb2_val_at_c1_top': dict(sorted(rb2_val_at_c1.items(),
                                        key=lambda kv: -kv[1])[:12]),
        'rb2_val_at_c1_support': len(rb2_val_at_c1),
        'two_pow_minus_degree': 2.0 ** -degree,
        'ref_fire_per_shot': float(rfire.sum()),
        'ref_fire_times': [int(x) for x in np.nonzero(rfire)[0]],
        'ref_live_mul_positions': ref_live,
        'union_live_positions': int(union_all.sum()),
        'union_dev_extra_positions': int((union_dev & ~(ref_bm.reshape(n_iters, m_e))).sum()),
        'safe_freeze_fraction': float(1 - union_all.sum() / union_all.size),
        'majority_freeze_fraction': float(1 - ref_live / union_all.size),
        'ref_live_per_iteration': [int(x) for x in ref_live_per_it],
        # ---- persistence / structure ---------------------------------------
        'persist_at_last_iter': persist_last,
        'fire_pattern_diff_hist': dict(sorted(Counter(fire_pattern_dev).items())),
        'first_div_mod6': dict(sorted(first_div_mod6.items())),
        'first_div_mod3': dict(sorted(first_div_mod3.items())),
        'ndiff_iters_hist': dict(sorted(ndiff_hist.items())),
        'live_mul_delta_hist': dict(sorted(live_delta_hist.items())),
        'dev_examples': dev_shots_examples,
        'per_batch': per_batch,
    }
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('dev_examples', 'ref_live_per_iteration',
                                   'fire_count_hist', 'ndiff_iters_hist')},
                     indent=2, default=str), flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2, default=str))
        print('wrote', args.json)


if __name__ == '__main__':
    main()

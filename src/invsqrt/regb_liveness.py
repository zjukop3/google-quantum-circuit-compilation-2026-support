"""E4: what does `reg_b` actually hold, iteration by iteration?

Question
--------
At `n = 70, degree = 12` the reference decoder allocates

    reg_a = 3 fresh words + 70 syndrome registers   ->  36 + 840 =  876 qubits
    reg_b = 73 fresh words                          ->             876 qubits

i.e. 1752 of its 1893 qubits (92 %) are the two EEA polynomials, and `reg_b`
alone is 876 (46 %).  The paper calls this "explicit remainder storage" and says
it can be improved.  But *how wide* do those two polynomials really have to be?

Method (no model, the real circuit)
-----------------------------------
Splice a `DEBUG_PRINT` of every `reg_a` and `reg_b` qubit right after each
*forward* EEA iteration, then reconstruct the per-shot slot contents from the
simulator's debug callback.  `DEBUG_PRINT` costs zero qubits, so the traced
circuit has exactly the same width and the same gate stream as the reference.
kickmix maps computational-basis states to computational-basis states, so the
mid-circuit register contents are well-defined classical values per shot even
though the register is in a superposition across shots.

Reported per forward-EEA iteration (max over shots, plus the per-shot maximum):

    span_a / span_b   index span `max_nz - min_nz + 1` of each polynomial
    nnz_a / nnz_b     number of non-zero coefficient slots
    deg_a / deg_b     highest occupied slot index (=-1 for the zero polynomial)

The packed-store width is `max_it (nnz_a + nnz_b)` (plus shift headroom); it is
the width a sliding / packed two-polynomial layout would need, against today's
`2 * m_e = 146` fixed slots.

Usage
-----
    python src/e4/regb_liveness.py --n 10 --shots 16
    python src/e4/regb_liveness.py --n 70 --shots 4 --json notes/runs/e4_liveness_n70.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py'

# --------------------------------------------------------------------------- #
# Source patch: dump reg_a + reg_b after every *forward* EEA iteration.
# --------------------------------------------------------------------------- #

ANCHOR_END = """                if not is_adjoint:
                    self._update_counter(builder, counter, flag, is_adjoint=False)
"""

PROBE_END = """
                if not is_adjoint and _dump[0] is not None:
                    _dump[0](builder, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag)
"""

ANCHOR_START = """                ctrl = self._counter_controls(builder, counter, is_c3, is_hc_zero)
"""

PROBE_START = """                ctrl = self._counter_controls(builder, counter, is_c3, is_hc_zero)
                if not is_adjoint and _dump[0] is not None:
                    _dump[0](builder, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag)
"""

_dump = [None]


def load_decoder_class(probe_at: str = 'end'):
    """Loads the reference module with the trace hook spliced into `_synchronized_eea`."""
    src = REF.read_text()
    if probe_at == 'end':
        if ANCHOR_END not in src:
            raise RuntimeError('anchor (end) not found in the reference source')
        src = src.replace(ANCHOR_END, ANCHOR_END + PROBE_END, 1)
    elif probe_at == 'start':
        if ANCHOR_START not in src:
            raise RuntimeError('anchor (start) not found in the reference source')
        src = src.replace(ANCHOR_START, PROBE_START, 1)
    else:
        raise ValueError(probe_at)
    ns: dict = {'__name__': 'traced_dqi_reference', '_dump': _dump}
    exec(compile(src, str(REF), 'exec'), ns)  # noqa: S102 - local trusted file
    return ns


def build_traced(n: int, degree: int, iters_cap: int | None = None, probe_at: str = 'end',
                 logm_extra: int = 1):
    """Builds `initial_shift + forward_eea` with a per-iteration register dump."""
    ns = load_decoder_class(probe_at)
    decoder_cls = ns['ProosAndZalkaDecoder']
    default_n_iters = ns['default_n_iters']

    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    # The Chien search is the only consumer of `error_mbuc` and cannot influence the
    # forward schedule, so a single 1-position stub keeps the build cheap.
    builder.create_classical_register(degree, name='E0')

    n_iters = default_n_iters(n)
    if iters_cap is not None:
        n_iters = min(n_iters, iters_cap)

    m = n + 3
    logm = m.bit_length() + logm_extra
    counts = {'a': m * degree, 'b': m * degree, 'meta': 5 * logm + 1, 'dumps': 0}

    def dump(b, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag):
        b.debug_print(km.array([q for r in reg_a for q in r]))
        b.debug_print(km.array([q for r in reg_b for q in r]))
        b.debug_print(km.array([*n_A, *n_B, *n_a, *n_b, *n_q, flag]))
        counts['dumps'] += 1

    decoder = decoder_cls(n, km.GF2Field(degree))
    reg_a = [builder.alloc_qubits(degree) if i < 3 else syndromes[i - 3] for i in range(m)]
    reg_b = [builder.alloc_qubits(degree) for _ in range(m)]
    n_A = builder.alloc_qubits(logm)
    n_B = builder.alloc_qubits(logm)
    n_a = builder.alloc_qubits(logm)
    n_b = builder.alloc_qubits(logm)
    n_q = builder.alloc_qubits(logm)
    flag = builder.alloc_qubits(1)[0]
    counter = builder.alloc_qubits(2)
    halting_counter = builder.alloc_qubits(n_iters.bit_length())

    builder.x(km.array([reg_a[0][0], reg_b[m - 1][0], n_a[0], flag]))
    builder.gf2_iadd(n, target=n_A)
    builder.gf2_iadd(n + 1, target=n_B)
    with builder.mark('initial_shift'):
        decoder._initial_shift_loop(builder, reg_a, n_A, is_adjoint=False)

    _dump[0] = dump
    try:
        with builder.mark('forward_eea'):
            decoder._synchronized_eea(builder, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag,
                                      counter, halting_counter, n_iters=n_iters,
                                      is_adjoint=False)
    finally:
        _dump[0] = None

    circuit = builder.finish_circuit()
    return circuit, syndromes, m, counts


def run_trace(n: int, degree: int, shots: int, seed: int,
              iters_cap: int | None = None, syndrome_mode: str = 'uniform',
              probe_at: str = 'end', logm_extra: int = 1):
    """Returns `(slot_a, slot_b, meta, n_dumps)` with shapes `(dumps, shots, m*degree)`."""
    circuit, syndromes, m, counts = build_traced(n, degree, iters_cap, probe_at, logm_extra)
    n_dumps = counts['dumps']
    # each `debug_print(...)` call also emits one empty DEBUG_PRINT terminator
    n_per = counts['a'] + counts['b'] + counts['meta'] + 3

    rng = np.random.default_rng(seed)
    n_syn = len(syndromes)
    if syndrome_mode == 'uniform':
        values = rng.integers(0, 1 << degree, size=(shots, n_syn), dtype=np.int64)
    elif syndrome_mode == 'sparse':
        # Realistic DQI syndromes: weight <= n/2 error words.
        order = (1 << degree) - 1
        logt = np.zeros(1 << degree, dtype=np.int64)
        antilog = np.zeros(order, dtype=np.int64)
        gf = km.GF2Field(degree)
        prim = gf.primitive_element
        for k in range(order):
            antilog[k] = gf.pow(prim, k)
        logt[antilog] = np.arange(order)
        values = np.zeros((shots, n_syn), dtype=np.int64)
        rows = np.arange(1, n_syn + 1, dtype=np.int64)[:, None]
        for s in range(shots):
            k = int(rng.integers(0, n_syn // 2 + 1))
            pos = rng.choice(max(n_syn, m), size=k, replace=False)
            pos = pos[pos < m]
            vals = rng.integers(1, order + 1, size=len(pos), dtype=np.int64)
            exps = (logt[vals][None, :] + rows * pos[None, :]) % order
            values[s] = np.bitwise_xor.reduce(antilog[exps], axis=1)
    else:
        raise ValueError(syndrome_mode)

    bits = ((values[:, :, None] >> np.arange(degree)) & 1 == 1)
    bits = bits.transpose(1, 2, 0).reshape(-1, shots)

    got: list[np.ndarray] = []

    def cb(target, target_vals, cond, cond_vals):
        got.append(np.asarray(target_vals))

    sim = km.Simulator(batch_size=shots, debug_print=cb, seed=seed)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(km.array([q for reg in syndromes for q in reg]), bits)
    sim.do(circuit)

    if len(got) != n_dumps * n_per:
        raise RuntimeError(f'expected {n_dumps * n_per} debug events, got {len(got)}')
    arr = np.stack(got[:n_dumps * n_per]).reshape(n_dumps, n_per, shots)
    a = arr[:, :counts['a'], :]
    b = arr[:, counts['a']:counts['a'] + counts['b'], :]
    meta = arr[:, counts['a'] + counts['b']:counts['a'] + counts['b'] + counts['meta'], :]
    return a, b, meta, n_dumps


def word_values(flat_bits: np.ndarray, degree: int) -> np.ndarray:
    """`(dumps, slots*degree, shots)` bit slice -> `(dumps, slots, shots)` integers."""
    d, nbits, shots = flat_bits.shape
    b = flat_bits.reshape(d, nbits // degree, degree, shots).astype(np.int64)
    weights = (1 << np.arange(degree)).reshape(1, 1, degree, 1).astype(np.int64)
    return (b * weights).sum(axis=2)


def summarize(n: int, degree: int, shots: int, seed: int,
              iters_cap: int | None = None, syndrome_mode: str = 'uniform',
              probe_at: str = 'end', logm_extra: int = 1) -> dict:
    a, b, meta, n_dumps = run_trace(n, degree, shots, seed, iters_cap, syndrome_mode, probe_at,
                                    logm_extra)
    m = n + 3
    wa = word_values(a, degree)      # (dumps, m, shots)
    wb = word_values(b, degree)
    nz_a = wa != 0
    nz_b = wb != 0

    nnz_a = nz_a.sum(axis=1)         # (dumps, shots)
    nnz_b = nz_b.sum(axis=1)
    idx = np.arange(m).reshape(1, m, 1)
    hi_a = np.where(nz_a, idx, -1).max(axis=1)
    lo_a = np.where(nz_a, idx, m).min(axis=1)
    hi_b = np.where(nz_b, idx, -1).max(axis=1)
    lo_b = np.where(nz_b, idx, m).min(axis=1)
    span_a = np.where(nnz_a > 0, hi_a - lo_a + 1, 0)
    span_b = np.where(nnz_b > 0, hi_b - lo_b + 1, 0)

    logm = m.bit_length() + logm_extra

    def meta_word(k: int) -> np.ndarray:
        return word_values(meta[:, k * logm:(k + 1) * logm, :], logm)[:, 0, :]

    n_A = meta_word(0)
    n_B = meta_word(1)
    n_a = meta_word(2)
    n_b = meta_word(3)
    n_q = meta_word(4)

    return {
        'n': n, 'degree': degree, 'm_e': m, 'shots': shots, 'seed': seed, 'probe_at': probe_at,
        'syndrome_mode': syndrome_mode, 'n_dumps': n_dumps,
        'iters': n_dumps,
        'nnz_a_max': int(nnz_a.max()), 'nnz_a_mean': float(nnz_a.mean()),
        'nnz_b_max': int(nnz_b.max()), 'nnz_b_mean': float(nnz_b.mean()),
        'span_a_max': int(span_a.max()), 'span_b_max': int(span_b.max()),
        'deg_a_max': int(hi_a.max()), 'deg_b_max': int(hi_b.max()),
        'packed_max': int((nnz_a + nnz_b).max()),
        'packed_mean': float((nnz_a + nnz_b).mean()),
        'packed_per_iter_max': (nnz_a + nnz_b).max(axis=1).tolist(),
        'nnz_a_per_iter_max': nnz_a.max(axis=1).tolist(),
        'nnz_b_per_iter_max': nnz_b.max(axis=1).tolist(),
        'span_a_per_iter_max': span_a.max(axis=1).tolist(),
        'span_b_per_iter_max': span_b.max(axis=1).tolist(),
        'n_A_first_last': [int(n_A[0].max()), int(n_A[-1].max())],
        'n_B_first_last': [int(n_B[0].max()), int(n_B[-1].max())],
        'n_a_first_last': [int(n_a[0].max()), int(n_a[-1].max())],
        'n_b_first_last': [int(n_b[0].max()), int(n_b[-1].max())],
        'fixed_slots_today': 2 * m,
        'degree_sum_invariant_violations': int(
            ((hi_a + 1) + (hi_b + 1) > n + 2).sum()
        ),
        'degree_sum_max': int(((hi_a + 1) + (hi_b + 1)).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--iters-cap', type=int, default=None)
    ap.add_argument('--mode', choices=('uniform', 'sparse'), default='uniform')
    ap.add_argument('--probe-at', choices=('start', 'end'), default='end')
    ap.add_argument('--json', type=str, default=None)
    args = ap.parse_args()

    out = summarize(args.n, args.degree, args.shots, args.seed, args.iters_cap, args.mode,
                    args.probe_at)
    head = {k: v for k, v in out.items() if not k.endswith('_per_iter_max')}
    for k, v in head.items():
        print(f'{k:36s} {v}')
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

"""E4: adder ancilla/Toffoli trade-off for the invsqrt circuit.

The E3 note left the adder ancilla term as an unexplained constant ("inside the
compiled kickmix").  `invsqrt_qbreak.py` shows Q = persistent (211) + transient,
where the transient is the widest `iadd`/`isub` in the circuit: the parallel-prefix
adder built into kickmix costs **L-1 Toffolis and L-2 ancillas** on an L-bit slice
(L-1 ancillas when the offset is a classical constant, since there is no source
register to hold carries).

This file asks the only question that matters for a further qubit diet: *can the
ancilla be bought down with Toffolis, and at what price?*

Every candidate is checked **against kickmix's own `iadd`** on the full input space
for small L, using the shipped semantics:
    |a> |t> |0...>  ->  |a> |t + a mod 2^L> |0...>      (source preserved, scratch clean)

Usage
-----
    ../2609-xingzhi-drug/.venv/bin/python src/e4/invsqrt_adders.py
    ../2609-xingzhi-drug/.venv/bin/python src/e4/invsqrt_adders.py --out notes/runs/e4_adders.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import kickmix as km  # noqa: E402
import numpy as np  # noqa: E402


# --------------------------------------------------------------------------- #
# candidate adders
# --------------------------------------------------------------------------- #

def _maj(b, a, c, t):
    """Classical majority MAJ(a, c, t): t <- maj, a <- a^t_old, c <- c^t_old."""
    b.cx(t, c)
    b.cx(t, a)
    b.ccx(a, c, t)


def _uma(b, a, c, t):
    """Exact inverse of `_maj`."""
    b.ccx(a, c, t)
    b.cx(t, a)
    b.cx(t, c)


def cuccaro(b: km.CircuitBuilder, src: km.array, tgt: km.array) -> None:
    """Ripple-carry add with ONE carry ancilla (Cuccaro/Draper-Kutin-Moulton style).

    Up-sweep propagates the carry through the source qubits, down-sweep restores
    them and deposits the sum bits in the target.
    """
    L = len(src)
    carry = b.alloc_qubits(1)
    for i in range(L):
        if i == L - 1:
            b.cx(carry, tgt[i])
        else:
            _maj(b, src[i], tgt[i], carry)
    for i in reversed(range(L)):
        if i == L - 1:
            b.cx(carry, tgt[i])
        else:
            _uma(b, src[i], tgt[i], carry)
    b.free(carry)


def ripple2(b: km.CircuitBuilder, src: km.array, tgt: km.array) -> None:
    """Textbook ripple carry with an explicit carry ancilla per bit (0 optimisation)."""
    L = len(src)
    carrys = b.alloc_qubits(L + 1)
    for i in range(L):
        b.ccx(src[i], tgt[i], carrys[i + 1])
        b.cx(carrys[i], carrys[i + 1])
        b.cx(src[i], tgt[i])
        b.cx(tgt[i], carrys[i + 1])
    # note: leave the landed sum, then unwind the carries
    for i in range(L):
        b.cx(src[i], tgt[i])
    b.free(carrys)


def builtin(b: km.CircuitBuilder, src: km.array, tgt: km.array) -> None:
    b.iadd(src, target=tgt)


CANDS = {'builtin': builtin, 'cuccaro': cuccaro}


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def build(builder_fn, L: int, *, named: bool = True) -> km.Circuit:
    b = km.CircuitBuilder()
    if named:
        a = b.create_quantum_register(L, name='a')
        t = b.create_quantum_register(L, name='t')
    else:
        a, t = b.alloc_qubits(L), b.alloc_qubits(L)
    builder_fn(b, a, t)
    return b.finish_circuit(), a, t


def measure(builder_fn, L: int) -> dict:
    c, a, t = build(builder_fn, L, named=False)
    st = c.operation_statistics()
    return {'L': L, 'ancilla': c.num_qubits - 2 * L, 'toffoli': st.num_toffoli_like,
            'instructions': st.num_instructions, 'non_clifford': st.num_non_clifford}


def check_semantics(builder_fn, L: int) -> dict:
    """Exhaustive check against `t := t + a mod 2^L` with source preserved."""
    c, a, t = build(builder_fn, L, named=True)
    pairs = np.array(list(itertools.product(range(1 << L), repeat=2)), dtype=np.uint64)
    x = pairs[:, 0]
    y = pairs[:, 1]
    bits_a = (x[None, :] >> np.arange(L, dtype=np.uint64)[:, None]) & 1 == 1
    bits_t = (y[None, :] >> np.arange(L, dtype=np.uint64)[:, None]) & 1 == 1
    nn = len(x)
    sim = km.Simulator(batch_size=nn, debug_print=False, counting_operations=True, seed=7)
    sim.clear_for_shot()
    sim.write_across_shots(a, bits_a)
    sim.write_across_shots(t, bits_t)
    sim.do(c)
    ra = sim.read_across_shots(a).astype(np.uint64)
    rt = sim.read_across_shots(t).astype(np.uint64)
    got_a = (ra << np.arange(L, dtype=np.uint64)[:, None]).sum(axis=0)
    got_t = (rt << np.arange(L, dtype=np.uint64)[:, None]).sum(axis=0)
    want = (x + y) % (1 << L)
    scratch = km.array([km.q(i) for i in range(c.num_qubits)
                        if km.q(i) not in set(list(a) + list(t))])
    dirty = sim.read_across_shots(scratch).any(axis=1)
    return {'L': L, 'source_preserved': bool((got_a == x).all()),
            'sum_ok': bool((got_t == want).all()),
            'scratch_clean': bool(not dirty.any()),
            'n_bad_sum': int((got_t != want).sum())}


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'notes' / 'runs' / 'e4_adders.json')
    ap.add_argument('--max-check-L', type=int, default=5)
    args = ap.parse_args()

    print('=== semantics (exhaustive vs t := t + a mod 2^L) ===')
    sem = {}
    for name, fn in CANDS.items():
        rows = []
        ok = True
        for L in range(1, args.max_check_L + 1):
            r = check_semantics(fn, L)
            rows.append(r)
            ok &= r['sum_ok'] and r['source_preserved'] and r['scratch_clean']
        sem[name] = rows
        r = rows[-1]
        print(f'  {name:10s} L={args.max_check_L}: sum_ok={r["sum_ok"]} '
              f'preserved={r["source_preserved"]} clean={r["scratch_clean"]} -> {"PASS" if ok else "FAIL"}')

    print('\n=== cost vs slice width ===')
    widths = [3, 4, 5, 6, 8, 12, 16, 17, 20, 24, 27, 28, 29, 30, 32, 40, 64]
    table = {}
    for name, fn in CANDS.items():
        table[name] = [measure(fn, L) for L in widths]
    hdr = f'{"L":>3s} ' + ' '.join(f'{n:>16s}' for n in CANDS)
    print(hdr)
    for i, L in enumerate(widths):
        cells = []
        for name in CANDS:
            r = table[name][i]
            cells.append(f'{r["ancilla"]:>6d}anc/{r["toffoli"]:>4d}tof')
        print(f'{L:3d} ' + ' '.join(f'{c:>16s}' for c in cells))

    print('\n=== delta vs builtin (ancilla saved / Toffoli added) ===')
    print(f'{"L":>3s} {"danc":>5s} {"dtof":>5s} {"per-op test":>12s}')
    deltas = []
    for i, L in enumerate(widths):
        danc = table['builtin'][i]['ancilla'] - table['cuccaro'][i]['ancilla']
        dtof = table['cuccaro'][i]['toffoli'] - table['builtin'][i]['toffoli']
        worth = dtof < 32 * danc
        deltas.append({'L': L, 'danc': danc, 'dtof': dtof, 'worth': worth})
        print(f'{L:3d} {danc:5d} {dtof:5d}   {"worth it" if worth else "not worth":>12s}')

    # WARNING: the per-op column above is NOT the circuit-level verdict.  Q's transient term is a
    # high-water mark over the 180 iadd/isub calls, and the adder ancillas are allocated and
    # released inside each call, so converting *every* call to 1 ancilla lowers Q by
    # (peak transient - 1) ONCE -- it does not scale with the op count.  The real test therefore
    # weighs sum(L-1) extra Toffolis against 32 * (peak transient - 1).
    print('\n=== circuit-level verdict (the per-op column above is NOT the verdict) ===')
    qbreak_path = ROOT / 'notes' / 'runs' / 'e4_invsqrt_qbreak.json'
    if qbreak_path.exists():
        ops = json.loads(qbreak_path.read_text())['ops']
        peak = json.loads(qbreak_path.read_text())['identity']['max_transient']
        extra = sum(o['target_len'] - 1 for o in ops)
        saved = peak - 1
        print(f'{len(ops)} iadd/isub calls; swapping all of them for a 1-ancilla adder:')
        print(f'   Toffoli   +{extra:,}   (sum of L-1)')
        print(f'   qubits    -{saved}       (transient {peak} -> 1)')
        print(f'   budget    32 x {saved} = {32 * saved:,} Toffolis')
        verdict = 'WORTH IT' if extra < 32 * saved else 'LOSS'
        print(f'   verdict   {verdict}: {extra / (32 * saved):.2f}x over budget')
        global_swap = {'ops': len(ops), 'extra_toffoli': extra, 'qubits_saved': saved,
                       'budget': 32 * saved, 'verdict': verdict,
                       'over_budget_factor': round(extra / (32 * saved), 3)}
    else:
        print(f'   (run {qbreak_path.name} first to get the op ledger)')

    out = {'semantics': sem, 'cost': table, 'deltas': deltas}
    out['global_swap'] = locals().get('global_swap')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

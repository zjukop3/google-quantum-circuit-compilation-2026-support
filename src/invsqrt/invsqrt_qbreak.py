"""E4: exact qubit (Q) anatomy of the optimized invsqrt circuit (Benchmark I).

Three independent measurements, all on the *real* circuit:

1. **Allocation ledger** -- every named register, its width, and its role.  The
   optimized builder allocates all registers up front and never frees, so the
   sum of the widths is the *persistent* width.

2. **Per-operation peak trace** -- `CircuitBuilder.finish_circuit()` is
   non-destructive and can be called mid-build, so `finish_circuit().num_qubits`
   is a free oracle for the running peak.  Wrapping `iadd`/`isub` (the only two
   primitives that allocate transient scratch) and probing after every call
   attributes Q to the exact operation that sets it.  The probe costs
   ~0.7 us/call -- 180 calls for the whole circuit.

3. **Adder micro-benchmark** -- isolated `iadd`/`isub` on an `L`-bit slice,
   swept over L, giving the transient ancilla cost `a(L)` as a function of the
   slice width.  This is the term the E3 note left as "inside the compiled
   kickmix"; here it is measured, not assumed.

The identity that must close:  `Q = persistent + max_over_ops a(L_op)`.

Usage
-----
    ../2609-xingzhi-drug/.venv/bin/python src/e4/invsqrt_qbreak.py \
        --out notes/runs/e4_invsqrt_qbreak.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'invsqrt'))

import kickmix as km  # noqa: E402

from nr_reference_circuit import Precision  # noqa: E402
import invsqrt_opt_circuit as O  # noqa: E402

BEST = dict(delta_bits=12, coef_bits=17, square_bits=28, product_bits=27, result_bits=28)
N_P = 10


# --------------------------------------------------------------------------- #
# 1. persistent allocation ledger
# --------------------------------------------------------------------------- #

def persistent_ledger(prec: Precision, n_p: int = N_P) -> dict:
    """Declared widths of every register the builder allocates up front."""
    width = 2 * n_p + 2
    out_bits = prec.output_bits(n_p)
    scratch_len = max(prec.square_bits, prec.product_bits + 2, prec.result_bits + 1)
    rows = [
        ('x', width, 'datapath', 'momentum pair; normalized in place, then un-normalized'),
        ('y1', out_bits, 'datapath',
         f'output p 2^s; out_bits = result_bits + n_p + 1 = {prec.result_bits}+{n_p}+1'),
        ('flags', n_p, 'control', 'unary leading-zero-pair count s (1 flag per shift step)'),
        ('selectors', 3, 'control', 't_a, t_b (interval) + half (delta 1/2 bit)'),
        ('coef a', prec.coef_bits, 'datapath', 'c0 slot -> y0'),
        ('coef b', prec.coef_bits, 'datapath', 'c3 -> c1 -> r2'),
        ('coef c', prec.coef_bits, 'datapath', 'c2 -> r1'),
        ('square', prec.square_bits, 'datapath', 'y0^2'),
        ('z', prec.product_bits + 2, 'datapath', 'M y0^2 then 3 - that (2 integer bits)'),
        ('scratch', scratch_len, 'ancilla', 'AND-product scratch, aligned with target'),
    ]
    total = sum(w for _, w, _, _ in rows)
    return {'rows': rows, 'persistent': total, 'out_bits': out_bits,
            'scratch_len': scratch_len, 'width': width}


# --------------------------------------------------------------------------- #
# 2. per-operation peak trace
# --------------------------------------------------------------------------- #

class Probe:
    """Wrap the builder primitives that can raise the peak; probe the peak after each."""

    def __init__(self):
        self.stack: list[str] = []
        self.ops: list[dict] = []

    def install(self) -> None:
        B = km.CircuitBuilder
        self._mark, self._iadd, self._isub = B.mark, B.iadd, B.isub
        _mark, _iadd, _isub = self._mark, self._iadd, self._isub

        probe = self

        class MarkCtx:
            def __init__(self, inner, label):
                self.inner, self.label = inner, label

            def __enter__(self):
                probe.stack.append(self.label)
                return self.inner.__enter__()

            def __exit__(self, *a):
                probe.stack.pop()
                return self.inner.__exit__(*a)

        def mark(self, name):
            return MarkCtx(_mark(self, name), name)

        def make(fn, name):
            def g(self, offset, *, target, control=True, btol=float('inf')):
                before_alloc = self.num_allocated_qubits
                fn(self, offset, target=target, control=control, btol=btol)
                try:
                    L = len(offset)
                except TypeError:
                    L = 0  # constant offset (e.g. `correct`'s 3 - z)
                probe.ops.append({
                    'op': name, 'label': '/'.join(probe.stack), 'slice': L,
                    'target_len': len(target), 'alloc_before': before_alloc,
                    'peak_after': self.finish_circuit().num_qubits,
                })
            return g

        B.mark = mark
        B.iadd = make(_iadd, 'iadd')
        B.isub = make(_isub, 'isub')

    def uninstall(self) -> None:
        B = km.CircuitBuilder
        B.mark, B.iadd, B.isub = self._mark, self._iadd, self._isub


def peak_trace(prec: Precision, n_p: int = N_P) -> dict:
    probe = Probe()
    probe.install()
    try:
        c = O.build_opt_circuit(n_p, prec=prec)
    finally:
        probe.uninstall()
    ops = probe.ops
    peak = c.num_qubits

    # first op to reach the final peak, and the peak path
    first_at_peak = next((i for i, o in enumerate(ops) if o['peak_after'] == peak), None)
    running = 0
    growth = []
    for i, o in enumerate(ops):
        if o['peak_after'] > running:
            growth.append({'op_index': i, 'op': o['op'], 'label': o['label'],
                           'slice': o['slice'], 'target_len': o['target_len'],
                           'peak': o['peak_after'], 'alloc_before': o['alloc_before']})
            running = o['peak_after']

    by_label = collections.defaultdict(lambda: {'n': 0, 'max_slice': 0, 'max_peak': 0})
    for o in ops:
        d = by_label[o['label']]
        d['n'] += 1
        d['max_slice'] = max(d['max_slice'], o['slice'])
        d['max_peak'] = max(d['max_peak'], o['peak_after'])
    return {'peak': peak, 'ops': ops, 'growth': growth,
            'first_at_peak': first_at_peak,
            'by_label': dict(sorted(by_label.items()))}


# --------------------------------------------------------------------------- #
# 3. adder micro-benchmark
# --------------------------------------------------------------------------- #

def adder_cost(L: int, op: str = 'iadd') -> dict:
    """Transient peak of an isolated `op` on an L-bit slice, with nothing else live."""
    b = km.CircuitBuilder()
    src = b.alloc_qubits(L)
    tgt = b.alloc_qubits(L)
    base = b.finish_circuit().num_qubits
    getattr(b, op)(src, target=tgt)
    c = b.finish_circuit()
    st = c.operation_statistics()
    return {'L': L, 'peak': c.num_qubits, 'base': base, 'ancilla': c.num_qubits - 2 * L,
            'toffoli': st.num_toffoli_like, 'instructions': st.num_instructions}


def adder_table(lo: int = 3, hi: int = 70) -> list[dict]:
    rows = []
    for L in range(lo, hi + 1):
        add = adder_cost(L, 'iadd')
        sub = adder_cost(L, 'isub')
        rows.append({'L': L, 'iadd_ancilla': add['ancilla'], 'isub_ancilla': sub['ancilla'],
                     'iadd_toffoli': add['toffoli'], 'isub_toffoli': sub['toffoli']})
    return rows


def and_cost() -> dict:
    """`init_and` / `del_and` price -- the other structural primitive."""
    b = km.CircuitBuilder()
    r = b.alloc_qubits(3)
    b.init_and(r[0], r[1], r[2])
    c = b.finish_circuit()
    st = c.operation_statistics()
    return {'peak': c.num_qubits, 'toffoli': st.num_toffoli_like}

def transient(op: dict) -> int:
    """Transient scratch of one recorded op: L-2 for a slice add, |target|-1 for a constant."""
    if op['slice']:
        return max(0, op['slice'] - 2)
    return max(0, op['target_len'] - 1)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-p', type=int, default=N_P)
    ap.add_argument('--out', type=Path, default=ROOT / 'notes' / 'runs' / 'e4_invsqrt_qbreak.json')
    args = ap.parse_args()

    prec = Precision(**BEST)
    led = persistent_ledger(prec, args.n_p)
    trace = peak_trace(prec, args.n_p)

    print(f'=== persistent ledger (prec {BEST}) ===')
    print(f'{"register":10s} {"width":>5s}  {"kind":8s} role')
    for name, w, kind, role in led['rows']:
        print(f'{name:10s} {w:5d}  {kind:8s} {role}')
    print(f'{"TOTAL":10s} {led["persistent"]:5d}   (declared, persistent)')

    print(f'\n=== per-operation peak trace ===')
    print(f'measured circuit Q = {trace["peak"]}')
    for g in trace['growth']:
        print(f'  #{g["op_index"]:>3d}  peak -> {g["peak"]:>3d}  {g["op"]:5s} L={g["slice"]:>2d} '
              f'|target|={g["target_len"]:>2d} alloc={g["alloc_before"]:>3d}  [{g["label"]}]')

    rows = adder_table(3, 70)
    print('\n=== adder transient ancilla (isolated) ===')
    print(f'{"L":>3s} {"iadd_anc":>8s} {"isub_anc":>8s} {"iadd_toff":>9s}')
    for r in rows:
        if r['L'] in range(3, 12) or r['L'] in (16, 24, 27, 28, 29, 30, 32, 39, 40, 64, 70):
            print(f'{r["L"]:3d} {r["iadd_ancilla"]:8d} {r["isub_ancilla"]:8d} '
                  f'{r["iadd_toffoli"]:9d}')
    print('closed-form check: ancilla == max(0, L-2)?',
          all(max(0, r['L'] - 2) == r['iadd_ancilla'] for r in rows))

    trans = [transient(o) for o in trace['ops']]
    worst = max(trans)
    predicted = led['persistent'] + worst
    print(f'\nidentity  Q = persistent + max transient = {led["persistent"]} + {worst} = '
          f'{predicted} (measured {trace["peak"]})')

    # Which ops exceed a candidate transient cap?
    print('\n=== transient histogram over all iadd/isub ops (both passes) ===')
    hist = collections.Counter(trans)
    for k in sorted(hist, reverse=True)[:12]:
        print(f'  transient {k:3d}  x{hist[k]:3d}   (slice L={k+2})')
    print('\n=== ops exceeding a cap k (these must be replaced to reach that peak) ===')
    print(f'{"cap k":>6s} {"#ops> k":>8s} {"worst L":>8s} {"full-sweep Toffoli delta (1-ancilla)":>36s}')
    for k in (26, 24, 20, 16, 12, 8, 6, 4, 2, 1):
        bad = [o for o, t in zip(trace['ops'], trans) if t > k]
        # replacing with a 1-ancilla adder costs L-1 extra Toffolis per call
        delta = sum(max(0, o['slice'] - 1) if o['slice'] else max(0, o['target_len'] - 2)
                    for o in bad)
        print(f'{k:6d} {len(bad):8d} {max((o["slice"] or o["target_len"]) for o in bad) if bad else 0:8d}'
              f'{delta:36d}')

    out = {'n_p': args.n_p, 'precision': BEST, 'ledger': led,
           'peak': trace['peak'], 'growth': trace['growth'], 'ops': trace['ops'],
           'first_at_peak': trace['first_at_peak'],
           'by_label': trace['by_label'], 'adder_table': rows,
           'and_cost': and_cost(),
           'identity': {'persistent': led['persistent'],
                        'max_transient': worst, 'predicted': predicted,
                        'measured': trace['peak']}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

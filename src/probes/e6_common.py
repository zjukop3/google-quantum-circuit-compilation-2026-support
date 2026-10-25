"""E6 shared machinery: local forward-only EEA probe with an *event-pattern* trace.

Question this exists to answer
------------------------------
E3 measured that the dqi forward-EEA schedule deviates from the generic one with
probability ~8.9e-3, that the deviation is *born* at the sub-step-1 branch
`flip_if_equal(reg_b[m-2], 0, target=flag, control=aux)` with
`aux = c1 AND NOT(n_B == 1)`, and that once born it never disappears.

What E3 did **not** establish is what the deviation *is* as a function of the
data.  This module makes that measurable.  Per forward iteration `it` it records

  * `z1[it]` -- toggled on every iteration where the sub-step-1 branch's own
    enable signal (`c1 AND NOT(n_B == 1)`) is 1.  XOR is self-inverse, so the
    final value is the *parity* of the number of firings.
  * `z2[it]` -- the same for the sub-step-2 branch (`c2 AND NOT(n_b == 0)`).
  * `rb[it]` -- snapshot of `reg_b[m-2] == 0` at the top of the iteration
    (unconditional: independent of `c1`), so the classical model can be
    validated against the circuit without relying on the branch being taken.
  * `nB[it]`, `ctrl[it]` -- `n_B` and the one-hot sub-step decode, for the same
    purpose.

Every probe gate is a `cx` / `flip_if_equal` / `multi_controlled_x` on freshly
allocated scratch qubits and touches no algorithm register, so the traced
circuit follows exactly the same trajectory as the untraced one
(`e6_deviant_trace.py --compare-no-probe` asserts this).

Usage
-----
    python src/e6/e6_deviant_trace.py --n 70 --shots 64
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py'
SRC = REF.read_text()

# --------------------------------------------------------------------------- #
# source patching
# --------------------------------------------------------------------------- #

CTRL_ANCHOR = ('                ctrl = self._counter_controls(builder, counter, is_c3, '
               'is_hc_zero)')

# top of every forward iteration: unconditional snapshots (no branch dependence)
ITER_PROBE = '''                if not is_adjoint and getattr(self, '_pat', None) is not None:
                    _p = self._pat
                    builder.cx(ctrl, _p['ctrl'][it])
                    _cp = [ctrl[_k] for _k in range(4)]
                    if not is_adjoint:
                        builder.cx(_cp[0], _p['ctrlA'][it][0])
                        builder.cx(_cp[1], _p['ctrlA'][it][1])
                        builder.cx(_cp[2], _p['ctrlA'][it][2])
                        builder.cx(_cp[3], _p['ctrlA'][it][3])
                    builder.cx(km.array([flag]), _p['flg'][it])
                    if getattr(self, '_frozen_target', None) is not None and self._frozen_target[it]:
                        builder.x(_p['fz'][it][0])
                    builder.cx(km.array([is_hc_zero]), _p['hcz'][it])
                    builder.flip_if_equal(reg_b[len(reg_b) - 2], 0, target=_p['rb'][it][0])
                    builder.flip_if_equal(reg_b[0], 0, target=_p['rb0'][it][0])
'''

FREE_ANCHOR = '            builder.free(aux)\n'

FREE_PROBE = '''            if getattr(self, '_pat', None) is not None:
                _p = self._pat
                builder.cx(aux, _p['z1'][it])
'''

# expose c2 for the sub-step-2 probe (before it is folded into `flag`)
C2_ANCHOR = '            builder.cx(c2, flag)\n            builder.flip_if_equal(n_b, 0'
C2_PROBE = '''            if getattr(self, '_pat', None) is not None:
                builder.cx(c2, self._pat['c2g'][it][0])
            builder.cx(c2, flag)
            builder.flip_if_equal(n_b, 0'''

# --- frozen-schedule mode: replace the quantum `mul` qubit by a compile-time
# --- constant taken from a recorded generic-run target bitmap.
MUL_ANCHOR = '''                    mul = builder.alloc_qubits(1)[0]
'''
MUL_FROZEN = '''                    mul = builder.alloc_qubits(1)[0]
                    _ft = getattr(self, 'frozen_target', None)
                    _frozen = (_ft is not None
                               and bool((_ft[1] if is_adjoint else _ft[0])[it]))
'''

MUL_PROBE = '''                    if getattr(self, '_mul_probe', None) is not None:
                        builder.cx(mul, self._mul_probeA[it][0] if is_adjoint
                                   else self._mul_probe[it][0])'''

CORE_MULT = '''                        if is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], field=gf,
                                                 control=mul)
'''
CORE_MULT_FROZEN = '''                        if not _frozen and is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], field=gf,
                                                 control=mul)
'''

CORE_MULT2 = '''                        if not is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], field=gf,
                                                 control=mul)
'''
CORE_MULT2_FROZEN = '''                        if not _frozen and not is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], field=gf,
                                                 control=mul)
'''

CSWAP_MULT = '                            builder.cswap(swaps[0], q_i, reg_a[i])'
CSWAP_MULT_FROZEN = '                            builder.cswap(mul if _frozen else swaps[0], q_i, reg_a[i])'

# The sweep tail.  The probe goes BETWEEN `endpoint(prev)` and the final
# `free(mul)`, i.e. while `mul` still holds the iteration's liveness value --
# `endpoint(prev)` uncomputes it, so a probe placed after it reads a hard 0.
TAIL_TAIL = '                    endpoint(prev)\n                    builder.free(mul)'
TAIL_TAIL_PROBED = (MUL_PROBE + '\n                    endpoint(prev)'
                    + '\n                    builder.free(mul)')


# frozen-mode tail: in frozen mode the sweep's `mul` is never used as an
# algorithm qubit, so the real tail's `free(mul)` is dropped.
BREAK_MULT = '''                    endpoint(prev)
                    builder.free(mul)'''
BREAK_MULT_FROZEN = '''                    endpoint(prev)'''

WRAP_ANCHOR = '    def _initial_shift_loop('

PRELIM = '''

def _pat_iadd1(builder, target, *controls):
    """XOR-increment `target` by 1, gated on all of `controls` (qubits)."""
    for j in range(len(target)):
        builder.multi_controlled_x(km.array(list(controls)
                                           + [target[k] for k in range(j)]), target[j])

'''

WRAP = '''
    def _alloc_pat_probe(self, builder, n_iters, logm):
        q = builder.create_quantum_register

        self._pat = {
            'z1': [q(1, name=f'Pz1_{it}') for it in range(n_iters)],
            'c2g': [q(1, name=f'Pc2g_{it}') for it in range(n_iters)],
            'ctrl': [q(4, name=f'Pctl_{it}') for it in range(n_iters)],
            'ctrl2': [q(4, name=f'Pctl2_{it}') for it in range(n_iters)],
            'ctrlA': [q(4, name=f'PctlA_{it}') for it in range(n_iters)],
            'fz': [q(1, name=f'Pfz_{it}') for it in range(n_iters)],
            'muls': [q(1, name=f'Pmul_{it}') for it in range(n_iters)],
            'mulsA': [q(1, name=f'PmulA_{it}') for it in range(n_iters)],
            'flg': [q(1, name=f'Pflg_{it}') for it in range(n_iters)],
            'hcz': [q(1, name=f'Phcz_{it}') for it in range(n_iters)],
            'rb': [q(1, name=f'Prb_{it}') for it in range(n_iters)],
            'rb0': [q(1, name=f'Prb0_{it}') for it in range(n_iters)],
        }

    def build_syndrome_decoding_mbuc(self, builder, *, syndromes, error_mbuc, n_iters):
        self._pat = None
        self._probe_regs = []
        self.probe_qubit_lo = builder.num_allocated_qubits
        if getattr(self, 'pat_probe_enabled', True):
            self._alloc_pat_probe(builder, n_iters, logm=None)
        self._mul_probe = self._pat['muls'] if self._pat else None
        self._mul_probeA = self._pat['mulsA'] if self._pat else None
        self.probe_qubit_hi = builder.num_allocated_qubits
        self.probe_reg_names = [q for r in syndromes for q in r]
        self.probe_set = set()
        if self._pat:
            for v in self._pat.values():
                for r in (v if isinstance(v, list) else [v]):
                    self.probe_set.update(r)
        self._orig_build_syndrome_decoding_mbuc(
            builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters)


'''

TAIL_START = "        with builder.mark('chien_forney_mbuc'):"
TAIL_END = "            builder.free(reg)\n"


DEL_ANCHOR = '                self._del_counter_controls(builder, counter, is_c3, is_hc_zero, ctrl)'
DEL_PROBE = '''                if not is_adjoint and getattr(self, '_pat', None) is not None:
                    builder.cx(ctrl, self._pat['ctrl2'][it])
                self._del_counter_controls(builder, counter, is_c3, is_hc_zero, ctrl)'''

FLAG_ANCHOR = ('                     n_q: km.array, flag, ctrl) -> None:\n'
               '        """Toggles `flag` when the active EEA sub-step reaches its '
               'termination condition."""')
FLAG_SIG = ('                     n_q: km.array, flag, ctrl, it: int) -> None:\n'
            '        """Toggles `flag` when the active EEA sub-step reaches its '
            'termination condition."""')

CALL_ANCHOR = ('lambda: self._update_flag(builder, reg_b, n_A, n_B, n_a, n_b, n_q,\n'
               '                                              flag, ctrl),')
CALL_FIXED = ('lambda: self._update_flag(builder, reg_b, n_A, n_B, n_a, n_b, n_q,\n'
              '                                              flag, ctrl, it),')


def build_probe_source(full: bool = False) -> str:
    src = SRC
    for anchor in (CTRL_ANCHOR, FREE_ANCHOR, C2_ANCHOR, WRAP_ANCHOR, FLAG_ANCHOR,
                   CALL_ANCHOR, MUL_ANCHOR):
        assert anchor in src, f'anchor missing: {anchor!r}'
    # truncate the tail FIRST: `TAIL_START`/`TAIL_END` must be located in the
    # original source, before the wrap inserts any extra method.
    if full:
        # drop only the Chien search (the sole consumer of `error_mbuc`): keep the
        # reverse EEA and the uncompute of the initial shift, so the syndrome
        # restoration contract is directly checkable on this circuit.
        a = src.index(TAIL_START)
        b = src.index('        with builder.mark(\'reverse_eea\'):', a)
        src = src[:a] + src[b:]
    else:
        start = src.index(TAIL_START)
        end = src.index(TAIL_END, start) + len(TAIL_END)
        src = src[:start] + '        # [E6 local forward-only probe] Chien/reverse/unshift omitted\n' + src[end:]
    src = src.replace(FLAG_ANCHOR, FLAG_SIG, 1)
    src = src.replace(CALL_ANCHOR, CALL_FIXED, 1)
    src = src.replace(CTRL_ANCHOR, CTRL_ANCHOR + '\n' + ITER_PROBE.rstrip('\n'), 1)
    src = src.replace(DEL_ANCHOR, DEL_PROBE, 1)
    src = src.replace(C2_ANCHOR, C2_PROBE.strip('\n'), 1)
    assert CORE_MULT in src and CORE_MULT2 in src and CSWAP_MULT in src
    src = src.replace(MUL_ANCHOR, MUL_FROZEN, 1)
    src = src.replace(CORE_MULT, CORE_MULT_FROZEN, 1)
    src = src.replace(CORE_MULT2, CORE_MULT2_FROZEN, 1)
    # NB: the sweep's `cswap` is controlled by a *position* signal
    # (`_active(eq_i, i, lim_q)`), NOT by `mul`, so it stays untouched: the frozen
    # schedule only removes the multiplier, never the data movement.

    # Two distinct tails: the live one (`builder.free(mul)`) and the frozen
    # template rendered into the source (`if not _frozen:`).  Patch both.
    assert src.count(TAIL_TAIL) == 1, src.count(TAIL_TAIL)
    src = src.replace(TAIL_TAIL, TAIL_TAIL_PROBED, 1)
    assert src.count(MUL_PROBE) == 1
    src = src.replace(BREAK_MULT, BREAK_MULT_FROZEN, 1)
    assert src.count(MUL_PROBE) == 1, src.count(MUL_PROBE)
    src = src.replace('            builder.free(mul)', '            builder.free(mul)', 1)
    src = src.replace(FREE_ANCHOR, FREE_PROBE.rstrip('\n') + '\n' + FREE_ANCHOR, 1)
    # dedent to module level, define the free helper, then reopen the class body
    src = src.replace(WRAP_ANCHOR, '\n' + PRELIM.strip('\n')
                      + '\n\n\nclass _E6Wrap(ProosAndZalkaDecoder):\n'
                      + WRAP.strip('\n') + '\n\n' + WRAP_ANCHOR, 1)
    src = src.replace(
        '    def build_syndrome_decoding_mbuc(self, builder: km.CircuitBuilder, *,',
        '    def _orig_build_syndrome_decoding_mbuc(self, builder: km.CircuitBuilder, *,',
        1)
    return src


_MODULE_CACHE: dict[bool, dict] = {}


def load_probe_module(full: bool = False) -> dict:
    if full not in _MODULE_CACHE:
        ns = {'__name__': 'e6probe_full' if full else 'e6probe'}
        exec(compile(build_probe_source(full=full), '<e6probe>', 'exec'), ns)
        _MODULE_CACHE[full] = ns
    return _MODULE_CACHE[full]


def build(n, degree, ns=None, probe=True, frozen_target=None, full=False):
    """Build the forward-only probe circuit (initial_shift + forward EEA).

    `frozen_target`: optional `(forward, adjoint)` pair of length-`n_iters` bool
    sequences.  When given, the sweep control `mul` is replaced by that
    compile-time constant instead of the quantum qubit -- i.e. the Stage A "freeze
    the liveness bitmap" candidate applied to the iteration-level liveness, which
    is what has to be right for the sweep to be gated safely.  The two passes
    need *separate* bitmaps: the forward and adjoint sweeps use opposite position
    orders, and their measured liveness is complementary.
    """
    if ns is None:
        ns = load_probe_module(full=full)
    n_iters = 3 * (n + 1) + 9
    dec = ns['_E6Wrap'](n, km.GF2Field(degree))
    dec.pat_probe_enabled = probe
    if frozen_target is not None and not isinstance(frozen_target, tuple):
        frozen_target = (frozen_target, frozen_target)
    dec.frozen_target = frozen_target
    dec.full_circuit = full
    b = km.CircuitBuilder()
    syn = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    err = [b.create_classical_register(degree, name=f'E{j}') for j in range(n)]
    dec.build_syndrome_decoding_mbuc(b, syndromes=syn, error_mbuc=err, n_iters=n_iters)
    return b.finish_circuit(), dec, n_iters, n + 3, syn


def bits_of(data, degree):
    mat = np.asarray(data, dtype=np.int64)
    rows = []
    for reg in range(mat.shape[1]):
        for b in range(degree):
            rows.append(((mat[:, reg] >> b) & 1).astype(bool))
    return np.array(rows, dtype=bool)


def run_batch(circuit, dec, syn, data, degree, seed=7, n_iters=None, want_pat=True):
    """Simulate one batch; return a dict of per-iteration probe traces."""
    shots = data.shape[0]
    sim = km.Simulator(batch_size=shots, counting_operations=True, seed=seed)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(km.array([q for r in syn for q in r]), bits_of(data, degree))
    sim.do(circuit)
    if not want_pat:
        return {'T': circuit.operation_statistics().num_toffoli_like}
    p = dec._pat
    n_iters = n_iters or len(p['z1'])
    out = {k: np.zeros((n_iters, shots), dtype=np.uint8)
           for k in ('z1', 'c2g', 'rb', 'rb0', 'flg', 'hcz', 'muls', 'mulsA', 'fz')
           if k in p}
    out['ctrl'] = np.zeros((n_iters, 4, shots), dtype=np.uint8)
    out['ctrl2'] = np.zeros((n_iters, 4, shots), dtype=np.uint8)
    out['ctrlA'] = np.zeros((n_iters, 4, shots), dtype=np.uint8)
    for it in range(n_iters):
        for k in out:
            if k in ('ctrl', 'ctrl2', 'ctrlA'):
                for j in range(4):
                    out[k][it, j] = np.asarray(
                        sim.read_across_shots(p[k][it])[j]).astype(np.uint8)
            else:
                out[k][it] = np.asarray(sim.read_across_shots(p[k][it])[0]).astype(np.uint8)
    out['T'] = circuit.operation_statistics().num_toffoli_like
    c = out['ctrl']
    out['c1'] = c[:, 1, :]
    out['c2'] = c[:, 2, :]
    out['c0'] = c[:, 0, :]
    out['c3'] = c[:, 3, :]
    return out


def gen_data(kind, n, degree, shots, rng):
    full = (1 << degree) - 1
    if kind == 'generic':
        d = rng.integers(1, full + 1, size=(n, shots)).T.astype(np.int64)
        d[:, n - 1] = rng.integers(1, full + 1, size=shots)
        return d
    if kind == 'natural':
        return rng.integers(0, full + 1, size=(shots, n)).astype(np.int64)
    raise ValueError(kind)


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial (Clopper-Pearson) CI."""
    def betacf(a, b, x, itmax=400, eps=3e-16):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < 1e-300:
            d = 1e-300
        d = 1.0 / d
        h = d
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-300:
                d = 1e-300
            c = 1.0 + aa / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-300:
                d = 1e-300
            c = 1.0 + aa / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < eps:
                break
        return h

    from math import exp, lgamma, log

    def betainc(a, b, x):
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        lbeta = lgamma(a + b) - lgamma(a) - lgamma(b)
        front = exp(lbeta + a * log(x) + b * log(1.0 - x))
        if x < (a + 1.0) / (a + b + 2.0):
            return front * betacf(a, b, x) / a
        return 1.0 - exp(lbeta + b * log(1.0 - x) + a * log(x)) * betacf(b, a, 1.0 - x) / b

    def beta_ppf(p, a, b, lo=0.0, hi=1.0, iters=200):
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if betainc(a, b, mid) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

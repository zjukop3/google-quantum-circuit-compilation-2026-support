"""E8-B — source-level instrument for the three `cswap` families.

Three call sites are instrumented:

  A  `_swap_a_and_b`          `builder.cswap(swap_ctrl, lhs, rhs)`
                              one call per (pass, iteration); 892 bit-pairs.
                              site grid: (pass, it)                       -> 892 T/site
  B  `_shift_b_array_right`   `builder.cswap(do_shift, reg_b[i-1], reg_b[i])`
                              one call per (pass, iteration, i=1..m-1); 12 pairs.
                              site grid: (pass, it, i)                   ->  12 T/site
  C  `sweep_positions`        `builder.cswap(swaps[0]|both, q_i, reg_a[i])`
                              one call per (pass, iteration, i=0..m-1); 12 pairs.
                              site grid: (pass, it, i)                   ->  12 T/site

A site is *dead on a shot* when the control qubit of that `cswap` is |0> on that
shot, because `cswap(0, a, b)` is the identity.  Two things are therefore built
from one patched source:

  * `freeze=`  : delete the `cswap` when the site's mask bit is set.
  * `probe=`   : CNOT the live control qubit into a fresh ancilla indexed by the
                 site, so the *actual* control value can be read back from the
                 simulator for both EEA passes -- no closed form required.

The `it` / `is_adjoint` pair needed by site A is tracked on the decoder instance
inside `_synchronized_eea` (A is a `@staticmethod` in the shipped source; it is
turned into a normal method here).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py'

# --------------------------------------------------------------------------- #
# anchors / substitutions
# --------------------------------------------------------------------------- #
A_DECOR = '    @staticmethod\n    def _swap_a_and_b(builder: km.CircuitBuilder,'
A_DECOR_NEW = '    def _swap_a_and_b(self, builder: km.CircuitBuilder,'

A_CALL = '            builder.cswap(swap_ctrl, lhs, rhs)'
A_PROBE = ("            _p8 = _e8_probe_reg(self, 'swap', None, None)\n"
           "            if _p8 is not None:\n"
           "                builder.cx(swap_ctrl, _p8)\n")
A_FREEZE = ("            if _e8_frozensite(self, 'swap') is not True:\n"
            "                builder.cswap(swap_ctrl, lhs, rhs)\n")

B_CALL = '                    builder.cswap(do_shift, reg_b[i - 1], reg_b[i])'
B_PROBE = ("                    _p8 = _e8_probe_reg(self, 'shift', it, i - 1)\n"
           "                    if _p8 is not None:\n"
           "                        builder.cx(do_shift, _p8)\n")
B_FREEZE = ("                    if _e8_frozensite(self, 'shift', it, i - 1) is not True:\n"
            "                        builder.cswap(do_shift, reg_b[i - 1], reg_b[i])\n")

C_BLOCK = '''                        if len(swaps) == 1:
                            builder.cswap(swaps[0], q_i, reg_a[i])
                        elif swaps:
                            both = builder.alloc_qubits(1)[0]
                            builder.cx(km.array(swaps), both)
                            builder.cswap(both, q_i, reg_a[i])
                            builder.cx(km.array(swaps), both)
                            builder.free(both)
'''
C_PROBE = '''                        _p8 = _e8_probe_reg(self, 'sweep', it, i)
                        if _p8 is not None:
                            if len(swaps) == 1:
                                builder.cx(swaps[0], _p8)
                            elif swaps:
                                _b8 = builder.alloc_qubits(1)[0]
                                builder.cx(km.array(swaps), _b8)
                                builder.cx(_b8, _p8)
                                builder.cx(km.array(swaps), _b8)
                                builder.free(_b8)
'''
C_FREEZE = '''                        if _e8_frozensite(self, 'sweep', it, i) is not True:
                            if len(swaps) == 1:
                                builder.cswap(swaps[0], q_i, reg_a[i])
                            elif swaps:
                                both = builder.alloc_qubits(1)[0]
                                builder.cx(km.array(swaps), both)
                                builder.cswap(both, q_i, reg_a[i])
                                builder.cx(km.array(swaps), both)
                                builder.free(both)
'''

MUL_ANCHOR_F = ('                        if not is_adjoint and multiplies:\n'
                '                            builder.ixor_gf2_mul(')
MUL_ANCHOR_A = ('                        if is_adjoint and multiplies:\n'
                '                            builder.ixor_gf2_mul(')
MUL_FREEZE_F = ('                        if (_e8_frozensite(self, \'mul\', it, i) '
                'is not True\n'
                '                                and not is_adjoint and multiplies):\n'
                '                            builder.ixor_gf2_mul(')
MUL_FREEZE_A = ('                        if (_e8_frozensite(self, \'mul\', it, i) '
                'is not True\n'
                '                                and is_adjoint and multiplies):\n'
                '                            builder.ixor_gf2_mul(')
MUL_PROBE = ("                        _p8m = _e8_probe_reg(self, 'mul', it, i)\n"
             "                        if _p8m is not None and multiplies:\n"
             "                            builder.cx(mul, _p8m)\n")

IT_ANCHOR = ('        for it in (range(n_iters - 1, -1, -1) if is_adjoint '
             'else range(n_iters)):\n            with eea_step:\n')
IT_ANCHOR_NEW = (IT_ANCHOR
                 + '                self._e8_it = it\n'
                 + '                self._e8_adj = is_adjoint\n')

# module level helpers (inserted just before the decoder class definition)
HELPERS = '''

def _e8_frozensite(self, kind, it=None, i=None):
    """True  => that `cswap` call is deleted (the site is frozen)."""
    fz = getattr(self, '_e8_freeze', None)
    if fz is None:
        return False
    arr = fz.get(kind)
    if arr is None:
        return False
    p = 1 if getattr(self, '_e8_adj', False) else 0
    if kind == 'swap':
        return bool(arr[p, getattr(self, '_e8_it', 0)])
    if it is None or i is None or i < 0 or i >= arr.shape[2]:
        return False
    return bool(arr[p, it, i])


def _e8_probe_reg(self, kind, it, i):
    """The ancilla this site's control qubit is XORed into (or None)."""
    pr = getattr(self, '_e8_probe', None)
    if pr is None:
        return None
    p = 1 if getattr(self, '_e8_adj', False) else 0
    if kind == 'swap':
        return pr['swap'][p][getattr(self, '_e8_it', 0)]
    if kind == 'shift':
        return pr['shift'][p][it][i]
    if kind == 'sweep':
        return pr['sweep'][p][it][i]
    return pr['mul'][p][it][i]

'''

CENSUS_HELPERS = """

_E8_CENSUS = {'swap': [], 'shift': [], 'sweep': [], 'mul': []}


def _e8_census(kind, it, i, nswaps=None):  # noqa: D401
    _E8_CENSUS[kind].append((it, i, nswaps))

"""
# census recorders, inserted in the *unfrozen* variant only
A_CENSUS = "            _e8_census('swap', self._e8_it, -1)\n"
B_CENSUS = "                    _e8_census('shift', it, i - 1)\n"
C_CENSUS = "                        _e8_census('sweep', it, i, len(swaps))\n"
MUL_CENSUS = ("                        if multiplies:\n"
              "                            _e8_census('mul', it, i)\n")

CLASS_ANCHOR = 'class ProosAndZalkaDecoder:'
WRAP_ANCHOR = '    def _initial_shift_loop('

PROBE_WRAP = '''    def _e8_alloc_probe(self, builder, n_iters, m_):
        def q(name, *idx):
            return builder.create_quantum_register(
                1, name='P8' + name + '_'.join(str(v) for v in idx))[0]
        self._e8_probe = {
            'swap': [[q('A', p, it) for it in range(n_iters)] for p in range(2)],
            'shift': [[[q('B', p, it, i) for i in range(m_ - 1)]
                       for it in range(n_iters)] for p in range(2)],
            'sweep': [[[q('C', p, it, i) for i in range(m_)]
                       for it in range(n_iters)] for p in range(2)],
            'mul': [[[q('M', p, it, i) for i in range(m_)]
                     for it in range(n_iters)] for p in range(2)],
        }

    def build_syndrome_decoding_mbuc(self, builder, *, syndromes, error_mbuc, n_iters):
        self._e8_probe = None
        if getattr(self, 'e8_probe_enabled', False):
            self._e8_alloc_probe(builder, n_iters, len(syndromes) + 3)
        return self._e8_orig_build(builder, syndromes=syndromes,
                                  error_mbuc=error_mbuc, n_iters=n_iters)

'''

ORIG_SIG = '    def build_syndrome_decoding_mbuc(self, builder: km.CircuitBuilder, *,'

TAIL_START = "        with builder.mark('chien_forney_mbuc'):"
TAIL_END = "            builder.free(reg)\n"


def patched_source(*, freeze=False, probe=False, forward_only=False, census=False,
                   families=('swap', 'shift', 'sweep')) -> str:
    src = REF.read_text()
    assert src.count(IT_ANCHOR) == 1, 'it anchor'
    src = src.replace(IT_ANCHOR, IT_ANCHOR_NEW, 1)
    assert src.count(A_DECOR) == 1, 'staticmethod anchor'
    src = src.replace(A_DECOR, A_DECOR_NEW, 1)

    if probe:
        assert src.count(A_CALL) == 1, 'A'
        src = src.replace(A_CALL, A_CALL + '\n' + A_PROBE, 1)
        assert src.count(B_CALL) == 1, 'B'
        src = src.replace(B_CALL, B_PROBE + B_CALL, 1)
        assert src.count(C_BLOCK) == 1, 'C'
        src = src.replace(C_BLOCK, C_PROBE + C_BLOCK, 1)
        assert src.count(MUL_ANCHOR_A) == 1, 'mulA'
        src = src.replace(MUL_ANCHOR_A, MUL_PROBE + MUL_ANCHOR_A, 1)
        assert src.count(ORIG_SIG) == 1, 'orig sig'
        src = src.replace(ORIG_SIG,
                          '    def _e8_orig_build(self, builder: km.CircuitBuilder, *,', 1)
        assert src.count(WRAP_ANCHOR) == 1, 'wrap anchor'
        src = src.replace(WRAP_ANCHOR, PROBE_WRAP + WRAP_ANCHOR, 1)

    if freeze:
        if 'swap' in families:
            assert src.count(A_CALL) == 1, 'A'
            src = src.replace(A_CALL, A_FREEZE, 1)
        if 'shift' in families:
            assert src.count(B_CALL) == 1, 'B'
            src = src.replace(B_CALL, B_FREEZE, 1)
        if 'sweep' in families:
            assert src.count(C_BLOCK) == 1, 'C'
            src = src.replace(C_BLOCK, C_FREEZE, 1)
        if 'mul' in families:
            assert src.count(MUL_ANCHOR_F) == 1, 'mulF'
            src = src.replace(MUL_ANCHOR_F, MUL_FREEZE_F, 1)
            assert src.count(MUL_ANCHOR_A) == 1, 'mulA'
            src = src.replace(MUL_ANCHOR_A, MUL_FREEZE_A, 1)

    if census:
        assert src.count(A_CALL) == 1, 'A'
        src = src.replace(A_CALL, A_CENSUS + A_CALL, 1)
        assert src.count(B_CALL) == 1, 'B'
        src = src.replace(B_CALL, B_CENSUS + B_CALL, 1)
        assert src.count(C_BLOCK) == 1, 'C'
        src = src.replace(C_BLOCK, C_CENSUS + C_BLOCK, 1)
        assert src.count(MUL_ANCHOR_A) == 1, 'mulA'
        src = src.replace(MUL_ANCHOR_A, MUL_CENSUS + MUL_ANCHOR_A, 1)

    if freeze or probe or census:
        assert src.count(CLASS_ANCHOR) == 1, 'class anchor'
        head = HELPERS.strip('\n')
        if census:
            head = CENSUS_HELPERS.strip('\n') + '\n\n\n' + head
        src = src.replace(CLASS_ANCHOR, head + '\n\n\n' + CLASS_ANCHOR, 1)

    if forward_only:
        start = src.index(TAIL_START)
        end = src.index(TAIL_END, start) + len(TAIL_END)
        src = (src[:start]
               + '        # [e8 probe] Chien / reverse EEA / unshift omitted\n'
               + src[end:])
    return src


def load_module(tag: str, **kw) -> dict:
    ns: dict = {'__name__': f'e8src_{tag}'}
    exec(compile(patched_source(**kw), f'<e8src_{tag}>', 'exec'), ns)
    return ns


# --------------------------------------------------------------------------- #
# grids
# --------------------------------------------------------------------------- #

def grid_shapes(n: int):
    m = n + 3
    n_iters = 3 * (n + 1) + 9
    return {'n_iters': n_iters, 'm': m,
            'swap': (2, n_iters),
            'shift': (2, n_iters, m - 1),
            'sweep': (2, n_iters, m),
            'mul': (2, n_iters, m)}


def empty_freeze(n: int) -> dict:
    sh = grid_shapes(n)
    return {k: np.zeros(sh[k], dtype=bool)
            for k in ('swap', 'shift', 'sweep', 'mul')}


MUL_T_PER_SITE = 75


def load_mul_mask(path, model: str, n: int):
    """E7-style `freeze_sites.json` -> (2, n_iters, m) bool mask."""
    import json
    js = json.loads(Path(path).read_text())
    spec = js[model] if model in js else js
    FS = np.zeros(grid_shapes(n)['mul'], dtype=bool)
    for it, i in spec['sites']:
        FS[0, int(it), int(i)] = True
        FS[1, int(it), int(i)] = True
    return FS


def full_freeze(U: dict, n: int) -> dict:
    """Broadcast a per-pass freeze union to the (2, ...) mask the builder wants.

    Both passes share one `(it, ...)` grid, so the same mask is applied to both
    (validated by `e8_mirror.py`).
    """
    return {f: np.broadcast_to(U[f], grid_shapes(n)[f]).copy() for f in U}


def site_capacity(n: int, degree: int = 12) -> dict:
    """Sites and T carried by each family's grid (design values)."""
    sh = grid_shapes(n)
    m, ni = sh['m'], sh['n_iters']
    swap_t = 2 * (m.bit_length() + 1) + m * degree
    return {
        'n_iters': ni, 'm': m,
        'swap_sites': 2 * ni, 'swap_T_per_site': swap_t, 'swap_T': 2 * ni * swap_t,
        'shift_sites': 2 * ni * (m - 1), 'shift_T_per_site': degree,
        'shift_T': 2 * ni * (m - 1) * degree,
        'sweep_sites': 2 * ni * m, 'sweep_T_per_site': degree,
    }


def build_circuit(ns: dict, m_chien: int, n: int, degree: int, *,
                  freeze=None, probe=False):
    n_iters = 3 * (n + 1) + 9
    dec = ns['ProosAndZalkaDecoder'](n, km.GF2Field(degree))
    b = km.CircuitBuilder()
    if freeze is not None:
        dec._e8_freeze = freeze
    if probe:
        dec.e8_probe_enabled = True
    syn = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    err = [b.create_classical_register(degree, name=f'E{j}') for j in range(m_chien)]
    dec.build_syndrome_decoding_mbuc(b, syndromes=syn, error_mbuc=err, n_iters=n_iters)
    return b.finish_circuit(), dec, syn


def probe_sites(dec, n: int):
    """{family: nested list of probe qubits indexed by (pass, ...)}."""
    return dec._e8_probe


def census(n: int = 70, degree: int = 12, m_chien: int = 70, **kw) -> dict:
    """Which `cswap` calls the unfrozen circuit actually emits, per family."""
    ns = load_module('census', census=True, **kw)
    circ, dec, syn = build_circuit(ns, m_chien, n, degree)
    return {'census': ns['_E8_CENSUS'],
            'T': int(circ.operation_statistics().num_toffoli_like),
            'Q': circ.num_qubits}


__all__ = ['patched_source', 'load_module', 'grid_shapes', 'empty_freeze',
           'load_mul_mask', 'MUL_T_PER_SITE',
           'full_freeze', 'site_capacity', 'build_circuit', 'probe_sites',
           'REF', 'ROOT']

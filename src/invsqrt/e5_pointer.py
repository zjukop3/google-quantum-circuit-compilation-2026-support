"""E5: the pointer-driven EEA prototype — `_swap_a_and_b` replaced by a role pointer.

Design (semantics-preserving, one site pointerized)
--------------------------------------------------
`_swap_a_and_b` physically exchanges `reg_a` (892 qubits) with `reg_b` once per
EEA round, conditional on `swap_ctrl = c3_hc AND flag`.  In this prototype the
*arrays* are never exchanged: a single coherent role bit `p` records which
physical array currently plays role A.

    role_A array = reg_a          if p == 0 else reg_b
    role_B array = reg_b          if p == 0 else reg_a

* `_swap_a_and_b`  -> `cswap(swap_ctrl, n_A|n_a, n_B|n_b)` (16 qubits, unchanged)
                      plus `cx(swap_ctrl, p)`   (free)
* every role-dependent array access is wrapped: `cswap(p, reg_a[k], reg_b[k])`
  before, the access body, `cswap(p, reg_a[k], reg_b[k])` after.  With `p == 1`
  that pair moves the role-B word into `reg_b` for the duration of the body and
  the modified word back into `reg_a` afterwards, which is exactly the
  conditional access the pointer semantics require.
* the Chien/Forney stage is left untouched: the role is *normalised* with two
  whole-array cswaps around it (2 x 876 T for the entire circuit, not per round).

Everything else (metadata degrees, sub-step controls, quantisation of `q_i`) is
untouched, so the prototype must pass the official three-condition contract on
every shot -- it is a *cost* experiment, not a semantics experiment.

Usage
-----
    python src/e5/e5_pointer.py --m 280 --n 70                 # Q / T only
    python src/e5/e5_pointer.py --m 280 --n 70 --shots 512     # + official check
    python src/e5/e5_pointer.py --m 280 --n 10 --shots 512     # small instance
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py'


# --------------------------------------------------------------------------- #
# source substitutions
# --------------------------------------------------------------------------- #

A_ALLOC = """        reg_b = [builder.alloc_qubits(degree) for _ in range(m)]

        logm = m.bit_length() + 1"""
A_ALLOC_NEW = """        reg_b = [builder.alloc_qubits(degree) for _ in range(m)]
        role_ptr = builder.alloc_qubits(1)[0]

        logm = m.bit_length() + 1"""

A_TUPLE = ('        eea = (reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag, counter, '
           'halting_counter)')
A_TUPLE_NEW = ('        eea = (reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag, counter, '
               'halting_counter, role_ptr)')

A_FREE = """        for reg in (halting_counter, counter, flag, n_q, n_b, n_a, n_B, n_A, *reg_a[:3], *reg_b):
            builder.free(reg)"""
A_FREE_NEW = """        for reg in (halting_counter, counter, flag, n_q, n_b, n_a, n_B, n_A, *reg_a[:3], *reg_b):
            builder.free(reg)
        builder.free(role_ptr)"""

A_CHIEN = """        with builder.mark('chien_forney_mbuc'):
            self._chien_search_and_forney_mbuc(builder, reg_b, error_mbuc, n_b, n_B)
        with builder.mark('reverse_eea'):"""
A_CHIEN_NEW = """        _role_a_all = km.array([q for r in reg_a for q in r])
        _role_b_all = km.array([q for r in reg_b for q in r])
        with builder.mark('normalize_role'):
            builder.cswap(role_ptr, _role_a_all, _role_b_all)
        with builder.mark('chien_forney_mbuc'):
            self._chien_search_and_forney_mbuc(builder, reg_b, error_mbuc, n_b, n_B)
        with builder.mark('unnormalize_role'):
            builder.cswap(role_ptr, _role_a_all, _role_b_all)
        with builder.mark('reverse_eea'):"""

A_SIG = """                          n_b: km.array, n_q: km.array, flag,
                          counter: km.array, halting_counter: km.array, *,
                          n_iters: int, is_adjoint: bool) -> None:"""
A_SIG_NEW = """                          n_b: km.array, n_q: km.array, flag,
                          counter: km.array, halting_counter: km.array,
                          role_ptr, *,
                          n_iters: int, is_adjoint: bool) -> None:"""

A_FLAG_CALL = """                    lambda: self._update_flag(builder, reg_b, n_A, n_B, n_a, n_b, n_q,
                                              flag, ctrl),"""
A_FLAG_CALL_NEW = """                    lambda: self._update_flag(builder, reg_a, reg_b, n_A, n_B, n_a, n_b,
                                              n_q, flag, ctrl, role_ptr),"""

A_MUL_CALL = """                    lambda: self._multiply_and_add_or_sub(builder, reg_a, reg_b, n_A, n_a,
                                                          n_q, ctrl, it,
                                                          is_adjoint=is_adjoint),"""
A_MUL_CALL_NEW = """                    lambda: self._multiply_and_add_or_sub(builder, reg_a, reg_b, n_A, n_a,
                                                          n_q, ctrl, it, role_ptr,
                                                          is_adjoint=is_adjoint),"""

A_SHIFT_CALL = """                    lambda: self._shift_b_array_right(builder, reg_b, n_B, n_b, flag, ctrl,
                                                      it, is_adjoint=is_adjoint),"""
A_SHIFT_CALL_NEW = """                    lambda: self._shift_b_array_right(builder, reg_a, reg_b, n_B, n_b,
                                                      flag, ctrl, it, role_ptr,
                                                      is_adjoint=is_adjoint),"""

A_SWAP_CALL = """                    lambda: self._swap_a_and_b(builder, reg_a, reg_b, n_A, n_B, n_a, n_b,
                                               flag, ctrl),"""
A_SWAP_CALL_NEW = """                    lambda: self._swap_a_and_b(builder, reg_a, reg_b, n_A, n_B, n_a, n_b,
                                               flag, ctrl, role_ptr),"""

# ---- _update_flag: role-dependent reads of reg_b[m-2] and reg_b[0] --------- #
A_UF_SIG = """    def _update_flag(self, builder: km.CircuitBuilder, reg_b: Sequence[km.array],
                     n_A: km.array, n_B: km.array, n_a: km.array, n_b: km.array,
                     n_q: km.array, flag, ctrl) -> None:"""
A_UF_SIG_NEW = """    def _update_flag(self, builder: km.CircuitBuilder, reg_a: Sequence[km.array],
                     reg_b: Sequence[km.array],
                     n_A: km.array, n_B: km.array, n_a: km.array, n_b: km.array,
                     n_q: km.array, flag, ctrl, role_ptr) -> None:"""

A_UF_HI = '            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)'
A_UF_HI_NEW = """            builder.cswap(role_ptr, reg_a[m - 2], reg_b[m - 2])
            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)
            builder.cswap(role_ptr, reg_a[m - 2], reg_b[m - 2])"""

A_UF_LO = '            builder.flip_if_equal(reg_b[0], 0, target=flag, control=aux)'
A_UF_LO_NEW = """            builder.cswap(role_ptr, reg_a[0], reg_b[0])
            builder.flip_if_equal(reg_b[0], 0, target=flag, control=aux)
            builder.cswap(role_ptr, reg_a[0], reg_b[0])"""

# ---- _multiply_and_add_or_sub ---------------------------------------------- #
A_MAS_SIG = """                                 n_A: km.array, n_a: km.array, n_q: km.array,
                                 ctrl, it: int, *, is_adjoint: bool) -> None:"""
A_MAS_SIG_NEW = """                                 n_A: km.array, n_a: km.array, n_q: km.array,
                                 ctrl, it: int, role_ptr, *, is_adjoint: bool) -> None:"""

A_DIV_IN = """                ratio, scaffold = builder.init_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], field=gf)"""
A_DIV_IN_NEW = """                builder.cswap(role_ptr, reg_a[index], reg_b[index])
                ratio, scaffold = builder.init_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], field=gf)"""

A_DIV_OUT = """                builder.del_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], target=ratio, scaffold=scaffold, field=gf)
                builder.free(scaffold + ratio)"""
A_DIV_OUT_NEW = """                builder.del_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], target=ratio, scaffold=scaffold, field=gf)
                builder.cswap(role_ptr, reg_a[index], reg_b[index])
                builder.free(scaffold + ratio)"""

A_SWEEP_BODY = """                        if is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], field=gf,
                                                 control=mul)
                        if len(swaps) == 1:
                            builder.cswap(swaps[0], q_i, reg_a[i])
                        elif swaps:
                            both = builder.alloc_qubits(1)[0]
                            builder.cx(km.array(swaps), both)
                            builder.cswap(both, q_i, reg_a[i])
                            builder.cx(km.array(swaps), both)
                            builder.free(both)
                        if not is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], field=gf,
                                                 control=mul)"""
A_SWEEP_BODY_NEW = """                        builder.cswap(role_ptr, reg_a[i], reg_b[i])
                        if is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], field=gf,
                                                 control=mul)
                        if len(swaps) == 1:
                            builder.cswap(swaps[0], q_i, reg_a[i])
                        elif swaps:
                            both = builder.alloc_qubits(1)[0]
                            builder.cx(km.array(swaps), both)
                            builder.cswap(both, q_i, reg_a[i])
                            builder.cx(km.array(swaps), both)
                            builder.free(both)
                        if not is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], field=gf,
                                                 control=mul)
                        builder.cswap(role_ptr, reg_a[i], reg_b[i])"""

# ---- _shift_b_array_right -------------------------------------------------- #
A_SH_SIG = """    def _shift_b_array_right(self, builder: km.CircuitBuilder,
                             reg_b: Sequence[km.array], n_B: km.array, n_b: km.array,
                             flag, ctrl, it: int, *, is_adjoint: bool) -> None:"""
A_SH_SIG_NEW = """    def _shift_b_array_right(self, builder: km.CircuitBuilder,
                             reg_a: Sequence[km.array],
                             reg_b: Sequence[km.array], n_B: km.array, n_b: km.array,
                             flag, ctrl, it: int, role_ptr, *, is_adjoint: bool) -> None:"""

A_SH_MOVE = '                    builder.cswap(do_shift, reg_b[i - 1], reg_b[i])'
A_SH_MOVE_NEW = """                    _rc = builder.alloc_qubits(2)
                    builder.init_and(do_shift, role_ptr, _rc[0])
                    builder.cx(do_shift, _rc[1])
                    builder.cx(_rc[0], _rc[1])
                    builder.cswap(_rc[0], reg_a[i - 1], reg_a[i])
                    builder.cswap(_rc[1], reg_b[i - 1], reg_b[i])
                    builder.cx(_rc[0], _rc[1])
                    builder.cx(do_shift, _rc[1])
                    builder.del_and(do_shift, role_ptr, _rc[0])
                    builder.free(_rc)"""

# ---- _swap_a_and_b --------------------------------------------------------- #
A_SW_SIG = """    def _swap_a_and_b(builder: km.CircuitBuilder, reg_a: Sequence[km.array],
                      reg_b: Sequence[km.array], n_A: km.array, n_B: km.array,
                      n_a: km.array, n_b: km.array, flag, ctrl) -> None:"""
A_SW_SIG_NEW = """    def _swap_a_and_b(builder: km.CircuitBuilder, reg_a: Sequence[km.array],
                      reg_b: Sequence[km.array], n_A: km.array, n_B: km.array,
                      n_a: km.array, n_b: km.array, flag, ctrl, role_ptr) -> None:"""

A_SW_BODY = """            lhs = km.array([q for r in (n_A, n_a, *reg_a) for q in r])
            rhs = km.array([q for r in (n_B, n_b, *reg_b) for q in r])
            builder.cswap(swap_ctrl, lhs, rhs)"""
A_SW_BODY_NEW = """            lhs = km.array([q for r in (n_A, n_a) for q in r])
            rhs = km.array([q for r in (n_B, n_b) for q in r])
            builder.cswap(swap_ctrl, lhs, rhs)
            builder.cx(swap_ctrl, role_ptr)"""

SUBS = [
    (A_ALLOC, A_ALLOC_NEW), (A_TUPLE, A_TUPLE_NEW), (A_FREE, A_FREE_NEW),
    (A_CHIEN, A_CHIEN_NEW), (A_SIG, A_SIG_NEW),
    (A_FLAG_CALL, A_FLAG_CALL_NEW), (A_MUL_CALL, A_MUL_CALL_NEW),
    (A_SHIFT_CALL, A_SHIFT_CALL_NEW), (A_SWAP_CALL, A_SWAP_CALL_NEW),
    (A_UF_SIG, A_UF_SIG_NEW), (A_UF_HI, A_UF_HI_NEW), (A_UF_LO, A_UF_LO_NEW),
    (A_MAS_SIG, A_MAS_SIG_NEW), (A_DIV_IN, A_DIV_IN_NEW), (A_DIV_OUT, A_DIV_OUT_NEW),
    (A_SWEEP_BODY, A_SWEEP_BODY_NEW),
    (A_SH_SIG, A_SH_SIG_NEW), (A_SH_MOVE, A_SH_MOVE_NEW),
    (A_SW_SIG, A_SW_SIG_NEW), (A_SW_BODY, A_SW_BODY_NEW),
]


def build_pointer_source() -> str:
    src = REF.read_text()
    for old, new in SUBS:
        if old not in src:
            raise RuntimeError(f'anchor not found:\n{old!r}')
        src = src.replace(old, new, 1)
    return src


def load_variant(name: str):
    if name == 'ref':
        src = REF.read_text()
    elif name == 'pointer':
        src = build_pointer_source()
    else:
        raise ValueError(name)
    ns: dict = {'__name__': f'e5_{name}'}
    exec(compile(src, str(REF), 'exec'), ns)  # noqa: S102 - local trusted file
    return ns


def build(name: str, m: int, n: int, degree: int):
    ns = load_variant(name)
    return ns['build_reference_circuit'](m, n, degree), ns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', action='append', default=None,
                    choices=('ref', 'pointer'))
    ap.add_argument('--m', type=int, default=280)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=0,
                    help='0 = Q/T only; >0 = run the official contract check')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--mode', choices=('official', 'maxweight'), default='official')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    names = args.variant or ['ref', 'pointer']
    if 'ref' in names:
        names = ['ref'] + [x for x in names if x != 'ref']

    checker = None
    if args.shots:
        sys.path.insert(0, str(ROOT / 'src' / 'e4'))
        sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'dqi'))
        import semantic as SEM  # noqa: E402
        import verify_submission_dqi as V  # noqa: E402
        checker = (SEM, V)

    out: dict = {'m': args.m, 'n': args.n, 'degree': args.degree,
                 'shots': args.shots, 'mode': args.mode}
    base = None
    for name in names:
        t0 = time.time()
        try:
            circuit, _ = build(name, args.m, args.n, args.degree)
        except Exception as exc:  # noqa: BLE001
            print(f'{name:10s} BUILD ERROR {type(exc).__name__}: {exc}')
            if name == 'pointer':
                import traceback
                traceback.print_exc()
            continue
        q = circuit.num_qubits
        t = int(circuit.operation_statistics().num_toffoli_like)
        tb = time.time() - t0
        rec = {'num_qubits': q, 'static_toffoli': t, 'build_seconds': round(tb, 2)}
        if base is None:
            base = rec
            print(f'{name:10s} Q={q:5d}  T={t:12,d}  build={tb:.1f}s   (baseline)')
        else:
            dq = 100.0 * (q - base['num_qubits']) / base['num_qubits']
            dt = 100.0 * (t - base['static_toffoli']) / base['static_toffoli']
            mult = (base['num_qubits'] / q) ** 2 / (t / base['static_toffoli'])
            print(f'{name:10s} Q={q:5d} ({dq:+.3f}%)  T={t:12,d} ({dt:+.3f}%)  '
                  f'build={tb:.1f}s  score x{mult:.4f}')
            rec.update({'score_multiple': mult, 'dQ_pct': dq, 'dT_pct': dt})

        if checker is not None:
            SEM, V = checker
            spec = V.ProblemSpecDQI.benchmark_instance(m=args.m, n=args.n,
                                                       degree=args.degree,
                                                       num_shots=args.shots)
            tables = V.FieldTables(km_degree(args.degree))
            shots = SEM.make_shots(spec, tables, 20261025, args.shots, args.mode)
            t1 = time.time()
            res = SEM.check_circuit(circuit, spec, shots, batch_size=args.batch_size)
            rec.update({'semantic': res, 'sim_seconds': round(time.time() - t1, 2)})
            print(f'           semantics[{args.mode}] {args.shots} shots: '
                  f'value={res["value_errors"]} phase={res["phase_errors"]} '
                  f'-> {res["verdict"]}  ({time.time()-t1:.1f}s)')
        out[name] = rec

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, default=float))
        print(f'\nwrote {p}')


def km_degree(degree: int):
    import kickmix as km
    return km.GF2Field(degree)


if __name__ == '__main__':
    main()

"""E6 experiment J: end-to-end semantic proof of the frozen sweep schedule.

Two questions, both answered against the **shipped** reference module and the
**official** DQI verifier:

1. *What is the real per-iteration liveness of the sweep's `mul` qubit?*  A probe
   CNOT is inserted into the reference `_multiply_and_add_or_sub` immediately
   before the tail `endpoint(prev)` (the fan-in that uncomputes `mul`), one probe
   qubit per iteration and per pass, and read back from the simulator.
2. *If that bitmap is frozen into a compile-time constant, does the computed
   answer change?*  The same method is re-bound with the two
   `ixor_gf2_mul(..., control=mul)` calls skipped on iterations the bitmap marks
   dead, keeping `mul` allocated (so the register layout is bit-identical) and
   leaving the position-controlled `cswap` untouched.  Reference and frozen
   circuits are then run on the *same* officially-sampled shots.

Usage
-----
    python src/e6/e6_semantics.py --n 6 --degree 6 --m 63 --shots 4096
    python src/e6/e6_semantics.py --n 70 --degree 12 --m 280 --shots 1024
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import kickmix as km

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src' / 'e3'))

import e6_common as C  # noqa: E402

TAIL = ('                    endpoint(prev)\n'
        '                    builder.free(mul)')


def load_module(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def rewritten_method_source(ref, *, freeze=True, probe=None) -> str:
    """`_multiply_and_add_or_sub` with the sweep gated and/or probed.

    `freeze='tail'` gates the two `ixor_gf2_mul(..., control=mul)` calls on the
    iteration-level bitmap read at the tail of the sweep; `freeze='sites'` gates
    them on the per-`(it, i)` bitmap read at the multiplier itself.  Either way
    `mul` is still allocated and still fan-in'ed, so the register layout and every
    other gate stay bit-identical and only the multipliers disappear.  The
    position-controlled `cswap` is deliberately never touched.

    `probe='tail'` adds one CNOT from `mul` into `self._mp[is_adjoint][it][0]`
    immediately before the tail `endpoint(prev)`.  That reads only the *final*
    accumulation, so `probe='site'` instead adds, for every `(it, i)` where
    `multiplies` is true, one CNOT from `mul` into `self._mpi[is_adjoint][it][i]`
    at the exact point the multipliers read it -- the object a freeze would
    actually have to reproduce.
    """
    src = inspect.getsource(ref.ProosAndZalkaDecoder._multiply_and_add_or_sub)

    if freeze == 'iters':
        # iteration-level freeze: whole iterations the bitmap marks dead lose all
        # their multipliers.
        old_mul = '                    mul = builder.alloc_qubits(1)[0]'
        assert old_mul in src
        src = src.replace(
            old_mul,
            old_mul + '\n'
            + '\n                    _fi = getattr(self, "_frozen_iters", None)',
            1)
        for pass_expr, pass_id in (('is_adjoint', 1), ('not is_adjoint', 0)):
            old = ('                        if ' + pass_expr + ' and multiplies:\n'
                   '                            builder.ixor_gf2_mul(')
            assert src.count(old) == 1, old[:60]
            src = src.replace(
                old,
                old.replace(
                    'if ',
                    'if not (_fi is not None and bool(_fi[%d][it])) and ' % pass_id,
                    1), 1)
    elif freeze == 'sites':
        old_mul = '                    mul = builder.alloc_qubits(1)[0]'
        assert old_mul in src
        src = src.replace(
            old_mul,
            old_mul + '\n'
            + '\n                    _fs = getattr(self, "_frozen_sites", None)',
            1)
        for pass_expr, pass_id in (('is_adjoint', 1), ('not is_adjoint', 0)):
            old = ('                        if ' + pass_expr + ' and multiplies:\n'
                   '                            builder.ixor_gf2_mul(')
            assert src.count(old) == 1, old[:60]
            src = src.replace(
                old, old.replace('if ', 'if not (_fs is not None and bool(_fs[%d][it][i])) and ' % pass_id), 1)
    elif freeze:
        old_mul = '                    mul = builder.alloc_qubits(1)[0]'
        assert old_mul in src
        src = src.replace(
            old_mul,
            old_mul + '\n'
            + '\n                    _ft = getattr(self, "_frozen_liveness", None)'
            + '\n                    _frozen = (_ft is not None'
            + ' and bool(_ft[is_adjoint][it]))',
            1)
        for old in ('                        if is_adjoint and multiplies:\n'
                    '                            builder.ixor_gf2_mul(',
                    '                        if not is_adjoint and multiplies:\n'
                    '                            builder.ixor_gf2_mul('):
            assert old in src, old[:60]
            src = src.replace(old, old.replace('if ', 'if not _frozen and '), 1)

    if probe == 'tail':
        assert src.count(TAIL) == 1
        src = src.replace(
            TAIL,
            '                    if getattr(self, "_mp", None) is not None:\n'
            '                        builder.cx(mul, self._mp[1 if is_adjoint else 0][it][0])\n'
            + TAIL, 1)
    elif probe == 'sitexor':
        # one probe qubit per (pass, it); several multiplier sites in the same
        # iteration XOR into it, so a 1 means *at least one* site saw `mul == 1`.
        old_first = ('                        if is_adjoint and multiplies:\n'
                     '                            builder.ixor_gf2_mul(')
        assert src.count(old_first) == 1
        src = src.replace(
            old_first,
            '                        if multiplies and getattr(self, "_mpx", None) is not None:\n'
            '                            builder.cx('
            'mul, self._mpx[1 if is_adjoint else 0][it][0])\n'
            + old_first, 1)
    elif probe == 'site':
        old_first = ('                        if is_adjoint and multiplies:\n'
                     '                            builder.ixor_gf2_mul(')
        assert src.count(old_first) == 1
        src = src.replace(
            old_first,
            '                        if multiplies and getattr(self, "_mpi", None) is not None:\n'
            '                            builder.cx('
            'mul, self._mpi[1 if is_adjoint else 0][it][i])\n'
            + old_first, 1)

    # `inspect.getsource` keeps the class-body indent; drop it so the text is a
    # standalone function definition.
    return '\n'.join(l[4:] if l.startswith('    ') else l for l in src.split('\n'))


def _install_method(ref, key, source_builder):
    module_ns = vars(ref)
    fn = module_ns.get(key)
    if fn is None:
        ns = {}
        exec(compile(source_builder(), '<e6>' + key, 'exec'), module_ns, ns)
        fn = ns['_multiply_and_add_or_sub']
        module_ns[key] = fn
    return fn


def build_circuit(ref, m, n, degree, *, frozen_liveness=None, frozen_sites=None,
                  frozen_iters=None, probe_regs=None):
    decoder = ref.ProosAndZalkaDecoder(n, km.GF2Field(degree))
    b = km.CircuitBuilder()
    if frozen_liveness is not None:
        fn = _install_method(ref, '_e6_frozen_mult',
                             lambda: rewritten_method_source(ref, freeze='tail'))
        decoder._multiply_and_add_or_sub = fn.__get__(decoder, type(decoder))
        decoder._frozen_liveness = frozen_liveness
    if frozen_iters is not None:
        fn = _install_method(
            ref, '_e6_frozen_iter_mult',
            lambda: rewritten_method_source(ref, freeze='iters'))
        decoder._multiply_and_add_or_sub = fn.__get__(decoder, type(decoder))
        decoder._frozen_iters = frozen_iters
    if frozen_sites is not None:
        fn = _install_method(ref, '_e6_frozen_site_mult',
                             lambda: rewritten_method_source(ref, freeze='sites'))
        decoder._multiply_and_add_or_sub = fn.__get__(decoder, type(decoder))
        decoder._frozen_sites = frozen_sites
    syndromes = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    if probe_regs is not None:
        n_iters = 3 * (n + 1) + 9
        regs = [[b.create_quantum_register(1, name=f'PF{it}') for it in range(n_iters)],
                [b.create_quantum_register(1, name=f'PA{it}') for it in range(n_iters)]]
        fn = _install_method(
            ref, '_e6_probe_mult',
            lambda: rewritten_method_source(ref, freeze=False, probe=True))
        decoder._multiply_and_add_or_sub = fn.__get__(decoder, type(decoder))
        decoder._mp = regs
        probe_regs.extend([regs, syndromes])
    error_mbuc = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    decoder.build_syndrome_decoding_mbuc(
        b, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=3 * (n + 1) + 9)
    return b.finish_circuit()


def measure_liveness(ref, m, n, degree, *, shots=256, batch_size=128, seed=999):
    """Majority (forward, adjoint) `mul` liveness bitmaps, from the shipped module."""
    n_iters = 3 * (n + 1) + 9
    holder: list = []
    circ = build_circuit(ref, m, n, degree, probe_regs=holder)
    regs, syn = holder[0], holder[1]
    rng = np.random.default_rng(seed)
    data = C.gen_data('generic', n, degree, shots, rng)
    bits = C.bits_of(data, degree)
    F = np.zeros((n_iters, shots), dtype=bool)
    A = np.zeros((n_iters, shots), dtype=bool)
    for start in range(0, shots, batch_size):
        stop = min(start + batch_size, shots)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True,
                           seed=7 + start)
        sim.use_same_registers_as(circ)
        sim.write_across_shots(km.array([q for r in syn for q in r]),
                               bits[:, start:stop])
        sim.do(circ)
        for it in range(n_iters):
            F[it, start:stop] = np.asarray(
                sim.read_across_shots(regs[0][it][0])).astype(bool)
            A[it, start:stop] = np.asarray(
                sim.read_across_shots(regs[1][it][0])).astype(bool)
    fkey, fn_ = Counter(tuple(F[:, s].tolist()) for s in range(shots)).most_common(1)[0]
    akey, an_ = Counter(tuple(A[:, s].tolist()) for s in range(shots)).most_common(1)[0]
    Fm = np.array(fkey, dtype=bool)
    Am = np.array(akey, dtype=bool)
    return Fm, Am, dict(forward_votes=fn_, adjoint_votes=an_,
                        forward_distinct=len(set(tuple(F[:, s].tolist())
                                                for s in range(shots))),
                        adjoint_distinct=len(set(tuple(A[:, s].tolist())
                                                for s in range(shots))),
                        shots=shots)


def measure_sitexor_liveness(ref, m, n, degree, *, shots=256, batch_size=128,
                             seed=999, bits=None):
    """Per-iteration “does any multiplier site see `mul == 1`”, both passes.

    One probe qubit per `(pass, it)`; the several multiplier sites of an iteration
    XOR into it, so 1 means at least one site is live.
    """
    n_iters = 3 * (n + 1) + 9
    decoder = ref.ProosAndZalkaDecoder(n, km.GF2Field(degree))
    b = km.CircuitBuilder()
    regs = [[b.create_quantum_register(1, name=f'PX{p}_{it}')
             for it in range(n_iters)] for p in range(2)]
    fn = _install_method(
        ref, '_e6_probe_sitexor',
        lambda: rewritten_method_source(ref, freeze=False, probe='sitexor'))
    decoder._multiply_and_add_or_sub = fn.__get__(decoder, type(decoder))
    decoder._mpx = regs
    syndromes = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    error_mbuc = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    decoder.build_syndrome_decoding_mbuc(
        b, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters)
    circ = b.finish_circuit()

    if bits is None:
        rng = np.random.default_rng(seed)
        data = C.gen_data('generic', n, degree, shots, rng)
        bits = C.bits_of(data, degree)
    shots = bits.shape[1]
    out = [np.zeros((n_iters, shots), dtype=bool) for _ in range(2)]
    for start in range(0, shots, batch_size):
        stop = min(start + batch_size, shots)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True,
                           seed=7 + start)
        sim.use_same_registers_as(circ)
        sim.write_across_shots(km.array([q for r in syndromes for q in r]),
                               bits[:, start:stop])
        sim.do(circ)
        for p in range(2):
            for it in range(n_iters):
                out[p][it, start:stop] = np.asarray(
                    sim.read_across_shots(regs[p][it][0])).astype(bool)
    return out[0], out[1], circ


def measure_site_liveness(ref, m, n, degree, *, shots=64, batch_size=64, seed=999):
    """Per-`(pass, it, i)` value of `mul` at the exact point the multipliers read it.

    Returns `(F, A)` with shape `(n_iters, m, shots)` -- `F` for the forward EEA
    pass, `A` for the adjoint one.  `True` means the multiplier at that site is
    *not* a no-op, i.e. the site cannot be dropped by a compile-time schedule.
    """
    n_iters = 3 * (n + 1) + 9
    decoder = ref.ProosAndZalkaDecoder(n, km.GF2Field(degree))
    b = km.CircuitBuilder()
    regs = [[[b.create_quantum_register(1, name=f'Q{p}_{it}_{i}')
              for i in range(m)] for it in range(n_iters)] for p in range(2)]
    fn = _install_method(
        ref, '_e6_probe_site',
        lambda: rewritten_method_source(ref, freeze=False, probe='site'))
    decoder._multiply_and_add_or_sub = fn.__get__(decoder, type(decoder))
    decoder._mpi = regs
    syndromes = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    error_mbuc = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    decoder.build_syndrome_decoding_mbuc(
        b, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters)
    circ = b.finish_circuit()

    rng = np.random.default_rng(seed)
    data = C.gen_data('generic', n, degree, shots, rng)
    bits = C.bits_of(data, degree)
    out = [np.zeros((n_iters, m, shots), dtype=bool) for _ in range(2)]
    for start in range(0, shots, batch_size):
        stop = min(start + batch_size, shots)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True,
                           seed=7 + start)
        sim.use_same_registers_as(circ)
        sim.write_across_shots(km.array([q for r in syndromes for q in r]),
                               bits[:, start:stop])
        sim.do(circ)
        for p in range(2):
            for it in range(n_iters):
                for i in range(m):
                    v = np.asarray(sim.read_across_shots(regs[p][it][i])).astype(bool)
                    if v.any():
                        out[p][it, i, start:stop] = v
    return out[0], out[1], circ


def run_verify(V, circ, spec, shots, batch_size=128, max_workers=2):
    """Official verifier, tolerant of the `eps_plus < 1/16` assertion at small n.

    `verify` raises when the Clopper-Pearson upper bound of the *observed* error
    rate exceeds 1/16, which at small `n` can happen even with zero observed
    errors.  For an equivalence question only the counts matter, so on that
    assertion the counts embedded in its message are returned.
    """
    try:
        r = V.verify(circ, spec, seed=12345, batch_size=batch_size,
                     max_workers=max_workers, shots=shots)
        return r['value_errors'], r['phase_errors'], r['toffoli_count'], True
    except AssertionError as exc:
        tail = str(exc).split('(')[-1]
        ve = pe = None
        for part in tail.rstrip(')').split(','):
            part = part.strip()
            if part.startswith('value_errors='):
                ve = int(part.split('=')[1].split('/')[0])
            elif part.startswith('phase_errors='):
                pe = int(part.split('=')[1].split('/')[0])
        if ve is None or pe is None:
            raise
        return ve, pe, circ.operation_statistics().num_toffoli_like, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=6)
    ap.add_argument('--m', type=int, default=63)
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--max-workers', type=int, default=2)
    ap.add_argument('--liveness-shots', type=int, default=256)
    ap.add_argument('--json', default=None)
    ap.add_argument('--site-check', action='store_true',
                    help='also measure per-site liveness and test a site freeze')
    ap.add_argument('--site-shots', type=int, default=64)
    ap.add_argument('--freeze-mode', choices=('tail', 'sites', 'both'), default='sites')
    args = ap.parse_args()
    n, degree, m = args.n, args.degree, args.m
    n_iters = 3 * (n + 1) + 9

    V = load_module('vdqi', ROOT / 'data' / 'verifiers' / 'dqi'
                    / 'verify_submission_dqi.py')
    ref = load_module('refdqi', ROOT / 'data' / 'verifiers' / 'dqi'
                      / 'dqi_reference_circuit.py')

    t0 = time.time()
    F, A, meta = measure_liveness(ref, m, n, degree, shots=args.liveness_shots)
    print(f'[liveness] forward={int(F.sum())}/{n_iters} '
          f'({meta["forward_votes"]}/{meta["shots"]} votes, '
          f'{meta["forward_distinct"]} distinct), adjoint={int(A.sum())} '
          f'({meta["adjoint_votes"]} votes, {meta["adjoint_distinct"]} distinct) '
          f'({time.time() - t0:.1f}s)', flush=True)

    spec = V.ProblemSpecDQI.benchmark_instance(m=m, n=n, degree=degree,
                                               num_shots=args.shots)
    tables = V.FieldTables(km.GF2Field(degree))
    rng = np.random.default_rng(20261025)
    wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    shots = V.sample_shots(spec, tables, rng, args.shots, wp)

    out = {}
    modes = [('ref', None), ('frozen', (F, A))] if args.freeze_mode != 'sites' \
        else [('ref', None)]
    for tag, fake in modes:
        t0 = time.time()
        circ = build_circuit(ref, m, n, degree, frozen_liveness=fake)
        T = circ.operation_statistics().num_toffoli_like
        print(f'[{tag:6s}] Q={circ.num_qubits} T={T:,} ({time.time() - t0:.1f}s)',
              flush=True)
        ve, pe, T, passed = run_verify(V, circ, spec, shots,
                                       batch_size=args.batch_size,
                                       max_workers=args.max_workers)
        out[tag] = dict(value_errors=ve, phase_errors=pe, toffoli_count=T,
                        eps_ok=passed)
        print(f'[{tag:6s}] value_errors={ve} phase_errors={pe} toffolis={T:,} '
              f'eps_plus<1/16={passed}', flush=True)

    site = None
    if args.site_check or args.freeze_mode in ('sites', 'both'):
        t0 = time.time()
        SF, SA, scirc = measure_site_liveness(ref, m, n, degree,
                                              shots=args.site_shots)
        print(f'[site] Q={scirc.num_qubits} '
              f'({time.time() - t0:.1f}s)', flush=True)
        # `SF[:, :, s]` is per-shot; sites that are never live on any generic shot
        # are the only ones a constant schedule could drop.
        anyF = SF.any(axis=2)
        allF = SF.all(axis=2)
        anyA = SA.any(axis=2)
        allA = SA.all(axis=2)
        print(f'[site] forward  sites ever-live {int(anyF.sum())} / '
              f'always-live {int(allF.sum())} / ever-dead '
              f'{int((~anyF).sum())}', flush=True)
        print(f'[site] adjoint  sites ever-live {int(anyA.sum())} / '
              f'always-live {int(allA.sum())} / ever-dead '
              f'{int((~anyA).sum())}', flush=True)
        site = dict(forward_ever_live=int(anyF.sum()),
                    forward_always_live=int(allF.sum()),
                    adjoint_ever_live=int(anyA.sum()),
                    adjoint_always_live=int(allA.sum()),
                    total_sites=int(anyF.size))
        dead_sites = (~anyF).tolist(), (~anyA).tolist()
        if args.freeze_mode in ('sites', 'both'):
            t0 = time.time()
            scirc = build_circuit(ref, m, n, degree, frozen_sites=dead_sites)
            Ts = scirc.operation_statistics().num_toffoli_like
            print(f'[sites ] Q={scirc.num_qubits} T={Ts:,} '
                  f'({time.time() - t0:.1f}s)', flush=True)
            ve, pe, Ts, passed = run_verify(V, scirc, spec, shots,
                                            batch_size=args.batch_size,
                                            max_workers=args.max_workers)
            out['sites'] = dict(value_errors=ve, phase_errors=pe,
                                toffoli_count=Ts, eps_ok=passed)
            print(f'[sites ] value_errors={ve} phase_errors={pe} '
                  f'toffolis={Ts:,} eps_plus<1/16={passed}', flush=True)

    def clean(tag):
        return tag in out and out[tag]['value_errors'] == 0 and out[tag]['phase_errors'] == 0
    ok = clean('ref')
    for tag in ('frozen', 'sites'):
        if tag in out:
            ok = ok and out[tag]['value_errors'] == out['ref']['value_errors'] \
                and out[tag]['phase_errors'] == out['ref']['phase_errors']
    best = min((out[t]['toffoli_count'] for t in out), default=0)
    ratio = out['ref']['toffoli_count'] / best if best else 1.0
    live = int(F.sum()) + int(A.sum())
    print('\n== J. semantics of the frozen sweep ======================')
    print(f'  reference errors : {out["ref"]["value_errors"]} val / '
          f'{out["ref"]["phase_errors"]} phase')
    print(f'  frozen    errors : {out["frozen"]["value_errors"]} val / '
          f'{out["frozen"]["phase_errors"]} phase')
    print(f'  semantic equivalence on {args.shots} official shots : {ok}')
    print(f'  live iterations {live}/{2 * n_iters}; Toffoli ratio ref/frozen '
          f': {ratio:.6f}')
    if args.json:
        Path(args.json).write_text(json.dumps({
            'n': n, 'degree': degree, 'm': m, 'shots': args.shots,
            'forward_live': int(F.sum()), 'adjoint_live': int(A.sum()),
            'liveness_meta': meta,
            **{f'{k}_{kk}': vv for k, v in out.items() for kk, vv in v.items()},
            'semantically_equal': bool(ok), 'toffoli_ratio': ratio,
            'site': site,
        }, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

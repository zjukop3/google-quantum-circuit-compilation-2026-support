"""E6-D: are the fan-out probes harmless?  (validation of the main instrument)

Every measurement in this note relies on CNOT fan-out probes inserted into the
shipped module.  Edits in that region turned out to be dangerous in this
toolchain (even a logical no-op causes 70% damage), so the probes must be
validated, not assumed:

(A) full decoder + fire probe inside `_update_flag`:
    do the syndrome registers still come back exactly?  (0 errors expected)

(B) local forward-only probe (the tail is dropped) vs the shipped decoder
    instrumented *only* with a right-after-`forward_eea` read of the syndromes:
    are the two forward-EEA outputs bit-identical on the same shots?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'data' / 'verifiers' / 'dqi'))
sys.path.insert(0, str(HERE))

import kickmix as km                                              # noqa: E402
import verify_submission_dqi as V                                 # noqa: E402
import e6d_schedule as E                                          # noqa: E402
import e6d_full_fire as F                                         # noqa: E402

SRC = (ROOT / 'data' / 'verifiers' / 'dqi' / 'dqi_reference_circuit.py').read_text()
MARK = ("        with builder.mark('chien_forney_mbuc'):")


def build_eea_readout(n, degree, m):
    """Shipped decoder + a copy of the syndromes taken right after forward EEA."""
    probe = ("        self._snp = [builder.create_quantum_register(degree, "
             "name=f'SN{i}') for i in range(n)]\n"
             "        builder.cx(km.array([q for r in syndromes for q in r]),"
             " km.array([q for r in self._snp for q in r]))\n")
    src = SRC.replace(MARK, probe + MARK, 1)
    ns = {'__name__': 'e6dsafe'}
    exec(compile(src, '<e6dsafe>', 'exec'), ns)
    b = km.CircuitBuilder()
    syn = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    err = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    dec = ns['ProosAndZalkaDecoder'](n, km.GF2Field(degree))
    dec.build_syndrome_decoding_mbuc(b, syndromes=syn, error_mbuc=err,
                                     n_iters=3 * (n + 1) + 9)
    return b.finish_circuit(), dec, syn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=63)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--shots', type=int, default=1024)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    n, degree, m = args.n, args.degree, args.m
    n_iters = 3 * (n + 1) + 9
    rng = np.random.default_rng(11)
    full = (1 << degree) - 1
    data = rng.integers(0, full + 1, size=(args.shots, n)).astype(np.int64)

    out = {'n': n, 'm': m, 'degree': degree, 'shots': args.shots}

    # ---- (A) fire probe is harmless --------------------------------------
    circ, dec, syn = F.build(n, degree, m, 2 * n_iters)
    sim = km.Simulator(batch_size=args.shots, counting_operations=True, seed=5)
    sim.use_same_registers_as(circ)
    rows = []
    for reg in range(n):
        for b in range(degree):
            rows.append(((data[:, reg] >> b) & 1).astype(bool))
    ib = np.array(rows, dtype=bool)
    sim.write_across_shots(km.array([q for r in syn for q in r]), ib)
    sim.do(circ)
    syn_arr = km.array([q for r in syn for q in r])
    rd = np.asarray(sim.read_across_shots(syn_arr)).astype(bool) ^ ib
    out['A_fire_probe_syndrome_errors'] = int(rd.any(axis=0).sum())
    print(f'(A) full decoder + fire probe: syndrome errors = '
          f'{out["A_fire_probe_syndrome_errors"]}/{args.shots}', flush=True)

    # ---- (B) local probe == shipped forward EEA --------------------------
    circ2, dec2, syn2 = build_eea_readout(n, degree, m)
    sim2 = km.Simulator(batch_size=args.shots, counting_operations=True, seed=5)
    sim2.use_same_registers_as(circ2)
    sim2.write_across_shots(km.array([q for r in syn2 for q in r]), ib)
    sim2.do(circ2)
    ref_eea = np.asarray(sim2.read_across_shots(
        km.array([q for r in dec2._snp for q in r]))).astype(bool)

    ns = E.load_probe_module()
    circ3, dec3, ni3, me3, syn3 = E.build(n, degree, n, ns)
    sim3 = km.Simulator(batch_size=args.shots, counting_operations=True, seed=5)
    sim3.use_same_registers_as(circ3)
    sim3.write_across_shots(km.array([q for r in syn3 for q in r]), ib)
    sim3.do(circ3)
    # local probe's syndrome-qubit values at the end of forward EEA: the probe
    # circuit shares the S0..S{n-1} register objects, so read them directly.
    loc_eea = np.asarray(sim3.read_across_shots(
        km.array([q for r in syn3 for q in r]))).astype(bool)
    mismatch = int((loc_eea ^ ref_eea).any(axis=0).sum())
    out['B_local_vs_shipped_forward_eea_mismatch_shots'] = mismatch
    out['B_total_bits_compared'] = int(loc_eea.size)
    print(f'(B) local forward-only probe vs shipped forward EEA: '
          f'{mismatch}/{args.shots} shots differ', flush=True)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print('wrote', args.json)


if __name__ == '__main__':
    main()

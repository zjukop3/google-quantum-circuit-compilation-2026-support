"""E7-E — site-level check of the freeze set, in *both* EEA passes.

Two questions the freeze decision hinges on:

  H1  do the forward and adjoint passes have the same per-`(it, i)` multiplier
      liveness?  (`freeze = complement of union` is only valid if they do;
      if the adjoint bitmap were the iteration-mirror of the forward one, the
      mirror-closed freeze would be **0** for every model in `e7_enum.py`.)
  H2  is every site in the chosen freeze set really dead on real shots?

Method: patch `_multiply_and_add_or_sub` in the shipped reference module so that,
immediately before each multiplier reads the `mul` qubit, a CNOT copies `mul`
into a per-site probe qubit -- but only for a *chosen* set of `(pass, it, i)`
triples.  Then run the **full** decoder (forward EEA + Chien + reverse EEA) on
official-distribution shots and read the probes back.

Usage
-----
    $P src/e7/e7_adjoint.py --freeze notes/runs/e7/freeze_sites.json \
        --model nA070_69_K2 --shots 256 --controls 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'e6'))
sys.path.insert(0, str(HERE))

import e6_semantics as ES        # noqa: E402
import e6_common as C            # noqa: E402

import kickmix as km             # noqa: E402

PROBE = '''                    if multiplies and getattr(self, "_e7_map", None) is not None:
                        _pq = self._e7_map.get((1 if is_adjoint else 0, it, i))
                        if _pq is not None:
                            builder.cx(mul, _pq)
'''


def patched_source(ref):
    src = ES.rewritten_method_source(ref, freeze=False, probe=None)
    anchor = ('                    if is_adjoint and multiplies:\n'
              '                        builder.ixor_gf2_mul(')
    assert src.count(anchor) == 1
    return src.replace(anchor, PROBE + anchor, 1)


def build_probed(ref, m, n, degree, sites):
    """Full decoder with per-site `mul` probes at the given (pass, it, i) triples."""
    decoder = ref.ProosAndZalkaDecoder(n, km.GF2Field(degree))
    b = km.CircuitBuilder()
    ns = {}
    exec(compile(patched_source(ref), '<e7probe>', 'exec'), vars(ref), ns)
    fn = ns['_multiply_and_add_or_sub']
    # bind the rewritten function to the decoder class so `getattr` works
    type(decoder)._multiply_and_add_or_sub = fn
    decoder._e7_map = {key: b.create_quantum_register(1, name=f'P{key[0]}_{key[1]}_{key[2]}')[0]
                       for key in sites}
    syndromes = [b.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    error_mbuc = [b.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    decoder.build_syndrome_decoding_mbuc(
        b, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=3 * (n + 1) + 9)
    return b.finish_circuit(), decoder, syndromes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', default=str(ROOT / 'notes/runs/e7/freeze_sites.json'))
    ap.add_argument('--model', default='nA070_69_K2')
    ap.add_argument('--shots', type=int, default=128)
    ap.add_argument('--controls', type=int, default=120)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--degree', type=int, default=12)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    n, degree = args.n, args.degree
    m = n + 3
    frozen = json.loads(Path(args.freeze).read_text())[args.model]
    dead_sites = [(0, int(a), int(b)) for a, b in frozen['sites']]
    dead_sites += [(1, int(a), int(b)) for a, b in frozen['sites']]
    print(f'[freeze] model={args.model} freeze_sites={len(frozen["sites"])} '
          f'(both passes probed: {len(dead_sites)})')

    # control sites: sample live sites (from the reference majority bitmap) --
    # we know these are live for the reference schedule, so the probe must show 1.
    ctrl_live = json.loads(
        (ROOT / 'notes/runs/e7/freeze_sites.json').read_text())['_meta']['live_sites']
    rng = np.random.default_rng(7)
    live = np.array(ctrl_live)
    sel = rng.choice(len(live), size=min(args.controls, len(live)), replace=False)
    control_sites = [(0, int(live[s][0]), int(live[s][1])) for s in sel]
    control_sites += [(1, int(live[s][0]), int(live[s][1])) for s in sel[:len(sel) // 2]]
    print(f'[control] {len(control_sites)} known-live sites')

    ref = ES.load_module('e7ref', ROOT / 'data' / 'verifiers' / 'dqi'
                         / 'dqi_reference_circuit.py')
    sites = dead_sites + control_sites
    circ, dec, syn = build_probed(ref, m, n, degree, sites)

    r = np.random.default_rng(31337)
    data = C.gen_data('natural', n, degree, args.shots, r)
    bits = C.bits_of(data, degree)
    hits = {k: 0 for k in sites}
    anyprobe = {k: np.zeros(args.shots, dtype=bool) for k in sites}
    for start in range(0, args.shots, args.batch):
        stop = min(start + args.batch, args.shots)
        sim = km.Simulator(batch_size=stop - start, counting_operations=True,
                           seed=11 + start)
        sim.use_same_registers_as(circ)
        sim.write_across_shots(km.array([q for r0 in syn for q in r0]),
                               bits[:, start:stop])
        sim.do(circ)
        for key, reg in dec._e7_map.items():
            v = np.asarray(sim.read_across_shots(reg)).astype(bool)
            anyprobe[key][start:stop] = v
    dead_hits = sum(int(anyprobe[k].sum()) for k in dead_sites)
    ctrl_hits = sum(int(anyprobe[k].sum()) for k in control_sites)
    fwd_dead = sum(int(anyprobe[k].sum()) for k in dead_sites if k[0] == 0)
    adj_dead = sum(int(anyprobe[k].sum()) for k in dead_sites if k[0] == 1)
    print(f'[result] shots={args.shots}')
    print(f'[result] frozen sites    : forward live-reads={fwd_dead}  '
          f'adjoint live-reads={adj_dead}   (must be 0 / 0)')
    print(f'[result] control (live)  : live-reads={ctrl_hits} / {len(control_sites) * args.shots}'
          f'  ({ctrl_hits / max(1, len(control_sites) * args.shots):.4f})  (must be > 0)')
    if args.json:
        Path(args.json).write_text(json.dumps({
            'model': args.model, 'shots': args.shots,
            'frozen_sites': len(frozen['sites']),
            'frozen_forward_live': fwd_dead, 'frozen_adjoint_live': adj_dead,
            'control_sites': len(control_sites), 'control_live_reads': ctrl_hits,
        }, indent=2))
        print('wrote', args.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())

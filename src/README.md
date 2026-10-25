# `src/` — construction and verification toolchain

These are the scripts that generated the submitted circuits, plus the instruments used to measure
them. They are published to satisfy **Rules §6.4**: the source that *produces* the winning
submissions, together with the compute environment needed to run it.

Only personal absolute paths have been redacted. The code itself is as it was run.

## Directory layout

| directory | role |
|---|---|
| `invsqrt/` | invsqrt constructors: the `opt2`/`opt3` builder chain, the E5 `flags` binary-`s` builder, adder and projection libraries, exhaustive equivalence checkers, and the qubit-break / metadata-width instruments |
| `dqi/` | dqi constructors: the E7 safe-freeze and E8 stacked-CSWAP freeze builders, frozen-site union enumerators, fixpoint solvers, small-scale exhaustive verifiers, and the stacked-zip packer |
| `probes/` | measurement instruments: deviation rate and law, flag semantics, basis/predicate rules, register liveness, price/cap models |
| `common/` | submission assembly (`make_submission.py`), the file-level format validator and its negative test suite, the end-to-end verifier runner, and the simulation-free fast cost model (`dfthc_fastscore.py`, `dfthc_frontier.py`) |

## Environment

| item | value |
|---|---|
| machine | Apple Silicon macOS, 8 cores, 16 GB RAM |
| Python | 3.14.3 in an isolated venv (`$VENV`) |
| libraries | `numpy` 2.5.3, `scipy` 1.18.1, `requests` |
| native dependency | `kickmix`, built from the upstream Qualtran source tree with **Homebrew LLVM 20** |

`kickmix` ships with the competition data as a CPython 3.13 / x86-64 Linux extension and will not
load on Apple Silicon. Build it yourself:

```bash
cd <QUALTRAN_KICKMIX_SOURCE>          # Qualtran `main` — only `main` carries pyproject.toml
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export SDKROOT=$(xcrun --show-sdk-path)
"$VENV/bin/pip" install ./kickmix
```

> **LLVM 20 is required; Apple clang 15 is not sufficient.** Apple clang 15 lacks C++20
> parenthesized aggregate initialisation (P0960) and fails on `cmd/main_diagram.cc`
> (`RegisterId(reg_idx)`) and `mem/stride_span.h:82` (`stride_ptr<const const BitId>(a, b)`).
> This is an upstream source dependency, not a local misconfiguration.

## Typical invocations

```bash
P="$VENV/bin/python"

# --- invsqrt ---
$P invsqrt/invsqrt_flags_circuit.py     # build the E5 flags circuit      -> invsqrt_flags.kmb
$P invsqrt/invsqrt_opt3_circuit.py      # build the E3 opt3 circuit       -> invsqrt_opt3.kmb
$P invsqrt/e5_flags_check.py            # exhaustive equivalence, n_p = 2..6, all x
$P invsqrt/e5_flags_measure.py          # measure Q / T / C
$P invsqrt/invsqrt_equiv.py             # circuit-level equivalence harness
$P invsqrt/invsqrt_qbreak.py            # qubit-count break-down by stage

# --- dqi ---
$P dqi/e8_union_trend.py                # frozen-site union vs register width
$P dqi/e8_fixpoint.py                   # solve for the freeze fixpoint
$P dqi/e8_build.py                      # build the E8 stacked freeze      -> dqi_stacked.kmb
$P dqi/e8_verify_small.py               # small-scale exhaustive correctness proof
$P dqi/e7_freeze_from_union.py          # E7 (multiply-register) freeze
$P dqi/make_stacked_zip.py              # swap dqi.kmb in an existing archive

# --- probes ---
$P probes/e6_deviation_rate.py          # eps+ and its law
$P probes/e6_deviation_law.py
$P probes/e6_liveness.py                # register liveness
$P probes/e6_predicate_rules.py
$P probes/e6_flags_port.py

# --- assembly and verification ---
$P common/make_submission.py --invsqrt invsqrt_flags.kmb --dqi dqi_stacked.kmb \
                             --dfthcsos <OFFICIAL_DFTHCSOS_DIR> --out submission.zip
$P common/validate_submission_format.py submission.zip
$P common/test_format_validator.py      # nine negative cases; the validator must not be vacuous
$P common/verify_all.py --runs 3        # three official verifiers, multi-salt
```

`verify_all.py` records status, wall clock and **peak RSS** per verifier, and automatically adds
`--max-workers 2` to `dqi` — the default of four workers peaks near 6 GB each and will exhaust a
16 GB machine.

## Why `common/validate_submission_format.py` exists

**Passing the standalone verifier is not the same as having the submission accepted.** The official
metric performs file-level checks *before* any simulation, and those checks appear nowhere in the
`verifiers/` tree:

1. exactly one `invsqrt.km[xb]`, exactly one `dqi.km[xb]`, and exactly one `dfthcsos/` directory,
   found by **recursive** search. The `find_one` helper globs the literal pattern `invsqrt.km[xb]`,
   so a file renamed to `backup_invsqrt.kmb` is invisible to the check — but nesting the real file in
   a subdirectory, or shipping both `.kmb` and `.kmx`, rejects the whole archive.
2. maximum qubit/bit id `< 699,050` (`2**30 // 1536`). Not checked by the standalone verifiers at all.
3. `submission.npz` must be npy format 1.0, float dtype, little-endian, total ≤ 64 MiB. The standalone
   verifiers accept integer arrays and big-endian.

The validator re-uses the metric's own `find_one` / `read_circuit` / `read_npz` helpers so that it
cannot drift from the real rules. It **simulates nothing** — it only proves the archive reaches the
verifier stage. Correctness must come from the official `verifiers/`, which need the competition
data.

## Notes

* Benchmark members other than the one under construction are copied from the official sample
  archive. `dfthcsos` was never modified, which is why its ratio is exactly `1.000000` throughout.
* A free upload rehearsal (start the blob upload and `PUT` the archive, without creating a
  submission) was run before every credited submission. No competition credit was ever spent
  discovering that an archive was malformed.
* These scripts read competition inputs from a local `data/` directory that is **not** part of this
  repository. See the top-level `README.md` §4.2 for how to obtain them.

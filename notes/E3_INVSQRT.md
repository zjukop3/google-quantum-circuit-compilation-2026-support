# E3 — invsqrt: the E2 config is infeasible; a different 1.146x was landed instead

Status: **submitted** (id `57612302`, 2026-10-25T01:03:44Z). One credit used, four remaining
today. Successor to `E2_OPTIMIZATION_FRONTIER.md`.

## 0. TL;DR

1. **The "verified 1.13x" configuration in `E2_OPTIMIZATION_FRONTIER.md` §3 does not exist.**
   `delta_bits=13 / coef_bits=16 / square_bits=27 / product_bits=27 / result_bits=28` has worst-case
   relative error **7.4998e-08**, which is **1.258x the 2^-24 = 5.9605e-08 tolerance**. The shipped
   verifier returns `FAIL`, not a score. E2's numbers for that row (`3.801e-08`, "margin 1.57x")
   are wrong, as is its baseline entry (`1.998e-08`, "margin 2.98x").
2. What *is* real: **a strictly-better Horner register layout** that removes one whole
   `coef_bits`-wide coefficient register at **identical arithmetic and identical Toffoli count**,
   worth **1.0711x** on its own.
3. Combined with the best *feasible* precision point, the verified result is

   | | Q | Toffolis | rotation_eps | C | score |
   | --- | --- | --- | --- | --- | --- |
   | reference (submitted 1.00000) | 260 | 7735 | 1.5289e-12 | 7987.59 | 2,076,774.23 |
   | **E3 (submitted)** | **239** | **7327** | 3.6809e-13 | **7583.13** | **1,812,367.40** |

   **invsqrt ratio = 1.145898x**, 3 salts bit-identical, `status: PASS`,
   `worst_error_over_tolerance = 0.9251076980`.

- dqi and dfthcsos in the uploaded archive are **byte-identical** to the reference submission
  (verified member-by-member: `dqi.kmb` sha256 `959471d0…`, all 11 `dfthcsos/*` unchanged).
- **Submitted and scored: public 1.04644, rank 652/769** (was 1.00000, 653/662). One credit of
  five spent; the reserved second submission was not needed.

## 1. Falsifying the E2 configuration

### 1.1 The analytic model *is* the verifier

`nr_reference_circuit.relative_error` evaluates `|y1/2**F - 1/sqrt(x)| * sqrt(x)` over the whole
domain, and the verifier's pass/fail test is `|theta - target| <= tol(x)` with
`tol(x) = 2^-24/sqrt(x)`. Dividing through, the verifier's ratio is exactly
`max_x(relative error) / 2^-24`. The shipped verifier reports that ratio as
`worst_error_over_tolerance`:

```
analytic max relative error / 2**-24 = 0.47691161085718864
verifier worst_error_over_tolerance  = 0.4769116097939472     (notes/runs/invsqrt_sample_full.json)
```

Agreement to 9 significant figures (the residue is the 2^-64 rounding of `half_turns_to_units`).
So the 1-second numpy model prices a candidate as exactly as the 4 m 42 s, 200M-shot run.

### 1.2 What that says about the E2 table

```python
Precision(delta_bits=13, coef_bits=16, square_bits=27, product_bits=27, result_bits=28)
relative_error(...).max() = 7.499782e-08      # peak at x = 188134
tolerance 2**-24           = 5.960464e-08
ratio                      = 1.2583          -> over tolerance
```

Note `Toffolis = 7097` and `Q = 251` are reproduced **exactly**, so E2 identified the right
circuit parameters — only the error column is wrong. The module's own docstring quotes
`2.84e-8` for the defaults, i.e. the value measured here, not E2's `1.998e-08`.

### 1.3 Empirical confirmation (the real verifier, not the model)

```
$ python data/verifiers/invsqrt/verify_submission_invsqrt.py \
      build/e3/candA_e2_config.kmb --max-x 200192 --shots-per-x 1 --salt e2check
{
  "status": "FAIL",
  "error": "x=2943: phase error -1.159e-09 turns exceeds tolerance 1.099e-09."
}
exit=1
```

The run fails at `x = 2943`, long before its analytic peak at `x = 188134`. Uploading that config
would have burned a credit and produced **no score for invsqrt at all**. It was not uploaded.

E2's §3 conclusion ("the precision knob has ~1.5 bits left, row 2 is worth taking as a floor")
is therefore unsound: the defaults already sit at `ratio = 0.4769`, i.e. **just over one bit** of
relative-accuracy headroom, not 2.98x.

## 2. The lever that is real: three coefficient registers instead of four

Measured stage costs (baseline, forward pass, `num_toffoli_like` deltas):

| stage | Toffolis | | stage | Toffolis |
| --- | --- | --- | --- | --- |
| normalize | 120 | | correct | 28 |
| select_interval | 3 | | multiply_y0 | 867 |
| load_coefficients | **0** | | denormalize | 300 |
| horner | **1071** | | phase_by_bits | 0 |
| square | 442 | | *forward total* | **3874** |
| multiply_m | 1043 | | *circuit total* | **7735** |

`_load_coefficients` costs **zero Toffolis** — it is `gf2_iadd`, a Clifford XOR network. So the four
`coef_bits`-wide coefficient registers are pure qubit cost, and `Q` is a first-class term of the
score.

Horner's rule runs three steps and the reference keeps all four registers live for all of them:

```
step 1:  c2 -= T(c3 * delta)      -> c3 is dead the moment step 1 ends
step 2:  c1 -= T(r1 * delta)      -> r1 lives in c2's slot
step 3:  c0 -= T(r2 * delta)      -> r2 lives in c1's slot; c0 is untouched until now
```

`c3` dies at the end of step 1; `c0` is not needed until step 3. Three registers suffice
(`src/invsqrt_opt_circuit.py`):

```
A <- c0                    (clean at allocation; idle until step 3)
C <- c2 ;  B <- c3
step 1:  C -= T(B * delta)          C = r1
B ^= c3 ;  B <- c1                  (c3 dead -> classical clear, free)
step 2:  B -= T(C * delta)          B = r2
step 3:  A -= T(B * delta)          A = y0
```

**Two registers are not enough**, and this is worth recording: `C` cannot be recycled for `c0`
because after step 1 it holds the *quantum* state `r1`, which no classical XOR can clear. Three is
the floor for this evaluation order.

Consequences: Toffoli count **unchanged** (every load/clear is Clifford), `num_qubits` drops by
exactly `coef_bits`, and — because the arithmetic is untouched — `fixed_point_model` and
`relative_error` still describe the circuit *exactly*.

### 2.1 Equivalence testing

Circuit-level equivalence against the reference, every `x` in `[1, 2^(2n_p+2))`, phases compared
as exact `Fraction`s after `read_shot_phase`:

| n_p | Q_ref | Q_opt | Toffolis (ref/opt) | x restored | scratch clean | opt == ref | opt == model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 228 | 211 | 4205 / 4205 | yes | yes | yes | yes |
| 4 | 236 | 219 | 5395 / 5395 | yes | yes | yes | yes |
| 5 | 240 | 223 | 5948 / 5948 | yes | yes | yes | yes |
| 6 | 244 | 227 | 6473 / 6473 | yes | yes | yes | yes |
| 7 | 248 | 231 | 6970 / 6970 | yes | yes | yes | yes |
| 8 | 252 | 235 | 7229 / 7229 | yes | yes | yes | yes |

`Q_opt = Q_ref - coef_bits` exactly, in every case.

**A bug this caught.** The first version of `_horner3`'s reverse pass emitted the coefficient
re-loads in the wrong order and omitted one of them (`b` is loaded three times forward, so it must
be inverted three times backward). Phases stayed exact and `x` was restored — only the *scratch*
came back dirty, which the small-`n_p` test flagged and the 4 m 42 s run would have flagged by
failing. Fixed; see the `inverse` branch of `_horner3`.

## 3. The feasible precision frontier

`src/invsqrt_frontier.py` prices a candidate in ~1 s: one `fixed_point_model` pass gives both the
accuracy ratio and the exact `min_x (tol(x) - |error(x)|)` the verifier uses to size
`rotation_eps`, hence `T` and `C = Toffolis + T/4`. It also reports `Q_opt = Q_ref - coef_bits`.

Grid: `delta_bits 12..15 x coef_bits 16..19 x square_bits 27..30 x product_bits 27..30 x
result_bits 28..32` (1280 points), swept **exhaustively** — `delta_bits in {12,13,14}` from the
first pass plus `{11,12,15}` re-swept with corrected margins
(`notes/runs/invsqrt_frontier_d111215.json`). Feasible-point counts: `delta=11 -> 0`,
`delta=12 -> 138`, `delta=15 -> 191`. Best feasible points:

| delta/coef/sq/prod/res | ratio | Q_opt | Toffolis | n_rot | eps | C | score | gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **12/17/28/27/28** | 0.9251 | **239** | **7327** | 36 | 3.681e-13 | 7583.13 | **1,812,367** | **1.1459x** |
| 12/17/29/27/28 | 0.9251 | 240 | 7335 | 36 | 3.681e-13 | 7591.13 | 1,821,871 | 1.1399x |
| 13/17/28/27/28 | 0.8469 | 239 | 7447 | 36 | 4.860e-13 | 7701.11 | 1,840,565 | 1.1283x |
| 12/17/28/27/29 | 0.8385 | 241 | 7411 | 37 | 6.822e-13 | 7669.62 | 1,848,379 | 1.1236x |
| 13/17/29/27/28 | 0.7718 | 240 | 7455 | 36 | 7.244e-13 | 7706.20 | 1,849,489 | 1.1229x |

`delta_bits <= 11` is infeasible at **any** point of the grid (0 feasible out of the 320-point
`delta=11` block), so 12 is the edge of the feasible region and `12/17/28/27/28` is the global
optimum of the sweep, not a local one.

Split of the 1.1459x: **1.0711x** from the register merge (260 -> 239 at fixed Toffolis) and
**1.0698x** from the precision point (7735 -> 7327 Toffolis, plus the larger `eps`).

**Closed-form qubit accounting**, which predicts every measured `Q` exactly:

```
Q = 49 + result_bits + 3*coef_bits + square_bits + 2*product_bits
      + max(square_bits, product_bits + 2, result_bits + 1)
```

`num_qubits` is "highest qubit id appearing in an operation + 1", and the registers are allocated
contiguously, so the 28-29 qubits above the register sum are the internal ancillas of the widest
modular adder (`iadd`/`isub` on an `n`-bit slice costs `n-2` ancillas — measured for
`n in {8,16,24,30,32,40,64}`). That peak is `len(z) - 2 = product_bits`, hence the trailing
`product_bits + 1`. Reducing it would need a cheaper adder inside the compiled `kickmix`; the
algorithm-level structure is otherwise tight.

## 4. Full verification (3 salts, 200,540,928 shots each)

`data/verifiers/invsqrt/verify_submission_invsqrt.py build/e3/invsqrt_opt.kmb --salt <S>`

| salt | status | score | Q | toffoli_count | n_rot | rotation_eps | worst ratio | worst_x | sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e3alpha | PASS | 1812367.3976837783 | 239 | 7327 | 36 | 3.6809236372811597e-13 | 0.9251076979764682 | 1119994 | 37c8e6b0… |
| e3bravo | PASS | 1812367.3976837783 | 239 | 7327 | 36 | 3.6809236372811597e-13 | 0.9251076979764682 | 1119994 | 37c8e6b0… |
| e3charlie | PASS | 1812367.3976837783 | 239 | 7327 | 36 | 3.6809236372811597e-13 | 0.9251076979764682 | 1119994 | 37c8e6b0… |

Bit-identical across salts (the tracked phase is deterministic; the salt only reseeds
measurement-based uncomputation). Wall time 183-193 s, peak RSS ~0.5 GB. Logs:
`notes/runs/invsqrt_opt_e3{alpha,bravo,charlie}.json`.

The verifier's own numbers reproduce the model's prediction **exactly** — predicted
`C = 7583.13`, `score = 1,812,367`; verified `toffoli_equivalent = 7583.127186961416`,
`score = 1812367.3976837783`. The 1 s pricer is calibrated.

## 5. Submission

- Archive: `build/submission_e3.zip` (and `build/e3/submission.zip`, same bytes) —
  73,802,172 B. `src/validate_submission_format.py` -> **ACCEPT**.
- Member-by-member diff against the 1.00000 submission: **only `invsqrt.kmb` changed**
  (`e2bf8eb6…` 579,721 B -> `37c8e6b0…` 546,482 B). `dqi.kmb` and all 11 `dfthcsos/*` are
  byte-identical.
- Three free `--upload-only` handshakes ran before the paid step.

**Trap found:** `CreateSubmission` rejects the blob unless it is named exactly `submission.zip`:

```
HTTP 400 {"message":"Submission not allowed: Submission files must be named \"submission.zip\"
for this Competition.","status":"FAILED_PRECONDITION"}
```

`--upload-only build/submission_e3.zip` uploads fine but can never be submitted; the blob name is
`os.path.basename(--zip)`. Use a file literally named `submission.zip`. The failed attempt used no
credit.

| field | value |
| --- | --- |
| submission id | **57612302** |
| submitted | 2026-10-25T01:03:44Z |
| credits | 5 allowed at start, **1 used**, 4 remaining |
| next credit | the daily allowance resets on Kaggle's schedule (`GetSubmissionLimitInfo`) |

Result to be filled in from the leaderboard poll — see §6.

## 6. Score

Expected: `invsqrt = 2,076,774.23 / 1,812,367.40 = 1.145898x`; the other two benchmarks are
untouched at ratio 1, so the geometric mean over three benchmarks is

```
overall = 1.145898 ** (1/3) = 1.04644   (was 1.00000, rank 653/662)
```

### 6.1 Confirmed

| field | value |
| --- | --- |
| status | **COMPLETE** |
| public score | **1.04644** (was 1.00000) |
| stored score on the LB row | `1.04644` |
| rank | **652 / 769** (was 653 / 662) |
| predicted | `1.145898^(1/3) = 1.04644` — matched to all 5 printed decimals |

The whole-competition geometric mean moved by exactly the geometric-mean-of-one-factor
predicted from the single verified benchmark, which independently confirms that the other two
benchmarks are being scored at ratio 1 (byte-identical members).

Fresh leaderboard context (`notes/runs/leaderboard_e3.json`, fetched 2026-10-25):

| | value |
| --- | --- |
| top | 20.10132 |
| median | 6.37294 |
| teams in [0.99, 1.01] | 75 |
| min | 0.97849 |

Neighbours: rank 651 = 1.05071, rank 652 = **1.04644**, rank 653 = 1.04432.

### 6.2 Quota after the run

```
GetSubmissionLimitInfo -> {"numToday": 1, "numTotal": 2, "numAllowedNow": 4}
```

One credit spent (submission `57612302`). The failed filename attempt burned nothing. Four
credits remain today. **No second submission was needed** — the first landed exactly on
prediction, so the reserved second slot is unspent.

## 7. What is left on invsqrt

- **Toffolis 7327, of which ~441 are structural floor** (`normalize` 120, `denormalize` 290,
  `correct` 27, `select_interval` 3) and roughly half of everything else is the mandatory reverse
  pass: the reverse half re-pays every AND, because `del_and` is free but the reverse `init_and`
  is not.
- Retaining ANDs instead of recomputing them buys Toffolis with qubits at ~1:1, and the exchange
  rate is set by `Q/C ~ 239/7583 = 0.0315`: **one saved qubit is worth up to ~32 Toffolis**. That
  is the direction any future structural work has to push — qubit savings, not Toffoli savings.
  The 3-register Horner is exactly such a trade (0 Toffolis, -17 qubits).
- The adder-ancilla term (`product_bits + 1` qubits) is the largest remaining non-algorithmic
  qubit cost, and it is inside the compiled `kickmix` `iadd`/`isub`.
- `coef_bits` is unusually valuable here: it costs **3 qubits per bit** on `Q` (three registers)
  and it buys enough accuracy to shrink `square/product/result` — the frontier's best point uses
  `coef_bits=17` with the smallest feasible `product_bits`.

## 8. Files

| path | what |
| --- | --- |
| `src/invsqrt_opt_circuit.py` | the optimized builder (3-register Horner) + CLI |
| `src/invsqrt_frontier.py` | exact ~1 s pricer / precision frontier sweep |
| `build/e3/invsqrt_opt.kmb` | the submitted circuit (Q=239, 7327 Toffolis, sha256 `37c8e6b0…`) |
| `build/e3/candA_e2_config.kmb` | the E2 config, kept as the FAIL reproduction |
| `build/e3/submission.zip`, `build/submission_e3.zip` | the submitted archive |
| `notes/runs/invsqrt_opt_e3*.json` | the three 200M-shot verifier runs |
| `notes/runs/invsqrt_frontier.json` | the full frontier sweep |

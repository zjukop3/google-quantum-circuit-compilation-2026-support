# E2 — Optimization frontier: where the score can actually move

Status: reconnaissance + negative results. **No new submission was spent.**
Companion to `E1_REPRODUCE.md` (§7 has the pre-upload state) and `UPLOAD_FLOW.md`.

## 0. Where we stand

Overall score is the geometric mean over the three benchmarks
(`notes/metric_kernel_extracted.py:278`):

```
Score = ( prod_{b=1..3} baseline_cost_b / cost_b ) ** (1/3)
```

Our uploaded submission (id `57596361`) scored **exactly 1.00000**, i.e. all three
ratios are 1. Rank **653 / 662**.

| | leaderboard |
| --- | --- |
| top | 17.71378 |
| median | 5.85489 |
| teams in [0.99, 1.01] | 64 |
| min | 0.97849 |

So ~650 teams beat the baseline, the median team is ~**200x** aggregate better
(5.855^3), and the leader is ~**5556x** aggregate better (17.714^3).

## 1. Per-benchmark cost laws (confirmed by reading the shipped verifiers)

```
invsqrt    score = Q * C,        C = Toffolis + T/4,   Q = circuit.num_qubits
dqi        score = R(eps) * Q**2 * T
dfthcsos   score = lambda_eff * T_total * Q_total
dfthcsos   lambda_eff = sqrt( E_gap * (2*Lambda - E_gap) )
```

Baselines (from `notes/runs/*_sample_full.json`, all re-confirmed by the dfthcsos
fast model):

| | Q | T | score |
| --- | --- | --- | --- |
| invsqrt | 260 | 7987.59 Toffoli-equiv | 2.076774e6 |
| dqi | 1893 | 5.725108e6 | 2.443946e13 |
| dfthcsos | 730 | 93237 | 3.3556e9 |

## 2. dfthcsos — exactly reproduced, and the levers are closed

Exact decomposition of the baseline (T = 93237, Q = 730):

| term | value | share of T |
| --- | --- | --- |
| ANGLES (2 passes) | 79480 | **85.25%** |
| inner_state_prep | 9461 | 10.15% |
| givens_rotations | 3450 | 3.70% |
| outer_state_prep | 846 | 0.91% |

Q = 730 = persistent 186 (outer 46 + inner 44 + `J*brot` 96) + temporary 544
(inner QROAM, `k2_opt = 5`).

### 2.1 The ANGLES family is a closed form

`A = N + R*B = 76 + 65*60 = 3976`, `K = ceil((N-1)/J) = ceil(75/8) = 10`, and each
circuit is a full binary segment-tree unary-iteration QROM. With `init_and` = 1
CCX and `del_and` uncomputed by measurement (free), the per-circuit cost is
exactly the number of internal nodes = `A - 2 = 3974` (matches
`max_magic 3974` on every circuit).

```
T_angles = 2 * K * (A - 2) = 2 * ceil(75/J) * (76 + R*B - 2)
```

The three controlling quantities, and why each is pinned:

* **`brot = 12` is forced.** Word registers need `12 + J*brot + (sel-1) <= 128`,
  and `brot` is the angle quantisation width. Sweep on the baseline words:

  | brot | error_2 | lambda_eff | J | K | T_total | verdict |
  | --- | --- | --- | --- | --- | --- | --- |
  | **12** | **0.018640** | 49.30137 | 8 | 10 | 93237 | **the only feasible point** |
  | 11 | 0.05111 | 50.40211 | 9 | 9 | 84989 | REJECT (>0.020) |
  | 10 | 0.13918 | 56.67143 | 10 | 8 | 76741 | REJECT |
  | 8 | 0.67407 | 109.10477 | 13 | 6 | 60245 | REJECT |

  11 bits already blows the residual gate by 2.6x. `brot = 12` sits on the
  boundary of feasibility, not in the middle of it.

* **`J = 8` is forced** by the 128-qubit cap once `brot = 12` and the 11 tree
  ancillas are paid for: `12 + 96 + 11 = 119`. Hence `K = 10`.

* **`A = 3976` is forced** — see §2.3.

### 2.2 The angle data is incompressible

`angle_words` is `(3976, 75)`, all 4096 12-bit values occur, entropy 11.47 bits,
zero all-zero rows, zero constant columns, 31/75 columns need 11-12 bits. So
Qualtran's QROM `_break_early` collapse (`all rows in subtree equal`) never fires
— every internal node must be built. There is no data-side shortcut.

### 2.3 (R, B, C) cannot shrink — measured, not assumed

Truncating the baseline is *itself a legal submission* (the verifier only reads
array shapes and checks behaviour), so this is a hard feasibility probe; and a
re-trained factorization of the same size would beat a truncated one. Every
truncation point fails:

| (R',B',C') | A | error_2 | verdict |
| --- | --- | --- | --- |
| (65,60,60) | 3976 | 0.018640 | baseline |
| (56,60,60) | 3436 | 0.29536 | REJECT (16x over) |
| (48,60,60) | 2956 | 1.02872 | REJECT |
| (65,56,60) | 3716 | 0.46082 | REJECT |
| (65,50,60) | 3326 | 0.86674 | REJECT |
| (65,60,56) | 3976 | 0.55036 | REJECT |
| (65,60,48) | 3976 | 1.27264 | REJECT |
| (40,40,40) | 1676 | 8.17539 | REJECT |

The organizers trained the baseline *right against* the 20 mHa wall: error_2 is
0.018640 vs a 0.020 gate, and dropping **one** rank costs a factor of 16. There
is no free rank to give back. Lowering `A` requires a genuinely better
factorization, not a smaller one.

### 2.4 The only live dfthcsos lever

`lambda_eff` is the sole term that can still move: `E_gap` is pinned near
3.914898 by the floor, so

```
lambda_eff ~ sqrt(2 * E_gap * Lambda),   Lambda = 308.456 -> floor 49.0
```

| Lambda | lambda_eff | gain |
| --- | --- | --- |
| 308.456 (baseline) | 49.3014 | 1.000x |
| 49.0 (hard floor) | 19.1920 | **2.57x** |

That is the entire remaining dfthcsos upside from λ alone, and it needs Lambda
pushed 6.3x down (a 6.3x smaller two-body LCU one-norm) while holding error_2
under 20 mHa. Pushing `E_gap` down from 3.9655 to the 3.914898 floor is worth
only 1.006x on its own.

**Conclusion: dfthcsos is capped near 2.6x from where we are, absent a research-
grade factorization breakthrough.** Do not spend submission credits here early.

## 3. Where the headroom actually is

### invsqrt — `score = Q * C`, both terms equal weight

`C = 7735 Toffoli + (1 T + 37 rotations)/4 = 7987.59`. The rotations contribute
252.59 = **3.2%** of C, so synthesising them away is worth at most 3.2%. The
7735 Toffolis are 96.8% of the cost. Need ~1.73x on *both* Q and C for a 3x.
The paper's own hint: a cubic polynomial over a small rescaled range may be
replaceable by a simpler polynomial over a larger range.

### invsqrt — structure, and what is (and is not) available

Spec: implement `|x>|0..0> -> exp(2 pi i / sqrt(x))|x>|0..0>` for every
`x in [1, max_x]`, `max_x = 3(2^10 - 2)^2 = 3,133,452`, `width = 22`, `beta = 1`,
tolerance `tol(x) = 2^-24 / sqrt(x)` (a **relative** bound; absolute floor
`2^-48` turns). x must be restored, all scratch back to `|0>`.

The baseline (`data/verifiers/invsqrt/nr_reference_circuit.py`, 401 lines) is
normalize -> cubic initial guess on `[1,4)` -> one Newton-Raphson step -> phase
by bits -> uncompute, with `Precision(delta_bits=14, coef_bits=17,
square_bits=28, product_bits=28, result_bits=29)`.

Three things measured:

1. **The reverse pass pays full Toffoli cost.** The docstring says "~3.9k
   Toffolis per direction", matching `7735 = 2 x ~3867`. `_and()` correctly uses
   the free `del_and` when `inverse=True`, but **`_add_row()` -- which every
   multiplier (`_square`, `_multiply_m`, `_multiply_y0`, `_horner`) is built
   from -- calls `init_and` unconditionally in both directions.** So every
   re-computed AND in the reverse half costs a fresh Toffoli.

2. **But retaining those ANDs is a losing trade.** Skipping the forward
   `del_and` and reusing the live ancillas in reverse would halve C (~3870) at
   the cost of keeping ~3870 ancillas live simultaneously (the `scratch` slice is
   reused per call today). Score is `Q*C`, so that is `4100 * 3870 = 1.6e7`
   against today's `260 * 7988 = 2.08e6` -- **7.6x worse**. `Q` is far too
   valuable for that. Any scheme that buys Toffolis with qubits at worse than
   1:1 is dead on this benchmark.

3. **The precision knob has ~1.5 bits left in it, no more.** `main()`'s
   `relative_error` / `fixed_point_model` evaluate the worst case over the whole
   domain in numpy, and `build_nr_circuit` prices a candidate in <0.1 s, so this
   is a genuine instant inner loop:

   | Precision | worst rel. err | margin vs 2^-24 | Toffolis | Q |
   | --- | --- | --- | --- | --- |
   | shipped defaults | 1.998e-08 | 2.98x | 7735 | 260 |
   | delta 13, coef 16, sq 27, prod 27, res 28 | 3.801e-08 | 1.57x | 7097 | 251 |
   | delta 14, coef 16, sq 26, prod 26, res 27 | 5.087e-08 | 1.17x | 7017 | 246 |
   | delta 13, coef 15, sq 25, prod 25, res 26 | overflow | -- | -- | -- |

   Row 2 is roughly a **1.13x** score gain (251 x (7097 + ~230) vs 2.077e6) and
   leaves 36% of the tolerance for rotation-synthesis error, so it is worth
   taking as a floor -- but it is a 13% win, not a 3x one. The shipped defaults
   are already well tuned; the error is `2^-bits`, so the whole 2.98x margin
   buys ~1.5 bits.

   For reference the *simulated* run only used 47.7% of tolerance
   (`worst_error_over_tolerance: 0.4769`), so the analytic worst case is
   conservative -- but the verifier spends the leftover on rotation synthesis, so
   it is not free either.

**Conclusion for invsqrt:** the Toffoli budget is a straight-line sequence of
column-truncated multiplier rows, already trimmed, and the qubit budget is
already lean. A real win has to come from **replacing the algorithm**, not from
tuning it. The paper's own hint is the direction: a *simpler* polynomial over a
*larger* range instead of the cubic on `[1, 1.5]`-class intervals. A
phase-gradient construction (segment index -> QROM -> base phase + slope, one
controlled rotation, no multiplier chain) is the other candidate; the linear-
interpolation error budget `(3/32)(L/x)^2 <= 2^-24` puts the segment count
around `2^16`-`2^17`, which is the number to check first.

### dqi — `score = R(eps) * Q**2 * T`

**Q is squared.** `R(eps) = 1.19126` is already at its floor (`eps+` ~ 8.7e-4,
0 failures in 102400 shots). So the levers are `Q = 1893` and `T = 5.725e6`,
and a 1% qubit cut is worth a 2% score cut. The paper lists the targets:
explicit remainder storage, pointer-driven EEA steps, polynomial-basis F2^12
arithmetic, per-position Itoh-Tsujii inversion.

**Arithmetic check on what reaching the median requires:** if dfthcsos tops out
at 2.6x, then matching the median 5.855x needs `invsqrt * dqi ~ 77x`, e.g. 10x
and 7.7x. The leader at 17.714x needs `invsqrt * dqi ~ 2138x`. That is where the
real work is, and it is in invsqrt + dqi, not dfthcsos.

## 4. Tooling built this session (fast iteration loop)

| tool | what | cost |
| --- | --- | --- |
| `src/dfthc_fastscore.py` | exact verifier metrics **without** the kickmix simulation; plus `cost()` for T_total/Q_total from (A, J, brot, R, B, C) | **15.8 s** |
| `src/dfthc_frontier.py` | (R,B,C) truncation feasibility sweep | 25 s for 19 points |
| `example_submission/angle_k_reference_circuit.py` | rebuild the 10 ANGLE circuits from a `submission.npz` | **2.0 s** |
| `invsqrt.nr_reference_circuit.relative_error` + `build_nr_circuit` | invsqrt accuracy **and** (Toffolis, Q) for any `Precision` | **< 0.1 s** |
| `python -m dfthcsos.verify RUN_DIR --salt S` | full ground-truth verify | 1 m 54 s |

The invsqrt pair is the best inner loop in the repo: `relative_error` is pure
numpy over the whole domain and `build_nr_circuit` returns a circuit whose
`operation_statistics().num_toffoli_like` and `num_qubits` give Q and C
directly, so a precision/algorithm candidate is priced instantly. Only the
finalist needs `verify_submission_invsqrt.py` (200M shots).

`dfthc_fastscore.py` reproduces the shipped verifier bit-for-bit on the baseline
(score 3.356e9, T 93237, Q 730, lambda_eff 49.30137, error_2 0.018640), so a
candidate can be screened in 16 s and only the finalist needs the 2 min run.

## 5. Competition economics (relevant to "win a prize")

* 1st $50k / 2nd $25k / 3rd $10k / 4th $5k.
* **Mid-Competition Milestone Prize $10,000** — highest public leaderboard score
  at 11:59 PM UTC **2026-11-20**; requires a **public notebook**, kept public
  until the final deadline.
* **Longest Leader $10,000** — best score held longest; notebooks public by
  **2027-01-04**.
* **Single Task Performance Prizes $30,000** — $10k for the best score on *each*
  individual task. This is the most reachable of the extras: it needs leadership
  on one benchmark, not on the geometric mean.
* Entry deadline 2027-01-13, final submission 2027-01-20.

The mid-competition and single-task prizes both reward depth on one benchmark,
and both require a **public notebook** — which has to be written and published,
not just a submission uploaded.

## 6. Recommended order of work

1. **Do not spend credits on dfthcsos micro-gains** (§2 — capped at ~2.6x).
2. **invsqrt first** — smallest circuit (260 qubits, 7735 Toffolis), fastest to
   iterate, and the paper names a concrete structural change. Target: the
   7735-Toffoli polynomial evaluation and the 260-qubit width.
3. **dqi second** — `Q**2` makes a qubit diet the highest-leverage change; the
   paper lists four specific subsystems.
4. **dfthcsos only if a factorization that reaches `Lambda -> 49` appears.**
5. Publish a notebook early — the $10k milestone prize needs one by 2026-11-20
   and recent leaderboard duration needs it public by 2027-01-04.

---

# E2b — dqi is fully decomposed (measured, not inferred)

Everything below is measured by building the reference circuit at reduced
parameters and differencing; the extrapolation reproduces the verifier's own
static count **exactly**.

## b.1 The cost split

`build_reference_circuit(m, n=70, degree=12)` with `m` = number of classical
`E` registers = number of Chien positions (`num_positions = min(4095, len(E))`):

| m | `num_toffoli_like` |
| --- | --- |
| 70 | 4,312,686 |
| 140 | 4,337,256 |
| 280 | 4,386,396 |

slope = **351.00 T / position**, intercept = **4,288,116 T**.

```
extrapolate m -> 4095:  4288116 + 4095*351.00 = 5,725,461
verifier max_toffoli_count                    = 5,725,461   EXACT
```

So

```
T = 5,725,108  (mean)  =  EEA 4,288,116 (74.9%)  +  Chien 1,437,345 (25.1%)
```

and `max - mean = 353`, i.e. **only 353 Toffolis were skipped over the whole
run**. The EEA is also cleanly linear:

| n_iters | `num_toffoli_like` |
| --- | --- |
| 20 | 441,942 |
| 40 | 802,502 |
| 80 | 1,558,742 |
| 160 | 3,108,894 |

slope = **19,049.7 per `n_iters` unit** = 2 directions x **9,524.8 per
single-direction EEA iteration**; 222 iterations -> 4,289,973 (measured
intercept 4,288,116).

## b.2 What the 353 skipped Toffolis prove

The verifier counts `T` as the **mean** number of Toffoli-like gates *executed*
per shot, and `push_condition`/`pop_condition` on a **classical** bit makes the
enclosed instructions not execute at all on shots where the bit is false.

The reference has exactly one such condition: in the Chien loop,

```python
builder.bit_store(is_error[0], True, control=error_mbuc[j])
builder.push_condition(is_error[0])
```

`error_mbuc[j] = M_j` is uniform in GF(2^12), so `P(M_j = 0) = 1/4096` and the
expected skipped work is `4095 * 351 / 4096 = 351.0` -- which is exactly the
observed gap **353**. 

Two conclusions:

1. **Classical conditioning demonstrably works on this verifier** and is worth
   real Toffolis -- this is a proven-effective lever, not a hypothesis.
2. **The EEA (74.9% of T) has ZERO classical conditioning.** Its `mul` flag, its
   four `UnaryIterationCursor`s and its counters `n_A, n_a, n_q, n_b, n_B` are
   all **quantum** (`alloc_qubits`), so the per-position `ixor_gf2_mul` executes
   on every shot even though `mul` selects a narrow active window:
   `lim_a = min(m-2, it//2 + 1)`, `lim_A = m-2`.

## b.3 Primitive prices (measured, GF(2^12), one call each)

| call | toffoli_like | note |
| --- | --- | --- |
| `gf2_imul(const, target=reg)` | **0** | lowered to `gf_imul_classical` -> `linear_map` -> CX/SWAP only |
| `del_and` / `del_gf2_inverse_with_scaffold` / `del_gf2_div_with_scaffold` | **0** | stabilizer + measurement |
| `multi_controlled_x` (12 controls) | 11 | |
| `flip_if_equal(12-bit, 0)` | 12 | |
| `init_and` (12 lanes) | 12 | |
| `init_gf2_mul(a, b)` / `ixor_gf2_mul(a, b -> c)` | 63 | |
| `init_gf2_inverse_with_scaffold` | **315** | |
| `init_gf2_inverse` (no scaffold) | 315 | same T, 24 more qubits |
| `init_gf2_div_with_scaffold(a,b)` | **378** | |

Replicating the exact Chien loop body standalone gives **351.0 T/position**,
matching the measured slope to the Toffoli -- so the Chien cost is fully
accounted for by `315 (inversion) + 12 + 12 + 12` and the free uncomputations.

**A constant GF(2^12) multiply is already free.** The 70 `step_multipliers` in
the Chien loop (`gf2_imul` / `gf2_idiv` by `primitive ** k`) cost nothing. My
first estimate of `1398/position = 70 x 20` was an arithmetic coincidence, not
the mechanism -- do not "optimize" the step multipliers, they are already free.

## b.4 Where the EEA's 9,525 T/iteration goes

`_multiply_and_add_or_sub` sweeps **all m = 73 positions** every iteration:

```python
for i in order:                                  # 73 iterations
    if is_adjoint and multiplies:  builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], control=mul)
    if len(swaps) == 1:            builder.cswap(swaps[0], q_i, reg_a[i])   # 12 CSWAP = 12 T
    if not is_adjoint and multiplies: builder.ixor_gf2_mul(...)             # 63 T
```

`multiplies = i < lim_a or i >= 2` is true for ~71 of the 73 positions, so the
sweep emits ~71 x 63 = 4,473 T of `ixor_gf2_mul` plus 73 x 12 = 876 T of
`cswap`, on top of two `init_gf2_div_with_scaffold` (2 x 378) and the four
unary-iteration cursors. The control `mul` is a **qubit**, so none of it is
elidable.

## b.5 What this means for the score

`score = R(eps_plus) * Q**2 * T`, and `R` is **not improvable**: the run reports
`value_errors: 0, phase_errors: 0` out of 102,400 shots, so `eps_plus` is the
one-sided Clopper-Pearson bound at `delta = 2^-128` (~8.66e-4) and
`R = 1.1913` is a pure statistical floor. **Only Q and T remain.**

* T is 74.9% EEA. Making the EEA sweep classically conditioned -- the one
  lever the 353-Toffoli proof says would work -- is the single biggest target on
  this benchmark. It is a real restructuring (counters and the `mul` flag would
  have to carry classical state and still be uncomputed), not a knob.
* Q = 1893, of which `reg_b` is 73 x 12 = 876 (**46%**) and the 70 syndromes are
  840. Because Q is **squared**, a 10% qubit cut is worth 19% of score -- the
  exchange rate is: Q can grow 1% only if T drops 2.03%.
* Note the two benchmarks have **opposite exchange rates**: dqi rewards trading
  Toffolis for qubits (up to 2:1), invsqrt punishes it (Q*C is symmetric, and
  the AND-retention idea lost 7.6x).

## b.6 Cost of a candidate run

`build_reference_circuit(70, 70, 12)` -> `411`, `(m=4095)` builds in ~3 s at
m<=280; the full m=4095 circuit is the 886 MB `build/dqi_reference.kmb`. The
verifier itself is the expensive step (102,400 shots over 86M instructions).
Iterate on `num_toffoli_like` / `num_qubits` first, verify once at the end.

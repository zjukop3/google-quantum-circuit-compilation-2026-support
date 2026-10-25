# Google Quantum AI — Verifiable Quantum Circuit Compilation (2026) — Support Materials

Reproducibility package for our submissions to the Kaggle competition
**"Google Quantum AI — Verifiable Quantum Circuit Compilation"**, published as required by
**Rules §6.4 (Open Source and Prize Obligations, B.5 / B.7 / C.6.b)**.

License: **Apache 2.0** (see [`LICENSE`](LICENSE)) — applies to the submission artifacts and to all
code in this repository.

**Current leaderboard state: public score `1.11751`, rank `710`** (submission `57625941`,
2026-10-25T10:07:51.693Z).

---

## 1. What this package is

The competition scores three independent circuit-compilation benchmarks. The overall score is the
geometric mean of the three cost ratios against a reference implementation:

```
Score = ( prod_{b=1..3}  baseline_cost_b / cost_b ) ** (1/3)
```

| benchmark | cost law (read from the shipped verifiers) |
|---|---|
| `invsqrt` | `cost = Q * C`,  `C = Toffoli + T/4`,  `Q = circuit.num_qubits` |
| `dqi` | `cost = R(eps+) * Q**2 * T` |
| `dfthcsos` | `cost = lambda_eff * T_total * Q_total`,  `lambda_eff = sqrt(E_gap * (2*Lambda - E_gap))` |

This repository contains:

| Path | Contents |
|---|---|
| [`notes/`](notes/) | The eight methodology reports (`E1`–`E8`): reproduction of the reference solution, the optimization frontier, the invsqrt redesign, the dqi deviation analysis and the two dqi register-freeze constructions. **Written in Chinese.** |
| [`notes/SUBMISSIONS.md`](notes/SUBMISSIONS.md) | Full submission ledger — every submission id, timestamp, package digest and measured public score. |
| [`notes/REPO_PUBLISHED.md`](notes/REPO_PUBLISHED.md) | Publication evidence for this repository (URL, commit hash, timestamp, pre-push scan result). |
| [`src/invsqrt/`](src/invsqrt/) | Constructors for the invsqrt submission: `opt3` and `flags` circuit builders, adder/projection libraries, equivalence and qubit-break instruments. |
| [`src/dqi/`](src/dqi/) | Constructors for the dqi submission: the `E7` safe-freeze and `E8` CSWAP-freeze builders, union enumerators, fixpoint solvers, small-scale verifiers, and the stacked-zip packer. |
| [`src/probes/`](src/probes/) | The probe instruments used to measure deviation rates, flag semantics, basis/predicate rules and register liveness on the real verifier. |
| [`src/common/`](src/common/) | Submission assembly and format validation, the fast (simulation-free) cost model, and the end-to-end verifier runner. |
| [`metrics/`](metrics/) | Machine-readable copy of the benchmark table above ([`scores.csv`](metrics/scores.csv), [`scores.json`](metrics/scores.json)). |

### What is deliberately **not** in this repository

* No competition data, no verifier sources, no submission binaries. `dqi.kmb` alone is ~887 MB and each
  `submission.zip` is ~72 MB; the rules forbid redistributing the competition's own inputs anyway.
  See §4 for how to re-obtain the inputs and rebuild them.
* No credentials, cookies, session files or tokens of any kind. The package was scanned for
  credential-shaped strings before publication — see [`notes/REPO_PUBLISHED.md`](notes/REPO_PUBLISHED.md).
* No scraped forum content and no third-party material.

---

## 2. Results

### 2.1 Submitted lineage

Six submissions were spent over the competition window. Scores below are **measured public LB scores**,
not predictions.

| # | submission id | submitted (UTC) | content | public score | rank at settle |
|---|---|---|---|---|---|
| 1 | `57596361` | 2026-10-24T13:51:19.913Z | reference reproduction (all three = baseline) | `1.00000` | 653 / 662 |
| 2 | `57612302` | 2026-10-25T01:03:44Z | invsqrt `opt3` (E3) | `1.04644` | 652 / 769 |
| 3 | `57617295` | 2026-10-25T04:51:52.263Z | invsqrt `opt3` + qubit trade (E4) | `1.04905` | 681 |
| 4 | `57618221` | 2026-10-25T05:32:54.983Z | invsqrt `flags` binary-`s` encoding (E5) | `1.08145` | 678 |
| 5 | `57623058` | 2026-10-25T08:27:07Z | dqi multiply-register safe freeze (E7) | `1.09489` | 713 |
| 6 | **`57625941`** | **2026-10-25T10:07:51.693Z** | **dqi CSWAP-register stacked freeze (E8)** | **`1.11751`** | **710** |

The productive line is 1 → 6: **+11.75 %** overall, driven by two independent levers (§2.3).

### 2.2 Final per-benchmark decomposition

| benchmark | reference cost | our cost | ratio | note |
|---|---:|---:|---:|---|
| `invsqrt` | `Q=260`, `T=7,735`, `C=7,987.59`, score `2,076,774.23` | `Q=231`, `T=6,852`, `C=7,108.127187`, score `1,641,977.380` | **`1.264800755`** | three-salt verifier agreement, bit-for-bit |
| `dqi` | `Q=1,893`, `T=5,725,109.856`, score `2.443946e13` | `Q=1,893`, `T=5,164,840.586`, `eps+ = 9.1042e-4` | **`1.103410`** | ratio under the *official salted* sample point |
| `dfthcsos` | `Q=730`, `T=93,237`, score `3.3556e9` | unchanged | `1.000000` | no submission was spent on this benchmark |

```
(1.264800755 × 1.103409985 × 1.000000) ** (1/3) = 1.11751    <- matches the measured 1.11751
```

### 2.3 The two levers, in one paragraph each

**invsqrt — `flags` re-encoding (E5, ratio `1.26480`).** The reference carries a 10-bit *unary
thermometer* `flags[k] = [s >= k+1]` tracking the number of leading zero pairs `s` of the input. We
showed `s` can be carried as a compact binary register and re-expanded on demand inside the two
control paths that actually read it (`_select_interval` never does), reducing the peak qubit count
`Q` from 237 to **231** while *also* removing 482 Toffoli-equivalents (`T` 7,334 → **6,852**). The
trade is negative — qubits and gates fall together. Exhaustive equivalence over all inputs for
`n_p = 2..6`, and three independent salted full verifier runs agreeing bit-for-bit.

**dqi — register-freeze constructions (E7/E8, ratio `1.103410`).** The dqi score is
`R(eps+) * Q^2 * T`, where `R` is a repetition factor driven by the sampled error rate. We
characterised *where* the residual deviation on the frozen registers actually lives (E6), then built
two successive freezes: a multiply-register freeze (E7) and a stacked CSWAP-register freeze (E8),
each validated by enumerating the frozen site union and solving for a fixpoint, and each given a
small-scale exhaustive correctness proof before pricing. The `T` side is sample-independent and was
confirmed bit-identical across independent salts; the `R` side moves with the sample point and is
reported honestly in §3.

---

## 3. Honest limits of these claims

These are stated because §6.4 asks for a *reproducible methodology*, and a methodology that hides its
own variance is not reproducible.

1. **`R` contains a measurable "salt lottery".** The official scorer merges each row's `seed_salt`
   with the submission hash; a different salt draws a different sample point. For our *final*
   artifact, two independent salts both produced exactly **one** erroneous shot out of 102,400,
   raising `eps+` from `8.6606e-4` (unsalted) to `9.1042e-4` and `R` by `+0.4593 %`. This is why the
   measured `1.11751` sits `−0.00171` below the unsalted prediction `1.11922`. Both numbers are
   reported; the residual against the *salted* model is `−0.00000` to five decimals.
2. **Offline (`salt=''`) reproduction is optimistically biased by ≈ `+0.46 %`** on the dqi term.
   Future budgets must report `LB_expected` and `LB_optimistic(k=0)` separately.
3. **"Zero violations over 102,400 shots" is not a distributional guarantee.** The E8 post-mortem
   proved the frozen sites *do* light up on other sample points; the magnitude is ~1 shot in 102,400
   and the resulting `eps+` still passes the 1/16 threshold by a wide margin (`9.1e-4` vs `0.0625`).
4. **`dfthcsos` was explored and then left alone.** The angle-preparation family accounts for 85.25 %
   of its `T` and is expressible in closed form; the levers we could reach inside the remaining
   budget were priced as unattractive. Ratio `1.000000` is therefore a deliberate choice, not an
   oversight.
5. **The reference reproduction is exact, not approximate.** Submission `57596361` scored exactly
   `1.00000` with `contentLength = 73,803,082` bytes read back byte-for-byte identical to the local
   archive. The scoring model was then confirmed in the other direction: `1.145898 ** (1/3) = 1.04644`
   matched submission `57612302` to all five printed decimals.
6. **No competition submission credit was spent on anything in this repository after 2026-10-25T10:07:51Z.**
   Everything in `notes/E6` onward beyond the ledger is analysis and construction.

---

## 4. Reproducing our submission

### 4.1 Compute environment (required by §6.4)

| item | value |
|---|---|
| machine | Apple Silicon macOS, 8 cores, 16 GB RAM |
| Python | 3.14.3, in an isolated venv |
| libraries | `numpy` 2.5.3, `scipy` 1.18.1, `requests` |
| native dependency | `kickmix` — **compiled locally from the Qualtran source tree** (the copy shipped with the competition data is CPython 3.13 / x86-64 Linux only) |
| toolchain for `kickmix` | **Homebrew LLVM 20** (`clang`/`clang++`), `SDKROOT=$(xcrun --show-sdk-path)` |
| resource note | the `dqi` verifier must run with `--max-workers 2`; the default of 4 peak at roughly 6 GB each and will exhaust a 16 GB machine |

`kickmix` build recipe:

```bash
cd <QUALTRAN_KICKMIX_SOURCE>          # a checkout of Qualtran `main` (only `main` has pyproject.toml)
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export SDKROOT=$(xcrun --show-sdk-path)
"$VENV/bin/pip" install ./kickmix
```

> **LLVM 20 is mandatory; Apple clang 15 will not work.** Apple clang 15 rejects C++20 parenthesized
> aggregate initialisation (P0960) in `cmd/main_diagram.cc` (`RegisterId(reg_idx)`) and
> `mem/stride_span.h:82` (`stride_ptr<const const BitId>(a, b)`). Upstream depends on P0960.

### 4.2 Obtaining the competition inputs

The competition data and the Flash-THC starter-kit dataset are fetched from Kaggle's public data
endpoints. Two things bite here and are worth recording:

* **The download endpoint requires percent-encoded path separators.** A literal `/` in the `path`
  segment returns `404` with an HTML error page (this was initially misread as a rules-acceptance
  gate — `hasAcceptedRules` was `false` throughout and did not matter):

  ```bash
  # 404
  .../api/v1/competitions/data/download/<SLUG>/verifiers/invsqrt/verify_submission_invsqrt.py
  # 200  ('/' -> '%2F')
  .../api/v1/competitions/data/download/<SLUG>/verifiers%2Finvsqrt%2Fverify_submission_invsqrt.py
  ```

* **Single-stream throughput is 200–320 KB/s**, which makes the 887 MB `dqi.kmb` a ~50 minute
  download. Range-chunked parallel fetching (GCS honours `206`) reaches **3.18 MB/s — a 16× speedup**;
  887 MB in 279 s. Authenticate with an API token supplied through the environment; **never commit
  a token**, and never place one in a notebook cell.

Verify every file by byte count and (where published) by SHA-256. Our reference reproduction
confirmed `FeMoCo76o113e.fcidump` at 128,128,348 B with the SHA-256 declared in the competition
`README`.

### 4.3 Rebuilding the three artifacts

The constructors in [`src/`](src/) are the scripts that produced the submitted circuits. The
`invsqrt` and `dqi` members are rebuilt as follows (the `dfthcsos` member is the official sample,
copied through unchanged):

```bash
# ---- invsqrt: the E5 `flags` construction (Q=231, T=6852) ----
"$VENV/bin/python" src/invsqrt/invsqrt_flags_circuit.py     # -> invsqrt_flags.kmb
"$VENV/bin/python" src/invsqrt/e5_flags_check.py            # exhaustive equivalence, n_p = 2..6
"$VENV/bin/python" src/invsqrt/e5_flags_measure.py          # Q / T / C measurement

# ---- dqi: the E8 stacked CSWAP freeze ----
"$VENV/bin/python" src/dqi/e8_union_trend.py                # frozen-site union
"$VENV/bin/python" src/dqi/e8_fixpoint.py                   # fixpoint solve
"$VENV/bin/python" src/dqi/e8_build.py                      # -> dqi_stacked.kmb
"$VENV/bin/python" src/dqi/e8_verify_small.py               # small-scale exhaustive proof

# ---- assemble and validate the submission archive ----
"$VENV/bin/python" src/common/make_submission.py \
    --invsqrt invsqrt_flags.kmb --dqi dqi_stacked.kmb \
    --dfthcsos <OFFICIAL_DFTHCSOS_DIR> --out submission.zip
"$VENV/bin/python" src/common/validate_submission_format.py submission.zip
```

`make_submission.py` runs the format validator automatically after packing.

### 4.4 Verifying correctness against the official verifiers

```bash
"$VENV/bin/python" src/common/verify_all.py            # all three, with peak RSS and wall clock
"$VENV/bin/python" src/common/verify_all.py --salt N   # a specific salt
"$VENV/bin/python" src/common/verify_all.py --runs 3   # cross-validate over several salts
```

**A passing independent verifier does not mean the submission is accepted.** The official metric
performs a whole set of *file-level* checks before running any simulation, and those checks are
absent from the standalone `verifiers/` tree:

1. exactly one `invsqrt.km[xb]`, exactly one `dqi.km[xb]`, exactly one `dfthcsos/` directory —
   located by **recursive** search. Note `find_one` globs the literal pattern `invsqrt.km[xb]`, so
   renaming to `backup_invsqrt.kmb` slips through, but nesting in a subdirectory, or shipping both
   `.kmb` and `.kmx`, rejects the entire archive.
2. maximum qubit/bit id `< 699,050` (= `2**30 // 1536`) — not checked by the standalone verifiers at all.
3. `submission.npz` must be npy format 1.0, float dtype, little-endian, total ≤ 64 MiB — the standalone
   verifiers accept integer arrays and big-endian.

[`src/common/validate_submission_format.py`](src/common/validate_submission_format.py) implements
these checks using the metric's own `find_one` / `read_circuit` / `read_npz` helpers, and
[`src/common/test_format_validator.py`](src/common/test_format_validator.py) is its negative test
suite (nine rejection cases) so that the validator is demonstrably not vacuous. **The validator does
not simulate anything** — it proves the archive reaches the verifier, nothing about correctness.

### 4.5 Reading order for the methodology

```
notes/E1_REPRODUCE.md            reproduction of the reference solution, exact cost table
notes/E2_OPTIMIZATION_FRONTIER.md the three cost laws, and which levers are open or closed
notes/E3_INVSQRT.md              invsqrt opt3: first scored improvement
notes/E4_DQI_QUBIT.md            invsqrt qubit trade + the dqi qubit-count analysis
notes/E4_SUBMISSION.md           the E4 submission record
notes/E5_FLAGS.md                invsqrt flags re-encoding (the 1.26480 lever)
notes/E6_DQI_DEVIATION.md        where the dqi residual deviation actually lives
notes/E7_DQI_SAFE_FREEZE.md      first dqi register freeze
notes/E8_DQI_CSWAP_FREEZE.md     stacked CSWAP freeze, post-mortem and honest-variance analysis
notes/SUBMISSIONS.md             the submission ledger
```

---

## 5. Citation and license

Copyright 2026 the authors. Licensed under the **Apache License, Version 2.0**; see [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE).

Everything here is a description of a computational procedure applied to synthetic benchmark
circuits. No measured physical quantity is claimed anywhere in this package.

---

## 6. 中文摘要

本仓库是 Kaggle 竞赛 **「Google Quantum AI — Verifiable Quantum Circuit Compilation」** 的
**复现包**，按竞赛规则 **§6.4（开源与领奖义务，B.5 / B.7 / C.6.b）** 公开，许可证 **Apache 2.0**。

* **当前成绩：public `1.11751`，排名 `710`**（提交号 `57625941`，2026-10-25T10:07:51.693Z）。
* 总分 = 三个赛道成本比值之积的 **立方根**。最终分解：
  * `invsqrt` 比值 **1.264800755**（`Q 260→231`、`T 7,735→6,852`，做法是把 `flags` 的
    10 位一元温度计换成紧凑二元 `s`，在真正读它的两条控制路径里按需展开）；
  * `dqi` 比值 **1.103410**（官方带盐样点下；离线无盐为 1.108478，做法是两轮寄存器冻结）；
  * `dfthcsos` 比值 **1.000000**（刻意不做，方向已探明但收益不足）。
* **提交谱系**：`57596361`→1.00000（复现参考解）· `57612302`→1.04644 · `57617295`→1.04905 ·
  `57618221`→1.08145 · `57623058`→1.09489 · **`57625941`→1.11751**。
* **诚实边界**：`dqi` 的 `R` 因子含约 ±0.46 % 的「盐彩票」波动，因此实测 `1.11751` 比无盐预测
  `1.11922` 低 0.00171；对带盐模型的残差是 −0.00000（五位小数逐位吻合）。详见 §3。
* `notes/` 下的方法论报告为**中文**，`src/` 下的构造脚本为原样保留的施工代码（仅做了个人绝对路径的
  脱敏）。本仓库不含任何竞赛数据、提交二进制、凭据或第三方素材。

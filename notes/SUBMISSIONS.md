# Submission ledger

Every submission that consumed competition credit, in order. All scores are **measured public
leaderboard scores** returned by the scorer — none of the numbers in this file are predictions.

Scoring model (confirmed in both directions, see §3):

```
Score = ( prod_{b=1..3}  baseline_cost_b / cost_b ) ** (1/3)
```

| # | submission id | submitted (UTC) | content | package bytes | public score | rank at settle | credit |
|---|---|---|---|---|---|---|---|
| 1 | `57596361` | 2026-10-24T13:51:19.913Z | reference reproduction — all three benchmarks = baseline | 73,803,082 | `1.00000` | 653 / 662 | 1 |
| 2 | `57612302` | 2026-10-25T01:03:44Z | invsqrt `opt3` (E3) | — | `1.04644` | 652 / 769 | 1 |
| 3 | `57617295` | 2026-10-25T04:51:52.263Z | invsqrt `opt3` + 3-register Horner qubit trade (E4) | — | `1.04905` | 681 | 1 |
| 4 | `57618221` | 2026-10-25T05:32:54.983Z | invsqrt `flags`: binary `s` (4 bit) replaces 10-bit thermometer (E5) | — | `1.08145` | 678 | 1 |
| 5 | `57623058` | 2026-10-25T08:27:07Z | dqi multiply-register safe freeze (E7) | 72,509,253 | `1.09489` | 713 | 1 |
| 6 | `57625941` | 2026-10-25T10:07:51.693Z | dqi CSWAP-register stacked freeze (E8) | 71,668,652 | `1.11751` | 710 | 1 |

**Six credits total; none remained after #6.**

---

## 1. Submission #1 — reference reproduction

Purpose: prove that our local toolchain reproduces the reference implementation *exactly*, which
establishes the scoring model before any optimization is attempted.

| field | value |
|---|---|
| `CreateSubmission` | HTTP 200 |
| id | `57596361` |
| state | `Pending` → `COMPLETE` (≈ 39 minutes) |
| `publicScoreFormatted` | `1.00000` |
| rank | 653 / 662 |
| `contentLength` (server read-back) | `73,803,082` — byte-identical to the local archive |
| local sha256 of the archive | `52427f9823a541f2611c9a6779ce16bb54c151cc44b2c26a06fe6a3ff19bc6e6` |
| `privateEligibleSubmissionIds` | `[57596361]` — entered the private leaderboard |
| credit after | `numToday=1, numTotal=1, numAllowedNow=4` |

Corroboration that `1.00000` means "exactly the baseline" rather than a coincidence: of 662 teams,
**64 scored inside `[0.99, 1.01]`**, the median was `5.85489` and the minimum `0.97849`.

## 2. Submission #2 — invsqrt `opt3` (E3)

First scored improvement. `1.145898 ** (1/3) = 1.046442381…` → printed `1.04644`, matching all five
decimals. Neighbours on the leaderboard at settle time: rank 651 `1.05071`, rank 652 `1.04644`,
rank 653 `1.04432`.

## 3. Submissions #3–#6 — the optimization line

Each later submission re-used the *same* `dfthcsos` member as submission #1 (the official sample,
copied through byte-for-byte) and swapped only the benchmark it was targeting. This is why the
`dfthcsos` ratio stays at `1.000000` throughout and the overall score is the cube root of the
product of the other two ratios.

| submission | invsqrt ratio | dqi ratio | dfthcsos ratio | product | `product ** (1/3)` | measured |
|---|---:|---:|---:|---:|---:|---:|
| `57618221` (E5) | 1.264800755 | 1.000000 | 1.000000 | 1.264800755 | 1.08145 | **1.08145** |
| `57623058` (E7) | 1.264800755 | 1.037766 | 1.000000 | 1.312566 | 1.09490 | **1.09489** |
| `57625941` (E8) | 1.264800755 | 1.103410 | 1.000000 | 1.395593 | 1.11751 | **1.11751** |

`57623058` (E7) is the clean validation of the model: prediction `1.09490` vs measured `1.09489`,
agreeing to five decimals.

### 3.1 `57625941` — the salt residual, explained rather than hidden

The E8 prediction was `1.11922`, the measurement was `1.11751` — a gap of `−0.00171`.

Root cause, located by re-running the *same* artifact under two independent random salts: the
official scorer merges each row's `seed_salt` with the submission hash before sampling, so the
sample point is not the one we priced against offline. Under both salts, exactly **one shot in
102,400** landed on a frozen site:

| quantity | offline, unsalted | salted (two independent salts) | official, back-solved from the LB |
|---|---:|---:|---:|
| erroneous shots | 0 | **1** | 1 |
| `eps+` | `8.660587301657926e-4` | `9.104197583220583e-4` | `9.104197583220583e-4` |
| `R` (repetition factor) | `1.1912604648115832` | `1.1967314991687559` (+0.4593 %) | same |
| mean Toffoli | 5,164,839.448 | 5,164,840.586 / 5,164,841.460 | — |
| CSWAP execution count | 84,039,270,400 | **84,039,270,400 (identical)** | — |
| dqi ratio | 1.108478 | **1.103410** | 1.103423 |
| `LB = (C · ratio) ** (1/3)` | 1.11921 | **1.11751** | **1.11751 (measured)** |

Two things follow, and both are useful:

* The **`T` side is sample-independent.** The CSWAP execution count reproduced bit-for-bit across
  salts and the mean Toffoli count varied by ±2 out of 5.16 million (4e-7). Every `ΔT` claim in
  `notes/E7` and `notes/E8` stands unchanged.
* The **`R` side moves with the sample point** — a "salt lottery" worth about ±0.46 % on the dqi
  ratio. A control run of the *reference* circuit under the same two salts produced
  `value_errors=0 / phase_errors=0` and `eps+ = 8.6606e-4`, so the single erroneous shot comes from
  the frozen artifact itself and not from the verifier or sampler. Relative to the *salted* model the
  residual is `−0.00000` to five decimals.

**Reporting consequence:** offline (`salt=''`) reproduction is optimistically biased by about
`+0.46 %` on the dqi term. Anyone rebuilding this artifact should expect to land between the
`k=0` (`1.11922`) and `k=1` (`1.11751`) predictions, and should report both.

---

## 4. Reproducing the ledger

Submission headers, scores and ranks were read back with the competition's read-only API endpoints
(`competitions.SubmissionService/ListSubmissions`, `competitions.LeaderboardService/*`), and public
leaderboard snapshots were saved after each settle. The package-level digests above were taken with
`shasum -a 256` on the local archive and cross-checked against the server-side `contentLength` and,
for `57625941`, against the GCS `PUT` receipt (`35c67b4e727791b190aae0a7aa62b567ec7903aedffd6ad7ffba2c267d6394dc`).

A free upload rehearsal (`--upload-only`, steps 1–2: `StartBlobUpload` then `PUT`) was run before
every credited submission, so no credit was ever spent discovering that an archive was malformed.

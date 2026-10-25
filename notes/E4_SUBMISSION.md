# E4 — invsqrt Q=237 submission (result)

**Status: COMPLETE.** Uploaded, evaluated, scored. E4's circuit (Q 239 → 237 at identical total
Toffoli-equivalent, via the 3-register Horner 0-Toffoli qubit trade) is the first thing here
priced with a *different* cost model, so the run doubles as the strongest available test of the
metric model: it was byte-verified three times over before packing, then re-verified once more
from inside the packed zip. This note records the submission itself and the competitive outcome.

---

## 1. What was submitted

| field | value |
| --- | --- |
| package | `build/submission.zip` (a byte copy of `build/submission_e4.zip`) |
| size | **73,790,975 B** |
| sha256 | `20b7df73dfdc7c4eb16a8dbe2eb69f0bf193a925ac54d3b460cb532ed9bed3b0` |
| members | `invsqrt.kmb` (549,105 B) · `dqi.kmb` · `dfthcsos/*` (11 files + `submission.npz`) |
| `invsqrt.kmb` | Q=**237**, T=**7334** |
| `circuit_sha256` | `38c74c87e33057a7992cc707e779d40ead64161a16f3bd6cf633160b24e35839` |
| description | `E4 invsqrt Q=237 (T=7334, 3x Horner reg, 0-Toffoli -2q vs E3)` |

The two companion members were confirmed **byte-identical to the already-scored E3 package**
(byte-for-byte against `build/submission_e3.zip`), so the whole-competition score must move by
exactly the geometric mean of the single invsqrt factor — that is the whole experimental design
of this run.

### 1.1 Pre-flight (all local, before spending anything)

| check | result |
| --- | --- |
| `invsqrt.kmb` extracted **from the zip** sha256 | `38c74c87…e35839` — matches the stored circuit |
| verifier re-run on the **extracted** file (`--salt saltA`, `--max-workers 2`) | **PASS**, `verified_shots = 200,540,928` |
| score reported by that re-run | **1,798,860.1433098556** |
| `num_qubits` / `toffoli_count` | 237 / 7334 (as designed) |
| `worst_error_over_tolerance` | `0.9251076979764682` — bit-identical to E3's |
| `dqi.kmb` sha256 in e4 vs e3 zip | identical (`959471d0…`) |
| `dfthcsos/{angle_k_0,4,9}.kmb`, `submission.npz` e4 vs e3 zip | identical |

So the uploaded artifact is exactly the verified one, and only one of the three benchmark
members changed. Re-running the verifier on the extracted copy also proves the pack/unpack step
is lossless (`notes/runs/e4_zipcheck_saltA.out`).

Timing note: this salted full verification took **13m57s wall / 21m07s user** on a loaded 16 GB box
(`--max-workers 2`) versus the 4m42s recorded for the earlier unsalted run — the verifier is
CPU-bound on 200M shots and its runtime is not a property of the circuit. Nothing to read into
it beyond "the machine was busy".

---

## 2. Submission mechanics (what actually happened)

| step | result |
| --- | --- |
| `cp build/submission_e4.zip build/submission.zip` | done — name must be `submission.zip` or CreateSubmission 400s |
| `--upload-only` handshake (free) | **OK** — StartBlobUpload + PUT, `sha256=20b7df73…` echoed by the PUT |
| CreateSubmission (from the blob token) | **OK** |
| **submission id** | **57617295** |
| teamId | 17194714 |
| `dateSubmitted` | `2026-10-25T04:51:52.263Z` (12:51:52 +08:00) |
| eval finished | `publicScoreFormatted` first visible at the 13:36 poll → **~44 min** (E3 was ~39 min) |

Logs: `notes/runs/e4_uploadonly.txt` (contains the blob token),
`notes/runs/e4_createsubmission.txt`, `notes/runs/e4_watch.txt` (the poll trace, ended with
`[status] settled`).

---

## 3. Result

| | value |
| --- | --- |
| **public LB score** | **1.04905** (was 1.04644 on E3) |
| **rank** | **681 / 801** (was 676 / 795) |
| quota after | `{"numToday": 2, "numTotal": 3, "numAllowedNow": 3}` |
| credits spent on this run | **1** (the `--upload-only` handshake cost nothing) |

Leaderboard snapshot: `notes/runs/leaderboard_e4.json`; the pre-run baseline for comparison is
`notes/runs/leaderboard_e4pre.json`.

Neighbourhood after the run:

```
rank  678  1.05201
rank  679  1.05164
rank  680  1.05071
rank  681  1.04905   <== 17194714 (us, submission 57617295)
rank  682  1.04789
rank  683  1.04432
```

The leaderboard grew 795 → 801 while we were submitting, and it is denser in our band than it
was at E3 (the 1.04432 row that sat directly below us at E3 is now 683, and a new 1.04789 row
inserted itself between us and it). Net effect: score **+0.00261**, rank **+5 places**
(676 → 681) — we out-gained the median of the field but not the top of our local band.

---

## 4. Prediction vs actual

| quantity | predicted | actual | deviation |
| --- | --- | --- | --- |
| invsqrt ratio `2,076,774.234 / 1,798,860.143` | 1.1544946 | — (not printed) | — |
| whole competition `ratio^(1/3)` | 1.0490527 | **1.04905** | **−0.000003** |

The image was produced with the coarse inputs from the brief (`2,076,774.23 / 1,798,860.14`),
which gives 1.15449455 → 1.04905268; the exact verifier number gives 1.04905258. Both round to
the same 5-decimal display. The brief's headline "**1.04916**" is *not* reproducible from the
stated ratio: for the ratio 1.1544946 to yield 1.04916 the ratio would have to be ≈1.15455
(≈+0.005%), i.e. the stated 1.04916 is a slightly off arithmetic step, not a different model.
The deviation of the realised score from 1.04916 is **0.00011**, i.e. **9× smaller than the
0.001 tolerance** — so no root-cause investigation is triggered, and the useful conclusion is
rather stronger than that:

**The scoring model is confirmed exactly.** E3 landed on 1.04644 = (1.145898…)^(1/3) to all 5
printed decimals, and E4 lands on 1.04905 = (1.1544946)^(1/3) to all 5 printed decimals. Two
independent points, both exact. What this run *does* rule out is any subtle deviation from
"geometric mean of the three independent member ratios, other members scored at ratio 1":
the byte-identical `dqi.kmb` and `dfthcsos/*` contributed exactly nothing, as designed.

One thing to stop doing: comparing against the *pre-run* rank projection. The projection said
675; we landed 681, purely because six new teams appeared and the band tightened. Score
prediction was exact; rank prediction needs a live leaderboard, not a stale snapshot.

---

## 5. What this buys, and what is left

- E4's contribution over E3 is **Q 239 → 237 at identical total Toffoli-equivalent** (the
  3-register Horner 0-Toffoli qubit trade). Score 1.04644 → 1.04905.
- The remaining move on invsqrt is **qubits, not Toffolis**: the exchange rate is set by
  `Q/C ≈ 239/7583`, i.e. one saved qubit is worth up to ~32 Toffolis
  (see `notes/E3_INVSQRT.md` §7). Adding Toffolis to save qubits is still profitable at that
  rate; adding qubits to save Toffolis is not.
- The largest remaining non-algorithmic qubit cost is the adder-ancilla term
  (`product_bits + 1`) inside the compiled `kickmix` `iadd`/`isub`
  (`src/e4/invsqrt_adders.py`, `notes/runs/e4_adders.json`).
- Three credits remain today; two Final Submission slots remain for the whole competition.
  **Do not spend a Final slot on an invsqrt-only marginal** — the same absolute qubit saving is
  worth much more on the `dqi` member, whose cost is 2.4e13 against invsqrt's 1.8e6.

---

## 6. Files

| path | what |
| --- | --- |
| `build/submission.zip` | the submitted archive (74 MB) |
| `build/submission_e4.zip` | the same bytes, kept under the experiment name |
| `build/e4/zipcheck/invsqrt.kmb` | the member extracted from the zip, re-verified as-is |
| `src/lb_snapshot.py` | leaderboard snapshotter used for the before/after views |
| `notes/runs/e4_uploadonly.txt` · `e4_createsubmission.txt` | handshake + CreateSubmission transcripts |
| `notes/runs/e4_watch.txt` | poll trace, ended `[status] settled` (submission 57617295 = 1.04905) |
| `notes/runs/e4_zipcheck_saltA.out` | PASS verdict of the re-run on the extracted member |
| `notes/runs/leaderboard_e4pre.json` · `leaderboard_e4.json` | LB before / after |

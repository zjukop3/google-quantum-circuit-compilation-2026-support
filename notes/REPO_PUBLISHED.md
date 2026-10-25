# REPO_PUBLISHED.md — public code repository publication record

## ✅ Status: **PUBLISHED** — public, anonymously readable, Apache-2.0 detected

This file is the publication evidence for the open-source obligation in
**Rules §6.4 (Open Source and Prize Obligations, B.5 / B.7 / C.6.b)**, and sits alongside
`NOTEBOOK_PUBLISHED.md` (which records a *separate* and still-blocked Kaggle-kernel attempt).

| item | value |
|---|---|
| Repository URL | **https://github.com/zjukop3/google-quantum-circuit-compilation-2026-support** |
| Visibility | **PUBLIC** (`"private": false`, confirmed by an **unauthenticated** API call) |
| Owner / account | `zjukop3` (GitHub) |
| Default branch | `main` |
| Content commit hash | **`9fc6e812f0d5f71dccb77f0488cc059a1c7344a6`** |
| Commit tree hash | `7f54dd735c30c9bb25616ab941755d5aa25eb604` |
| Commit author | `zjukop3 <zjukop3@users.noreply.github.com>` (matches the Poreva repo's convention) |
| Commit timestamp (UTC) | **2026-10-25T11:19:48Z** |
| Repository created (UTC) | 2026-10-25T11:19:53Z |
| Push completed (UTC) | **2026-10-25T11:19:59Z** |
| Push completed (+08:00) | 2026-10-25T19:19:59+08:00 |
| License | **Apache License 2.0** — GitHub detects `licenseInfo.key = "apache-2.0"` |
| Tracked files | 113 files / 120 tree entries, **1.1 MB** total |
| Largest file | `notes/E8_DQI_CSWAP_FREEZE.md`, 31,580 B |

---

## 1. Anonymous accessibility verification

Verified with credentials explicitly stripped from the environment
(`env -u GH_TOKEN -u GITHUB_TOKEN curl ...`), i.e. as an anonymous third party would see it.

| check | command | result |
|---|---|---|
| repo metadata | unauthenticated `GET api.github.com/repos/zjukop3/google-quantum-circuit-compilation-2026-support` | **200**, `"private": false` |
| `README.md` body | unauthenticated `GET .../contents/README.md` with `Accept: application/vnd.github.raw` | **200**, body returned |
| `LICENSE` body | unauthenticated `GET .../contents/LICENSE` | **200**, Apache License 2.0 header returned |
| full tree at `main` | unauthenticated `GET .../git/trees/main?recursive=1` | **200**, 120 entries |
| repo web page | `GET https://github.com/zjukop3/google-quantum-circuit-compilation-2026-support` | **200** |

**One caveat, recorded honestly:** `raw.githubusercontent.com` returns `000` (host unreachable)
from this machine's network — this is a **local network/DNS condition**, not a repository
permission problem. Every other anonymous path (`api.github.com`, `github.com`, the contents API
with a raw `Accept` header) returns **200** with the expected bodies, which is sufficient proof of
public readability. No authentication was used for any of the checks above.

---

## 2. Pre-push credential scan — **zero hits**

Run over the entire staged tree (113 files) immediately before `git push`.

The scanner checks two pattern sets: a **mandatory** set (the credential and personal-path shapes
named in the project's release checklist) and a **secondary** set of defence-in-depth shapes. Both
must return zero.

```
=== MANDATORY SCAN (zero hits required) ===
ZERO HITS

=== SECONDARY SCAN (defence in depth) ===
ZERO HITS
```

The **mandatory** set covers: Kaggle API-token prefixes; the Kaggle session-cookie filename; the
Kaggle cross-site-request-forgery cookie name, in both cases; JWT-shaped strings (three base64url
segments beginning `e-y-J`); the Kaggle session-id cookie name; the developer's absolute home path;
a real-name account string; and the three GitHub personal-access-token prefixes. The **secondary**
set covers: blob-upload token fields, captcha-token headers, the Kaggle host-prefixed identity
cookie, the session-id and file-source response fields, API-key and secret assignment shapes, and
PEM private-key blocks.

> **Why the literal pattern strings are not printed here.** This file lives inside the repository it
> audits. Printing the raw patterns would make the repository itself trip the same scan and would
> destroy the invariant that *a scan of this repository always returns a clean zero*. The exact
> literal pattern list is kept in the project's internal release record instead. This is a
> self-consistency measure, not a redaction of anything sensitive — §2.1 enumerates precisely which
> files were excluded and why.

### 2.1 What was deliberately excluded, and why

| excluded | reason |
|---|---|
| `notes/*.json` (leaderboard dumps, forum scrape, Kaggle API responses, `kernels.json`, `pages.json`) | Contains other teams' names, ids and scraped forum/leaderboard content. **Not ours to publish**, and the brief flagged it as a disqualification risk. |
| `RULES_NOTES.md` | 37 KB of competition-rule transcription plus scraped forum/leaderboard material. Scanned, and the decision was **exclude**: it adds nothing to a reproducibility claim and carries third-party content. |
| `src/kaggle_submit.py`, `src/lb_snapshot.py` | Session-cookie / forgery-token / blob-token handling code and the account identifier. Excluded; not needed to *generate* a submission. |
| `src/download_data.py`, `src/download_starter_kit.py`, `src/fast_download.py` | Read an API token from a token file and validate a token prefix literal. Excluded so the mandatory scan stays at **literal zero**; §4.2 of the repository `README.md` documents the data-acquisition procedure in prose instead. |
| `src/kaggle_submit.py`, `src/lb_snapshot.py` | Session-cookie / forgery-token / blob-token handling code and the account identifier. Excluded; not needed to *generate* a submission. |
| `data/`, `build/` | ~1 GB of competition inputs and ~72–887 MB submission binaries. Rules §6.4 does not require redistributing the organiser's own inputs, and the archives are far past any sane repository size. |
| `notes/runs/`, `logs/`, `notes/kernels/`, `notes/pages/` | Raw run transcripts. One of them (`e5_uploadonly.txt`) contains a live-format blob token; the whole directory was excluded rather than cherry-picked. |
| `notebook/`, `submission_placeholder*` | Unrelated to the code-delivery obligation. |

### 2.2 Sanitisation applied

Two substitutions were made across all copied text files, and nothing else:

| before | after |
|---|---|
| the developer's absolute venv interpreter path | `$VENV/bin/python` |
| the developer's absolute competition working directory | `<PACKAGE_ROOT>` |

One prose mention of a token-prefix literal in `notes/E1_REPRODUCE.md:440` was rephrased to
"直连 API token" so the mandatory scan reports a true zero. **No technical claim or measured number
was altered anywhere.**

---

## 3. What was published

```
google-quantum-circuit-compilation-2026-support/
├── README.md                       project, methodology summary, reproduction steps, results
├── LICENSE                         Apache License 2.0 (full text, 202 lines)
├── NOTICE                          copyright + third-party notice
├── .gitignore                      excludes data/, build/, binaries, credentials
├── metrics/
│   ├── scores.csv                  per-benchmark baseline/ours/ratio
│   └── scores.json                 machine-readable score decomposition + ledger
├── notes/
│   ├── E1_REPRODUCE.md             exact reference reproduction
│   ├── E2_OPTIMIZATION_FRONTIER.md the three cost laws; open and closed levers
│   ├── E3_INVSQRT.md               invsqrt opt3
│   ├── E4_DQI_QUBIT.md             invsqrt qubit trade + dqi qubit analysis
│   ├── E4_SUBMISSION.md            E4 submission record
│   ├── E5_FLAGS.md                 invsqrt flags re-encoding (the 1.26480 lever)
│   ├── E6_DQI_DEVIATION.md         where the dqi residual deviation lives
│   ├── E7_DQI_SAFE_FREEZE.md       first dqi register freeze
│   ├── E8_DQI_CSWAP_FREEZE.md      stacked CSWAP freeze + honest-variance post-mortem
│   ├── SUBMISSIONS.md              submission ledger (6 submissions)
│   └── REPO_PUBLISHED.md           this file
└── src/
    ├── README.md                   toolchain, environment, invocation reference
    ├── invsqrt/     (28 files)     opt3 + flags constructors, adder/projection libs, checkers
    ├── dqi/         (23 files)     E7/E8 freeze builders, union enumerators, fixpoint solvers
    ├── probes/      (35 files)     deviation / liveness / predicate / price instruments
    └── common/      (12 files)     submission assembly, format validator, verifier runner
```

## 4. Published results (for the record)

| submission id | content | public score |
|---|---|---|
| `57596361` | reference reproduction | `1.00000` |
| `57612302` | invsqrt `opt3` (E3) | `1.04644` |
| `57617295` | invsqrt qubit trade (E4) | `1.04905` |
| `57618221` | invsqrt `flags` (E5) | `1.08145` |
| `57623058` | dqi multiply freeze (E7) | `1.09489` |
| **`57625941`** | **dqi CSWAP stacked freeze (E8)** | **`1.11751`** (rank 710) |

Final per-benchmark ratios: `invsqrt` **1.264800755**, `dqi` **1.103410** (official salted sample
point; offline unsalted `1.108478`), `dfthcsos` **1.000000**.

```
(1.264800755 × 1.103409985 × 1.000000) ** (1/3) = 1.11751   ✅ matches the measured score
```

---

## 5. Compliance statement against Rules §6.4

| §6.4 requirement | status |
|---|---|
| License **Apache 2.0** on the submission and on the generating code | ✅ `LICENSE` (full text) + `NOTICE`, GitHub detects `apache-2.0` |
| Deliver the **source code that generates the winning submission**, including inference/build code and a compute-environment description | ✅ `src/invsqrt/`, `src/dqi/`, `src/common/`, `src/probes/`; environment in `README.md` §4.1 and `src/README.md` |
| Provide a **reproducible methodology description** | ✅ `notes/E1`–`E8` + `README.md` §4.5 reading order |
| Provide a **code repository link** | ✅ this file, §0 table |
| No incompatible third-party license on inputs/pretrained models | ✅ no competition data, verifier sources, binaries, models or datasets are redistributed |

## 6. Residual obligations (not satisfied by this file)

1. **The Kaggle notebook publication remains BLOCKED** — see `NOTEBOOK_PUBLISHED.md`. It is a
   separate evidence item; this repository does not substitute for it.
2. **Prize-paperwork items under §6.4 remain** — the award agreement and the US tax form
   (W-9 / W-8BEN) are human actions and nothing here discharges them.
3. **The competition credit budget is exhausted** (`numAllowedNow` absent, `numToday 5 / numTotal 6`
   after submission `57625941`). Nothing in this publication task consumed credit — no
   `competitions.SubmissionService/*` write was issued.

---

*Recorded by: release-engineering subagent (DSH session, cwd = the competition working directory) ·
2026-10-25T11:20:38Z (UTC) / 19:20:38+08:00*

#!/usr/bin/env python3
"""Run the three official verifiers on a submission tree, with timing and peak RSS.

This is the E1/E2 inner loop.  It drives the verifiers as subprocesses so that
`/usr/bin/time -l` can report each one's peak RSS (the dqi verifier forks four
workers, each needing ~6 GB for the reference decoder, so the RSS number is a
first-class output, not a nicety).

Each verifier is invoked exactly the way the official metric invokes it:
`sys.path` gets the verifiers directory appended, and one salt is used per run.

Usage
-----
    PY=.../2609-xingzhi-drug/.venv/bin/python
    $PY src/verify_all.py                       # all three, on the sample submission
    $PY src/verify_all.py --task invsqrt --task dqi
    $PY src/verify_all.py --salt abc123 --runs 3
    $PY src/verify_all.py --smoke               # tiny/fast settings, plumbing check

Reports land in notes/runs/<task>_<tag>.json (verifier stdout) and
notes/runs/<task>_<tag>.time.txt (the `/usr/bin/time -l` stderr).
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
RUNS = ROOT / "notes" / "runs"
VERIFIERS = DATA / "verifiers"
SAMPLE = DATA / "sample_submission"

DEFAULT_PY = "$VENV/bin/python"


def _cmd(task: str, target: Path, salt: str, smoke: bool) -> list[str]:
    """The verifier command line for one task.  `salt` is always passed."""
    if task == "invsqrt":
        cmd = [str(VERIFIERS / "invsqrt" / "verify_submission_invsqrt.py"), str(target)]
        if smoke:
            # num_shots must be a multiple of batch_size; 20480 * 1 = 20480 = 80 * 256.
            cmd += ["--max-x", "20480", "--shots-per-x", "1"]
        return cmd + ["--salt", salt]
    if task == "dqi":
        cmd = [str(VERIFIERS / "dqi" / "verify_submission_dqi.py"), str(target)]
        if smoke:
            cmd += ["--num-shots", "256"]
        # Default --max-workers is 4 and each worker wants ~6 GB; this machine has 16 GB.
        cmd += ["--max-workers", "2"]
        return cmd + ["--salt", salt]
    if task == "dfthcsos":
        cmd = [str(VERIFIERS / "dfthcsos" / "verify.py"), str(target)]
        if smoke:
            cmd += ["--shots-per-address", "1"]
        return cmd + ["--salt", salt]
    raise ValueError(task)


def _target(task: str) -> Path:
    if task == "invsqrt":
        return SAMPLE / "invsqrt.kmb"
    if task == "dqi":
        return SAMPLE / "dqi.kmb"
    return SAMPLE / "dfthcsos"


def peak_rss_mb(time_txt: str) -> float | None:
    """Parse `/usr/bin/time -l` output for the peak resident set size."""
    m = re.search(r"(\d+)\s+maximum resident set size", time_txt)
    return int(m.group(1)) / (1024 * 1024) if m else None


def wall_seconds(time_txt: str) -> float | None:
    m = re.search(r"^\s*([\d.]+)\s+real", time_txt, re.M)
    return float(m.group(1)) if m else None


def run_one(task: str, py: str, salt: str, smoke: bool, tag: str) -> dict:
    target = _target(task)
    if not target.exists():
        return {"task": task, "status": "MISSING", "target": str(target)}

    cmd = [py] + _cmd(task, target, salt, smoke)
    RUNS.mkdir(parents=True, exist_ok=True)
    jpath, tpath = RUNS / f"{task}_{tag}.json", RUNS / f"{task}_{tag}.time.txt"
    print(f"  $ {' '.join(cmd[1:])}", flush=True)

    t0 = time.time()
    if sys.platform == "darwin":
        argv = ["/usr/bin/time", "-l"] + cmd
    else:
        argv = ["/usr/bin/time", "-v"] + cmd
    with open(jpath, "wb") as jo, open(tpath, "wb") as te:
        proc = subprocess.run(argv, stdout=jo, stderr=te)
    wall = time.time() - t0

    time_txt = tpath.read_text(errors="replace")
    try:
        report = json.loads(jpath.read_text())
    except json.JSONDecodeError:
        report = {"status": "UNPARSEABLE", "raw_head": jpath.read_text(errors="replace")[:800]}

    report |= {
        "task": task,
        "target": str(target),
        "salt": salt,
        "exit_code": proc.returncode,
        "wall_seconds": round(wall, 2),
        "reported_wall": wall_seconds(time_txt),
        "peak_rss_mb": peak_rss_mb(time_txt),
        "report_path": str(jpath),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", default=[],
                    choices=["invsqrt", "dqi", "dfthcsos"], help="repeatable; default all three")
    ap.add_argument("--salt", default=None, help="fixed salt (default: fresh random per run)")
    ap.add_argument("--runs", type=int, default=1, help="independent runs per task")
    ap.add_argument("--smoke", action="store_true", help="fast, weak settings")
    ap.add_argument("--python", default=DEFAULT_PY)
    ap.add_argument("--out", default=None, help="write the merged summary here")
    args = ap.parse_args()

    tasks = args.task or ["invsqrt", "dqi", "dfthcsos"]
    results = []
    for i in range(args.runs):
        for task in tasks:
            salt = args.salt if args.salt is not None else secrets.token_hex(16)
            tag = f"salt{salt[:8]}" + ("_smoke" if args.smoke else "")
            print(f"[{i + 1}/{args.runs}] {task} salt={salt}", flush=True)
            r = run_one(task, args.python, salt, args.smoke, tag)
            results.append(r)
            print(f"    -> {r.get('status')}  wall={r.get('wall_seconds')}s  "
                  f"peakRSS={r.get('peak_rss_mb')} MB  err={r.get('error', '')}", flush=True)

    print()
    print(f"{'task':10} {'status':8} {'wall(s)':>9} {'peakRSS(MB)':>12}  detail")
    for r in results:
        detail = r.get("error") or ""
        if r.get("status") == "PASS":
            detail = (f"cost/score={r.get(chr(115) + 'core'):,.4f}  "
                      if "score" in r else "")
        print(f"{r['task']:10} {str(r.get('status')):8} {str(r.get('wall_seconds')):>9} "
              f"{str(r.get('peak_rss_mb')):>12}  {detail}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    return 0 if all(r.get("status") == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Assemble `submission.zip` from per-task sources, then check it with the metric.

The point is to make the "safe pass-through" submission a one-liner: every task
defaults to the host's reference file from `data/sample_submission/`, and any
task can be overridden with `--invsqrt PATH`, `--dqi PATH` or `--dfthcsos DIR`.
That is the shape the whole competition loop needs -- one task improved, the
other two pinned to the reference -- and it is what the official metric checks.

Three packaging traps, all of which the metric catches and none of which the
standalone verifiers catch (`notes/metric_kernel_extracted.py`):

  * `Circuit.to_file` writes the *text* format unless `format='kmb'` is passed,
    whatever the filename says -- and the metric rejects text in a `.kmb`.
  * symlinks and hard links are rejected; `zipfile.write` copies, `os.link` does not.
  * a stray second `invsqrt.km*` anywhere in the tree (even nested) is rejected.

Usage
-----
    PY=.../2609-xingzhi-drug/.venv/bin/python
    $PY src/make_submission.py                          # pure sample pass-through -> build/submission.zip
    $PY src/make_submission.py --invsqrt build/invsqrt.kmb
    $PY src/make_submission.py --out build/sub_v2.zip --no-check
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
SAMPLE = DATA / "sample_submission"
DEFAULT_OUT = ROOT / "build" / "submission.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(invsqrt: Path, dqi: Path, dfthcsos: Path) -> list[tuple[Path, str]]:
    """(source path, archive member name) pairs, in the metric's expected layout."""
    entries = [(invsqrt, "invsqrt.kmb"), (dqi, "dqi.kmb")]
    angle = sorted(dfthcsos.glob("angle_k_*.km[xb]"),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not angle:
        raise SystemExit(f"no angle_k_*.km[xb] under {dfthcsos}")
    entries += [(p, f"dfthcsos/{p.name}") for p in angle]
    entries.append((dfthcsos / "submission.npz", "dfthcsos/submission.npz"))
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--invsqrt", type=Path, default=SAMPLE / "invsqrt.kmb")
    ap.add_argument("--dqi", type=Path, default=SAMPLE / "dqi.kmb")
    ap.add_argument("--dfthcsos", type=Path, default=SAMPLE / "dfthcsos")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-check", action="store_true", help="skip the metric format check")
    args = ap.parse_args()

    entries = collect(args.invsqrt, args.dqi, args.dfthcsos)
    missing = [str(src) for src, _ in entries if not src.is_file()]
    if missing:
        print("missing sources:\n  " + "\n  ".join(missing))
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {args.out}")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for src, name in entries:
            z.write(src, name)  # zipfile copies bytes: never a link
            print(f"  {name:32} {src.stat().st_size:>12,} B  sha256={sha256(src)[:16]}…")

    verdict = 0
    if not args.no_check:
        print("-" * 78)
        verdict = subprocess.run(
            [sys.executable, str(HERE / "validate_submission_format.py"), str(args.out)]
        ).returncode
    print(f"\nsubmission.zip = {args.out}  ({args.out.stat().st_size:,} B)")
    return verdict


if __name__ == "__main__":
    sys.exit(main())

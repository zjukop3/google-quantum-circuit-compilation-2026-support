#!/usr/bin/env python3
"""Negative tests for validate_submission_format.py.

A validator that says ACCEPT on everything is worthless. These cases each break
one rule that the *official metric* enforces but the standalone verifiers do not,
and assert that the metric's own checks reject them.

    python src/test_format_validator.py
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import zipfile

import numpy as np

from validate_submission_format import load_metric, validate
from make_placeholder_submission import ORBITALS, build

HERE = pathlib.Path(__file__).resolve().parent.parent
METRIC_PY = HERE / "notes" / "metric_kernel_extracted.py"


def zip_dir(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(src))
    return dst


def case_clean(root: pathlib.Path) -> pathlib.Path:
    """Baseline: should be accepted."""
    return root


def case_angle_gap(root: pathlib.Path) -> pathlib.Path:
    """Renumber angle_k_1 -> angle_k_2: gap in the numbering. Metric rejects."""
    (root / "dfthcsos" / "angle_k_1.kmb").rename(root / "dfthcsos" / "angle_k_2.kmb")
    return root


def case_two_invsqrt(root: pathlib.Path) -> pathlib.Path:
    """A copy of invsqrt.kmb inside a subfolder.

    Note the glob is the literal `invsqrt.km[xb]`, so a *renamed* backup
    (backup_invsqrt.kmb) is NOT found -- but rglob searches recursively, so a
    copy under any subdirectory IS, and gets the submission rejected.
    """
    (root / "backup").mkdir()
    shutil.copy(root / "invsqrt.kmb", root / "backup" / "invsqrt.kmb")
    return root


def case_both_formats(root: pathlib.Path) -> pathlib.Path:
    """Same circuit as both .kmb and .kmx, in the same folder. Two matches -> reject."""
    import kickmix as km

    c = km.Circuit.from_file(root / "invsqrt.kmb", format="kmb")
    c.to_file(root / "invsqrt.kmx", format="kmx")
    return root


def case_two_dfthcsos_dirs(root: pathlib.Path) -> pathlib.Path:
    """A second directory named dfthcsos/ one level down."""
    (root / "old").mkdir()
    shutil.copytree(root / "dfthcsos", root / "old" / "dfthcsos")
    return root


def case_bigendian_npz(root: pathlib.Path) -> pathlib.Path:
    """tree_edges in big-endian: metric rejects, dfthcsos verifier accepts."""
    p = root / "dfthcsos" / "submission.npz"
    arrays = dict(np.load(p))
    arrays["tree_edges"] = arrays["tree_edges"].astype(">i8")
    np.savez(p, **arrays)
    return root


def case_unnamed_register(root: pathlib.Path) -> pathlib.Path:
    """A circuit with a register but no REGISTER line -> no name. Metric rejects."""
    import kickmix as km

    text = "APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER q1 r0\nCX q0 q1\nCX q0 q1"
    km.Circuit(text).to_file(root / "dfthcsos" / "angle_k_0.kmb", format="kmb")
    return root


def case_npz_wrong_dtype(root: pathlib.Path) -> pathlib.Path:
    """U as integers: metric requires a float kind for U."""
    p = root / "dfthcsos" / "submission.npz"
    arrays = dict(np.load(p))
    arrays["U"] = arrays["U"].astype(np.int64)
    np.savez(p, **arrays)
    return root


def case_npz_bad_shape(root: pathlib.Path) -> pathlib.Path:
    """H2_I shape disagrees with the named 'R' dimension from U."""
    p = root / "dfthcsos" / "submission.npz"
    arrays = dict(np.load(p))
    arrays["H2_I"] = np.zeros((arrays["H2_I"].shape[0] + 1, arrays["H2_I"].shape[1]))
    np.savez(p, **arrays)
    return root


def case_unsupported_instruction(root: pathlib.Path) -> pathlib.Path:
    """An instruction outside the metric's whitelist."""
    import kickmix as km

    p = root / "dfthcsos" / "angle_k_0.kmb"
    c = km.Circuit.from_file(p, format="kmb")
    km.Circuit(str(c) + "\nDEBUG_PRINT").to_file(p, format="kmb")
    return root


CASES = [
    ("clean baseline", case_clean, "ACCEPT"),
    ("angle numbering gap", case_angle_gap, "REJECT"),
    ("invsqrt copy in subfolder", case_two_invsqrt, "REJECT"),
    ("invsqrt in both formats", case_both_formats, "REJECT"),
    ("second dfthcsos/ directory", case_two_dfthcsos_dirs, "REJECT"),
    ("big-endian tree_edges", case_bigendian_npz, "REJECT"),
    ("unnamed register", case_unnamed_register, "REJECT"),
    ("npz U not float", case_npz_wrong_dtype, "REJECT"),
    ("npz H2_I shape mismatch", case_npz_bad_shape, "REJECT"),
]


def main() -> int:
    metric = load_metric(METRIC_PY)
    build()  # ensure the pristine placeholder exists
    pristine = HERE / "submission_placeholder"

    failures = 0
    print(f"{'case':28s} {'expected':9s} {'got':9s} ok")
    print("-" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        for name, mutate, expected in CASES:
            work = pathlib.Path(tmp) / name.replace(" ", "_")
            shutil.copytree(pristine, work)
            mutate(work)
            z = zip_dir(work, pathlib.Path(tmp) / f"{work.name}.zip")
            verdict, _log = validate(metric, z)
            got = verdict.split(":")[0]
            ok = got == expected
            failures += not ok
            print(f"{name:28s} {expected:9s} {got:9s} {'OK' if ok else 'FAIL <<<'}")
            if not ok:
                print(f"    {verdict}")

    print("-" * 70)
    print("all negative tests passed" if not failures else f"{failures} unexpected result(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

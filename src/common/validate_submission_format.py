#!/usr/bin/env python3
"""Validate a submission archive with the *official metric's own* file checks.

This runs the real `find_one` / `read_circuit` / `read_npz` from
https://www.kaggle.com/code/metric/google-quantum-ai-metric (saved as
notes/metric_kernel_extracted.py) plus the pre-verification section of
`generate_standard_submission`.

It does NOT simulate anything, so it cannot tell you whether a circuit is
correct -- only whether the archive would be accepted far enough to reach the
verifiers. That is precisely the class of failure the standalone verifiers do
not catch, and it costs no submission quota.

Usage:
    python src/validate_submission_format.py [path/to/submission.zip | unpacked/dir]

Exit code 0 = ACCEPT, 1 = REJECT, 2 = could not run the check.
"""

from __future__ import annotations

import importlib.util
import pathlib
import stat
import sys
import tempfile
import zipfile

HERE = pathlib.Path(__file__).resolve().parent.parent
METRIC_PY = HERE / "notes" / "metric_kernel_extracted.py"

# From the metric source.
MAX_NPZ_BYTES = 2**26
# From the verifiers (re-stated here because verifiers/ needs the gated data;
# values taken from the metric source and the public pitfalls notebook).
ORBITALS = 76
DQI_N = 70
DQI_M = 4095
ANGLE_QUBIT_CAP = 128

NPZ_SPEC = {
    "U": ("f", ("R", "B", ORBITALS)),
    "W": ("f", ("R", "B", "C")),
    "H2_I": ("f", ("R", "C")),
    "H1_I": ("f", ()),
    "Hsym_lower": ("f", (ORBITALS * (ORBITALS + 1) // 2,)),
    "tree_edges": ("iu", (ORBITALS - 1, 2)),
}
MAX_REGISTERS = {"invsqrt": 1, "dqi": DQI_N + DQI_M, "angle": ANGLE_QUBIT_CAP}


def load_metric(path: pathlib.Path):
    if not path.is_file():
        sys.exit(
            f"could not run the check: {path} not found.\n"
            "Fetch it with:\n"
            "  curl -H 'Authorization: Bearer <TOKEN>' \\\n"
            "    'https://www.kaggle.com/api/v1/kernels/pull/metric/google-quantum-ai-metric' \\\n"
            "    | python -c \"import json,sys;print(json.load(sys.stdin)['blob']['source'])\" > notes/metric_raw.json\n"
            "then extract the notebook's single cell (see notes/metric_kernel_extracted.py).\n"
            "Exit code 2."
        )
    spec = importlib.util.spec_from_file_location("official_metric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(metric, target: pathlib.Path) -> tuple[str, list[str]]:
    """Return (verdict, log). Mirrors generate_standard_submission's pre-verifier section."""
    log: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        if target.is_dir():
            root = target
        else:
            with zipfile.ZipFile(target) as archive:
                archive.extractall(tmp)
            root = pathlib.Path(tmp)

        def read(file: pathlib.Path, reader, spec):
            # The metric's read(): regular file, single hard link, inside the unpack root.
            status = file.lstat()
            if not stat.S_ISREG(status.st_mode) or status.st_nlink > 1 or not file.resolve().is_relative_to(root.resolve()):
                raise metric.ParticipantVisibleError(f"{file.relative_to(root)} must be a regular file, not a link or a special file.")
            return reader(file, spec)

        try:
            p = metric.find_one(root, "invsqrt.km[xb]", "invsqrt.kmx or invsqrt.kmb")
            c = read(p, metric.read_circuit, MAX_REGISTERS["invsqrt"])
            log.append(
                f"invsqrt      OK  {p.relative_to(root)}  qubits={c.num_qubits} bits={c.num_bits} "
                f"registers={c.num_registers} ops={len(c)}"
            )

            p = metric.find_one(root, "dqi.km[xb]", "dqi.kmx or dqi.kmb")
            c = read(p, metric.read_circuit, MAX_REGISTERS["dqi"])
            log.append(
                f"dqi          OK  {p.relative_to(root)}  qubits={c.num_qubits} bits={c.num_bits} "
                f"registers={c.num_registers} ops={len(c)}"
            )

            directory = metric.find_one(root, "dfthcsos", "dfthcsos directory", is_dir=True)
            angle_files = list(directory.glob("angle_k_*.km[xb]"))
            angle_paths = {f.stem: f for f in angle_files}
            stems = [f"angle_k_{k}" for k in range(len(angle_files))]
            if not stems or sorted(angle_paths) != sorted(stems):
                raise metric.ParticipantVisibleError(
                    "dfthcsos/ must hold the circuits angle_k_0 ... angle_k_<n-1>, each once, as .kmx or .kmb."
                )
            log.append(f"dfthcsos/    OK  {len(stems)} ANGLE circuits: {', '.join(stems)}")
            for stem in stems:
                c = read(angle_paths[stem], metric.read_circuit, MAX_REGISTERS["angle"])
                log.append(f"  {stem:10s} OK  qubits={c.num_qubits} registers={c.num_registers} ops={len(c)}")

            npz = read(
                metric.find_one(directory, "submission.npz", "dfthcsos/submission.npz"),
                metric.read_npz,
                NPZ_SPEC,
            )
            for key, arr in npz.items():
                total = arr.size * arr.dtype.itemsize
                log.append(f"  npz[{key:12s}] OK  dtype={arr.dtype} shape={arr.shape} {total:,} B")
            log.append(f"  npz total    OK  {sum(a.size * a.dtype.itemsize for a in npz.values()):,} B (limit {MAX_NPZ_BYTES:,})")
        except Exception as err:  # noqa: BLE001 - report whatever the metric raises
            return f"REJECT: {type(err).__name__}: {err}", log

    return "ACCEPT", log


def main() -> int:
    if len(sys.argv) > 1:
        target = pathlib.Path(sys.argv[1]).resolve()
    else:
        target = HERE / "submission_placeholder.zip"
    if not target.exists():
        print(f"target does not exist: {target}")
        return 2

    metric = load_metric(METRIC_PY)
    print(f"checking {target}")
    print(f"metric source: {METRIC_PY}")
    print("-" * 78)
    verdict, log = validate(metric, target)
    for line in log:
        print(line)
    print("-" * 78)
    print(verdict)
    print()
    print("NOTE: this only proves the archive layout/dtypes are acceptable.")
    print("      It does NOT simulate the circuits. Run the verifiers (which need the")
    print("      competition data) to check correctness and get a cost.")
    return 0 if verdict == "ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())

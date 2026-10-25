#!/usr/bin/env python3
"""make_stacked_zip.py — build build/e8/submission.zip from build/submission.zip
by swapping ONLY dqi.kmb for build/e8/dqi_stacked.kmb.

Everything else (invsqrt.kmb, dfthcsos/*) is copied byte-for-byte at the entry level,
so the frozen invsqrt (Q=231, sha 0687a7b2...) and the 11 dfthcsos files are untouched.

The payload basename MUST be submission.zip (Kaggle 400 FAILED_PRECONDITION otherwise).
"""
from __future__ import annotations

import hashlib
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.dirname(os.path.dirname(HERE))
SRC_ZIP = os.path.join(COMP, "build", "submission.zip")
NEW_DQI = os.path.join(COMP, "build", "e8", "dqi_stacked.kmb")
OUT_ZIP = os.path.join(COMP, "build", "e8", "submission.zip")

EXPECT_DQI_SHA = "639f9713d63257bb392f8e48f201ae73fbf0b382049e0aabb10ef056143d4c14"


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    got = sha256_of(NEW_DQI)
    print("dqi_stacked.kmb sha256 = %s" % got)
    if got != EXPECT_DQI_SHA:
        print("FATAL: dqi_stacked.kmb sha mismatch (expected %s)" % EXPECT_DQI_SHA)
        return 1

    with zipfile.ZipFile(SRC_ZIP) as zin:
        infos = zin.infolist()
        print("source entries: %s" % ", ".join(i.filename for i in infos))
        with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
            for info in infos:
                if info.filename == "dqi.kmb":
                    zout.write(NEW_DQI, "dqi.kmb",
                               compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
                    print("  swapped  dqi.kmb  <- %s" % NEW_DQI)
                    continue
                with zin.open(info) as fin, zout.open(info, "w") as fout:
                    while True:
                        block = fin.read(1 << 20)
                        if not block:
                            break
                        fout.write(block)
                    print("  copied   %s" % info.filename)

    print("wrote %s (%d bytes)" % (OUT_ZIP, os.path.getsize(OUT_ZIP)))
    with zipfile.ZipFile(OUT_ZIP) as z:
        print("verify: %d entries" % len(z.infolist()))
        for info in z.infolist():
            print("  %-28s %12d %s" % (info.filename, info.file_size, info.compress_type))
    return 0


if __name__ == "__main__":
    sys.exit(main())

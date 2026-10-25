#!/usr/bin/env python3
"""project_rank.py — where a given score would land on a saved leaderboard snapshot."""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--target", type=float, default=1.08163)
    ap.add_argument("--lo", type=float, default=1.05)
    ap.add_argument("--hi", type=float, default=1.14)
    args = ap.parse_args()

    j = json.load(open(args.snapshot))
    rows = j.get("publicLeaderboard") or []
    sc = []
    for r in rows:
        try:
            rank = r.get("rank")
            sc.append((float(r["displayScore"]),
                       -1 if rank is None else int(rank),
                       r.get("teamId")))
        except (KeyError, TypeError, ValueError):
            pass
    sc.sort(reverse=True)
    above = [x for x in sc if x[0] > args.target]
    print("snapshot rows=%d  scored=%d" % (len(rows), len(sc)))
    print("teams strictly above %.5f: %d" % (args.target, len(above)))
    print("window [%.4f, %.4f]:" % (args.lo, args.hi))
    for s, r, t in sc:
        if args.lo <= s <= args.hi:
            print("   rank=%s score=%.5f team=%s" % (r, s, t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""poll_e5.py — poll submission 57618221 until it settles, logging timestamped lines.

Reads the cookie session via src/kaggle_submit.py (never prints credential material).
Exits 0 when a public score is visible, or when the submission errors.

  PY=../2609-xingzhi-drug/.venv/bin/python
  $PY src/e5/poll_e5.py --id 57618221 --team-id 17194714 --interval 60
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from kaggle_submit import Session, list_by_id, limits  # noqa: E402

COMPETITION_ID = 161954


def stamp() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--team-id", type=int, required=True)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--limit", type=int, default=180)
    args = ap.parse_args()

    sess = Session(verbose=False)
    for i in range(args.limit):
        sc, j = list_by_id(sess, args.team_id, [args.id])
        subs = j.get("submissions", []) or []
        s = next((x for x in subs if x.get("id") == args.id), None)
        print("[%s] poll %d http=%d body=%s" % (
            stamp(), i, sc, json.dumps(s if s is not None else subs, ensure_ascii=False)[:700]),
            flush=True)
        if s:
            status = (s.get("status") or "").lower()
            score = s.get("publicScore")
            if score not in (None, "") or status in ("complete", "completed",
                                                     "error", "failed", "cancelled"):
                print("[%s] SETTLED status=%s publicScore=%s formatted=%s" % (
                    stamp(), status, score, s.get("publicScoreFormatted")), flush=True)
                sc2, lim = limits(sess, COMPETITION_ID)
                print("[%s] limits http=%d %s" % (stamp(), sc2,
                                                  json.dumps(lim, ensure_ascii=False)), flush=True)
                return 0
        time.sleep(args.interval)
    print("[%s] gave up after %d polls" % (stamp(), args.limit), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

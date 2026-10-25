#!/usr/bin/env python3
"""e5_deviation.py — compare the E5 prediction with the realised public score."""
from __future__ import annotations

import json

REF = 2076774.23              # organiser reference score for invsqrt (brief)
E4 = 1798860.1433098556       # E4 invsqrt score (verifier)
E5 = 1641977.380188087        # E5 invsqrt score (verifier, 3 salts)
E4_LB = 1.04905               # E4 realised public score
E5_LB = 1.08145               # E5 realised public score


def main() -> None:
    r_e4 = REF / E4
    r_e5 = REF / E5
    p_e4 = r_e4 ** (1 / 3)
    p_e5 = r_e5 ** (1 / 3)
    print("invsqrt ratio  E4 = %.9f -> whole = %.9f   (realised %.5f, dev %+.6f)"
          % (r_e4, p_e4, E4_LB, E4_LB - p_e4))
    print("invsqrt ratio  E5 = %.9f -> whole = %.9f   (realised %.5f, dev %+.6f)"
          % (r_e5, p_e5, E5_LB, E5_LB - p_e5))

    # invert: what invsqrt ratio does the printed 1.08145 imply?
    implied = E5_LB ** 3
    print("\nprinted score %.5f  ->  implied invsqrt ratio = %.9f" % (E5_LB, implied))
    print("  verifier ratio                                 = %.9f" % r_e5)
    print("  relative gap                                   = %+.3e" % (implied / r_e5 - 1))
    # the 5-decimal printing quantisation band
    lo, hi = (E5_LB - 5e-6) ** 3, (E5_LB + 5e-6) ** 3
    print("  5-decimal print band                           = [%.9f, %.9f]" % (lo, hi))
    print("  verifier ratio inside band?                    = %s" % (lo <= r_e5 <= hi))

    # what score would byte-identical companions have to have to explain a 0.002 gap?
    for gap in (0.00018, 0.002):
        s_other = (E5_LB + gap) / p_e5
        print("\nE5 baseline %.5f: explaining a %+.5f gap needs the other two members "
              "at ratio %.6f (whole-competition factor %.6f)" % (E5_LB, gap, s_other ** 3, s_other))

    print("\nLB history: E3 1.04644 -> E4 1.04905 -> E5 1.08145  (delta %+.5f)"
          % (E5_LB - E4_LB))
    print("row counts: E3 795 -> E4 801 -> E5 806")

    with open("notes/runs/leaderboard_e5.json") as fh:
        rows = (json.load(fh).get("publicLeaderboard") or [])
    print("LB rows in snapshot: %d" % len(rows))


if __name__ == "__main__":
    main()

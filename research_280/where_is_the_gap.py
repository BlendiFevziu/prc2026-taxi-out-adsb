"""Given our official score, where must the 27 s to the leader live? Part three of the capacity study.

We never see per-row official feedback and must not probe for it. But the official score is one equation, the
scored set's composition is known exactly, and that is enough to bound the possibilities.

Known: 344,841 scored rows, 339,551 main (98.466 %) and 5,290 fallback (1.534 %), from the ranking file.
Our official RMSE is 266.4508 and the leader's best is 239.13.

Total squared error splits as  n * RMSE^2 = n_main * M^2 + n_fallback * F^2, so fixing either lane's RMSE fixes
the other. The table sweeps our plausible official main-lane RMSE, reports the fallback RMSE each implies, and
then asks what the leader would need if they matched us in one lane and beat us in the other. That localises the
gap without any leaderboard probing.

Anchors for reading it. On the 2025 January/July holdout our main lane scores 216.77 and our fallback 1756.91,
and the holdout total is 307.26 against an official 266.45, so the scored year is materially easier and neither
lane's holdout number transfers directly.

  python3 -B research_280/where_is_the_gap.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "research_280/results/where_is_the_gap.json"
N, N_MAIN, N_FB = 344841, 339551, 5290
OURS, LEADER = 266.4508, 239.13
HOLD_MAIN, HOLD_FB, HOLD_TOTAL = 216.773, 1756.91, 307.26


def other(total_rmse, main_rmse):
    """Fallback RMSE implied by a total and a main-lane RMSE (nan if impossible)."""
    rest = N * total_rmse ** 2 - N_MAIN * main_rmse ** 2
    return float(np.sqrt(rest / N_FB)) if rest > 0 else float("nan")


def main():
    total_sse = N * OURS ** 2
    print(f"scored set {N:,} rows: main {N_MAIN:,} ({N_MAIN/N:.3%}), fallback {N_FB:,} ({N_FB/N:.3%})")
    print(f"our official {OURS}, leader {LEADER}, gap {OURS - LEADER:.2f} s")
    print(f"our total squared error {total_sse:,.0f}\n")
    rows = []
    print(f"{'if our main is':>15s} {'implied fallback':>17s} {'main share of SSE':>18s} "
          f"{'leader needs main':>18s} {'or leader needs fb':>19s}")
    for main_rmse in (170, 180, 190, 200, 210, 216.773, 225, 240):
        fb = other(OURS, main_rmse)
        if not np.isfinite(fb):
            continue
        share = N_MAIN * main_rmse ** 2 / total_sse
        leader_main = other(LEADER, 0) * 0 + np.sqrt(max(N * LEADER ** 2 - N_FB * fb ** 2, 0) / N_MAIN)
        leader_fb = other(LEADER, main_rmse)
        rows.append({"our_main_rmse": main_rmse, "implied_our_fallback_rmse": fb,
                     "main_share_of_sse": share, "leader_main_if_fallback_tied": float(leader_main),
                     "leader_fallback_if_main_tied": float(leader_fb) if np.isfinite(leader_fb) else None})
        lf = f"{leader_fb:,.0f}" if np.isfinite(leader_fb) else "impossible"
        print(f"{main_rmse:>15.1f} {fb:>17,.0f} {share:>17.1%} {leader_main:>18.1f} {lf:>19s}")

    print("\nReading the last two columns: to reach 239.13 the leader must either beat us in the main lane by the\n"
          "amount in 'leader needs main' while tying our fallback, or beat us in the fallback by the amount in\n"
          "'or leader needs fb' while tying our main lane. Any mix lies between.")
    OUTPUT.write_text(json.dumps({"note": "algebra on the official score and the known scored composition; "
                                          "no leaderboard probing, no per-row feedback used",
                                  "scored_rows": N, "main_rows": N_MAIN, "fallback_rows": N_FB,
                                  "our_official": OURS, "leader_official": LEADER,
                                  "holdout": {"main": HOLD_MAIN, "fallback": HOLD_FB, "total": HOLD_TOTAL},
                                  "scenarios": rows}, indent=2) + "\n")
    print(f"\nwritten {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()

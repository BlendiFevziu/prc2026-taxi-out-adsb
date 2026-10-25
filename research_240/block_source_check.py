"""Which timestamp does the airport's block time equal in the 'neither' class? Descriptive, 2025 release cache.

Candidates at rank time: EOBT_1, LOBT and IOBT (TAXITIME equal to ersatz_eobt_s, ersatz_lobt_s or ersatz_iobt_s),
the schedule shifted by one day either way, and whole-hour offsets of AOBT_3 (time-zone slips: rest = AOBT_3 - BLOCK
near 3,600, 7,200 or 10,800 s, with neighbouring windows as controls). Share of rows within 60 s per group; 5-minute
histogram of rest for the 'neither' class (main lane, not a schedule copy, rest > 1,800 s). No model, no leaderboard.

Run:  python3 -B research_240/block_source_check.py <output.json>
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
cols = ["ADEP_mvt", "y", "ersatz_s", "ersatz_eobt_s", "ersatz_lobt_s", "ersatz_iobt_s", "versp_min", "hat_aobt"]
d = pd.read_parquet(P / ".cache/clock_queue_release_20260912/data/months_1_2_3_4_5_6_7_8_9_10_11_12.parquet", columns=cols)
d = d[d.y.notna()].reset_index(drop=True)
y = d.y.to_numpy(float)
delay = d.versp_min.to_numpy(float) * 60
copied = np.abs(y - delay) <= 10
main = d.hat_aobt.eq(1).to_numpy()
rest = y - d.ersatz_s.to_numpy(float)
groups = {"all_rows": np.ones(len(d), bool), "main_not_copied": main & ~copied,
          "neither": main & ~copied & (rest > 1800), "fallback_extreme_not_copied": ~main & ~copied & (y > 5400)}
candidates = {"eobt_1": np.abs(y - d.ersatz_eobt_s.to_numpy(float)) <= 60, "lobt": np.abs(y - d.ersatz_lobt_s.to_numpy(float)) <= 60,
              "iobt": np.abs(y - d.ersatz_iobt_s.to_numpy(float)) <= 60,
              "sched_plus_day": np.abs(y - (delay + 86400)) <= 60, "sched_minus_day": np.abs(y - (delay - 86400)) <= 60}
for hours in (1, 2, 3):
    for label, centre in ((f"aobt_minus_{hours}h", 3600 * hours), (f"control_{hours}h_minus_5min", 3600 * hours - 300),
                          (f"control_{hours}h_plus_5min", 3600 * hours + 300)):
        candidates[label] = main & (np.abs(rest - centre) <= 60)
report = {"note": "descriptive; 2025 release cache, all months", "groups": {}}
for g, mask in groups.items():
    report["groups"][g] = {"rows": int(mask.sum()), **{c: {"share": float(np.mean(hit[mask])), "rows": int(hit[mask].sum())}
                                                        for c, hit in candidates.items()}}
neither = groups["neither"]
bins = np.arange(1800, 14401, 300)
hist, _ = np.histogram(rest[neither], bins=bins)
report["neither_rest_histogram_5min"] = {f"{int(lo)}-{int(lo + 300)}": int(c) for lo, c in zip(bins[:-1], hist)}
airports = d.ADEP_mvt.astype(str).to_numpy()
report["neither_by_airport"] = {a: {"rows": int((neither & (airports == a)).sum()),
                                    "rest_median_s": float(np.median(rest[neither & (airports == a)])) if (neither & (airports == a)).any() else None,
                                    **{c: int(candidates[c][neither & (airports == a)].sum()) for c in ("eobt_1", "lobt", "iobt", "aobt_minus_1h", "aobt_minus_2h")}}
                                for a in sorted(set(airports))}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
for g, s in report["groups"].items():
    print(g, s["rows"], {c: (round(v["share"], 4), v["rows"]) for c, v in s.items() if c != "rows"})
print("neither rest histogram (5 min bins):", report["neither_rest_histogram_5min"])
print("neither by airport:", report["neither_by_airport"])

"""Is the departure block time sometimes the in-block time of the last arrival at the same stand? Descriptive, 2025.

If an airport database falls back to the rotation's arrival in-block time when no off-block was reported, the target
TAXITIME = MVT_TIME - BLOCK_TIME equals um_seit_ank = MVT_TIME - (in-block of the last arrival at the same airport and
STAND_mvt at or before MVT_TIME), a feature both lanes already have. Compares the share of rows with
|TAXITIME - um_seit_ank| <= 10 s and <= 60 s across: all rows, schedule copies (|TAXITIME - delay| <= 10 s),
main rows with AOBT_3 - BLOCK_TIME > 1,800 s that are not copies (a participant's "neither" class), and fallback rows above
5,400 s that are not copies. v45 release cache, all twelve 2025 months. No model, no leaderboard feedback.

Run:  python3 -B research_240/inblock_substitution_check.py <output.json>
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
d = pd.read_parquet(P / ".cache/clock_queue_release_20260912/data/months_1_2_3_4_5_6_7_8_9_10_11_12.parquet",
                    columns=["ADEP_mvt", "y", "ersatz_s", "versp_min", "um_seit_ank", "hat_aobt", "monat"])
d = d[d.y.notna()].reset_index(drop=True)
y, since = d.y.to_numpy(float), d.um_seit_ank.to_numpy(float)
copied = np.abs(y - d.versp_min.to_numpy(float) * 60) <= 10
main = d.hat_aobt.eq(1).to_numpy()
rest = y - d.ersatz_s.to_numpy(float)
groups = {"all_rows": np.ones(len(d), bool), "schedule_copies": copied, "main_not_copied": main & ~copied,
          "main_neither_rest_over_1800": main & ~copied & (rest > 1800), "fallback_all": ~main,
          "fallback_extreme_not_copied": ~main & ~copied & (y > 5400)}


def summary(mask):
    ok = mask & np.isfinite(since)
    diff = np.abs(y[ok] - since[ok])
    return {"rows": int(mask.sum()), "with_um_seit_ank": int(ok.sum()),
            "share_within_10s": float(np.mean(diff <= 10)) if ok.any() else None,
            "share_within_60s": float(np.mean(diff <= 60)) if ok.any() else None,
            "rows_within_60s": int(np.sum(diff <= 60))}


airports = d.ADEP_mvt.astype(str).to_numpy()
report = {"note": "descriptive; 2025 release cache, all months; TAXITIME compared with um_seit_ank",
          "groups": {name: summary(mask) for name, mask in groups.items()},
          "by_airport": {a: {name: summary(mask & (airports == a)) for name, mask in groups.items()
                             if name in ("all_rows", "main_neither_rest_over_1800", "fallback_extreme_not_copied")}
                         for a in sorted(set(airports))}}
match = np.isfinite(since) & (np.abs(y - since) <= 60)
report["matched_rows_mse_view"] = {"rows_within_60s": int(match.sum()), "share_of_rows": float(match.mean()),
                                   "sum_y_squared_share_of_all": float(np.sum(y[match]**2) / np.sum(y**2)),
                                   "median_y_matched_s": float(np.median(y[match])) if match.any() else None}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
for name, s in report["groups"].items():
    print(f"{name:32s} rows {s['rows']:>9,}  <=10s {s['share_within_10s']:.4f}  <=60s {s['share_within_60s']:.4f}  n<=60s {s['rows_within_60s']:,}")
print("by airport (neither class <=60s share, fallback extreme <=60s share):",
      {a: (v["main_neither_rest_over_1800"]["rows"], None if v["main_neither_rest_over_1800"]["share_within_60s"] is None else round(v["main_neither_rest_over_1800"]["share_within_60s"], 3),
           v["fallback_extreme_not_copied"]["rows"], None if v["fallback_extreme_not_copied"]["share_within_60s"] is None else round(v["fallback_extreme_not_copied"]["share_within_60s"], 3))
       for a, v in report["by_airport"].items()})
print(json.dumps(report["matched_rows_mse_view"]))

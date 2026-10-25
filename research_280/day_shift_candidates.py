"""Diagnostic (16.09.2026): is there an observable subset that contains the day-shift labels?

Day shifts are the largest untouched block of error in the model. The airport writes a block time carrying the
previous day's date, so the label lands near 86,400 s plus the real taxi while the model predicts about 2,000 s.
the team's six-pair error map put 14 such rows at 14.5 % of ALL squared error, 13 of them at Rome, and found no clock
that explains them.

Why they deserve another look even with a weak signal: squared error is convex, so for a subset whose day-shift rate
is r, moving every prediction by r * 86,400 lowers the expected squared error by r^2 * 86,400^2 per row. At r = 0.04
that is 12 million per row, an order of magnitude above anything the copy work produced. The question is therefore
not "can we identify them" but "is there an observable subset with a high enough rate, and how many rows does it
cost to cover".

Candidate subsets, all observable at rank time (no label, no block time):
  A  movement and scheduled timestamps fall on different calendar dates
  B  the scheduled timestamp is on the day before the movement
  C  movement hour between 00:00 and 05:59 UTC
  D  airport is LIRF
  E  combinations of the above
Reported per subset: rows, day shifts inside, rate, recall, and the expected-value arithmetic above, together with
the cost side, which is what a real screen would have to survive.

Diagnostic only. Challenge data only.
Run from the repository root:  python3 -B research_280/day_shift_candidates.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
           "TAXITIME_SEC_mvt", "AOBT_3_flt"]
OUTPUT = Path("research_280/results/day_shift_candidates.json")
GAP = 86400.0

frames = [pq.ParquetFile(p).read(columns=COLUMNS, use_threads=False).to_pandas()
          for p in sorted(glob.glob("data/training_2025-*.parquet"))]
data = pd.concat(frames, ignore_index=True)
data = data.loc[data.PHASE_mvt.eq("DEP")].reset_index(drop=True)
data["t"] = pd.to_datetime(data.MVT_TIME_UTC_mvt, utc=True)
data["s"] = pd.to_datetime(data.SCHED_TIME_UTC_mvt, utc=True)
data["y"] = data.TAXITIME_SEC_mvt.astype(float)
data = data.dropna(subset=["y"]).reset_index(drop=True)
data["fallback"] = data.AOBT_3_flt.isna()
data["shift"] = data.y.between(80000, 93600)
data["date_cross"] = data.t.dt.date.astype(str).ne(data.s.dt.date.astype(str))
data["sched_previous_day"] = (data.t.dt.normalize() - data.s.dt.normalize()).dt.days.eq(1)
data["night"] = data.t.dt.hour.lt(6)
data["rome"] = data.ADEP_mvt.eq("LIRF")

fallback = data.loc[data.fallback]
report = {"departures": int(len(data)), "fallback_rows": int(len(fallback)),
          "day_shifts_all_rows": int(data["shift"].sum()), "day_shifts_fallback": int(fallback["shift"].sum()),
          "day_shifts_by_airport": data.loc[data["shift"]].ADEP_mvt.value_counts().to_dict(),
          "day_shifts_in_main_lane": int(data.loc[~data.fallback, "shift"].sum())}
print(f"2025 departures {len(data):,}, fallback {len(fallback):,}")
print(f"day shifts: {int(data['shift'].sum())} in total, {int(fallback['shift'].sum())} of them fallback, "
      f"by airport {report['day_shifts_by_airport']}")

subsets = {
    "A date_cross (fallback)": fallback.date_cross,
    "B sched_previous_day (fallback)": fallback.sched_previous_day,
    "C night 00-06 UTC (fallback)": fallback.night,
    "D Rome (fallback)": fallback.rome,
    "E Rome and date_cross": fallback.rome & fallback.date_cross,
    "F Rome and night": fallback.rome & fallback.night,
    "G Rome and sched_previous_day": fallback.rome & fallback.sched_previous_day,
    "H date_cross, any lane": data.date_cross,
}
print("\nsubset                            rows   shifts     rate   recall   EV gain per row   EV gain total")
report["subsets"] = {}
total_shifts = int(fallback["shift"].sum())
for name, mask in subsets.items():
    frame = data if "any lane" in name else fallback
    mask = mask.to_numpy(bool)
    rows = int(mask.sum())
    shifts = int(frame.loc[mask, "shift"].sum())
    rate = shifts / rows if rows else float("nan")
    gain_per_row = rate ** 2 * GAP ** 2
    block = {"rows": rows, "shifts": shifts, "rate": rate,
             "recall_of_fallback_shifts": shifts / max(total_shifts, 1),
             "ev_gain_per_row": gain_per_row, "ev_gain_total": gain_per_row * rows,
             "ev_shift_seconds": rate * GAP}
    report["subsets"][name] = block
    print(f"{name:32s} {rows:6d} {shifts:8d} {rate:8.4f} {block['recall_of_fallback_shifts']:8.2f} "
          f"{gain_per_row:17,.0f} {block['ev_gain_total']:15,.0f}")

best = max(report["subsets"].items(), key=lambda kv: kv[1]["ev_gain_total"])
print(f"\nlargest expected value: {best[0]}, shifting {best[1]['ev_shift_seconds']:.0f} s onto {best[1]['rows']} rows")
print("translated to the fallback screen scale (mean gain per fallback row):",
      round(best[1]["ev_gain_total"] / len(fallback)))
mechanical = np.sqrt(max(266.4508 ** 2 - (5290 / 344841) * best[1]["ev_gain_total"] / len(fallback), 0))
print(f"mechanical official RMSE if that transferred in full (NOT a forecast): {mechanical:.4f} from 266.4508")
report["best_subset"] = best[0]
report["mechanical_leaderboard_rmse_not_forecast"] = float(mechanical)

print("\nhow the model currently predicts the fallback day shifts (from the saved six-pair rows):")
try:
    rows = pd.read_parquet(".cache/research_280/rome_copy_hard_gate_rows.parquet")
    shifts = rows.loc[rows.y.between(80000, 93600)]
    print(shifts[["fold", "airport", "y", "delay", "prod_anchor", "p"]].round(0).to_string(index=False))
    report["model_on_day_shifts"] = shifts[["fold", "airport", "y", "delay", "prod_anchor", "p"]].round(2).to_dict("records")
except FileNotFoundError:
    print("  (saved rows not present)")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
print("\nwritten", OUTPUT)

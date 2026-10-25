"""Diagnostic (16.09.2026): are Rome's copies and day shifts concentrated on disrupted days, and is 2026 different?

Prompted by the owner's account of Fiumicino in 2026: nationwide transport strikes and schedule backlogs in January,
a security-screening walkout on 5 July 2026 from 10:00 to 18:00 CEST, and an Italy-wide drop in ground-equipment
utilisation through the first half of the year.

Two reasons not to hand-code those dates as a feature. The training data is 2025 and the flagged dates are 2026, so
nothing could validate the effect. And the sign is not obvious: a terminal bottleneck delays the push-back, which
moves the block time later, and taxi-out is take-off minus block time, so it may not lengthen at all.

What is testable is a day-level disruption measure built only from quantities the ranking file also carries:
movement time against scheduled time, departure counts, and the share of departures with no NM record. None of them
needs a label, so whatever holds in 2025 can be applied in 2026.

Questions, all measured at LIRF unless stated:
  1. On the most disrupted 2025 days, is taxi-out longer, shorter or unchanged? This settles the sign.
  2. Do copies and day shifts concentrate on disrupted days? If so, a day-level measure could sharpen the copy
     classifier, whose missed copies currently sit at a median probability of 0.74.
  3. Does the fallback share itself rise on disrupted days, which would tie the missing NM records to the same cause?
  4. Label-free drift: do January and July 2026 show a higher delay level than the same months of 2025, and does
     5 July 2026, 08:00 to 16:00 UTC, stand out in the ranking inputs at all?

Diagnostic only, no model, no gate. Challenge data only.
Run from the repository root:  python3 -B research_280/rome_disruption_days.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
           "TAXITIME_SEC_mvt", "AOBT_3_flt"]
OUTPUT = Path("research_280/results/rome_disruption_days.json")
AIRPORT = "LIRF"


def load(paths, labelled):
    frames = []
    for path in paths:
        table = pq.ParquetFile(path).read(columns=COLUMNS, use_threads=False).to_pandas()
        frames.append(table.loc[table.PHASE_mvt.eq("DEP") & table.ADEP_mvt.eq(AIRPORT)])
    data = pd.concat(frames, ignore_index=True)
    data["t"] = pd.to_datetime(data.MVT_TIME_UTC_mvt, utc=True)
    data["sched_delay"] = (data.t - pd.to_datetime(data.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    data["day"] = data.t.dt.strftime("%Y-%m-%d")
    data["fallback"] = data.AOBT_3_flt.isna()
    if labelled:
        data["y"] = data.TAXITIME_SEC_mvt.astype(float)
        data = data.dropna(subset=["y"])
        data["copy"] = data.sched_delay.notna() & (data.y - data.sched_delay).abs().le(60)
        data["day_shift"] = data.y.ge(80000)
    return data.reset_index(drop=True)


train = load(sorted(glob.glob("data/training_2025-*.parquet")), True)
report = {"rows_2025": int(len(train))}
print(f"LIRF departures 2025 with a label: {len(train):,}")

days = train.groupby("day").agg(departures=("y", "size"), median_delay=("sched_delay", "median"),
                                fallback_share=("fallback", "mean"), median_taxi=("y", "median"),
                                copies=("copy", "sum"), shifts=("day_shift", "sum"))
days["quintile"] = pd.qcut(days.median_delay, 5, labels=False)
print("\n1 and 2. LIRF days grouped by that day's median movement-minus-schedule delay (observable in 2026 too):")
grouped = days.groupby("quintile").agg(days=("departures", "size"), median_delay=("median_delay", "median"),
                                       departures=("departures", "median"), median_taxi=("median_taxi", "median"),
                                       fallback_share=("fallback_share", "median"), copies=("copies", "sum"),
                                       shifts=("shifts", "sum"))
print(grouped.round(3).to_string())
report["by_delay_quintile"] = grouped.round(4).to_dict("index")

print("\n3. Same days, but measured on the rows themselves:")
train = train.merge(days[["quintile", "median_delay"]], left_on="day", right_index=True)
rows = train.groupby("quintile").agg(rows=("y", "size"), fallback=("fallback", "mean"), taxi_median=("y", "median"),
                                     taxi_p90=("y", lambda v: float(np.quantile(v, .9))))
fb = train.loc[train.fallback].groupby("quintile").agg(fallback_rows=("y", "size"), copy_rate=("copy", "mean"),
                                                       shift_rate=("day_shift", "mean"))
print(rows.join(fb).round(4).to_string())
report["rows_by_quintile"] = rows.join(fb).round(4).to_dict("index")

worst = days.nlargest(8, "median_delay")
print("\nthe eight most delayed LIRF days of 2025:")
print(worst.drop(columns=["quintile"]).round(2).to_string())
report["worst_days_2025"] = worst.drop(columns=["quintile"]).round(3).to_dict("index")

print("\n4. Label-free comparison with the scored months (ranking inputs, no labels read):")
rank = load(["data/ranking.parquet"], False)
for name, frame in (("2025", train), ("2026", rank)):
    for month in (1, 7):
        part = frame.loc[pd.to_datetime(frame.t).dt.month.eq(month)]
        if not len(part):
            continue
        key = f"{name}_month_{month}"
        block = {"departures": int(len(part)), "median_sched_delay_s": float(part.sched_delay.median()),
                 "p90_sched_delay_s": float(np.quantile(part.sched_delay.dropna(), .9)),
                 "fallback_share": float(part.fallback.mean())}
        report[key] = block
        print(f"  {key}: {block['departures']:6d} departures, median delay {block['median_sched_delay_s']:7.0f} s, "
              f"p90 {block['p90_sched_delay_s']:8.0f} s, fallback share {block['fallback_share']:.3%}")

window = rank.loc[rank.t.between("2026-07-05 08:00:00+00:00", "2026-07-05 16:00:00+00:00")]
normal = rank.loc[pd.to_datetime(rank.t).dt.month.eq(7) & ~rank.day.eq("2026-07-05")]
strike = {"departures_in_window": int(len(window)),
          "median_sched_delay_s": float(window.sched_delay.median()) if len(window) else float("nan"),
          "fallback_share": float(window.fallback.mean()) if len(window) else float("nan"),
          "july_2026_other_days_median_delay_s": float(normal.sched_delay.median()),
          "july_2026_other_days_fallback_share": float(normal.fallback.mean())}
report["july_5_2026_window"] = strike
print("\n5 July 2026, 08:00-16:00 UTC, against the rest of July 2026:")
print(json.dumps(strike, indent=1))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
print("\nwritten", OUTPUT)

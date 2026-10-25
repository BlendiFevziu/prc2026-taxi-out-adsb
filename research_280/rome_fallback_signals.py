"""Diagnostic (16.09.2026): are Rome's copied block times clustered, and does anything label-free mark them?

Why Rome. The six-pair error map (research_270/results/fallback_error_map.json) puts 37 % of all squared error on
22,470 fallback rows, and Rome carries most of the worst mechanisms: 720 of 1,123 schedule copies, 13 of 14 day
shifts, and 640 of 777 extremes above 5,400 s. A perfect prediction on those mechanisms is worth roughly 20 s
locally, which is the distance to the 240 target.

What has already failed, and must not be repeated: a copy classifier on the production fallback inputs (EV and hard
substitution, research_270/copy_classifier_ev_screen.py), the NM taxi-time source as a copy marker (refuted
16.09.2026: all 19 matched copies came through CDM source), the own-arrival copy flag, and the inbound-stand flag.

The untested idea here. A copied block time and a missing NM record plausibly share one cause: the airport's
reporting for that period failed. If so, copies arrive in bursts, and burst membership is observable at rank time
because AOBT_3 presence is an input column, not a label. This script measures, on 2025 labels:
  1. the copy rate by hour, stand, operator and month at LIRF,
  2. whether copies sit in windows with an unusually high share of fallback departures,
  3. whether copies are contiguous in time (gap to the nearest other fallback departure),
against non-copy fallback rows at the same airport.

Diagnostic only. No model is fitted and no gate is claimed; a separating signal would still need a pre-registered
six-pair screen. Challenge data only.

Run from the repository root:  python3 -B research_280/rome_fallback_signals.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
           "TAXITIME_SEC_mvt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt"]
OUTPUT = Path("research_280/results/rome_fallback_signals.json")
WINDOWS = (30, 60, 120)


def load():
    frames = []
    for path in sorted(glob.glob("data/training_2025-*.parquet")):
        table = pq.ParquetFile(path).read(columns=COLUMNS, use_threads=False).to_pandas()
        frames.append(table.loc[table.PHASE_mvt.eq("DEP")])
    data = pd.concat(frames, ignore_index=True)
    data["t"] = pd.to_datetime(data.MVT_TIME_UTC_mvt, utc=True)
    data["sched_delay"] = (data.t - pd.to_datetime(data.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    data["y"] = data.TAXITIME_SEC_mvt.astype(float)
    data["fallback"] = data.AOBT_3_flt.isna()
    data["copy"] = data.sched_delay.notna() & (data.y - data.sched_delay).abs().le(60)
    return data.dropna(subset=["y"]).sort_values(["ADEP_mvt", "t"]).reset_index(drop=True)


def burst_features(group):
    """Label-free: how much of the local traffic also lost its NM record, and how close the nearest one is."""
    t = group.t.to_numpy("datetime64[s]").astype("int64")
    fb = group.fallback.to_numpy()
    out = {}
    for minutes in WINDOWS:
        span = minutes * 60
        left = np.searchsorted(t, t - span, side="left")
        right = np.searchsorted(t, t + span, side="right")
        total = right - left
        cumulative = np.concatenate([[0], np.cumsum(fb)])
        fallback_count = cumulative[right] - cumulative[left]
        out[f"n_dep_{minutes}"] = total
        out[f"n_fallback_{minutes}"] = fallback_count
        out[f"share_fallback_{minutes}"] = fallback_count / np.maximum(total, 1)
    index = np.flatnonzero(fb)
    nearest = np.full(len(group), np.nan)
    if len(index) > 1:
        times = t[index]
        previous = np.concatenate([[np.nan], np.diff(times)])
        following = np.concatenate([np.diff(times), [np.nan]])
        nearest[index] = np.nanmin(np.vstack([previous, following]), axis=0)
    out["gap_to_nearest_fallback_s"] = nearest
    return pd.DataFrame(out, index=group.index)


def describe(frame, key, minimum=20):
    grouped = frame.groupby(key, observed=True).agg(rows=("copy", "size"), copies=("copy", "sum"),
                                                    copy_rate=("copy", "mean"), median_y=("y", "median"))
    return grouped.loc[grouped.rows >= minimum].sort_values("copy_rate", ascending=False)


data = load()
data = pd.concat([data, data.groupby("ADEP_mvt", group_keys=False).apply(burst_features)], axis=1)
fallback = data.loc[data.fallback]
report = {"rows": len(data), "fallback_rows": int(len(fallback)), "copies": int(fallback["copy"].sum())}
print(f"2025 departures {len(data):,}, fallback {len(fallback):,}, copies among fallback {int(fallback['copy'].sum()):,}")

by_airport = fallback.groupby("ADEP_mvt", observed=True).agg(rows=("copy", "size"), copy_rate=("copy", "mean"),
                                                             extreme_rate=("y", lambda v: float((v > 5400).mean())))
report["by_airport"] = by_airport.round(4).to_dict("index")
print("\ncopy and extreme rate among fallback rows, by airport:")
print(by_airport.round(3).sort_values("copy_rate", ascending=False).to_string())

rome = fallback.loc[fallback.ADEP_mvt.eq("LIRF")].copy()
print(f"\nLIRF fallback rows {len(rome)}, copies {int(rome['copy'].sum())} ({rome['copy'].mean():.1%})")
print("\nDo copies sit in bursts? medians among LIRF fallback rows:")
burst = {}
for minutes in WINDOWS:
    for column in (f"share_fallback_{minutes}", f"n_fallback_{minutes}", f"n_dep_{minutes}"):
        a = rome.loc[rome["copy"], column].median()
        b = rome.loc[~rome["copy"], column].median()
        burst[column] = {"copies": float(a), "non_copies": float(b)}
        print(f"  {column:22s} copies {a:8.3f}   non-copies {b:8.3f}")
gap_copy = rome.loc[rome["copy"], "gap_to_nearest_fallback_s"].median()
gap_other = rome.loc[~rome["copy"], "gap_to_nearest_fallback_s"].median()
burst["gap_to_nearest_fallback_s"] = {"copies": float(gap_copy), "non_copies": float(gap_other)}
print(f"  {'gap to nearest fallback':22s} copies {gap_copy:8.0f} s non-copies {gap_other:8.0f} s")
report["burst_medians_lirf"] = burst

print("\nLIRF copy rate by hour of day:")
rome["hour"] = rome.t.dt.hour
print(describe(rome, "hour", minimum=10).round(3).to_string())
report["lirf_by_hour"] = describe(rome, "hour", minimum=10).round(4).to_dict("index")

for key in ("STAND_mvt", "AIRCRAFT_OPERATOR_flt", "RUNWAY_mvt"):
    table = describe(rome, key, minimum=15)
    print(f"\nLIRF copy rate by {key} (groups of 15+, top 10):")
    print(table.head(10).round(3).to_string())
    report[f"lirf_by_{key}"] = table.head(20).round(4).to_dict("index")

rome["day"] = rome.t.dt.strftime("%Y-%m-%d")
days = rome.groupby("day").agg(rows=("copy", "size"), copies=("copy", "sum"))
days = days.loc[days.rows >= 3]
report["lirf_days"] = {"days_with_3plus_fallback": int(len(days)),
                       "days_with_any_copy": int((days.copies > 0).sum()),
                       "days_where_all_fallback_rows_are_copies": int((days.copies == days.rows).sum()),
                       "copies_share_in_top_10_days": float(days.copies.nlargest(10).sum() / max(days.copies.sum(), 1))}
print("\nLIRF days with at least three fallback departures:", json.dumps(report["lirf_days"], indent=1))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
print("\nwritten", OUTPUT)

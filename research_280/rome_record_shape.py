"""Diagnostic (16.09.2026): does a Rome movement record whose block time is a schedule copy LOOK different?

Where this comes from. research_280/rome_fallback_signals.py showed copies are not an outage burst and that hour and
stand, both already available to the fallback model, carry most of the obvious structure. What remains unexplained is
the team's join forensics: fallback rows lack FLIGHT_ID_mvt on 99.8 % of cases, so the movement record itself looks
defective. If Rome writes some records from the schedule rather than from operations, the record should carry tells
that have nothing to do with the label: rounded timestamps, missing stand or runway, unusual flight identifiers.

Every quantity here is observable at rank time (no _flt column, no block time, no label), so anything that separates
can become a feature for the whole scored set, not only the six days the NM archives cover.

Tested on 2025 fallback departures, copies (|y - (movement - schedule)| <= 60 s) against non-copies, at LIRF and,
as a control, at the other nine airports:
  seconds and minute of the movement timestamp (a synthesised time tends to be round)
  seconds of the scheduled timestamp
  stand and runway missing
  flight identifier shape: length, digit count, whether the same identifier appears on other days
  schedule delay and taxi label, to show what is at stake

Diagnostic only, no model, no gate. Challenge data only.
Run from the repository root:  python3 -B research_280/rome_record_shape.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
           "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_RULE_mvt"]
OUTPUT = Path("research_280/results/rome_record_shape.json")

frames = []
for path in sorted(glob.glob("data/training_2025-*.parquet")):
    table = pq.ParquetFile(path).read(columns=COLUMNS, use_threads=False).to_pandas()
    frames.append(table.loc[table.PHASE_mvt.eq("DEP")])
data = pd.concat(frames, ignore_index=True)
data["t"] = pd.to_datetime(data.MVT_TIME_UTC_mvt, utc=True)
data["s"] = pd.to_datetime(data.SCHED_TIME_UTC_mvt, utc=True)
data["sched_delay"] = (data.t - data.s).dt.total_seconds()
data["y"] = data.TAXITIME_SEC_mvt.astype(float)
data = data.dropna(subset=["y"]).reset_index(drop=True)
data["fallback"] = data.AOBT_3_flt.isna()
data["copy"] = data.sched_delay.notna() & (data.y - data.sched_delay).abs().le(60)

data["mvt_second"] = data.t.dt.second
data["mvt_minute"] = data.t.dt.minute
data["sched_second"] = data.s.dt.second
data["mvt_second_zero"] = data.mvt_second.eq(0)
data["mvt_minute_mod5"] = data.mvt_minute.mod(5).eq(0)
data["sched_second_zero"] = data.sched_second.eq(0)
data["stand_missing"] = data.STAND_mvt.isna() | data.STAND_mvt.astype(str).str.strip().eq("")
data["runway_missing"] = data.RUNWAY_mvt.isna() | data.RUNWAY_mvt.astype(str).str.strip().eq("")
data["flight_id_missing"] = data.FLIGHT_ID_mvt.isna()
flight = data.FLIGHT_mvt.astype(str)
data["flight_len"] = flight.str.len()
data["flight_digits"] = flight.str.count(r"\d")
seen = data.groupby(["ADEP_mvt", "FLIGHT_mvt"], observed=True).MVT_ID_mvt.transform("size")
data["identifier_days_seen"] = seen

BINARY = ["mvt_second_zero", "mvt_minute_mod5", "sched_second_zero", "stand_missing", "runway_missing",
          "flight_id_missing"]
NUMERIC = ["mvt_second", "flight_len", "flight_digits", "identifier_days_seen", "sched_delay", "y"]
report = {}


def compare(frame, name):
    copies, others = frame.loc[frame["copy"]], frame.loc[~frame["copy"]]
    block = {"rows": int(len(frame)), "copies": int(len(copies)), "non_copies": int(len(others))}
    print(f"\n=== {name}: {len(frame)} fallback rows, {len(copies)} copies, {len(others)} non-copies")
    for column in BINARY:
        a, b = float(copies[column].mean()), float(others[column].mean())
        block[column] = {"copies": a, "non_copies": b}
        print(f"  {column:22s} copies {a:7.3%}   non-copies {b:7.3%}")
    for column in NUMERIC:
        a, b = float(copies[column].median()), float(others[column].median())
        block[column + "_median"] = {"copies": a, "non_copies": b}
        print(f"  {column:22s} copies {a:10.1f}   non-copies {b:10.1f}   (median)")
    return block


fallback = data.loc[data.fallback]
report["LIRF"] = compare(fallback.loc[fallback.ADEP_mvt.eq("LIRF")], "LIRF fallback")
report["other_airports"] = compare(fallback.loc[~fallback.ADEP_mvt.eq("LIRF")], "other airports, fallback")
report["LIRF_main_lane_control"] = compare(
    data.loc[data.ADEP_mvt.eq("LIRF") & ~data.fallback], "LIRF main lane (control: rows that DO have AOBT_3)")

rome = fallback.loc[fallback.ADEP_mvt.eq("LIRF")]
print("\nLIRF fallback: movement-second distribution, copies against non-copies (share per 10 s bucket)")
buckets = pd.cut(rome.mvt_second, bins=range(0, 61, 10), right=False)
table = pd.crosstab(buckets, rome["copy"], normalize="columns")
print(table.round(3).to_string())
report["LIRF_second_buckets"] = {str(k): {str(c): float(v) for c, v in row.items()} for k, row in table.iterrows()}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
print("\nwritten", OUTPUT)

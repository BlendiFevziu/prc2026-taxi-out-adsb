"""Diagnostic (16.09.2026): does an airport's ARRIVAL reporting behaviour predict its DEPARTURE block times?

The idea comes from what the NM work established about the target. We do not predict a physical taxi duration, we
predict the timestamp the airport wrote into BLOCK_TIME, and the main-lane target is exactly the disagreement
between NM's clock and that written value (rest = AOBT_3 - BLOCK_TIME). Nothing in the model measures how an
airport is writing timestamps at the moment of a departure.

The arrival side makes that observable, and it stays observable at rank time: in ranking.parquet the arrival rows
carry BLOCK_TIME, TAXITIME and SCHED_TIME at 100 %, while the departure rows carry none of them. features_arrivals.py
already uses arrival taxi-in as a congestion measure; this asks a different question of the same rows.

Three candidate measures per airport and two-hour bucket, all from input columns only:
  sek_anteil   share of arrival block times whose seconds field is not zero (a machine-written stamp has seconds,
               a typed or derived one tends to be round)
  kopie        share of arrivals whose block time equals the scheduled time within 60 s
  nm_fehlt     share of arrivals with no NM arrival record (ARVT_3_flt missing)

Tested against, on 2025 departures of the same airport and bucket:
  main lane   rest = y - ersatz_s = AOBT_3 - BLOCK_TIME, its median and its RMSE
  fallback    the copy rate and the extreme rate

Diagnostic only, no model and no gate. A separating measure would then be built as features and screened with the
queue-screen protocol. Challenge data only.

Run from the repository root:  python3 -B research_280/reporting_state_signal.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
           "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "ARVT_3_flt"]
OUTPUT = Path("research_280/results/reporting_state_signal.json")
BUCKET = "2h"

frames = [pq.ParquetFile(p).read(columns=COLUMNS, use_threads=False).to_pandas()
          for p in sorted(glob.glob("data/training_2025-*.parquet"))]
data = pd.concat(frames, ignore_index=True)
data["t"] = pd.to_datetime(data.MVT_TIME_UTC_mvt, utc=True)
data["block"] = pd.to_datetime(data.BLOCK_TIME_UTC_mvt, utc=True)
data["sched"] = pd.to_datetime(data.SCHED_TIME_UTC_mvt, utc=True)
data["bucket"] = data.t.dt.floor(BUCKET)
# The airport whose reporting we are measuring is the departure airport for DEP rows and the destination for ARR.
data["airport"] = np.where(data.PHASE_mvt.eq("ARR"), data.ADES_mvt.astype(str), data.ADEP_mvt.astype(str))

arrivals = data.loc[data.PHASE_mvt.eq("ARR") & data.block.notna()].copy()
arrivals["sek"] = arrivals.block.dt.second.ne(0)
arrivals["kopie"] = (arrivals.block - arrivals.sched).dt.total_seconds().abs().le(60)
arrivals["nm_fehlt"] = arrivals.ARVT_3_flt.isna()
state = arrivals.groupby(["airport", "bucket"], observed=True).agg(
    sek_anteil=("sek", "mean"), kopie=("kopie", "mean"), nm_fehlt=("nm_fehlt", "mean"),
    ank_n=("sek", "size"), ank_taxi_med=("TAXITIME_SEC_mvt", "median")).reset_index()
print(f"arrival rows {len(arrivals):,} in {len(state):,} airport-buckets")
print("arrival reporting by airport (whole year):")
by_airport = arrivals.groupby("airport", observed=True).agg(rows=("sek", "size"), sek_anteil=("sek", "mean"),
                                                            kopie=("kopie", "mean"), nm_fehlt=("nm_fehlt", "mean"))
print(by_airport.round(4).to_string())

departures = data.loc[data.PHASE_mvt.eq("DEP") & data.TAXITIME_SEC_mvt.notna()].copy()
departures["y"] = departures.TAXITIME_SEC_mvt.astype(float)
departures["ersatz"] = (departures.t - pd.to_datetime(departures.AOBT_3_flt, utc=True)).dt.total_seconds()
departures["rest"] = departures.y - departures.ersatz
departures["fallback"] = departures.AOBT_3_flt.isna()
departures["sched_delay"] = (departures.t - departures.sched).dt.total_seconds()
departures["copy"] = departures.sched_delay.notna() & (departures.y - departures.sched_delay).abs().le(60)
departures = departures.merge(state, on=["airport", "bucket"], how="left")
print(f"\ndeparture rows {len(departures):,}, matched to an arrival bucket "
      f"{departures.sek_anteil.notna().mean():.1%}")

report = {"arrival_rows": int(len(arrivals)), "departure_rows": int(len(departures)),
          "by_airport": by_airport.round(5).to_dict("index"), "measures": {}}
main = departures.loc[~departures.fallback & departures.rest.notna()]
fallback = departures.loc[departures.fallback]
for measure in ("sek_anteil", "kopie", "nm_fehlt"):
    valid = main.loc[main[measure].notna()]
    try:
        quintile = pd.qcut(valid[measure], 5, labels=False, duplicates="drop")
    except ValueError:
        print(f"\n{measure}: not enough distinct values to split")
        continue
    grouped = valid.groupby(quintile).agg(rows=("rest", "size"), level=(measure, "median"),
                                          rest_median=("rest", "median"),
                                          rest_rmse=("rest", lambda v: float(np.sqrt((v ** 2).mean()))),
                                          abs_rest_median=("rest", lambda v: float(v.abs().median())))
    print(f"\nmain lane, by quintile of {measure} (arrivals in the same airport and two-hour bucket):")
    print(grouped.round(3).to_string())
    fb = fallback.loc[fallback[measure].notna()]
    fq = pd.qcut(fb[measure], 5, labels=False, duplicates="drop")
    fgrouped = fb.groupby(fq).agg(rows=("y", "size"), level=(measure, "median"), copy_rate=("copy", "mean"),
                                  extreme_rate=("y", lambda v: float((v > 5400).mean())))
    print(f"fallback rows, by quintile of {measure}:")
    print(fgrouped.round(4).to_string())
    report["measures"][measure] = {"main": grouped.round(4).to_dict("index"),
                                   "fallback": fgrouped.round(4).to_dict("index")}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
print("\nwritten", OUTPUT)

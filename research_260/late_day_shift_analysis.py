"""Day-shift regime among departures delayed past midnight: where it occurs, how it separates from schedule copies, 2026 exposure.

Descriptive. No model fit, no leaderboard feedback. Definitions fixed 14.09.2026 after research_260/fallback_regime_census.py
found 14 day-shift fallback rows carrying 52.8 % of the v46 fallback squared error, 11 of them Rome departures whose movement
date differs from the scheduled date.

Mechanism under test: when a flight is delayed past midnight, the airport records the actual off-block clock time under the
SCHEDULED date (day shift, y about 86,400 s plus taxi) or, when no off-block is recorded, copies the schedule (copy, y = delay).
Both lanes are counted, because main-lane departures carry the same airport block time and would add examples.

Definitions (all 2025 departures with a label; y = TAXITIME_SEC_mvt, delay = MVT - SCHED in seconds; lane = main when AOBT_3 exists):
  late        MVT date differs from the scheduled date (tag_versch)
  copy        |y - delay| <= 60 s
  day_shift   not copy, 80,000 <= y <= 93,600 s
  normal      not copy, -600 <= y <= 5,400 s
  other       everything else
Tables: regime counts by lane, airport and late flag; for late Rome departures the regime by delay band (0-6, 6-12, 12-18, 18-24,
24+ h) and by movement hour band; v46 fallback squared error per cell from the six-pair held-out predictions; for main-lane
January/July rows the v45 candidate-stage holdout error of day-shift rows; 2026 ranking exposure of late Rome departures by lane,
delay band and hour band with the served fallback predictions (label-free). Row-level details are printed only.

Run from the repository root:  python3 -B research_260/late_day_shift_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt",
       "TAXITIME_SEC_mvt"]
DELAY_BANDS = [(-1e9, 6), (6, 12), (12, 18), (18, 24), (24, 1e9)]
HOUR_BANDS = [(0, 6), (6, 12), (12, 18), (18, 24)]
OUTPUT = ROOT / "research_260/results/late_day_shift_analysis.json"


def frame(paths, labelled=True):
    raw = pd.concat([pd.read_parquet(p, columns=[c for c in RAW if labelled or c != "TAXITIME_SEC_mvt"]) for p in paths], ignore_index=True)
    d = raw[raw.PHASE_mvt.eq("DEP")].reset_index(drop=True)
    t_mvt, t_sch = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True), pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True)
    d["delay"] = (t_mvt - t_sch).dt.total_seconds()
    d["delay_h"] = d.delay / 3600
    d["late"] = (t_mvt.dt.date != t_sch.dt.date).to_numpy()
    d["mvt_hour"] = t_mvt.dt.hour
    d["lane"] = np.where(d.AOBT_3_flt.notna(), "main", "fallback")
    d["airport"] = d.ADEP_mvt.astype(str)
    if labelled:
        y = d.TAXITIME_SEC_mvt.to_numpy(float)
        copy = np.abs(y - d.delay.to_numpy()) <= 60
        shift = ~copy & (y >= 80000) & (y <= 93600)
        normal = ~copy & (y >= -600) & (y <= 5400)
        d["regime"] = np.select([copy, shift, normal], ["copy", "day_shift", "normal"], "other")
        d = d[np.isfinite(y)].reset_index(drop=True)
    return d


def band(values, bands):
    out = np.full(len(values), "", dtype=object)
    for lo, hi in bands:
        label = f"{max(lo, 0):g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        out[(values >= lo) & (values < hi)] = label
    return out


def crosstab(d, row, col="regime"):
    t = pd.crosstab(d[row], d[col])
    return {str(i): {str(c): int(v) for c, v in r.items()} for i, r in t.iterrows()}


d = frame(sorted((ROOT / "data").glob("training_2025-*.parquet")))
report = {"note": "descriptive, all labelled 2025 departures; row-level details printed only", "rows": int(len(d))}
report["day_shift_rows_by_lane_airport_late"] = crosstab(d[d.regime.eq("day_shift")].assign(key=lambda x: x.lane + "|" + x.airport + "|late=" + x.late.astype(str)), "key", "lane")
late = d[d.late]
report["late_regimes_by_lane_airport"] = crosstab(late.assign(key=late.lane + "|" + late.airport), "key")
rome = late[late.airport.eq("LIRF")].copy()
rome["delay_band"], rome["hour_band"] = band(rome.delay_h.to_numpy(), DELAY_BANDS), band(rome.mvt_hour.to_numpy(float), HOUR_BANDS)
report["rome_late_by_lane_delay_band"] = crosstab(rome.assign(key=rome.lane + "|" + rome.delay_band), "key")
report["rome_late_by_lane_hour_band"] = crosstab(rome.assign(key=rome.lane + "|" + rome.hour_band), "key")
report["rome_late_by_delay_and_hour_band_both_lanes"] = crosstab(rome.assign(key=rome.delay_band + "|" + rome.hour_band), "key")
rome["long_flight_code"] = rome.FLIGHT_mvt.astype(str).str.len() >= 8
report["rome_late_long_flight_code_share"] = {r: float(g.long_flight_code.mean()) for r, g in rome.groupby("regime")}

# v46 fallback squared error per late-Rome cell (six-pair held-out predictions)
pred = pd.concat([pd.read_parquet(ROOT / f".cache/research_240/fallback_anchor_production/predictions_{p}.parquet", columns=["MVT_ID_mvt", "prod_anchor"])
                  for p in ["1_7", "2_8", "3_9", "4_10", "5_11", "6_12"]], ignore_index=True)
fb = d[d.lane.eq("fallback")].merge(pred, on="MVT_ID_mvt", validate="one_to_one")
fb["se"] = (fb.prod_anchor - fb.TAXITIME_SEC_mvt) ** 2
sse = fb.se.sum()
fr = fb[fb.late & fb.airport.eq("LIRF")].copy()
fr["delay_band"] = band(fr.delay_h.to_numpy(), DELAY_BANDS)
report["rome_late_fallback_se_share_by_delay_band_regime"] = {f"{b}|{r}": {"rows": int(len(g)), "se_share_of_fallback": float(g.se.sum() / sse),
                                                                            "mean_prediction_h": float(g.prod_anchor.mean() / 3600)}
                                                               for (b, r), g in fr.groupby(["delay_band", "regime"])}

# main-lane January/July holdout error on day-shift rows (v45 candidate-stage ensemble)
hold = pd.read_parquet(ROOT / ".cache/clock_queue_release_20260912/candidate/holdout_predictions.parquet", columns=["MVT_ID_mvt", "hat_aobt", "y", "prediction"])
hm = hold[hold.hat_aobt.eq(1)].merge(d[["MVT_ID_mvt", "regime", "airport", "late"]], on="MVT_ID_mvt", validate="one_to_one")
hse = (hm.prediction - hm.y) ** 2
report["main_holdout_jan_jul"] = {"rows": int(len(hm)), "rmse": float(np.sqrt(hse.mean())),
                                  "by_regime": {r: {"rows": int((hm.regime == r).sum()), "se_share": float(hse[hm.regime == r].sum() / hse.sum()),
                                                    "rmse": float(np.sqrt(hse[hm.regime == r].mean()))} for r in ("copy", "day_shift", "normal", "other")}}

# 2026 exposure, label-free
r26 = frame([ROOT / "data/ranking.parquet"], labelled=False)
r26 = r26[r26.late & r26.airport.eq("LIRF")].copy()
r26["delay_band"], r26["hour_band"] = band(r26.delay_h.to_numpy(), DELAY_BANDS), band(r26.mvt_hour.to_numpy(float), HOUR_BANDS)
served = pd.read_parquet(ROOT / ".cache/research_240/anchor_preview_2026_production.parquet", columns=["MVT_ID_mvt", "anchored", "q"])
r26 = r26.merge(served, on="MVT_ID_mvt", how="left")
report["rome_late_2026"] = {"by_lane_delay_band": {f"{lane}|{b}": int(n) for (lane, b), n in r26.groupby(["lane", "delay_band"]).size().items()},
                            "by_lane_hour_band": {f"{lane}|{b}": int(n) for (lane, b), n in r26.groupby(["lane", "hour_band"]).size().items()},
                            "fallback_served_prediction_h_by_delay_band": {b: {"rows": int(len(g)), "mean_served_h": float(g.anchored.mean() / 3600),
                                                                               "mean_q": float(g.q.mean()), "mean_delay_h": float(g.delay_h.mean())}
                                                                           for b, g in r26[r26.lane.eq("fallback")].groupby("delay_band")}}
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

print("day-shift rows by lane|airport|late:", report["day_shift_rows_by_lane_airport_late"])
print("\nlate departures, regimes by lane|airport:")
for k, v in report["late_regimes_by_lane_airport"].items():
    print(f"  {k:16s} {v}")
for name in ("rome_late_by_lane_delay_band", "rome_late_by_lane_hour_band", "rome_late_by_delay_and_hour_band_both_lanes"):
    print(f"\n{name}:")
    for k, v in report[name].items():
        print(f"  {k:16s} {v}")
print("\nlong flight code share by regime (late Rome):", report["rome_late_long_flight_code_share"])
print("\nlate Rome fallback, v46 SE share by delay band|regime:")
for k, v in report["rome_late_fallback_se_share_by_delay_band_regime"].items():
    print(f"  {k:16s} rows {v['rows']:3d}  se share {v['se_share_of_fallback']:.4f}  mean prediction {v['mean_prediction_h']:.2f} h")
print("\nmain holdout Jan/Jul:", json.dumps(report["main_holdout_jan_jul"]))
print("\n2026 late Rome:", json.dumps(report["rome_late_2026"], indent=1))
view = rome[rome.regime.isin(["day_shift"]) | (rome.lane.eq("fallback") & rome.delay_h.ge(12))][["lane", "FLIGHT_mvt", "delay_h", "mvt_hour", "regime", "TAXITIME_SEC_mvt"]]
print("\nlate Rome rows that are day shifts, or fallback rows delayed 12 h or more (printed only):")
print(view.sort_values(["lane", "delay_h"]).to_string(index=False))

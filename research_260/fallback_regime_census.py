"""Hidden block-time regimes of fallback departures (no AOBT_3): census, error shares, oracle bounds and 2026 exposure.

Descriptive. No model fit, no leaderboard feedback. Definitions fixed 14.09.2026 before the tables were looked at.

Hypothesis. Fallback departures carry no Network Manager flight record at all (EOBT_1, LOBT, IOBT and AOBT_3 are missing on
every fallback row, research_260/census_2025.json). The NM rules that derive an off-block time from a DPI/ADI event or from
take-off minus taxi time therefore never apply, and the airport's BLOCK_TIME is the only off-block clock. That clock is
produced by distinct hidden regimes: (a) an operational off-block event, (b) the schedule copied into the block time,
(c) an operational clock time stamped with the scheduled date (a day shift), (d) other errors. Each regime implies a
different taxi-out value, so for these rows the regime, not the feature set, bounds the error.

Inputs: six-pair held-out predictions of the scored v46 fallback recipe (research_240/fallback_anchor_production.py, column
prod_anchor, 22,470 rows), the held-out production tables (research_240/fallback_rescreen_production.py) and the raw 2025
movement files for the block, schedule and movement clocks. 2026 exposure, label-free: the cached ranking fallback table and
the v46 fallback predictions that v47 also carries (research_240/anchor_preview_2026_production.py).

Regimes (y = TAXITIME_SEC_mvt, delay = 60 * versp_min = MVT - SCHED):
  copy        |y - delay| <= 60 s
  day_shift   not copy and 80,000 <= y <= 93,600 s
  normal      not copy and -600 <= y <= 5,400 s
  negative    not copy and y < -600 s
  other       not copy and 5,400 < y < 80,000 s or y > 93,600 s
Oracle: copy -> delay; day_shift -> 86,400 s + the median implied taxi (y - 86,400) of all day-shift rows; normal, negative
and other keep the v46 prediction. Reported with each regime's share of the v46 fallback squared error, by airport and by
movement date versus scheduled date (tag_versch). Row-level details are printed, never written to the result file.

Run from the repository root:  python3 -B research_260/fallback_regime_census.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ["1_7", "2_8", "3_9", "4_10", "5_11", "6_12"]
PREDICTIONS = ROOT / ".cache/research_240/fallback_anchor_production"
TABLES = ROOT / ".cache/research_240/rescreen_production"
PREVIEW = ROOT / ".cache/research_240/anchor_preview"
RAW = ["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "FLIGHT_RULE_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
       "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "AOBT_3_flt", "FLIGHT_ID_mvt", "ADES_mvt"]
FEATURES = ["MVT_ID_mvt", "versp_min", "tag_versch", "sched_sek", "ruf_len", "id_abw", "auslast", "STAND_mvt"]
ROWS_2026, FALLBACK_2026 = 344841, 5290
OUTPUT = ROOT / "research_260/results/fallback_regime_census.json"


def clock(t):
    return (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy(float)


def circular_close(a, b, tolerance=60.):
    d = np.abs(a - b) % 86400
    return np.minimum(d, 86400 - d) <= tolerance


def share(mask):
    return float(np.mean(mask)) if len(mask) else None


# ---- 2025 held-out fallback rows with v46 predictions ----
pred = pd.concat([pd.read_parquet(PREDICTIONS / f"predictions_{p}.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "y", "nt_anchor", "prod_anchor", "fold"])
                  for p in PAIRS], ignore_index=True)
index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet") for s in TABLES.glob("rescreen_*.json")}
tables = pd.concat([pd.read_parquet(index[tuple(sorted(map(int, p.split("_"))))], columns=FEATURES) for p in PAIRS], ignore_index=True)
raw = pd.concat([pd.read_parquet(p, columns=RAW) for p in sorted((ROOT / "data").glob("training_2025-*.parquet"))], ignore_index=True)
raw = raw[raw.PHASE_mvt.eq("DEP") & raw.MVT_ID_mvt.isin(pred.MVT_ID_mvt)]
d = pred.merge(tables, on="MVT_ID_mvt", validate="one_to_one").merge(raw, on="MVT_ID_mvt", validate="one_to_one")
assert len(d) == len(pred) == 22470
y, delay, v46 = d.y.to_numpy(float), 60 * d.versp_min.to_numpy(float), d.prod_anchor.to_numpy(float)
copy = np.abs(y - delay) <= 60
shift = ~copy & (y >= 80000) & (y <= 93600)
normal = ~copy & (y >= -600) & (y <= 5400)
negative = ~copy & (y < -600)
other = ~copy & ~shift & ~normal & ~negative
regimes = {"copy": copy, "day_shift": shift, "normal": normal, "negative": negative, "other": other}
se = (v46 - y) ** 2
sse = se.sum()
t_mvt, t_blk, t_sch = (pd.to_datetime(d[c], utc=True) for c in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"))
blk_date_is_sched = (t_blk.dt.date == t_sch.dt.date).to_numpy()
blk_date_is_mvt = (t_blk.dt.date == t_mvt.dt.date).to_numpy()
blk_clock_is_sched = circular_close(clock(t_blk), clock(t_sch))
blk_zero, sch_zero, mvt_zero = (t.dt.second.eq(0).to_numpy() for t in (t_blk, t_sch, t_mvt))
nm_any = (d.EOBT_1_flt.notna() | d.AOBT_3_flt.notna()).to_numpy()
callsign = d.CALLSIGN_flt.notna().to_numpy()
tag = d.tag_versch.to_numpy() == 1
airports = d.ADEP_mvt.astype(str).to_numpy()
rule = d.FLIGHT_RULE_mvt.astype(str).to_numpy()


def summary(mask):
    n = int(mask.sum())
    if not n:
        return {"rows": 0}
    return {"rows": n, "row_share": n / len(d), "v46_se_share": float(se[mask].sum() / sse), "v46_rmse": float(np.sqrt(se[mask].mean())),
            "median_y_s": float(np.median(y[mask])), "median_delay_h": float(np.median(delay[mask]) / 3600),
            "tag_versch_share": share(tag[mask]), "block_date_equals_sched_date": share(blk_date_is_sched[mask]),
            "block_date_equals_mvt_date": share(blk_date_is_mvt[mask]), "block_clock_equals_sched_clock": share(blk_clock_is_sched[mask]),
            "block_seconds_zero": share(blk_zero[mask]), "sched_seconds_zero": share(sch_zero[mask]), "mvt_seconds_zero": share(mvt_zero[mask]),
            "nm_times_present": share(nm_any[mask]), "callsign_present": share(callsign[mask]),
            "flight_rule": {k: int(v) for k, v in pd.Series(rule[mask]).value_counts().head(4).items()},
            "airports": {k: int(v) for k, v in pd.Series(airports[mask]).value_counts().head(6).items()}}


report = {"note": "descriptive; six-pair held-out rows of the v46 fallback recipe; row-level details are printed only",
          "rows": len(d), "v46_fallback_rmse": float(np.sqrt(se.mean())),
          "nm_times_present_on_any_fallback_row": bool(nm_any.any()),
          "regimes": {name: summary(mask) for name, mask in regimes.items()},
          "by_tag_versch": {str(t): {name: summary(mask & (tag == t)) for name, mask in regimes.items()} for t in (False, True)},
          "by_airport": {a: {name: {"rows": int((mask & (airports == a)).sum()), "se_share": float(se[mask & (airports == a)].sum() / sse)}
                             for name, mask in regimes.items()} for a in sorted(set(airports))}}
# other extremes: offsets relative to the schedule
off = y - delay
report["other_offsets"] = {"rows": int(other.sum()),
                           "y_minus_delay_quantiles_h": {q: float(np.quantile(off[other], q) / 3600) for q in (.05, .25, .5, .75, .95)} if other.any() else None,
                           "schedule_copy_with_day_offset": int((other & (np.minimum(np.abs(off - 86400), np.abs(off + 86400)) <= 60)).sum()),
                           "schedule_copy_with_hour_offset": {str(k): int((other & (np.abs(off - k * 3600) <= 60)).sum()) for k in (-3, -2, -1, 1, 2, 3)},
                           "block_clock_equals_sched_clock": int((other & blk_clock_is_sched).sum()),
                           "block_seconds_zero": int((other & blk_zero).sum())}
implied_taxi = np.median(y[shift] - 86400) if shift.any() else 900.
report["day_shift_implied_taxi_s"] = {"median": float(implied_taxi),
                                      "quantiles": {q: float(np.quantile(y[shift] - 86400, q)) for q in (.1, .5, .9)} if shift.any() else None}
oracle = v46.copy()
oracle[copy] = delay[copy]
oracle[shift] = 86400 + implied_taxi
gain_per_row = {name: float(((v46 - y) ** 2 - (np.where(mask, oracle, v46) - y) ** 2).mean()) for name, mask in (("copy", copy), ("day_shift", shift))}
report["oracle"] = {"v46_fallback_rmse": float(np.sqrt(se.mean())), "oracle_fallback_rmse": float(np.sqrt(((oracle - y) ** 2).mean())),
                    "fallback_mse_gain_per_row": gain_per_row,
                    "leaderboard_mse_equivalent_at_2026_fallback_share": {k: v * FALLBACK_2026 / ROWS_2026 for k, v in gain_per_row.items()}}

# ---- label-free separability: day shifts against copies among delayed-past-midnight rows ----
late = tag & (copy | shift)
fields = {"delay_h": delay / 3600, "mvt_hour": t_mvt.dt.hour.to_numpy(float), "sched_hour": t_sch.dt.hour.to_numpy(float),
          "sched_seconds_zero": sch_zero.astype(float), "mvt_seconds_zero": mvt_zero.astype(float), "ruf_len": d.ruf_len.to_numpy(float),
          "id_abw": d.id_abw.to_numpy(float), "callsign_present": callsign.astype(float), "ifr": (rule == "I").astype(float)}
report["late_rows_shift_vs_copy"] = {"rows_copy": int((late & copy).sum()), "rows_shift": int((late & shift).sum()),
                                     "medians": {k: {"copy": float(np.nanmedian(v[late & copy])) if (late & copy).any() else None,
                                                     "day_shift": float(np.nanmedian(v[late & shift])) if (late & shift).any() else None}
                                                 for k, v in fields.items()}}

# ---- 2026 exposure, label-free ----
rank = pd.read_parquet(next(PREVIEW.glob("fallback_2026_ranking_*.parquet")), columns=["MVT_ID_mvt", "ADEP_mvt", "versp_min", "tag_versch"])
served = pd.read_parquet(ROOT / ".cache/research_240/anchor_preview_2026_production.parquet", columns=["MVT_ID_mvt", "anchored", "q"])
r = rank.merge(served, on="MVT_ID_mvt", validate="one_to_one")
assert len(r) == FALLBACK_2026
r_tag = r.tag_versch.to_numpy() == 1
r_air = r.ADEP_mvt.astype(str).to_numpy()
expected_shift = 0.
per_airport = {}
for a in sorted(set(airports) | set(r_air)):
    base = tag & (airports == a)
    rate = float(shift[base].mean()) if base.any() else 0.
    n26 = int((r_tag & (r_air == a)).sum())
    expected_shift += rate * n26
    per_airport[a] = {"2025_tag_versch_rows": int(base.sum()), "2025_day_shift_rate_among_them": rate, "2026_tag_versch_rows": n26,
                      "2025_tag_versch_share_of_fallback": float(tag[airports == a].mean()) if (airports == a).any() else None,
                      "2026_tag_versch_share_of_fallback": float(r_tag[r_air == a].mean()) if (r_air == a).any() else None}
mean_se_shift = float(se[shift].mean()) if shift.any() else 0.
report["exposure_2026"] = {"fallback_rows": FALLBACK_2026, "tag_versch_rows": int(r_tag.sum()), "by_airport": per_airport,
                           "expected_day_shift_rows": expected_shift,
                           "v46_mean_squared_error_on_2025_day_shift_rows": mean_se_shift,
                           "expected_leaderboard_mse_at_stake": expected_shift * mean_se_shift / ROWS_2026,
                           "served_prediction_on_2026_tag_versch_rows_h": {q: float(np.quantile(r.anchored.to_numpy(float)[r_tag], q) / 3600)
                                                                           for q in (.1, .5, .9)} if r_tag.any() else None,
                           "delay_on_2026_tag_versch_rows_h": {q: float(np.quantile(60 * r.versp_min.to_numpy(float)[r_tag], q) / 3600)
                                                               for q in (.1, .5, .9)} if r_tag.any() else None}
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

pd.set_option("display.width", 220)
print(f"rows {len(d)}, v46 fallback RMSE {np.sqrt(se.mean()):.1f}, any NM time on a fallback row: {nm_any.any()}")
table = pd.DataFrame({k: {c: v.get(c) for c in ("rows", "v46_se_share", "v46_rmse", "median_y_s", "median_delay_h", "tag_versch_share",
                                                "block_date_equals_sched_date", "block_clock_equals_sched_clock", "block_seconds_zero",
                                                "sched_seconds_zero", "callsign_present")} for k, v in report["regimes"].items()}).T
print(table.round(3).to_string())
print("by tag_versch (rows / se share):", {t: {k: (v["rows"], round(v.get("v46_se_share", 0), 3)) for k, v in g.items()} for t, g in report["by_tag_versch"].items()})
print("other offsets:", json.dumps(report["other_offsets"]))
print("day shift implied taxi:", report["day_shift_implied_taxi_s"])
print("oracle:", json.dumps(report["oracle"]))
print("late rows shift vs copy:", json.dumps(report["late_rows_shift_vs_copy"]))
print("2026 exposure:", json.dumps({k: v for k, v in report["exposure_2026"].items() if k != "by_airport"}))
print("2026 by airport:", {a: (v["2025_tag_versch_rows"], round(v["2025_day_shift_rate_among_them"], 3), v["2026_tag_versch_rows"]) for a, v in per_airport.items()})
cols = ["fold", "ADEP_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "tag_versch", "versp_min", "y", "prod_anchor"]
print("\nday-shift rows (printed only):")
view = d.loc[shift, cols].copy()
view["delay_h"], view["mvt_clock"], view["sched_clock"], view["block_clock"] = view.versp_min / 60, t_mvt[shift].dt.strftime("%H:%M:%S"), t_sch[shift].dt.strftime("%m-%d %H:%M:%S"), t_blk[shift].dt.strftime("%m-%d %H:%M:%S")
print(view.drop(columns="versp_min").to_string(index=False))
print("\nlargest other rows by v46 squared error (printed only):")
view = d.loc[other, cols].assign(se=se[other]).sort_values("se", ascending=False).head(15)
print(view.to_string(index=False))

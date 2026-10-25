"""Where does the v46-style holdout error sit, and is there month-level drift to correct?

No training, no leaderboard feedback, no 2026 label.

1. Error map on January/July 2025 (learning months 2-6 and 8-12): main-lane predictions of the
   complete v45-recipe ensemble (release candidate stage) and fallback predictions of the scored v46
   fallback recipe (prod_anchor from research_240/fallback_anchor_production.py; its prod_v44 column
   must reproduce the release candidate's fallback predictions). Squared error as total MSE over all
   holdout rows, by class: main copied (|y - 60*versp| <= 10 s), main over one hour and not copied,
   main ordinary, fallback extreme (y > 5,400 s), fallback normal; and by airport and class.
2. Month-level bias of the main lane: squared error removable by subtracting each airport-month's mean
   residual (in-sample, and cross-fitted by day parity) and each airport-day's mean (in-sample, and
   cross-fitted by row parity within the day).
3. 2025 versus 2026, label-free, per airport for January and July: departures' AOBT_3 share, median
   MVT - AOBT_3 and median delay MVT - SCHED; arrivals' median ARVT_3 - MVT_TIME, median taxi-in
   BLOCK - MVT_TIME and copied-in-block share.
4. Can arrivals track departure reporting drift? Over the 120 airport-months of 2025: correlation of
   each month's deviation from the airport's yearly level, departure AOBT_3 - BLOCK median against
   arrival ARVT_3 - MVT median, arrival taxi-in median and departure MVT - AOBT_3 median.

Run from the repository root:
  python3 research_240/error_and_drift_diagnostics.py
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
FILE = re.compile(r"^training_2025-(\d{2})-01_\d{4}-\d{2}-01\.parquet$")
RAW = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
       "AOBT_3_flt", "ARVT_3_flt"]


def seconds(a, b):
    return (pd.to_datetime(a, utc=True, errors="coerce") - pd.to_datetime(b, utc=True, errors="coerce")).dt.total_seconds().to_numpy()


def median(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None


def bounds(resid, keys, split, n_total):
    frame = pd.DataFrame({"r": resid, "k": keys, "s": split})
    groups = frame.groupby("k").r
    in_sample = float((groups.size() * groups.mean() ** 2).sum() / n_total)
    removed = 0.0
    for part in (0, 1):
        means = frame[frame.s != part].groupby("k").r.mean()
        test = frame[frame.s == part]
        shift = test.k.map(means).fillna(0.0).to_numpy()
        removed += float((test.r.to_numpy() ** 2 - (test.r.to_numpy() - shift) ** 2).sum())
    return {"in_sample_total_mse": in_sample, "cross_fit_total_mse": removed / n_total, "bins": int(groups.ngroups)}


def movement_frame(paths):
    d = pd.concat([pd.read_parquet(p, columns=RAW) for p in paths], ignore_index=True)
    phase = d.PHASE_mvt.astype(str).str.upper().str[:3].to_numpy()
    return {"phase": phase, "month": pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy(),
            "dep_airport": d.ADEP_mvt.astype(str).to_numpy(), "arr_airport": d.ADES_mvt.astype(str).to_numpy(),
            "ersatz": seconds(d.MVT_TIME_UTC_mvt, d.AOBT_3_flt), "delay_min": seconds(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt) / 60,
            "rest": seconds(d.AOBT_3_flt, d.BLOCK_TIME_UTC_mvt), "arr_nm": seconds(d.ARVT_3_flt, d.MVT_TIME_UTC_mvt),
            "taxi_in": seconds(d.BLOCK_TIME_UTC_mvt, d.MVT_TIME_UTC_mvt),
            "arr_copied": np.abs(seconds(d.BLOCK_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)) <= 10}


def month_stats(m, airport, month):
    dep = (m["phase"] == "DEP") & (m["dep_airport"] == airport) & (m["month"] == month)
    arr = (m["phase"] == "ARR") & (m["arr_airport"] == airport) & (m["month"] == month)
    return {"departures": int(dep.sum()), "aobt3_share": float(np.isfinite(m["ersatz"][dep]).mean()) if dep.any() else None,
            "median_mvt_minus_aobt3_s": median(m["ersatz"][dep]), "median_delay_min": median(m["delay_min"][dep]),
            "median_aobt3_minus_block_s": median(m["rest"][dep]),
            "arrivals": int(arr.sum()), "median_arvt3_minus_mvt_s": median(m["arr_nm"][arr]),
            "median_taxi_in_s": median(m["taxi_in"][arr]),
            "arrival_copied_share": float(m["arr_copied"][arr].mean()) if arr.any() else None}


def correlation(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return None
    ranks = lambda x: pd.Series(x).rank().to_numpy()
    return {"points": int(ok.sum()), "pearson": float(np.corrcoef(a[ok], b[ok])[0, 1]),
            "spearman": float(np.corrcoef(ranks(a[ok]), ranks(b[ok]))[0, 1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path(".cache/clock_queue_release_20260912"))
    parser.add_argument("--anchor", type=Path, default=Path(".cache/research_240/fallback_anchor_production/predictions_1_7.parquet"))
    parser.add_argument("--output", type=Path, default=Path("research_240/results/error_and_drift_diagnostics.json"))
    args = parser.parse_args()
    report = {"note": "no training; January/July 2025 holdout of the v45 recipe with the v46 fallback; 2026 label-free"}

    pred = pd.read_parquet(args.release / "candidate/holdout_predictions.parquet")
    table = pd.read_parquet(args.release / "data/months_1_7.parquet", columns=["MVT_ID_mvt", "versp_min"])
    pred = pred.merge(table, on="MVT_ID_mvt", how="left", validate="one_to_one")
    anchor = pd.read_parquet(args.anchor, columns=["MVT_ID_mvt", "prod_v44", "prod_anchor"]).set_index("MVT_ID_mvt")
    fallback = pred.hat_aobt.eq(0).to_numpy()
    ids = pred.loc[fallback, "MVT_ID_mvt"].to_numpy()
    assert set(ids) == set(anchor.index)
    report["fallback_replication_max_abs_difference_s"] = float(np.abs(anchor.loc[ids, "prod_v44"].to_numpy()
                                                                        - pred.loc[fallback, "prediction"].to_numpy()).max())
    v46 = pred.prediction.to_numpy(float).copy()
    v46[fallback] = anchor.loc[ids, "prod_anchor"].to_numpy()
    y = pred.y.to_numpy(float)
    n_all = len(pred)
    e45, e46 = (pred.prediction.to_numpy(float) - y) ** 2, (v46 - y) ** 2
    copied = np.abs(y - pred.versp_min.to_numpy(float) * 60) <= 10
    cls = np.where(fallback, np.where(y > 5400, "fallback_extreme", "fallback_normal"),
                   np.where(copied, "main_copied", np.where(y > 3600, "main_over_1h_not_copied", "main_ordinary")))
    airports = pred.ADEP_mvt.astype(str).to_numpy()
    report["error_map"] = {"rmse_v45": float(np.sqrt(e45.mean())), "rmse_v46_style": float(np.sqrt(e46.mean())),
                           "total_mse_v46_style": float(e46.mean()),
                           "classes": {c: {"rows": int((cls == c).sum()), "total_mse": float(e46[cls == c].sum() / n_all),
                                           "share_of_squared_error": float(e46[cls == c].sum() / e46.sum()),
                                           "rmse": float(np.sqrt(e46[cls == c].mean()))} for c in sorted(set(cls))}}
    cells = pd.DataFrame({"airport": airports, "cls": cls, "e": e46}).groupby(["airport", "cls"]).e.agg(["size", "sum"]).reset_index()
    cells["share"] = cells["sum"] / e46.sum()
    report["error_map"]["top_airport_classes"] = [
        {"airport": r.airport, "class": r.cls, "rows": int(r.size), "total_mse": float(r.sum / n_all), "share": float(r.share)}
        for r in cells.sort_values("sum", ascending=False).head(15).itertuples()]
    ordinary = cls == "main_ordinary"
    abs_error = np.sqrt(e46)
    report["error_map"]["main_ordinary_share_from_errors_over_30_min"] = float(e46[ordinary & (abs_error > 1800)].sum() / e46[ordinary].sum())

    main_rows = ~fallback
    resid = y[main_rows] - pred.prediction.to_numpy(float)[main_rows]
    stamps = pd.to_datetime(pred.MVT_TIME_UTC_mvt, utc=True)
    a = pd.Series(airports[main_rows])
    month_key = (a + "|" + pd.Series(stamps.dt.month.to_numpy()[main_rows]).astype(str)).to_numpy()
    day_key = (a + "|" + pd.Series(stamps.dt.strftime("%m-%d").to_numpy()[main_rows])).to_numpy()
    day_parity = stamps.dt.dayofyear.to_numpy()[main_rows] % 2
    row_parity = np.arange(main_rows.sum()) % 2
    monthly = pd.DataFrame({"k": month_key, "r": resid}).groupby("k").r.agg(["size", "mean"])
    report["main_bias"] = {"airport_month": bounds(resid, month_key, day_parity, n_all),
                           "airport_day": bounds(resid, day_key, row_parity, n_all),
                           "airport_month_mean_residual_s": {k: {"rows": int(v["size"]), "mean": float(v["mean"])}
                                                             for k, v in monthly.iterrows()}}

    paths = {int(FILE.match(p.name).group(1)): p for p in Path("data").glob("training_2025-*.parquet") if FILE.match(p.name)}
    assert sorted(paths) == list(range(1, 13))
    m2025 = movement_frame([paths[1], paths[7]])
    m2026 = movement_frame([Path("data/ranking.parquet")])
    drift = {}
    for airport in AIRPORTS:
        for month in (1, 7):
            before, after = month_stats(m2025, airport, month), month_stats(m2026, airport, month)
            delta = {k: (after[k] - before[k]) for k in before if isinstance(before[k], float) and isinstance(after[k], float)}
            drift[f"{airport}|{month}"] = {"2025": before, "2026": after, "change": delta}
    report["label_free_2025_vs_2026"] = drift

    year = movement_frame([paths[m] for m in range(1, 13)])
    rows = []
    for airport in AIRPORTS:
        stats = [month_stats(year, airport, month) for month in range(1, 13)]
        frame = pd.DataFrame(stats)
        for col in ("median_aobt3_minus_block_s", "median_arvt3_minus_mvt_s", "median_taxi_in_s", "median_mvt_minus_aobt3_s"):
            frame[col + "_dev"] = frame[col] - frame[col].median()
        frame["airport"] = airport
        rows.append(frame)
    months = pd.concat(rows, ignore_index=True)
    dep_dev = months["median_aobt3_minus_block_s_dev"].to_numpy(float)
    report["arrival_tracks_departure_drift_2025"] = {
        "monthly_departure_rest_median_spread_s": {a: float(g["median_aobt3_minus_block_s"].std()) for a, g in months.groupby("airport")},
        "vs_arrival_arvt3_minus_mvt": correlation(dep_dev, months["median_arvt3_minus_mvt_s_dev"].to_numpy(float)),
        "vs_arrival_taxi_in": correlation(dep_dev, months["median_taxi_in_s_dev"].to_numpy(float)),
        "vs_departure_mvt_minus_aobt3": correlation(dep_dev, months["median_mvt_minus_aobt3_s_dev"].to_numpy(float))}

    # 5. Day-level bias proxies: can the airport-day mean main residual be predicted label-free?
    raw = pd.concat([pd.read_parquet(paths[m], columns=RAW) for m in (1, 7)], ignore_index=True)
    phase = raw.PHASE_mvt.astype(str).str.upper().str[:3].to_numpy()
    day_text = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    dep = pd.DataFrame({"key": raw.ADEP_mvt.astype(str).to_numpy() + "|" + day_text,
                        "ersatz": seconds(raw.MVT_TIME_UTC_mvt, raw.AOBT_3_flt),
                        "delay": seconds(raw.MVT_TIME_UTC_mvt, raw.SCHED_TIME_UTC_mvt) / 60})[phase == "DEP"]
    arr = pd.DataFrame({"key": raw.ADES_mvt.astype(str).to_numpy() + "|" + day_text,
                        "copied": (np.abs(seconds(raw.BLOCK_TIME_UTC_mvt, raw.SCHED_TIME_UTC_mvt)) <= 10).astype(float),
                        "taxi_in": seconds(raw.BLOCK_TIME_UTC_mvt, raw.MVT_TIME_UTC_mvt),
                        "arr_nm": seconds(raw.ARVT_3_flt, raw.MVT_TIME_UTC_mvt)})[phase == "ARR"]
    aggregates = pd.DataFrame({"dep_n": dep.groupby("key").size(),
                               "fallback_share": dep.groupby("key").ersatz.apply(lambda s: s.isna().mean()),
                               "ersatz_med": dep.groupby("key").ersatz.median(), "delay_med": dep.groupby("key").delay.median(),
                               "arr_copied": arr.groupby("key").copied.mean(), "taxi_in_med": arr.groupby("key").taxi_in.median(),
                               "arr_nm_med": arr.groupby("key").arr_nm.median()})
    main_keys = (a + "|" + pd.Series(stamps.dt.strftime("%Y-%m-%d").to_numpy()[main_rows])).to_numpy()
    days = pd.DataFrame({"key": main_keys, "r": resid}).groupby("key").r.agg(n="size", bias="mean").join(aggregates, how="left")
    days["airport"] = days.index.str[:4]
    proxies = list(aggregates.columns)
    for col in proxies:
        days[col + "_dev"] = days[col] - days.groupby("airport")[col].transform("median")
    correlations = {col: correlation(days[col + "_dev"].to_numpy(float), days.bias.to_numpy(float)) for col in proxies}
    design = np.column_stack([np.ones(len(days)), days[[c + "_dev" for c in proxies]].fillna(0.0).to_numpy()])
    parity = (pd.to_datetime(days.index.str[5:]).dayofyear % 2).to_numpy()
    by_key = pd.DataFrame({"key": main_keys, "r": resid})
    removed = 0.0
    for part in (0, 1):
        fit_rows = parity != part
        weights = np.sqrt(days.n.to_numpy(float)[fit_rows])
        beta = np.linalg.lstsq(design[fit_rows] * weights[:, None], days.bias.to_numpy(float)[fit_rows] * weights, rcond=None)[0]
        shift = pd.Series(design[~fit_rows] @ beta, index=days.index[~fit_rows])
        test = by_key[by_key.key.isin(shift.index)]
        applied = test.key.map(shift).to_numpy()
        removed += float((test.r.to_numpy() ** 2 - (test.r.to_numpy() - applied) ** 2).sum())
    report["day_bias_proxies"] = {"airport_days": int(len(days)),
                                  "row_weighted_day_bias_sd_s": float(np.sqrt(np.average(days.bias ** 2, weights=days.n))),
                                  "correlations_with_day_bias": correlations,
                                  "cross_fit_linear_removal_total_mse": removed / n_all}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    em = report["error_map"]
    print(f"holdout RMSE v45 {em['rmse_v45']:.3f}, v46-style {em['rmse_v46_style']:.3f}; fallback replication {report['fallback_replication_max_abs_difference_s']:.2e}")
    for c, v in em["classes"].items():
        print(f"  {c:26s} rows {v['rows']:>7,}  total MSE {v['total_mse']:>9.1f}  share {v['share_of_squared_error']:.3f}  RMSE {v['rmse']:.0f}")
    print("  top airport-class cells:", [(t["airport"], t["class"], round(t["share"], 3)) for t in em["top_airport_classes"][:8]])
    print("  main ordinary share from |error| > 30 min:", round(em["main_ordinary_share_from_errors_over_30_min"], 3))
    print("main bias:", {k: v for k, v in report["main_bias"].items() if k != "airport_month_mean_residual_s"})
    print("largest airport-month mean residuals:", sorted(((k, round(v["mean"], 1)) for k, v in report["main_bias"]["airport_month_mean_residual_s"].items()), key=lambda t: -abs(t[1]))[:6])
    print("arrival tracks departure drift:", json.dumps(report["arrival_tracks_departure_drift_2025"]))
    big = {k: {kk: round(vv, 1) for kk, vv in v["change"].items() if kk in ("median_mvt_minus_aobt3_s", "median_arvt3_minus_mvt_s", "median_taxi_in_s", "aobt3_share", "arrival_copied_share")}
           for k, v in drift.items()}
    print("2026 minus 2025 (label-free):")
    for k, v in big.items():
        print("  ", k, v)


if __name__ == "__main__":
    main()

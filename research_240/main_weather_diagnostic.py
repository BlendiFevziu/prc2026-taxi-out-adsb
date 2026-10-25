"""Can weather or operating direction remove squared error from v45's saved holdout predictions?

Uses the complete v45-recipe ensemble predictions for January and July 2025 saved by the release
gate (.cache/clock_queue_release_20260912/candidate/holdout_predictions.parquet), trained without
those months. Nothing is trained. Each row gets the METAR weather at or before its runway time
and the operating direction from the same months' arrivals (research_240/features_weather.py).

Bound: subtract the mean residual of each bin and report the squared error removed, as total MSE
over all holdout rows (the units of the 13 September de-icing check). Bins: airport x weather
state (temperature band, frozen or freezing precipitation, icing risk), airport x direction, and
their combination (descriptive). In-sample bin means are an optimistic upper bound; the cross-fit
version estimates bin means on odd days and applies them to even days, and vice versa.

Pre-registered decision, 13.09.2026, before running: the main lane is pursued only if the
in-sample bound for weather states or for directions reaches 500 total MSE (about 0.9 s at the
official level). Otherwise weather is tested only in the fallback lane.

Run from the repository root after research_240/fetch_metar.py:
  python3 research_240/main_weather_diagnostic.py
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_240 import features_weather as W

THRESHOLD = 500.0
FILE = re.compile(r"^training_2025-(\d{2})-01_\d{4}-\d{2}-01\.parquet$")


def state_key(wx):
    temp = wx.wx_temp_c.to_numpy(float)
    band = np.where(np.isnan(temp), 4, np.digitize(np.nan_to_num(temp), [0.0, 3.0, 10.0]))
    precip = ((wx.wx_frozen.to_numpy(float) == 1) | (wx.wx_freezing.to_numpy(float) == 1)).astype(int)
    icing = np.nan_to_num(wx.wx_icing_risk.to_numpy(float), nan=2).astype(int)
    return (pd.Series(band).astype(str) + "|" + pd.Series(precip).astype(str) + "|" + pd.Series(icing).astype(str)).to_numpy()


def bounds(resid, keys, parity, n_total):
    frame = pd.DataFrame({"r": resid, "k": keys, "p": parity})
    groups = frame.groupby("k").r
    in_sample = float((groups.size() * groups.mean() ** 2).sum() / n_total)
    removed = 0.0
    for part in (0, 1):
        means = frame[frame.p != part].groupby("k").r.mean()
        test = frame[frame.p == part]
        shift = test.k.map(means).fillna(0.0).to_numpy()
        removed += float((test.r.to_numpy() ** 2 - (test.r.to_numpy() - shift) ** 2).sum())
    return {"in_sample_total_mse": in_sample, "cross_fit_total_mse": removed / n_total, "bins": int(groups.ngroups)}


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return None
    return float(np.corrcoef(pd.Series(a[m]).rank().to_numpy(), pd.Series(b[m]).rank().to_numpy())[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", type=Path, default=Path(".cache/research_240/weather"))
    parser.add_argument("--predictions", type=Path,
                        default=Path(".cache/clock_queue_release_20260912/candidate/holdout_predictions.parquet"))
    parser.add_argument("--output", type=Path, default=Path("research_240/results/main_weather_diagnostic.json"))
    args = parser.parse_args()
    pred = pd.read_parquet(args.predictions)
    resid = pred.y.to_numpy(float) - pred.prediction.to_numpy(float)
    n_total = len(pred)
    airports = pred.ADEP_mvt.astype(str).to_numpy()
    stamps = W.utc_ns(pred.MVT_TIME_UTC_mvt)
    months = sorted(int(m) for m in stamps.dt.month.unique())
    paths = [p for p in sorted(Path("data").glob("training_2025-*.parquet")) if FILE.match(p.name) and int(FILE.match(p.name).group(1)) in months]
    assert len(paths) == len(months), (months, paths)
    raw = pd.concat([pd.read_parquet(p, columns=["PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]) for p in paths], ignore_index=True)
    arrivals = raw[raw.PHASE_mvt.astype(str).str.upper().str.startswith("ARR").to_numpy()]
    reports = W.report_features(W.load_metar(args.weather))
    wx = W.weather_for(airports, pred.MVT_TIME_UTC_mvt, reports)
    od = W.direction_for(airports, pred.MVT_TIME_UTC_mvt, W.operating_direction(arrivals), wx.wind_dir_deg, wx.wx_wind_kt)
    parity = stamps.dt.dayofyear.to_numpy() % 2
    weather_key = (pd.Series(airports) + "|" + pd.Series(state_key(wx))).to_numpy()
    direction_key = (pd.Series(airports) + "|" + od.od_dir.fillna(-1).astype(int).astype(str)).to_numpy()
    joint_key = (pd.Series(weather_key) + "|" + pd.Series(direction_key)).to_numpy()

    report = {"note": "v45-recipe ensemble holdout predictions for January and July 2025; residual = y - prediction; "
                      "bounds in total MSE over all holdout rows; no training",
              "months": months, "rows": n_total, "overall_mse": float(np.mean(resid ** 2)),
              "threshold_total_mse": THRESHOLD, "lanes": {}}
    for lane, mask in (("main", pred.hat_aobt.eq(1).to_numpy()), ("fallback", pred.hat_aobt.eq(0).to_numpy())):
        r = resid[mask]
        entry = {"rows": int(mask.sum()), "squared_error_share_total_mse": float((r ** 2).sum() / n_total),
                 "weather_coverage": float(np.isfinite(wx.wx_temp_c.to_numpy(float)[mask]).mean()),
                 "direction_coverage": float(np.isfinite(od.od_dir.to_numpy(float)[mask]).mean()),
                 "bound_weather_state": bounds(r, weather_key[mask], parity[mask], n_total),
                 "bound_direction": bounds(r, direction_key[mask], parity[mask], n_total),
                 "bound_weather_and_direction_descriptive": bounds(r, joint_key[mask], parity[mask], n_total),
                 "spearman_residual": {c: spearman(r, wx[c].to_numpy(float)[mask]) for c in W.WEATHER}}
        entry["spearman_residual"]["od_headwind_kt"] = spearman(r, od.od_headwind_kt.to_numpy(float)[mask])
        report["lanes"][lane] = entry
    main_lane = report["lanes"]["main"]
    best = max(main_lane["bound_weather_state"]["in_sample_total_mse"], main_lane["bound_direction"]["in_sample_total_mse"])
    report["decision"] = {"main_lane_best_in_sample_bound": best, "pursue_main_lane": bool(best >= THRESHOLD)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"rows": n_total, "overall_mse": report["overall_mse"], "decision": report["decision"],
                      "lanes": {k: {kk: vv for kk, vv in v.items() if kk != "spearman_residual"} for k, v in report["lanes"].items()}}, indent=2))
    print("spearman (main):", {k: (None if v is None else round(v, 4)) for k, v in main_lane["spearman_residual"].items()})


if __name__ == "__main__":
    main()

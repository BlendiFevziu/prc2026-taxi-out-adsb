"""Where the main ensemble errs relative to the stand-runway reference. Descriptive, January/July holdout.

Reads the v45 candidate-stage holdout predictions (complete ensemble trained without January/July) and the
release-cache feature tables. References from the learning months only, as in fallback_reference_screen.py.
No model fit, no leaderboard feedback. Bin-bias bound: the main-lane MSE removable by correcting the mean
residual of each ex_p10 bin (and of each airport x bin cell), in-sample and cross-fitted between January and July.
For scale: 0.3 s of main RMSE at 216.77 is about 130 MSE.

Run:  python3 -B research_240/main_residual_by_reference.py <output.json>
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REL = PROJECT / ".cache/clock_queue_release_20260912"
LEVELS = (("stand", ["airport", "stand", "runway"], 2), ("runway", ["airport", "runway"], 1), ("airport", ["airport"], 0))
EDGES = [-np.inf, -600, -300, -120, 0, 120, 300, 600, 1200, np.inf]


def text(values):
    s = pd.Series(values).reset_index(drop=True)
    return np.where(s.notna().to_numpy(), s.astype(str).to_numpy(), "").astype(object)


def reference_tables(frame):
    p = frame.ersatz_s.to_numpy(float)
    keep = frame.hat_aobt.eq(1).to_numpy() & (p > 0) & (p <= 7200)
    base = pd.DataFrame({"airport": frame.ADEP_mvt.astype(str).to_numpy()[keep], "stand": text(frame.STAND_mvt)[keep],
                         "runway": text(frame.RUNWAY_mvt)[keep], "p": p[keep]})
    tables = {}
    for name, keys, _ in LEVELS:
        data = base
        for k in keys[1:]:
            data = data[data[k].to_numpy() != ""]
        grouped = data.groupby(keys, sort=False).p
        table = pd.DataFrame({"p10": grouped.quantile(.1), "p50": grouped.median(), "n": grouped.size()})
        if name == "stand":
            joined = data.join(table.p10.rename("q"), on=keys)
            below = joined[joined.p.to_numpy() <= joined.q.to_numpy()].groupby(keys, sort=False).size()
            table = table[below.reindex(table.index).fillna(0).to_numpy() >= 10]
        tables[name] = table
    return tables


def lookup(frame, tables):
    keys = pd.DataFrame({"airport": frame.ADEP_mvt.astype(str).to_numpy(), "stand": text(frame.STAND_mvt),
                         "runway": text(frame.RUNWAY_mvt)})
    out = np.full((len(keys), 3), np.nan)
    for name, cols, level in LEVELS:
        joined = keys.join(tables[name], on=cols)
        fill = np.isnan(out[:, 0]) & joined.p50.notna().to_numpy()
        out[fill, 0], out[fill, 1], out[fill, 2] = joined.p10.to_numpy(float)[fill], joined.p50.to_numpy(float)[fill], level
    return out


columns = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "ersatz_s", "hat_aobt"]
tables = reference_tables(pd.read_parquet(REL / "data/months_2_3_4_5_6_8_9_10_11_12.parquet", columns=columns))
held = pd.read_parquet(REL / "data/months_1_7.parquet", columns=columns + ["y", "versp_min", "MVT_TIME_UTC_mvt"])
pred = pd.read_parquet(REL / "candidate/holdout_predictions.parquet", columns=["MVT_ID_mvt", "prediction"])
main = held[held.hat_aobt.eq(1) & held.y.notna()].merge(pred, on="MVT_ID_mvt", how="inner", validate="one_to_one").reset_index(drop=True)
ref = lookup(main, tables)
main["ex_p10"], main["ex_p50"] = main.ersatz_s.to_numpy(float) - ref[:, 0], main.ersatz_s.to_numpy(float) - ref[:, 1]
residual = main.y.to_numpy(float) - main.prediction.to_numpy(float)
n = len(main)
raw_month = pd.to_datetime(main.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
month = np.where(np.isin(raw_month, [1, 7]), raw_month, raw_month % 12 + 1)
copied = np.abs(main.y.to_numpy(float) - main.versp_min.to_numpy(float) * 60) <= 10
bins = pd.cut(main.ex_p10, EDGES).astype(str).to_numpy()
airports = main.ADEP_mvt.astype(str).to_numpy()


def bias_bound(keys):
    """MSE removed by subtracting each cell's mean residual: in-sample, and cross-fitted January <-> July."""
    frame = pd.DataFrame({"k": keys, "r": residual, "m": month})
    inside = frame.groupby("k").r.transform("mean").to_numpy()
    cross = np.zeros(n)
    for m, other in ((1, 7), (7, 1)):
        means = frame[frame.m == other].groupby("k").r.mean()
        rows = (month == m)
        cross[rows] = pd.Series(keys[rows]).map(means).fillna(0).to_numpy()
    base = np.mean(residual**2)
    return {"in_sample": float(base - np.mean((residual - inside)**2)), "cross_fitted": float(base - np.mean((residual - cross)**2))}


table = []
for label in pd.cut(pd.Series([0.0]), EDGES).cat.categories.astype(str):
    rows = bins == label
    if rows.sum():
        table.append({"ex_p10_bin_s": label, "rows": int(rows.sum()), "share": float(rows.mean()), "copied_share": float(copied[rows].mean()),
                      "mean_residual_s": float(residual[rows].mean()), "rmse_s": float(np.sqrt(np.mean(residual[rows]**2))),
                      "main_mse_share": float(np.sum(residual[rows]**2) / np.sum(residual**2))})
short = main.ex_p10.to_numpy(float) < -120
report = {"note": "descriptive; v45 candidate-stage holdout ensemble, January/July main lane; references from learning months",
          "rows": n, "main_rmse_s": float(np.sqrt(np.mean(residual**2))),
          "missing_reference": int(np.isnan(ref[:, 0]).sum()), "stand_level_share": float((ref[:, 2] == 2).mean()),
          "by_ex_p10_bin": table,
          "bias_bound_mse": {"bin": bias_bound(bins), "airport_x_bin": bias_bound(np.char.add(np.char.add(airports.astype(str), "|"), bins.astype(str))),
                             "airport_only": bias_bound(airports.astype(str))},
          "implausibly_short_by_airport": {a: {"share": float(short[airports == a].mean()),
                                               "rmse_s": float(np.sqrt(np.mean(residual[(airports == a) & short]**2))) if (short & (airports == a)).any() else None,
                                               "mean_residual_s": float(residual[(airports == a) & short].mean()) if (short & (airports == a)).any() else None}
                                           for a in sorted(set(airports))}}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ("rows", "main_rmse_s", "missing_reference", "stand_level_share", "bias_bound_mse")}, indent=1))
for row in table:
    print(f"{row['ex_p10_bin_s']:>16} rows {row['rows']:>7,} share {row['share']:6.1%} copied {row['copied_share']:6.1%} bias {row['mean_residual_s']:+7.1f} rmse {row['rmse_s']:7.1f} mse share {row['main_mse_share']:6.1%}")
print({a: (round(v["share"], 3), None if v["rmse_s"] is None else round(v["rmse_s"]), None if v["mean_residual_s"] is None else round(v["mean_residual_s"])) for a, v in report["implausibly_short_by_airport"].items()})

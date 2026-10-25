"""Robustness of the six-pair fallback anchor screen, from its saved predictions only.

These diagnostics were defined AFTER the first three pairs had been seen, and extended on
12.09.2026 with validation rules shared by another team on the competition Discord
(another participant): require January and July to improve separately, and reject a gain that
lives in fewer than 100 rows. They are descriptive and do not change the pre-specified
decision rule in fallback_anchor_screen.py.

  gain after removing the k most favourable rows, k = 1, 5, 20, 100
  number of most favourable rows that already account for 50 % and 90 % of the net gain
  gain after removing the most favourable 1 and 3 airport-days
  gain per calendar month inside each pair, and how many pairs improve in both months
  share of net gain on disrupted hours (nt_anchor >= 1800 s)
  share of net gain on fabricated-like rows, |y - delay| <= 60 s (label-defined,
  diagnostic only, never usable at prediction time)
  positive and negative squared-error mass behind each net gain
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--screen", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
files = sorted(args.screen.glob("predictions_*.parquet"))
if len(files) != 6:
    raise SystemExit(f"Expected six pair prediction files, found {len(files)}")
d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
pooled = json.loads((args.screen / "pooled.json").read_text())
y = d.y.to_numpy(float)
base = (d.v44_fallback.to_numpy(float)-y)**2
n = len(d)
stamps = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True)
day = (d.ADEP_mvt.astype(str) + " " + stamps.dt.strftime("%Y-%m-%d")).to_numpy()
fabricated_like = np.abs(y - d.versp_min.to_numpy(float)*60) <= 60
disrupted = np.nan_to_num(d.nt_anchor.to_numpy(float), nan=-1.0) >= 1800
folds = d.fold.to_numpy()
raw_month = stamps.dt.month.to_numpy()


def calendar_month(fold, months):
    """Two source files hold records stamped 3 s before their month; count them in it."""
    pair = [int(m) for m in fold.split("_")]
    return np.where(np.isin(months, pair), months, months % 12 + 1)


def share(part, whole):
    return float(part/whole) if whole != 0 else None


report = {"rows": n, "pairs": sorted(set(folds)),
          "status": "descriptive robustness diagnostics defined after three pairs had been seen; "
                    "not the pre-specified decision rule"}
for variant in ("nt_features", "nt_anchor_normal"):
    gain = base - (d[variant].to_numpy(float)-y)**2
    total = gain.sum()
    r = {"pooled_mse_gain": float(total/n),
         "pooled_json_mse_gain": pooled["metrics"][variant]["mse_gain_vs_v44"],
         "positive_sse_mass": float(gain[gain > 0].sum()),
         "negative_sse_mass": float(gain[gain < 0].sum())}
    order = np.argsort(gain)[::-1]
    for k in (1, 5, 20, 100):
        keep = np.ones(n, dtype=bool)
        keep[order[:k]] = False
        r[f"mse_gain_without_top{k}_rows"] = float(gain[keep].mean())
    r["survives_removing_top100_rows"] = bool(r["mse_gain_without_top100_rows"] > 0)
    if total > 0:
        cumulative = np.cumsum(gain[order])
        r["rows_for_50pct_of_net_gain"] = int(np.argmax(cumulative >= .5*total)+1)
        r["rows_for_90pct_of_net_gain"] = int(np.argmax(cumulative >= .9*total)+1)
    by_day = pd.Series(gain).groupby(day).sum().sort_values(ascending=False)
    for k in (1, 3):
        dropped = list(by_day.index[:k])
        keep = ~np.isin(day, dropped)
        r[f"mse_gain_without_top{k}_airport_days"] = float(gain[keep].mean())
        r[f"top{k}_airport_days"] = dropped
    r["disrupted_rows"] = {"rows": int(disrupted.sum()), "net_gain_share": share(gain[disrupted].sum(), total)}
    r["fabricated_like_rows_label_defined"] = {"rows": int(fabricated_like.sum()),
                                              "net_gain_share": share(gain[fabricated_like].sum(), total)}
    r["folds"], both = {}, 0
    for fold in sorted(set(folds)):
        m = folds == fold
        months = calendar_month(fold, raw_month[m])
        per_month = {int(c): {"rows": int((months == c).sum()), "mse_gain": float(gain[m][months == c].mean())}
                     for c in sorted(set(months))}
        both += int(all(v["mse_gain"] > 0 for v in per_month.values()))
        r["folds"][fold] = {"rows": int(m.sum()), "mse_gain": float(gain[m].mean()),
                            "disrupted_rows": int((m & disrupted).sum()),
                            "disrupted_net_gain_share": share(gain[m & disrupted].sum(), gain[m].sum()),
                            "months": per_month}
    r["pairs_improving_in_both_months"] = both
    report[variant] = r
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

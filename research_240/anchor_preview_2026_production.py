"""Label-free 2026 preview of the neighbour-taxi anchor, with production categorical encoding.

The first preview (anchor_preview_2026.py) stopped at its replication gate because it used the
research re-encoding. This one keeps H.baue's category dtypes and real missing values, as
main_model.py --alle does, so its baseline refit must reproduce v45's official fallback
predictions exactly. It reuses the twelve-month and ranking fallback tables cached by the first
preview, reads no 2026 label, selects nothing from any score, writes no bundle and uploads
nothing. A prediction change is not an error change; its RMSE effect is unknown without labels.
"""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--tables", default=Path(".cache/research_240/anchor_preview"), type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--official", default=Path(".cache/clock_queue_release_20260912/candidate.parquet"), type=Path)
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--tolerance", type=float, default=0.05)
args = parser.parse_args()
project = args.project.resolve()
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import features_stand_rate as MSQ
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_240 import features_neighbour_taxi as NT

F = list(H.MERKMALE_RUECK)
WITH_NT = F + NT.MERKMALE
tables = project / args.tables if not args.tables.is_absolute() else args.tables
learn_files = sorted(tables.glob("fallback_2025_all_*.parquet"))
rank_files = sorted(tables.glob("fallback_2026_ranking_*.parquet"))
if len(learn_files) != 1 or len(rank_files) != 1:
    raise SystemExit(f"Expected one cached table each, found {len(learn_files)} and {len(rank_files)}")
lo = pd.read_parquet(learn_files[0])
rank = pd.read_parquet(rank_files[0])
official_path = args.official if args.official.is_absolute() else project / args.official
official = pd.read_parquet(official_path).set_index("MVT_ID_mvt", verify_integrity=True)


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                             min_child_samples=minimum, verbose=-1, n_jobs=args.jobs, random_state=0)


lo["z"] = (lo.y > H.SCHWELLE_EXTREM).astype(int)
quote = MSQ.passe_an(lo)
lo = MSQ.ausser_faltung(lo)
z0 = lo.z.eq(0).to_numpy()
y = lo.y.to_numpy(float)
classifier = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                                min_child_samples=20, verbose=-1, n_jobs=args.jobs, random_state=0).fit(lo[F], lo.z)
normal_model = reg(400, 31, 20).fit(lo.loc[z0, F], y[z0])
height_model = reg(300, 15, 10).fit(lo.loc[~z0, F], y[~z0])

frame = MSQ.wende_an(quote, rank)
q = classifier.predict_proba(frame[F])[:, 1] ** H.HOCH
high = (1-H.ANTEIL_HM)*frame.versp_min.to_numpy(float)*60 + H.ANTEIL_HM*height_model.predict(frame[F])
normal = normal_model.predict(frame[F])
baseline = np.clip((1-q)*normal + q*high, 0, None)
ids = frame.MVT_ID_mvt.to_numpy()
scored = official.loc[ids, "TAXITIME_SEC_mvt"].to_numpy(float)
gate = {"rows": int(len(ids)), "max_abs_difference_s": float(np.abs(baseline-scored).max()), "tolerance_s": args.tolerance}
gate["passed"] = gate["max_abs_difference_s"] <= args.tolerance
print("GATE", json.dumps(gate), flush=True)
if not gate["passed"]:
    args.output.write_text(json.dumps({"replication_gate": gate}, indent=2))
    sys.exit("Production-encoded baseline does not reproduce v45's official fallback predictions; stop.")

anchor_lo, anchor_rank = lo.nt_anchor.to_numpy(float), frame.nt_anchor.to_numpy(float)
rows = z0 & np.isfinite(anchor_lo)
offset = reg(400, 31, 20).fit(lo.loc[rows, WITH_NT], y[rows]-anchor_lo[rows])
has = np.isfinite(anchor_rank)
normal_anchor = normal.copy()
normal_anchor[has] = anchor_rank[has] + offset.predict(frame.loc[has, WITH_NT])
anchored = np.clip((1-q)*normal_anchor + q*high, 0, None)
delta = anchored - baseline

airports = frame.ADEP_mvt.astype(str).to_numpy()
when = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
disrupted = np.nan_to_num(anchor_rank, nan=-1.0) >= 1800


def summary(mask):
    d = delta[mask]
    if not d.size:
        return {"rows": 0}
    return {"rows": int(d.size), "mean_delta_s": float(d.mean()), "median_delta_s": float(np.median(d)),
            "mean_abs_delta_s": float(np.abs(d).mean()), "rows_up_over_5_min": int((d > 300).sum()),
            "rows_down_over_5_min": int((d < -300).sum()), "rows_over_15_min": int((np.abs(d) > 900).sum()),
            "rows_over_30_min": int((np.abs(d) > 1800).sum()), "max_abs_delta_s": float(np.abs(d).max()),
            "mean_q": float(q[mask].mean())}


cluster = (airports == "EHAM") & when.dt.strftime("%Y-%m-%d").between("2026-01-02", "2026-01-06").to_numpy()
bands = {"anchor < 20 min": np.nan_to_num(anchor_rank, nan=-1) < 1200,
         "anchor 20-30 min": (anchor_rank >= 1200) & (anchor_rank < 1800),
         "anchor 30-45 min": (anchor_rank >= 1800) & (anchor_rank < 2700),
         "anchor >= 45 min": anchor_rank >= 2700, "no anchor": ~has}
report = {"note": "label-free; production-encoded fallback refit on all twelve 2025 months; prediction changes "
                  "on the 2026 ranking fallback rows when the normal branch is anchored; not an error change",
          "replication_gate": gate, "all_rows": summary(np.ones(len(ids), bool)),
          "disrupted_hours": summary(disrupted), "ordinary_hours": summary(~disrupted),
          "schiphol_2_to_6_january": summary(cluster),
          "anchor_bands": {k: summary(v) for k, v in bands.items()},
          "by_month": {int(m): summary((when.dt.month == m).to_numpy()) for m in sorted(when.dt.month.unique())},
          "by_airport": {a: summary(airports == a) for a in sorted(set(airports))}}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2))
rowlevel = pd.DataFrame({"MVT_ID_mvt": ids, "ADEP_mvt": airports, "MVT_TIME_UTC_mvt": frame.MVT_TIME_UTC_mvt.to_numpy(),
                         "nt_anchor": anchor_rank, "q": q, "v45_official": scored, "anchored": anchored, "delta": delta})
rowlevel.to_parquet(project / ".cache/research_240/anchor_preview_2026_production.parquet", index=False)
print(json.dumps({k: report[k] for k in ("all_rows", "disrupted_hours", "ordinary_hours", "schiphol_2_to_6_january")}, indent=2))
print("anchor bands: " + " | ".join(f"{k}: n={v['rows']}, mean {v.get('mean_delta_s', 0):+.0f}s" for k, v in report["anchor_bands"].items()))
print("by airport:   " + " | ".join(f"{a}: mean {v['mean_delta_s']:+.0f}s, >15min {v['rows_over_15_min']}" for a, v in report["by_airport"].items()))
print("by month:     " + " | ".join(f"{m}: mean {v['mean_delta_s']:+.0f}s, >15min {v['rows_over_15_min']}" for m, v in report["by_month"].items()), flush=True)

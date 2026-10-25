"""Pre-registered six-pair fallback screen: a soft schedule-copy gate on top of the v46 fallback.

Pre-specified 14.09.2026 before any fit of this script. No leaderboard feedback. Designed after
research_260/fallback_regime_components.py on the same six pairs: schedule copies (block time = schedule, y = delay) that the
extreme gate misses (q < 0.1: 102 rows, 9.5 % of fallback squared error) carry most of the copy error, the copy height is
already close (median 112 s), and a copy classifier on the 35 fallback inputs reaches AUC 0.980 among extremes and 0.934
among non-extremes. Related rejections: research_250's cause classifier and the September 'y = versp' classifier REPLACED the
extreme classifier. The main lane's accepted soft gate (main_model.py ERF_HOCH) corrects an existing prediction instead; this
screen transfers that design.

Baseline prod_anchor: the scored v46 fallback (production encoding), rebuilt per pair; GATE: reproduces
research_240/fallback_anchor_production.py within 1e-6 s. Copy classifier: LightGBM binary with the extreme classifier's
settings (600 rounds, learning rate 0.05, 31 leaves, min_child_samples 20, seed 0) on the 35 fallback inputs of the training
months, label |y - delay| <= 60 s over all training fallback rows. Variants, no parameter chosen on held-out months:
  G25  max(v46 + c ** 2.5 * (delay - v46), 0)     the main lane's exponent
  G10  max(v46 + c * (delay - v46), 0)            the unshrunk probability
Decision rule against prod_anchor, pooled over six pairs: (1) MSE gain positive, (2) positive without its 100 most favourable
rows, (3) at least 4 of 6 pairs gain, (4) January and July both gain inside pair 1_7. No column is added, so no permutation
control. The candidate is the passing variant with the larger pooled gain; otherwise nothing is built. Reported with each
regime's contribution (copy extreme, copy non-extreme, day shift, normal, other). Fallback-only; not a leaderboard forecast.

Run from the repository root:  python3 -B research_260/fallback_copy_gate_screen.py --project .
"""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, default=Path("."))
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--bootstrap", type=int, default=2000)
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

BASE = list(H.MERKMALE_RUECK)
OFFSET = BASE + [c for c in NT.MERKMALE if c not in BASE]
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
TABLES = project / ".cache/research_240/rescreen_production"
SAVED = project / ".cache/research_240/fallback_anchor_production"
OUTPUT = project / "research_260/results/fallback_copy_gate_screen_pooled.json"
V47, SHARE_2026 = 266.9416, 5290 / 344841
VARIANTS = {"G25": 2.5, "G10": 1.0}
index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet") for s in TABLES.glob("rescreen_*.json")}


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves, min_child_samples=minimum,
                             verbose=-1, n_jobs=args.jobs, random_state=0)


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31, min_child_samples=20,
                              verbose=-1, n_jobs=args.jobs, random_state=0)


frames = []
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    learn = [m for m in range(1, 13) if m not in pair]
    tr, va = pd.read_parquet(index[tuple(learn)]).reset_index(drop=True), pd.read_parquet(index[tuple(sorted(pair))]).reset_index(drop=True)
    tr["z"] = (tr.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(tr)
    tr, va = MSQ.ausser_faltung(tr), MSQ.wende_an(mapping, va)
    z0, ytr = tr.z.eq(0).to_numpy(), tr.y.to_numpy(float)
    p = classifier().fit(tr[BASE], tr.z).predict_proba(va[BASE])[:, 1]
    delay = 60 * va.versp_min.to_numpy(float)
    height = (1 - H.ANTEIL_HM) * delay + H.ANTEIL_HM * reg(300, 15, 10).fit(tr.loc[~z0, BASE], ytr[~z0]).predict(va[BASE])
    normal = reg(400, 31, 20).fit(tr.loc[z0, BASE], ytr[z0]).predict(va[BASE])
    a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
    rows = z0 & np.isfinite(a_tr)
    offset = reg(400, 31, 20).fit(tr.loc[rows, OFFSET], ytr[rows] - a_tr[rows])
    has = np.isfinite(a_va)
    anchored = normal.copy()
    anchored[has] = a_va[has] + offset.predict(va.loc[has, OFFSET])
    q = p ** H.HOCH
    v46 = np.maximum((1 - q) * anchored + q * height, 0)
    saved = pd.read_parquet(SAVED / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[va.MVT_ID_mvt, "prod_anchor"].to_numpy(float)
    gap = float(np.abs(v46 - saved).max())
    print("GATE", key, gap, flush=True)
    if gap > 1e-6:
        raise SystemExit(f"rebuild does not reproduce prod_anchor on {key}")
    train_copy = (np.abs(ytr - 60 * tr.versp_min.to_numpy(float)) <= 60).astype(int)
    c = classifier().fit(tr[BASE], train_copy).predict_proba(va[BASE])[:, 1]
    frame = pd.DataFrame({"fold": key, "MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "MVT_TIME_UTC_mvt": va.MVT_TIME_UTC_mvt.to_numpy(),
                          "airport": va.ADEP_mvt.astype(str).to_numpy(), "y": va.y.to_numpy(float), "delay": delay, "prod_anchor": v46, "c": c})
    for name, h in VARIANTS.items():
        frame[name] = np.maximum(v46 + c ** h * (delay - v46), 0)
    yv = frame.y.to_numpy()
    print("PAIR", key, json.dumps({k: round(float(np.sqrt(np.mean((frame[k].to_numpy() - yv) ** 2))), 3) for k in ["prod_anchor"] + list(VARIANTS)}), flush=True)
    frames.append(frame)

pooled = pd.concat(frames, ignore_index=True)
y, delay, base = pooled.y.to_numpy(), pooled.delay.to_numpy(), pooled.prod_anchor.to_numpy()
folds = pooled.fold.to_numpy()
stamps = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
month = np.array([m if str(m) in f.split("_") else m % 12 + 1 for m, f in zip(stamps.dt.month.to_numpy(), folds)])
_, inverse = np.unique((pooled.airport + "|" + stamps.dt.strftime("%Y-%m-%d")).to_numpy(), return_inverse=True)
cluster_rows = np.bincount(inverse)
draws = np.random.default_rng(20260914).integers(0, cluster_rows.size, size=(args.bootstrap, cluster_rows.size))
copy = np.abs(y - delay) <= 60
extreme = y > H.SCHWELLE_EXTREM
shift = ~copy & (y >= 80000) & (y <= 93600)
normal = ~copy & (y >= -600) & (y <= 5400)
regimes = {"copy_extreme": copy & extreme, "copy_nonextreme": copy & ~extreme, "day_shift": shift, "normal": normal,
           "other": ~copy & ~shift & ~normal}


def compare(candidate):
    gain = (base - y) ** 2 - (candidate - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    pair_gains = {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}
    boot = np.bincount(inverse, weights=gain)[draws].sum(axis=1) / cluster_rows[draws].sum(axis=1)
    out = {"pooled_mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
           "pair_gains": pair_gains, "pairs_positive": int(sum(g > 0 for g in pair_gains.values())),
           "january_gain_1_7": float(gain[(folds == "1_7") & (month == 1)].mean()), "july_gain_1_7": float(gain[(folds == "1_7") & (month == 7)].mean()),
           "airport_day_bootstrap_mse_gain_95pct": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
           "regime_contributions": {r: float(gain[m].sum() / len(gain)) for r, m in regimes.items()},
           "airport_contributions": {a: float(gain[pooled.airport.to_numpy() == a].sum() / len(gain)) for a in sorted(pooled.airport.unique())},
           "rows_changed_over_60s": int(np.sum(np.abs(candidate - base) > 60)),
           "mechanical_leaderboard_rmse_if_full_transfer_not_forecast": float(np.sqrt(max(V47 ** 2 - SHARE_2026 * float(gain.mean()), 0)))}
    out["checks"] = {"1_pooled_gain_positive": out["pooled_mse_gain"] > 0, "2_survives_top100_removal": out["mse_gain_without_top100_rows"] > 0,
                     "3_at_least_4_of_6_pairs": out["pairs_positive"] >= 4,
                     "4_january_and_july_gain_in_1_7": out["january_gain_1_7"] > 0 and out["july_gain_1_7"] > 0}
    out["advance"] = bool(all(out["checks"].values()))
    return out


report = {"validation": "fallback-only six-pair screen against the scored v46 fallback recipe, production encoding; not a leaderboard forecast",
          "rmse": {k: float(np.sqrt(np.mean((pooled[k].to_numpy() - y) ** 2))) for k in ["prod_anchor"] + list(VARIANTS)},
          "variants": {name: compare(pooled[name].to_numpy()) for name in VARIANTS}}
advancing = [n for n, e in report["variants"].items() if e["advance"]]
report["decision"] = {"advancing": advancing, "candidate": max(advancing, key=lambda n: report["variants"][n]["pooled_mse_gain"]) if advancing else None}
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print("RMSE", {k: round(v, 3) for k, v in report["rmse"].items()})
for n, e in report["variants"].items():
    print(n, json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in e.items()
                         if k in ("pooled_mse_gain", "mse_gain_without_top100_rows", "pairs_positive", "january_gain_1_7", "july_gain_1_7",
                                  "rows_changed_over_60s", "advance")}),
          "| regimes", {r: round(v) for r, v in e["regime_contributions"].items()},
          "| pairs", {k: round(v) for k, v in e["pair_gains"].items()}, "| boot", [round(x) for x in e["airport_day_bootstrap_mse_gain_95pct"]])
print("DECISION", report["decision"])

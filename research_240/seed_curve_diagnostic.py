"""How much would more ensemble seeds help the main lane? Estimated from saved release checkpoints.

Pre-specified 13.09.2026, before running. No training and no leaderboard feedback. Uses the v45 release
candidate stage (learn months 2-6 and 8-12, test January/July): five LightGBM seeds, three CatBoost
seeds, five neural-network states, ten airport trees and the copied-block-time gate, combined exactly
as research_250/release_common.predict does.

Replication gate: with every seed, the recombined main-lane predictions must reproduce the saved
candidate holdout predictions within 1e-4 s.
Seed curve: main-lane MSE averaged over all seed subsets of size k for one model family at a time, the
other families complete. MSE(k) = a + b/k is fitted per family and extrapolated to twice the current
seed count. Decision: pursue extra seeds only if the summed extrapolated gain is worth at least 0.15 s
of RMSE at the official level (main-lane MSE gain x 339,551 / 344,841 / (2 x 268.24)).

Run from the repository root:
  python3 research_240/seed_curve_diagnostic.py --project . --output research_240/results/seed_curve_diagnostic.json
"""
import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--release", default=Path(".cache/clock_queue_release_20260912"), type=Path)
args = parser.parse_args()
project = args.project.resolve()
output = args.output if args.output.is_absolute() else project / args.output
sys.path.insert(0, str(project))
os.chdir(project)
release = args.release if args.release.is_absolute() else project / args.release

import main_model as H  # torch before tree libraries on macOS
import neural_model as NM
import joblib
import numpy as np
import pandas as pd
from research_250 import candidate_features as B

OFFICIAL, MAIN_ROWS, ROWS, BAR_S = 268.2394, 339551, 344841, 0.15
started = time.time()
checkpoints = release / "candidate/checkpoints"


def load(name):
    obj = joblib.load(checkpoints / f"{name}.pkl")
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


frame = pd.read_parquet(release / "data/months_1_7.parquet")
main = frame.loc[frame.hat_aobt.eq(1)].reset_index(drop=True)
features = B.main_features(H.MERKMALE_HAUPT)
x = main[features]
y, ersatz, versp = main.y.to_numpy(float), main.ersatz_s.to_numpy(float), main.versp_min.to_numpy(float) * 60
airports = main.ADEP_mvt.astype(str).to_numpy()

lgb_seeds, cb_seeds, nn_seeds = [0, 7, 13, 21, 29], [0, 7, 13], [13, 21, 29, 37, 45]
lgb = {s: load(f"lgb_{s}").predict(x) for s in lgb_seeds}
print("LightGBM done", round(time.time() - started), "s", flush=True)
local = np.empty(len(main))
for airport in sorted(set(airports)):
    mask = airports == airport
    local[mask] = load(f"airport_{airport}").predict(x.loc[mask])
cat_x = H.kat_str(x)
cat = {s: load(f"catboost_{s}").predict(cat_x) for s in cb_seeds}
print("CatBoost done", round(time.time() - started), "s", flush=True)
items = {s: load(f"neural_{s}") for s in nn_seeds}
net_single = {}
for s in nn_seeds:
    net = dict(items[s])
    net["zustaende"] = list(items[s]["zustaende"])
    net_single[s] = NM.sage_voraus(net, x)
combined = dict(items[nn_seeds[0]])
combined["zustaende"] = [state for s in nn_seeds for state in items[s]["zustaende"]]
net_all = NM.sage_voraus(combined, x)
p_gate = load("reporting_classifier").predict_proba(x)[:, 1] ** H.ERF_HOCH
print("neural and gate done", round(time.time() - started), "s", flush=True)


def predict(lgb_mean, cat_mean, net_mean):
    residual = lgb_mean
    if H.ANTEIL_FLH:
        residual = (1 - H.ANTEIL_FLH) * residual + H.ANTEIL_FLH * local
    residual = (1 - H.ANTEIL_CB) * residual + H.ANTEIL_CB * cat_mean
    residual = (1 - H.ANTEIL_NETZ) * residual + H.ANTEIL_NETZ * net_mean
    output = ersatz + residual
    output = (1 - p_gate) * output + p_gate * versp
    return np.maximum(np.clip(output, 0, None), 300)


def mse(prediction):
    return float(np.mean((prediction - y) ** 2))


lgb_full = np.mean([lgb[s] for s in lgb_seeds], axis=0)
cat_full = np.mean([cat[s] for s in cb_seeds], axis=0)
net_mean_single = np.mean([net_single[s] for s in nn_seeds], axis=0)
full = predict(lgb_full, cat_full, net_all)
saved = pd.read_parquet(release / "candidate/holdout_predictions.parquet").set_index("MVT_ID_mvt").loc[main.MVT_ID_mvt]
gap = float(np.abs(full - saved.prediction.to_numpy(float)).max())
report = {"validation": "saved v45 candidate-stage checkpoints, January/July main lane; no training",
          "rows": int(len(main)), "replication_gate": {"max_abs_difference_s": gap, "passed": gap <= 1e-4},
          "neural_mean_of_single_states_vs_combined_max_abs_s": float(np.abs(net_mean_single - net_all).max()),
          "full_main_rmse": float(np.sqrt(mse(full))), "families": {}}
print("REPLICATION", json.dumps(report["replication_gate"]), flush=True)
if not report["replication_gate"]["passed"]:
    output.write_text(json.dumps(report, indent=2))
    sys.exit("Recombined predictions do not reproduce the saved candidate holdout predictions")

total_gain = 0.0
for family, seeds in (("lightgbm", lgb_seeds), ("catboost", cb_seeds), ("neural", nn_seeds)):
    curve = {}
    for k in range(1, len(seeds) + 1):
        values = []
        for subset in combinations(seeds, k):
            if family == "lightgbm":
                prediction = predict(np.mean([lgb[s] for s in subset], axis=0), cat_full, net_all)
            elif family == "catboost":
                prediction = predict(lgb_full, np.mean([cat[s] for s in subset], axis=0), net_all)
            else:
                prediction = predict(lgb_full, cat_full, np.mean([net_single[s] for s in subset], axis=0))
            values.append(mse(prediction))
        curve[k] = float(np.mean(values))
    ks = np.array(sorted(curve), float)
    design = np.column_stack([np.ones_like(ks), 1 / ks])
    a, b = np.linalg.lstsq(design, np.array([curve[int(k)] for k in ks]), rcond=None)[0]
    current, doubled = len(seeds), 2 * len(seeds)
    gain = float((a + b / current) - (a + b / doubled))
    total_gain += gain
    report["families"][family] = {"mse_by_seed_count": curve, "fit_a": float(a), "fit_b": float(b),
                                  "extrapolated_mse_gain_doubling": gain}
    print(family, json.dumps(report["families"][family]), flush=True)

official_gain_s = total_gain * MAIN_ROWS / ROWS / (2 * OFFICIAL)
report["decision"] = {"summed_main_mse_gain_doubling_all": total_gain, "official_level_rmse_gain_s": official_gain_s,
                      "bar_s": BAR_S, "pursue_extra_seeds": bool(official_gain_s >= BAR_S)}
report["seconds"] = time.time() - started
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2))
print("DECISION", json.dumps(report["decision"]), flush=True)

"""Pre-registered cross-month selection of the sequence-net weight in the frozen main ensemble (January/July).

Pre-specified 14.09.2026 before any evaluation. No leaderboard feedback. v47 put origin-grouped sequence nets into the neural
quarter of the frozen v45 ensemble but kept the weight 0.25 that was tuned for the weaker flat net. The sequence net alone is
now about as strong as the single LightGBM (the team, January/July: 221.75 against 223.70), so the fixed weight may be too low.

Inputs (research_260/sequence_release_v47.py validate stages, ignored cache .cache/sequence_release_v47/validate): the
candidate-stage frozen ensemble components on the January/July main rows (ensemble_baseline.parquet: tree residual without the
neural part, reporting gate, schedule value, ersatz, y, saved prediction) and ten-month origin sequence-net residuals for seeds
13 and 21. Seed 21 is fitted here with the identical recipe and arrays (stage fit-seed21).

Prediction for weight w: max((1 - gate) * (ersatz + (1 - w) * tree + w * mean sequence residual) + gate * schedule, 300).
Grid w in {0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}.
Cross-month selection: w_jan = the grid weight with the lowest main RMSE on January rows, applied to July; w_jul = the lowest on
July, applied to January.
Decision to rebuild v47 with a new weight (nothing is uploaded here):
  (1) both cross-month applications lower main MSE against w = 0.25 on the month they are applied to,
  (2) the combined cross-applied improvement stays positive without its 100 most favourable rows,
  (3) |w_jan - w_jul| <= 0.10; the release weight is their mean rounded to the nearest 0.05, and it must differ from 0.25.
Two seeds stand in for the release's five, which biases the selected weight downward. Single split, as for ERF_HOCH.

Run from the repository root:
  python3 -u -B research_260/sequence_blend_weight.py fit-seed21
  python3 -B research_260/sequence_blend_weight.py select
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import neural_model as NM  # noqa: F401  torch before tree libraries on macOS
import torch
import joblib
import numpy as np
import pandas as pd
from research_260 import sequence_features as F, sequence_model as S
from research_260.airport_ablation import sha, write_json, log

PROJECT = Path(__file__).resolve().parents[1]
VALIDATE = PROJECT / ".cache/sequence_release_v47/validate"
MODE, SEED = "origin", 21
GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
OUTPUT = PROJECT / "research_260/results/sequence_blend_weight.json"


def fit_seed21():
    name = f"validate_{MODE}_{SEED}"
    if (VALIDATE / f"{name}.json").exists():
        log(f"{name} already complete")
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing a silent CPU fallback")
    prep = joblib.load(VALIDATE / "preprocessing.pkl")
    xk, xn, y = (np.load(VALIDATE / f"train_{part}.npy") for part in ("xk", "xn", "residual"))
    xs, ms = np.load(VALIDATE / f"train_{MODE}_xs.npy"), np.load(VALIDATE / f"train_{MODE}_ms.npy")
    log(f"FIT {name} on MPS, {len(y)} rows")
    package = S.fit_net(True, SEED, xk, xn, xs, ms, y, prep["kdim"], log=lambda message: log(f"{name} {message}"))
    package.update(preprocessing=prep, mode=MODE, feature_schema=F.SCHEMA)
    partial = VALIDATE / f"{name}.pkl.partial"
    joblib.dump(package, partial)
    partial.replace(VALIDATE / f"{name}.pkl")
    del xk, xn, xs, ms, y
    torch.mps.empty_cache()
    vk, vn = np.load(VALIDATE / "valid_xk.npy"), np.load(VALIDATE / "valid_xn.npy")
    vs, vm = np.load(VALIDATE / f"valid_{MODE}_xs.npy"), np.load(VALIDATE / f"valid_{MODE}_ms.npy")
    residual = S.predict(package, vk, vn, vs, vm)
    rows = pd.read_parquet(VALIDATE / "valid_rows.parquet")
    rows["residual"] = residual
    rows.to_parquet(VALIDATE / f"{name}_predictions.parquet", index=False)
    error = np.maximum(rows.ersatz_s.to_numpy(float) + residual, 300.) - rows.y.to_numpy(float)
    result = {"mode": MODE, "seed": SEED, "floored_component_rmse": float(np.sqrt(np.mean(error ** 2))), "seconds": package["seconds"],
              "kevin_origin_seed21_reference": 225.04809149029367, "model_sha256": sha(VALIDATE / f"{name}.pkl")}
    write_json(VALIDATE / f"{name}.json", result)
    log("RESULT " + json.dumps(result))


def select():
    info = json.loads((VALIDATE / "ensemble_baseline.json").read_text())
    if sha(VALIDATE / "ensemble_baseline.parquet") != info["baseline_sha256"]:
        raise ValueError("ensemble baseline changed")
    rows = pd.read_parquet(VALIDATE / "ensemble_baseline.parquet")
    residuals = []
    for seed in (13, 21):
        pred = pd.read_parquet(VALIDATE / f"validate_{MODE}_{seed}_predictions.parquet").set_index("MVT_ID_mvt")
        if set(pred.index) != set(rows.MVT_ID_mvt):
            raise ValueError("prediction IDs differ from the ensemble baseline")
        residuals.append(pred.loc[rows.MVT_ID_mvt, "residual"].to_numpy(float))
    seq = np.mean(residuals, axis=0)
    y = rows.y.to_numpy(float)
    ersatz, tree, gate, schedule = (rows[c].to_numpy(float) for c in ("ersatz_s", "non_neural_residual", "gate", "schedule"))
    month = pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    month = np.where(np.isin(month, [1, 7]), month, month % 12 + 1)
    jan, jul = month == 1, month == 7

    def predict(w):
        return np.maximum((1 - gate) * (ersatz + (1 - w) * tree + w * seq) + gate * schedule, 300.)

    predictions = {w: predict(w) for w in GRID}
    rmse = {str(w): {"january": float(np.sqrt(np.mean((p[jan] - y[jan]) ** 2))), "july": float(np.sqrt(np.mean((p[jul] - y[jul]) ** 2))),
                     "both": float(np.sqrt(np.mean((p - y) ** 2)))} for w, p in predictions.items()}
    w_jan = min(GRID, key=lambda w: rmse[str(w)]["january"])
    w_jul = min(GRID, key=lambda w: rmse[str(w)]["july"])
    base = predictions[0.25]
    cross = base.copy()
    cross[jul] = predictions[w_jan][jul]
    cross[jan] = predictions[w_jul][jan]
    gain = (base - y) ** 2 - (cross - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    release_weight = round(round((w_jan + w_jul) / 2 / 0.05) * 0.05, 2)
    checks = {"1_both_cross_month_applications_improve": bool(gain[jul].mean() > 0 and gain[jan].mean() > 0),
              "2_positive_without_top100_rows": bool(gain[keep].mean() > 0),
              "3_selected_weights_within_0_10": bool(abs(w_jan - w_jul) <= 0.10 + 1e-9),
              "4_release_weight_differs_from_0_25": bool(abs(release_weight - 0.25) > 1e-9)}
    report = {"note": "January/July cross-month weight selection on the candidate-stage frozen ensemble; two sequence seeds; not six-pair",
              "baseline_saved_main_rmse": info["main_rmse"], "rmse_by_weight": rmse, "w_selected_on_january": w_jan, "w_selected_on_july": w_jul,
              "cross_applied": {"july_mse_gain": float(gain[jul].mean()), "january_mse_gain": float(gain[jan].mean()),
                                "main_mse_gain": float(gain.mean()), "main_mse_gain_without_top100_rows": float(gain[keep].mean()),
                                "main_rmse_w025": float(np.sqrt(np.mean((base - y) ** 2))), "main_rmse_cross": float(np.sqrt(np.mean((cross - y) ** 2)))},
              "release_weight": release_weight, "checks": checks, "decision": {"rebuild_with_new_weight": bool(all(checks.values()))}}
    write_json(OUTPUT, report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["fit-seed21", "select"])
    {"fit-seed21": fit_seed21, "select": select}[parser.parse_args().stage]()

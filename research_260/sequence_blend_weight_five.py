"""Blend-weight selection with the release's full five seeds. Rules fixed 21.09.2026 before fitting.

Why this rerun. research_260/sequence_blend_weight.py selected the sequence-net weight on the January/July
candidate-stage ensemble with two seeds (13 and 21) and rejected the change: January chose 0.30, July 0.45, the
weights lay 0.15 apart and the cross-applied gain went negative without its 100 most favourable rows. That script
states the flaw itself: "Two seeds stand in for the release's five, which biases the selected weight downward."
A two-seed mean is noisier than the release's five-seed mean, so the blend leans on the trees more than it should.

This rerun fits seeds 29, 37 and 45 on the identical prepared arrays and repeats the selection with all five, so the
neural part is exactly what v47 ships. Nothing else changes: same tensors, same recipe, same grid, same rules.

What a true six-pair version would need, and why it is not this: the frozen ensemble's components (tree residual
without the neural part, reporting gate, schedule value) exist only for pair 1_7. The other five pairs would mean
retraining the v45 ensemble per pair, and the release log records about 2.5 hours per CatBoost seed alone. So this
remains one split, and the result must be read as such.

Decision, fixed here before the fits:
  1. the five-seed grid must place its minimum away from 0.25, otherwise the current weight stands
  2. January and July must both improve against 0.25 at the selected weight
  3. the selected weight from January applied to July, and from July applied to January, must both improve
  4. the combined cross-applied gain must survive removing its 100 most favourable rows
Only with all four does a rebuild get proposed; nothing is built or uploaded here.

Run from the repository root:
  python3 -u -B research_260/sequence_blend_weight_five.py fit --seeds 29 37 45
  python3 -B research_260/sequence_blend_weight_five.py select
"""
import argparse
import json
from pathlib import Path
import sys
import time

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
MODE = "origin"
RELEASE_SEEDS = (13, 21, 29, 37, 45)
GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
OUTPUT = PROJECT / "research_260/results/sequence_blend_weight_five.json"


def fit(seed):
    name = f"validate_{MODE}_{seed}"
    if (VALIDATE / f"{name}.json").exists():
        log(f"{name} already complete")
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing a silent CPU fallback")
    prep = joblib.load(VALIDATE / "preprocessing.pkl")
    xk, xn, y = (np.load(VALIDATE / f"train_{part}.npy") for part in ("xk", "xn", "residual"))
    xs, ms = np.load(VALIDATE / f"train_{MODE}_xs.npy"), np.load(VALIDATE / f"train_{MODE}_ms.npy")
    log(f"FIT {name} on MPS, {len(y):,} rows")
    package = S.fit_net(True, seed, xk, xn, xs, ms, y, prep["kdim"], log=lambda m: log(f"{name} {m}"))
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
    result = {"mode": MODE, "seed": seed, "floored_component_rmse": float(np.sqrt(np.mean(error ** 2))),
              "seconds": package["seconds"], "model_sha256": sha(VALIDATE / f"{name}.pkl")}
    write_json(VALIDATE / f"{name}.json", result)
    log("RESULT " + json.dumps(result))


def select():
    info = json.loads((VALIDATE / "ensemble_baseline.json").read_text())
    if sha(VALIDATE / "ensemble_baseline.parquet") != info["baseline_sha256"]:
        raise ValueError("ensemble baseline changed")
    rows = pd.read_parquet(VALIDATE / "ensemble_baseline.parquet")
    residuals, used = [], []
    for seed in RELEASE_SEEDS:
        path = VALIDATE / f"validate_{MODE}_{seed}_predictions.parquet"
        if not path.exists():
            continue
        pred = pd.read_parquet(path).set_index("MVT_ID_mvt")
        if set(pred.index) != set(rows.MVT_ID_mvt):
            raise ValueError(f"seed {seed} prediction IDs differ from the ensemble baseline")
        residuals.append(pred.loc[rows.MVT_ID_mvt, "residual"].to_numpy(float))
        used.append(seed)
    seq = np.mean(residuals, axis=0)
    log(f"using seeds {used}")
    y = rows.y.to_numpy(float)
    ersatz, tree, gate, schedule = (rows[c].to_numpy(float) for c in ("ersatz_s", "non_neural_residual", "gate", "schedule"))
    month = pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    month = np.where(np.isin(month, [1, 7]), month, month % 12 + 1)
    jan, jul = month == 1, month == 7

    def predict(w):
        return np.maximum((1 - gate) * (ersatz + (1 - w) * tree + w * seq) + gate * schedule, 300.)

    predictions = {w: predict(w) for w in GRID}
    rmse = {str(w): {"january": float(np.sqrt(np.mean((p[jan] - y[jan]) ** 2))),
                     "july": float(np.sqrt(np.mean((p[jul] - y[jul]) ** 2))),
                     "both": float(np.sqrt(np.mean((p - y) ** 2)))} for w, p in predictions.items()}
    w_jan = min(GRID, key=lambda w: rmse[str(w)]["january"])
    w_jul = min(GRID, key=lambda w: rmse[str(w)]["july"])
    w_both = min(GRID, key=lambda w: rmse[str(w)]["both"])
    base = predictions[0.25]
    cross = base.copy()
    cross[jul] = predictions[w_jan][jul]
    cross[jan] = predictions[w_jul][jan]
    gain = (base - y) ** 2 - (cross - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    both_improve = {str(w): bool(rmse[str(w)]["january"] < rmse["0.25"]["january"]
                                 and rmse[str(w)]["july"] < rmse["0.25"]["july"]) for w in GRID}
    checks = {"1_grid_minimum_away_from_0_25": w_both != 0.25,
              "2_both_months_improve_at_the_minimum": both_improve[str(w_both)],
              "3_cross_month_applications_improve": bool(gain[jul].mean() > 0 and gain[jan].mean() > 0),
              "4_survives_top100_removal": bool(gain[keep].mean() > 0)}
    report = {"note": "January/July candidate-stage ensemble with the release's five sequence seeds; one split, not six pairs",
              "seeds_used": used, "rmse_by_weight": rmse, "weight_minimising_january": w_jan,
              "weight_minimising_july": w_jul, "weight_minimising_both": w_both,
              "both_months_improve_against_0_25": both_improve,
              "cross_applied": {"july_mse_gain": float(gain[jul].mean()), "january_mse_gain": float(gain[jan].mean()),
                                "main_mse_gain": float(gain.mean()),
                                "main_mse_gain_without_top100_rows": float(gain[keep].mean()),
                                "main_rmse_w025": float(np.sqrt(np.mean((base - y) ** 2))),
                                "main_rmse_cross": float(np.sqrt(np.mean((cross - y) ** 2)))},
              "checks": checks, "propose_rebuild": bool(all(checks.values())),
              "baseline_saved_main_rmse": info["main_rmse"]}
    write_json(OUTPUT, report)
    print(json.dumps({k: report[k] for k in ("seeds_used", "weight_minimising_january", "weight_minimising_july",
                                             "weight_minimising_both", "checks", "propose_rebuild")}, indent=1))
    print("RMSE by weight (main lane):")
    for w in GRID:
        mark = "  <== current" if w == 0.25 else ("  <== minimum" if w == w_both else "")
        print(f"  {w:.2f}  january {rmse[str(w)]['january']:.4f}  july {rmse[str(w)]['july']:.4f}  "
              f"both {rmse[str(w)]['both']:.4f}{mark}")
    print("cross-applied:", json.dumps({k: round(v, 1) for k, v in report["cross_applied"].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["fit", "select"])
    parser.add_argument("--seeds", type=int, nargs="*", default=[29, 37, 45])
    args = parser.parse_args()
    if args.stage == "fit":
        started = time.time()
        for seed in args.seeds:          # sequential: one large fit at a time, as the project rules require
            fit(seed)
            log(f"elapsed {time.time() - started:.0f}s")
    else:
        select()

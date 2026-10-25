"""Pilot 5 (PREREG_stacker.md): the ADS-B columns in a second-stage corrector over out-of-fold first-stage predictions.
Run from the repository root:  python3 -u -B research_730_adsb_learner/pilot5_screen.py
"""
import json
from pathlib import Path
import sys
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_730_adsb_learner import features_adsb as F
from research_730_adsb_learner import pilot_screen as P
from research_300 import incremental as I
from research_300.heavy_lock import heavy_work

CACHE, ROOT = P.CACHE, P.ROOT
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot5_screen.json"
AD18 = P.AD + list(F.STAND_MERKMALE)
STAGE2_BASE = ["p1", "ersatz_s", "stunde", "ADEP_mvt"]
ONE_STAGE_REFERENCE = 220.824                                     # pilot 2, same 18 columns, one stage
ROUNDS = 600


def stage2(table, fold, features, seed, label):
    cache = CACHE / f"stack_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    pred = np.zeros(len(table))
    resid = (table.y - table.p1).to_numpy()
    for k in range(P.FOLDS):
        learn, held = table[fold != k].copy(), table[fold == k].copy()
        for column in ("ADEP_mvt",):
            cats = pd.unique(learn[column].astype(str))
            learn[column] = pd.Categorical(learn[column].astype(str), categories=cats)
            held[column] = pd.Categorical(held[column].astype(str), categories=cats)
        with heavy_work(f"research_730 pilot 5 {label} seed {seed} fold {k}", log=P.log):
            model = lgb.LGBMRegressor(objective="l2", n_estimators=ROUNDS, learning_rate=.03, num_leaves=63, min_child_samples=100,
                                      subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)
            model.fit(learn[features], resid[fold != k])
            pred[fold == k] = np.maximum(held.p1.to_numpy(float) + model.predict(held[features]), 300)
    np.save(cache, pred)
    P.log(f"{label} seed {seed}: RMSE {np.sqrt(np.mean((pred - table.y) ** 2)):.3f}")
    return pred


def main():
    table = pd.read_parquet(CACHE / "table_1_7_stand.parquet")
    date = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    fold = date.map(lambda d: zlib.crc32(d.encode()) % P.FOLDS).to_numpy()
    table["_day"] = table.ADEP_mvt.astype(str) + "|" + date
    airport = table.ADEP_mvt.astype(str).to_numpy()
    month = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    y = table.y.to_numpy(float)
    preds = {}
    for seed in P.SEEDS:
        table["p1"] = np.load(CACHE / f"oof_base_s{seed}.npy")
        preds[("base", seed)] = table.p1.to_numpy().copy()
        preds[("stack", seed)] = stage2(table, fold, STAGE2_BASE + AD18, seed, "stack")
        preds[("stack0", seed)] = stage2(table, fold, STAGE2_BASE, seed, "stack0")
        for k in P.CONTROLS:
            ctrl = P.permute_within(table, AD18, table._day, 9000 + k)
            preds[(f"control_{k}", seed)] = stage2(ctrl, fold, STAGE2_BASE + AD18, seed, f"control_{k}")
    loss = {key: (p - y) ** 2 for key, p in preds.items()}
    mean = lambda name: np.mean([loss[(name, s)] for s in P.SEEDS], axis=0)
    stack, stack0, base = mean("stack"), mean("stack0"), mean("base")
    ctrl = np.mean([loss[(f"control_{k}", s)] for s in P.SEEDS for k in P.CONTROLS], axis=0)
    diff = ctrl - stack
    rmse = lambda l: float(np.sqrt(l.mean()))
    per_seed = {s: float(np.mean([loss[(f"control_{k}", s)].mean() for k in P.CONTROLS]) - loss[("stack", s)].mean()) for s in P.SEEDS}
    report = {"prereg": "research_730_adsb_learner/PREREG_stacker.md", "rows": int(len(y)),
              "rmse": {"stack": rmse(stack), "mean_control": rmse(ctrl), "stack0": rmse(stack0), "base": rmse(base), "one_stage_reference": ONE_STAGE_REFERENCE},
              "rmse_gain_vs_control_s": rmse(ctrl) - rmse(stack), "rmse_gain_vs_base_s": rmse(base) - rmse(stack),
              "share_of_one_stage_gain_kept": (rmse(base) - rmse(stack)) / (rmse(base) - ONE_STAGE_REFERENCE),
              "per_seed_mse_gain": per_seed, "months": {str(m): float(diff[month == m].mean()) for m in (1, 7)},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, table._day.to_numpy()),
              "airports": {a: {"rows": int((airport == a).sum()), "vs_control": float(diff[airport == a].mean()),
                               "vs_base": float((base - stack)[airport == a].mean())} for a in sorted(set(airport))}}
    checks = {"1_every_seed_beats_mean_control": all(v > 0 for v in per_seed.values()),
              "2_both_months_positive": all(v > 0 for v in report["months"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_rmse_gain_vs_control_at_least_0_3_s": report["rmse_gain_vs_control_s"] >= 0.3,
              "6_stack_beats_base": report["rmse_gain_vs_base_s"] > 0}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

"""One focused main-lane weather experiment with equal-width permutation controls and paired training seeds.

Rules and decision: research_300/PREREG_weather.md, committed before the first fit. Private research: no production
change, no candidate, no upload.

Why this shape. The earlier main-lane weather evidence was a residual diagnostic on v45's holdout (bin corrections,
at most 211 total MSE, research_240/main_weather_diagnostic.py), never a fit with weather as features; the fallback
weather screen was rejected and is not repeated. And the single-seed main-lane screen confounds a small feature
effect with the column count (STATUS.md, 23 September, as corrected), so the real arm is compared with controls of
the SAME width, and every arm is fitted with the same three training seeds.

Arms, same rows and hyperparameters, fixed 1,200 rounds (no early stopping, so no held-out label picks the model):
  real        the 91 release features + the 8 METAR columns of research_240/features_weather.WEATHER
  control_k   the same 8 columns permuted JOINTLY within airport and calendar month, k = 1, 2, 3; learning and
              held-out rows use independent permutation seeds
  base        the 91 features alone: descriptive only, a different width, never the comparator
Training seeds 177, 277, 377. Learning months 2-6 and 8-12, held out 1 and 7, main lane, target y - ersatz_s,
prediction max(ersatz_s + model, 300), as in the repository's screens. METAR: cached IEM reports
(.cache/research_240/weather), latest at or before the movement time, at most 3 h old. Every fit runs under the
shared heavy-work lock, one at a time. Predictions are cached per arm and seed, so a rerun resumes.

Run from the repository root:  python3 -u -B research_300/weather_screen.py [1_7 | 3_9]
The default 1_7 is the pre-registered screen; 3_9 is stage 1 of its confirmation (PREREG_weather.md addendum B),
using the release's prepared tables (learning months 2, 4, 5, 6, 8, 10, 11, 12; held out 3 and 9).
"""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main_model as H  # torch before tree libraries on macOS
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_240 import features_weather as W
from research_250 import candidate_features as B
from research_300 import incremental as I
from research_300.heavy_lock import heavy_work

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / ".cache/clock_queue_release_20260912/data"
METAR = ROOT / ".cache/research_240/weather"
PAIRS = {"1_7": ((2, 3, 4, 5, 6, 8, 9, 10, 11, 12), (1, 7)), "3_9": ((2, 4, 5, 6, 8, 10, 11, 12), (3, 9))}
PAIR = sys.argv[1] if len(sys.argv) > 1 else "1_7"
CACHE = ROOT / ".cache/research_300/weather" / ("" if PAIR == "1_7" else PAIR)
OUTPUT = ROOT / ("research_300/results/weather_screen.json" if PAIR == "1_7" else f"research_300/results/weather_screen_{PAIR}.json")
BASE = B.main_features(H.MERKMALE_HAUPT)
WX = list(W.WEATHER)
SEEDS, CONTROLS, ROUNDS = (177, 277, 377), (1, 2, 3), 1200
EXTRA = ["y", "ersatz_s", "hat_aobt", "MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def permute_within(frame, columns, groups, seed):
    """Permute the given columns JOINTLY (same row order for all of them) within each group; returns a copy."""
    out = frame.copy()
    rng = np.random.default_rng(seed)
    values = frame[columns].to_numpy()
    new = values.copy()
    for idx in pd.Series(np.arange(len(frame))).groupby(np.asarray(groups)).indices.values():
        new[idx] = values[rng.permutation(idx)]
    out[columns] = new
    return out


def table(months, reports):
    frame = pd.read_parquet(RELEASE / ("months_" + "_".join(map(str, months)) + ".parquet"),
                            columns=list(dict.fromkeys(BASE + EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    wx = W.weather_for(frame.ADEP_mvt.astype(str).to_numpy(), frame.MVT_TIME_UTC_mvt, reports)
    for col in WX:
        frame[col] = wx[col].to_numpy(float)
    frame["_group"] = frame.ADEP_mvt.astype(str) + "|" + pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%m")
    return frame


def fit_predict(learn, held, features, seed, label):
    cache = CACHE / f"pred_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    learn, held = learn.copy(), held.copy()
    for column in H.KAT:
        if column in features:
            cats = pd.unique(learn[column].astype(str))
            learn[column] = pd.Categorical(learn[column].astype(str), categories=cats)
            held[column] = pd.Categorical(held[column].astype(str), categories=cats)
    with heavy_work(f"weather screen {PAIR} {label} seed {seed}", log=log):
        started = time.time()
        model = lgb.LGBMRegressor(objective="l2", n_estimators=ROUNDS, learning_rate=.03, num_leaves=255,
                                  min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                                  verbose=-1, n_jobs=8, random_state=seed)
        model.fit(learn[features], (learn.y - learn.ersatz_s).to_numpy())
        pred = np.maximum(held.ersatz_s.to_numpy(float) + model.predict(held[features]), 300)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, pred)
    log(f"{label} seed {seed}: {time.time() - started:.0f} s, RMSE {np.sqrt(np.mean((pred - held.y) ** 2)):.3f}")
    return pred


def main():
    reports = W.report_features(W.load_metar(METAR))
    learn, held = table(PAIRS[PAIR][0], reports), table(PAIRS[PAIR][1], reports)
    log(f"learn {len(learn):,} rows, held out {len(held):,}; weather present on "
        f"{learn[WX[0]].notna().mean():.1%} / {held[WX[0]].notna().mean():.1%}")
    preds = {}
    for seed in SEEDS:
        preds[("real", seed)] = fit_predict(learn, held, BASE + WX, seed, "real")
        for k in CONTROLS:
            pl = permute_within(learn, WX, learn._group, 1000 + k)
            ph = permute_within(held, WX, held._group, 2000 + k)
            preds[(f"control_{k}", seed)] = fit_predict(pl, ph, BASE + WX, seed, f"control_{k}")
        preds[("base", seed)] = fit_predict(learn, held, BASE, seed, "base")
    y = held.y.to_numpy(float)
    loss = {key: (p - y) ** 2 for key, p in preds.items()}
    real = np.mean([loss[("real", s)] for s in SEEDS], axis=0)
    ctrl = np.mean([loss[(f"control_{k}", s)] for s in SEEDS for k in CONTROLS], axis=0)
    diff = ctrl - real                                   # positive: the real weather columns help
    month = pd.to_datetime(held.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    per_seed = {s: float(np.mean([loss[(f"control_{k}", s)].mean() for k in CONTROLS]) - loss[("real", s)].mean())
                for s in SEEDS}
    rmse = lambda l: float(np.sqrt(l.mean()))
    blocks = (held.ADEP_mvt.astype(str) + "|" + pd.to_datetime(held.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")).to_numpy()
    report = {"rows": int(len(y)), "rmse": {"real": rmse(real), "mean_control": rmse(ctrl),
                                            "base_91_descriptive": rmse(np.mean([loss[("base", s)] for s in SEEDS], axis=0))},
              "rmse_gain_vs_control_s": rmse(ctrl) - rmse(real),
              "mse_gain_per_row": float(diff.mean()), "per_seed_mse_gain": per_seed,
              "months": {str(m): float(diff[month == m].mean()) for m in PAIRS[PAIR][1]},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, blocks),
              "airports": {a: float(diff[held.ADEP_mvt.astype(str).eq(a).to_numpy()].mean())
                           for a in sorted(held.ADEP_mvt.astype(str).unique())}}
    checks = {"every_seed_beats_mean_control": all(v > 0 for v in per_seed.values()),
              "both_held_out_months_positive": all(v > 0 for v in report["months"].values()),
              "without_best_100_positive": report["without_best_100"] > 0,
              "airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "rmse_gain_at_least_0_3_s": report["rmse_gain_vs_control_s"] >= 0.3}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

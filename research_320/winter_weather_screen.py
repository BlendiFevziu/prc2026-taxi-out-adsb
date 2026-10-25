"""Pre-registered winter-weather screen (research_320/PREREG_winter_weather.md). No candidate, no upload.

Pairs 2_8 and 6_12, learning on the ten other months, the release's all-year table filtered by month. Arms: real
(91 features + four winter METAR columns), three equal-width controls (the four columns permuted jointly within airport
and month), base (91). Seeds 177/277/377, fixed 1,200 rounds, each fit under the shared lock, predictions cached.
Judged on February and December only.
Run from the repository root:  python3 -u -B research_320/winter_weather_screen.py
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
from research_300.weather_screen import permute_within

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / ".cache/clock_queue_release_20260912/data/months_1_2_3_4_5_6_7_8_9_10_11_12.parquet"
METAR = ROOT / ".cache/research_240/weather"
CACHE = ROOT / ".cache/research_320/winter_weather"
OUT = ROOT / "research_320/results/winter_weather.json"
WX = ["wx_visibility_km", "wx_frozen", "wx_freezing", "wx_icing_risk"]
PAIRS = {"2_8": (2, 8), "6_12": (6, 12)}
WINTER = {"2_8": 2, "6_12": 12}
SEEDS, CONTROLS, ROUNDS = (177, 277, 377), (1, 2, 3), 1200
BASE = B.main_features(H.MERKMALE_HAUPT)
EXTRA = ["y", "ersatz_s", "hat_aobt", "MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def fit_predict(learn, held, features, seed, label):
    path = CACHE / f"pred_{label}_s{seed}.npy"
    if path.exists():
        return np.load(path)
    learn, held = learn.copy(), held.copy()
    for column in H.KAT:
        if column in features:
            cats = pd.unique(learn[column].astype(str))
            learn[column] = pd.Categorical(learn[column].astype(str), categories=cats)
            held[column] = pd.Categorical(held[column].astype(str), categories=cats)
    with heavy_work(f"winter weather {label} seed {seed}", log=log):
        started = time.time()
        model = lgb.LGBMRegressor(objective="l2", n_estimators=ROUNDS, learning_rate=.03, num_leaves=255,
                                  min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                                  verbose=-1, n_jobs=8, random_state=seed)
        model.fit(learn[features], (learn.y - learn.ersatz_s).to_numpy())
        pred = np.maximum(held.ersatz_s.to_numpy(float) + model.predict(held[features]), 300)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, pred)
    log(f"{label} seed {seed}: {time.time() - started:.0f} s, RMSE {np.sqrt(np.mean((pred - held.y) ** 2)):.3f}")
    return pred


def main():
    frame = pd.read_parquet(TABLE, columns=list(dict.fromkeys(BASE + EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    wx = W.weather_for(frame.ADEP_mvt.astype(str).to_numpy(), frame.MVT_TIME_UTC_mvt, W.report_features(W.load_metar(METAR)))
    for col in WX:
        frame[col] = wx[col].to_numpy(float)
    t = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
    frame["_month"] = t.dt.month.to_numpy()
    frame["_group"] = frame.ADEP_mvt.astype(str) + "|" + t.dt.strftime("%m")
    log(f"{len(frame):,} main-lane rows; weather present on {frame[WX[0]].notna().mean():.1%}")
    losses, winter_rows = {}, []
    for pair, months in PAIRS.items():
        learn = frame[~frame._month.isin(months)].reset_index(drop=True)
        held = frame[frame._month.isin(months)].reset_index(drop=True)
        y = held.y.to_numpy(float)
        preds = {}
        for seed in SEEDS:
            preds[("real", seed)] = fit_predict(learn, held, BASE + WX, seed, f"{pair}_real")
            for k in CONTROLS:
                pl, ph = permute_within(learn, WX, learn._group, 1000 + k), permute_within(held, WX, held._group, 2000 + k)
                preds[(f"control_{k}", seed)] = fit_predict(pl, ph, BASE + WX, seed, f"{pair}_control_{k}")
            preds[("base", seed)] = fit_predict(learn, held, BASE, seed, f"{pair}_base")
        loss = {key: (p - y) ** 2 for key, p in preds.items()}
        losses[pair] = (held, loss)
        winter_rows.append(held._month.eq(WINTER[pair]).to_numpy())
    # pooled winter rows across both pairs
    def pooled(fn):
        return np.concatenate([fn(losses[p][1])[w] for p, w in zip(PAIRS, winter_rows)])
    real = pooled(lambda L: np.mean([L[("real", s)] for s in SEEDS], axis=0))
    ctrl = pooled(lambda L: np.mean([L[(f"control_{k}", s)] for s in SEEDS for k in CONTROLS], axis=0))
    base = pooled(lambda L: np.mean([L[("base", s)] for s in SEEDS], axis=0))
    held_w = pd.concat([losses[p][0][w] for p, w in zip(PAIRS, winter_rows)], ignore_index=True)
    diff, vs_base = ctrl - real, base - real
    keep = np.ones(len(diff), bool)
    keep[np.argsort(diff, kind="stable")[-100:]] = False
    blocks = (held_w.ADEP_mvt.astype(str) + "|" + pd.to_datetime(held_w.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")).to_numpy()
    per_seed = {s: float(pooled(lambda L, s=s: np.mean([L[(f"control_{k}", s)] for k in CONTROLS], axis=0) - L[("real", s)]).mean())
                for s in SEEDS}
    rmse = lambda l: float(np.sqrt(l.mean()))
    report = {"winter_rows": int(len(diff)), "rmse_winter": {"real": rmse(real), "mean_control": rmse(ctrl), "base": rmse(base)},
              "gain_vs_control_mse": float(diff.mean()), "gain_vs_base_mse": float(vs_base.mean()),
              "per_seed_vs_control": per_seed,
              "months_vs_control": {"02": float(diff[held_w._month.eq(2).to_numpy()].mean()), "12": float(diff[held_w._month.eq(12).to_numpy()].mean())},
              "without_best_100_vs_control": float(diff[keep].mean()), "airport_day_ci_vs_control": I.block_bootstrap(diff, blocks),
              "reported_summer": {}}
    for pair, (held, L) in losses.items():
        summer = ~held._month.eq(WINTER[pair]).to_numpy()
        r = np.mean([L[("real", s)] for s in SEEDS], axis=0)[summer]
        b = np.mean([L[("base", s)] for s in SEEDS], axis=0)[summer]
        report["reported_summer"][pair] = {"rows": int(summer.sum()), "gain_vs_base_mse": float((b - r).mean())}
    rules = {"every_seed_beats_mean_control": all(v > 0 for v in per_seed.values()),
             "february_and_december_positive": min(report["months_vs_control"].values()) > 0,
             "without_best_100_positive": report["without_best_100_vs_control"] > 0,
             "airport_day_ci_above_zero": report["airport_day_ci_vs_control"][0] > 0,
             "beats_base": report["gain_vs_base_mse"] > 0,
             "minimum_200_mse_vs_base": report["gain_vs_base_mse"] >= 200}
    report["rules"], report["verdict"] = rules, "pass" if all(rules.values()) else "fail"
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

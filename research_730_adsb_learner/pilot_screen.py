"""Pilot: per-flight ADS-B ground-track features in a learned model, all of January and July 2025 (PREREG_pilot.md).

Arms: base (91 features), real (91 + 12 ADS-B), control_k (the 12 permuted jointly within airport and day, k = 1..3).
Four day-blocked folds over both months, two training seeds, fixed 1,200 rounds. Every fit under the shared heavy-work
lock; out-of-fold predictions cached per arm and seed, so a rerun resumes. Private research: no candidate, no upload.

Run from the repository root:  python3 -u -B research_730_adsb_learner/pilot_screen.py
"""
import glob
import json
from pathlib import Path
import sys
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main_model as H  # torch before tree libraries on macOS
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_250 import candidate_features as B
from research_300 import incremental as I
from research_300.heavy_lock import heavy_work
from research_730_adsb_learner import features_adsb as F

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / ".cache/clock_queue_release_20260912/data/months_1_7.parquet"
TRACKS = ROOT.parent / "adsb-lol/tracks"
CACHE = ROOT / ".cache/research_730"
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot_screen.json"
BASE = B.main_features(H.MERKMALE_HAUPT)
AD = list(F.MERKMALE)
SEEDS, CONTROLS, ROUNDS, FOLDS = (177, 277), (1, 2, 3), 1200, 4
EXTRA = ["y", "ersatz_s", "hat_aobt", "MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def build_table():
    cached = CACHE / "table_1_7.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    frame = pd.read_parquet(RELEASE, columns=list(dict.fromkeys(BASE + EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                     for p in sorted(glob.glob(str(ROOT / "data/training_2025-0[17]-01_*.parquet")))], ignore_index=True)
    raw = raw.drop_duplicates("MVT_ID_mvt")
    frame = frame.merge(raw[["MVT_ID_mvt", "CALLSIGN_flt"]], on="MVT_ID_mvt", how="left")
    files = sorted(glob.glob(str(TRACKS / "2025-01-??.parquet")) + glob.glob(str(TRACKS / "2025-07-??.parquet")))
    log(f"{len(frame):,} departures, {len(files)} track days")
    tracks = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ad = F.compute(frame, tracks)
    for c in AD:
        frame[c] = ad[c].to_numpy()
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cached, index=False)
    return frame


def fit_predict(table, fold, features, seed, label):
    cache = CACHE / f"oof_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    pred = np.zeros(len(table))
    for k in range(FOLDS):
        learn, held = table[fold != k].copy(), table[fold == k].copy()
        for column in list(H.KAT):
            if column in features:
                cats = pd.unique(learn[column].astype(str))
                learn[column] = pd.Categorical(learn[column].astype(str), categories=cats)
                held[column] = pd.Categorical(held[column].astype(str), categories=cats)
        with heavy_work(f"research_730 pilot {label} seed {seed} fold {k}", log=log):
            t0 = time.time()
            model = lgb.LGBMRegressor(objective="l2", n_estimators=ROUNDS, learning_rate=.03, num_leaves=255,
                                      min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                                      verbose=-1, n_jobs=8, random_state=seed)
            model.fit(learn[features], (learn.y - learn.ersatz_s).to_numpy())
            pred[fold == k] = np.maximum(held.ersatz_s.to_numpy(float) + model.predict(held[features]), 300)
        log(f"{label} seed {seed} fold {k}: {time.time() - t0:.0f} s")
    np.save(cache, pred)
    log(f"{label} seed {seed}: RMSE {np.sqrt(np.mean((pred - table.y) ** 2)):.3f}")
    return pred


def permute_within(frame, columns, groups, seed):
    out = frame.copy()
    rng = np.random.default_rng(seed)
    order = np.arange(len(frame))
    for idx in pd.Series(np.arange(len(frame))).groupby(np.asarray(groups)).indices.values():
        order[idx] = rng.permutation(idx)
    for column in columns:
        out[column] = frame[column].to_numpy()[order]
    return out


def main():
    table = build_table()
    date = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    fold = date.map(lambda d: zlib.crc32(d.encode()) % FOLDS).to_numpy()
    table["_day"] = table.ADEP_mvt.astype(str) + "|" + date
    airport, month = table.ADEP_mvt.astype(str).to_numpy(), pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    matched = table.ad_matched.eq(1).to_numpy()
    cover = {a: {"rows": int((airport == a).sum()), "matched": float(matched[airport == a].mean())} for a in sorted(set(airport))}
    log("fold sizes", np.bincount(fold).tolist(), "matched share", round(float(matched.mean()), 3), json.dumps(cover))
    preds = {}
    for seed in SEEDS:
        preds[("base", seed)] = fit_predict(table, fold, BASE, seed, "base")
        preds[("real", seed)] = fit_predict(table, fold, BASE + AD, seed, "real")
        for k in CONTROLS:
            ctrl = permute_within(table, AD, table._day, 1000 + k)
            preds[(f"control_{k}", seed)] = fit_predict(ctrl, fold, BASE + AD, seed, f"control_{k}")
    y = table.y.to_numpy(float)
    loss = {key: (p - y) ** 2 for key, p in preds.items()}
    real = np.mean([loss[("real", s)] for s in SEEDS], axis=0)
    ctrl = np.mean([loss[(f"control_{k}", s)] for s in SEEDS for k in CONTROLS], axis=0)
    base = np.mean([loss[("base", s)] for s in SEEDS], axis=0)
    diff = ctrl - real
    rmse = lambda l: float(np.sqrt(l.mean()))
    per_seed = {s: float(np.mean([loss[(f"control_{k}", s)].mean() for k in CONTROLS]) - loss[("real", s)].mean()) for s in SEEDS}
    report = {"prereg": "research_730_adsb_learner/PREREG_pilot.md", "rows": int(len(y)), "coverage": cover,
              "matched_share": float(matched.mean()),
              "rmse": {"real": rmse(real), "mean_control": rmse(ctrl), "base_91": rmse(base)},
              "rmse_gain_vs_control_s": rmse(ctrl) - rmse(real), "rmse_gain_vs_base_s": rmse(base) - rmse(real),
              "mse_gain_per_row": float(diff.mean()), "per_seed_mse_gain": per_seed,
              "months": {str(m): float(diff[month == m].mean()) for m in (1, 7)},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, table._day.to_numpy()),
              "airports": {a: {"rows": int((airport == a).sum()), "vs_control": float(diff[airport == a].mean()),
                               "vs_base": float((base - real)[airport == a].mean())} for a in sorted(set(airport))},
              "matched_rows": {"rows": int(matched.sum()), "vs_control": float(diff[matched].mean()), "vs_base": float((base - real)[matched].mean())},
              "unmatched_rows": {"rows": int((~matched).sum()), "vs_control": float(diff[~matched].mean()), "vs_base": float((base - real)[~matched].mean())}}
    checks = {"1_every_seed_beats_mean_control": all(v > 0 for v in per_seed.values()),
              "2_both_months_positive": all(v > 0 for v in report["months"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_rmse_gain_vs_control_at_least_0_3_s": report["rmse_gain_vs_control_s"] >= 0.3,
              "6_real_beats_the_base": report["rmse_gain_vs_base_s"] > 0}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

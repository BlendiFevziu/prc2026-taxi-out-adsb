"""Main-lane screen of the runway-configuration and destination-bearing columns against equal-width permutation
controls and the 91-feature base, three paired training seeds.

Rules and decision: research_370_konfig/PREREG_konfig.md, committed before the first fit. Private research: no
production change, no candidate, no upload. The protocol is research_300/weather_screen.py's (same prepared tables,
main-lane rows, seeds, hyperparameters and fixed 1,200 rounds) with the five columns of features_konfig.MERKMALE in
place of the METAR columns, and a sixth rule, because the weather columns passed against their controls on 1_7 while
losing to the plain base on 3_9: the real arm must also beat the 91-feature base.

Arms: real (91 + 5), control_k (the 5 permuted jointly within airport and calendar month, k = 1, 2, 3; learning and
held-out rows with independent seeds), base (91). Every fit under the shared heavy-work lock, one at a time;
predictions cached per arm and seed, so a rerun resumes.

Run from the repository root:  python3 -u -B research_370_konfig/konfig_screen.py [1_7 | 3_9]
Needs .cache/research_370/airports.csv (OurAirports, public domain); its SHA-256 goes into the report.
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
from research_250 import candidate_features as B
from research_300 import incremental as I
from research_300.heavy_lock import heavy_work
from research_370_konfig import features_konfig as K

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / ".cache/clock_queue_release_20260912/data"
COORDS = ROOT / ".cache/research_370/airports.csv"
PAIRS = {"1_7": ((2, 3, 4, 5, 6, 8, 9, 10, 11, 12), (1, 7)), "3_9": ((2, 4, 5, 6, 8, 10, 11, 12), (3, 9))}
PAIR = sys.argv[1] if len(sys.argv) > 1 else "1_7"
CACHE = ROOT / ".cache/research_370/screen" / PAIR
OUTPUT = ROOT / f"research_370_konfig/results/konfig_screen_{PAIR}.json"
BASE = B.main_features(H.MERKMALE_HAUPT)
KF = list(K.MERKMALE)
SEEDS, CONTROLS, ROUNDS = (177, 277, 377), (1, 2, 3), 1200
EXTRA = ["y", "ersatz_s", "hat_aobt", "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]
HUBS = ("LIRF", "LTFM", "LFPG")
MIN_BEARING_COVERAGE = 0.98


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def permute_within(frame, columns, groups, seed):
    """Permute the given columns JOINTLY (one row order for all) within each group, keeping each column's dtype."""
    out = frame.copy()
    rng = np.random.default_rng(seed)
    order = np.arange(len(frame))
    for idx in pd.Series(np.arange(len(frame))).groupby(np.asarray(groups)).indices.values():
        order[idx] = rng.permutation(idx)
    for column in columns:
        out[column] = frame[column].to_numpy()[order]
    return out


def raw_months(months):
    """Movements of the given 2025 months from the challenge's training files, label-free columns only."""
    paths = [p for m in months for p in sorted((ROOT / "data").glob(f"training_2025-{m:02d}-01_*.parquet"))]
    if len(paths) != len(months):
        raise SystemExit(f"expected {len(months)} training files, found {len(paths)}")
    return pd.concat([pd.read_parquet(p, columns=K.RAW_COLUMNS) for p in paths], ignore_index=True)


def table(months, coords):
    frame = pd.read_parquet(RELEASE / ("months_" + "_".join(map(str, months)) + ".parquet"),
                            columns=list(dict.fromkeys(BASE + EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    kf = K.attach(frame, raw_months(months), coords)
    for column in KF:
        frame[column] = kf[column].to_numpy()
    frame["_group"] = frame.ADEP_mvt.astype(str) + "|" + pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%m")
    return frame


def coverage(frame):
    return {"bearing": float(frame.kf_peil_sin.notna().mean()), "konfig_wenig": float(frame.kf_konfig.eq(K.WENIG).mean()),
            "konfig_distinct": int(frame.kf_konfig.nunique()), "seit_capped": float(frame.kf_seit_umkehr_min.eq(K.CAP_MIN).mean()),
            "bis_capped": float(frame.kf_bis_umkehr_min.eq(K.CAP_MIN).mean()),
            "flow_missing": float(frame.kf_seit_umkehr_min.isna().mean())}


def fit_predict(learn, held, features, seed, label):
    cache = CACHE / f"pred_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    learn, held = learn.copy(), held.copy()
    for column in list(H.KAT) + list(K.KATEGORISCH):
        if column in features:
            cats = pd.unique(learn[column].astype(str))
            learn[column] = pd.Categorical(learn[column].astype(str), categories=cats)
            held[column] = pd.Categorical(held[column].astype(str), categories=cats)
    with heavy_work(f"research_370 konfig screen {PAIR} {label} seed {seed}", log=log):
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
    if not COORDS.exists():
        raise SystemExit(f"coordinates missing: {COORDS} (OurAirports airports.csv, {K.COORDS_URL})")
    coords = K.load_coords(COORDS)
    learn, held = table(PAIRS[PAIR][0], coords), table(PAIRS[PAIR][1], coords)
    cover = {"learn": coverage(learn), "held_out": coverage(held)}
    log(f"learn {len(learn):,} rows, held out {len(held):,}; coverage {json.dumps(cover)}")
    source = {"coords_url": K.COORDS_URL, "coords_sha256": K.sha256(COORDS), "coords_codes": len(coords)}
    if min(cover["learn"]["bearing"], cover["held_out"]["bearing"]) < MIN_BEARING_COVERAGE:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps({"coverage": cover, "source": source, "verdict": "gate_failed"}, indent=2) + "\n")
        raise SystemExit(f"bearing coverage below {MIN_BEARING_COVERAGE}: {cover}")
    preds = {}
    for seed in SEEDS:
        preds[("real", seed)] = fit_predict(learn, held, BASE + KF, seed, "real")
        for k in CONTROLS:
            pl = permute_within(learn, KF, learn._group, 1000 + k)
            ph = permute_within(held, KF, held._group, 2000 + k)
            preds[(f"control_{k}", seed)] = fit_predict(pl, ph, BASE + KF, seed, f"control_{k}")
        preds[("base", seed)] = fit_predict(learn, held, BASE, seed, "base")
    y = held.y.to_numpy(float)
    loss = {key: (p - y) ** 2 for key, p in preds.items()}
    real = np.mean([loss[("real", s)] for s in SEEDS], axis=0)
    ctrl = np.mean([loss[(f"control_{k}", s)] for s in SEEDS for k in CONTROLS], axis=0)
    base = np.mean([loss[("base", s)] for s in SEEDS], axis=0)
    diff = ctrl - real                                   # positive: the real columns help
    month = pd.to_datetime(held.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    airport = held.ADEP_mvt.astype(str).to_numpy()
    near = ((held.kf_seit_umkehr_min <= 60) | (held.kf_bis_umkehr_min <= 60)).to_numpy()
    per_seed = {s: float(np.mean([loss[(f"control_{k}", s)].mean() for k in CONTROLS]) - loss[("real", s)].mean())
                for s in SEEDS}
    rmse = lambda l: float(np.sqrt(l.mean()))
    blocks = (held.ADEP_mvt.astype(str) + "|" + pd.to_datetime(held.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")).to_numpy()
    report = {"prereg": "research_370_konfig/PREREG_konfig.md", "pair": PAIR, "rows": int(len(y)), "coverage": cover,
              "source": source,
              "rmse": {"real": rmse(real), "mean_control": rmse(ctrl), "base_91": rmse(base)},
              "rmse_per_seed": {f"{arm}_s{s}": rmse(loss[(arm, s)]) for arm in ["real", "base"] + [f"control_{k}" for k in CONTROLS]
                                for s in SEEDS},
              "rmse_gain_vs_control_s": rmse(ctrl) - rmse(real), "rmse_gain_vs_base_s": rmse(base) - rmse(real),
              "mse_gain_per_row": float(diff.mean()), "per_seed_mse_gain": per_seed,
              "months": {str(m): float(diff[month == m].mean()) for m in PAIRS[PAIR][1]},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, blocks),
              "airports": {a: {"rows": int((airport == a).sum()), "vs_control": float(diff[airport == a].mean()),
                               "vs_base": float((base - real)[airport == a].mean())} for a in sorted(set(airport))},
              "hubs_vs_control": float(diff[np.isin(airport, HUBS)].mean()),
              "within_60_min_of_a_reversal": {"rows": int(near.sum()), "vs_control": float(diff[near].mean()),
                                              "vs_base": float((base - real)[near].mean())},
              "elsewhere": {"rows": int((~near).sum()), "vs_control": float(diff[~near].mean()),
                            "vs_base": float((base - real)[~near].mean())}}
    checks = {"1_every_seed_beats_mean_control": all(v > 0 for v in per_seed.values()),
              "2_both_held_out_months_positive": all(v > 0 for v in report["months"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_rmse_gain_vs_control_at_least_0_3_s": report["rmse_gain_vs_control_s"] >= 0.3,
              "6_real_beats_the_91_feature_base": report["rmse_gain_vs_base_s"] > 0}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

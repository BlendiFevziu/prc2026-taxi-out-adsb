"""Main-lane screen of the gate-retention columns against equal-width permutation controls and the 91-feature base, three
paired training seeds.

Rules and decision: research_720_retention/PREREG_retention.md, committed before the first fit. Private research: no
production change, no candidate, no upload. Protocol as research_370_konfig/konfig_screen.py (same prepared tables,
main-lane rows, seeds, hyperparameters and fixed 1,200 rounds) with the seven columns of features_retention.MERKMALE.

Arms: real (91 + 7), control_k (the 7 permuted jointly within airport and calendar month, k = 1, 2, 3; learning and
held-out rows with independent seeds), base (91). Every fit under the shared heavy-work lock, one at a time;
predictions cached per arm and seed, so a rerun resumes.

Run from the repository root:  python3 -u -B research_720_retention/retention_screen.py [1_7 | 3_9]
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
from research_720_retention import features_retention as R

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / ".cache/clock_queue_release_20260912/data"
PAIRS = {"1_7": ((2, 3, 4, 5, 6, 8, 9, 10, 11, 12), (1, 7)), "3_9": ((2, 4, 5, 6, 8, 10, 11, 12), (3, 9))}
PAIR = sys.argv[1] if len(sys.argv) > 1 else "1_7"
CACHE = ROOT / ".cache/research_720/screen" / PAIR
OUTPUT = ROOT / f"research_720_retention/results/retention_screen_{PAIR}.json"
BASE = B.main_features(H.MERKMALE_HAUPT)
RK = list(R.MERKMALE)
SEEDS, CONTROLS, ROUNDS = (177, 277, 377), (1, 2, 3), 1200
EXTRA = ["y", "ersatz_s", "hat_aobt", "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]
HUBS = ("LIRF", "LTFM", "LFPG")
MIN_COVERAGE = 0.98


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
    return pd.concat([pd.read_parquet(p, columns=R.RAW_COLUMNS) for p in paths], ignore_index=True)


def table(months):
    frame = pd.read_parquet(RELEASE / ("months_" + "_".join(map(str, months)) + ".parquet"),
                            columns=list(dict.fromkeys(BASE + EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    rk = R.attach(frame.MVT_ID_mvt.to_numpy(), raw_months(months))
    for column in RK:
        frame[column] = rk[column].to_numpy()
    frame["_group"] = frame.ADEP_mvt.astype(str) + "|" + pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%m")
    return frame


def coverage(frame):
    return {"wait_apt_present": float(frame.ret_wait_apt.notna().mean()), "wait_bahn_present": float(frame.ret_wait_bahn.notna().mean()),
            "delay15_present": float(frame.ret_delay15_s.notna().mean()),
            "wait_apt_mean": float(frame.ret_wait_apt.mean()), "roll_apt_mean": float(frame.ret_roll_apt.mean())}


def fit_predict(learn, held, features, seed, label):
    cache = CACHE / f"pred_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    learn, held = learn.copy(), held.copy()
    for column in list(H.KAT):
        if column in features:
            cats = pd.unique(learn[column].astype(str))
            learn[column] = pd.Categorical(learn[column].astype(str), categories=cats)
            held[column] = pd.Categorical(held[column].astype(str), categories=cats)
    with heavy_work(f"research_720 retention screen {PAIR} {label} seed {seed}", log=log):
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
    learn, held = table(PAIRS[PAIR][0]), table(PAIRS[PAIR][1])
    cover = {"learn": coverage(learn), "held_out": coverage(held)}
    log(f"learn {len(learn):,} rows, held out {len(held):,}; coverage {json.dumps(cover)}")
    if min(cover["learn"]["wait_apt_present"], cover["held_out"]["wait_apt_present"]) < MIN_COVERAGE:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps({"coverage": cover, "verdict": "gate_failed"}, indent=2) + "\n")
        raise SystemExit(f"coverage below {MIN_COVERAGE}: {cover}")
    preds = {}
    for seed in SEEDS:
        preds[("real", seed)] = fit_predict(learn, held, BASE + RK, seed, "real")
        for k in CONTROLS:
            pl = permute_within(learn, RK, learn._group, 1000 + k)
            ph = permute_within(held, RK, held._group, 2000 + k)
            preds[(f"control_{k}", seed)] = fit_predict(pl, ph, BASE + RK, seed, f"control_{k}")
        preds[("base", seed)] = fit_predict(learn, held, BASE, seed, "base")
    y = held.y.to_numpy(float)
    loss = {key: (p - y) ** 2 for key, p in preds.items()}
    real = np.mean([loss[("real", s)] for s in SEEDS], axis=0)
    ctrl = np.mean([loss[(f"control_{k}", s)] for s in SEEDS for k in CONTROLS], axis=0)
    base = np.mean([loss[("base", s)] for s in SEEDS], axis=0)
    diff = ctrl - real                                   # positive: the real columns help
    month = pd.to_datetime(held.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    airport = held.ADEP_mvt.astype(str).to_numpy()
    busy = (held.ret_wait_apt > held.ret_wait_apt.median()).to_numpy()
    per_seed = {s: float(np.mean([loss[(f"control_{k}", s)].mean() for k in CONTROLS]) - loss[("real", s)].mean())
                for s in SEEDS}
    rmse = lambda l: float(np.sqrt(l.mean()))
    blocks = (held.ADEP_mvt.astype(str) + "|" + pd.to_datetime(held.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")).to_numpy()
    report = {"prereg": "research_720_retention/PREREG_retention.md", "pair": PAIR, "rows": int(len(y)), "coverage": cover,
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
              "wait_above_median": {"rows": int(busy.sum()), "vs_control": float(diff[busy].mean()), "vs_base": float((base - real)[busy].mean())},
              "wait_at_or_below_median": {"rows": int((~busy).sum()), "vs_control": float(diff[~busy].mean()),
                                          "vs_base": float((base - real)[~busy].mean())}}
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

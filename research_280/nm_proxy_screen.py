"""Pre-registered main-lane screen: label-free proxies for the NM findings. Fixed 23.09.2026, before any fit.

Pre-registration, inventory and reasoning: research_280/NM_PROXY_INVENTORY.md. Features:
research_280/features_nm_proxy.py (tests: tests/test_nm_proxy_features.py). Three families, each added to the 91
release features:

  H1 provenance flags        AOBT_3 exactly equal to EOBT_1, IOBT, LOBT
  H2 precision signature     AOBT_3, ARVT_1, ARVT_3 carry a non-zero seconds field
  H3 NM-side regime          share of OTHER main-lane departures at the airport, strictly earlier in the last hour
                             and earlier the same UTC day, whose AOBT_3 equals a planned clock; and their count

Protocol, unchanged from research_280/reporting_state_main_screen.py: one LightGBM, 1400 rounds, learning rate
0.03, 255 leaves, min_child_samples 50, subsample and colsample 0.8, seed 177, early stopping 80 on the held-out
fold; learning months 2-6 and 8-12, held out 1 and 7; target y - ersatz_s, prediction max(ersatz_s + model, 300).
The baseline is fitted once and shared. Early stopping on the held-out fold selects the round count on validation
data; it is the project setting and is applied identically to every arm, controls included.

Arms per hypothesis: candidate, and five permutation controls in which the family's columns are shuffled JOINTLY
within every (split, airport, UTC date) group, separately in training and held-out rows, seeds 1 to 5.

DECISION, all six must hold for a hypothesis to justify a future full-ensemble run:
  R1 RMSE gain >= 0.3 s
  R2 MSE gain > 0 after removing the 100 most favourable rows
  R3 January and July both gain
  R4 candidate MSE gain exceeds every one of the five permutation-control gains
  R5 airport-day cluster bootstrap (2,000 resamples) 95 % interval of the MSE gain lies above zero
  R6 |standardised mean difference| of every new column between held-out 2025 and the 2026 ranking rows < 0.25

INTEGRITY, hard, the run aborts on failure:
  I1 model replay: the baseline reproduces the recorded 223.6976 within 0.05 s, and every saved model reloaded from
     disk reproduces its own predictions to 1e-6 s
  I2 row alignment: IDs unique; every table row found in the raw-built proxy table; one row order for every arm
  I3 train/validation ID separation: disjoint IDs; training IDs only from source months 2-6 and 8-12, held-out
     IDs only from 1 and 7, by strict filename start-month parsing
  I4 fallback preserved: only hat_aobt == 1 rows are modelled or scored
Feature invariance (I5) is covered by the unit tests.

Row-level predictions and models go to .cache/nm_proxy_screen/ (never committed); the manifest with their SHA-256
goes to research_280/results/. Each arm is checkpointed, so an interrupted run resumes without refitting.

Run from the repository root:  python3 -u -B research_280/nm_proxy_screen.py
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=int, default=8)
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_model as H  # torch before tree libraries on macOS
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_250 import candidate_features as B
from research_280 import features_nm_proxy as NP

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT / ".cache/clock_queue_release_20260912/data"
CACHE = PROJECT / ".cache/nm_proxy_screen"
OUTPUT = PROJECT / "research_280/results/nm_proxy_screen.json"
FEATURES = B.main_features(H.MERKMALE_HAUPT)
EXTRA = ["y", "ersatz_s", "hat_aobt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt", "ADEP_mvt"]
LEARN, HELD = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], [1, 7]
RECORDED_BASELINE, REPLAY_TOL = 223.6976292444155, 0.05
SEEDS = (1, 2, 3, 4, 5)
# Part of every checkpoint key. Bump it whenever the permutation or fitting code changes, or a resumed run
# would silently reload predictions made by different code.
RECIPE = "nm_proxy_screen/1"
RAW = ["MVT_ID_mvt", "PHASE_mvt"] + list(NP.REQUIRED)
started = time.time()


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def month_of(path):
    """Strict start-month parsing: training_2025-MM-01_... -> MM."""
    return int(Path(path).name.split("training_2025-")[1][:2])


def proxies_2025():
    """Proxy features for every 2025 departure, built separately per source month, keyed by MVT_ID."""
    parts = []
    for path in sorted(glob.glob(str(PROJECT / "data/training_2025-*.parquet"))):
        raw = pd.read_parquet(path, columns=RAW)
        dep = raw[raw.PHASE_mvt.astype(str).str.upper().str.startswith("DEP")].reset_index(drop=True)
        built = NP.baue(dep)
        built["MVT_ID_mvt"] = dep.MVT_ID_mvt.to_numpy()
        built["source_month"] = month_of(path)
        parts.append(built)
    table = pd.concat(parts, ignore_index=True)
    if table.MVT_ID_mvt.duplicated().any():
        raise SystemExit("I2 failed: duplicate MVT_ID across the 2025 source files")
    log(f"proxy features built for {len(table):,} departures across 12 source months")
    return table.set_index("MVT_ID_mvt")


def proxies_2026():
    """Same features on the 2026 ranking departures, built separately per calendar month, main lane only."""
    raw = pd.read_parquet(PROJECT / "data/ranking.parquet", columns=RAW)
    dep = raw[raw.PHASE_mvt.astype(str).str.upper().str.startswith("DEP")].reset_index(drop=True)
    month = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    parts = [NP.baue(dep[month == m].reset_index(drop=True)).assign(month=m) for m in np.unique(month)]
    main = dep.AOBT_3_flt.notna().to_numpy()
    out = pd.concat([p for p in parts], ignore_index=True)
    order = np.concatenate([np.where(month == m)[0] for m in np.unique(month)])
    return out[main[order]].reset_index(drop=True)


def table(months, proxies):
    frame = pd.read_parquet(RELEASE / ("months_" + "_".join(map(str, months)) + ".parquet"),
                            columns=list(dict.fromkeys(FEATURES + EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    found = frame.MVT_ID_mvt.isin(proxies.index)
    if not found.all():
        raise SystemExit(f"I2 failed: {int((~found).sum()):,} rows of months {months} missing from the proxy table")
    matched = proxies.loc[frame.MVT_ID_mvt.to_numpy()]
    for column in NP.MERKMALE:
        frame[column] = matched[column].to_numpy(float)
    frame["source_month"] = matched["source_month"].to_numpy()
    return frame


def airport_day(frame):
    """Integer code per (airport, UTC date). Vectorised; strftime on 1.7 million rows is far too slow."""
    day = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.normalize().to_numpy()
    keys = pd.DataFrame({"a": frame.ADEP_mvt.astype(str).to_numpy(), "d": day})
    return keys.groupby(["a", "d"], sort=False).ngroup().to_numpy()


def permuted(frame, columns, seed):
    """Shuffle `columns` jointly within (airport, UTC date) groups; one bijection per group."""
    codes = airport_day(frame)
    n = len(frame)
    base_order = np.lexsort((np.arange(n), codes))
    rand_order = np.lexsort((np.random.default_rng(seed).random(n), codes))
    source = np.empty(n, dtype=np.int64)
    source[base_order] = rand_order
    out = frame.copy()
    out[columns] = frame[columns].to_numpy()[source]
    return out


def fit(name, columns, train, valid, y_train, y_target, ersatz, valid_ids_sha):
    """Fit one arm, or reload its checkpoint if it exists for the same held-out rows."""
    CACHE.mkdir(parents=True, exist_ok=True)
    pred_path, meta_path = CACHE / f"pred_{name}.npy", CACHE / f"pred_{name}.json"
    if pred_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get("recipe") == RECIPE and meta["valid_ids_sha256"] == valid_ids_sha
                and meta["columns"] == columns):
            log(f"{name}: reloaded checkpoint (best iteration {meta['best_iteration']})")
            return np.load(pred_path), meta, None
    model = lgb.LGBMRegressor(objective="l2", n_estimators=1400, learning_rate=.03, num_leaves=255,
                              min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                              verbose=-1, n_jobs=args.jobs, random_state=177)
    model.fit(train[columns], y_train, eval_set=[(valid[columns], y_target)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(80, verbose=False)])
    pred = np.maximum(ersatz + model.predict(valid[columns]), 300)
    importance = model.booster_.feature_importance("gain")
    model.booster_.save_model(str(CACHE / f"model_{name}.txt"))
    meta = {"recipe": RECIPE, "best_iteration": int(model.best_iteration_), "columns": columns,
            "valid_ids_sha256": valid_ids_sha, "prediction_sha256": sha(pred),
            "importance_gain": {c: float(g) for c, g in zip(columns, importance)}}
    np.save(pred_path, pred)
    meta_path.write_text(json.dumps(meta) + "\n")
    return pred, meta, model


def replay(name, valid, columns, ersatz, pred):
    """I1(b): the model written to disk and read back must reproduce the predictions. Runs on resumed arms too,
    since fit() writes the model file at training time and replay only ever reads it."""
    path = CACHE / f"model_{name}.txt"
    again = np.maximum(ersatz + lgb.Booster(model_file=str(path)).predict(valid[columns]), 300)
    diff = float(np.abs(again - pred).max())
    if diff > 1e-6:
        raise SystemExit(f"I1 failed: {name} reloaded from disk differs by {diff} s")
    return diff


def metrics(gain, valid, rng_seed=0):
    """Untrimmed MSE gain, month split, best-100 removal, airports, airport-day bootstrap."""
    month = pd.to_datetime(valid.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    airport = valid.ADEP_mvt.astype(str).to_numpy()
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    clusters = airport_day(valid)
    c_gain = np.bincount(clusters, weights=gain)
    c_n = np.bincount(clusters).astype(float)
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, len(c_n), size=(args.bootstrap, len(c_n)))
    boot = c_gain[draws].sum(axis=1) / c_n[draws].sum(axis=1)
    return {"mse_gain": float(gain.mean()), "mse_gain_without_top100": float(gain[keep].mean()),
            "january_mse_gain": float(gain[month == 1].mean()), "july_mse_gain": float(gain[month == 7].mean()),
            "rows_improved": int((gain > 0).sum()), "rows_worsened": int((gain < 0).sum()),
            "airport_day_clusters": int(len(c_n)),
            "bootstrap_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "airport_mse_gain": {a: float(gain[airport == a].mean()) for a in sorted(np.unique(airport))}}


def smd(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0


def main():
    proxies = proxies_2025()
    train, valid = table(LEARN, proxies), table(HELD, proxies)

    # I2, I3, I4
    for name, frame in (("train", train), ("valid", valid)):
        if frame.MVT_ID_mvt.duplicated().any():
            raise SystemExit(f"I2 failed: duplicate IDs in {name}")
        if not frame.hat_aobt.eq(1).all():
            raise SystemExit(f"I4 failed: fallback rows in {name}")
    overlap = set(train.MVT_ID_mvt) & set(valid.MVT_ID_mvt)
    bad_train = sorted(set(train.source_month) - set(LEARN))
    bad_valid = sorted(set(valid.source_month) - set(HELD))
    if overlap or bad_train or bad_valid:
        raise SystemExit(f"I3 failed: overlap {len(overlap)}, train months {bad_train}, valid months {bad_valid}")
    integrity = {"I2_ids_unique_and_all_found": True, "I3_id_overlap": 0,
                 "I3_train_source_months": sorted(map(int, set(train.source_month))),
                 "I3_valid_source_months": sorted(map(int, set(valid.source_month))),
                 "I4_main_lane_only": True, "train_rows": int(len(train)), "valid_rows": int(len(valid))}
    log(f"integrity I2 I3 I4 passed: {len(train):,} train, {len(valid):,} held-out rows")

    for column in H.KAT:
        if column in train.columns:
            categories = pd.unique(train[column].astype(str))
            train[column] = pd.Categorical(train[column].astype(str), categories=categories)
            valid[column] = pd.Categorical(valid[column].astype(str), categories=categories)
    coverage = {c: {"train_nonnull": float(train[c].notna().mean()), "valid_nonnull": float(valid[c].notna().mean()),
                    "valid_mean": float(np.nanmean(valid[c].to_numpy(float)))} for c in NP.MERKMALE}

    y_train = (train.y - train.ersatz_s).to_numpy()
    y_target = (valid.y - valid.ersatz_s).to_numpy()
    y = valid.y.to_numpy(float)
    ersatz = valid.ersatz_s.to_numpy(float)
    ids_sha = sha(valid.MVT_ID_mvt.to_numpy())
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))

    base_pred, base_meta, base_model = fit("baseline", FEATURES, train, valid, y_train, y_target, ersatz, ids_sha)
    base_rmse = rmse(base_pred)
    integrity["I1_baseline_rmse"] = base_rmse
    integrity["I1_recorded_baseline"] = RECORDED_BASELINE
    if abs(base_rmse - RECORDED_BASELINE) > REPLAY_TOL:
        raise SystemExit(f"I1 failed: baseline {base_rmse:.4f} against recorded {RECORDED_BASELINE:.4f}")
    integrity["I1_reload_max_diff_baseline"] = replay("baseline", valid, FEATURES, ersatz, base_pred)
    log(f"baseline rmse {base_rmse:.4f} (recorded {RECORDED_BASELINE:.4f}); I1 passed")

    p26 = proxies_2026()
    results = {}
    for family, columns in NP.FAMILIES.items():
        cols = FEATURES + columns
        cand_pred, cand_meta, cand_model = fit(f"{family}_candidate", cols, train, valid, y_train, y_target,
                                               ersatz, ids_sha)
        integrity[f"I1_reload_max_diff_{family}"] = replay(f"{family}_candidate", valid, cols, ersatz, cand_pred)
        gain = (base_pred - y) ** 2 - (cand_pred - y) ** 2
        entry = {"columns": columns, "baseline_rmse": base_rmse, "candidate_rmse": rmse(cand_pred),
                 "rmse_gain": base_rmse - rmse(cand_pred), "best_iteration": cand_meta["best_iteration"],
                 **metrics(gain, valid)}
        imp = cand_meta["importance_gain"]
        ranked = sorted(imp, key=imp.get, reverse=True)
        entry["importance_rank"] = {c: ranked.index(c) + 1 for c in columns}
        entry["importance_gain_share"] = {c: imp[c] / sum(imp.values()) for c in columns}

        controls = []
        for seed in SEEDS:
            ptrain, pvalid = permuted(train, columns, seed), permuted(valid, columns, seed + 100)
            if not (pvalid.MVT_ID_mvt.to_numpy() == valid.MVT_ID_mvt.to_numpy()).all():
                raise SystemExit("I2 failed: permutation changed the held-out row order")
            ppred, pmeta, _ = fit(f"{family}_perm{seed}", cols, ptrain, pvalid, y_train, y_target, ersatz, ids_sha)
            pg = (base_pred - y) ** 2 - (ppred - y) ** 2
            controls.append({"seed": seed, "mse_gain": float(pg.mean()), "rmse_gain": base_rmse - rmse(ppred),
                             "best_iteration": pmeta["best_iteration"]})
            log(f"{family} control seed {seed}: mse gain {pg.mean():+.2f}")
        entry["permutation_controls"] = controls

        held_mask = valid.source_month.isin(HELD).to_numpy()
        entry["drift_smd_heldout2025_vs_2026"] = {
            c: smd(valid.loc[held_mask, c].to_numpy(float), p26[c].to_numpy(float)) for c in columns}
        entry["drift_smd_learning_vs_heldout2025"] = {
            c: smd(train[c].to_numpy(float), valid[c].to_numpy(float)) for c in columns}

        checks = {"R1_rmse_gain_at_least_0_3_s": entry["rmse_gain"] >= .3,
                  "R2_survives_top100_removal": entry["mse_gain_without_top100"] > 0,
                  "R3_january_and_july_gain": entry["january_mse_gain"] > 0 and entry["july_mse_gain"] > 0,
                  "R4_beats_every_permutation_control": entry["mse_gain"] > max(c["mse_gain"] for c in controls),
                  "R5_bootstrap_interval_above_zero": entry["bootstrap_95"][0] > 0,
                  "R6_2025_2026_comparable": all(abs(v) < .25 for v in entry["drift_smd_heldout2025_vs_2026"].values())}
        entry["checks"] = checks
        entry["justifies_full_ensemble_run"] = bool(all(checks.values()))
        results[family] = entry
        log(f"{family}: rmse gain {entry['rmse_gain']:+.4f} s, mse gain {entry['mse_gain']:+.2f}, "
            f"controls {[round(c['mse_gain'], 1) for c in controls]}, passes {sum(checks.values())}/6")

    manifest = {name: json.loads((CACHE / f"pred_{name}.json").read_text())["prediction_sha256"]
                for name in ["baseline"] + [f"{f}_{s}" for f in NP.FAMILIES for s in
                                            ["candidate"] + [f"perm{k}" for k in SEEDS]]}
    report = {"validation": "single-LightGBM January/July main-lane screen with permutation controls; not the full "
                            "ensemble and not a leaderboard forecast",
              "preregistration": "research_280/NM_PROXY_INVENTORY.md",
              "integrity": integrity, "feature_coverage": coverage, "families": results,
              "predictions_sha256": manifest, "predictions_dir": str(CACHE.relative_to(PROJECT)),
              "seconds": time.time() - started}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print("\nSUMMARY")
    for family, e in results.items():
        print(f"  {family}: rmse {e['baseline_rmse']:.4f} -> {e['candidate_rmse']:.4f} ({e['rmse_gain']:+.4f} s), "
              f"mse {e['mse_gain']:+.1f}, top100 {e['mse_gain_without_top100']:+.1f}, "
              f"Jan {e['january_mse_gain']:+.1f}, Jul {e['july_mse_gain']:+.1f}, "
              f"boot {[round(v, 1) for v in e['bootstrap_95']]}, "
              f"best control {max(c['mse_gain'] for c in e['permutation_controls']):+.1f}, "
              f"passes {sum(e['checks'].values())}/6")
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()

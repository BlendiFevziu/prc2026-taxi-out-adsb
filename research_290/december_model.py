"""Out-of-sample comparator for December 2025: one LightGBM on source months 1-11, predicting December's main lane.

Why it is needed. research_290/PREREG_stand_detector.md compares the frozen stand detector with the model on
identical rows, but no saved out-of-sample prediction exists for December: every production model trained on it.

Settings are the repository's main-lane screen (research_280/reporting_state_main_screen.py): 255 leaves,
learning rate 0.03, min_child_samples 50, subsample and colsample 0.8, seed 177, the 91 release features, target
y - ersatz_s, prediction max(ersatz_s + model, 300). ONE deliberate difference, documented: a FIXED 1,200 rounds
with no early stopping, because the screens stop early on the held-out fold and here that fold is December itself,
whose labels must not choose the model. The screens' best iterations were 1,179 to 1,281.

This comparator is weaker than the production ensemble (January/July 2025 main lane: 223.70 against 216.77), so a
detector advantage measured against it is overstated for production.

Integrity, hard: IDs disjoint between training and December; training rows only from source months 1-11 and test
rows only from 12, by strict filename start-month parsing; main lane only. Predictions go to .cache (never
committed); the manifest with their SHA-256 goes to research_290/results/.

Run from the repository root:  python3 -u -B research_290/december_model.py
"""
import glob
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / ".cache/clock_queue_release_20260912/data/months_1_2_3_4_5_6_7_8_9_10_11_12.parquet"
CACHE = ROOT / ".cache/munich_stand_detector"
OUTPUT = ROOT / "research_290/results/december_model.json"
FEATURES = B.main_features(H.MERKMALE_HAUPT)
EXTRA = ["y", "ersatz_s", "hat_aobt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt", "ADEP_mvt"]
ROUNDS = 1200


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def source_months():
    """MVT_ID -> start month of the training file it came from."""
    parts = []
    for path in sorted(glob.glob(str(ROOT / "data/training_2025-*.parquet"))):
        month = int(Path(path).name.split("training_2025-")[1][:2])
        ids = pd.read_parquet(path, columns=["MVT_ID_mvt"]).MVT_ID_mvt
        parts.append(pd.Series(month, index=ids.to_numpy()))
    out = pd.concat(parts)
    if out.index.duplicated().any():
        raise SystemExit("duplicate MVT_ID across source files")
    return out


def main():
    started = time.time()
    frame = pd.read_parquet(TABLE, columns=list(dict.fromkeys(FEATURES + EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    months = source_months()
    missing = ~frame.MVT_ID_mvt.isin(months.index)
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} table rows have no source file")
    frame["source_month"] = months.loc[frame.MVT_ID_mvt.to_numpy()].to_numpy()
    train = frame[frame.source_month.between(1, 11)].reset_index(drop=True)
    test = frame[frame.source_month.eq(12)].reset_index(drop=True)
    overlap = set(train.MVT_ID_mvt) & set(test.MVT_ID_mvt)
    if overlap or train.MVT_ID_mvt.duplicated().any() or test.MVT_ID_mvt.duplicated().any():
        raise SystemExit(f"ID separation failed: overlap {len(overlap)}")
    log(f"train {len(train):,} rows (months 1-11), December {len(test):,} rows; IDs disjoint")

    for column in H.KAT:
        if column in train.columns:
            categories = pd.unique(train[column].astype(str))
            train[column] = pd.Categorical(train[column].astype(str), categories=categories)
            test[column] = pd.Categorical(test[column].astype(str), categories=categories)
    model = lgb.LGBMRegressor(objective="l2", n_estimators=ROUNDS, learning_rate=.03, num_leaves=255,
                              min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                              verbose=-1, n_jobs=8, random_state=177)
    model.fit(train[FEATURES], (train.y - train.ersatz_s).to_numpy())
    ersatz = test.ersatz_s.to_numpy(float)
    pred = np.maximum(ersatz + model.predict(test[FEATURES]), 300)

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "december_model.txt"
    model.booster_.save_model(str(path))
    again = np.maximum(ersatz + lgb.Booster(model_file=str(path)).predict(test[FEATURES]), 300)
    replay = float(np.abs(again - pred).max())
    if replay > 1e-6:
        raise SystemExit(f"model replay failed: {replay}")
    out = pd.DataFrame({"MVT_ID_mvt": test.MVT_ID_mvt.to_numpy(), "airport": test.ADEP_mvt.astype(str).to_numpy(),
                        "pred": pred, "ersatz_s": ersatz, "y": test.y.to_numpy(float)})
    out.to_parquet(CACHE / "december_model.parquet", index=False)
    y = out.y.to_numpy()
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    munich = out.airport.eq("EDDM").to_numpy()
    report = {"note": "out-of-sample comparator for December 2025; single LightGBM, weaker than production",
              "rounds": ROUNDS, "train_rows": int(len(train)), "december_rows": int(len(test)),
              "id_overlap": 0, "replay_max_diff_s": replay,
              "december_main_rmse": rmse(out.pred.to_numpy()), "december_main_rmse_ersatz_only": rmse(out.ersatz_s.to_numpy()),
              "munich_rows": int(munich.sum()),
              "munich_rmse": float(np.sqrt(np.mean((out.pred.to_numpy()[munich] - y[munich]) ** 2))),
              "prediction_sha256": hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest(),
              "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "seconds": round(time.time() - started, 1)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

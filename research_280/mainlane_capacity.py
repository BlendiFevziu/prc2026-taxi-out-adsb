"""Is the main lane data-limited, capacity-limited, or at its ceiling? Part two of the capacity study.

Part one (research_280/mainlane_noise_floor.py) could not bound the floor: conditioning on airport, stand,
runway, aircraft type and day moves the spread of `rest` only from 415.8 to 368.2 s, while the ensemble reaches
216.8, so the ensemble's power is in continuous and temporal features that near-replicate matching cannot hold
fixed. The remaining question has to be answered empirically.

Three curves, all on one single LightGBM predicting `rest` so the comparison is clean:
  learning curve   12.5 %, 25 %, 50 %, 100 % of the ten learning months at fixed capacity
  capacity curve   63, 255 and 1023 leaves at full data
  fit gap          training RMSE beside holdout RMSE at each point

Reading it: if the learning curve is flat at 100 %, more rows will not help and full NM archives buy coverage
rather than accuracy. If the capacity curve still falls at 1023 leaves, the released ensemble is under-fitted and
a bigger model is worth building. If training RMSE is far below holdout RMSE everywhere, the lane is
variance-limited and more averaging, not more capacity, is the answer.

Honest limits. One model, not the released ensemble, so absolute numbers sit above the ensemble's 216.77; the
reference point is the single-tree `rest` view at 223.731 from research_240/main_target_views. Rounds are fixed,
never chosen on the holdout. The holdout is January and July, untouched by every fit.

  python3 -u -B research_280/mainlane_capacity.py
"""
import json
import time
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import main_model as H
import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = PROJECT / ".cache/clock_queue_release_20260912/data"
OUTPUT = PROJECT / "research_280/results/mainlane_capacity.json"
ENSEMBLE, SINGLE_TREE = 216.773, 223.731
SEED = 177
RUNS = [("learning 12.5%", 0.125, 255, 1400),
        ("learning 25%", 0.25, 255, 1400),
        ("learning 50%", 0.50, 255, 1400),
        ("capacity 63 leaves", 1.0, 63, 1400),
        ("capacity 255 leaves", 1.0, 255, 1400),
        ("capacity 1023 leaves", 1.0, 1023, 1400),
        ("capacity 255 x 3000 rounds", 1.0, 255, 3000)]


def log(*p):
    print(time.strftime("%H:%M:%S"), *p, flush=True)


def main():
    features = [c for c in H.MERKMALE_HAUPT]
    columns = sorted(set(features) | {"y", "ersatz_s", "hat_aobt"})
    have = set(__import__("pyarrow.parquet", fromlist=["p"]).ParquetFile(DATA / "months_1_7.parquet").schema_arrow.names)
    columns = [c for c in columns if c in have]
    features = [c for c in features if c in have]
    learn = pd.read_parquet(DATA / "months_2_3_4_5_6_8_9_10_11_12.parquet", columns=columns)
    hold = pd.read_parquet(DATA / "months_1_7.parquet", columns=columns)
    learn = learn.loc[learn.hat_aobt.astype(int).eq(1)]
    hold = hold.loc[hold.hat_aobt.astype(int).eq(1)]
    for frame in (learn, hold):
        frame["rest"] = frame.y - frame.ersatz_s
    log(f"{len(features)} features | learn {len(learn):,} main rows | hold {len(hold):,}")

    y_hold = hold.y.to_numpy(float)
    ersatz_hold = hold.ersatz_s.to_numpy(float)
    results = []
    for name, fraction, leaves, rounds in RUNS:
        part = learn if fraction >= 1. else learn.sample(frac=fraction, random_state=SEED)
        model = lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.03, num_leaves=leaves,
                                  min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                                  verbose=-1, n_jobs=-1, random_state=SEED)
        started = time.time()
        model.fit(part[features], part.rest)
        seconds = time.time() - started
        train_pred = np.maximum(part.ersatz_s.to_numpy(float) + model.predict(part[features]), 300.)
        hold_pred = np.maximum(ersatz_hold + model.predict(hold[features]), 300.)
        entry = {"run": name, "rows": int(len(part)), "leaves": leaves, "rounds": rounds,
                 "train_rmse": float(np.sqrt(np.mean((train_pred - part.y.to_numpy(float)) ** 2))),
                 "hold_rmse": float(np.sqrt(np.mean((hold_pred - y_hold) ** 2))), "seconds": seconds}
        entry["fit_gap"] = entry["hold_rmse"] - entry["train_rmse"]
        results.append(entry)
        log(f"{name:28s} rows {len(part):>9,} leaves {leaves:>5} rounds {rounds:>5}  "
            f"train {entry['train_rmse']:7.2f}  hold {entry['hold_rmse']:7.2f}  "
            f"gap {entry['fit_gap']:+7.2f}  {seconds:.0f}s")
        OUTPUT.write_text(json.dumps({"note": "single LightGBM on rest; holdout January/July never fitted on; "
                                              "rounds fixed, never chosen on the holdout",
                                      "ensemble_rmse": ENSEMBLE, "single_tree_reference": SINGLE_TREE,
                                      "runs": results}, indent=2) + "\n")
    print()
    print(pd.DataFrame(results)[["run", "rows", "leaves", "rounds", "train_rmse", "hold_rmse", "fit_gap"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\nreleased ensemble on the same rows: {ENSEMBLE:.2f}; single-tree reference {SINGLE_TREE:.2f}")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()

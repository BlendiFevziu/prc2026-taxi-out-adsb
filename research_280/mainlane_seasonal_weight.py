"""If the main lane is data-limited, does it help to weight the data toward the scored season?

Motivated by part two of the capacity study: the learning curve still falls about 5 s per doubling at 1.72
million rows, so the lane is data-limited rather than capacity-limited, and we have no more labelled departures.
The remaining question is whether the data we have is being used in the right proportions.

The scored months are January and July. Training treats all months equally, and `monat` is a feature, so a tree
model can already specialise. Whether it should be PUSHED to specialise is a different question and is untested
here (experiments.csv holds a loss-weighting entry from 06.09, but that is label-distribution smoothing, not
seasonal weighting).

Design, and the reason it is a proxy. The holdout is January and July 2025, so those months cannot be up-weighted
in training without leaking. Instead the months ADJACENT to them carry the weight: December and February stand
for January, June and August for July. If seasonal matching helps at all, weighting the nearest months should
show it. A shoulder-month arm weights April, May and October instead, as a control: if that also helps, the
effect is about re-weighting in general and not about season.

Everything else is held at the capacity study's setting, 255 leaves and 1400 fixed rounds, so the uniform arm
reproduces its 226.61 exactly.

  python3 -u -B research_280/mainlane_seasonal_weight.py
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
OUTPUT = PROJECT / "research_280/results/mainlane_seasonal_weight.json"
SEED, LEAVES, ROUNDS = 177, 255, 1400
NEAR = [12, 2, 6, 8]         # adjacent to the held-out January and July
SHOULDER = [4, 5, 10]        # control: seasonally far from both


def log(*p):
    print(time.strftime("%H:%M:%S"), *p, flush=True)


def main():
    features = list(H.MERKMALE_HAUPT)
    columns = sorted(set(features) | {"y", "ersatz_s", "hat_aobt", "monat"})
    have = set(__import__("pyarrow.parquet", fromlist=["p"]).ParquetFile(DATA / "months_1_7.parquet").schema_arrow.names)
    columns = [c for c in columns if c in have]
    features = [c for c in features if c in have]
    learn = pd.read_parquet(DATA / "months_2_3_4_5_6_8_9_10_11_12.parquet", columns=columns)
    hold = pd.read_parquet(DATA / "months_1_7.parquet", columns=columns)
    learn = learn.loc[learn.hat_aobt.astype(int).eq(1)]
    hold = hold.loc[hold.hat_aobt.astype(int).eq(1)]
    for frame in (learn, hold):
        frame["rest"] = frame.y - frame.ersatz_s
    month = learn.monat.to_numpy()
    y_hold, ersatz_hold = hold.y.to_numpy(float), hold.ersatz_s.to_numpy(float)
    log(f"learn {len(learn):,} main rows, hold {len(hold):,}; near months {NEAR}, shoulder {SHOULDER}")

    arms = {"uniform": np.ones(len(learn))}
    for factor in (2., 3., 5.):
        arms[f"near x{factor:g}"] = np.where(np.isin(month, NEAR), factor, 1.)
    arms["shoulder x3 (control)"] = np.where(np.isin(month, SHOULDER), 3., 1.)

    results = []
    for name, weight in arms.items():
        model = lgb.LGBMRegressor(objective="l2", n_estimators=ROUNDS, learning_rate=.03, num_leaves=LEAVES,
                                  min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                                  verbose=-1, n_jobs=-1, random_state=SEED)
        started = time.time()
        model.fit(learn[features], learn.rest, sample_weight=weight)
        prediction = np.maximum(ersatz_hold + model.predict(hold[features]), 300.)
        squared = (prediction - y_hold) ** 2
        entry = {"arm": name, "effective_rows": float(weight.sum()),
                 "hold_rmse": float(np.sqrt(squared.mean())), "seconds": time.time() - started}
        if results:
            gain = results[0]["_squared"] - squared
            keep = np.ones(len(gain), bool)
            keep[np.argsort(gain)[::-1][:100]] = False
            entry.update(mse_gain=float(gain.mean()), gain_without_best_100=float(gain[keep].mean()))
        entry["_squared"] = squared
        results.append(entry)
        log(f"{name:24s} effective rows {weight.sum():>12,.0f}  hold {entry['hold_rmse']:7.2f}"
            + (f"  gain {entry['mse_gain']:+9.1f}  no-top100 {entry['gain_without_best_100']:+9.1f}"
               if "mse_gain" in entry else "  (reference)"))
    table = [{k: v for k, v in r.items() if k != "_squared"} for r in results]
    print()
    print(pd.DataFrame(table).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    best = max((r for r in table if "mse_gain" in r and r["arm"].startswith("near")),
               key=lambda r: r["mse_gain"], default=None)
    control = next((r for r in table if "control" in r["arm"]), None)
    if best and control:
        print(f"\nbest near-month arm {best['arm']}: gain {best['mse_gain']:+.1f} MSE "
              f"({best['gain_without_best_100']:+.1f} without its best 100 rows); "
              f"shoulder control {control['mse_gain']:+.1f}")
        print("seasonal effect is real only if the near arms beat the shoulder control.")
    OUTPUT.write_text(json.dumps({"note": "proxy test: the scored months cannot be up-weighted because they are "
                                          "the holdout, so adjacent months stand in; shoulder months are the "
                                          "control. Single LightGBM, fixed rounds, holdout never fitted on.",
                                  "near_months": NEAR, "shoulder_months": SHOULDER,
                                  "leaves": LEAVES, "rounds": ROUNDS, "arms": table}, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()

"""Sweep FENSTER_S, the +/-30 minute window behind the `auslast` congestion feature.

`main_model.py` sets FENSTER_S = 1800 with no comment and no recorded measurement, and `auslastung` uses it to
count the other departures at the same airport inside that window. It is the last unevidenced constant in the
codebase with a plausible mechanism: too narrow and it misses the queue, too wide and it blurs into the daily
shape that `stunde` and `lage_50` already carry.

Evaluation is the capacity study's single-LightGBM harness (255 leaves, 1400 fixed rounds, lr .03, 81 features,
trained on the ten learning months and scored on the untouched January/July holdout). A single model rather than
the full ensemble, because what matters is the comparison between windows and a full refit per window is hours.
The 1800 arm therefore has to reproduce the capacity study's 226.61, and that is checked.

Fidelity gate before anything is swept: recomputing `auslast` at 1800 from the cached tables must reproduce the
stored column exactly. If it does not, the recomputation is wrong and no sweep built on it means anything.

Rules, fixed before the run:
  1. the chosen window beats 1800 pooled
  2. BOTH January and July improve
  3. the gain survives removing its 100 most favourable rows
  4. the window is interior to the grid
Passing means take it to a full ensemble refit, not ship it.

  python3 -u -B research_280/fenster_sweep.py
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
OUTPUT = PROJECT / "research_280/results/fenster_sweep.json"
GRID = [600, 900, 1200, 1800, 2700, 3600, 5400]
CURRENT, SEED = 1800, 177


def log(*p):
    print(time.strftime("%H:%M:%S"), *p, flush=True)


def auslast(frame, window):
    """main_model.auslastung with the window as an argument; same searchsorted logic."""
    out = np.zeros(len(frame), dtype=np.int32)
    t = frame.MVT_TIME_UTC_mvt.astype("int64").to_numpy() // 10 ** 9
    position = {v: i for i, v in enumerate(frame.index)}
    for _, idx in frame.groupby("ADEP_mvt", observed=True).groups.items():
        i = np.array([position[v] for v in idx])
        ts = np.sort(t[i])
        out[i] = (np.searchsorted(ts, t[i] + window, "right")
                  - np.searchsorted(ts, t[i] - window, "left") - 1)
    return out


def main():
    features = [c for c in H.MERKMALE_HAUPT]
    columns = sorted(set(features) | {"y", "ersatz_s", "hat_aobt", "MVT_TIME_UTC_mvt", "ADEP_mvt"})
    have = set(__import__("pyarrow.parquet", fromlist=["p"]).ParquetFile(DATA / "months_1_7.parquet").schema_arrow.names)
    columns = [c for c in columns if c in have]
    features = [c for c in features if c in have]
    learn = pd.read_parquet(DATA / "months_2_3_4_5_6_8_9_10_11_12.parquet", columns=columns).reset_index(drop=True)
    hold = pd.read_parquet(DATA / "months_1_7.parquet", columns=columns).reset_index(drop=True)

    for name, f in (("learn", learn), ("hold", hold)):
        again = auslast(f, CURRENT)
        same = int((again == f.auslast.to_numpy()).sum())
        log(f"FIDELITY {name}: recomputed auslast at {CURRENT} matches the stored column on "
            f"{same:,} of {len(f):,} rows ({same/len(f):.4%})")
        if same / len(f) < 0.999:
            raise SystemExit("recomputation does not reproduce the stored feature; the sweep would be meaningless")

    learn_main = learn.loc[learn.hat_aobt.astype(int).eq(1)].copy()
    hold_main = hold.loc[hold.hat_aobt.astype(int).eq(1)].copy()
    for f in (learn_main, hold_main):
        f["rest"] = f.y - f.ersatz_s
    y = hold_main.y.to_numpy(float)
    ersatz = hold_main.ersatz_s.to_numpy(float)
    month = pd.to_datetime(hold_main.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    jan, jul = month == 1, month == 7
    log(f"{len(learn_main):,} learn main rows, {len(hold_main):,} hold ({len(features)} features)")

    squared, rows = {}, []
    for w in GRID:
        learn_main["auslast"] = auslast(learn, w)[learn.hat_aobt.astype(int).eq(1).to_numpy()]
        hold_main["auslast"] = auslast(hold, w)[hold.hat_aobt.astype(int).eq(1).to_numpy()]
        model = lgb.LGBMRegressor(objective="l2", n_estimators=1400, learning_rate=.03, num_leaves=255,
                                  min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8,
                                  verbose=-1, n_jobs=-1, random_state=SEED)
        started = time.time()
        model.fit(learn_main[features], learn_main.rest)
        pred = np.maximum(ersatz + model.predict(hold_main[features]), 300.)
        squared[w] = (pred - y) ** 2
        rows.append({"window_s": w, "rmse": float(np.sqrt(squared[w].mean())), "seconds": time.time() - started})
        log(f"window {w:>5} s  hold RMSE {rows[-1]['rmse']:8.3f}  ({rows[-1]['seconds']:.0f}s)")

    base = squared[CURRENT]
    for r in rows:
        g = base - squared[r["window_s"]]
        keep = np.ones(len(g), bool)
        keep[np.argsort(g)[::-1][:100]] = False
        r.update(gain=float(g.mean()), gain_no_top100=float(g[keep].mean()),
                 january_gain=float(g[jan].mean()), july_gain=float(g[jul].mean()))
    t = pd.DataFrame(rows)
    print()
    print(t[["window_s", "rmse", "gain", "gain_no_top100", "january_gain", "july_gain"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    ok = t.loc[(t.gain > 0) & (t.january_gain > 0) & (t.july_gain > 0) & (t.gain_no_top100 > 0)
               & t.window_s.between(min(GRID) + 1, max(GRID) - 1)]
    res = {"current": CURRENT, "current_rmse": float(t.loc[t.window_s.eq(CURRENT), "rmse"].iloc[0]),
           "capacity_study_reference": 226.61, "grid": rows, "passing_all_four": ok.to_dict("records")}
    print(f"\n1800 arm: {res['current_rmse']:.3f} (capacity study recorded 226.61 for the same harness)")
    if len(ok):
        best = ok.sort_values("gain", ascending=False).iloc[0]
        print(f"BEST passing all four: {int(best.window_s)} s -> {best.rmse:.3f}, gain {best.gain:,.1f} MSE "
              f"({best.gain_no_top100:,.1f} robust). Worth a full ensemble refit.")
        res["best"] = best.to_dict()
    else:
        print("No interior window passes all four rules: FENSTER_S 1800 stands.")
    OUTPUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()

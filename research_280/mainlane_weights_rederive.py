"""Re-derive the main-lane blend weights ANTEIL_FLH and ANTEIL_CB, the last untested tuning lever.

ANTEIL_NETZ was stale and worth 0.33 s when moved (v50). Its two neighbours have never been re-derived for the
current model: ANTEIL_FLH (0.2, per-airport models) was tuned on one half of a holdout, ANTEIL_CB (0.6, CatBoost)
on an unrecorded basis, and both predate several changes to the ensemble.

Unlike the fallback grid, this operates on 339,046 rows where a gain is not concentrated by construction, so the
top-100 rule is a fair test here.

Step one, this file: run the validation-stage ensemble with each component ISOLATED and save the four residuals
(trees, per-airport, CatBoost, neural) on the January/July holdout. `ensemble_baseline.parquet` stores the first
three already blended, which is why they have to be recomputed. Nothing is fitted; the models are the ones that
produced the saved baseline.

Step two, the same file: grid the two weights at FIXED values, as the v50 correction requires, and judge them
with the pre-registered rules below.

Rules, fixed before the grid:
  1. the chosen pair beats the current (0.2, 0.6) pooled
  2. BOTH January and July improve at that same fixed pair
  3. the gain survives removing its 100 most favourable rows
  4. the pair is interior to the grid
Passing means take it to six pairs, not ship it.

  python3 -u -B research_280/mainlane_weights_rederive.py
"""
import json
import time
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import neural_model as NM  # noqa: F401  torch before tree libraries on macOS
import joblib
import numpy as np
import pandas as pd
import main_model as H
from research_250 import release_common as R

MODEL = PROJECT / ".cache/clock_queue_release_20260912/candidate/model.pkl"
FRAME = PROJECT / ".cache/clock_queue_release_20260912/data/months_1_7.parquet"
BASELINE = PROJECT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
PARTS = PROJECT / ".cache/sequence_release_v47/validate/components_main.parquet"
OUTPUT = PROJECT / "research_280/results/mainlane_weights_rederive.json"
FLH_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]
CB_GRID = [0.3, 0.45, 0.6, 0.75]


def log(*p):
    print(time.strftime("%H:%M:%S"), *p, flush=True)


def components():
    if PARTS.exists():
        log(f"reusing {PARTS.name}")
        return pd.read_parquet(PARTS)
    log("loading the validation-stage model (1.8 GB)")
    model = joblib.load(MODEL)
    frame = pd.read_parquet(FRAME)
    frame = R.SR.wende_an(model["standquote"], frame)
    hat = frame.hat_aobt.astype(int).eq(1).to_numpy()
    x = frame.loc[hat, model["merkmale_haupt"]]
    log(f"{int(hat.sum()):,} main rows, {len(model['merkmale_haupt'])} features")
    trees = model["mit_aobt"]
    trees = trees if isinstance(trees, list) else [trees]
    base = np.mean([t.predict(x) for t in trees], axis=0)
    log("trees done")
    local = base.copy()
    codes = frame.loc[hat, "ADEP_mvt"].astype(str).to_numpy()
    for airport, tree in model["mit_aobt_flh"].items():
        m = codes == airport
        if m.any():
            local[m] = tree.predict(x.loc[m])
    log("airport models done")
    cats = model["mit_aobt_cb"]
    cats = cats if isinstance(cats, (list, tuple)) else [cats]
    cat = np.mean([c.predict(H.kat_str(x)) for c in cats], axis=0)
    log("catboost done")
    out = pd.DataFrame({"MVT_ID_mvt": frame.loc[hat, "MVT_ID_mvt"].to_numpy(),
                        "trees": base, "local": local, "cat": cat,
                        "ersatz_s": frame.loc[hat, "ersatz_s"].to_numpy(float),
                        "gate": None, "schedule": None})
    out.to_parquet(PARTS, index=False)
    log(f"written {PARTS.name}")
    return out


def main():
    parts = components()
    base = pd.read_parquet(BASELINE)
    m = base.merge(parts[["MVT_ID_mvt", "trees", "local", "cat"]], on="MVT_ID_mvt", how="inner",
                   validate="one_to_one")
    log(f"joined {len(m):,} rows with the saved baseline")
    y = m.y.to_numpy(float)
    ersatz, gate, schedule = (m[c].to_numpy(float) for c in ("ersatz_s", "gate", "schedule"))
    net = m.old_neural_residual.to_numpy(float)
    trees, local, cat = (m[c].to_numpy(float) for c in ("trees", "local", "cat"))
    month = pd.to_datetime(m.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    jan, jul = month == 1, month == 7

    def predict(w_flh, w_cb, w_net=H.ANTEIL_NETZ):
        r = (1 - w_flh) * trees + w_flh * local
        r = (1 - w_cb) * r + w_cb * cat
        r = (1 - w_net) * r + w_net * net
        return np.maximum((1 - gate) * (ersatz + r) + gate * schedule, 300.)

    check = predict(H.ANTEIL_FLH, H.ANTEIL_CB)
    diff = float(np.abs(check - m.baseline.to_numpy(float)).max())
    log(f"REPLAY of the saved baseline at the current weights: max |diff| {diff:.4f} s")
    if diff > 1.0:
        raise SystemExit("component reconstruction does not reproduce the saved baseline; not grinding a grid on it")

    base_sq = (check - y) ** 2
    rows = []
    for a in FLH_GRID:
        for b in CB_GRID:
            sq = (predict(a, b) - y) ** 2
            g = base_sq - sq
            keep = np.ones(len(g), bool)
            keep[np.argsort(g)[::-1][:100]] = False
            rows.append({"anteil_flh": a, "anteil_cb": b, "rmse": float(np.sqrt(sq.mean())),
                         "gain": float(g.mean()), "gain_no_top100": float(g[keep].mean()),
                         "january_gain": float(g[jan].mean()), "july_gain": float(g[jul].mean())})
    t = pd.DataFrame(rows)
    print("\nmain-lane RMSE by (ANTEIL_FLH rows, ANTEIL_CB cols)")
    print(t.pivot(index="anteil_flh", columns="anteil_cb", values="rmse").to_string(float_format=lambda v: f"{v:,.3f}"))
    ok = t.loc[(t.gain > 0) & (t.january_gain > 0) & (t.july_gain > 0) & (t.gain_no_top100 > 0)
               & t.anteil_flh.between(min(FLH_GRID) + .01, max(FLH_GRID) - .01)
               & t.anteil_cb.between(min(CB_GRID) + .01, max(CB_GRID) - .01)]
    res = {"current": {"anteil_flh": H.ANTEIL_FLH, "anteil_cb": H.ANTEIL_CB,
                       "rmse": float(np.sqrt(base_sq.mean()))}, "replay_max_diff_s": diff, "grid": rows,
           "passing_all_four_rules": ok.to_dict("records")}
    print(f"\ncurrent ({H.ANTEIL_FLH}, {H.ANTEIL_CB}): RMSE {np.sqrt(base_sq.mean()):.4f}")
    if len(ok):
        best = ok.sort_values("gain", ascending=False).iloc[0]
        res["best"] = best.to_dict()
        per = best.gain * 0.98466
        print(f"BEST passing all four: FLH {best.anteil_flh}, CB {best.anteil_cb} -> {best.rmse:.4f}, "
              f"gain {best.gain:,.1f} MSE ({best.gain_no_top100:,.1f} robust), "
              f"Jan {best.january_gain:,.1f}, Jul {best.july_gain:,.1f}")
        print(f"  mechanical official: 266.1204 -> {np.sqrt(266.1204**2 - per):.4f} "
              f"(gain {266.1204 - np.sqrt(266.1204**2 - per):.3f} s), NOT a forecast")
    else:
        print("No interior point passes all four rules: the current setting stands.")
    OUTPUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()

"""Is our prediction tail under-dispersed relative to the optimum? A Discord hint, tested properly.

A competitor reported that what got them past this score band was "working deeply with the outliers in the
submission, specifically trying to force them to look as much like the training set outliers as possible".

The theory matters before the test. Under squared error the optimal predictor is the conditional mean, and
Var(E[y|x]) = Var(y) - E[Var(y|x)], so an OPTIMAL predictor is already narrower than the labels. Matching the two
distributions is therefore wrong in general and would hurt. The idea only pays if our predictor is under-dispersed
BEYOND that optimum, which is exactly the defect `s* = 1.237` exposed in the NM add-on's residual model today.

So this measures dispersion rather than assuming it, on the saved January/July holdout, main lane and fallback
together, reconstructed exactly (the fallback arm reproduces the production RMSE of 1756.91).

Two questions:
  1. diagnostic: how do predicted and observed quantiles compare in the upper tail?
  2. screen: does stretching predictions above a threshold improve holdout RMSE, and at what factor?

`submission.py` already has this lever (`RAND_SCHWELLE` 3600, `randfaktor`), and it was swept ON THE LEADERBOARD
on 10.09.2026 against a model scoring about 276, finding a sharp minimum at factor 1.0. It has never been
re-derived for the current model, and the leaderboard is no longer an acceptable place to tune it.

Rules, fixed before the run:
  1. the chosen (threshold, factor) beats factor 1.0 pooled
  2. BOTH January and July improve
  3. the gain survives removing its 100 most favourable rows
  4. the factor is interior to the grid
Passing means propose a candidate, not ship one.

  python3 -u -B research_280/tail_dispersion.py
"""
import json
import time
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import main_model as H
import features_stand_rate as MSQ
import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = PROJECT / ".cache/clock_queue_release_20260912/data"
BASELINE = PROJECT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
OUTPUT = PROJECT / "research_280/results/tail_dispersion.json"
THRESHOLDS = [1800, 2700, 3600, 5400, 7200]
FACTORS = [0.85, 0.95, 1.0, 1.05, 1.15, 1.30, 1.50]


def log(*p):
    print(time.strftime("%H:%M:%S"), *p, flush=True)


def fallback_predictions():
    columns = sorted(set(H.MERKMALE_RUECK) - set(MSQ.MERKMALE)
                     | {"y", "hat_aobt", "versp_min", "ADEP_mvt", "STAND_mvt", "monat"})
    learn = pd.read_parquet(DATA / "months_2_3_4_5_6_8_9_10_11_12.parquet", columns=columns)
    hold = pd.read_parquet(DATA / "months_1_7.parquet", columns=columns)
    for f in (learn, hold):
        f["hat_aobt"] = f.hat_aobt.astype(int)
        f["z"] = (f.y > H.SCHWELLE_EXTREM).astype(int)
    learn = learn.loc[learn.hat_aobt.eq(0)].copy()
    mapping = MSQ.passe_an(learn)
    learn = MSQ.ausser_faltung(learn)
    hold = MSQ.wende_an(mapping, hold)
    hold = hold.loc[hold.hat_aobt.eq(0)].copy()
    B = list(H.MERKMALE_RUECK)
    yl, extreme = learn.y.to_numpy(float), learn.z.eq(1).to_numpy()
    reg = lambda r, l, m: lgb.LGBMRegressor(objective="l2", n_estimators=r, learning_rate=.05, num_leaves=l,
                                            min_child_samples=m, verbose=-1, n_jobs=-1, random_state=0)
    p = (lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                            min_child_samples=20, verbose=-1, n_jobs=-1, random_state=0)
         .fit(learn[B], learn.z).predict_proba(hold[B])[:, 1] ** H.HOCH)
    height = ((1 - H.ANTEIL_HM) * hold.versp_min.to_numpy(float) * 60
              + H.ANTEIL_HM * reg(300, 15, 10).fit(learn.loc[extreme, B], yl[extreme]).predict(hold[B]))
    normal = reg(400, 31, 20).fit(learn.loc[~extreme, B], yl[~extreme]).predict(hold[B])
    pred = np.maximum((1 - p) * normal + p * height, 0)
    log(f"fallback arm reproduces RMSE {np.sqrt(np.mean((pred - hold.y.to_numpy(float))**2)):.2f} "
        f"(production baseline 1756.912)")
    return hold, pred


def main():
    main_rows = pd.read_parquet(BASELINE, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "baseline"])
    hold_fb, fb_pred = fallback_predictions()
    y = np.concatenate([main_rows.y.to_numpy(float), hold_fb.y.to_numpy(float)])
    pred = np.concatenate([main_rows.baseline.to_numpy(float), fb_pred])
    month = np.concatenate([pd.to_datetime(main_rows.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy(),
                            hold_fb.monat.to_numpy()])
    jan, jul = month == 1, month == 7
    log(f"{len(y):,} holdout rows, combined RMSE {np.sqrt(np.mean((pred-y)**2)):.3f}")

    print("\nupper-tail quantiles, predicted against observed:")
    print(f"{'quantile':>10s} {'predicted':>11s} {'observed':>11s} {'ratio':>7s}")
    diag = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999, 0.9999):
        a, b = float(np.quantile(pred, q)), float(np.quantile(y, q))
        diag.append({"q": q, "predicted": a, "observed": b, "ratio": a / b})
        print(f"{q:>10.4f} {a:>11,.0f} {b:>11,.0f} {a/b:>7.3f}")
    print("\nA ratio below 1 means the prediction tail is NARROWER than the labels, which an optimal")
    print("predictor is anyway. The screen below decides whether stretching it actually helps.")

    base_sq = (pred - y) ** 2
    rows = []
    for t in THRESHOLDS:
        over = pred > t
        for f in FACTORS:
            v = pred.copy()
            v[over] = t + f * (pred[over] - t)
            g = base_sq - (np.maximum(v, 0) - y) ** 2
            keep = np.ones(len(g), bool)
            keep[np.argsort(g)[::-1][:100]] = False
            rows.append({"threshold_s": t, "factor": f, "rows_touched": int(over.sum()),
                         "rmse": float(np.sqrt(np.mean((np.maximum(v, 0) - y) ** 2))),
                         "gain": float(g.mean()), "gain_no_top100": float(g[keep].mean()),
                         "january_gain": float(g[jan].mean()), "july_gain": float(g[jul].mean())})
    t = pd.DataFrame(rows)
    print("\ncombined holdout RMSE by (threshold, factor); factor 1.0 is the current setting")
    print(t.pivot(index="threshold_s", columns="factor", values="rmse").to_string(float_format=lambda v: f"{v:,.3f}"))
    ok = t.loc[(t.gain > 0) & (t.january_gain > 0) & (t.july_gain > 0) & (t.gain_no_top100 > 0)
               & t.factor.between(min(FACTORS) + .01, max(FACTORS) - .01) & t.factor.ne(1.0)]
    res = {"note": "holdout main lane plus reconstructed production fallback; factor 1.0 is what ships",
           "quantiles": diag, "grid": rows, "passing_all_four": ok.to_dict("records")}
    if len(ok):
        best = ok.sort_values("gain", ascending=False).iloc[0]
        V = 265.2839
        print(f"\nBEST passing all four: threshold {int(best.threshold_s)} s, factor {best.factor} "
              f"on {int(best.rows_touched):,} rows -> RMSE {best.rmse:.3f}, gain {best.gain:,.1f} MSE "
              f"({best.gain_no_top100:,.1f} robust)")
        print(f"  mechanical official: {V:.4f} -> {np.sqrt(max(V**2-best.gain,1)):.4f}, NOT a forecast")
        res["best"] = best.to_dict()
    else:
        print("\nNo interior factor passes all four rules: the tail is not under-dispersed in a usable way.")
    OUTPUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()

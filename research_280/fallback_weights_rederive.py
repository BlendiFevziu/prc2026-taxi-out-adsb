"""Re-derive the fallback blend weights, which were tuned on the leaderboard against a much older model.

`main_model.py` records that ANTEIL_HM (0.3) and HOCH (2.5) were read off the RANKING on 09.09.2026, when the
submission scored about 276. The model now scores 266.12, and the one other stale weight we tested today,
ANTEIL_NETZ, turned out to be worth 0.33 s when moved. These two govern the fallback blend

    p    = classifier probability ** HOCH
    high = (1 - ANTEIL_HM) * schedule delay + ANTEIL_HM * height model
    out  = (1 - p) * normal + p * high

and the fallback lane carries 39 to 51 % of official squared error, so a stale setting there is expensive.

Protocol follows the correction v50 forced on us: judge the FIXED pair we would actually ship, not a per-month
optimum. The component models are fitted once on the ten learning months and only the blend varies, so this is a
grid over arithmetic, not a retrain. The holdout is January and July and is never fitted on.

Rules, fixed before the grid runs:
  1. the chosen pair beats (2.5, 0.3) pooled
  2. BOTH January and July improve at that same fixed pair
  3. the gain survives removing its 100 most favourable rows
  4. the pair is not on the edge of the grid, so we are not extrapolating
Passing means take it to six pairs, not ship it.

  python3 -u -B research_280/fallback_weights_rederive.py
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
OUTPUT = PROJECT / "research_280/results/fallback_weights_rederive.json"
HOCH_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
HM_GRID = [0.0, 0.15, 0.30, 0.45, 0.60]
CURRENT = (H.HOCH, H.ANTEIL_HM)


def log(*p):
    print(time.strftime("%H:%M:%S"), *p, flush=True)


def main():
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
    y = hold.y.to_numpy(float)
    month = hold.monat.to_numpy()
    jan, jul = month == 1, month == 7
    log(f"learn {len(learn):,} fallback rows, hold {len(hold):,} (Jan {jan.sum():,}, Jul {jul.sum():,})")

    yl = learn.y.to_numpy(float)
    extreme = learn.z.eq(1).to_numpy()
    reg = lambda r, l, m: lgb.LGBMRegressor(objective="l2", n_estimators=r, learning_rate=.05, num_leaves=l,
                                            min_child_samples=m, verbose=-1, n_jobs=-1, random_state=0)
    prob = (lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                               min_child_samples=20, verbose=-1, n_jobs=-1, random_state=0)
            .fit(learn[B], learn.z).predict_proba(hold[B])[:, 1])
    height_model = reg(300, 15, 10).fit(learn.loc[extreme, B], yl[extreme]).predict(hold[B])
    normal = reg(400, 31, 20).fit(learn.loc[~extreme, B], yl[~extreme]).predict(hold[B])
    schedule = hold.versp_min.to_numpy(float) * 60
    log("components fitted once; the grid now varies only the blend")

    def predict(hoch, hm):
        p = prob ** hoch
        return np.maximum((1 - p) * normal + p * ((1 - hm) * schedule + hm * height_model), 0)

    base_sq = (predict(*CURRENT) - y) ** 2
    rows = []
    for hoch in HOCH_GRID:
        for hm in HM_GRID:
            sq = (predict(hoch, hm) - y) ** 2
            gain = base_sq - sq
            keep = np.ones(len(gain), bool)
            keep[np.argsort(gain)[::-1][:100]] = False
            rows.append({"hoch": hoch, "anteil_hm": hm, "rmse": float(np.sqrt(sq.mean())),
                         "gain": float(gain.mean()), "gain_no_top100": float(gain[keep].mean()),
                         "january_gain": float(gain[jan].mean()), "july_gain": float(gain[jul].mean())})
    t = pd.DataFrame(rows)
    print("\nfallback RMSE by (HOCH, ANTEIL_HM); current setting marked")
    piv = t.pivot(index="hoch", columns="anteil_hm", values="rmse")
    print(piv.to_string(float_format=lambda v: f"{v:,.1f}"))
    cur = t.loc[(t.hoch == CURRENT[0]) & (t.anteil_hm == CURRENT[1])].iloc[0]
    print(f"\ncurrent ({CURRENT[0]}, {CURRENT[1]}): RMSE {cur.rmse:.2f}")

    ok = t.loc[(t.gain > 0) & (t.january_gain > 0) & (t.july_gain > 0) & (t.gain_no_top100 > 0)
               & t.hoch.between(min(HOCH_GRID) + .01, max(HOCH_GRID) - .01)
               & t.anteil_hm.between(min(HM_GRID) + .001, max(HM_GRID) - .001)]
    result = {"current": {"hoch": CURRENT[0], "anteil_hm": CURRENT[1], "rmse": float(cur.rmse)},
              "grid": rows, "candidates_passing_all_four_rules": ok.to_dict("records")}
    if len(ok):
        best = ok.sort_values("gain", ascending=False).iloc[0]
        result["best"] = best.to_dict()
        print(f"\nBEST passing all four rules: HOCH {best.hoch}, ANTEIL_HM {best.anteil_hm} -> "
              f"RMSE {best.rmse:.2f} (from {cur.rmse:.2f}), gain {best.gain:,.0f} MSE "
              f"({best.gain_no_top100:,.0f} without best 100), Jan {best.january_gain:,.0f}, Jul {best.july_gain:,.0f}")
        per = best.gain * 5290 / 344841
        print(f"  mechanical official effect: 266.1204 -> {np.sqrt(266.1204**2 - per):.4f} "
              f"(gain {266.1204 - np.sqrt(266.1204**2 - per):.3f} s), NOT a forecast")
        result["mechanical_official"] = float(np.sqrt(266.1204 ** 2 - per))
    else:
        print("\nNo interior grid point passes all four rules: the current setting stands.")
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()

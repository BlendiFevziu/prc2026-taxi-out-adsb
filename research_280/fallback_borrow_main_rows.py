"""Can the fallback NORMAL branch learn from main-lane rows instead of only from its own 26,000?

The gap this asks about. `main_model.py` line 318 splits the training data with `lo = d[d.hat_aobt == 0]`, so the
fallback path learns from fallback rows alone while 1.74 million rows sit unused. The reason it cannot simply use
them is structural, and measured here rather than assumed: fallback rows are missing the whole flight-plan join,
not only the off-block time. On the January/July holdout, AIRCRAFT_OPERATOR_flt, FLIGHT_TYPE_flt,
MARKET_SEGMENT_flt, WK_TBL_CAT_flt, eobt_sched_min, ersatz_eobt_s and ersatz_s are NaN on 100 % of fallback rows
and on 0 % of main rows. A model trained on all rows would lean on those seven and then meet them absent.

So the candidate drops those seven everywhere and trains the normal branch on every row, fallback and main
together, giving it 1.74 million examples instead of about 26,000. The extreme classifier, the height model and
the blend are untouched: only the normal regressor changes, which is the part that serves the 92.8 % of fallback
rows below the extreme threshold and, per STATUS.md, carries 14.9 % of fallback squared error.

Why it might fail, stated before fitting: fallback rows are not missing at random. They are disrupted movements
(extreme rate 2.9 % against 0.04 % on main rows), so main-lane rows may describe a different population even
below the threshold. That is what the third arm controls for.

Arms:
  base        normal branch trained on fallback rows with z == 0, the scored recipe
  borrow      normal branch trained on ALL rows with z == 0, seven flight-plan features dropped
  borrow_eq   borrow, but on a random sample of main rows equal in size to the fallback set
              (separates "more data" from "different data"; seeds 101, 202, 303)

Production encoding throughout, as STATUS.md requires of every fallback screen: H.baue categories straight from
the cached month tables, real missing values, out-of-month stand rate through features_stand_rate.

Decision rule, fixed before any fit. This is a stage-one January/July screen, so passing means "take to six
pairs", never "build a candidate":
  1. pooled MSE gain on held-out fallback rows is positive
  2. it survives removing its 100 most favourable rows
  3. January and July both gain
  4. borrow beats every borrow_eq seed, otherwise the effect is the sample rather than the extra rows

  python3 -u -B research_280/fallback_borrow_main_rows.py
"""
import json
import time
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import main_model as H          # torch before tree libraries on macOS
import features_stand_rate as MSQ
import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = PROJECT / ".cache/clock_queue_release_20260912/data"
LEARN = DATA / "months_2_3_4_5_6_8_9_10_11_12.parquet"
HOLD = DATA / "months_1_7.parquet"
OUTPUT = PROJECT / "research_280/results/fallback_borrow_main_rows.json"
PLAN_ONLY = ["AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt",
             "eobt_sched_min", "ersatz_eobt_s", "ersatz_s"]
CONTROL_SEEDS = (101, 202, 303)
JOBS = 4


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                             min_child_samples=minimum, verbose=-1, n_jobs=JOBS, random_state=0)


def load(path, columns):
    frame = pd.read_parquet(path, columns=columns)
    frame["hat_aobt"] = frame.hat_aobt.astype(int)
    return frame


def main():
    columns = sorted(set(H.MERKMALE_RUECK) - set(MSQ.MERKMALE)
                     | {"y", "hat_aobt", "versp_min", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt"})
    learn, hold = load(LEARN, columns), load(HOLD, columns)
    log(f"learn {len(learn):,} rows, hold {len(hold):,} rows")
    learn["z"] = (learn.y > H.SCHWELLE_EXTREM).astype(int)
    hold["z"] = (hold.y > H.SCHWELLE_EXTREM).astype(int)

    # The stand rate is the share of EXTREME movements at a stand. Production builds it on fallback rows, where
    # the extreme rate is 2.9 %, not on main rows, where it is 0.04 %; building it on the pooled set would change
    # what the column means and break the base arm's fidelity to the scored recipe. So it is fitted on the
    # fallback learning rows exactly as production does, out of fold for those rows, and merely APPLIED to the
    # main learning rows and the holdout, which never entered it.
    fallback_learn = learn.loc[learn.hat_aobt.eq(0)].copy()
    mapping = MSQ.passe_an(fallback_learn)
    fallback_learn = MSQ.ausser_faltung(fallback_learn)
    main_learn = MSQ.wende_an(mapping, learn.loc[learn.hat_aobt.eq(1)])
    learn = pd.concat([fallback_learn, main_learn])
    hold = MSQ.wende_an(mapping, hold)
    log(f"stand rate fitted on {len(fallback_learn):,} fallback rows, base {mapping['grund']:.4f}")

    base_features = list(H.MERKMALE_RUECK)
    masked_features = [c for c in base_features if c not in PLAN_ONLY]
    log(f"{len(base_features)} features, {len(masked_features)} after dropping the flight-plan seven")

    learn_fb, hold_fb = fallback_learn, hold.loc[hold.hat_aobt.eq(0)]
    log(f"fallback: learn {len(learn_fb):,}, hold {len(hold_fb):,}; "
        f"normal rows learn {int(learn_fb.z.eq(0).sum()):,}")

    # classifier and height model are shared by every arm, fitted once on the fallback learning rows
    y_learn_fb = learn_fb.y.to_numpy(float)
    q = (lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                            min_child_samples=20, verbose=-1, n_jobs=JOBS, random_state=0)
         .fit(learn_fb[base_features], learn_fb.z).predict_proba(hold_fb[base_features])[:, 1] ** H.HOCH)
    extreme = learn_fb.z.eq(1).to_numpy()
    height = ((1 - H.ANTEIL_HM) * hold_fb.versp_min.to_numpy(float) * 60
              + H.ANTEIL_HM * reg(300, 15, 10).fit(learn_fb.loc[extreme, base_features],
                                                   y_learn_fb[extreme]).predict(hold_fb[base_features]))
    log("classifier and height model fitted")

    normal_rows = learn_fb.z.eq(0).to_numpy()
    arms = {"base": reg(400, 31, 20).fit(learn_fb.loc[normal_rows, base_features], y_learn_fb[normal_rows])
            .predict(hold_fb[base_features])}
    log("base fitted")

    all_normal = learn.loc[learn.z.eq(0)]
    arms["borrow"] = reg(400, 31, 20).fit(all_normal[masked_features], all_normal.y.to_numpy(float)) \
        .predict(hold_fb[masked_features])
    log(f"borrow fitted on {len(all_normal):,} rows")

    main_normal = all_normal.loc[all_normal.hat_aobt.eq(1)]
    for seed in CONTROL_SEEDS:
        sample = main_normal.sample(n=min(int(normal_rows.sum()), len(main_normal)), random_state=seed)
        pooled = pd.concat([learn_fb.loc[normal_rows], sample])
        arms[f"borrow_eq_{seed}"] = reg(400, 31, 20).fit(pooled[masked_features], pooled.y.to_numpy(float)) \
            .predict(hold_fb[masked_features])
        log(f"borrow_eq_{seed} fitted on {len(pooled):,} rows")

    y = hold_fb.y.to_numpy(float)
    month = pd.to_datetime(hold_fb.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    jan, jul = month == 1, month == 7
    squared = {}
    for name, normal in arms.items():
        prediction = np.maximum((1 - q) * normal + q * height, 0)
        squared[name] = (prediction - y) ** 2
    result = {"rows": int(len(hold_fb)), "learn_fallback_rows": int(len(learn_fb)),
              "borrow_training_rows": int(len(all_normal)), "features_base": len(base_features),
              "features_masked": len(masked_features), "dropped": PLAN_ONLY, "arms": {}}
    for name, sq in squared.items():
        gain = squared["base"] - sq
        keep = np.ones(len(gain), bool)
        keep[np.argsort(gain)[::-1][:100]] = False
        result["arms"][name] = {"rmse": float(np.sqrt(sq.mean())), "mse_gain_vs_base": float(gain.mean()),
                                "gain_without_best_100": float(gain[keep].mean()),
                                "january_gain": float(gain[jan].mean()), "july_gain": float(gain[jul].mean())}
    borrow, controls = result["arms"]["borrow"], [result["arms"][f"borrow_eq_{s}"] for s in CONTROL_SEEDS]
    result["checks"] = {"1_pooled_gain_positive": borrow["mse_gain_vs_base"] > 0,
                        "2_survives_top100_removal": borrow["gain_without_best_100"] > 0,
                        "3_both_months_gain": borrow["january_gain"] > 0 and borrow["july_gain"] > 0,
                        "4_beats_equal_size_controls": all(borrow["mse_gain_vs_base"] > c["mse_gain_vs_base"]
                                                           for c in controls)}
    result["take_to_six_pairs"] = all(result["checks"].values())
    print()
    print(pd.DataFrame(result["arms"]).T.to_string(float_format=lambda v: f"{v:,.2f}"))
    print("\nchecks " + json.dumps(result["checks"]))
    print("take to six pairs:", result["take_to_six_pairs"])
    OUTPUT.write_text(json.dumps({"note": "stage-one January/July screen, production encoding, fallback rows only "
                                          "in the evaluation; passing means take to six pairs, not build",
                                  **result}, indent=2, default=bool) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()

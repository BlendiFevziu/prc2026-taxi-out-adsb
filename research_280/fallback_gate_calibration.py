"""Is the fallback gate's probability calibrated, and is `p ** HOCH` the right blend weight?

Why this is not another variation of a rejected screen. Every previous fallback attempt changed the INPUTS
(new features, new anchors, new classifiers) or reshaped the probability with a single exponent. This asks
whether the number the gate already produces means what the blend assumes it means.

What the 50 worst holdout rows show (research_280 inspection, 22.09.2026): 35 of 50 are under-predictions
carrying 93.1 % of those rows' error, and the failure is symmetric in a telling way. Rows whose label equals the
schedule delay to within a few seconds, an unmistakable copy, come back with q = 0.000 (LIRF y 14,890 against
versp 14,885; 11,636 against 11,638; 7,679 against 7,679). Rows with q = 0.999 turn out to have y = 959 against
versp 14,339. A classifier with 600 trees on 17,097 rows returning p about 0.9996 on ambiguous flights is
overconfident, and `q = p ** 2.5` then commits to that overconfidence. Under squared loss, when the two
hypotheses differ by 10,000 s and nothing separates them, the optimal answer is the probability-weighted middle.

This matters out of all proportion to the row count: 50 rows of 344,419 carry 44.8 % of total holdout squared
error, and the fallback lane's worst 154 rows carry 95.7 % of its error.

Method. Production recipe throughout (H.baue categories, real missing values, out-of-month stand rate). The
learning months are cross-fitted in five folds to obtain out-of-fold p, normal and height; on those the
MSE-optimal blend weight per probability bin is measured directly, because for a bin
    w* = sum((y - normal) * (height - normal)) / sum((height - normal) ** 2)
minimises the squared error of w * height + (1 - w) * normal. A monotone (isotonic) map from p to w* is fitted
on the learning months only and applied to the holdout. Nothing is fitted on the holdout.

Arms: prod (q = p ** HOCH, the scored recipe), iso (the fitted map), and shrunk_<c> (q = c * p ** HOCH) as a
cheap control that separates "the map found real structure" from "any shrinkage would have helped".

Decision rule, fixed before fitting. Stage one on January/July; passing means take it to six pairs, not build:
  1. pooled MSE gain against prod is positive
  2. it survives removing its 100 most favourable rows
  3. January and July both gain
  4. iso beats the best constant-shrinkage control, otherwise the structure is just shrinkage

  python3 -u -B research_280/fallback_gate_calibration.py
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
from sklearn.isotonic import IsotonicRegression

DATA = PROJECT / ".cache/clock_queue_release_20260912/data"
LEARN = DATA / "months_2_3_4_5_6_8_9_10_11_12.parquet"
HOLD = DATA / "months_1_7.parquet"
OUTPUT = PROJECT / "research_280/results/fallback_gate_calibration.json"
SHRINK = (0.9, 0.75, 0.6, 0.5, 0.35)
FOLDS = 5
JOBS = 4


def log(*p):
    print(time.strftime("%H:%M:%S"), *p, flush=True)


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                             min_child_samples=minimum, verbose=-1, n_jobs=JOBS, random_state=0)


def clf():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=JOBS, random_state=0)


def components(train, test, features):
    """p, normal and height exactly as the production fallback path builds them."""
    y = train.y.to_numpy(float)
    extreme = train.z.eq(1).to_numpy()
    p = clf().fit(train[features], train.z).predict_proba(test[features])[:, 1]
    height = ((1 - H.ANTEIL_HM) * test.versp_min.to_numpy(float) * 60
              + H.ANTEIL_HM * reg(300, 15, 10).fit(train.loc[extreme, features], y[extreme]).predict(test[features]))
    normal = reg(400, 31, 20).fit(train.loc[~extreme, features], y[~extreme]).predict(test[features])
    return p, normal, height


def main():
    columns = sorted(set(H.MERKMALE_RUECK) - set(MSQ.MERKMALE)
                     | {"y", "hat_aobt", "versp_min", "ADEP_mvt", "STAND_mvt", "monat"})
    learn = pd.read_parquet(LEARN, columns=columns)
    hold = pd.read_parquet(HOLD, columns=columns)
    for frame in (learn, hold):
        frame["hat_aobt"] = frame.hat_aobt.astype(int)
        frame["z"] = (frame.y > H.SCHWELLE_EXTREM).astype(int)
    learn = learn.loc[learn.hat_aobt.eq(0)].copy()
    mapping = MSQ.passe_an(learn)
    learn = MSQ.ausser_faltung(learn)
    hold = MSQ.wende_an(mapping, hold)
    hold = hold.loc[hold.hat_aobt.eq(0)].copy()
    features = list(H.MERKMALE_RUECK)
    log(f"learn {len(learn):,} fallback rows, hold {len(hold):,}; extreme rate "
        f"{learn.z.mean():.2%} / {hold.z.mean():.2%}")

    # out-of-fold components on the learning months, folded by month so a month never trains on itself
    months = sorted(learn.monat.unique())
    blocks = [months[i::FOLDS] for i in range(FOLDS)]
    oof = np.zeros((len(learn), 3))
    for block in blocks:
        held = learn.monat.isin(block).to_numpy()
        oof[held] = np.column_stack(components(learn.loc[~held], learn.loc[held], features))
    log("out-of-fold components built on the learning months")

    p_oof, normal_oof, height_oof = oof[:, 0], oof[:, 1], oof[:, 2]
    y_oof = learn.y.to_numpy(float)
    span = height_oof - normal_oof
    keep = np.abs(span) > 1.0          # rows where the blend weight can matter at all
    target = np.clip((y_oof - normal_oof)[keep] / span[keep], -0.5, 1.5)
    weight = span[keep] ** 2           # weight by how much the choice costs, which is the MSE geometry
    iso = IsotonicRegression(y_min=0., y_max=1., increasing=True, out_of_bounds="clip")
    iso.fit(p_oof[keep], target, sample_weight=weight)
    log(f"isotonic map fitted on {int(keep.sum()):,} learning rows")

    grid = np.linspace(0, 1, 11)
    print("\n  p      p**HOCH   fitted w*")
    for value in grid:
        print(f"  {value:.2f}   {value ** H.HOCH:7.4f}   {float(iso.predict([value])[0]):9.4f}")

    p, normal, height = components(learn, hold, features)
    y = hold.y.to_numpy(float)
    month = hold.monat.to_numpy()
    jan, jul = month == 1, month == 7
    arms = {"prod": np.maximum((1 - p ** H.HOCH) * normal + (p ** H.HOCH) * height, 0)}
    w_iso = iso.predict(p)
    arms["iso"] = np.maximum((1 - w_iso) * normal + w_iso * height, 0)
    for c in SHRINK:
        q = c * p ** H.HOCH
        arms[f"shrunk_{c}"] = np.maximum((1 - q) * normal + q * height, 0)

    base = (arms["prod"] - y) ** 2
    result = {"learn_rows": int(len(learn)), "hold_rows": int(len(hold)), "arms": {}}
    for name, prediction in arms.items():
        squared = (prediction - y) ** 2
        gain = base - squared
        mask = np.ones(len(gain), bool)
        mask[np.argsort(gain)[::-1][:100]] = False
        result["arms"][name] = {"rmse": float(np.sqrt(squared.mean())), "mse_gain": float(gain.mean()),
                                "gain_without_best_100": float(gain[mask].mean()),
                                "january_gain": float(gain[jan].mean()), "july_gain": float(gain[jul].mean())}
    print()
    print(pd.DataFrame(result["arms"]).T.to_string(float_format=lambda v: f"{v:,.2f}"))

    best_shrunk = max((result["arms"][f"shrunk_{c}"]["mse_gain"] for c in SHRINK))
    a = result["arms"]["iso"]
    result["checks"] = {"1_pooled_gain_positive": a["mse_gain"] > 0,
                        "2_survives_top100_removal": a["gain_without_best_100"] > 0,
                        "3_both_months_gain": a["january_gain"] > 0 and a["july_gain"] > 0,
                        "4_beats_best_constant_shrinkage": a["mse_gain"] > best_shrunk}
    result["take_to_six_pairs"] = all(result["checks"].values())
    result["isotonic_map"] = {f"{v:.2f}": float(iso.predict([v])[0]) for v in grid}
    print("\nchecks " + json.dumps(result["checks"]))
    print("take to six pairs:", result["take_to_six_pairs"])
    OUTPUT.write_text(json.dumps({"note": "stage-one January/July screen, production encoding, fallback rows only; "
                                          "map fitted on the learning months out of fold", **result}, indent=2,
                                 default=bool) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()

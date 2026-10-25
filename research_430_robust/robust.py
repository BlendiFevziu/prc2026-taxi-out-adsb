"""Drift-robust partial-track correction: movement-only inputs and a second model family (research_430_robust/PREREG_robust.md).
Reuses research_320/volume_capacity.py for training set E, the B1/B2 cohorts, C1's recipe and the judge. No upload.
Run from the repository root:  python3 -u -B research_430_robust/robust.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import volume_capacity as V  # noqa: E402

S, MS, R = V.S, V.MS, V.R
from catboost import CatBoostRegressor  # noqa: E402

V.CAP.update({"c800": (800, 31)})
DROP = ["ground_points", "stand_min_m", "stand_first_m", "stand_median_m"]
CACHE = HERE / ".cache/research_430/models"
OUT = HERE / "research_430_robust/results/robust.json"
ROWS_2026, N, REF, ALPHA = 99059, 344841, 257.0986, 0.05 / 3


def x_of(frame, movement):
    x = MS.inputs(frame, "real")
    return x.drop(columns=DROP) if movement else x


def fit(train, name, kind, movement, tag):
    """Mean correction of three seeds; models cached per name/tag/seed."""
    target = (train.y - train.comparator).to_numpy(float)
    models = []
    for s in V.SEEDS:
        path = CACHE / f"{name}_{tag}_{s}.pkl"
        if not path.exists():
            with V.heavy_work(f"research_430 {name} {tag} seed {s}", poll_s=10.0):
                x = x_of(train, movement)
                m = (V.lgbm("c800", s).fit(x, target) if kind == "lgb" else
                     CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=10, loss_function="RMSE",
                                       thread_count=4, random_seed=s, verbose=0).fit(x, target))
                path.parent.mkdir(parents=True, exist_ok=True)
                R.joblib.dump(m, path)
        models.append(R.joblib.load(path))
    return models


def correction(models, frame, movement):
    x = x_of(frame, movement)
    return {s: m.predict(x) for s, m in zip(V.SEEDS, models)}


def arm_corrections(frame, bank):
    """Per-seed corrections of every arm on a frame; bank holds the fitted model lists."""
    c1 = bank["c1"](frame)
    lgb_m = correction(bank["lgb_m"], frame, True)
    cat_f = correction(bank["cat_f"], frame, False)
    cat_m = correction(bank["cat_m"], frame, True)
    return {"C1": c1, "M": lgb_m, "F": {s: (c1[s] + cat_f[s]) / 2 for s in V.SEEDS},
            "MF": {s: (lgb_m[s] + cat_m[s]) / 2 for s in V.SEEDS}}


def mean(c):
    return np.mean([c[s] for s in V.SEEDS], axis=0)


def main():
    cap = json.loads((HERE / "research_320/results/capacity2.json").read_text())
    if any(V.sha(V.model_path("c1", "real", s)) != cap["model_hashes"][f"c1_real_{s}.pkl"] for s in V.SEEDS):
        raise SystemExit("the frozen C1 models changed")
    train_s = S.extract(V.S_DATES, "train")
    extra = S.extract(V.E_EXTRA, "volume_extra")
    train_e = pd.concat([train_s[train_s.category.eq("partial_ground")], extra[extra.category.eq("partial_ground")]], ignore_index=True)
    if len(train_e) != cap["training_rows"]:
        raise SystemExit("training set E differs")
    with V.heavy_work("research_430: detector replay B1/B2", poll_s=10.0):
        b1 = V.eligible(S.extract(V.B1, "development"), V.detector_ids(V.B1))
        b2 = V.eligible(S.extract(V.B2, "confirm"), V.detector_ids(V.B2))
    frozen = lambda frame: V.seed_corrections(frame, "c1")
    bank = {"c1": frozen, "lgb_m": fit(train_e, "lgb_m", "lgb", True, "E"), "cat_f": fit(train_e, "cat_f", "cat", False, "E"),
            "cat_m": fit(train_e, "cat_m", "cat", True, "E")}
    report = {"prereg": "research_430_robust/PREREG_robust.md", "dropped": DROP, "rows": {"E": int(len(train_e)), "B1": int(len(b1)), "B2": int(len(b2))}}
    judged = {}
    for block, frame in (("B1", b1), ("B2", b2)):
        y = frame.y.to_numpy(float)
        se = lambda p: (p - y) ** 2
        corr = arm_corrections(frame, bank)
        ref = se(V.pred(frame, mean(corr["C1"])))
        for arm in ("M", "F", "MF"):
            seeds = {s: float((se(V.pred(frame, corr["C1"][s])) - se(V.pred(frame, corr[arm][s]))).mean()) for s in V.SEEDS}
            judged.setdefault(arm, {})[block] = V.judge(frame, ref, se(V.pred(frame, mean(corr[arm]))), seeds, ALPHA)
    # cross-season: every model refitted on one month of E, judged on the other month of B1+B2
    both = pd.concat([b1, b2], ignore_index=True)
    cross = {}
    for fit_month, test_month in (("01", "07"), ("07", "01")):
        tr = train_e[train_e.date.str[5:7].eq(fit_month)].reset_index(drop=True)
        te = both[both.date.str[5:7].eq(test_month)].reset_index(drop=True)
        tag = f"season{fit_month}"
        c1_models = fit(tr, "c1_refit", "lgb", False, tag)
        season_bank = {"c1": lambda frame, ms=c1_models: correction(ms, frame, False), "lgb_m": fit(tr, "lgb_m", "lgb", True, tag),
                       "cat_f": fit(tr, "cat_f", "cat", False, tag), "cat_m": fit(tr, "cat_m", "cat", True, tag)}
        y = te.y.to_numpy(float)
        corr = arm_corrections(te, season_bank)
        ref = (V.pred(te, mean(corr["C1"])) - y) ** 2
        cross[f"{fit_month}_to_{test_month}"] = {arm: float((ref - (V.pred(te, mean(corr[arm])) - y) ** 2).mean()) for arm in ("M", "F", "MF")}
        cross[f"{fit_month}_to_{test_month}"]["rows"] = int(len(te))
    report["cross_season"] = cross
    for arm in ("M", "F", "MF"):
        gain_b2 = judged[arm]["B2"]["gain_mse"]
        mse = ROWS_2026 * gain_b2 / N
        full_s = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
        rules = {"1_development_B1": judged[arm]["B1"]["pass"], "2_replication_B2": judged[arm]["B2"]["pass"],
                 "3_cross_season_both_directions": all(cross[k][arm] > 0 for k in ("01_to_07", "07_to_01")),
                 "4_planning_at_least_0_1_s": full_s >= 0.1}
        report[arm] = {"B1": {k: judged[arm]["B1"][k] for k in ("gain_mse", "without_best100", "interval", "by_month", "by_cell", "seeds", "rules")},
                       "B2": {k: judged[arm]["B2"][k] for k in ("gain_mse", "without_best100", "interval", "by_month", "by_cell", "seeds", "rules")},
                       "planning_full_s": full_s, "rules": rules, "verdict": "pass" if all(rules.values()) else "fail"}
        print(arm, report[arm]["verdict"], json.dumps({"B1": judged[arm]["B1"]["gain_mse"], "B2": gain_b2, "cross": {k: cross[k][arm] for k in cross}, "rules": rules}), flush=True)
    passing = [a for a in ("M", "F", "MF") if report[a]["verdict"] == "pass"]
    report["chosen"] = max(passing, key=lambda a: judged[a]["B1"]["gain_mse"]) if passing else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("chosen", report["chosen"], flush=True)


if __name__ == "__main__":
    main()

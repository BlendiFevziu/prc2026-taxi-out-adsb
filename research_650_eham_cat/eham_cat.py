"""Test of research_650_eham_cat/PREREG_eham_cat.md. Fits CatBoost once (saved). No upload.
Run from the repository root:  python3 -u -B research_650_eham_cat/eham_cat.py
"""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_480_weights", ROOT / "research_640_tail_rule"):
    sys.path.insert(0, str(p))
import weights as W  # noqa: E402
import catboost_layers as CL  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

V, S, L, H = W.V, W.S, W.L, W.H
ID, N, REF = "MVT_ID_mvt", 344841, 245.046
CACHE = ROOT / ".cache/research_650"
PLUS = ".cache/research_550/c1_more/c1plus_{seed}.pkl"
OUT = ROOT / "research_650_eham_cat/results/eham_cat.json"


def train_rows(base):
    tr = S.extract(tuple(S.MS.TRAIN), "train")
    f = tr[tr.airport.eq("EHAM") & tr.category.eq("partial_ground")].drop_duplicates(ID).reset_index(drop=True)
    f = f.merge(W.eham_features(tuple(S.MS.TRAIN)), on=ID, how="inner", validate="one_to_one", suffixes=("_ms", ""))
    W.check(f, base)
    return f


def fit(train):
    paths = [CACHE / f"cat_eham_{s}.pkl" for s in L.SEEDS]
    if not all(p.exists() for p in paths):
        CACHE.mkdir(parents=True, exist_ok=True)
        x, y = L.eham_inputs(train), (train.y - train.comparator).to_numpy(float)
        with V.heavy_work("research_650: CatBoost EHAM add-on", poll_s=10.0):
            for s, p in zip(L.SEEDS, paths):
                joblib.dump(CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=10, loss_function="RMSE",
                                              random_seed=s, thread_count=4, verbose=False).fit(x, y), p)
    return [joblib.load(p) for p in paths]


def test_frame(ext, days, base):
    f = W.keep_rows(ext, days, ("EHAM/01",))
    f = f.merge(W.eham_features(days), on=ID, how="inner", validate="one_to_one")
    W.check(f, base)
    return f


def gains(f, cat):
    y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
    xa = L.eham_inputs(f)
    addon = {s: m.predict(xa) for s, m in zip(L.SEEDS, L.load_models("eham_january"))}
    cb = {s: m.predict(xa) for s, m in zip(L.SEEDS, cat)}
    xp = L.inputs("partial", f)
    plus = {s: joblib.load(ROOT / PLUS.format(seed=s)).predict(xp) for s in L.SEEDS}
    se = lambda c: (np.maximum(b + c, 300.0) - y) ** 2
    m = lambda d: np.mean(list(d.values()), axis=0)
    g = se((m(addon) + m(plus)) / 2) - se(((m(addon) + m(cb)) / 2 + m(plus)) / 2)
    seeds = {s: float((se((addon[s] + plus[s]) / 2) - se(((addon[s] + cb[s]) / 2 + plus[s]) / 2)).mean()) for s in L.SEEDS}
    return g, seeds


def main():
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    train = train_rows(base)
    if len(train) != 3752:
        raise SystemExit(f"training rows {len(train)}, expected 3,752")
    x = L.eham_inputs(train)
    for s, m in zip(L.SEEDS, L.load_models("eham_january")):          # Codex's models must refit exactly on these rows
        r = lgb.LGBMRegressor(objective="regression", n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=80, reg_lambda=100.,
                              subsample=.8, subsample_freq=1, colsample_bytree=.9, max_bin=63, n_jobs=4, random_state=s, verbosity=-1)
        if not np.array_equal(r.fit(x, (train.y - train.comparator).to_numpy(float)).predict(x), m.predict(x)):
            raise SystemExit(f"Codex's seed {s} does not refit exactly")
    cat = fit(train)
    report = {"prereg": "research_650_eham_cat/PREREG_eham_cat.md", "train_rows": int(len(train))}
    dev_ext = pd.concat([S.extract(tuple(S.STAGES["development"]), "development"), S.extract(tuple(S.STAGES["confirm"]), "confirm")],
                        ignore_index=True).drop_duplicates(ID)
    for name, ext, days in (("development", dev_ext, W.DEV_EHAM), ("fresh", S.extract(W.JAN10, "fresh_january"), W.JAN10)):
        f = test_frame(ext, days, base)
        g, seeds = gains(f, cat)
        ci, nd = CL.day_bootstrap(g, (f.airport + "|" + f.date).to_numpy())
        report[name] = {"rows": int(len(f)), "gain_mse": float(g.mean()), "bootstrap_95": ci, "airport_days": nd, "seeds": seeds,
                        "by_day": {d: float(g[f.date.eq(d).to_numpy()].mean()) for d in sorted(f.date.unique())}}
    fr = report["fresh"]
    mse = 9752 * fr["gain_mse"] / N
    report["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    report["rules"] = {"pooled": fr["gain_mse"] > 0, "bootstrap_lower_above_zero": fr["bootstrap_95"][0] > 0,
                       "seven_of_ten_days": sum(v > 0 for v in fr["by_day"].values()) >= 7, "every_seed": all(v > 0 for v in fr["seeds"].values()),
                       "development_pooled": report["development"]["gain_mse"] > 0, "planning_0_02_s": report["planning_s"] >= 0.02}
    report["verdict"] = "pass" if all(report["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

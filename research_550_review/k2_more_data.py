"""Test of research_550_review/PREREG_k2_more_data.md. Fits K2+ once (saved), then scores. No upload.
Run from the repository root:  python3 -u -B research_550_review/k2_more_data.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
sys.path.insert(0, str(HERE / "research_550_review"))
from research_320 import stand_experiment as S  # noqa: E402
from research_320 import volume_capacity as V  # noqa: E402
from research_320 import stand_capacity as K  # noqa: E402
import stand_capacity_retest as RT  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, N, REF, ROWS_2026 = "MVT_ID_mvt", 344841, 252.91, 17147
MODELS = HERE / ".cache/research_550/k2_more/k2plus_{seed}.pkl"
OUT = HERE / "research_550_review/results/k2_more_data.json"


def training():
    parts = [S.extract(tuple(S.TRAIN), "train"), S.extract(V.E_EXTRA, "volume_extra"),
             S.extract(tuple(S.STAGES["development"]), "development"), S.extract(tuple(S.STAGES["confirm"]), "confirm")]
    return pd.concat([p[p.category.eq("observed_at_assigned_stand")] for p in parts], ignore_index=True).drop_duplicates(ID).reset_index(drop=True)


def models(train):
    paths = [Path(str(MODELS).format(seed=s)) for s in K.SEEDS]
    if not all(p.exists() for p in paths):
        target = (train.y - train.comparator).to_numpy(float)
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        with V.heavy_work("research_550: fit K2+ on E, B1 and B2", poll_s=10.0):
            for s, p in zip(K.SEEDS, paths):
                m = K.R.lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=.03, num_leaves=15,
                                          min_child_samples=80, reg_lambda=100., subsample=.8, subsample_freq=1, colsample_bytree=.9,
                                          max_bin=63, n_jobs=4, random_state=s, verbosity=-1).fit(S.inputs(train, "real"), target)
                joblib.dump(m, p)
    return [joblib.load(p) for p in paths]


def main():
    train = training()
    fresh = {d for _, ds in RT.BLOCKS for d in ds}
    if fresh & set(train.date):
        raise SystemExit("a test day is in the training set")
    plus = models(train)
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    f = RT.rows()
    if not np.array_equal(base.loc[f[ID]].to_numpy(float), f.comparator.to_numpy(float)):
        raise SystemExit("comparator not bit-identical")
    y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
    x = S.inputs(f, "real")
    k2 = K.corrections(f, "k2")
    new = {s: m.predict(x) for s, m in zip(K.SEEDS, plus)}
    se = lambda c: (np.maximum(b + c, 300.0) - y) ** 2
    g = se(np.mean([k2[s] for s in K.SEEDS], axis=0)) - se(np.mean([new[s] for s in K.SEEDS], axis=0))
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-100:]] = False
    by_day = {d: float(g[f.date.eq(d).to_numpy()].mean()) for d in sorted(f.date.unique())}
    by_cell = {c: float(g[f.cell.eq(c).to_numpy()].mean()) for c in RT.CELLS if f.cell.eq(c).any()}
    r = {"prereg": "research_550_review/PREREG_k2_more_data.md", "training_rows": int(len(train)), "test_rows": int(len(f)),
         "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
         "by_month": {m: float(g[f.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")},
         "by_cell": by_cell, "by_day": by_day, "seeds": {s: float((se(k2[s]) - se(new[s])).mean()) for s in K.SEEDS}}
    mse = ROWS_2026 * r["gain_mse"] / N
    r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    r["rules"] = {"pooled_positive": r["gain_mse"] > 0, "without_best100_positive": r["without_best100"] > 0,
                  "both_months": min(r["by_month"].values()) > 0,
                  "seventy_percent_of_days": np.mean([v > 0 for v in by_day.values()]) >= 0.7,
                  "two_thirds_of_cells": sum(v > 0 for v in by_cell.values()) >= 2 * len(by_cell) / 3,
                  "every_seed": all(v > 0 for v in r["seeds"].values()), "planning_at_least_0_1_s": r["planning_s"] >= 0.1}
    r["rules"] = {k: bool(v) for k, v in r["rules"].items()}
    r["verdict"] = "pass" if all(r["rules"].values()) else "fail"
    OUT.write_text(json.dumps(r, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "by_day"}, indent=1, default=str))


if __name__ == "__main__":
    main()

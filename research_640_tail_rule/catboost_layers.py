"""Test of research_640_tail_rule/PREREG_catboost_layers.md. Fits CatBoost once per family (saved). No upload.
Run from the repository root:  python3 -u -B research_640_tail_rule/catboost_layers.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_550_review", ROOT / "research_610_more_data"):
    sys.path.insert(0, str(p))
import rotation as RO  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

V, S, MS, L, H = RO.V, RO.S, RO.MS, RO.L, RO.H
ID, N, REF = "MVT_ID_mvt", 344841, 246.3837
DEPLOYED = {"c1": ROOT / ".cache/research_550/c1_more/c1plus_{seed}.pkl", "stand": ROOT / ".cache/research_550/k2_more/k2plus_{seed}.pkl"}
ROWS = {"c1": 84970, "stand": 17147}
BAR = {"c1": 0.1, "stand": 0.05}
CACHE = ROOT / ".cache/research_640"
OUT = ROOT / "research_640_tail_rule/results/catboost_layers.json"


def day_bootstrap(g, keys, draws=5000, seed=0):
    u, inv = np.unique(keys, return_inverse=True)
    sums, cnt = np.bincount(inv, weights=g), np.bincount(inv)
    rng = np.random.default_rng(seed)
    means = np.array([(lambda p: sums[p].sum() / cnt[p].sum())(rng.integers(0, len(u), len(u))) for _ in range(draws)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], int(len(u))


def fit_cat(fam, train):
    paths = [CACHE / f"cat_{fam}_EB1B2_{s}.pkl" for s in L.SEEDS]
    if not all(p.exists() for p in paths):
        CACHE.mkdir(parents=True, exist_ok=True)
        x, y = RO.inputs(fam, train), (train.y - train.comparator).to_numpy(float)
        with V.heavy_work(f"research_640: CatBoost {fam}", poll_s=10.0):
            for s, p in zip(L.SEEDS, paths):
                joblib.dump(CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=10, loss_function="RMSE",
                                              random_seed=s, thread_count=4, verbose=False).fit(x, y), p)
    return [joblib.load(p) for p in paths]


def main():
    b = RO.blocks()
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    fresh_days = tuple(sorted(set(b["F"].date)))
    report = {"prereg": "research_640_tail_rule/PREREG_catboost_layers.md"}
    for fam, spec in RO.FAMILIES.items():
        pick = lambda frame: frame[frame.category.eq(spec["category"])].drop_duplicates(ID)
        train = pd.concat([pick(b["E"]), pick(b["B1"]), pick(b["B2"])], ignore_index=True).drop_duplicates(ID)
        cat = fit_cat(fam, train)
        dep = [joblib.load(str(DEPLOYED[fam]).format(seed=s)) for s in L.SEEDS]
        t = RO.scored(fam, b["F"], fresh_days)
        if not np.array_equal(base.loc[t[ID]].to_numpy(float), t.comparator.to_numpy(float)):
            raise SystemExit("comparator not bit-identical")
        x, y, c = RO.inputs(fam, t), t.y.to_numpy(float), t.comparator.to_numpy(float)
        pd_ = {s: m.predict(x) for s, m in zip(L.SEEDS, dep)}
        pc = {s: m.predict(x) for s, m in zip(L.SEEDS, cat)}
        se = lambda k: (np.maximum(c + k, 300.0) - y) ** 2
        md, mc = np.mean(list(pd_.values()), axis=0), np.mean(list(pc.values()), axis=0)
        g = se(md) - se((md + mc) / 2)
        ci, days = day_bootstrap(g, (t.airport + "|" + t.date).to_numpy())
        by_cell = {cl: float(g[t.cell.eq(cl).to_numpy()].mean()) for cl in spec["cells"] if t.cell.eq(cl).any()}
        r = {"train_rows": int(len(train)), "test_rows": int(len(t)), "airport_days": days, "gain_mse": float(g.mean()), "bootstrap_95": ci,
             "by_month": {m: float(g[t.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")}, "by_cell": by_cell,
             "seeds": {s: float((se(pd_[s]) - se((pd_[s] + pc[s]) / 2)).mean()) for s in L.SEEDS}}
        mse = ROWS[fam] * r["gain_mse"] / N
        r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
        r["rules"] = {"pooled": r["gain_mse"] > 0, "bootstrap_lower_above_zero": ci[0] > 0, "both_months": min(r["by_month"].values()) > 0,
                      "two_thirds_cells": sum(v > 0 for v in by_cell.values()) >= 2 * len(by_cell) / 3,
                      "every_seed": all(v > 0 for v in r["seeds"].values()), "planning": r["planning_s"] >= BAR[fam]}
        r["verdict"] = "pass" if all(r["rules"].values()) else "fail"
        report[fam] = r
        print(fam, json.dumps({k: r[k] for k in ("gain_mse", "bootstrap_95", "by_month", "planning_s", "verdict")}), flush=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()

"""Test of research_630_c1_family/PREREG_c1_catboost_rotation.md. Fits and saves the rotation models. No upload.
Run from the repository root:  python3 -u -B research_630_c1_family/c1_catboost_rotation.py
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
ID, N, REF, ROWS = "MVT_ID_mvt", 344841, 246.3837, 84970
CACHE = ROOT / ".cache/research_630"
OUT = ROOT / "research_630_c1_family/results/c1_catboost_rotation.json"


def fit_cat(name, train):
    paths = [CACHE / f"cat_{name}_{s}.pkl" for s in L.SEEDS]
    if not all(p.exists() for p in paths):
        CACHE.mkdir(parents=True, exist_ok=True)
        x, y = MS.inputs(train, "real"), (train.y - train.comparator).to_numpy(float)
        with V.heavy_work(f"research_630: CatBoost {name}", poll_s=10.0):
            for s, p in zip(L.SEEDS, paths):
                m = CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=10, loss_function="RMSE",
                                      random_seed=s, thread_count=4, verbose=False).fit(x, y)
                joblib.dump(m, p)
    return [joblib.load(p) for p in paths]


def main():
    b = RO.blocks()
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    pick = lambda frame: frame[frame.category.eq("partial_ground")].drop_duplicates(ID)
    report = {"prereg": "research_630_c1_family/PREREG_c1_catboost_rotation.md"}
    for rot, held, other, held_days in (("R1", "B2", "B1", V.B2), ("R2", "B1", "B2", V.B1)):
        train = pd.concat([pick(b["E"]), pick(b[other]), pick(b["F"])], ignore_index=True).drop_duplicates(ID)
        lgbm = RO.fit("c1", f"{rot}_with_F", train)            # identical recipe and rows as research_610 (reused if saved)
        cat = fit_cat(f"{rot}_with_F", train)
        t = RO.scored("c1", b[held], tuple(held_days))
        if not np.array_equal(base.loc[t[ID]].to_numpy(float), t.comparator.to_numpy(float)):
            raise SystemExit("comparator not bit-identical")
        x, y, c = MS.inputs(t, "real"), t.y.to_numpy(float), t.comparator.to_numpy(float)
        pl = {s: m.predict(x) for s, m in zip(L.SEEDS, lgbm)}
        pc = {s: m.predict(x) for s, m in zip(L.SEEDS, cat)}
        se = lambda corr: (np.maximum(c + corr, 300.0) - y) ** 2
        ml, mc = np.mean(list(pl.values()), axis=0), np.mean(list(pc.values()), axis=0)
        g = se(ml) - se((ml + mc) / 2)
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-100:]] = False
        by_cell = {cl: float(g[t.cell.eq(cl).to_numpy()].mean()) for cl in RO.C1_CELLS if t.cell.eq(cl).any()}
        r = {"train_rows": int(len(train)), "test_rows": int(len(t)), "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
             "by_month": {m: float(g[t.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")}, "by_cell": by_cell,
             "seeds": {s: float((se(pl[s]) - se((pl[s] + pc[s]) / 2)).mean()) for s in L.SEEDS},
             "rmse": {"lgbm": float(np.sqrt(se(ml).mean())), "cat": float(np.sqrt(se(mc).mean())), "blend": float(np.sqrt(se((ml + mc) / 2).mean()))}}
        r["rules"] = {"pooled": r["gain_mse"] > 0, "without_best100": r["without_best100"] > 0, "both_months": min(r["by_month"].values()) > 0,
                      "two_thirds_cells": sum(v > 0 for v in by_cell.values()) >= 2 * len(by_cell) / 3,
                      "every_seed": all(v > 0 for v in r["seeds"].values())}
        r["pass"] = all(r["rules"].values())
        report[rot] = r
        print(rot, json.dumps({k: r[k] for k in ("gain_mse", "without_best100", "by_month", "rmse", "pass")}), flush=True)
    mse = (report["R1"]["gain_mse"] + report["R2"]["gain_mse"]) / 2 * ROWS / N
    report["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    report["verdict"] = "pass" if report["R1"]["pass"] and report["R2"]["pass"] and report["planning_s"] >= 0.1 else "fail"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"verdict": report["verdict"], "planning_s": report["planning_s"]}))


if __name__ == "__main__":
    main()

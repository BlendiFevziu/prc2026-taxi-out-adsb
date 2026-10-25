"""Test of research_610_more_data/PREREG_rotation.md. Fits the rotation models (saved). No upload.
Run from the repository root:  python3 -u -B research_610_more_data/rotation.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_550_review"):
    sys.path.insert(0, str(p))
import c1_more_data as C  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

V = C.V
S, MS = V.S, V.MS
ID, N, REF = "MVT_ID_mvt", 344841, 246.3837
C1_CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "EHAM/07", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07")
ST_CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07", "EHAM/01", "EHAM/07")
FAMILIES = {"c1": {"category": "partial_ground", "cells": C1_CELLS, "trees": 800, "leaves": 31, "rows": 106496},
            "stand": {"category": "observed_at_assigned_stand", "cells": ST_CELLS, "trees": 400, "leaves": 15, "rows": 32377}}
CACHE = ROOT / ".cache/research_610/models"
OUT = ROOT / "research_610_more_data/results/rotation.json"


def blocks():
    return {"E": pd.concat([S.extract(tuple(V.S_DATES), "train"), S.extract(V.E_EXTRA, "volume_extra")], ignore_index=True),
            "B1": S.extract(tuple(V.B1), "development"), "B2": S.extract(tuple(V.B2), "confirm"),
            "F": pd.concat([S.extract(d, t) for t, d in C.FRESH], ignore_index=True)}


def inputs(fam, frame):
    return MS.inputs(frame, "real") if fam == "c1" else S.inputs(frame, "real")


def fit(fam, name, train):
    spec = FAMILIES[fam]
    paths = [CACHE / f"{fam}_{name}_{s}.pkl" for s in L.SEEDS]
    if not all(p.exists() for p in paths):
        CACHE.mkdir(parents=True, exist_ok=True)
        x, y = inputs(fam, train), (train.y - train.comparator).to_numpy(float)
        with V.heavy_work(f"research_610: fit {fam} {name}", poll_s=10.0):
            for s, p in zip(L.SEEDS, paths):
                m = S.R.lgb.LGBMRegressor(objective="regression", n_estimators=spec["trees"], learning_rate=.03, num_leaves=spec["leaves"],
                                          min_child_samples=80, reg_lambda=100., subsample=.8, subsample_freq=1, colsample_bytree=.9,
                                          max_bin=63, n_jobs=4, random_state=s, verbosity=-1).fit(x, y)
                joblib.dump(m, p)
    return [joblib.load(p) for p in paths]


def scored(fam, block, tag_days):
    spec = FAMILIES[fam]
    with V.heavy_work("research_610: detector replay", poll_s=10.0):
        det = set(V.detector_ids(tag_days))
    f = block.copy()
    f["cell"] = f.airport + "/" + f.date.str[5:7]
    return f[f.category.eq(spec["category"]) & f.cell.isin(spec["cells"]) & ~f[ID].isin(det)].drop_duplicates(ID).reset_index(drop=True)


def main():
    b = blocks()
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    report = {"prereg": "research_610_more_data/PREREG_rotation.md"}
    for fam, spec in FAMILIES.items():
        pick = lambda frame: frame[frame.category.eq(spec["category"])].drop_duplicates(ID)
        fr = {}
        for rot, held, other, held_days in (("R1", "B2", "B1", V.B2), ("R2", "B1", "B2", V.B1)):
            small = pd.concat([pick(b["E"]), pick(b[other])], ignore_index=True).drop_duplicates(ID)
            large = pd.concat([small, pick(b["F"])], ignore_index=True).drop_duplicates(ID)
            m_small, m_large = fit(fam, f"{rot}_without_F", small), fit(fam, f"{rot}_with_F", large)
            t = scored(fam, b[held], tuple(held_days))
            if not np.array_equal(base.loc[t[ID]].to_numpy(float), t.comparator.to_numpy(float)):
                raise SystemExit("comparator not bit-identical")
            x, y, c = inputs(fam, t), t.y.to_numpy(float), t.comparator.to_numpy(float)
            ps = {s: m.predict(x) for s, m in zip(L.SEEDS, m_small)}
            pl = {s: m.predict(x) for s, m in zip(L.SEEDS, m_large)}
            se = lambda corr: (np.maximum(c + corr, 300.0) - y) ** 2
            g = se(np.mean(list(ps.values()), axis=0)) - se(np.mean(list(pl.values()), axis=0))
            keep = np.ones(len(g), bool)
            keep[np.argsort(g, kind="stable")[-100:]] = False
            by_cell = {cl: float(g[t.cell.eq(cl).to_numpy()].mean()) for cl in spec["cells"] if t.cell.eq(cl).any()}
            r = {"train_rows_without_F": int(len(small)), "train_rows_with_F": int(len(large)), "test_rows": int(len(t)),
                 "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
                 "by_month": {m: float(g[t.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")}, "by_cell": by_cell,
                 "seeds": {s: float((se(ps[s]) - se(pl[s])).mean()) for s in L.SEEDS}}
            r["rules"] = {"pooled": r["gain_mse"] > 0, "without_best100": r["without_best100"] > 0, "both_months": min(r["by_month"].values()) > 0,
                          "two_thirds_cells": sum(v > 0 for v in by_cell.values()) >= 2 * len(by_cell) / 3,
                          "every_seed": all(v > 0 for v in r["seeds"].values())}
            r["pass"] = all(r["rules"].values())
            fr[rot] = r
            print(fam, rot, json.dumps({k: r[k] for k in ("gain_mse", "without_best100", "by_month", "pass")}), flush=True)
        mean_gain = (fr["R1"]["gain_mse"] + fr["R2"]["gain_mse"]) / 2
        mse = 0.5 * mean_gain * spec["rows"] / N
        fr["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
        fr["verdict"] = "pass" if fr["R1"]["pass"] and fr["R2"]["pass"] and fr["planning_s"] >= 0.1 else "fail"
        report[fam] = fr
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: (v["verdict"], v["planning_s"]) for k, v in report.items() if k != "prereg"}))


if __name__ == "__main__":
    main()

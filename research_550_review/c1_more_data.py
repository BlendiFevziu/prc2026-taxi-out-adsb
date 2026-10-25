"""Test of research_550_review/PREREG_c1_more_data.md. Fits C1+ once (saved), then scores. No upload.
Run from the repository root:  python3 -u -B research_550_review/c1_more_data.py
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
from research_320 import volume_capacity as V  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

S, MS = V.S, V.MS
ID, N, REF, ROWS_2026 = "MVT_ID_mvt", 344841, 253.36, 84970
CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "EHAM/07", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07")
FRESH = (("fresh_july", ("2025-07-01", "2025-07-05", "2025-07-19", "2025-07-20", "2025-07-24", "2025-07-28", "2025-07-30")),
         ("eddf_july_confirm", ("2025-07-06", "2025-07-08", "2025-07-10")),
         ("fresh_january", ("2025-01-03", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-15", "2025-01-17",
                            "2025-01-22", "2025-01-27", "2025-01-29", "2025-01-31")))
MODELS = HERE / ".cache/research_550/c1_more/c1plus_{seed}.pkl"
OUT = HERE / "research_550_review/results/c1_more_data.json"


def training():
    parts = [S.extract(tuple(V.S_DATES), "train"), S.extract(V.E_EXTRA, "volume_extra"),
             S.extract(tuple(V.B1), "development"), S.extract(tuple(V.B2), "confirm")]
    t = pd.concat([p[p.category.eq("partial_ground")] for p in parts], ignore_index=True).drop_duplicates(ID)
    return t.reset_index(drop=True)


def models(train):
    paths = [Path(str(MODELS).format(seed=s)) for s in L.SEEDS]
    if not all(p.exists() for p in paths):
        with V.heavy_work("research_550: fit C1+ on E, B1 and B2", poll_s=10.0):
            fitted = L.fit("partial", train)
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        for m, p in zip(fitted, paths):
            joblib.dump(m, p)
    return [joblib.load(p) for p in paths]


def test_rows():
    frames = []
    for tag, days in FRESH:
        f = S.extract(days, tag)
        with V.heavy_work(f"research_550: detector replay {tag}", poll_s=10.0):
            det = set(V.detector_ids(days))
        f["cell"] = f.airport + "/" + f.date.str[5:7]
        frames.append(f[f.date.isin(days) & f.category.eq("partial_ground") & f.cell.isin(CELLS) & ~f[ID].isin(det)])
    return pd.concat(frames, ignore_index=True).drop_duplicates(ID).reset_index(drop=True)


def main():
    train = training()
    fresh_days = {d for _, ds in FRESH for d in ds}
    if fresh_days & set(train.date):
        raise SystemExit("a test day is in the training set")
    new_models = models(train)
    f = test_rows()
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    if not np.array_equal(base.loc[f[ID]].to_numpy(float), f.comparator.to_numpy(float)):
        raise SystemExit("comparator not bit-identical")
    y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
    x = MS.inputs(f, "real")
    old = {s: m.predict(x) for s, m in zip(L.SEEDS, L.load_models("partial"))}
    new = {s: m.predict(x) for s, m in zip(L.SEEDS, new_models)}
    se = lambda c: (np.maximum(b + c, 300.0) - y) ** 2
    g = se(np.mean(list(old.values()), axis=0)) - se(np.mean(list(new.values()), axis=0))
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-100:]] = False
    by_day = {d: float(g[f.date.eq(d).to_numpy()].mean()) for d in sorted(f.date.unique())}
    by_cell = {c: float(g[f.cell.eq(c).to_numpy()].mean()) for c in CELLS if f.cell.eq(c).any()}
    r = {"prereg": "research_550_review/PREREG_c1_more_data.md", "training_rows": int(len(train)), "test_rows": int(len(f)),
         "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
         "by_month": {m: float(g[f.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")},
         "by_cell": by_cell, "cell_rows": {c: int(f.cell.eq(c).sum()) for c in CELLS}, "by_day": by_day,
         "seeds": {s: float((se(old[s]) - se(new[s])).mean()) for s in L.SEEDS}}
    mse = ROWS_2026 * r["gain_mse"] / N
    r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    r["rules"] = {"pooled_positive": r["gain_mse"] > 0, "without_best100_positive": r["without_best100"] > 0,
                  "both_months": min(r["by_month"].values()) > 0, "six_of_eight_cells": sum(v > 0 for v in by_cell.values()) >= 6,
                  "seventy_percent_of_days": np.mean([v > 0 for v in by_day.values()]) >= 0.7,
                  "every_seed": all(v > 0 for v in r["seeds"].values()), "planning_at_least_0_1_s": r["planning_s"] >= 0.1}
    r["rules"] = {k: bool(v) for k, v in r["rules"].items()}
    r["verdict"] = "pass" if all(r["rules"].values()) else "fail"
    OUT.write_text(json.dumps(r, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "by_day"}, indent=1, default=str))


if __name__ == "__main__":
    main()

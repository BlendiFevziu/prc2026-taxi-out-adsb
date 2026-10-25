"""The second partial-track model of research_490_second/PREREG_second.md: fit on C1's training set E, then the arm
max(base + (C1 + new) / 2, 300) against max(base + C1, 300) on B1, B2 and the twenty fresh days. No upload.
Needs every feature file of extract_robust.py. Run from the repository root:  python3 -u -B research_490_second/second.py
"""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import volume_capacity as V  # noqa: E402
from research_320.episode_experiment import detector_ids  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

S = V.S
ID, N, REF = "MVT_ID_mvt", 344841, 255.89          # REF: v68's expected score, fixed before its upload
ROWS_2026 = 72977 + 11993                          # C1-owned rows of the eight cells (v62 owner "partial" + v63 EDDF July)
CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "EHAM/07", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07")
INPUTS = list(L.CONTEXT) + list(L.EHAM_TRACK) + list(L.AIRPORT_FLAGS) + ["is_july"]
E_TAGS = (("train", tuple(V.S_DATES)), ("volume_extra", V.E_EXTRA))
FRESH = (("fresh_july", ("2025-07-01", "2025-07-05", "2025-07-19", "2025-07-20", "2025-07-24", "2025-07-28", "2025-07-30")),
         ("eddf_july_confirm", ("2025-07-06", "2025-07-08", "2025-07-10")),
         ("fresh_january", ("2025-01-03", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-15", "2025-01-17",
                            "2025-01-22", "2025-01-27", "2025-01-29", "2025-01-31")))
FEATURES = HERE / ".cache/research_490/features"
MODELS = HERE / ".cache/research_490/models"
OUT = HERE / "research_490_second/results/second.json"


def with_features(frame):
    """Attach the seven sampling-aware features; every row must have them."""
    days = sorted(set(zip(frame.airport, frame.date)))
    feats = pd.concat([pd.read_parquet(FEATURES / f"{a}_{d}.parquet", columns=[ID] + list(L.EHAM_TRACK)) for a, d in days],
                      ignore_index=True)
    out = frame.merge(feats, on=ID, how="left", validate="one_to_one")
    missing = ~frame[ID].isin(feats[ID])
    if missing.any() or len(out) != len(frame):
        raise SystemExit(f"{int(missing.sum())} rows without sampling-aware features")
    return out


def rows(tag, dates):
    ext = S.extract(tuple(dates), tag)
    with V.heavy_work(f"research_490: detector replay {tag}", poll_s=10.0):
        det = set(detector_ids(dates))
    f = ext.copy()
    f["cell"] = f.airport + "/" + f.date.str[5:7]
    f = f[f.date.isin(dates) & f.cell.isin(CELLS) & f.category.eq("partial_ground") & ~f[ID].isin(det)].reset_index(drop=True)
    return with_features(f)


def fit(train):
    paths = [MODELS / f"second_{s}.pkl" for s in L.SEEDS]
    if all(p.exists() for p in paths):
        return [joblib.load(p) for p in paths]
    import lightgbm as lgb
    MODELS.mkdir(parents=True, exist_ok=True)
    x, y = train[INPUTS], (train.y - train.comparator).to_numpy(float)
    models = []
    with V.heavy_work("research_490: fit the second model", poll_s=10.0):
        for s, p in zip(L.SEEDS, paths):
            m = lgb.LGBMRegressor(objective="regression", n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=80,
                                  reg_lambda=100., subsample=.8, subsample_freq=1, colsample_bytree=.9, max_bin=63, n_jobs=4,
                                  random_state=s, verbosity=-1).fit(x, y)
            joblib.dump(m, p)
            models.append(m)
    return models


def second(frame, models):
    x = frame[INPUTS]
    preds = [m.predict(x) for m in models]
    for m, p in zip(models, preds):
        if not np.array_equal(m.predict(x.iloc[::-1])[::-1], p):
            raise SystemExit("reverse-order replay failed")
    return np.mean(preds, axis=0)


def gain(frame, models, base):
    if not np.array_equal(base.loc[frame[ID]].to_numpy(float), frame.comparator.to_numpy(float)):
        raise SystemExit("comparator not bit-identical")
    y, b = frame.y.to_numpy(float), frame.comparator.to_numpy(float)
    c1, new = L.correction("partial", frame), second(frame, models)
    return (np.maximum(b + c1, 300.0) - y) ** 2 - (np.maximum(b + (c1 + new) / 2, 300.0) - y) ** 2


def stats(g, frame):
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-100:]] = False
    return {"rows": int(len(g)), "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
            "by_month": {m: float(g[frame.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")},
            "by_cell": {c: float(g[frame.cell.eq(c).to_numpy()].mean()) for c in CELLS if frame.cell.eq(c).any()},
            "cell_rows": {c: int(frame.cell.eq(c).sum()) for c in CELLS},
            "by_day": {d: float(g[frame.date.eq(d).to_numpy()].mean()) for d in sorted(frame.date.unique())}}


def main():
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    train = pd.concat([rows(t, d) for t, d in E_TAGS], ignore_index=True)
    fresh_days = {d for _, ds in FRESH for d in ds}
    if fresh_days & set(train.date) or fresh_days & (set(V.B1) | set(V.B2)):
        raise SystemExit("a confirmation day overlaps training or development")
    models = fit(train)
    report = {"prereg": "research_490_second/PREREG_second.md", "inputs": INPUTS,
              "training": {"rows": int(len(train)), "by_cell": {c: int(n) for c, n in train.groupby("cell").size().items()}}}
    for name, tag, dates in (("B1", "development", V.B1), ("B2", "confirm", V.B2)):
        f = rows(tag, dates)
        report[name] = stats(gain(f, models, base), f)
    conf = pd.concat([rows(t, d) for t, d in FRESH], ignore_index=True)
    report["confirmation"] = c = stats(gain(conf, models, base), conf)
    mse = ROWS_2026 * c["gain_mse"] / N
    report["planning_full_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    report["rules"] = {"B1_positive": report["B1"]["gain_mse"] > 0, "B1_both_months": min(report["B1"]["by_month"].values()) > 0,
                       "B2_positive": report["B2"]["gain_mse"] > 0, "confirmation_positive": c["gain_mse"] > 0,
                       "confirmation_without_best100": c["without_best100"] > 0,
                       "confirmation_both_months": min(c["by_month"].values()) > 0,
                       "six_of_eight_cells": sum(v > 0 for v in c["by_cell"].values()) >= 6,
                       "seventy_percent_of_days": np.mean([v > 0 for v in c["by_day"].values()]) >= 0.7,
                       "planning_at_least_0_2_s": report["planning_full_s"] >= 0.2}
    report["rules"] = {k: bool(v) for k, v in report["rules"].items()}
    report["verdict"] = "pass" if all(report["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "inputs"}, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()

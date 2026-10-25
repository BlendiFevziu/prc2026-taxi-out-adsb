"""The frozen stand model at Zurich January and Amsterdam January on ten unread January 2025 days
(research_450_stand_january/PREREG_stand_january.md). No fit, no upload.
Run from the repository root:  python3 -u -B research_450_stand_january/stand_january.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import volume_capacity as V  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

S = V.S
ID, N, REF, WEIGHT = "MVT_ID_mvt", 344841, 256.4921, 0.5
DAYS = ("2025-01-03", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-15", "2025-01-17", "2025-01-22", "2025-01-27", "2025-01-29", "2025-01-31")
HALVES = {"first_half": DAYS[:5], "second_half": DAYS[5:]}
CELLS = {"LSZH/01": 2869, "EHAM/01": 1137}
OUT = HERE / "research_450_stand_january/results/stand_january.json"


def main():
    missing = [d for d in DAYS if not (S.MS.R.MD.ADSB / f"tracks/{d}.parquet").exists()]
    if missing:
        raise SystemExit(f"tracks missing: {missing}")
    with V.heavy_work("research_450: extraction and detector replay of ten fresh January days", poll_s=10.0):
        ext = S.extract(DAYS, "fresh_january")
        det = set(V.detector_ids(DAYS))
    ext["cell"] = ext.airport + "/" + ext.date.str[5:7]
    rows = ext[ext.category.eq("observed_at_assigned_stand") & ext.cell.isin(CELLS) & ~ext[ID].isin(det)].reset_index(drop=True)
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    checks = {"comparator_bit_identical": bool(np.array_equal(base.loc[rows[ID]].to_numpy(float), rows.comparator.to_numpy(float))),
              "rows": rows.cell.value_counts().to_dict()}
    if not checks["comparator_bit_identical"]:
        raise SystemExit("comparator not bit-identical")
    corr = L.correction("stand", rows)
    rev = L.correction("stand", rows.iloc[::-1].reset_index(drop=True))[::-1]
    checks["reverse_replay"] = bool(np.array_equal(corr, rev))
    y, b = rows.y.to_numpy(float), rows.comparator.to_numpy(float)
    gain = (b - y) ** 2 - (L.apply(rows, corr, WEIGHT) - y) ** 2
    report = {"prereg": "research_450_stand_january/PREREG_stand_january.md", "checks": checks, "cells": {}}
    passing_mse = 0.0
    for cell, rows26 in CELLS.items():
        m = rows.cell.eq(cell).to_numpy()
        g, f = gain[m], rows[m]
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-100:]] = False
        days = {d: float(g[f.date.eq(d).to_numpy()].mean()) for d in DAYS if f.date.eq(d).any()}
        halves = {k: float(g[f.date.isin(v).to_numpy()].mean()) for k, v in HALVES.items()}
        mse = rows26 * float(g.mean()) / N
        r = {"rows": int(m.sum()), "base_rmse": float(np.sqrt(np.mean((b[m] - y[m]) ** 2))),
             "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()), "halves": halves, "by_day": days,
             "planning_mse_per_scored_row": mse,
             "rules": {"1_pooled_positive": float(g.mean()) > 0, "2_best100_removed_positive": float(g[keep].mean()) > 0,
                       "3_each_half_positive": all(v > 0 for v in halves.values()),
                       "4_seven_of_ten_days_positive": sum(v > 0 for v in days.values()) >= 7}}
        r["passes_1_to_4"] = all(r["rules"].values())
        if r["passes_1_to_4"]:
            passing_mse += mse
        report["cells"][cell] = r
        print(cell, json.dumps({k: r[k] for k in ("rows", "gain_mse", "without_best100", "halves", "rules")}), flush=True)
    joint_s = float(REF - np.sqrt(REF ** 2 - passing_mse)) if passing_mse > 0 else 0.0
    report["joint_planning_s"] = joint_s
    report["passing_cells"] = [c for c, r in report["cells"].items() if r["passes_1_to_4"] and r["planning_mse_per_scored_row"] > 0] if joint_s >= 0.05 else []
    report["verdict"] = "pass" if report["passing_cells"] and checks["reverse_replay"] else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("joint planning", round(joint_s, 3), "s; passing", report["passing_cells"], report["verdict"], flush=True)


if __name__ == "__main__":
    main()

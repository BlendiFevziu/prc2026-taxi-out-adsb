"""Retest of research_550_review/PREREG_stand_capacity_retest.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_550_review/stand_capacity_retest.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import stand_experiment as S  # noqa: E402
from research_320 import volume_capacity as V  # noqa: E402
from research_320 import stand_capacity as K  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, N, REF, ROWS_2026 = "MVT_ID_mvt", 344841, 253.57, 17147
CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07", "EHAM/01", "EHAM/07")
BLOCKS = (("fresh_july", ("2025-07-01", "2025-07-05", "2025-07-19", "2025-07-20", "2025-07-24", "2025-07-28", "2025-07-30")),
          ("eddf_july_confirm", ("2025-07-06", "2025-07-08", "2025-07-10")),
          ("fresh_january", ("2025-01-03", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-15", "2025-01-17",
                             "2025-01-22", "2025-01-27", "2025-01-29", "2025-01-31")))
OUT = HERE / "research_550_review/results/stand_capacity_retest.json"


def rows():
    frames = []
    for tag, days in BLOCKS:
        f = S.extract(days, tag)
        with V.heavy_work(f"research_550: detector replay {tag}", poll_s=10.0):
            det = set(V.detector_ids(days))
        f["cell"] = f.airport + "/" + f.date.str[5:7]
        frames.append(f[f.date.isin(days) & f.category.eq("observed_at_assigned_stand") & f.cell.isin(CELLS) & ~f[ID].isin(det)])
    return pd.concat(frames, ignore_index=True).drop_duplicates(ID).reset_index(drop=True)


def main():
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    f = rows()
    if not np.array_equal(base.loc[f[ID]].to_numpy(float), f.comparator.to_numpy(float)):
        raise SystemExit("comparator not bit-identical")
    y = f.y.to_numpy(float)
    se = lambda p: (p - y) ** 2
    ref_c = K.corrections(f, "ref")
    ref_mean = np.mean([ref_c[s] for s in K.SEEDS], axis=0)
    ref_se = se(K.pred(f, ref_mean, 1.0))
    report = {"prereg": "research_550_review/PREREG_stand_capacity_retest.md", "rows": int(len(f)),
              "cell_rows": {c: int(f.cell.eq(c).sum()) for c in CELLS}}
    for arm in ("k1", "k2", "k3"):
        c = K.corrections(f, arm)
        g = ref_se - se(K.pred(f, np.mean([c[s] for s in K.SEEDS], axis=0), 1.0))
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-100:]] = False
        by_day = {d: float(g[f.date.eq(d).to_numpy()].mean()) for d in sorted(f.date.unique())}
        by_cell = {cl: float(g[f.cell.eq(cl).to_numpy()].mean()) for cl in CELLS if f.cell.eq(cl).any()}
        r = {"gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
             "by_month": {m: float(g[f.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")},
             "by_cell": by_cell, "by_day": by_day,
             "seeds": {s: float((se(K.pred(f, ref_c[s], 1.0)) - se(K.pred(f, c[s], 1.0))).mean()) for s in K.SEEDS}}
        mse = ROWS_2026 * r["gain_mse"] / N
        r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
        r["rules"] = {"pooled_positive": r["gain_mse"] > 0, "without_best100_positive": r["without_best100"] > 0,
                      "both_months": min(r["by_month"].values()) > 0,
                      "seventy_percent_of_days": np.mean([v > 0 for v in by_day.values()]) >= 0.7,
                      "two_thirds_of_cells": sum(v > 0 for v in by_cell.values()) >= 2 * len(by_cell) / 3,
                      "every_seed": all(v > 0 for v in r["seeds"].values()),
                      "planning_at_least_0_1_s": r["planning_s"] >= 0.1}
        r["rules"] = {k: bool(v) for k, v in r["rules"].items()}
        r["verdict"] = "pass" if all(r["rules"].values()) else "fail"
        report[arm] = r
    report["decision"] = report["k2"]["verdict"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: (v if k not in ("k1", "k2", "k3") else {kk: vv for kk, vv in v.items() if kk != "by_day"}) for k, v in report.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()

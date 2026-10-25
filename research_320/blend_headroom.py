"""EXPLORATORY, development dates only, no fitting: is the fixed shrink 0.5 of the ADS-B residual models too cautious,
and does the best weight depend on the evidence (idea 1, coverage-conditioned blending)?

v56 partial-track model: its eight development-test dates, the seven deployed cells, partial_ground rows, without the
Amsterdam-detector rows (research_320/audit_v56_errors.py cohort). v57 stand model: its fourteen development dates,
five kept cells. Corrections from the saved models; prediction max(comparator + s x correction, 300) for a grid of s,
overall and within evidence strata. These dates are spent; any shrink rule must be confirmed on unread data.
Run from the repository root:  python3 -u -B research_320/blend_headroom.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import stand_experiment as S  # noqa: E402
from research_320 import audit_v56_errors as A  # noqa: E402

GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
OUT = HERE / "research_320/results/blend_headroom.json"


def curve(y, base, corr):
    return {str(s): float(np.mean((base - y) ** 2 - (np.maximum(base + s * corr, 300.0) - y) ** 2)) for s in GRID}


def strata(frame, y, base, corr):
    out = {}
    for name, col in (("ground_points", "ground_points"), ("still_fraction", "still_fraction"),
                      ("first_ground_age_s", "first_ground_age_s")):
        q = frame[col].rank(pct=True).to_numpy()
        for lo, hi, label in ((0, 1 / 3, "low"), (1 / 3, 2 / 3, "mid"), (2 / 3, 1.01, "high")):
            k = (q > lo) & (q <= hi)
            c = curve(y[k], base[k], corr[k])
            out[f"{name}:{label}"] = {"rows": int(k.sum()), "best_s": max(c, key=c.get), "gain_at_0.5": c["0.5"], "gain_at_best": max(c.values())}
    return out


def main():
    report = {}
    # v56 partial-track model
    pred = pd.read_parquet(A.REBUILD / "predictions.parquet")
    pred["month"] = pred.date.str[5:7]
    cell = np.array([(a, m) in A.DEPLOYED for a, m in zip(pred.airport, pred.month)])
    det = A.detector_rows()
    p = pred[cell & pred.category.eq("partial_ground").to_numpy() & ~pred.MVT_ID_mvt.isin(det).to_numpy()].reset_index(drop=True)
    import joblib
    x = p[list(S.MS.COLS)]
    corr = np.mean([joblib.load(A.REBUILD / f"real_{s}.pkl").predict(x) for s in S.SEEDS], axis=0)
    y, base = p.y.to_numpy(float), p.comparator.to_numpy(float)
    assert np.allclose(np.maximum(base + .5 * corr, 300.0), p.real.to_numpy(float))
    report["v56_partial"] = {"rows": int(len(p)), "curve": curve(y, base, corr), "strata": strata(p, y, base, corr),
                             "by_cell_best_s": {c: max(curve(y[k], base[k], corr[k]).items(), key=lambda kv: kv[1])[0]
                                                for c, k in ((f"{a}/{m}", (p.airport.eq(a) & p.month.eq(m)).to_numpy()) for a, m in sorted(A.DEPLOYED))}}
    # v57 stand model
    d = S.extract(S.STAGES["development"], "development")
    d = d[(d.airport + "/" + d.date.str[5:7]).isin(["EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07"]) & d.category.eq("observed_at_assigned_stand")].reset_index(drop=True)
    xs = S.inputs(d, "real")
    cs = np.mean([joblib.load(S.CACHE / "models" / f"real_{s}.pkl").predict(xs) for s in S.SEEDS], axis=0)
    ys, bs = d.y.to_numpy(float), d.comparator.to_numpy(float)
    report["v57_stand"] = {"rows": int(len(d)), "curve": curve(ys, bs, cs), "strata": strata(d, ys, bs, cs)}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for k in ("v56_partial", "v57_stand"):
        c = report[k]["curve"]
        best = max(c, key=c.get)
        print(f"{k}: rows {report[k]['rows']}, gain at s=0.5 {c['0.5']:.0f}, best s {best} gain {c[best]:.0f} | curve " +
              " ".join(f"{s}:{v:.0f}" for s, v in c.items()))
        for name, r in report[k]["strata"].items():
            print(f"   {name:24s} rows {r['rows']:5d} best s {r['best_s']:>4s}  gain@0.5 {r['gain_at_0.5']:7.0f}  gain@best {r['gain_at_best']:7.0f}")
    print("v56 best s by cell", report["v56_partial"]["by_cell_best_s"])


if __name__ == "__main__":
    main()

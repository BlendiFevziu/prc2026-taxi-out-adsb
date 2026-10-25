"""Test of research_590_stand_new/PREREG_drift_robust_munich.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_590_stand_new/drift_robust_munich.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_590_stand_new", ROOT / "research_550_review"):
    sys.path.insert(0, str(p))
import drift_robust as DR  # noqa: E402

T, S, H, CC = DR.T, DR.S, DR.H, DR.CC
ID, N, REF = "MVT_ID_mvt", 344841, 246.3837
OUT = ROOT / "research_590_stand_new/results/drift_robust_munich.json"


def main():
    cols = DR.cols()
    ms = DR.models(None, cols)
    comps = H.components_2025()
    comps["comparator"] = DR.L.compose_base(comps)
    spec = CC.CELLS["EDDM/01"]
    f = T.frame_2025("EDDM", spec["days"], comps)
    f = f[~f[ID].isin(T.munich_detector_rows(spec["days"]))].reset_index(drop=True)
    r = DR.judge(f, ms, cols, spec["blocks"])
    own = H.ownership_v62().set_index(ID).owner
    r26 = T.rows_2026("EDDM")
    months, mse = {}, 0.0
    for m in ("01", "07"):
        t = r26[r26.month.eq(m) & own.reindex(r26[ID]).eq("base").to_numpy()]
        fl = DR.flags(f, t)
        ok = r["pass"] and not fl
        months[f"EDDM/{m}"] = {"rows_2026": int(len(t)), "drift_flags_kept": fl, "admitted": ok}
        if ok:
            mse += r["gain_mse"] * len(t) / N
    report = {"prereg": "research_590_stand_new/PREREG_drift_robust_munich.md", "test": r, "months": months,
              "planning_s": float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else 0.0}
    report["verdict"] = "pass" if report["planning_s"] >= 0.03 else "fail"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

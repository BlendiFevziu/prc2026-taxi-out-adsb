"""Test of research_590_stand_new/PREREG_drift_robust_c1_rome.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_590_stand_new/drift_robust_c1_rome.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_590_stand_new",
          ROOT / "research_550_review", ROOT / "research_380_rome"):
    sys.path.insert(0, str(p))
import drift_robust_c1 as DC  # noqa: E402
import stage2_c1_transfer as ST2  # noqa: E402

H, L, MS = DC.H, DC.L, DC.MS
ID, N, REF, ROWS_2026 = "MVT_ID_mvt", 344841, 247.13, 6767
OUT = ROOT / "research_590_stand_new/results/drift_robust_c1_rome.json"


def main():
    cols = json.loads((ROOT / "research_590_stand_new/results/drift_robust_c1.json").read_text())["inputs"]
    ms = DC.models(None, cols)
    f = ST2.frame(ST2.FRESH)
    x = MS.inputs(f, "real")[cols]
    c = {s: m.predict(x) for s, m in zip(L.SEEDS, ms)}
    r = DC.judge(f, c, f.comparator.to_numpy(float), {"first_half": ST2.FRESH[:8], "second_half": ST2.FRESH[8:]})
    own = H.ownership_v62().set_index(ID).owner
    audit = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_320/airport_coverage/day_2026-01-*.parquet")))], ignore_index=True)
    t = audit[audit.airport.eq("LIRF") & audit.category.eq("partial_ground") & own.reindex(audit[ID]).eq("base").to_numpy()]
    fl = DC.flags(f, t)
    mse = ROWS_2026 * r["gain_mse"] / N
    report = {"prereg": "research_590_stand_new/PREREG_drift_robust_c1_rome.md", "test": r, "rows_2026": int(len(t)), "drift_flags_kept": fl,
              "planning_s": float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))}
    report["verdict"] = "pass" if r["pass"] and not fl and report["planning_s"] >= 0.1 else "fail"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

"""Test of research_670_fallback_lfpg/PREREG_fallback_lfpg.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_670_fallback_lfpg/fallback_lfpg.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_550_review", ROOT / "research_640_tail_rule"):
    sys.path.insert(0, str(p))
import fallback_detector as FD  # noqa: E402
import catboost_layers as CL  # noqa: E402

N, REF, AP, ROWS_2026 = 344841, 245.020, "LFPG", 734
OUT = ROOT / "research_670_fallback_lfpg/results/fallback_lfpg.json"
CACHE = ROOT / ".cache/research_670/lfpg_rows.parquet"


def main():
    FD.TOW[AP] = False
    if CACHE.exists():
        d = pd.read_parquet(CACHE)
    else:
        mapping = FD.AT.OB.callsign_mapping()
        maps = (FD.SD.stand_table(AP), {})
        rows = []
        for i, day in enumerate(FD.DAYS):
            rows += FD.day_rows(day, AP, mapping, maps)
            if i % 10 == 0:
                print(day, len(rows), flush=True)
        d = pd.DataFrame(rows)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(CACHE, index=False)
    pred = pd.concat([pd.read_parquet(ROOT / f".cache/research_240/fallback_anchor_production/predictions_{p}.parquet",
                                      columns=["MVT_ID_mvt", "y", "prod_anchor"]) for p in FD.PAIRS]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    d["y"], d["ref"] = d.MVT_ID_mvt.map(pred.y), d.MVT_ID_mvt.map(pred.prod_anchor)
    if d.ref.isna().any():
        raise SystemExit(f"{int(d.ref.isna().sum())} fallback rows without a shipped value")
    c = d[d.detector.notna()].reset_index(drop=True)
    y = c.y.to_numpy(float)
    g = (c.ref.to_numpy(float) - y) ** 2 - (c.detector.to_numpy(float) - y) ** 2
    m = c.date.str[5:7]
    ci, days = CL.day_bootstrap(g, (c.airport + "|" + c.date).to_numpy()) if len(c) else ([0.0, 0.0], 0)
    diff = c.detector.to_numpy(float) - c.ref.to_numpy(float)
    sup = (diff > -600.0) & (diff <= 3600.0)
    r = {"prereg": "research_670_fallback_lfpg/PREREG_fallback_lfpg.md", "fallback_rows": int(len(d)), "covered": int(len(c)),
         "share": float(len(c) / max(len(d), 1)), "by_match": c.how.value_counts().to_dict(),
         "detector_rmse": float(np.sqrt(np.mean((c.detector - c.y) ** 2))) if len(c) else None,
         "shipped_rmse": float(np.sqrt(np.mean((c.ref - c.y) ** 2))) if len(c) else None,
         "gain_mse": float(g.mean()) if len(g) else 0.0, "bootstrap_95": ci, "airport_days": days,
         "by_month": {k: float(g[(m == k).to_numpy()].mean()) for k in ("01", "07", "12") if (m == k).any()},
         "rows_by_month": m.value_counts().to_dict(), "inside_support": int(sup.sum()),
         "gain_inside_support": float(g[sup].mean()) if sup.any() else 0.0}
    mse = r["gain_mse"] * r["share"] * ROWS_2026 / N
    r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    r["rules"] = {"pooled": r["gain_mse"] > 0, "bootstrap_lower_above_zero": ci[0] > 0,
                  "january_and_july": r["by_month"].get("01", -1) > 0 and r["by_month"].get("07", -1) > 0,
                  "twenty_rows": len(c) >= 20, "planning_0_05_s": r["planning_s"] >= 0.05}
    r["verdict"] = "pass" if all(r["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=2, default=str) + "\n")
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()

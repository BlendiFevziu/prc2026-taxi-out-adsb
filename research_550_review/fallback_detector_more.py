"""Test of research_550_review/PREREG_fallback_detector_more.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_550_review/fallback_detector_more.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_550_review"))
import fallback_detector as FD  # noqa: E402

N = 344841
AIRPORTS = {"LEBL": 442, "LSZH": 659, "EDDF": 421}
OUT = ROOT / "research_550_review/results/fallback_detector_more.json"


def main():
    for ap in AIRPORTS:
        FD.TOW[ap] = False
    mapping = FD.AT.OB.callsign_mapping()
    rows = []
    for ap in AIRPORTS:
        maps = (FD.SD.stand_table(ap), {})          # empty X-Plane table: xplane_first falls back to OSM stands
        for d in FD.DAYS:
            rows += FD.day_rows(d, ap, mapping, maps)
        print(ap, len(rows), flush=True)
    d = pd.DataFrame(rows)
    pred = pd.concat([pd.read_parquet(ROOT / f".cache/research_240/fallback_anchor_production/predictions_{p}.parquet",
                                      columns=["MVT_ID_mvt", "y", "prod_anchor"]) for p in FD.PAIRS]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    d["y"], d["ref"] = d.MVT_ID_mvt.map(pred.y), d.MVT_ID_mvt.map(pred.prod_anchor)
    if d.ref.isna().any():
        raise SystemExit(f"{int(d.ref.isna().sum())} fallback rows without a shipped value")
    report = {"prereg": "research_550_review/PREREG_fallback_detector_more.md", "airports": {}}
    mse = 0.0
    for ap in AIRPORTS:
        a = d[d.airport.eq(ap)]
        c = a[a.detector.notna()]
        y = c.y.to_numpy(float)
        g = (c.ref.to_numpy(float) - y) ** 2 - (c.detector.to_numpy(float) - y) ** 2
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-10:]] = False
        m = c.date.str[5:7]
        r = {"fallback_rows": int(len(a)), "covered": int(len(c)), "share": float(len(c) / max(len(a), 1)),
             "by_match": c.how.value_counts().to_dict(),
             "detector_rmse": float(np.sqrt(np.mean((c.detector - c.y) ** 2))) if len(c) else None,
             "shipped_rmse": float(np.sqrt(np.mean((c.ref - c.y) ** 2))) if len(c) else None,
             "gain_mse": float(g.mean()) if len(g) else 0.0, "without_best10": float(g[keep].mean()) if keep.any() else 0.0,
             "by_month": {k: float(g[(m == k).to_numpy()].mean()) for k in ("01", "07", "12") if (m == k).any()}}
        r["rules"] = {"pooled": r["gain_mse"] > 0, "without_best10": r["without_best10"] > 0,
                      "january_and_july": r["by_month"].get("01", -1) > 0 and r["by_month"].get("07", -1) > 0, "twenty_rows": len(c) >= 20}
        r["pass"] = all(r["rules"].values())
        if r["pass"]:
            mse += r["gain_mse"] * r["share"] * AIRPORTS[ap] / N
        report["airports"][ap] = r
    ref = json.loads((ROOT / "research_550_review/results/v74_build.json").read_text())["prediction_fixed_before_upload"]["central"]
    report["reference_central"] = ref
    report["planning_s"] = float(ref - np.sqrt(ref ** 2 - mse)) if mse > 0 else 0.0
    report["passing_airports"] = [ap for ap in AIRPORTS if report["airports"][ap]["pass"]] if report["planning_s"] >= 0.05 else []
    report["verdict"] = "pass" if report["passing_airports"] else "fail"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

"""Parts B and C of research_470_three/PREREG_three.md (part A is excluded label-free there). No fit, no upload.
B: the stand model (stand rows) and C1 (July partial rows) against the raw Amsterdam detector on 27 held-out 2025 days.
C: the frozen C1 at Paris July on all 31 July 2025 days.
Run from the repository root:  python3 -u -B research_470_three/three.py
"""
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
sys.path.insert(0, str(HERE / "research_400_cells"))
import cells_confirm as CC  # noqa: E402  (frozen extraction for new airports, v56 modules loaded there)
import harness as H  # noqa: E402
import layers as L  # noqa: E402
from error_by_track_quality import stratum  # noqa: E402

S = CC.S
ID, N, REF = "MVT_ID_mvt", 344841, 256.0858
JUL31 = tuple(f"2025-07-{d:02d}" for d in range(1, 32))
OUT = HERE / "research_470_three/results/three.json"


def summary(gain, dates):
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    days = {d: float(gain[dates == d].mean()) for d in sorted(set(dates))}
    return {"rows": int(len(gain)), "gain_mse": float(gain.mean()), "without_best100": float(gain[keep].mean()),
            "days_positive_share": float(np.mean([v > 0 for v in days.values()])), "by_day": days}


def part_b(base):
    det = pd.read_parquet(HERE / ".cache/research_360/detector_2025.parquet")
    det = det[det.airport.eq("EHAM")].set_index(ID).detector_taxi
    parts = [S.extract(tuple(S.STAGES["development"]), "development"), S.extract(tuple(S.STAGES["confirm"]), "confirm")]
    extra = HERE / ".cache/research_320/stand/shrink_confirm.parquet"
    if os.path.exists(extra):
        parts.append(pd.read_parquet(extra))
    e = pd.concat(parts, ignore_index=True).drop_duplicates(ID)
    e = e[e[ID].isin(det.index)].reset_index(drop=True)
    if len(e) != len(det) or not np.array_equal(base.loc[e[ID]].to_numpy(float), e.comparator.to_numpy(float)):
        raise SystemExit("B: detector rows missing from the extractions or comparator not bit-identical")
    out = {}
    for name, cat, layer, months in (("B1_stand", "observed_at_assigned_stand", "stand", ("01", "07")),
                                     ("B2_partial_july", "partial_ground", "partial", ("07",))):
        f = e[e.category.eq(cat) & e.date.str[5:7].isin(months)].reset_index(drop=True)
        y = f.y.to_numpy(float)
        pred = L.apply(f, L.correction(layer, f), 1.0)
        dv = det.loc[f[ID]].to_numpy(float)
        gain = (dv - y) ** 2 - (pred - y) ** 2
        r = summary(gain, f.date.to_numpy())
        r["detector_rmse"], r["correction_rmse"] = float(np.sqrt(np.mean((dv - y) ** 2))), float(np.sqrt(np.mean((pred - y) ** 2)))
        r["by_month"] = {m: float(gain[f.date.str[5:7].eq(m).to_numpy()].mean()) for m in months}
        r["rules"] = {"pooled_positive": r["gain_mse"] > 0, "without_best100_positive": r["without_best100"] > 0,
                      "sixty_percent_of_days": r["days_positive_share"] >= 0.6, "every_month_positive": all(v > 0 for v in r["by_month"].values())}
        r["passes"] = all(r["rules"].values())
        out[name] = r
    rows26 = {"B1_stand": 2346, "B2_partial_july": 1050}
    mse = sum(rows26[k] * out[k]["gain_mse"] for k in out if out[k]["passes"])
    joint = float(REF - np.sqrt(REF ** 2 - mse / N)) if mse > 0 else 0.0
    out["joint_planning_s"] = joint
    out["passing"] = [k for k in rows26 if out[k]["passes"] and out[k]["gain_mse"] > 0] if joint >= 0.05 else []
    return out


def part_c(comps):
    f = CC.frame_new_airport("LFPG", JUL31, comps)
    y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
    p = L.apply(f, L.correction("partial", f), 1.0)
    gain = (b - y) ** 2 - (p - y) ** 2
    r = summary(gain, f.date.to_numpy())
    st = pd.Series(stratum(f).to_numpy())
    degraded = (st.str.startswith("sparse") | st.str.endswith("/late")).to_numpy()
    groups = {k: (float(gain[m].mean()) if m.sum() >= 30 else None) for k, m in (("degraded", degraded), ("well_observed", ~degraded))}
    use = {k: (v if v is not None else r["gain_mse"]) for k, v in groups.items()}
    a26 = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(str(HERE / ".cache/research_320/airport_coverage/day_2026-07-*.parquet")))], ignore_index=True)
    a26 = a26[a26.airport.eq("LFPG") & a26.category.eq("partial_ground")].reset_index(drop=True)
    s26 = pd.Series(stratum(a26).to_numpy())
    deg26 = float((s26.str.startswith("sparse") | s26.str.endswith("/late")).mean())
    weighted = deg26 * use["degraded"] + (1 - deg26) * use["well_observed"]
    mse = 1896 * weighted / N
    full_s = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    halves = {"1_to_16": float(gain[f.date.str[8:10].astype(int).le(16).to_numpy()].mean()),
              "17_to_31": float(gain[f.date.str[8:10].astype(int).ge(17).to_numpy()].mean())}
    r.update({"base_rmse": float(np.sqrt(np.mean((b - y) ** 2))), "c1_rmse": float(np.sqrt(np.mean((p - y) ** 2))), "halves": halves,
              "groups": {k: {"rows": int(m.sum()), "gain_mse": groups[k]} for k, m in (("degraded", degraded), ("well_observed", ~degraded))},
              "degraded_share_2026": deg26, "weighted_gain": weighted, "planning_full_s": full_s})
    r["rules"] = {"pooled_positive": r["gain_mse"] > 0, "without_best100_positive": r["without_best100"] > 0,
                  "each_half_positive": all(v > 0 for v in halves.values()),
                  "groups_positive": all(v > 0 for v in groups.values() if v is not None), "planning_at_least_0_1_s": full_s >= 0.1}
    r["passes"] = all(r["rules"].values())
    return r


def main():
    comps = H.components_2025()
    comps["comparator"] = L.compose_base(comps)
    base = pd.Series(comps.comparator.to_numpy(float), index=comps[ID].to_numpy())
    report = {"prereg": "research_470_three/PREREG_three.md", "A_heathrow_july_stand": "excluded label-free (drift rule, three flags)",
              "B_amsterdam_detector": part_b(base), "C_paris_july": part_c(comps)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    b, c = report["B_amsterdam_detector"], report["C_paris_july"]
    for k in ("B1_stand", "B2_partial_july"):
        x = b[k]
        print(k, "PASS" if x["passes"] else "fail", json.dumps({kk: x[kk] for kk in ("rows", "detector_rmse", "correction_rmse", "gain_mse", "without_best100", "days_positive_share", "by_month")}), flush=True)
    print("B joint planning", round(b["joint_planning_s"], 3), "passing", b["passing"])
    print("C_paris_july", "PASS" if c["passes"] else "fail", json.dumps({kk: c[kk] for kk in ("rows", "base_rmse", "c1_rmse", "gain_mse", "without_best100", "halves", "groups", "planning_full_s", "rules")}), flush=True)


if __name__ == "__main__":
    main()

"""Heathrow July (frozen C1) and the C1 + CatBoost blend, on seven fresh July 2025 days (research_440_fresh_july/
PREREG_fresh_july.md). No fit, no upload.
Run from the repository root:  python3 -u -B research_440_fresh_july/fresh_july.py
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
from error_by_track_quality import stratum  # noqa: E402

S, MS, R = V.S, V.MS, V.R
ID, N, REF = "MVT_ID_mvt", 344841, 256.6145
DAYS = ("2025-07-01", "2025-07-05", "2025-07-19", "2025-07-20", "2025-07-24", "2025-07-28", "2025-07-30")
HALVES = {"first_half": DAYS[:4], "second_half": DAYS[4:]}
F_CELLS = ("EDDF/07", "EHAM/07", "LEBL/07", "LSZH/07")
C1_ROWS_V65 = 106584
OUT = HERE / "research_440_fresh_july/results/fresh_july.json"


def summary(gain, frame, keep_rows=100):
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-keep_rows:]] = False
    return {"rows": int(len(gain)), "gain_mse": float(gain.mean()), "without_best100": float(gain[keep].mean()),
            "halves": {k: float(gain[frame.date.isin(v).to_numpy()].mean()) for k, v in HALVES.items()},
            "by_day": {d: float(gain[frame.date.eq(d).to_numpy()].mean()) for d in DAYS if frame.date.eq(d).any()}}


def part_a(egll):
    y, b = egll.y.to_numpy(float), egll.comparator.to_numpy(float)
    p = L.apply(egll, L.correction("partial", egll), 1.0)
    gain = (b - y) ** 2 - (p - y) ** 2
    st = pd.Series(stratum(egll).to_numpy())
    degraded = (st.str.startswith("sparse") | st.str.endswith("/late")).to_numpy()
    groups = {k: (float(gain[m].mean()) if m.sum() >= 30 else None) for k, m in (("degraded", degraded), ("well_observed", ~degraded))}
    # 2026 EGLL July mix on base-owned partial rows; sister-month (EGLL January) stratum gains where the fresh sample is thin
    f26 = pd.read_parquet(HERE / ".cache/codex_v56/full_coverage_2026/features.parquet")
    f26 = f26[(f26.airport + "/" + f26.date.str[5:7]).eq("EGLL/07") & f26.category.eq("partial_ground")].reset_index(drop=True)
    own = H.ownership_v62().set_index(ID).owner
    f26 = f26[own.loc[f26[ID]].eq("base").to_numpy()].reset_index(drop=True)
    mix = pd.Series(stratum(f26).to_numpy()).value_counts(normalize=True)
    sister = json.loads((HERE / "research_360_integration/results/error_by_track_quality.json").read_text())["partial_cells"]["EGLL/01"]["strata"]
    used = {}
    for s_, share in mix.items():
        m = (st == s_).to_numpy()
        if m.sum() >= 30:
            used[s_] = ("fresh", float(gain[m].mean()))
        else:
            g = sister.get(s_, {}).get("layer_gain_per_row_2025")
            used[s_] = ("sister", float(g) if g is not None else 0.0)
    weighted = float(sum(share * used[s_][1] for s_, share in mix.items()))
    mse = len(f26) * weighted / N
    full_s = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    out = summary(gain, egll)
    out.update({"base_rmse": float(np.sqrt(np.mean((b - y) ** 2))), "c1_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                "groups": {"degraded": {"rows": int(degraded.sum()), "gain_mse": groups["degraded"]},
                           "well_observed": {"rows": int((~degraded).sum()), "gain_mse": groups["well_observed"]}},
                "rows_2026_base_owned": int(len(f26)), "mix_2026": mix.round(4).to_dict(), "stratum_gain_used": used,
                "weighted_gain": weighted, "planning": {"mse_per_scored_row": mse, "full_s": full_s}})
    out["rules"] = {"1_pooled_positive": out["gain_mse"] > 0, "2_best100_removed_positive": out["without_best100"] > 0,
                    "3_each_half_positive": all(v > 0 for v in out["halves"].values()),
                    "4_groups_positive": all(v > 0 for v in groups.values() if v is not None),
                    "5_planning_at_least_0_1_s": weighted > 0 and full_s >= 0.1}
    out["verdict"] = "pass" if all(out["rules"].values()) else "fail"
    return out


def part_b(frame):
    y = frame.y.to_numpy(float)
    c1 = V.seed_corrections(frame, "c1")
    x = MS.inputs(frame, "real")
    cat = {s: R.joblib.load(HERE / f".cache/research_430/models/cat_f_E_{s}.pkl").predict(x) for s in V.SEEDS}
    mean = lambda c: np.mean([c[s] for s in V.SEEDS], axis=0)
    p_c1 = V.pred(frame, mean(c1))
    p_f = V.pred(frame, mean({s: (c1[s] + cat[s]) / 2 for s in V.SEEDS}))
    gain = (p_c1 - y) ** 2 - (p_f - y) ** 2
    cell = (frame.airport + "/" + frame.date.str[5:7]).to_numpy()
    out = summary(gain, frame)
    out["by_cell"] = {c: float(gain[cell == c].mean()) for c in sorted(set(cell))}
    mse = C1_ROWS_V65 * out["gain_mse"] / N
    out["planning_full_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    out["rules"] = {"1_pooled_positive": out["gain_mse"] > 0, "2_best100_removed_positive": out["without_best100"] > 0,
                    "3_each_half_positive": all(v > 0 for v in out["halves"].values()),
                    "4_three_of_four_cells": sum(out["by_cell"].get(c, 0) > 0 for c in F_CELLS) >= 3,
                    "5_planning_at_least_0_1_s": out["planning_full_s"] >= 0.1}
    out["verdict"] = "pass" if all(out["rules"].values()) else "fail"
    return out


def main():
    missing = [d for d in DAYS if not (S.MS.R.MD.ADSB / f"tracks/{d}.parquet").exists()]
    if missing:
        raise SystemExit(f"tracks missing: {missing}")
    with V.heavy_work("research_440: extraction and detector replay of seven fresh July days", poll_s=10.0):
        ext = S.extract(DAYS, "fresh_july")
        det = set(V.detector_ids(DAYS))
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    ext = ext[ext.category.eq("partial_ground") & ~ext[ID].isin(det)].reset_index(drop=True)
    checks = {"comparator_bit_identical": bool(np.array_equal(base.loc[ext[ID]].to_numpy(float), ext.comparator.to_numpy(float))),
              "rows_by_airport": ext.airport.value_counts().to_dict()}
    if not checks["comparator_bit_identical"]:
        raise SystemExit("comparator not bit-identical")
    report = {"prereg": "research_440_fresh_july/PREREG_fresh_july.md", "days": list(DAYS), "checks": checks}
    report["A_heathrow_july"] = part_a(ext[ext.airport.eq("EGLL")].reset_index(drop=True))
    cells_b = list(F_CELLS) + (["EGLL/07"] if report["A_heathrow_july"]["verdict"] == "pass" else [])
    frame_b = ext[(ext.airport + "/" + ext.date.str[5:7]).isin(cells_b)].reset_index(drop=True)
    report["B_blend"] = part_b(frame_b)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    for part in ("A_heathrow_july", "B_blend"):
        r = report[part]
        print(part, r["verdict"], json.dumps({k: r[k] for k in ("rows", "gain_mse", "without_best100", "halves", "rules")}), flush=True)


if __name__ == "__main__":
    main()

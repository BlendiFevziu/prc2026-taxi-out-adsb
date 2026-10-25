"""Where the exact v62 pipeline still errs on held-out 2025 rows, by track quality, and how common each track-quality
stratum is in 2026. Gate for the coverage-robustness experiment (Codex's proposal of 29 September). No fit.

Rows: every main-lane departure of 27 held-out days (B1, B2 and 5, 7, 19 January, 15, 17 July; none is a training
day of any layer, all are in the base's held-out months). Owners as in v62: direct detector (Munich, Amsterdam),
partial (C1), stand, Amsterdam January, else the base; fallback frozen and scored separately.
Strata for partial-track rows, fixed before any error is looked at, on the two axes where Codex's audit found 2025
and 2026 apart: density = ground points (<= 10 / 11-40 / > 40) and start = distance of the first ground point from
the assigned stand (<= 200 m / 200-800 m / > 800 m). Other categories are strata of their own.
Run from the repository root:  python3 -u -B research_360_integration/error_by_track_quality.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import layers as L  # noqa: E402
from prepare_detectors_2025 import DAYS  # noqa: E402

ID = H.ID
C = HERE / ".cache"
OUT = HERE / "research_360_integration/results/error_by_track_quality.json"
N_SCORED = 344841
DENSITY, START = (10, 40), (200, 800)


def stratum(frame):
    d = np.where(frame.ground_points <= DENSITY[0], "sparse", np.where(frame.ground_points <= DENSITY[1], "medium", "dense"))
    s = np.where(frame.stand_first_m <= START[0], "at_stand", np.where(frame.stand_first_m <= START[1], "near", "late"))
    s = np.where(frame.stand_first_m.isna(), "no_stand", s)
    return pd.Series(np.where(frame.category.eq("partial_ground"), pd.Series(d) + "/" + pd.Series(s), frame.category.fillna("not_extracted")),
                     index=frame.index)


def eham_features():
    paths = [C / f"codex_eham/eham_recovery/features_{d}.parquet" for d in DAYS if (C / f"codex_eham/eham_recovery/features_{d}.parquet").exists()]
    paths += [C / f"research_320/eham_weight/features/features_{d}.parquet" for d in DAYS if (C / f"research_320/eham_weight/features/features_{d}.parquet").exists()]
    paths += [C / f"research_360/eham_features/features_{d}.parquet" for d in DAYS if (C / f"research_360/eham_features/features_{d}.parquet").exists()]
    f = pd.concat([pd.read_parquet(p, columns=[ID] + list(L.EHAM_TRACK)) for p in paths]).drop_duplicates(ID)
    return f


def main():
    ext = pd.concat([pd.read_parquet(C / f"research_320/stand/{t}.parquet") for t in ("development", "confirm", "shrink_confirm")])
    ext = ext[ext.date.isin(DAYS)].drop_duplicates(ID).reset_index(drop=True)
    ext["cell"] = ext.airport + "/" + ext.date.str[5:7]
    det = pd.read_parquet(C / "research_360/detector_2025.parquet").set_index(ID).detector_taxi
    comps = H.components_2025()
    rows = comps[comps.date.isin(DAYS)].reset_index(drop=True)
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    free = ~ext[ID].isin(det.index)
    frames = {"partial": ext[free & ext.category.eq("partial_ground") & ext.cell.isin(L.LAYERS["partial"].cells)].reset_index(drop=True),
              "stand": ext[free & ext.category.eq("observed_at_assigned_stand") & ext.cell.isin(L.LAYERS["stand"].cells)].reset_index(drop=True)}
    eh = ext[free & ext.category.eq("partial_ground") & ext.cell.eq("EHAM/01")]
    frames["eham_january"] = eh.merge(eham_features(), on=ID, how="inner", validate="one_to_one").reset_index(drop=True)
    keep, pred = H.policy_2025(base, rows, frames, set_aside=det.index)
    pol = pd.Series(pred, index=keep[ID].to_numpy())
    pol = pd.concat([pol, det.loc[det.index.isin(rows[ID])]])          # detector rows at their detector value
    rows = rows.set_index(ID).loc[pol.index].rename_axis(ID).reset_index()
    owner = pd.Series("base", index=rows[ID].to_numpy())
    for k, f in frames.items():
        owner.loc[f[ID].to_numpy()] = k
    owner.loc[owner.index.isin(det.index)] = "direct_detector"
    info = rows[[ID, "airport", "date", "y"]].merge(ext[[ID, "category", "cell", "ground_points", "stand_first_m"]], on=ID, how="left")
    info["cell"] = info.cell.fillna(info.airport + "/" + info.date.str[5:7])
    info["stratum"] = stratum(info).to_numpy()
    info["owner"] = owner.loc[info[ID]].to_numpy()
    y = info.y.to_numpy(float)
    info["e_policy"] = (pol.loc[info[ID]].to_numpy(float) - y) ** 2
    info["e_base"] = (base.loc[info[ID]].to_numpy(float) - y) ** 2
    total = info.e_policy.sum()

    def summary(g):
        return pd.Series({"rows": len(g), "rmse_policy": float(np.sqrt(g.e_policy.mean())), "rmse_base": float(np.sqrt(g.e_base.mean())),
                          "layer_gain_per_row": float((g.e_base - g.e_policy).mean()), "share_of_main_sse": float(g.e_policy.sum() / total)})

    report = {"days": list(DAYS), "main_rows": int(len(info)), "rmse_policy": float(np.sqrt(info.e_policy.mean())),
              "rmse_base": float(np.sqrt(info.e_base.mean())),
              "by_owner": info.groupby("owner").apply(summary).round(4).to_dict(orient="index"),
              "by_stratum": info.groupby("stratum").apply(summary).round(4).to_dict(orient="index")}
    # 2026 exposure of each stratum, per cell, label-free
    f26 = pd.read_parquet(C / "codex_v56/full_coverage_2026/features.parquet", columns=[ID, "airport", "date", "category", "ground_points", "stand_first_m"])
    f26["cell"] = f26.airport + "/" + f26.date.str[5:7]
    f26["stratum"] = stratum(f26).to_numpy()
    part25 = info[info.stratum.str.contains("/")]
    part26 = f26[f26.stratum.str.contains("/")]
    cells = {}
    for cell in sorted(set(part26.cell)):
        a, b = part25[part25.cell.eq(cell)], part26[part26.cell.eq(cell)]
        share25 = a.stratum.value_counts(normalize=True)
        share26 = b.stratum.value_counts(normalize=True)
        strata = sorted(set(share25.index) | set(share26.index))
        cells[cell] = {"deployed": cell in L.LAYERS["partial"].cells or cell == "EHAM/01", "rows_2025": int(len(a)), "rows_2026": int(len(b)),
                       "strata": {s: {"share_2025": float(share25.get(s, 0.0)), "share_2026": float(share26.get(s, 0.0)),
                                      "rows_2025": int((a.stratum == s).sum()),
                                      "layer_gain_per_row_2025": float((a.e_base - a.e_policy)[a.stratum.eq(s)].mean()) if (a.stratum == s).any() else None,
                                      "rmse_policy_2025": float(np.sqrt(a.e_policy[a.stratum.eq(s)].mean())) if (a.stratum == s).any() else None}
                                  for s in strata}}
    report["partial_cells"] = cells
    # the gate: degraded = sparse or late; how many 2026 rows, how many 2025 proxy rows, how much worse corrected
    deg = lambda s: s.str.startswith("sparse") | s.str.endswith("/late")
    report["gate"] = {
        "rows_2026_degraded_deployed_cells": int((deg(part26.stratum) & part26.cell.isin(list(L.LAYERS["partial"].cells) + ["EHAM/01"])).sum()),
        "rows_2026_degraded_withheld_cells": int((deg(part26.stratum) & part26.cell.isin(["EDDF/07", "EGLL/07"])).sum()),
        "rows_2026_withheld_cells_all": int(part26.cell.isin(["EDDF/07", "EGLL/07"]).sum()),
        "proxy_rows_2025_degraded": int(deg(part25.stratum).sum()),
        "proxy_rows_2025_degraded_by_cell": part25[deg(part25.stratum)].cell.value_counts().to_dict(),
        "gain_per_row_2025_degraded": float((part25.e_base - part25.e_policy)[deg(part25.stratum)].mean()),
        "gain_per_row_2025_well_observed": float((part25.e_base - part25.e_policy)[~deg(part25.stratum)].mean()),
        "base_mse_2025_degraded": float(part25.e_base[deg(part25.stratum)].mean()),
        "base_mse_2025_well_observed": float(part25.e_base[~deg(part25.stratum)].mean()),
        "policy_mse_2025_degraded": float(part25.e_policy[deg(part25.stratum)].mean()),
        "policy_mse_2025_well_observed": float(part25.e_policy[~deg(part25.stratum)].mean())}
    g = report["gate"]
    rel_good = g["gain_per_row_2025_well_observed"] / g["base_mse_2025_well_observed"]
    rel_deg = g["gain_per_row_2025_degraded"] / g["base_mse_2025_degraded"]
    ceiling = max(rel_good - rel_deg, 0.0) * g["base_mse_2025_degraded"] * g["rows_2026_degraded_deployed_cells"] / N_SCORED
    g["relative_gain_well_observed"], g["relative_gain_degraded"] = rel_good, rel_deg
    g["ceiling_mse_per_scored_row_deployed"] = ceiling
    g["ceiling_s_deployed"] = float(258.5995 - np.sqrt(258.5995 ** 2 - ceiling))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("main_rows", "rmse_policy", "rmse_base", "gate")}, indent=1, default=str))
    print(pd.DataFrame(report["by_owner"]).T.round(3).to_string())
    print(pd.DataFrame(report["by_stratum"]).T.sort_values("share_of_main_sse", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()

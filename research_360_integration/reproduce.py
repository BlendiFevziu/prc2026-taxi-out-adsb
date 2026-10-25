"""Real-data replay: with the base left unchanged, the harness must reproduce the existing pipeline exactly.

Light work only (loads saved tables and the frozen seed models, predicts; no fit, no download, no lock needed).
  R1  2026: ownership covers all 344,841 rows once; the pre-NM base equals every layer frame's comparator;
      assemble_2026(current base) is byte-identical to the scored v62 file.
  R2  2025: compose_base(saved components) equals the `comparator` column of every research_320 extraction table.
  R3  2025 partial layer: on the 16,029 B1 rows, C1 against arm 4 reproduces research_320/results/capacity2.json.
  R4  2025 stand layer: on B1's stand rows of the five deployed cells, the deployed model at s = 0.5 reproduces
      research_320/results/stand_development.json cell by cell.
  R5  descriptive: the complete old-base policy on B1 and B2 (all ten airports), the reference point for a new base.
Run from the repository root:  python3 -u -B research_360_integration/reproduce.py
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

ID, T = H.ID, H.T
OUT = HERE / "research_360_integration/results/reproduce.json"


def main():
    r = {}
    # R1 ------------------------------------------------------------------ 2026
    own = H.ownership_v62()
    r["r1_owner_counts"] = own.owner.value_counts().to_dict()
    r["r1_rows"] = int(len(own))
    base = H.current_base_2026()
    ctx = pd.read_parquet(H.CONTEXT_2026).set_index(ID)
    layer_ids = own.loc[own.owner.isin(["partial", "stand", "eham_january"]), ID]
    r["r1_base_equals_layer_comparator"] = bool(np.array_equal(base.loc[layer_ids].to_numpy(float), ctx.loc[layer_ids, "comparator"].to_numpy(float)))
    v62 = pd.read_parquet(H.V62)
    base_rows = own.loc[own.owner.eq("base"), ID]
    r["r1_v62_base_rows_equal_base"] = bool(np.array_equal(v62.set_index(ID).loc[base_rows, T].to_numpy(float), base.loc[base_rows].to_numpy(float)))
    rebuilt = H.assemble_2026(base, own)
    r["r1_ids_order_equal"] = bool(rebuilt[ID].equals(v62[ID]))
    r["r1_values_identical_to_v62"] = bool(np.array_equal(rebuilt[T].to_numpy(), v62[T].to_numpy()))
    r["r1_dtype_equal"] = str(rebuilt[T].dtype) == str(v62[T].dtype)
    r["r1_rows_differing"] = int((rebuilt[T].to_numpy() != v62[T].to_numpy()).sum())

    # R2 ------------------------------------------------------------------ 2025 base
    comps = H.components_2025()
    old = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    r["r2_components_rows"] = int(len(comps))
    r["r2_old_base_rmse"] = float(np.sqrt(np.mean((old.to_numpy() - comps.y.to_numpy(float)) ** 2)))
    for tag in ("train", "development", "confirm", "volume_extra", "shrink_confirm"):
        f = pd.read_parquet(HERE / f".cache/research_320/stand/{tag}.parquet")
        r[f"r2_{tag}_comparator_equal"] = bool(np.array_equal(old.loc[f[ID]].to_numpy(float), f.comparator.to_numpy(float)))

    # R3 ------------------------------------------------------------------ 2025 partial, B1
    dev = pd.read_parquet(HERE / ".cache/research_320/stand/development.parquet").set_index(ID)
    ids = pd.read_parquet(HERE / ".cache/research_320/volume/b1_predictions.parquet")[ID]
    frame = dev.loc[ids].reset_index()
    rows = comps[comps[ID].isin(set(ids))].set_index(ID).loc[ids].reset_index()
    arm4 = [__import__("joblib").load(HERE / f".cache/research_320/volume/models/arm4_real_{s}.pkl") for s in L.SEEDS]
    _, p_c1 = H.policy_2025(old, rows, {"partial": frame})
    _, p_a4 = H.policy_2025(old, rows, {"partial": frame}, models={"partial": arm4})
    y = rows.y.to_numpy(float)
    gain = float((((p_a4 - y) ** 2) - ((p_c1 - y) ** 2)).mean())
    rec = json.loads((HERE / "research_320/results/capacity2.json").read_text())["development"]["c1"]["gain_mse"]
    r["r3_c1_vs_arm4_gain_b1"], r["r3_recorded"] = gain, rec
    r["r3_equal"] = bool(abs(gain - rec) <= 1e-9 * max(1.0, abs(rec)))

    # R4 ------------------------------------------------------------------ 2025 stand, B1
    dev_r = dev.reset_index()
    dev_r["cell"] = dev_r.airport + "/" + dev_r.date.str[5:7]
    stand = dev_r[dev_r.category.eq("observed_at_assigned_stand") & dev_r.cell.isin(L.LAYERS["stand"].cells)].reset_index(drop=True)
    srows = comps.set_index(ID).loc[stand[ID]].reset_index()
    _, p_s = H.policy_2025(old, srows, {"stand": stand})
    base_s = old.loc[stand[ID]].to_numpy(float)
    ys = stand.y.to_numpy(float)
    g = (base_s - ys) ** 2 - (p_s - ys) ** 2
    recs = json.loads((HERE / "research_320/results/stand_development.json").read_text())["arms"]["real"]["by_cell"]
    r["r4_stand_rows"] = int(len(stand))
    r["r4_by_cell"] = {c: {"replayed": float(g[stand.cell.eq(c).to_numpy()].mean()), "recorded": recs[c]["gain_mse"]} for c in L.LAYERS["stand"].cells}
    r["r4_equal"] = bool(all(abs(v["replayed"] - v["recorded"]) <= 1e-9 * max(1.0, abs(v["recorded"])) for v in r["r4_by_cell"].values()))

    # R5 ------------------------------------------------------------------ descriptive complete policy, old base
    r["r5"] = {}
    for tag, pred_file in (("development", "b1_predictions"), ("confirm", "b2_predictions")):
        f = pd.read_parquet(HERE / f".cache/research_320/stand/{tag}.parquet")
        f["cell"] = f.airport + "/" + f.date.str[5:7]
        part_ids = pd.read_parquet(HERE / f".cache/research_320/volume/{pred_file}.parquet")[ID]
        frames = {"partial": f.set_index(ID).loc[part_ids].reset_index(),
                  "stand": f[f.category.eq("observed_at_assigned_stand") & f.cell.isin(L.LAYERS["stand"].cells)].reset_index(drop=True)}
        dates = set(f.date)
        all_rows = comps[comps.date.isin(dates)].reset_index(drop=True)
        keep, pred = H.policy_2025(old, all_rows, frames)
        base_only = old.loc[keep[ID]].to_numpy(float)
        fb = H.fallback_2025(dates)
        res = H.compare(keep, base_only, pred, fallback=fb)
        r["r5"][tag] = {"dates": sorted(dates), "main_rows": res["rows"], "rmse_old_base_alone": res["rmse_before"],
                        "rmse_old_base_plus_v62_layers": res["rmse_after"], "layer_mse_gain": res["mse_gain"],
                        "best100_removed": res["best100_removed_mse_gain"], "airport_day_ci95": res["airport_day"]["ci95"],
                        "rows_owned": {k: int(len(v)) for k, v in frames.items()}, "fallback": res["fallback"],
                        "combined_rmse_old_base_alone": res["combined_rmse_before"], "combined_rmse_old_base_plus_v62_layers": res["combined_rmse_after"],
                        "note": "detector rows (Munich, Amsterdam) not set aside yet; EHAM January layer not applied in 2025 (features missing on NM days)"}
    checks = [k for k in r if k.startswith(("r1_", "r2_", "r3_equal", "r4_equal")) and isinstance(r[k], bool)]
    r["all_reproduced"] = bool(all(r[k] for k in checks))
    r["failed"] = [k for k in checks if not r[k]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=2, default=str) + "\n")
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()

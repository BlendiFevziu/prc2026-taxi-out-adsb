"""The stand model at weight 1.0 instead of 0.5 (research_450_stand_january/PREREG_stand_weight.md). No fit, no upload.
Run from the repository root:  python3 -u -B research_450_stand_january/stand_weight.py
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

S = V.S
ID, N, REF = "MVT_ID_mvt", 344841, 256.4921
CELLS5 = ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07")
JUL_FRESH = ("2025-07-01", "2025-07-05", "2025-07-19", "2025-07-20", "2025-07-24", "2025-07-28", "2025-07-30")
JUL_READ = ("2025-07-06", "2025-07-08", "2025-07-10")
JAN = ("2025-01-03", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-15", "2025-01-17", "2025-01-22", "2025-01-27", "2025-01-29", "2025-01-31")
OUT = HERE / "research_450_stand_january/results/stand_weight.json"


def stand_rows(ext, dates, cells):
    with V.heavy_work("research_450: detector replay for the stand weight test", poll_s=10.0):
        det = set(V.detector_ids(dates))
    ext = ext.copy()
    ext["cell"] = ext.airport + "/" + ext.date.str[5:7]
    return ext[ext.category.eq("observed_at_assigned_stand") & ext.cell.isin(cells) & ~ext[ID].isin(det)].reset_index(drop=True)


def gains(f, base):
    if not np.array_equal(base.loc[f[ID]].to_numpy(float), f.comparator.to_numpy(float)):
        raise SystemExit("comparator not bit-identical")
    corr = L.correction("stand", f)
    y = f.y.to_numpy(float)
    return (L.apply(f, corr, 0.5) - y) ** 2 - (L.apply(f, corr, 1.0) - y) ** 2


def main():
    missing = [d for d in JAN if not (S.MS.R.MD.ADSB / f"tracks/{d}.parquet").exists()]
    if missing:
        raise SystemExit(f"January tracks missing: {missing}")
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    dev_dates = tuple(S.STAGES["development"]) + tuple(S.STAGES["confirm"])
    dev = pd.concat([S.extract(tuple(S.STAGES["development"]), "development"), S.extract(tuple(S.STAGES["confirm"]), "confirm")], ignore_index=True)
    dev = stand_rows(dev.drop_duplicates(ID), dev_dates, CELLS5)
    jul = pd.concat([S.extract(JUL_FRESH, "fresh_july"), S.extract(JUL_READ, "eddf_july_confirm")], ignore_index=True)
    jul = stand_rows(jul, JUL_FRESH + JUL_READ, ("EDDF/07", "LEBL/07"))
    with V.heavy_work("research_450: extraction of the ten fresh January days", poll_s=10.0):
        jan_ext = S.extract(JAN, "fresh_january")
    jan = stand_rows(jan_ext, JAN, ("EDDF/01", "EGLL/01", "LEBL/01"))
    g_dev = gains(dev, base)
    conf = pd.concat([jul, jan], ignore_index=True)
    g = gains(conf, base)
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-100:]] = False
    month = conf.date.str[5:7].to_numpy()
    by_month = {m: float(g[month == m].mean()) for m in ("01", "07")}
    rows_v = 13141
    jan_stand = HERE / "research_450_stand_january/results/stand_january.json"
    if jan_stand.exists() and json.loads(jan_stand.read_text()).get("verdict") == "pass":
        r = json.loads(jan_stand.read_text())
        rows_v += sum({"LSZH/01": 2869, "EHAM/01": 1137}[c] for c in r["passing_cells"])
    mse = rows_v * float(g.mean()) / N
    full_s = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    rules = {"1_development_positive": float(g_dev.mean()) > 0, "2_confirmation_positive": float(g.mean()) > 0,
             "3_confirmation_without_best_100": float(g[keep].mean()) > 0, "4_both_months": all(v > 0 for v in by_month.values()),
             "5_planning_at_least_0_1_s": full_s >= 0.1}
    cell = conf.cell.to_numpy()
    rep = {"prereg": "research_450_stand_january/PREREG_stand_weight.md",
           "development": {"rows": int(len(dev)), "gain_mse": float(g_dev.mean()),
                           "by_cell": {c: float(g_dev[dev.cell.eq(c).to_numpy()].mean()) for c in CELLS5}},
           "confirmation": {"rows": int(len(conf)), "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
                            "by_month": by_month, "by_cell": {c: {"rows": int((cell == c).sum()), "gain_mse": float(g[cell == c].mean())}
                                                              for c in sorted(set(cell))}},
           "rows_at_weight_one": rows_v, "planning_full_s": full_s, "rules": rules, "verdict": "pass" if all(rules.values()) else "fail"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: rep[k] for k in ("development", "confirmation", "planning_full_s", "rules", "verdict")}, indent=1), flush=True)


if __name__ == "__main__":
    main()

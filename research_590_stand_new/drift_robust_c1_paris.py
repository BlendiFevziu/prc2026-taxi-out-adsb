"""Test of research_590_stand_new/PREREG_drift_robust_c1_paris.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_590_stand_new/drift_robust_c1_paris.py
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
ID, N, REF = "MVT_ID_mvt", 344841, 247.13
DAYS = tuple(sorted(ST2.FRESH + ST2.CODEX))
OUT = ROOT / "research_590_stand_new/results/drift_robust_c1_paris.json"


def frame(days):
    rows = ST2.S1.extract_2025(ST2.S1.departures_2025(list(days)), list(days))
    rows = rows[rows.airport.eq("LFPG") & rows.category.eq("partial_ground")]
    comps = H.components_2025()
    comps["comparator"] = L.compose_base(comps)
    f = rows.merge(comps[[ID, "y", "comparator", "ersatz_s", "schedule", "gate", "MVT_TIME_UTC_mvt"]], on=ID, how="left", validate="one_to_one")
    if f.y.isna().any():
        raise SystemExit(f"{int(f.y.isna().sum())} LFPG rows lack a held-out component row")
    epoch = (pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True) - ST2.R.MD.EPOCH0).dt.total_seconds()
    phase = 2 * np.pi * (epoch % 86400) / 86400
    f["hour_sin"], f["hour_cos"] = np.sin(phase), np.cos(phase)
    for ap in ("EDDF", "EGLL", "EHAM", "LEBL", "LSZH"):
        f[f"airport_{ap}"] = 0.0
    f["is_july"] = 1.0
    return f.reset_index(drop=True)


def main():
    cols = json.loads((ROOT / "research_590_stand_new/results/drift_robust_c1.json").read_text())["inputs"]
    ms = DC.models(None, cols)
    f = frame(DAYS)
    c = {s: m.predict(MS.inputs(f, "real")[cols]) for s, m in zip(L.SEEDS, ms)}
    r = DC.judge(f, c, f.comparator.to_numpy(float), {"first_half": DAYS[:12], "second_half": DAYS[12:]})
    own = H.ownership_v62().set_index(ID).owner
    audit = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_320/airport_coverage/day_2026-*.parquet")))], ignore_index=True)
    audit = audit[audit.airport.eq("LFPG") & audit.category.eq("partial_ground") & own.reindex(audit[ID]).eq("base").to_numpy()]
    months, mse = {}, 0.0
    for m in ("07", "01"):
        t = audit[audit.date.str[5:7].eq(m)]
        fl = DC.flags(f, t)
        ok = r["pass"] and not fl
        months[f"LFPG/{m}"] = {"rows_2026": int(len(t)), "drift_flags_kept": fl, "admitted": ok}
        if ok:
            mse += r["gain_mse"] * len(t) / N
    report = {"prereg": "research_590_stand_new/PREREG_drift_robust_c1_paris.md", "test": r, "months": months,
              "planning_s": float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else 0.0}
    report["verdict"] = "pass" if report["planning_s"] >= 0.05 else "fail"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

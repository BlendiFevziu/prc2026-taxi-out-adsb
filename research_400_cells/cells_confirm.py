"""The frozen C1 model in three unused cells, judged cell by cell (research_400_cells/PREREG_cells.md). No fit, no upload.

EDDM/01 and LEMD/07: the frozen extraction (full_coverage_2026.extract_day at 9884150, OSM stands, as the 2026 coverage
audit ran it) on the local 2025 days, joined to the held-out 2025 components; EGLL/07: the cached v56 extraction of
6/8/10 July 2025. C1 through layers.correction("partial", ...), applied with layers.apply at weight 1.0.
Run from the repository root:  python3 -u -B research_400_cells/cells_confirm.py
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
R = MS.R
from research_310 import full_coverage_2026 as F  # noqa: E402  (Codex's frozen module, from the checkout)
from research_320 import stand_experiment as S  # noqa: E402
from research_300.heavy_lock import heavy_work  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402
from error_by_track_quality import stratum  # noqa: E402

ID, N, REF, SEEDS = "MVT_ID_mvt", 344841, 258.0008, (177, 277, 377)
FLAGS = ("EDDF", "EGLL", "EHAM", "LEBL", "LSZH")
TRACKS = R.MD.ADSB / "tracks"
CACHE = HERE / ".cache/research_400/days"
OUT = HERE / "research_400_cells/results/cells_confirm.json"
JAN = tuple(f"2025-01-{d}" for d in ("01", "02", "04", "05", "06", "07", "08", "09", "13", "14",
                                     "16", "18", "19", "20", "21", "23", "24", "25", "26", "28", "30"))
JUL = tuple(f"2025-07-{d}" for d in ("02", "03", "04", "06", "07", "08", "09", "10", "11", "12", "13", "14",
                                     "15", "16", "17", "18", "21", "22", "23", "25", "26", "27", "29", "31"))
EGLL_DAYS = ("2025-07-06", "2025-07-08", "2025-07-10")
CELLS = {"EDDM/01": {"days": JAN, "blocks": {"first_half": JAN[:10], "second_half": JAN[10:]}, "rows_2026": 6802, "degraded_2026": 0.1307},
         "LEMD/07": {"days": JUL, "blocks": {"first_half": JUL[:12], "second_half": JUL[12:]}, "rows_2026": 8692, "degraded_2026": 0.5383},
         "EGLL/07": {"days": EGLL_DAYS, "blocks": {d: (d,) for d in EGLL_DAYS}, "rows_2026": 9048, "degraded_2026": 0.0619}}
COLUMNS = [ID, "PHASE_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt"]


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def departures(airport, days):
    paths = [p for m in sorted({d[:7] for d in days}) for p in sorted((HERE / "data").glob(f"training_{m}-01_*.parquet"))]
    raw = pd.concat([pd.read_parquet(p, columns=COLUMNS) for p in paths], ignore_index=True)
    dep = raw[raw.PHASE_mvt.eq("DEP") & raw.AOBT_3_flt.notna() & raw.ADEP_mvt.eq(airport)].copy()
    dep["date"] = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    dep = dep[dep.date.isin(days)].copy()
    dep["airport"] = airport
    dep["callsign"] = dep.FLIGHT_mvt.astype(str).str.strip().str.upper()
    dep["mvt_epoch"] = (pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True) - R.MD.EPOCH0).dt.total_seconds()
    return dep


def extracted(airport, days):
    """Frozen extraction for one new airport, one cached file per day."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dep = departures(airport, days)
    todo = [d for d in days if not (CACHE / f"{airport}_{d}.parquet").exists()]
    if todo:
        mapping, maps = R.AT.OB.callsign_mapping(), {airport: (R.SD.stand_table(airport), {})}
        saved = MS.AIRPORTS
        with heavy_work(f"research_400: frozen extraction {airport}, {len(todo)} days", log=log):
            try:
                MS.AIRPORTS = (airport,)
                for d in todo:
                    F.extract_day(dep[dep.date.eq(d)], pd.read_parquet(TRACKS / f"{d}.parquet"), mapping, maps).to_parquet(
                        CACHE / f"{airport}_{d}.parquet", index=False)
            finally:
                MS.AIRPORTS = saved
    rows = pd.concat([pd.read_parquet(CACHE / f"{airport}_{d}.parquet") for d in days], ignore_index=True)
    if set(rows[ID]) != set(dep[ID]):
        raise SystemExit(f"{airport}: extracted rows are not exactly the main-lane departures")
    return rows


def frame_new_airport(airport, days, comps):
    rows = extracted(airport, days)
    rows = rows[rows.category.eq("partial_ground")]
    f = rows.merge(comps[[ID, "y", "comparator", "ersatz_s", "schedule", "gate", "MVT_TIME_UTC_mvt"]], on=ID, how="left",
                   validate="one_to_one")
    if f.y.isna().any():
        raise SystemExit(f"{airport}: {int(f.y.isna().sum())} partial rows lack a held-out component row")
    epoch = (pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True) - R.MD.EPOCH0).dt.total_seconds()
    phase = 2 * np.pi * (epoch % 86400) / 86400
    f["hour_sin"], f["hour_cos"] = np.sin(phase), np.cos(phase)
    for ap in FLAGS:
        f[f"airport_{ap}"] = 0.0                                   # C1 never saw this airport
    f["is_july"] = f.date.str[5:7].eq("07").astype(float)
    return f.reset_index(drop=True)


def frame_egll(comps):
    ext = S.extract(EGLL_DAYS, "eddf_july_confirm")                # cached by the EDDF July confirmation
    f = ext[ext.airport.eq("EGLL") & ext.date.isin(EGLL_DAYS) & ext.category.eq("partial_ground")].reset_index(drop=True)
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    if not np.array_equal(base.loc[f[ID]].to_numpy(float), f.comparator.to_numpy(float)):
        raise SystemExit("EGLL: comparator is not bit-identical to the recomposed base")
    return f


def judge(cell, f):
    spec = CELLS[cell]
    corr = L.correction("partial", f)
    rev = L.correction("partial", f.iloc[::-1].reset_index(drop=True))[::-1]
    y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
    p = L.apply(f, corr, 1.0)
    gain = (b - y) ** 2 - (p - y) ** 2
    st = pd.Series(stratum(f).to_numpy())
    degraded = (st.str.startswith("sparse") | st.str.endswith("/late")).to_numpy()
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    groups = {"degraded": degraded, "well_observed": ~degraded}
    group_gain = {k: (float(gain[m].mean()) if m.sum() >= 30 else None) for k, m in groups.items()}
    use = {k: (v if v is not None else float(gain.mean())) for k, v in group_gain.items()}
    reweighted = spec["degraded_2026"] * use["degraded"] + (1 - spec["degraded_2026"]) * use["well_observed"]
    mse = spec["rows_2026"] * reweighted / N
    full_s = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    blocks = {k: float(gain[f.date.isin(v).to_numpy()].mean()) for k, v in spec["blocks"].items()}
    rules = {"1_pooled_positive": float(gain.mean()) > 0, "2_best100_removed_positive": float(gain[keep].mean()) > 0,
             "3_each_block_positive": all(v > 0 for v in blocks.values()),
             "4_groups_positive": all(v > 0 for v in group_gain.values() if v is not None),
             "5_reweighted_positive_and_at_least_0_1_s": reweighted > 0 and full_s >= 0.1}
    return {"rows": int(len(f)), "days": int(f.date.nunique()), "reverse_replay": bool(np.array_equal(corr, rev)),
            "base_rmse": float(np.sqrt(np.mean((b - y) ** 2))), "c1_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "gain_mse": float(gain.mean()), "without_best100": float(gain[keep].mean()),
            "best100_share_of_gain": float(np.sort(gain)[-100:].sum() / gain.sum()) if gain.sum() > 0 else None,
            "blocks": blocks, "by_day": {d: float(gain[f.date.eq(d).to_numpy()].mean()) for d in sorted(f.date.unique())},
            "groups": {k: {"rows": int(m.sum()), "gain_mse": group_gain[k]} for k, m in groups.items()},
            "by_stratum": {s: {"rows": int((st == s).sum()), "gain_mse": float(gain[(st == s).to_numpy()].mean())} for s in sorted(st.unique())},
            "reweighted_to_2026_mix": reweighted, "planning": {"mse_per_scored_row": mse, "full_s": full_s},
            "rules": rules, "verdict": "pass" if all(rules.values()) and bool(np.array_equal(corr, rev)) else "fail"}


def main():
    cap = json.loads((HERE / "research_320/results/capacity2.json").read_text())
    if not all(sha(HERE / f".cache/research_320/volume/models/c1_real_{s}.pkl") == cap["model_hashes"][f"c1_real_{s}.pkl"] for s in SEEDS):
        raise SystemExit("C1 models differ from the tested ones")
    comps = H.components_2025()
    comps["comparator"] = L.compose_base(comps)
    report = {"prereg": "research_400_cells/PREREG_cells.md", "reference_score": REF, "cells": {}}
    for cell in CELLS:
        airport = cell[:4]
        f = frame_egll(comps) if airport == "EGLL" else frame_new_airport(airport, CELLS[cell]["days"], comps)
        report["cells"][cell] = judge(cell, f)
        r = report["cells"][cell]
        log(f"{cell}: {r['rows']} rows, {r['base_rmse']:.1f} -> {r['c1_rmse']:.1f} s, gain {r['gain_mse']:.0f}, "
            f"w/o best 100 {r['without_best100']:.0f}, planning {r['planning']['full_s']:.2f} s, {r['verdict']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

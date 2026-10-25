"""Stage 1 of the Rome line: label-free track coverage at LIRF (and LFPG) in the 2025 months that could train and test a
Rome layer, against January 2026, the scored month with Rome coverage. No label is read, nothing is fitted.

Why. adsb.lol has ground points at LIRF in July 2025 (median 29,091 per day on the local days), December 2025 (13,375)
and January 2026 (13,203), none in January 2025 or July 2026. No layer runs at LIRF: every layer was validated within
its own airport and month, and January 2025 has no Rome tracks. In January 2026, 65.9 % of LIRF's 10,954 scored main
rows have partial or stand-level tracks (research_310/airport_coverage_audit.py).

What. Codex's frozen extraction (full_coverage_2026.extract_day at 9884150, the call the coverage audit makes for the
2026 days) on every local 2025 July and December track day, for the main-lane departures (DEP with AOBT_3_flt) of
LIRF and LFPG in the 2025 training files. Columns read: MVT_ID, PHASE, AOBT_3 (presence only), MVT_TIME, ADEP, FLIGHT,
CALLSIGN, STAND. The 2026 side is the audit's cached per-day table (.cache/research_320/airport_coverage).

Reported per airport and period (2025-07, 2025-12, 2026-01, 2026-07): rows, category shares, usable share, assigned
stand found in the stand table, median ground points and first-point stand distance of partial tracks, the track-quality
strata of research_360 (degraded = at most 10 ground points or first point more than 800 m from the stand), and v56's
drift rule on the 23 track summaries (mean shift above one reference SD or missingness change above 0.20), December
2025 and July 2025 each against January 2026.

Run from the repository root:  python3 -u -B research_380_rome/stage1_coverage.py
"""
import json
import os
from pathlib import Path
import re
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
from research_300.heavy_lock import heavy_work  # noqa: E402
from error_by_track_quality import stratum  # noqa: E402

ID = "MVT_ID_mvt"
AIRPORTS = ("LIRF", "LFPG")
PERIODS_2025 = ("2025-07", "2025-12")
TRACKS = R.MD.ADSB / "tracks"
CACHE = HERE / ".cache/research_380/stage1"
CACHE_2026 = HERE / ".cache/research_320/airport_coverage"
OUT = HERE / "research_380_rome/results/stage1_coverage.json"
STAND, PARTIAL = "observed_at_assigned_stand", "partial_ground"
COLUMNS = [ID, "PHASE_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt"]


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def local_days():
    names = (os.path.basename(p) for p in TRACKS.glob("2025-*.parquet"))
    return sorted(n[:10] for n in names if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.parquet", n) and n[:7] in PERIODS_2025)


def departures_2025(days):
    """Main-lane departures of LIRF and LFPG on those days, shaped like the audit's scored_departures()."""
    paths = [p for m in sorted({d[:7] for d in days}) for p in sorted((HERE / "data").glob(f"training_{m}-01_*.parquet"))]
    raw = pd.concat([pd.read_parquet(p, columns=COLUMNS) for p in paths], ignore_index=True)
    dep = raw[raw.PHASE_mvt.eq("DEP") & raw.AOBT_3_flt.notna() & raw.ADEP_mvt.isin(AIRPORTS)].copy()
    dep["date"] = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    dep = dep[dep.date.isin(days)].copy()
    dep["airport"] = dep.ADEP_mvt.astype(str)
    dep["callsign"] = dep.FLIGHT_mvt.astype(str).str.strip().str.upper()
    dep["mvt_epoch"] = (pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True) - R.MD.EPOCH0).dt.total_seconds()
    if not dep[ID].is_unique:
        raise SystemExit("duplicate movement IDs among the 2025 departures")
    return dep


def extract_2025(dep, days):
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not (CACHE / f"day_{d}.parquet").exists()]
    if todo:
        mapping = R.AT.OB.callsign_mapping()
        maps = {ap: (R.SD.stand_table(ap), {}) for ap in AIRPORTS}
        saved = MS.AIRPORTS
        with heavy_work(f"research_380 stage 1: label-free LIRF/LFPG extraction, {len(todo)} days", log=log):
            try:
                MS.AIRPORTS = AIRPORTS
                for date in todo:
                    started = time.time()
                    out = F.extract_day(dep[dep.date.eq(date)], pd.read_parquet(TRACKS / f"{date}.parquet"), mapping, maps)
                    out.to_parquet(CACHE / f"day_{date}.parquet", index=False)
                    log(f"{date}: {len(out)} rows in {time.time() - started:.0f} s")
            finally:
                MS.AIRPORTS = saved
    rows = pd.concat([pd.read_parquet(CACHE / f"day_{d}.parquet") for d in days], ignore_index=True)
    if set(rows[ID]) != set(dep[ID]):
        raise SystemExit("extracted rows are not exactly the 2025 main-lane departures")
    return rows


def rows_2026():
    frames = [pd.read_parquet(p) for p in sorted(CACHE_2026.glob("day_2026-*.parquet"))]
    rows = pd.concat(frames, ignore_index=True)
    return rows[rows.airport.isin(AIRPORTS)].reset_index(drop=True)


def describe(g):
    g = g.reset_index(drop=True)
    part = g[g.category.eq(PARTIAL)]
    st = stratum(part) if len(part) else pd.Series(dtype=str)
    degraded = st.str.startswith("sparse") | st.str.endswith("/late")
    return {"rows": int(len(g)), "days": int(g.date.nunique()),
            "shares": g.category.value_counts(normalize=True).round(4).to_dict(),
            "usable": round(float(g.category.isin([STAND, PARTIAL]).mean()), 4),
            "stand_level": round(float(g.category.eq(STAND).mean()), 4),
            "stand_available": round(float(g.stand_available.astype(bool).mean()), 4),
            "partial_rows": int(len(part)),
            "partial_median_points": float(part.ground_points.median()) if len(part) else None,
            "partial_median_first_stand_m": float(part.stand_first_m.median()) if len(part) else None,
            "partial_degraded_share": round(float(degraded.mean()), 4) if len(part) else None,
            "partial_strata": st.value_counts(normalize=True).round(4).to_dict()}


def drift(reference, other):
    """v56's drift rule with the reference period's partial tracks as the baseline."""
    shifts, flags = {}, []
    for col in MS.FEATURES:
        a, b = reference[col].astype(float), other[col].astype(float)
        sd, miss = a.std(), abs(b.isna().mean() - a.isna().mean())
        shift = abs(b.mean() - a.mean()) / sd if sd > 0 else 0.0
        shifts[col] = round(float(shift), 3)
        if shift > 1.0 or miss > 0.20:
            flags.append(col)
    return {"flagged": flags, "standardised_mean_shift": shifts}


def main():
    days = local_days()
    dep = departures_2025(days)
    log(f"{len(days)} local 2025 days ({', '.join(PERIODS_2025)}), {len(dep):,} main-lane departures at {'/'.join(AIRPORTS)}")
    rows = pd.concat([extract_2025(dep, days), rows_2026()], ignore_index=True)
    rows["period"] = rows.date.str[:7]
    report = {"days_2025": days, "airports": {}}
    for ap in AIRPORTS:
        a = rows[rows.airport.eq(ap)]
        cells = {p: describe(g) for p, g in a.groupby("period")}
        part = {p: g[g.category.eq(PARTIAL)] for p, g in a.groupby("period")}
        ref = part.get("2026-01")
        checks = {}
        if ref is not None and len(ref) >= 100:
            for p in PERIODS_2025:
                if p in part and len(part[p]) >= 100:
                    checks[f"{p}_against_2026-01"] = drift(ref, part[p])
        report["airports"][ap] = {"periods": cells, "drift": checks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    for ap, r in report["airports"].items():
        print(f"\n{ap}")
        for p, c in r["periods"].items():
            print(f"  {p}: rows {c['rows']:,} days {c['days']}, usable {c['usable']:.3f}, stand-level {c['stand_level']:.3f}, "
                  f"stand in table {c['stand_available']:.3f}, partial {c['partial_rows']:,} (median points "
                  f"{c['partial_median_points']}, first {c['partial_median_first_stand_m'] and round(c['partial_median_first_stand_m'])} m, "
                  f"degraded {c['partial_degraded_share']})")
        for k, v in r["drift"].items():
            print(f"  drift {k}: flagged {v['flagged']}")


if __name__ == "__main__":
    main()

"""LABEL-FREE airport coverage audit of open ADS-B ground evidence, all ten airports, all 62 scored days. No fitting.

Every 2026 scored main-lane departure is classified with Codex's frozen classifier (research_310/partial_track_audit
.evidence via full_coverage_2026.extract_day, commit 9884150: FLIGHT_mvt plus CALLSIGN_flt matching, unique visits,
X-Plane-first stands at EHAM and OSM elsewhere) into observed_at_assigned_stand (stand-level evidence),
partial_ground, sparse_ground, no_observed_ground, no_pre_takeoff_points, unmatched or ambiguous_visit.
EDDF, EGLL, EHAM, LEBL, LSZH: Codex's completed 62-day table (hash-checked copy in .cache/codex_v56/). EDDM, LEMD,
LFPG, LIRF, LTFM: the same extract_day on the same 62 cached track days, one cached file per day. No label is read.

Reported per airport and month: rows, category shares, median ground points and first-point stand distance of
partial tracks, and a season-stability check that applies v56's own drift rule (no feature's mean shifting by more
than one January SD, no missingness change above 0.20) to the 23 summaries of January against July partial tracks.
The owner's stated findings are then checked one by one against explicit rules, reported as supported or not.
Run from the repository root:  python3 -u -B research_310/airport_coverage_audit.py
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
R = MS.R
from research_310 import full_coverage_2026 as F  # noqa: E402  (Codex's frozen module, from the checkout)

ID = "MVT_ID_mvt"
CODEX = HERE / ".cache/codex_v56/full_coverage_2026/features.parquet"
CACHE = HERE / ".cache/research_320/airport_coverage"
OUT = HERE / "research_310/results/airport_coverage_audit.json"
OTHERS = ("EDDM", "LEMD", "LFPG", "LIRF", "LTFM")
STAND, PARTIAL = "observed_at_assigned_stand", "partial_ground"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scored_departures():
    raw = pd.read_parquet(HERE / "data/ranking.parquet", columns=[ID, "PHASE_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt",
                                                                  "MVT_TIME_UTC_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt"])
    dep = raw[raw.PHASE_mvt.eq("DEP") & raw.TAXITIME_SEC_mvt.isna() & raw.AOBT_3_flt.notna()].copy()
    dep["date"] = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    dep["airport"] = dep.ADEP_mvt.astype(str)
    dep["callsign"] = dep.FLIGHT_mvt.astype(str).str.strip().str.upper()
    dep["mvt_epoch"] = (pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True) - R.MD.EPOCH0).dt.total_seconds()
    return dep


def other_airports(dep):
    CACHE.mkdir(parents=True, exist_ok=True)
    mapping = R.AT.OB.callsign_mapping()
    maps = {ap: (R.SD.stand_table(ap), {}) for ap in OTHERS}
    saved = MS.AIRPORTS
    frames = []
    try:
        MS.AIRPORTS = OTHERS
        for date in sorted(dep.date.unique()):
            path = CACHE / f"day_{date}.parquet"
            if not path.exists():
                day = dep[dep.date.eq(date) & dep.airport.isin(OTHERS)]
                F.extract_day(day, pd.read_parquet(R.MD.ADSB / f"tracks/{date}.parquet"), mapping, maps).to_parquet(path, index=False)
                print(date, flush=True)
            frames.append(pd.read_parquet(path))
    finally:
        MS.AIRPORTS = saved
    return pd.concat(frames, ignore_index=True)


def stable(jan, jul):
    """v56's drift rule, January partial tracks as the reference."""
    flags = []
    for col in MS.FEATURES:
        a, b = jan[col].astype(float), jul[col].astype(float)
        sd, miss = a.std(), abs(b.isna().mean() - a.isna().mean())
        shift = abs(b.mean() - a.mean()) / sd if sd > 0 else 0.0
        if shift > 1.0 or miss > 0.20:
            flags.append(col)
    return flags


def main():
    recorded = json.loads(subprocess.run(["git", "show", "9884150:research_310/results/full_coverage_2026.json"], cwd=HERE,
                                         capture_output=True, text=True, check=True).stdout)
    if sha(CODEX) != recorded["features_sha256"]:
        raise SystemExit("Codex's 62-day table does not match his recorded hash")
    dep = scored_departures()
    rows = pd.concat([pd.read_parquet(CODEX), other_airports(dep)], ignore_index=True)
    if not rows[ID].is_unique or set(rows[ID]) != set(dep[ID]):
        raise SystemExit("the audit rows are not exactly the scored main-lane departures")
    rows["month"] = rows.date.str[5:7]
    report = {"scored_main_rows": int(len(rows)), "days": int(rows.date.nunique()), "cells": {}, "airports": {}}
    for (ap, m), g in rows.groupby(["airport", "month"]):
        p = g[g.category.eq(PARTIAL)]
        shares = g.category.value_counts(normalize=True).round(4).to_dict()
        report["cells"][f"{ap}/{m}"] = {
            "rows": int(len(g)), "shares": shares,
            "stand_level": round(float(g.category.eq(STAND).mean()), 4),
            "usable_ground": round(float(g.category.isin([STAND, PARTIAL]).mean()), 4),
            "partial_median_points": float(p.ground_points.median()) if len(p) else None,
            "partial_median_first_stand_m": float(p.stand_first_m.median()) if len(p) else None}
    for ap, g in rows.groupby("airport"):
        jan, jul = g[g.month.eq("01") & g.category.eq(PARTIAL)], g[g.month.eq("07") & g.category.eq(PARTIAL)]
        report["airports"][ap] = {
            "rows": int(len(g)), "stand_level": round(float(g.category.eq(STAND).mean()), 4),
            "usable_ground": round(float(g.category.isin([STAND, PARTIAL]).mean()), 4),
            "matched_unique": round(float((~g.category.isin(["unmatched", "ambiguous_visit"])).mean()), 4),
            "season_flags": stable(jan, jul) if len(jan) >= 100 and len(jul) >= 100 else "fewer than 100 partial tracks in a month"}
    allowed = set(recorded["allowed_cells"])
    for ap, g in rows.groupby("airport"):
        share = {m: float(g[g.month.eq(m)].category.eq(PARTIAL).mean()) for m in ("01", "07")}
        report["airports"][ap]["partial_share_jan_jul"] = [round(share["01"], 4), round(share["07"], 4)]
        report["airports"][ap]["partial_share_difference_pp"] = round(100 * abs(share["01"] - share["07"]), 1)
        report["airports"][ap]["v56_release_gate_both_months"] = (f"{ap}/01" in allowed and f"{ap}/07" in allowed) \
            if ap in ("EDDF", "EGLL", "EHAM", "LEBL", "LSZH") else "not in v56's partial-track scope"
    a = report["airports"]
    rank = sorted(a, key=lambda k: -a[k]["usable_ground"])
    cells = report["cells"]
    report["claims"] = {
        "EDDM, EHAM and EGLL have the strongest ground coverage (top three by usable ground evidence)":
            sorted(rank[:3]) == ["EDDM", "EGLL", "EHAM"],
        "LEBL, LEMD and LSZH are moderate to strong (ranks four to six)": sorted(rank[3:6]) == ["LEBL", "LEMD", "LSZH"],
        "EDDF has many tracks but weak stand-level evidence (matched above 80 %, stand-level below 5 %)":
            a["EDDF"]["matched_unique"] > 0.80 and a["EDDF"]["stand_level"] < 0.05,
        "LFPG is weak (usable ground evidence below 30 %)": a["LFPG"]["usable_ground"] < 0.30,
        "LIRF and LTFM have effectively zero usable ground evidence in July (below 2 %)":
            cells["LIRF/07"]["usable_ground"] < 0.02 and cells["LTFM/07"]["usable_ground"] < 0.02,
        "two-season partial-ground evidence stable at EHAM, EGLL, LEBL, LSZH and not at EDDF (v56 drift rule, Jan vs Jul)":
            all(a[k]["season_flags"] == [] for k in ("EHAM", "EGLL", "LEBL", "LSZH")) and a["EDDF"]["season_flags"] not in ([],)}
    report["usable_ground_rank"] = rank
    report["stand_level_rank"] = sorted(a, key=lambda k: -a[k]["stand_level"])
    report["season_definitions_note"] = ("The stated two-season finding is checked under three definitions: v56's drift rule "
                                         "on January against July 2026 partial tracks; the January-July difference of the "
                                         "partial-track share; v56's release gate (2025 training against 2026), both months.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"{report['scored_main_rows']} rows, {report['days']} days")
    for ap in rank:
        r = a[ap]
        print(f"{ap}: usable {r['usable_ground']:.3f} stand-level {r['stand_level']:.3f} matched {r['matched_unique']:.3f} season flags {r['season_flags']}")
    for c in ("LIRF/07", "LTFM/07", "LIRF/01", "LTFM/01"):
        print(c, "usable", cells[c]["usable_ground"])
    for ap in rank:
        r = a[ap]
        print(f"{ap}: partial share Jan/Jul {r['partial_share_jan_jul']} diff {r['partial_share_difference_pp']} pp, v56 gate both months {r['v56_release_gate_both_months']}")
    print("stand-level rank", report["stand_level_rank"])
    for claim, ok in report["claims"].items():
        print(("SUPPORTED   " if ok else "NOT SUPPORTED ") + claim)


if __name__ == "__main__":
    main()

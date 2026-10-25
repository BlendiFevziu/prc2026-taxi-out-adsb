"""Two-arm tests of the stand detector against production-strength predictions: geometry, then matching.

Each test changes ONE component and keeps the detector fixed: stand_detector.detect_onset_v2 plus the disclosed
arrival-and-stay guard of v54 (research_290/build_munich_candidate.py), applied identically in both arms:
  geometry   A = OSM stands (as validated)            B = X-Plane stand where X-Plane has the name, else OSM
             matching fixed at the validated rule (FLIGHT_mvt spellings)
  matching   A = FLIGHT_mvt spellings (as validated)  B = the same plus the filed callsign CALLSIGN_flt
             geometry fixed at whatever the geometry test decided (GEOMETRY_FOR_MATCHING)

Dates are those reserved in research_300/DATE_RESERVATION.md; rules and decision in PREREG_geometry.md and
PREREG_matching.md, each committed before its labels are read. Comparator: the full v45 ensemble's held-out
main-lane predictions (SHA-256 checked). Statistics: research_300/incremental.py.

CALLSIGN_flt is a column of the official challenge dataset, not of any third-party NM archive.
--dry reports coverage per arm only; it computes no error against a label.
Run from the repository root:  python3 -u -B research_300/arm_test.py {geometry,matching} [--dry]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADSB = Path("adsb-lol")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADSB))
import offblock_from_tracks as OB  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402
from research_300 import incremental as I  # noqa: E402

VALIDATE = ROOT / ".cache/sequence_release_v47/validate"
# per airport; set from the geometry decision in PREREG_matching.md, before the matching test reads any label
GEOMETRY_FOR_MATCHING = {"EHAM": "xplane_first", "EDDF": "osm"}   # geometry test, addendum B (8ed8741)
TAXI_CAP_S = 5400.0   # matching test, DESCRIPTIVE only: the same comparison with detector taxi times above 90 min withdrawn
GUARD_WINDOW_S, GUARD_STAY_S = 1800.0, 1200.0   # v54's guard, the same constants
TESTS = {
    "geometry": {"airports": ("EDDM", "EHAM", "EDDF"),
                 "dates": ("2025-01-09", "2025-01-13", "2025-01-28", "2025-01-30",   # 21 Jan: no release (addendum A)
                           "2025-07-02", "2025-07-09", "2025-07-16", "2025-07-23"),
                 "arms": (("osm", "flight_mvt"), ("xplane_first", "flight_mvt"))},
    "matching": {"airports": ("EHAM", "EDDF"),
                 "dates": ("2025-01-08", "2025-01-16", "2025-01-24", "2025-01-26",
                           "2025-07-04", "2025-07-11", "2025-07-18", "2025-07-25"),
                 "arms": (("per_airport", "flight_mvt"), ("per_airport", "plus_callsign_flt"))},
}
MIN_CHANGED = 20
OFFICIAL, N_SCORED, N_MAIN = 264.2328, 344841, 339551


def guarded(ts, lat, lon, gs, vertices, t_takeoff):
    """detect_onset_v2 plus v54's guard: withdraw a transition that follows motion above FAST_KT within 30 min AND
    after which the aircraft stays at the stand for more than 20 min (docking after a landing)."""
    o = SD.detect_onset_v2(ts, lat, lon, gs, vertices, t_takeoff, SD.R_PARK_M)
    if o["status"] != "ok":
        return o
    at = (ts >= o["t_after"]) & (SD._metres(lat, lon, vertices) <= SD.R_STAND_M)
    stay = ts[at].max() - o["t_after"] if at.any() else 0.0
    before = (ts >= o["t_before"] - GUARD_WINDOW_S) & (ts < o["t_after"]) & np.isfinite(gs)
    if stay > GUARD_STAY_S and before.any() and gs[before].max() > SD.FAST_KT:
        return dict(o, status="arrival_stay_guard")
    return o


def lookup(geometry, osm, xp, stand, airport=None):
    if geometry == "per_airport":
        geometry = GEOMETRY_FOR_MATCHING[airport]
    if geometry == "xplane_first":
        v, status = SD.lookup(xp, stand)
        if v is not None:
            return v, "xplane_" + status
    return SD.lookup(osm, stand)


def match_candidates(departures, visit_table, options):
    """offblock_from_tracks.match with the candidate spellings given explicitly (identical otherwise)."""
    expanded = pd.DataFrame({"MVT_ID_mvt": np.repeat(departures.MVT_ID_mvt.to_numpy(), [len(o) for o in options]),
                             "airport": np.repeat(departures.airport.to_numpy(), [len(o) for o in options]),
                             "mvt_epoch": np.repeat(departures.mvt_epoch.to_numpy(), [len(o) for o in options]),
                             "callsign_adsb": [c for o in options for c in o]})
    joined = expanded.merge(visit_table, on=["callsign_adsb", "airport"], how="inner")
    joined = joined.loc[(joined.t_last - joined.mvt_epoch).abs().le(300)]
    joined["closeness"] = (joined.t_last - joined.mvt_epoch).abs()
    joined = joined.sort_values(["closeness", "points"], ascending=[True, False]).drop_duplicates("MVT_ID_mvt")
    return joined.drop(columns=["closeness"])


def options_for(dep, matching, mapping):
    options = OB.candidates(dep.callsign, mapping)
    if matching == "plus_callsign_flt":
        filed = dep.CALLSIGN_flt.astype(object).where(dep.CALLSIGN_flt.notna(), None)
        options = [o | ({str(f).strip().upper()} if f else set()) for o, f in zip(options, filed)]
    return options


def estimates(date, airport, arms, mapping, maps, filed):
    """Detector taxi time per arm for every labelled departure of one airport-day (NaN = no estimate)."""
    tracks = pd.read_parquet(ADSB / f"tracks/{date}.parquet")
    t = tracks[tracks.airport.astype(str).eq(airport)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    dep = MD.departures(date, airport).merge(filed, on="MVT_ID_mvt", how="left")
    out = dep[["MVT_ID_mvt", "lane", "block_epoch", "mvt_epoch"]].copy()
    out["date"], out["airport"] = date, airport
    visits = OB.visits(t) if len(t) else None
    osm, xp = maps
    for geometry, matching in arms:
        col = f"{geometry}|{matching}"
        taxi = np.full(len(dep), np.nan)
        if visits is not None and len(visits):
            j = dep.merge(match_candidates(dep, visits, options_for(dep, matching, mapping)),
                          on=["MVT_ID_mvt", "airport"], how="inner", suffixes=("", "_v"))
            j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
            pos = {m: i for i, m in enumerate(dep.MVT_ID_mvt)}
            for row in j.itertuples(index=False):
                vertices, _ = lookup(geometry, osm, xp, row.STAND_mvt, airport)
                if vertices is None:
                    continue
                ts, lat, lon, gs, _ = MD.points_of(t, row)
                o = guarded(ts, lat, lon, gs, vertices, row.mvt_epoch)
                if o["status"] == "ok":
                    taxi[pos[row.MVT_ID_mvt]] = row.mvt_epoch - (o["t_before"] + o["t_after"]) / 2
        out[col] = taxi
    return out


def comparator():
    path, info = VALIDATE / "ensemble_baseline.parquet", json.loads((VALIDATE / "ensemble_baseline.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["baseline_sha256"]:
        raise SystemExit("ensemble_baseline.parquet does not match its recorded SHA-256")
    return pd.read_parquet(path, columns=["MVT_ID_mvt", "y", "baseline"])


def evaluate(rows, a, b, share):
    """rows: main-lane departures with y, baseline and the two arms' taxi times."""
    y, comp = rows.y.to_numpy(float), rows.baseline.to_numpy(float)
    g = I.gain_rows(y, comp, rows[a].to_numpy(float), rows[b].to_numpy(float))
    changed = g != 0
    rep = {"departures": int(len(rows)), "changed_rows": int(changed.sum()),
           "coverage": {arm: float(rows[arm].notna().mean()) for arm in (a, b)}}
    for arm in (a, b):
        cov = rows[arm].notna().to_numpy()
        e = (rows.mvt_epoch - rows[arm] - rows.block_epoch).to_numpy(float)[cov]
        rep[f"accuracy[{arm}]"] = {"n": int(cov.sum()), "median_s": float(np.median(e)) if len(e) else None,
                                  "mae_s": float(np.abs(e).mean()) if len(e) else None,
                                  "within_60": int((np.abs(e) <= 60).sum()),
                                  "policy_rmse_on_covered": float(np.sqrt(np.mean((rows[arm].to_numpy(float)[cov] - y[cov]) ** 2))) if cov.any() else None,
                                  "comparator_rmse_on_covered": float(np.sqrt(np.mean((comp[cov] - y[cov]) ** 2))) if cov.any() else None}
    rep["gain_per_departure"] = float(g.mean())
    rep["gain_per_changed_row"] = float(g[changed].mean()) if changed.any() else 0.0
    rep["gain_per_changed_row_without_top5pct"] = I.without_top(g)
    rep["airport_day_bootstrap_95"] = I.block_bootstrap(g, (rows.airport + "|" + rows.date).to_numpy())
    rep["months"] = {m: {"departures": int(k.sum()), "changed_rows": int((changed & k).sum()),
                         "gain_per_departure": float(g[k].mean()) if k.any() else None}
                     for m, k in (("January", rows.date.str[5:7].eq("01").to_numpy()),
                                  ("July", rows.date.str[5:7].eq("07").to_numpy()))}
    rep["official_scale_s"] = float(OFFICIAL - np.sqrt(OFFICIAL ** 2 - (N_MAIN / N_SCORED) * share * g.mean()))
    months_ok = all(v["gain_per_departure"] > 0 for v in rep["months"].values() if v["changed_rows"] >= MIN_CHANGED)
    if rep["changed_rows"] < MIN_CHANGED:
        rep["verdict"] = "insufficient"
    else:
        checks = {"pooled_gain_positive": rep["gain_per_departure"] > 0, "each_month_with_data_positive": months_ok,
                  "without_top5pct_positive": rep["gain_per_changed_row_without_top5pct"] > 0,
                  "airport_day_interval_above_zero": rep["airport_day_bootstrap_95"][0] > 0}
        rep["checks"], rep["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    return rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test", choices=sorted(TESTS))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    spec = TESTS[args.test]
    (a_geo, a_match), (b_geo, b_match) = spec["arms"]
    a, b = f"{a_geo}|{a_match}", f"{b_geo}|{b_match}"
    mapping = OB.callsign_mapping()
    filed = pd.concat(pd.read_parquet(p, columns=["MVT_ID_mvt", "CALLSIGN_flt"])
                      for p in sorted((ROOT / "data").glob("training_2025-0[17]-01_*.parquet")))
    maps = {ap: (SD.stand_table(ap), GC.xplane_table(ap)[0]) for ap in spec["airports"]}
    frames = []
    for date in spec["dates"]:
        for ap in spec["airports"]:
            frames.append(estimates(date, ap, spec["arms"], mapping, maps[ap], filed))
        print(date, "done", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    main_rows = rows[rows.lane.eq("main")]
    coverage = {ap: {arm: float(g[arm].notna().mean()) for arm in (a, b)} for ap, g in main_rows.groupby("airport")}
    if args.dry:
        print(json.dumps({"dry_run": True, "main_lane_coverage": coverage}, indent=1))
        return
    model = comparator()
    merged = main_rows.merge(model, on="MVT_ID_mvt", how="inner")
    covered = main_rows[main_rows[[a, b]].notna().any(axis=1)]
    if not covered.MVT_ID_mvt.isin(model.MVT_ID_mvt).all():
        raise SystemExit("row alignment failed: covered main-lane departures missing from the comparator")
    label_check = float(np.abs((merged.mvt_epoch - merged.block_epoch) - merged.y).max())
    scored = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt"])
    scored = scored[scored.PHASE_mvt.eq("DEP") & scored.TAXITIME_SEC_mvt.isna()]
    share = scored.ADEP_mvt.astype(str).value_counts(normalize=True).to_dict()
    report = {"test": args.test, "arms": {"A": a, "B": b}, "dates": list(spec["dates"]),
              "label_check_max_diff_s": label_check, "airports": {}}
    for ap, g in merged.groupby("airport"):
        report["airports"][ap] = evaluate(g, a, b, share.get(ap, 0.0))
        print(ap, report["airports"][ap]["verdict"], flush=True)
    report["pooled_descriptive"] = evaluate(merged, a, b, sum(share.get(ap, 0.0) for ap in spec["airports"]))
    if args.test == "matching":
        capped = merged.copy()
        for arm in (a, b):
            capped.loc[capped[arm] > TAXI_CAP_S, arm] = np.nan
        report["descriptive_taxi_cap_5400s"] = {ap: evaluate(g, a, b, share.get(ap, 0.0))
                                                for ap, g in capped.groupby("airport")}
    (ROOT / ".cache/research_300").mkdir(parents=True, exist_ok=True)
    rows.to_parquet(ROOT / f".cache/research_300/arm_test_{args.test}.parquet", index=False)
    (ROOT / f"research_300/results/arm_test_{args.test}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({ap: {k: v for k, v in r.items() if k in ("verdict", "changed_rows", "gain_per_departure", "official_scale_s", "coverage")}
                      for ap, r in report["airports"].items()}, indent=1))


if __name__ == "__main__":
    main()

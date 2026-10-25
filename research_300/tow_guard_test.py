"""Pre-registered test of the tow and false-onset guard (research_300/PREREG_towguard.md) on fresh dates.

Per departure, with the detector, thresholds, geometry and matching all fixed:
  unguarded   detector taxi where detect_onset_v2 plus v54's guard accept it (arm_test.guarded)
  guarded     the same, minus the estimates research_300/tow_guard.check withdraws
  production  no detector at all (the v45 comparator everywhere)
Comparison 1 (the guard on its own): unguarded against guarded at EHAM. Comparison 2 (the decision): production
against guarded at EHAM. Both with arm_test.evaluate's four rules. The rest is descriptive.
--dry reports coverage only and reads no label.
Run from the repository root:  python3 -u -B research_300/tow_guard_test.py [--dry]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "adsb-lol")
import offblock_from_tracks as OB  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402
from research_300 import arm_test as AT  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402
from research_300 import tow_guard as TG  # noqa: E402

DATES = ("2025-01-20", "2025-01-25", "2025-01-04", "2025-01-14",      # PREREG_towguard.md, seeded draw
         "2025-07-22", "2025-07-03", "2025-07-27", "2025-07-14")
GEOMETRY = {"EHAM": "xplane_first", "EDDM": "xplane_first", "EDDF": "osm"}
DECIDES = "EHAM"
OUT = ROOT / "research_300/results/tow_guard_test.json"


def estimates(date, airport, mapping, maps):
    tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
    t = tracks[tracks.airport.astype(str).eq(airport)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    dep = MD.departures(date, airport)
    out = dep[["MVT_ID_mvt", "lane", "block_epoch", "mvt_epoch", "callsign", "STAND_mvt"]].copy()
    out["date"], out["airport"] = date, airport
    for col in ("unguarded", "guarded", "production"):
        out[col] = np.nan
    out["guard"], out["rest_at_spot_s"], out["longest_stop_s"] = None, np.nan, np.nan
    if not len(t):
        return out
    visits = OB.visits(t)
    if not len(visits):
        return out
    j = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping)),
                  on=["MVT_ID_mvt", "airport"], how="inner", suffixes=("", "_v"))
    j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
    pos = {m: i for i, m in enumerate(dep.MVT_ID_mvt)}
    for row in j.itertuples(index=False):
        vertices, _ = AT.lookup(GEOMETRY[airport], *maps, row.STAND_mvt)
        if vertices is None:
            continue
        ts, lat, lon, gs, ground = MD.points_of(t, row)
        o = AT.guarded(ts, lat, lon, gs, vertices, row.mvt_epoch)
        if o["status"] != "ok":
            continue
        i = pos[row.MVT_ID_mvt]
        taxi = row.mvt_epoch - (o["t_before"] + o["t_after"]) / 2
        g = TG.check(ts, lat, lon, gs, ground, vertices, o, row.mvt_epoch)
        out.iloc[i, out.columns.get_loc("unguarded")] = taxi
        out.iloc[i, out.columns.get_loc("guard")] = g["status"]
        out.iloc[i, out.columns.get_loc("rest_at_spot_s")] = g["rest_at_spot_s"]
        out.iloc[i, out.columns.get_loc("longest_stop_s")] = g["longest_stop_s"]
        if g["status"] == "ok":
            out.iloc[i, out.columns.get_loc("guarded")] = taxi
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    missing = [d for d in DATES if not (MD.ADSB / f"tracks/{d}.parquet").exists()]
    if missing:
        raise SystemExit(f"tracks missing for {missing}: fetch them (research_300/fetch_preferred.py) first")
    mapping = OB.callsign_mapping()
    maps = {ap: (SD.stand_table(ap), GC.xplane_table(ap)[0]) for ap in GEOMETRY}
    frames = []
    for date in DATES:
        for ap in GEOMETRY:
            frames.append(estimates(date, ap, mapping, maps[ap]))
        print(date, "done", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    main_rows = rows[rows.lane.eq("main")]
    coverage = {ap: {arm: float(g[arm].notna().mean()) for arm in ("unguarded", "guarded")}
                for ap, g in main_rows.groupby("airport")}
    withdrawn = main_rows[main_rows.guard.isin(["false_onset", "long_stop"])]
    if args.dry:
        print(json.dumps({"dry_run": True, "main_lane_coverage": coverage,
                          "withdrawn_by_airport_and_rule": withdrawn.groupby(["airport", "guard"]).size()
                          .rename("n").reset_index().to_dict("records")}, indent=1))
        return
    model = AT.comparator()
    merged = main_rows.merge(model, on="MVT_ID_mvt", how="inner")
    covered = main_rows[main_rows.unguarded.notna()]
    if not covered.MVT_ID_mvt.isin(model.MVT_ID_mvt).all():
        raise SystemExit("row alignment failed: covered main-lane departures missing from the comparator")
    scored = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt"])
    scored = scored[scored.PHASE_mvt.eq("DEP") & scored.TAXITIME_SEC_mvt.isna()]
    share = scored.ADEP_mvt.astype(str).value_counts(normalize=True).to_dict()
    merged["err_unguarded"] = merged.unguarded - (merged.mvt_epoch - merged.block_epoch)
    report = {"prereg": "research_300/PREREG_towguard.md", "dates": list(DATES), "geometry": GEOMETRY,
              "label_check_max_diff_s": float(np.abs((merged.mvt_epoch - merged.block_epoch) - merged.y).max()),
              "main_lane_coverage": coverage}
    eham = merged[merged.airport.eq(DECIDES)]
    report["comparison_1_guard_alone"] = AT.evaluate(eham, "unguarded", "guarded", share.get(DECIDES, 0.0))
    report["comparison_2_decision"] = AT.evaluate(eham, "production", "guarded", share.get(DECIDES, 0.0))
    report["descriptive_unguarded_vs_production"] = AT.evaluate(eham, "production", "unguarded", share.get(DECIDES, 0.0))
    w = merged[merged.guard.isin(["false_onset", "long_stop"])]
    report["withdrawn_rows"] = w[["date", "airport", "callsign", "STAND_mvt", "guard", "unguarded", "err_unguarded",
                                  "rest_at_spot_s", "longest_stop_s"]].round(1).to_dict("records")
    report["unguarded_errors_over_2000_s_kept"] = merged[merged.guard.eq("ok") & merged.err_unguarded.abs().gt(2000)][
        ["date", "airport", "callsign", "STAND_mvt", "err_unguarded"]].round(1).to_dict("records")
    report["per_airport_withdrawals"] = {ap: {"covered": int(g.unguarded.notna().sum()),
                                              "withdrawn": int(g.guard.isin(["false_onset", "long_stop"]).sum())}
                                         for ap, g in merged.groupby("airport")}
    rows.to_parquet(ROOT / ".cache/research_300/tow_guard_test.parquet", index=False)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    for name in ("comparison_1_guard_alone", "comparison_2_decision", "descriptive_unguarded_vs_production"):
        r = report[name]
        print(name, json.dumps({k: r.get(k) for k in ("verdict", "changed_rows", "gain_per_departure", "months",
                                                     "gain_per_changed_row_without_top5pct", "airport_day_bootstrap_95",
                                                     "official_scale_s", "checks")}, default=str), flush=True)
    print("withdrawn", json.dumps(report["withdrawn_rows"], default=str))


if __name__ == "__main__":
    main()

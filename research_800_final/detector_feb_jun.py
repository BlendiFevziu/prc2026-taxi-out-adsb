"""Frozen ADS-B stand detectors (Munich, Amsterdam) on February and June: check on 2025, apply to 2026.

The detectors are exactly the validated ones (research_290/stand_detector.py detect_onset_v2 + v54's arrival-and-stay
guard, X-Plane-first stands, and for Amsterdam the tow and false-onset guard). Nothing is re-tuned. Code is imported,
not copied; the same functions as research_300/build_combined_candidate.py.

Stage 1  --check   2025 February/June days that have extracted tracks: every labelled main-lane departure of the two
                   airports gets a detector estimate where one exists; the estimate is compared with the label and with
                   the out-of-sample single-LightGBM comparator (comparator_feb_jun_2025.py, weaker than production).
                   Decision rules per airport, those of research_300/arm_test.evaluate: pooled gain positive, each month
                   with at least 30 changed rows positive, positive without the best 5 % of rows, airport-day bootstrap
                   interval above zero. The comparator is weaker than the production ensemble (it has one tree seed, no
                   CatBoost and no sequence nets), so the gain per row overstates the production gain; the report says so.
Stage 2  --apply   2026 February/June days with extracted tracks: scored main-lane departures of the airports that
                   passed stage 1 get the detector's taxi time. Output: a parquet of (MVT_ID_mvt, TAXITIME_SEC_mvt).

Run from the repository root:
  python3 -u -B research_800_final/detector_feb_jun.py --check
  python3 -u -B research_800_final/detector_feb_jun.py --apply [--airports EDDM EHAM]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADSB = Path("adsb-lol")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADSB))
import offblock_from_tracks as OB  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402
from research_300 import arm_test as AT  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402
from research_300 import incremental as I  # noqa: E402
from research_300 import tow_guard as TG  # noqa: E402

CONFIG = {"EDDM": ("xplane_first", False), "EHAM": ("xplane_first", True)}     # (geometry, tow guard)
OUT = ROOT / ".cache/research_800"
COMPARATOR = OUT / "comparator_feb_jun_2025.parquet"
MIN_CHANGED = 30


def detect(tracks, dep, airport, geometry, tow_guard, mapping, maps):
    """Detector taxi time per MVT id for the main-lane rows of `dep` (as build_combined_candidate.estimates)."""
    t = tracks[tracks.airport.astype(str).eq(airport)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    out, withdrawn = {}, 0
    visits = OB.visits(t) if len(t) else pd.DataFrame()
    if not len(visits) or not len(dep):
        return out, withdrawn
    j = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping)),
                  on=["MVT_ID_mvt", "airport"], how="inner", suffixes=("", "_v"))
    j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
    for row in j[j.lane.eq("main")].itertuples(index=False):
        vertices, _ = AT.lookup(geometry, *maps, row.STAND_mvt)
        if vertices is None:
            continue
        ts, lat, lon, gs, ground = MD.points_of(t, row)
        o = AT.guarded(ts, lat, lon, gs, vertices, row.mvt_epoch)
        if tow_guard and o["status"] == "ok":
            o = TG.check(ts, lat, lon, gs, ground, vertices, o, row.mvt_epoch)
            withdrawn += o["status"] != "ok"
        if o["status"] == "ok":
            out[row.MVT_ID_mvt] = row.mvt_epoch - (o["t_before"] + o["t_after"]) / 2
    return out, withdrawn


def available_days(prefixes):
    days = sorted(p.stem for p in (ADSB / "tracks").glob("20*.parquet") if len(p.stem) == 10)
    return [d for d in days if d[:7] in prefixes]


def maps_for(airports):
    return {ap: (SD.stand_table(ap), GC.xplane_table(ap)[0]) for ap in airports}


def check():
    days = available_days(("2025-02", "2025-06"))
    comparator = pd.read_parquet(COMPARATOR).set_index("MVT_ID_mvt")
    mapping, maps = OB.callsign_mapping(), maps_for(CONFIG)
    rows = []
    for date in days:
        tracks = pd.read_parquet(ADSB / f"tracks/{date}.parquet")
        for ap, (geometry, tow) in CONFIG.items():
            dep = MD.departures(date, ap)
            est, withdrawn = detect(tracks, dep, ap, geometry, tow, mapping, maps[ap])
            main = dep[dep.lane.eq("main")]
            print(date, ap, f"main departures {len(main)}, estimates {len(est)}, tow-guard withdrawn {withdrawn}", flush=True)
            for mid, taxi in est.items():
                rows.append((mid, date, ap, taxi))
    r = pd.DataFrame(rows, columns=["MVT_ID_mvt", "date", "airport", "taxi"])
    r = r[r.MVT_ID_mvt.isin(comparator.index)].copy()
    r["y"] = comparator.loc[r.MVT_ID_mvt, "y"].to_numpy()
    r["comparator"] = comparator.loc[r.MVT_ID_mvt, "comparator"].to_numpy()
    report = {"days": days, "note": "comparator is a single LightGBM, weaker than the production ensemble", "airports": {}}
    for ap, g in r.groupby("airport"):
        gain = (g.comparator - g.y) ** 2 - (g.taxi - g.y) ** 2
        g = g.assign(gain=gain.to_numpy())
        changed = g.gain.to_numpy() != 0
        months = {m: {"rows": int((g.date.str[5:7] == m).sum()),
                      "gain_per_row": float(g.gain[g.date.str[5:7] == m].mean()) if (g.date.str[5:7] == m).any() else None}
                  for m in ("02", "06")}
        lo, hi = I.block_bootstrap(g.gain.to_numpy(), (g.airport + "|" + g.date).to_numpy())
        checks = {"pooled_gain_positive": bool(g.gain.mean() > 0),
                  "each_month_positive": all(v["gain_per_row"] > 0 for v in months.values() if v["rows"] >= MIN_CHANGED),
                  "without_top5pct_positive": bool(I.without_top(g.gain.to_numpy()) > 0),
                  "bootstrap_lower_above_zero": bool(lo > 0)}
        report["airports"][ap] = {
            "rows": int(len(g)), "detector_rmse": float(np.sqrt(np.mean((g.taxi - g.y) ** 2))),
            "comparator_rmse_same_rows": float(np.sqrt(np.mean((g.comparator - g.y) ** 2))),
            "median_error_s": float(np.median(g.taxi - g.y)), "within_60_s": int(((g.taxi - g.y).abs() <= 60).sum()),
            "gain_per_row": float(g.gain.mean()), "gain_per_row_without_top5pct": float(I.without_top(g.gain.to_numpy())),
            "bootstrap_95": [float(lo), float(hi)], "months": months, "checks": checks,
            "verdict": "pass" if all(checks.values()) and len(g) >= MIN_CHANGED else "fail"}
        print(ap, json.dumps(report["airports"][ap], indent=1), flush=True)
    (OUT / "detector_check_feb_jun.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def apply(airports, months=("2026-02", "2026-06"), only_days=None, name="layer_detectors_feb_jun_2026"):
    final = pd.read_parquet(ROOT / "data/final_ranking.parquet",
                            columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
    final = final[final.PHASE_mvt.eq("DEP") & final.ADEP_mvt.astype(str).isin(airports)].copy()
    when = pd.to_datetime(final.MVT_TIME_UTC_mvt, utc=True)
    final = final[when.dt.strftime("%Y-%m").isin(list(months))].copy()
    final["mvt_epoch"] = (pd.to_datetime(final.MVT_TIME_UTC_mvt, utc=True) - MD.EPOCH0).dt.total_seconds()
    final["callsign"] = final.FLIGHT_mvt.astype(str).str.strip().str.upper()
    final["airport"] = final.ADEP_mvt.astype(str)
    final["lane"] = np.where(final.AOBT_3_flt.notna(), "main", "fallback")
    final["day"] = pd.to_datetime(final.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    mapping, maps = OB.callsign_mapping(), maps_for(airports)
    taxi, per_day = {}, {}
    days = sorted(final.day.unique())
    if only_days:
        days = [d for d in days if d in only_days]
    for date in days:
        path = ADSB / f"tracks/{date}.parquet"
        if not path.exists():
            per_day[date] = "no tracks"
            continue
        tracks = pd.read_parquet(path)
        rec = {}
        for ap in airports:
            geometry, tow = CONFIG[ap]
            est, withdrawn = detect(tracks, final[(final.day == date) & (final.airport == ap)].reset_index(drop=True),
                                    ap, geometry, tow, mapping, maps[ap])
            taxi.update(est)
            rec[ap] = {"replaced": len(est), "withdrawn_by_tow_guard": withdrawn}
        per_day[date] = rec
        print(date, rec, flush=True)
    out = pd.DataFrame({"MVT_ID_mvt": list(taxi.keys()), "TAXITIME_SEC_mvt": list(taxi.values())})
    target = OUT / f"{name}.parquet"
    out.to_parquet(target, index=False)
    (OUT / f"{name}.json").write_text(json.dumps({"airports": airports, "rows": int(len(out)),
                                                                        "days_without_tracks": [d for d, v in per_day.items() if v == "no tracks"],
                                                                        "per_day": per_day}, indent=2) + "\n")
    print(f"wrote {target}: {len(out)} rows; days without tracks: {sum(v == 'no tracks' for v in per_day.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--airports", nargs="*", default=list(CONFIG))
    parser.add_argument("--months", nargs="*", default=["2026-02", "2026-06"])
    parser.add_argument("--days", nargs="*", default=None, help="restrict to these days (testing)")
    parser.add_argument("--name", default="layer_detectors_feb_jun_2026")
    args = parser.parse_args()
    if args.check:
        check()
    if args.apply:
        apply(args.airports, tuple(args.months), set(args.days) if args.days else None, args.name)

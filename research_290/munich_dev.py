"""Development day for the assigned-stand detector: Munich (EDDM), 13 December 2025. Labels ARE read here.

This day is development data by design (Codex inspected it first, CLAUDE_LUIS_REVIEW_2026-09-23.md). It is used
for three things only, and every choice made here is frozen in research_290/PREREG_stand_detector.md BEFORE any
validation date is opened:
  1 replicate Codex's diagnostic independently (344 matched, 234 with a ground point within 60 s of block, 190 with
    a mapped assigned stand, 123 near both, old detector 55 detections at median +634.9 s)
  2 run the old detector unchanged, for the comparison
  3 choose ONE estimate among three fixed candidates of research_290/stand_detector.detect: t_rest (last stationary
    point at the stand), t_leave (first point moving or outside the radius), their midpoint

Matching reuses adsb-lol/offblock_from_tracks.py (callsign mapping, visits, match) unchanged. Departures are loaded
here rather than with adsb-lol/validate_offblock.departures, whose file name for December resolves to a month 13.

Run from the repository root:  python3 -u -B research_290/munich_dev.py
"""
import glob
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
from research_290 import stand_detector as SD  # noqa: E402

AIRPORT, DATE = "EDDM", "2025-12-13"
OUTPUT = ROOT / "research_290/results/munich_dev.json"
# Fixed before the onset grid was run, after the three `detect` estimates all failed (see stand_detector.detect_onset):
# parked-distance bound x estimate; the winner has the most detections within 60 s of block, then the lowest MAE.
R_PARK_GRID = (25.0, 30.0)
ESTIMATES = ("t_before", "t_after", "midpoint")
EPOCH0 = pd.Timestamp("1970-01-01", tz="UTC")
COLUMNS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt",
           "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"]


def month_file(date):
    """Strict start-month parsing: the training file whose name starts with that month."""
    hits = sorted(glob.glob(str(ROOT / f"data/training_{date[:7]}-01_*.parquet")))
    if len(hits) != 1:
        raise SystemExit(f"expected one training file for {date[:7]}, found {hits}")
    return hits[0]


def departures(date, airport):
    frame = pd.read_parquet(month_file(date), columns=COLUMNS)
    frame = frame[frame.PHASE_mvt.eq("DEP") & frame.ADEP_mvt.astype(str).eq(airport)].copy()
    t = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
    frame = frame[t.dt.strftime("%Y-%m-%d").eq(date)].reset_index(drop=True)
    frame["mvt_epoch"] = (pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True) - EPOCH0).dt.total_seconds()
    frame["block_epoch"] = (pd.to_datetime(frame.BLOCK_TIME_UTC_mvt, utc=True) - EPOCH0).dt.total_seconds()
    frame["callsign"] = frame.FLIGHT_mvt.astype(str).str.strip().str.upper()
    frame["airport"] = frame.ADEP_mvt.astype(str)
    frame["lane"] = np.where(frame.AOBT_3_flt.notna(), "main", "fallback")
    return frame


def matched_set(date, airport):
    tracks = pd.read_parquet(ADSB / f"tracks/{date}.parquet")
    tracks = tracks[tracks.airport.astype(str).eq(airport)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    dep = departures(date, airport)
    table = OB.visits(tracks)
    joined = dep.merge(OB.match(dep, table, OB.callsign_mapping()), on=["MVT_ID_mvt", "airport"], how="inner",
                       suffixes=("", "_v"))
    claims = joined.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size")
    joined["unique"] = claims.eq(1)
    return tracks, dep, joined


def points_of(tracks, row):
    g = tracks[(tracks.hex == row.hex) & (tracks.ts >= row.t_first) & (tracks.ts <= row.t_last)]
    return g.ts.to_numpy(float), g.lat.to_numpy(float), g.lon.to_numpy(float), \
        pd.to_numeric(g.gs, errors="coerce").to_numpy(float), g.on_ground.to_numpy(bool)


def evaluate(date, airport, verbose=True):
    tracks, dep, joined = matched_set(date, airport)
    table = SD.stand_table(airport)
    rows = []
    for row in joined.itertuples(index=False):
        ts, lat, lon, gs, ground = points_of(tracks, row)
        vertices, stand_status = SD.lookup(table, row.STAND_mvt)
        near_block = np.abs(ts - row.block_epoch)
        rec = {"MVT_ID_mvt": row.MVT_ID_mvt, "lane": row.lane, "unique": bool(row.unique),
               "block_epoch": row.block_epoch, "mvt_epoch": row.mvt_epoch, "stand_status": stand_status,
               "ground_within_60": bool((ground & (near_block <= 60)).any()),
               "ground_within_120": bool((ground & (near_block <= 120)).any()),
               "ground_both_sides_120": bool((ground & (ts < row.block_epoch) & (near_block <= 120)).any()
                                             and (ground & (ts >= row.block_epoch) & (near_block <= 120)).any()),
               "old_usable": bool(row.usable), "old_t": float(row.t_offblock) if row.usable else np.nan}
        if vertices is not None:
            at = SD._metres(lat, lon, vertices) <= SD.R_STAND_M
            rec["near_block_and_stand_60"] = bool((at & (near_block <= 60)).any())
            rec.update({f"det_{k}": v for k, v in SD.detect(ts, lat, lon, gs, vertices, row.mvt_epoch).items()})
            for r_park in R_PARK_GRID:
                onset = SD.detect_onset(ts, lat, lon, gs, vertices, row.mvt_epoch, r_park)
                rec.update({f"on{int(r_park)}_{k}": v for k, v in onset.items()})
        else:
            rec["near_block_and_stand_60"] = False
            rec["det_status"] = stand_status
        rows.append(rec)
    out = pd.DataFrame(rows)
    return dep, out


def error_summary(err):
    err = np.asarray(err, float)
    err = err[np.isfinite(err)]
    if not len(err):
        return {"n": 0}
    return {"n": int(len(err)), "median_s": float(np.median(err)), "mean_s": float(err.mean()),
            "mae_s": float(np.abs(err).mean()), "rmse_s": float(np.sqrt((err ** 2).mean())),
            "within_60": int((np.abs(err) <= 60).sum()), "within_120": int((np.abs(err) <= 120).sum())}


def main():
    dep, out = evaluate(DATE, AIRPORT)
    u = out[out.unique]
    rep = {"date": DATE, "airport": AIRPORT, "departures": int(len(dep)), "matched": int(len(out)),
           "uniquely_matched": int(len(u))}
    rep["codex_replication_on_unique"] = {
        "ground_within_60": int(u.ground_within_60.sum()), "ground_within_120": int(u.ground_within_120.sum()),
        "mapped_assigned_stand": int(u.stand_status.isin(["ok", "ok_subdivided"]).sum()),
        "near_block_and_stand_60": int(u.near_block_and_stand_60.sum()),
        "ground_both_sides_120": int(u.ground_both_sides_120.sum()),
        "old_detections": int(u.old_usable.sum()),
        "old_error_vs_block": error_summary(u.old_t - u.block_epoch),
        "codex_reported": {"matched": 344, "ground_within_60": 234, "ground_within_120": 293, "mapped": 190,
                           "near_block_and_stand_60": 123, "both_sides_120": 121, "old_detections": 55,
                           "old_median_delay_s": 634.91}}
    rep["stand_status"] = u.stand_status.value_counts().to_dict()
    rep["detector_status"] = u.det_status.value_counts().to_dict()
    ok = u[u.det_status.eq("ok")]
    rep["estimates_vs_block_on_ok"] = {
        "t_rest": error_summary(ok.det_t_rest - ok.block_epoch),
        "t_leave": error_summary(ok.det_t_leave - ok.block_epoch),
        "midpoint": error_summary((ok.det_t_rest + ok.det_t_leave) / 2 - ok.block_epoch)}
    rep["gap_s_on_ok"] = error_summary(ok.det_gap_s)
    grid = {}
    for r_park in R_PARK_GRID:
        c = f"on{int(r_park)}"
        good = u[u[f"{c}_status"].eq("ok")]
        est = {"t_before": good[f"{c}_t_before"], "t_after": good[f"{c}_t_after"],
               "midpoint": (good[f"{c}_t_before"] + good[f"{c}_t_after"]) / 2}
        grid[f"r_park_{int(r_park)}"] = {"status": u[f"{c}_status"].value_counts().to_dict(),
                                          **{e: error_summary(est[e] - good.block_epoch) for e in ESTIMATES}}
    rep["onset_grid"] = grid
    scored = [(g[e]["within_60"], -g[e]["mae_s"], k, e) for k, g in grid.items() for e in ESTIMATES if g[e]["n"]]
    best = max(scored)
    rep["onset_choice"] = {"r_park": best[2], "estimate": best[3], "within_60": best[0], "mae_s": -best[1],
                           "criterion": "most detections within 60 s of block, then lowest MAE"}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rep, indent=2, default=float) + "\n")
    out.to_parquet(ADSB / f"tracks/_dev_rows_{DATE}.parquet", index=False)
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()

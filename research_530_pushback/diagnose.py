"""Why the frozen v2 stand detector runs late away from Munich: a diagnostic on the first December 2025 block.

Labels of these days were read in research_290/transfer_2025.py (december part); this is a diagnostic, not a test.
For every v2 estimate, the track between the recorded block time and the estimate is classified:
  moved_slow   the position moves more than 10 m from the parked position while every reported speed is below 2 kt
  stationary   points exist but stay within 10 m of the parked position (block recorded before visible movement,
               or stale surface positions)
  gap          no point at all between the block time and the estimate
and a displacement onset is computed: the first point more than DISP_M from the parked position (median of the last
stationary run within R_PARK_M) that is followed by SUSTAIN further points all beyond DISP_M, as a contestant
described it ("first sustained movement away from the stand, over several consecutive points").
Row-level output in .cache/research_530/ (never committed); aggregates to research_530_pushback/results/.
Run from the repository root:  python3 -u -B research_530_pushback/diagnose.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADSB = Path("adsb-lol")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADSB))
sys.path.insert(0, str(ROOT / "research_290"))
import offblock_from_tracks as OB  # noqa: E402
import munich_dev as MD  # noqa: E402
import stand_detector as SD  # noqa: E402

DATES = ("2025-12-03", "2025-12-05", "2025-12-08", "2025-12-10", "2025-12-11")
AIRPORTS = ("EGLL", "LEMD", "LSZH", "EHAM")
DISP_M, SUSTAIN, LATE_S = 10.0, 2, 90.0
OUT = ROOT / "research_530_pushback/results/diagnose.json"
CACHE = ROOT / ".cache/research_530"


def metres(lat, lon, lat0, lon0):
    dy = (np.asarray(lat) - lat0) * 110540.0
    dx = (np.asarray(lon) - lon0) * 111320.0 * np.cos(np.radians(lat0))
    return np.sqrt(dx * dx + dy * dy)


def examine(ts, lat, lon, gs, vertices, mvt, block):
    out = SD.detect_onset_v2(ts, lat, lon, gs, vertices, mvt, SD.R_PARK_M)
    if out.get("status") != "ok":
        return None
    ts, lat, lon, gs = (np.asarray(a, float) for a in (ts, lat, lon, gs))
    o = np.argsort(ts, kind="stable")
    ts, lat, lon, gs = ts[o], lat[o], lon[o], gs[o]
    est = (out["t_before"] + out["t_after"]) / 2
    i_before = int(np.flatnonzero(ts == out["t_before"])[-1])
    # parked reference: consecutive still points within R_PARK_M ending at t_before
    d_stand = SD._metres(lat, lon, vertices)
    j = i_before
    while j > 0 and np.isfinite(gs[j - 1]) and gs[j - 1] < SD.REST_KT and d_stand[j - 1] <= SD.R_PARK_M:
        j -= 1
    lat0, lon0 = float(np.median(lat[j:i_before + 1])), float(np.median(lon[j:i_before + 1]))
    disp = metres(lat, lon, lat0, lon0)
    rec = {"estimate": est, "err_v2": est - block, "parked_points": int(i_before - j + 1)}
    w = (ts > block) & (ts < est)
    rec["points_between"] = int(w.sum())
    if not w.any():
        rec["kind"] = "gap"
    elif (disp[w] > DISP_M).any():
        rec["kind"] = "moved_slow" if np.nanmax(np.where(w, gs, np.nan)) < SD.REST_KT else "moved"
    else:
        rec["kind"] = "stationary"
    # displacement onset, searched from the first parked point onward and before take-off
    cand = np.flatnonzero((ts >= ts[j]) & (ts <= mvt) & (disp > DISP_M))
    onset = None
    for i in cand:
        nxt = np.flatnonzero((ts > ts[i]) & (ts <= mvt))[:SUSTAIN]
        if len(nxt) == SUSTAIN and (disp[nxt] > DISP_M).all():
            prev = np.flatnonzero(ts < ts[i])
            onset = (ts[prev[-1]] + ts[i]) / 2 if len(prev) else ts[i]
            break
    rec["err_disp"] = (onset - block) if onset is not None else np.nan
    return rec


def main():
    mapping, rows = OB.callsign_mapping(), []
    for date in DATES:
        tracks = pd.read_parquet(ADSB / f"tracks/{date}.parquet")
        for ap in AIRPORTS:
            t = tracks[tracks.airport.astype(str).eq(ap)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
            dep = MD.departures(date, ap)
            if not len(t) or not len(dep):
                continue
            joined = dep.merge(OB.match(dep, OB.visits(t), mapping), on=["MVT_ID_mvt", "airport"], how="inner", suffixes=("", "_v"))
            joined = joined[joined.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
            table = SD.stand_table(ap)
            for row in joined.itertuples(index=False):
                vertices, _ = SD.lookup(table, row.STAND_mvt)
                if vertices is None:
                    continue
                ts, lat, lon, gs, _ = MD.points_of(t, row)
                rec = examine(ts, lat, lon, gs, vertices, row.mvt_epoch, row.block_epoch)
                if rec:
                    rows.append({"date": date, "airport": ap, "MVT_ID_mvt": row.MVT_ID_mvt, **rec})
        print(date, len(rows), flush=True)
    d = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE / "diagnose_rows.parquet", index=False)
    report = {}
    for ap, g in d.groupby("airport"):
        late = g[g.err_v2 > LATE_S]
        both = g[g.err_disp.notna()]
        report[ap] = {"estimates": int(len(g)), "median_v2": float(g.err_v2.median()), "late_share": float(len(late) / len(g)),
                      "late_kinds": {k: int(v) for k, v in late.kind.value_counts().items()},
                      "disp_found": int(len(both)),
                      "median_disp": float(both.err_disp.median()) if len(both) else None,
                      "mae_v2_on_same": float(both.err_v2.abs().mean()) if len(both) else None,
                      "mae_disp": float(both.err_disp.abs().mean()) if len(both) else None,
                      "within60_v2": float((both.err_v2.abs() <= 60).mean()) if len(both) else None,
                      "within60_disp": float((both.err_disp.abs() <= 60).mean()) if len(both) else None}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

"""DEVELOPMENT ONLY, on dates already used: features for a tow / false-onset guard on the stand detector.

Every departure the current detector covers on the geometry and matching test dates (research_300/DATE_RESERVATION.md,
all already evaluated) gets two post-onset features, and its error against the recorded block:
  rest_at_park_s    how long the aircraft is still seen resting within R_PARK_M of its stand AFTER the detected
                    onset (span from the first to the last such point). A push-back leaves the parked position for
                    good; a speed blip at the stand does not (QY1239 on 16 January rested there 46 more minutes).
  longest_stop_s    the longest stationary span after the aircraft first leaves R_STAND_M: consecutive points within
                    STOP_RADIUS_M of the span's first point, on the ground, before take-off. A tow or reposition ends
                    in a long stop somewhere else (UA021 on 16 July: 51 minutes at another stand).
  rest_at_spot_s    the same, but within PARKED_SPOT_M of the aircraft's own parked point (at t_before) instead of any
                    stand vertex. A push-back line runs along the stand's lead-in vertices, so the post-push-back stop
                    often lies within R_PARK_M of a vertex; it never lies at the parked point itself. This is the
                    feature the guard uses (research_300/tow_guard.py), rest_at_park_s is kept for comparison.
Geometry per airport as currently decided (dry_day_report.CURRENT); matching with the filed callsign added, a
superset of the validated rule, to see as many examples as possible. Nothing here is a test: the rule and its
constants go into PREREG_towguard.md and are tested on fresh dates. Summary in results/tow_guard_dev.json.
Run from the repository root:  python3 -u -B research_300/tow_guard_dev.py
"""
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

GEOMETRY = {"EDDM": "xplane_first", "EHAM": "xplane_first", "EDDF": "osm"}
ALL_STANDS = None
OUT = ROOT / ".cache/research_300/tow_guard_dev.parquet"


def dev_extras(ts, lat, lon, gs, ground, vertices, t_after, t_takeoff, stop_s):
    """rest_at_park_s (any stand vertex) and whether the longest stop lies within R_STAND_M of any stand or apron."""
    order = np.argsort(ts, kind="stable")
    ts, lat, lon, gs, ground = (np.asarray(a)[order] for a in (ts, lat, lon, gs, ground))
    keep = (ts > t_after) & (ts <= t_takeoff)
    ts, lat, lon, gs, ground = (a[keep] for a in (ts, lat, lon, gs, ground))
    if not len(ts):
        return 0.0, False
    d = SD._metres(lat, lon, vertices)
    park = (d <= SD.R_PARK_M) & np.isfinite(gs) & (gs < SD.REST_KT)
    rest_at_park = float(ts[park].max() - ts[park].min()) if park.sum() > 1 else 0.0
    out = np.flatnonzero(d > SD.R_STAND_M)
    longest, at_stand = 0.0, False
    if len(out):
        anchor = None
        for i in np.flatnonzero(ground[out[0]:]) + out[0]:
            if anchor is None or SD._metres(lat[i], lon[i], np.array([[lat[anchor], lon[anchor]]]))[0] > TG.STOP_RADIUS_M:
                anchor, first = i, ts[i]
            if ts[i] - first > longest:
                longest = float(ts[i] - first)
                at_stand = bool(SD._metres(lat[anchor], lon[anchor], ALL_STANDS)[0] <= SD.R_STAND_M)
    assert abs(longest - stop_s) < 1e-6, "dev_extras and tow_guard disagree on the longest stop"
    return rest_at_park, at_stand


def main():
    mapping = OB.callsign_mapping()
    filed = pd.concat(pd.read_parquet(p, columns=["MVT_ID_mvt", "CALLSIGN_flt"])
                      for p in sorted((ROOT / "data").glob("training_2025-0[17]-01_*.parquet")))
    maps = {ap: (SD.stand_table(ap), GC.xplane_table(ap)[0]) for ap in GEOMETRY}
    stands = OB.stand_vertices()
    rows = []
    for test in ("geometry", "matching"):
        for date in AT.TESTS[test]["dates"]:
            tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
            for ap in AT.TESTS[test]["airports"]:
                global ALL_STANDS
                ALL_STANDS = stands[ap]
                t = tracks[tracks.airport.astype(str).eq(ap)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
                dep = MD.departures(date, ap).merge(filed, on="MVT_ID_mvt", how="left")
                j = dep.merge(AT.match_candidates(dep, OB.visits(t), AT.options_for(dep, "plus_callsign_flt", mapping)),
                              on=["MVT_ID_mvt", "airport"], suffixes=("", "_v"))
                j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
                for row in j.itertuples(index=False):
                    v, _ = AT.lookup(GEOMETRY[ap], *maps[ap], row.STAND_mvt)
                    if v is None:
                        continue
                    ts, lat, lon, gs, ground = MD.points_of(t, row)
                    o = AT.guarded(ts, lat, lon, gs, v, row.mvt_epoch)
                    if o["status"] != "ok":
                        continue
                    g = TG.check(ts, lat, lon, gs, ground, v, o, row.mvt_epoch)
                    spot, longest = g["rest_at_spot_s"], g["longest_stop_s"]
                    rest, at_stand = dev_extras(ts, lat, lon, gs, ground, v, o["t_after"], row.mvt_epoch, longest)
                    taxi = row.mvt_epoch - (o["t_before"] + o["t_after"]) / 2
                    rows.append({"test": test, "date": date, "airport": ap, "MVT_ID_mvt": row.MVT_ID_mvt,
                                 "lane": row.lane, "callsign": row.callsign, "stand": row.STAND_mvt,
                                 "taxi_det": taxi, "taxi_true": row.mvt_epoch - row.block_epoch,
                                 "onsets": o["onsets"], "onset_spread_s": o["onset_spread_s"],
                                 "rest_at_park_s": rest, "rest_at_spot_s": spot, "longest_stop_s": longest,
                                 "longest_stop_at_stand": at_stand, "guard": g["status"]})
            print(test, date, "done", flush=True)
    frame = pd.DataFrame(rows)
    frame["err"] = frame.taxi_det - frame.taxi_true
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    bad = frame.err.abs() > 300
    print(f"covered {len(frame)}, |err| > 300 s: {int(bad.sum())}")
    cols = ["date", "airport", "callsign", "stand", "taxi_det", "taxi_true", "err", "onsets", "onset_spread_s",
            "rest_at_park_s", "rest_at_spot_s", "longest_stop_s", "longest_stop_at_stand"]
    print(frame[frame.err.abs() > 450].sort_values("err")[cols].round(0).to_string(index=False))
    print("covered per airport", frame.airport.value_counts().to_dict())
    for f in ("rest_at_park_s", "rest_at_spot_s", "longest_stop_s"):
        good = frame.loc[~bad, f]
        print(f, "good rows quantiles", good.quantile([.5, .9, .95, .99, .999, 1]).round(0).to_dict())
        for thr in (60, 120, 300, 600, 900, 1200, 1800):
            k = frame[f] >= thr
            print(f"   >= {thr:5d} s: withdraws {int(k.sum()):4d} rows, of which bad {int((k & bad).sum())}, "
                  f"good within 60 s {int((k & (frame.err.abs() <= 60)).sum())}, by airport {frame[k].airport.value_counts().to_dict()}")
    for thr in (300, 600):
        k = frame.rest_at_spot_s >= thr
        print(f"rest_at_spot >= {thr} s rows:")
        print(frame[k][cols].round(0).to_string(index=False))
    k = frame.longest_stop_s >= 1200
    print("longest_stop >= 1200 s rows:")
    print(frame[k][cols].round(0).to_string(index=False))
    fired = frame.guard.ne("ok")
    summary = {"covered": int(len(frame)), "covered_by_airport": frame.airport.value_counts().to_dict(),
               "constants": {"PARKED_SPOT_M": TG.PARKED_SPOT_M, "REST_AT_SPOT_S": TG.REST_AT_SPOT_S,
                             "STOP_RADIUS_M": TG.STOP_RADIUS_M, "LONG_STOP_S": TG.LONG_STOP_S},
               "withdrawn": frame[fired][["date", "airport", "callsign", "stand", "guard", "taxi_det", "taxi_true", "err",
                                          "rest_at_spot_s", "longest_stop_s"]].round(1).to_dict("records"),
               "withdrawn_within_60_s": int((fired & frame.err.abs().le(60)).sum()),
               "errors_over_2000_s_not_withdrawn": frame[~fired & frame.err.abs().gt(2000)][
                   ["date", "airport", "callsign", "stand", "err"]].round(1).to_dict("records")}
    (ROOT / "research_300/results/tow_guard_dev.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("GUARD at the pre-registered constants:", json.dumps({k: v for k, v in summary.items() if k != "withdrawn"}, default=str))
    print(frame[fired][cols + ["guard"]].round(0).to_string(index=False))


if __name__ == "__main__":
    main()

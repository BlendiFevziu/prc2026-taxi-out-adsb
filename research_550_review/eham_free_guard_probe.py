"""Does the Amsterdam detector's guard remove the error tail of stand-parked free matches? Exploratory, nine December
2025 days (labels read in research_290/transfer_2025.py). No candidate.

Two groups of EHAM main-lane departures, both through the deployed detector (arm_test.guarded, X-Plane-first stands)
and the tow guard (tow_guard.check): rows the detector's own flight_mvt matcher finds, and rows only the stand-parked
free match finds (unique visit ending within 300 s of MVT with a guarded onset at the assigned stand).
Run from the repository root:  python3 -u -B research_550_review/eham_free_guard_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_300 import arm_test as AT, geometry_compare as GC, tow_guard as TG  # noqa: E402
from research_290 import munich_dev as MD, stand_detector as SD  # noqa: E402

DAYS = ("2025-12-03", "2025-12-05", "2025-12-08", "2025-12-10", "2025-12-11", "2025-12-13", "2025-12-14", "2025-12-17", "2025-12-20")
OUT = ROOT / "research_550_review/results/eham_free_guard_probe.json"


def estimate(t, hexid, t_first, t_last, vertices, mvt):
    row = pd.Series({"hex": hexid, "t_first": t_first, "t_last": t_last})
    ts, lat, lon, gs, ground = MD.points_of(t, row)
    o = AT.guarded(ts, lat, lon, gs, vertices, mvt)
    if o.get("status") != "ok":
        return None
    g = TG.check(ts, lat, lon, gs, ground, vertices, o, mvt)
    return (o["t_before"] + o["t_after"]) / 2 if g["status"] == "ok" else None


def main():
    mapping = AT.OB.callsign_mapping()
    maps = (SD.stand_table("EHAM"), GC.xplane_table("EHAM")[0])
    rows = []
    for date in DAYS:
        t = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
        t = t[t.airport.astype(str).eq("EHAM")].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
        dep = MD.departures(date, "EHAM")
        dep = dep[dep.lane.eq("main")]
        visits = AT.OB.visits(t)
        j = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping)), on=["MVT_ID_mvt", "airport"],
                      how="inner", suffixes=("", "_v"))
        j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
        own = {r.MVT_ID_mvt: (r.hex, r.t_first, r.t_last) for r in j.itertuples(index=False)}
        claimed = {(h, f) for h, f, _ in own.values()}
        free = {}
        for r in dep.itertuples(index=False):
            vertices, _ = AT.lookup("xplane_first", *maps, r.STAND_mvt)
            if vertices is None:
                continue
            if r.MVT_ID_mvt in own:
                h, f, l = own[r.MVT_ID_mvt]
                est = estimate(t, h, f, l, vertices, r.mvt_epoch)
                if est is not None:
                    rows.append({"date": date, "group": "callsign", "err": est - r.block_epoch})
                continue
            cand = visits[(visits.t_last - r.mvt_epoch).abs().le(300)]
            kept = []
            for v in cand.itertuples(index=False):
                if (v.hex, v.t_first) in claimed:
                    continue
                est = estimate(t, v.hex, v.t_first, v.t_last, vertices, r.mvt_epoch)
                if est is not None:
                    kept.append((v.hex, v.t_first, est))
            if len(kept) == 1:
                free[r.MVT_ID_mvt] = (kept[0], r.block_epoch)
        keys = pd.Series([k[0][0] + str(k[0][1]) for k in free.values()]).value_counts()
        for mid, (k, block) in free.items():
            if keys[k[0] + str(k[1])] == 1:
                rows.append({"date": date, "group": "free_only", "err": k[2] - block})
        print(date, len(rows), flush=True)
    d = pd.DataFrame(rows)
    rep = {}
    for g, x in d.groupby("group"):
        e = x.err
        rep[g] = {"n": int(len(e)), "median": float(e.median()), "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                  "within60": float((e.abs() <= 60).mean()), "over600": int((e.abs() > 600).sum()),
                  "rmse_without_over600": float(np.sqrt((e[e.abs() <= 600] ** 2).mean()))}
    OUT.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

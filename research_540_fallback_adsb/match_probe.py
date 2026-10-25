"""Callsign-free matching for rows without AOBT_3: exploratory probe on the nine December 2025 days (their labels were
read in research_290/transfer_2025.py). No fit, no candidate.

For each departure, candidate visits are all ADS-B visits at the airport whose last point lies within WINDOW_S of
the recorded take-off; a candidate is kept when the frozen v2 detector (stand_detector.detect_onset_v2) finds a parked
onset at the row's ASSIGNED stand in it. Exactly one kept candidate is a match.
  precision  on main-lane rows that the callsign matcher also matches: share where both pick the same aircraft
  reach      fallback rows matched by callsign vs callsign-free, per airport
  accuracy   detector estimate minus the recorded block on matched fallback rows
Run from the repository root:  python3 -u -B research_540_fallback_adsb/match_probe.py
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

DATES = ("2025-12-03", "2025-12-05", "2025-12-08", "2025-12-10", "2025-12-11", "2025-12-13", "2025-12-14", "2025-12-17", "2025-12-20")
AIRPORTS = ("EHAM", "LEBL", "LEMD", "EDDF", "LSZH", "EGLL", "EDDM")
WINDOW_S = 300.0
OUT = ROOT / "research_540_fallback_adsb/results/match_probe.json"
CACHE = ROOT / ".cache/research_540"


def free_match(dep, t, visits, table):
    """MVT_ID -> (hex, t_first, estimate) for rows with exactly one visit parked at their assigned stand."""
    by_hex = {h: g for h, g in t.groupby("hex", sort=False)}
    out = {}
    for row in dep.itertuples(index=False):
        vertices, _ = SD.lookup(table, row.STAND_mvt)
        if vertices is None:
            continue
        cand = visits[(visits.t_last - row.mvt_epoch).abs().le(WINDOW_S)]
        kept = []
        for v in cand.itertuples(index=False):
            g = by_hex[v.hex]
            g = g[g.ts.between(v.t_first, v.t_last)]
            r = SD.detect_onset_v2(g.ts.to_numpy(float), g.lat.to_numpy(float), g.lon.to_numpy(float),
                                   pd.to_numeric(g.gs, errors="coerce").to_numpy(float), vertices, row.mvt_epoch, SD.R_PARK_M)
            if r.get("status") == "ok":
                kept.append((v.hex, v.t_first, (r["t_before"] + r["t_after"]) / 2))
        if len(kept) == 1:
            out[row.MVT_ID_mvt] = kept[0]
    return out


def main():
    mapping, rows = OB.callsign_mapping(), []
    for date in DATES:
        tracks = pd.read_parquet(ADSB / f"tracks/{date}.parquet")
        for ap in AIRPORTS:
            t = tracks[tracks.airport.astype(str).eq(ap)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
            dep = MD.departures(date, ap)
            if not len(t) or not len(dep):
                continue
            visits = OB.visits(t)
            cs = dep.merge(OB.match(dep, visits, mapping), on=["MVT_ID_mvt", "airport"], how="inner", suffixes=("", "_v"))
            cs = cs[cs.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)].set_index("MVT_ID_mvt")
            free = free_match(dep, t, visits, SD.stand_table(ap))
            claims = pd.Series([v[0] + str(v[1]) for v in free.values()]).value_counts()
            for r in dep.itertuples(index=False):
                f = free.get(r.MVT_ID_mvt)
                if f is not None and claims[f[0] + str(f[1])] > 1:
                    f = None                                   # the same visit claimed by two departures
                rows.append({"date": date, "airport": ap, "MVT_ID_mvt": r.MVT_ID_mvt, "lane": r.lane,
                             "block_epoch": r.block_epoch, "mvt_epoch": r.mvt_epoch,
                             "cs_hex": cs.hex.get(r.MVT_ID_mvt), "free_hex": f[0] if f else None,
                             "free_estimate": f[2] if f else np.nan})
        print(date, len(rows), flush=True)
    d = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE / "match_probe_rows.parquet", index=False)
    report = {}
    for ap, g in d.groupby("airport"):
        m = g[g.lane.eq("main") & g.cs_hex.notna() & g.free_hex.notna()]
        fb = g[g.lane.eq("fallback")]
        fm = fb[fb.free_hex.notna()]
        e = (fm.free_estimate - fm.block_epoch)
        report[ap] = {"main_both_matched": int(len(m)), "main_same_aircraft": float((m.cs_hex == m.free_hex).mean()) if len(m) else None,
                      "fallback_rows": int(len(fb)), "fallback_callsign_matched": int(fb.cs_hex.notna().sum()),
                      "fallback_free_matched": int(len(fm)),
                      "fallback_error": {"median": float(e.median()), "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                                         "within60": float((e.abs() <= 60).mean())} if len(fm) else None}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

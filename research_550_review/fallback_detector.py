"""Test of research_550_review/PREREG_fallback_detector.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_550_review/fallback_detector.py
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

N, REF = 344841, 252.76
ROWS_2026 = {"EHAM": 1206, "EDDM": 238}
TOW = {"EHAM": True, "EDDM": False}
PAIRS = ("1_7", "2_8", "3_9", "4_10", "5_11", "6_12")
DAYS = sorted(p.stem for p in (MD.ADSB / "tracks").glob("2025-*.parquet") if "staging" not in p.stem)
OUT = ROOT / "research_550_review/results/fallback_detector.json"
CACHE = ROOT / ".cache/research_550/fallback_detector_rows.parquet"


def onset(t, hexid, t_first, t_last, vertices, mvt, tow):
    ts, lat, lon, gs, ground = MD.points_of(t, pd.Series({"hex": hexid, "t_first": t_first, "t_last": t_last}))
    o = AT.guarded(ts, lat, lon, gs, vertices, mvt)
    if o.get("status") != "ok":
        return None
    if tow:
        o = TG.check(ts, lat, lon, gs, ground, vertices, o, mvt)
        if o["status"] != "ok":
            return None
    return (o["t_before"] + o["t_after"]) / 2


def day_rows(date, ap, mapping, maps):
    tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
    t = tracks[tracks.airport.astype(str).eq(ap)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    if not len(t):
        return []
    dep = MD.departures(date, ap)
    filed = pd.read_parquet(Path(MD.month_file(date)), columns=["MVT_ID_mvt", "CALLSIGN_flt"])
    dep = dep.merge(filed, on="MVT_ID_mvt", how="left")
    visits = AT.OB.visits(t)
    if not len(visits):
        return []
    allm = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "plus_callsign_flt", mapping)), on=["MVT_ID_mvt", "airport"],
                     how="inner", suffixes=("", "_v"))
    claimed_all = set(zip(allm.hex, allm.t_first))
    own = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping)), on=["MVT_ID_mvt", "airport"],
                    how="inner", suffixes=("", "_v"))
    own = own[own.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)].set_index("MVT_ID_mvt")
    out = []
    for r in dep[dep.lane.eq("fallback")].itertuples(index=False):
        vertices, _ = AT.lookup("xplane_first", *maps, r.STAND_mvt)
        if vertices is None:
            continue
        est, how = None, None
        if r.MVT_ID_mvt in own.index:
            v = own.loc[r.MVT_ID_mvt]
            est, how = onset(t, v.hex, v.t_first, v.t_last, vertices, r.mvt_epoch, TOW[ap]), "callsign"
        else:
            kept = []
            for v in visits[(visits.t_last - r.mvt_epoch).abs().le(300)].itertuples(index=False):
                if (v.hex, v.t_first) in claimed_all:
                    continue
                e = onset(t, v.hex, v.t_first, v.t_last, vertices, r.mvt_epoch, TOW[ap])
                if e is not None:
                    kept.append(e)
            if len(kept) == 1:
                est, how = kept[0], "free"
        out.append({"MVT_ID_mvt": r.MVT_ID_mvt, "airport": ap, "date": date, "how": how,
                    "detector": max(r.mvt_epoch - est, 300.0) if est is not None else np.nan})
    return out


def main():
    mapping = AT.OB.callsign_mapping()
    rows = []
    for ap in ("EHAM", "EDDM"):
        maps = (SD.stand_table(ap), GC.xplane_table(ap)[0])
        for d in DAYS:
            rows += day_rows(d, ap, mapping, maps)
        print(ap, len(rows), flush=True)
    d = pd.DataFrame(rows)
    pred = pd.concat([pd.read_parquet(ROOT / f".cache/research_240/fallback_anchor_production/predictions_{p}.parquet",
                                      columns=["MVT_ID_mvt", "y", "prod_anchor"]) for p in PAIRS]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    d["y"], d["ref"] = d.MVT_ID_mvt.map(pred.y), d.MVT_ID_mvt.map(pred.prod_anchor)
    if d.ref.isna().any():
        raise SystemExit(f"{int(d.ref.isna().sum())} fallback rows without a shipped value")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE, index=False)
    c = d[d.detector.notna()].reset_index(drop=True)
    y = c.y.to_numpy(float)
    g = (c.ref.to_numpy(float) - y) ** 2 - (c.detector.to_numpy(float) - y) ** 2
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-20:]] = False
    c["block"] = c.airport + "/" + c.date.str[5:7]
    blocks = {b: float(g[c.block.eq(b).to_numpy()].mean()) for b in sorted(c.block.unique())}
    per_ap = {}
    mse = 0.0
    for ap in ("EHAM", "EDDM"):
        share = float(d.loc[d.airport.eq(ap), "detector"].notna().mean())
        gp = float(g[c.airport.eq(ap).to_numpy()].mean()) if c.airport.eq(ap).any() else 0.0
        per_ap[ap] = {"fallback_rows": int(d.airport.eq(ap).sum()), "covered": int(c.airport.eq(ap).sum()), "share": share,
                      "gain_per_covered_row": gp,
                      "detector_rmse": float(np.sqrt(np.mean((c.detector[c.airport.eq(ap)] - c.y[c.airport.eq(ap)]) ** 2))) if c.airport.eq(ap).any() else None,
                      "shipped_rmse": float(np.sqrt(np.mean((c.ref[c.airport.eq(ap)] - c.y[c.airport.eq(ap)]) ** 2))) if c.airport.eq(ap).any() else None}
        mse += gp * share * ROWS_2026[ap] / N
    r = {"prereg": "research_550_review/PREREG_fallback_detector.md", "days": len(DAYS), "covered_rows": int(len(c)),
         "by_match": c.how.value_counts().to_dict(), "gain_mse": float(g.mean()), "without_best20": float(g[keep].mean()),
         "by_month": {m: float(g[c.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07", "12") if c.date.str[5:7].eq(m).any()},
         "blocks": blocks, "airports": per_ap}
    r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    r["rules"] = {"pooled_positive": r["gain_mse"] > 0, "without_best20_positive": r["without_best20"] > 0,
                  "january_and_july": r["by_month"].get("01", -1) > 0 and r["by_month"].get("07", -1) > 0,
                  "four_of_six_blocks": sum(v > 0 for v in blocks.values()) >= 4, "planning_at_least_0_1_s": r["planning_s"] >= 0.1}
    r["verdict"] = "pass" if all(r["rules"].values()) else "fail"
    OUT.write_text(json.dumps(r, indent=2, default=str) + "\n")
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()

"""Test of research_550_review/PREREG_eham_free_match.md on the twenty fresh days. No fit, no upload.
Run from the repository root:  python3 -u -B research_550_review/eham_free_match.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_480_weights", ROOT / "research_550_review"):
    sys.path.insert(0, str(p))
from research_320 import stand_experiment as S  # noqa: E402
import weights as W  # noqa: E402
from research_300 import arm_test as AT, geometry_compare as GC  # noqa: E402
from research_290 import munich_dev as MD, stand_detector as SD  # noqa: E402
import eham_free_guard_probe as P  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, N, REF, ROWS_2026 = "MVT_ID_mvt", 344841, 252.76, 5600
BLOCKS = (("fresh_july", ("2025-07-01", "2025-07-05", "2025-07-19", "2025-07-20", "2025-07-24", "2025-07-28", "2025-07-30")),
          ("eddf_july_confirm", ("2025-07-06", "2025-07-08", "2025-07-10")),
          ("fresh_january", W.JAN10))
OUT = ROOT / "research_550_review/results/eham_free_match.json"


def free_only(date, mapping, maps):
    """MVT_ID -> detector taxi for Amsterdam main rows only the free match finds (guarded)."""
    t = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
    t = t[t.airport.astype(str).eq("EHAM")].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    dep = MD.departures(date, "EHAM")
    dep = dep[dep.lane.eq("main")]
    visits = AT.OB.visits(t)
    j = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping)), on=["MVT_ID_mvt", "airport"],
                  how="inner", suffixes=("", "_v"))
    j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
    covered = set(j.MVT_ID_mvt)
    claimed = set(zip(j.hex, j.t_first))
    free = {}
    for r in dep[~dep.MVT_ID_mvt.isin(covered)].itertuples(index=False):
        vertices, _ = AT.lookup("xplane_first", *maps, r.STAND_mvt)
        if vertices is None:
            continue
        kept = []
        for v in visits[(visits.t_last - r.mvt_epoch).abs().le(300)].itertuples(index=False):
            if (v.hex, v.t_first) in claimed:
                continue
            est = P.estimate(t, v.hex, v.t_first, v.t_last, vertices, r.mvt_epoch)
            if est is not None:
                kept.append((v.hex, v.t_first, r.mvt_epoch - est))
        if len(kept) == 1:
            free[r.MVT_ID_mvt] = kept[0]
    keys = pd.Series([f"{h}|{f}" for h, f, _ in free.values()]).value_counts()
    return {m: taxi for m, (h, f, taxi) in free.items() if keys[f"{h}|{f}"] == 1}


def policy(f, days):
    """Current policy value for Amsterdam rows of the extraction frame f."""
    b = f.comparator.to_numpy(float)
    val = b.copy()
    july = f.date.str[5:7].eq("07").to_numpy()
    part = f.category.eq("partial_ground").to_numpy()
    stand = f.category.eq("observed_at_assigned_stand").to_numpy()
    plus = [joblib.load(ROOT / f".cache/research_550/c1_more/c1plus_{s}.pkl") for s in L.SEEDS]
    k2p = [joblib.load(ROOT / f".cache/research_550/k2_more/k2plus_{s}.pkl") for s in L.SEEDS]
    m = part & july
    if m.any():
        x = L.inputs("partial", f[m])
        val[m] = np.maximum(b[m] + np.mean([mm.predict(x) for mm in plus], axis=0), 300.0)
    m = part & ~july
    if m.any():
        g = f[m].merge(W.eham_features(days), on=ID, how="left", validate="one_to_one")
        if g[list(L.EHAM_TRACK)].isna().all(axis=1).any():
            raise SystemExit("January partial rows without add-on features")
        c = (L.correction("eham_january", g) + L.correction("partial", g)) / 2
        val[m] = np.maximum(b[m] + c, 300.0)
    if stand.any():
        x = S.inputs(f[stand], "real")
        val[stand] = np.maximum(b[stand] + np.mean([mm.predict(x) for mm in k2p], axis=0), 300.0)
    return val


def main():
    mapping = AT.OB.callsign_mapping()
    maps = (SD.stand_table("EHAM"), GC.xplane_table("EHAM")[0])
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    rows = []
    for tag, days in BLOCKS:
        ext = S.extract(days, tag)
        ext = ext[ext.airport.eq("EHAM") & ext.date.isin(days)].reset_index(drop=True)
        for d in days:
            det = free_only(d, mapping, maps)
            f = ext[ext.date.eq(d) & ext[ID].isin(det)].reset_index(drop=True)
            if not len(f):
                continue
            if not np.array_equal(base.loc[f[ID]].to_numpy(float), f.comparator.to_numpy(float)):
                raise SystemExit("comparator not bit-identical")
            f["policy"] = policy(f, days)
            f["detector"] = np.maximum(f[ID].map(det).to_numpy(float), 300.0)
            rows.append(f[[ID, "date", "category", "y", "comparator", "policy", "detector"]])
            print(d, len(f), flush=True)
    d = pd.concat(rows, ignore_index=True)
    y = d.y.to_numpy(float)
    g = (d.policy.to_numpy(float) - y) ** 2 - (d.detector.to_numpy(float) - y) ** 2
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-100:]] = False
    by_day = {x: float(g[d.date.eq(x).to_numpy()].mean()) for x in sorted(d.date.unique())}
    r = {"prereg": "research_550_review/PREREG_eham_free_match.md", "rows": int(len(d)),
         "by_category": {c: {"rows": int(n), "gain": float(g[d.category.eq(c).to_numpy()].mean())} for c, n in d.category.value_counts().items()},
         "policy_rmse": float(np.sqrt(np.mean((d.policy - y) ** 2))), "detector_rmse": float(np.sqrt(np.mean((d.detector - y) ** 2))),
         "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
         "by_month": {m: float(g[d.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")}, "by_day": by_day}
    mse = ROWS_2026 * r["gain_mse"] / N
    r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    r["rules"] = {"pooled_positive": r["gain_mse"] > 0, "without_best100_positive": r["without_best100"] > 0,
                  "both_months": min(r["by_month"].values()) > 0,
                  "seventy_percent_of_days": bool(np.mean([v > 0 for v in by_day.values()]) >= 0.7),
                  "planning_at_least_0_1_s": r["planning_s"] >= 0.1}
    r["verdict"] = "pass" if all(r["rules"].values()) else "fail"
    OUT.write_text(json.dumps(r, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "by_day"}, indent=1, default=str))


if __name__ == "__main__":
    main()

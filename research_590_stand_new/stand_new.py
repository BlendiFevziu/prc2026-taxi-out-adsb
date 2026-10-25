"""Test of research_590_stand_new/PREREG_stand_new.md. No fit, no upload.
Run from the repository root:  python3 -u -B research_590_stand_new/stand_new.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_590_stand_new"):
    sys.path.insert(0, str(p))
import extract as E  # noqa: E402
import cells_confirm as CC  # noqa: E402
from research_320 import stand_experiment as S  # noqa: E402
from research_320 import volume_capacity as V  # noqa: E402
from research_300 import arm_test as AT, geometry_compare as GC  # noqa: E402
from research_290 import munich_dev as MD, stand_detector as SD  # noqa: E402
import harness as H  # noqa: E402

ID, N, REF = "MVT_ID_mvt", 344841, 249.3908
K2P = [joblib.load(ROOT / f".cache/research_550/k2_more/k2plus_{s}.pkl") for s in S.SEEDS]
TESTED = {"EDDM/01": "EDDM", "LEMD/07": "LEMD"}
SISTER = {"EDDM/07": "EDDM/01", "LEMD/01": "LEMD/07"}


def munich_detector_rows(days):
    """Rows the deployed Munich detector estimates (flight_mvt matcher, X-Plane-first stands, guarded onset)."""
    mapping, maps, out = AT.OB.callsign_mapping(), (SD.stand_table("EDDM"), GC.xplane_table("EDDM")[0]), set()
    for d in days:
        t = pd.read_parquet(MD.ADSB / f"tracks/{d}.parquet")
        t = t[t.airport.astype(str).eq("EDDM")].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
        dep = MD.departures(d, "EDDM")
        dep = dep[dep.lane.eq("main")]
        visits = AT.OB.visits(t)
        j = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping)), on=[ID, "airport"], how="inner",
                      suffixes=("", "_v"))
        j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
        for r in j.itertuples(index=False):
            v, _ = AT.lookup("xplane_first", *maps, r.STAND_mvt)
            if v is None:
                continue
            ts, lat, lon, gs, _ = MD.points_of(t, r)
            if AT.guarded(ts, lat, lon, gs, v, r.mvt_epoch)["status"] == "ok":
                out.add(r.MVT_ID_mvt)
    return out


def frame_2025(airport, days, comps):
    rows = E.load(airport, days)
    rows = rows[rows.category.eq("observed_at_assigned_stand")]
    f = rows.merge(comps[[ID, "y", "comparator", "ersatz_s", "schedule", "gate", "MVT_TIME_UTC_mvt"]], on=ID, how="left", validate="one_to_one")
    if f.y.isna().any():
        raise SystemExit(f"{airport}: stand rows without a held-out component row")
    epoch = (pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True) - S.R.MD.EPOCH0).dt.total_seconds()
    phase = 2 * np.pi * (epoch % 86400) / 86400
    f["hour_sin"], f["hour_cos"] = np.sin(phase), np.cos(phase)
    for ap in CC.FLAGS:
        f[f"airport_{ap}"] = 0.0
    f["is_july"] = f.date.str[5:7].eq("07").astype(float)
    return f.reset_index(drop=True)


def rows_2026(airport):
    dep = E.departures_2026(airport)
    f = E.load(airport, sorted(dep.date.unique()))
    return f[f.category.eq("observed_at_assigned_stand")].assign(month=lambda x: x.date.str[5:7])


def drift_ok(reference, target):
    stats = V.P.drift(reference, target)
    flags = [k for k, v in stats.items() if v["diagnostic_flag"]]
    return flags


def main():
    comps = H.components_2025()
    comps["comparator"] = __import__("layers").compose_base(comps)
    report, admitted = {"prereg": "research_590_stand_new/PREREG_stand_new.md", "cells": {}}, {}
    own = H.ownership_v62().set_index(ID).owner
    for cell, ap in TESTED.items():
        spec = CC.CELLS[cell]
        f = frame_2025(ap, spec["days"], comps)
        if ap == "EDDM":
            det = munich_detector_rows(spec["days"])
            f = f[~f[ID].isin(det)].reset_index(drop=True)
        y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
        x = S.inputs(f, "real")
        c = {s: m.predict(x) for s, m in zip(S.SEEDS, K2P)}
        se = lambda corr: (np.maximum(b + corr, 300.0) - y) ** 2
        g = (b - y) ** 2 - se(np.mean(list(c.values()), axis=0))
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-100:]] = False
        r26 = rows_2026(ap)
        base_owned = r26[r26.month.eq(cell[-2:]) & own.reindex(r26[ID]).eq("base").to_numpy()]
        flags = drift_ok(f, base_owned)
        r = {"rows": int(len(f)), "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
             "blocks": {k: float(g[f.date.isin(v).to_numpy()].mean()) for k, v in spec["blocks"].items()},
             "seeds": {s: float(((b - y) ** 2 - se(c[s])).mean()) for s in S.SEEDS}, "drift_flags": flags,
             "rows_2026_base_owned": int(len(base_owned))}
        r["rules"] = {"pooled": r["gain_mse"] > 0, "without_best100": r["without_best100"] > 0, "both_blocks": min(r["blocks"].values()) > 0,
                      "every_seed": all(v > 0 for v in r["seeds"].values()), "no_drift": not flags}
        r["pass"] = all(r["rules"].values())
        report["cells"][cell] = r
        if r["pass"]:
            admitted[cell] = (r["gain_mse"], len(base_owned))
            sister = [k for k, v in SISTER.items() if v == cell][0]
            s_rows = r26[r26.month.eq(sister[-2:]) & own.reindex(r26[ID]).eq("base").to_numpy()]
            s_flags = drift_ok(f, s_rows)
            report["cells"][sister] = {"sister_of": cell, "drift_flags": s_flags, "rows_2026_base_owned": int(len(s_rows)), "pass": not s_flags}
            if not s_flags:
                admitted[sister] = (r["gain_mse"], len(s_rows))
    mse = sum(g * n for g, n in admitted.values()) / N
    report["admitted"] = sorted(admitted)
    report["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else 0.0
    report["verdict"] = "pass" if admitted and report["planning_s"] >= 0.05 else "fail"
    Path(ROOT / "research_590_stand_new/results").mkdir(parents=True, exist_ok=True)
    (ROOT / "research_590_stand_new/results/stand_new.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

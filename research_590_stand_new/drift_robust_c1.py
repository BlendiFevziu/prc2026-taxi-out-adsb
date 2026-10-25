"""Test of research_590_stand_new/PREREG_drift_robust_c1.md. Fits the drift-robust C1 once (saved). No upload.
Run from the repository root:  python3 -u -B research_590_stand_new/drift_robust_c1.py
"""
import glob
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_590_stand_new", ROOT / "research_550_review"):
    sys.path.insert(0, str(p))
import stand_new as T  # noqa: E402
import extract as E  # noqa: E402
import cells_confirm as CC  # noqa: E402
import c1_more_data as C1M  # noqa: E402
import layers as L  # noqa: E402

S, V, H = T.S, T.V, T.H
MS = S.MS
ID, N, REF = "MVT_ID_mvt", 344841, 248.22
REMOVED = ["ground_points", "first_still_age_s", "last_still_age_s", "last_moving_age_s", "last_ground_age_s", "speed_p90",
           "sampled_distance_m", "stand_last_m", "stand_median_m", "last_near_stand_age_s", "last_still_near_stand_age_s"]
KEPT = [f for f in MS.FEATURES if f not in REMOVED]
JAN10 = ("2025-01-03", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-15", "2025-01-17", "2025-01-22", "2025-01-27", "2025-01-29", "2025-01-31")
MODELS = ROOT / ".cache/research_590/drift_robust_c1/drc1_{seed}.pkl"
OUT = ROOT / "research_590_stand_new/results/drift_robust_c1.json"


def columns(frame):
    return [c for c in MS.inputs(frame.head(5), "real").columns if c not in REMOVED]


def models(train, cols):
    paths = [Path(str(MODELS).format(seed=s)) for s in L.SEEDS]
    if not all(p.exists() for p in paths):
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        x, y = MS.inputs(train, "real")[cols], (train.y - train.comparator).to_numpy(float)
        with V.heavy_work("research_590: fit the drift-robust C1", poll_s=10.0):
            for s, p in zip(L.SEEDS, paths):
                m = S.R.lgb.LGBMRegressor(objective="regression", n_estimators=800, learning_rate=.03, num_leaves=31, min_child_samples=80,
                                          reg_lambda=100., subsample=.8, subsample_freq=1, colsample_bytree=.9, max_bin=63, n_jobs=4,
                                          random_state=s, verbosity=-1).fit(x, y)
                joblib.dump(m, p)
    return [joblib.load(p) for p in paths]


def judge(f, corr_by_seed, reference, blocks):
    y = f.y.to_numpy(float)
    b = f.comparator.to_numpy(float)
    se = lambda corr: (np.maximum(b + corr, 300.0) - y) ** 2
    g = (reference - y) ** 2 - se(np.mean(list(corr_by_seed.values()), axis=0))
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-100:]] = False
    r = {"rows": int(len(f)), "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
         "blocks": {k: float(g[f.date.isin(v).to_numpy()].mean()) for k, v in blocks.items()},
         "seeds": {s: float(((reference - y) ** 2 - se(c)).mean()) for s, c in corr_by_seed.items()}}
    r["rules"] = {"pooled": r["gain_mse"] > 0, "without_best100": r["without_best100"] > 0, "both_blocks": min(r["blocks"].values()) > 0,
                  "every_seed": all(v > 0 for v in r["seeds"].values())}
    r["pass"] = all(r["rules"].values())
    return r


def flags(reference, target):
    st = V.P.drift(reference, target)
    return [k for k, v in st.items() if v["diagnostic_flag"] and k in KEPT]


def main():
    train = C1M.training()
    cols = columns(train)
    ms = models(train, cols)
    comps = H.components_2025()
    comps["comparator"] = L.compose_base(comps)
    base = pd.Series(comps.comparator.to_numpy(float), index=comps[ID].to_numpy())
    own = H.ownership_v62().set_index(ID).owner
    ctx = pd.read_parquet(ROOT / ".cache/codex_v56/partial_release/context.parquet")
    report = {"prereg": "research_590_stand_new/PREREG_drift_robust_c1.md", "inputs": cols, "cells": {}}
    admitted = {}
    corr = lambda f: {s: m.predict(MS.inputs(f, "real")[cols]) for s, m in zip(L.SEEDS, ms)}
    # 1. EGLL/01 tested, EGLL/07 target
    fj = S.extract(JAN10, "fresh_january")
    eg = fj[fj.airport.eq("EGLL") & fj.category.eq("partial_ground") & fj.date.isin(JAN10)].reset_index(drop=True)
    if not np.array_equal(base.loc[eg[ID]].to_numpy(float), eg.comparator.to_numpy(float)):
        raise SystemExit("EGLL comparator not bit-identical")
    r = judge(eg, corr(eg), eg.comparator.to_numpy(float), {"first_half": JAN10[:5], "second_half": JAN10[5:]})
    tgt = ctx[ctx.airport.eq("EGLL") & ctx.date.str[5:7].eq("07") & ctx.category.eq("partial_ground")]
    tgt = tgt[own.reindex(tgt[ID]).eq("base").to_numpy()]
    fl = flags(eg, tgt)
    report["cells"]["EGLL/07"] = {"test_EGLL/01": r, "drift_flags_kept": fl, "rows_2026": int(len(tgt)), "admitted": r["pass"] and not fl}
    if r["pass"] and not fl:
        admitted["EGLL/07"] = r["gain_mse"] * len(tgt)
    # 2./3. LEMD/07 tested
    le = CC.frame_new_airport("LEMD", CC.CELLS["LEMD/07"]["days"], comps)
    blocks = CC.CELLS["LEMD/07"]["blocks"]
    c_dr = corr(le)
    r_base = judge(le, c_dr, le.comparator.to_numpy(float), blocks)
    c1 = np.maximum(le.comparator.to_numpy(float) + L.correction("partial", le), 300.0)
    r_c1 = judge(le, c_dr, c1, blocks)
    r26 = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / ".cache/research_590/days/LEMD_2026-*.parquet")))], ignore_index=True)
    r26 = r26[r26.category.eq("partial_ground") & own.reindex(r26[ID]).eq("base").to_numpy()]
    for m, key in (("01", "LEMD/01"), ("07", "LEMD/07")):
        t = r26[r26.date.str[5:7].eq(m)]
        fl = flags(le, t)
        if key == "LEMD/01":
            ok = r_base["pass"] and not fl
            report["cells"][key] = {"test_LEMD/07_vs_base": r_base, "drift_flags_kept": fl, "rows_2026": int(len(t)), "admitted": ok}
            if ok:
                admitted[key] = r_base["gain_mse"] * len(t)
        else:
            ok = r_base["pass"] and r_c1["gain_mse"] > 0 and r_c1["without_best100"] > 0 and not fl
            report["cells"][key] = {"test_vs_c1": r_c1, "drift_flags_kept": fl, "rows_2026": int(len(t)), "admitted": ok}
            if ok:
                admitted[key] = r_c1["gain_mse"] * len(t)
    mse = sum(admitted.values()) / N
    report["admitted"] = sorted(admitted)
    report["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else 0.0
    report["verdict"] = "pass" if admitted and report["planning_s"] >= 0.1 else "fail"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

"""Test of research_590_stand_new/PREREG_drift_robust.md. Fits the drift-robust stand model once (saved). No upload.
Run from the repository root:  python3 -u -B research_590_stand_new/drift_robust.py
"""
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
import k2_more_data as K2M  # noqa: E402
import cells_confirm as CC  # noqa: E402
import layers as L  # noqa: E402

S, V, H = T.S, T.V, T.H
ID, N, REF = "MVT_ID_mvt", 344841, 249.3908
REMOVED = ["ground_points", "last_ground_age_s", "last_still_age_s", "last_moving_age_s", "speed_p90", "sampled_distance_m",
           "stand_last_m", "stand_median_m", "near_stand_fraction"]
KEPT_SUMMARIES = [f for f in S.MS.FEATURES if f not in REMOVED]
JAN10 = ("2025-01-03", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-15", "2025-01-17", "2025-01-22", "2025-01-27", "2025-01-29", "2025-01-31")
MODELS = ROOT / ".cache/research_590/drift_robust/dr_{seed}.pkl"
OUT = ROOT / "research_590_stand_new/results/drift_robust.json"


def cols():
    full = list(S.inputs(K2M.training().head(5), "real").columns)
    return [c for c in full if c not in REMOVED]


def models(train, columns):
    paths = [Path(str(MODELS).format(seed=s)) for s in S.SEEDS]
    if not all(p.exists() for p in paths):
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        x, y = S.inputs(train, "real")[columns], (train.y - train.comparator).to_numpy(float)
        with V.heavy_work("research_590: fit the drift-robust stand model", poll_s=10.0):
            for s, p in zip(S.SEEDS, paths):
                m = S.R.lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=.03, num_leaves=15, min_child_samples=80,
                                          reg_lambda=100., subsample=.8, subsample_freq=1, colsample_bytree=.9, max_bin=63, n_jobs=4,
                                          random_state=s, verbosity=-1).fit(x, y)
                joblib.dump(m, p)
    return [joblib.load(p) for p in paths]


def judge(f, ms, columns, blocks):
    y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
    x = S.inputs(f, "real")[columns]
    c = {s: m.predict(x) for s, m in zip(S.SEEDS, ms)}
    se = lambda corr: (np.maximum(b + corr, 300.0) - y) ** 2
    g = (b - y) ** 2 - se(np.mean(list(c.values()), axis=0))
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-100:]] = False
    r = {"rows": int(len(f)), "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
         "blocks": {k: float(g[f.date.isin(v).to_numpy()].mean()) for k, v in blocks.items()},
         "seeds": {s: float(((b - y) ** 2 - se(c[s])).mean()) for s in S.SEEDS}}
    r["rules"] = {"pooled": r["gain_mse"] > 0, "without_best100": r["without_best100"] > 0, "both_blocks": min(r["blocks"].values()) > 0,
                  "every_seed": all(v > 0 for v in r["seeds"].values())}
    r["pass"] = all(r["rules"].values())
    return r


def flags(reference, target):
    st = V.P.drift(reference, target)
    return [k for k, v in st.items() if v["diagnostic_flag"] and k in KEPT_SUMMARIES]


def main():
    train = K2M.training()
    columns = cols()
    ms = models(train, columns)
    comps = H.components_2025()
    comps["comparator"] = L.compose_base(comps)
    own = H.ownership_v62().set_index(ID).owner
    ctx = pd.read_parquet(ROOT / ".cache/codex_v56/partial_release/context.parquet")
    report = {"prereg": "research_590_stand_new/PREREG_drift_robust.md", "inputs": columns, "cells": {}}
    admitted = {}
    # LEMD/07 tested, LEMD/01 as sister
    lemd = T.frame_2025("LEMD", CC.CELLS["LEMD/07"]["days"], comps)
    r = judge(lemd, ms, columns, CC.CELLS["LEMD/07"]["blocks"])
    r26 = T.rows_2026("LEMD")
    for cell in ("LEMD/07", "LEMD/01"):
        tgt = r26[r26.month.eq(cell[-2:]) & own.reindex(r26[ID]).eq("base").to_numpy()]
        fl = flags(lemd, tgt)
        report["cells"][cell] = {**({"test": r} if cell == "LEMD/07" else {"sister_of": "LEMD/07"}), "drift_flags_kept": fl, "rows_2026": int(len(tgt)),
                                 "admitted": r["pass"] and not fl}
        if r["pass"] and not fl:
            admitted[cell] = (r["gain_mse"], len(tgt))
    # EGLL/01 tested on the fresh January days, EGLL/07 as sister
    fj = S.extract(JAN10, "fresh_january")
    eg = fj[fj.airport.eq("EGLL") & fj.category.eq("observed_at_assigned_stand") & fj.date.isin(JAN10)].reset_index(drop=True)
    base = pd.Series(comps.comparator.to_numpy(float), index=comps[ID].to_numpy())
    if not np.array_equal(base.loc[eg[ID]].to_numpy(float), eg.comparator.to_numpy(float)):
        raise SystemExit("EGLL comparator not bit-identical")
    r = judge(eg, ms, columns, {"first_half": JAN10[:5], "second_half": JAN10[5:]})
    tgt = ctx[ctx.airport.eq("EGLL") & ctx.date.str[5:7].eq("07") & ctx.category.eq("observed_at_assigned_stand")]
    tgt = tgt[own.reindex(tgt[ID]).eq("base").to_numpy()]
    fl = flags(eg, tgt)
    report["cells"]["EGLL/01"] = {"test": r}
    report["cells"]["EGLL/07"] = {"sister_of": "EGLL/01", "drift_flags_kept": fl, "rows_2026": int(len(tgt)), "admitted": r["pass"] and not fl}
    if r["pass"] and not fl:
        admitted["EGLL/07"] = (r["gain_mse"], len(tgt))
    mse = sum(g * n for g, n in admitted.values()) / N
    report["admitted"] = sorted(admitted)
    report["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else 0.0
    report["verdict"] = "pass" if admitted and report["planning_s"] >= 0.1 else "fail"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

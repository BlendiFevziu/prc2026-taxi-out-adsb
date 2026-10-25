"""Test of research_640_tail_rule/PREREG_catboost_more.md. Fits CatBoost per model (saved). No upload.
Run from the repository root:  python3 -u -B research_640_tail_rule/catboost_more.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_590_stand_new",
          ROOT / "research_550_review", ROOT / "research_610_more_data", ROOT / "research_640_tail_rule"):
    sys.path.insert(0, str(p))
import drift_robust as DRS  # noqa: E402
import drift_robust_c1 as DRC  # noqa: E402
import catboost_layers as CL  # noqa: E402
import k2_more_data as K2M  # noqa: E402
import c1_more_data as C1M  # noqa: E402

S, MS, L, H, T, CC, V = DRS.S, DRC.MS, DRS.L, DRS.H, DRS.T, DRS.CC, DRS.V
ID, N, REF = "MVT_ID_mvt", 344841, 246.16
JAN10 = DRS.JAN10
CACHE = ROOT / ".cache/research_640"
OUT = ROOT / "research_640_tail_rule/results/catboost_more.json"


def cat_models(name, x, y):
    paths = [CACHE / f"cat_{name}_{s}.pkl" for s in L.SEEDS]
    if not all(p.exists() for p in paths):
        CACHE.mkdir(parents=True, exist_ok=True)
        from catboost import CatBoostRegressor
        with V.heavy_work(f"research_640: CatBoost {name}", poll_s=10.0):
            for s, p in zip(L.SEEDS, paths):
                joblib.dump(CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=10, loss_function="RMSE",
                                              random_seed=s, thread_count=4, verbose=False).fit(x, y), p)
    return [joblib.load(p) for p in paths]


def judge(f, x, ref_models, cat_models_, blocks):
    y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
    pr = {s: m.predict(x) for s, m in zip(L.SEEDS, ref_models)}
    pc = {s: m.predict(x) for s, m in zip(L.SEEDS, cat_models_)}
    se = lambda k: (np.maximum(b + k, 300.0) - y) ** 2
    mr, mc = np.mean(list(pr.values()), axis=0), np.mean(list(pc.values()), axis=0)
    g = se(mr) - se((mr + mc) / 2)
    ci, days = CL.day_bootstrap(g, (f.airport + "|" + f.date).to_numpy())
    r = {"rows": int(len(f)), "gain_mse": float(g.mean()), "bootstrap_95": ci, "airport_days": days,
         "blocks": {k: float(g[f.date.isin(v).to_numpy()].mean()) for k, v in blocks.items()},
         "seeds": {s: float((se(pr[s]) - se((pr[s] + pc[s]) / 2)).mean()) for s in L.SEEDS}}
    r["rules"] = {"pooled": r["gain_mse"] > 0, "bootstrap_lower": ci[0] > 0, "both_blocks": min(r["blocks"].values()) > 0,
                  "every_seed": all(v > 0 for v in r["seeds"].values())}
    r["pass"] = all(r["rules"].values())
    return r


def main():
    comps = H.components_2025()
    comps["comparator"] = L.compose_base(comps)
    report = {"prereg": "research_640_tail_rule/PREREG_catboost_more.md", "A": {}, "B": {}}
    fj = S.extract(JAN10, "fresh_january")
    halves_jan = {"first_half": JAN10[:5], "second_half": JAN10[5:]}
    # A: drift-robust stand
    st_cols = DRS.cols()
    st_train = K2M.training()
    dr_st = DRS.models(None, st_cols)
    cat_st = cat_models("dr_stand", S.inputs(st_train, "real")[st_cols], (st_train.y - st_train.comparator).to_numpy(float))
    lemd_st = T.frame_2025("LEMD", CC.CELLS["LEMD/07"]["days"], comps)
    report["A"]["stand LEMD/07"] = judge(lemd_st, S.inputs(lemd_st, "real")[st_cols], dr_st, cat_st, CC.CELLS["LEMD/07"]["blocks"])
    eg_st = fj[fj.airport.eq("EGLL") & fj.category.eq("observed_at_assigned_stand") & fj.date.isin(JAN10)].reset_index(drop=True)
    report["A"]["stand EGLL/01"] = judge(eg_st, S.inputs(eg_st, "real")[st_cols], dr_st, cat_st, halves_jan)
    # A: drift-robust C1
    c1_train = C1M.training()
    c1_cols = json.loads((ROOT / "research_590_stand_new/results/drift_robust_c1.json").read_text())["inputs"]
    dr_c1 = DRC.models(None, c1_cols)
    cat_c1 = cat_models("dr_c1", MS.inputs(c1_train, "real")[c1_cols], (c1_train.y - c1_train.comparator).to_numpy(float))
    eg_p = fj[fj.airport.eq("EGLL") & fj.category.eq("partial_ground") & fj.date.isin(JAN10)].reset_index(drop=True)
    report["A"]["c1 EGLL/01"] = judge(eg_p, MS.inputs(eg_p, "real")[c1_cols], dr_c1, cat_c1, halves_jan)
    lemd_p = CC.frame_new_airport("LEMD", CC.CELLS["LEMD/07"]["days"], comps)
    report["A"]["c1 LEMD/07"] = judge(lemd_p, MS.inputs(lemd_p, "real")[c1_cols], dr_c1, cat_c1, CC.CELLS["LEMD/07"]["blocks"])
    # B: old C1 (trained on E) with CatBoost on E
    e_train = pd.concat([S.extract(tuple(V.S_DATES), "train"), S.extract(V.E_EXTRA, "volume_extra")], ignore_index=True)
    e_train = e_train[e_train.category.eq("partial_ground")].drop_duplicates(ID)
    old_c1 = L.load_models("partial")
    cat_e = cat_models("c1_E", MS.inputs(e_train, "real"), (e_train.y - e_train.comparator).to_numpy(float))
    eddm = CC.frame_new_airport("EDDM", CC.CELLS["EDDM/01"]["days"], comps)
    report["B"]["EDDM/01"] = judge(eddm, MS.inputs(eddm, "real"), old_c1, cat_e, CC.CELLS["EDDM/01"]["blocks"])
    report["B"]["LEMD/07"] = judge(lemd_p, MS.inputs(lemd_p, "real"), old_c1, cat_e, CC.CELLS["LEMD/07"]["blocks"])
    rows = {"A": {"stand LEMD/07": {"LEMD/07 stand": 2812, "LEMD/01 stand": 2176}, "stand EGLL/01": {"EGLL/07 stand": 10242},
                  "c1 EGLL/01": {"EGLL/07 partial": 9048}, "c1 LEMD/07": {"LEMD/01 partial": 12478}},
            "B": {"EDDM/01": {"EDDM/01": 6802, "EDDM/07": 9539}, "LEMD/07": {"LEMD/07 C1": 8692}}}
    admitted, mse = {}, 0.0
    for dec in ("A", "B"):
        for k, r in report[dec].items():
            if r["pass"]:
                for tgt, n in rows[dec][k].items():
                    admitted[tgt] = n
                    mse += r["gain_mse"] * n / N
            print(dec, k, json.dumps({kk: r[kk] for kk in ("rows", "gain_mse", "bootstrap_95", "blocks", "pass")}), flush=True)
    report["admitted"] = admitted
    report["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else 0.0
    report["verdict"] = "pass" if admitted and report["planning_s"] >= 0.05 else "fail"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"admitted": admitted, "planning_s": report["planning_s"], "verdict": report["verdict"]}))


if __name__ == "__main__":
    main()

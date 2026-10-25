"""Test of research_570_recorded/PREREG_recorded.md. No upload.
Run from the repository root:  python3 -u -B research_570_recorded/recorded.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_510_rome_classifier"))
sys.path.insert(0, str(ROOT / "research_500_midnight"))
import rome_classifier as C  # noqa: E402  (loads main_model before lightgbm)
import rule as R  # noqa: E402

lgb = C.lgb
N, REF = 344841, 249.3908
OUT = ROOT / "research_570_recorded/results/recorded.json"


def late_rome(files, labelled=True):
    cols = R.COLS + (["TAXITIME_SEC_mvt"] if labelled else []) + ["FLIGHT_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt"]
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in files], ignore_index=True)
    d = R.annotate(d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF") & d.AOBT_3_flt.isna()]).reset_index(drop=True)
    return d[d.delay > 3600].reset_index(drop=True)


def fit_predict(x_tr, y_tr, x_te):
    x_tr, x_te = x_tr.copy(), x_te.copy()
    for c in C.CATS:
        levels = sorted(x_tr[c].unique())
        x_tr[c] = pd.Categorical(x_tr[c], categories=levels)
        x_te[c] = pd.Categorical(x_te[c], categories=levels)
    ps = []
    for s in C.SEEDS:
        m = lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=20,
                               reg_lambda=10., colsample_bytree=.9, subsample=.8, subsample_freq=1, random_state=s, n_jobs=4, verbosity=-1)
        m.fit(x_tr[C.NUM + C.CATS], y_tr, categorical_feature=C.CATS)
        ps.append(m.predict_proba(x_te[C.NUM + C.CATS])[:, 1])
    return np.mean(ps, axis=0)


def main():
    d = late_rome(sorted((ROOT / "data").glob("training_2025-*.parquet")))
    d["kind"] = R.label_kind(d.TAXITIME_SEC_mvt, d.delay)
    d["recorded"] = d.kind.ne("copy").astype(int)
    d["month"] = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True).dt.month
    shipped = pd.concat([pd.read_parquet(C.PREDS / f"predictions_{p}.parquet", columns=[R.ID, "prod_anchor"])
                         for p in ("1_7", "2_8", "3_9", "4_10", "5_11", "6_12")]).set_index(R.ID).prod_anchor
    d["shipped"] = d[R.ID].map(shipped).astype(float)
    target = (d.crossing >= 1) & (d.delay > 8 * 3600)
    x = C.features(d)
    d["p"], d["expected"] = np.nan, np.nan
    for m in sorted(d.loc[target, "month"].unique()):
        tr, te = d.month.ne(m).to_numpy(), (d.month.eq(m) & target).to_numpy()
        p = fit_predict(x[tr], d.recorded[tr].to_numpy(), x[te])
        t = float(d.loc[tr & d.kind.eq("normal").to_numpy(), "TAXITIME_SEC_mvt"].median())
        d.loc[te, "p"] = p
        d.loc[te, "expected"] = p * (86400.0 + t) + (1 - p) * d.delay.to_numpy(float)[te]
    f = d[target].reset_index(drop=True)
    y, ship = f.TAXITIME_SEC_mvt.to_numpy(float), f.shipped.to_numpy(float)
    from sklearn.metrics import roc_auc_score
    sub = f.kind.isin(["copy", "shift24"])
    auc = float(roc_auc_score(f.kind[sub].eq("shift24"), f.p[sub])) if f.kind[sub].nunique() == 2 else None
    report = {"prereg": "research_570_recorded/PREREG_recorded.md", "late_rows": int(len(d)), "target_rows": int(len(f)),
              "target_kinds": f.kind.value_counts().to_dict(), "auc_shift_vs_copy": auc,
              "p_by_kind": f.groupby("kind").p.mean().to_dict()}
    for name, arm in (("A", (ship + f.expected.to_numpy(float)) / 2), ("B", f.expected.to_numpy(float))):
        g = (ship - y) ** 2 - (arm - y) ** 2
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-3:]] = False
        by_month = {int(m): float(g[f.month.eq(m).to_numpy()].sum()) for m in sorted(f.month.unique())}
        report[f"arm_{name}"] = {"gain_sse": float(g.sum()), "gain_per_row": float(g.mean()), "without_best3": float(g[keep].sum()),
                                 "by_month": by_month, "months_positive": sum(v > 0 for v in by_month.values())}
    k = late_rome([ROOT / "data/ranking.parquet"], labelled=False)
    rows26 = int(((k.crossing >= 1) & (k.delay > 8 * 3600)).sum())
    a = report["arm_A"]
    mse = rows26 * a["gain_per_row"] / N
    report["rows_2026"] = rows26
    report["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    report["rules"] = {"pooled_positive": a["gain_sse"] > 0, "without_best3_positive": a["without_best3"] > 0,
                       "auc_at_least_0_65": auc is not None and auc >= 0.65,
                       "two_thirds_of_months": a["months_positive"] >= 2 * len(a["by_month"]) / 3,
                       "planning_at_least_0_1_s": report["planning_s"] >= 0.1}
    report["verdict"] = "pass" if all(report["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

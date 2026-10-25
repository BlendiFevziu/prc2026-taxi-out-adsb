"""Leave-one-month-out test of research_660_rome_arrivals/PREREG_rome_arrivals.md. No upload.
Run from the repository root:  python3 -u -B research_660_rome_arrivals/rome_arrivals.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_500_midnight", ROOT / "research_510_rome_classifier", ROOT / "research_640_tail_rule"):
    sys.path.insert(0, str(p))
import rome_classifier as RC  # noqa: E402  (imports main_model before lightgbm)
import rule as R  # noqa: E402
import catboost_layers as CL  # noqa: E402

lgb = RC.lgb
N, REF, ROWS_2026 = 344841, 245.020, 326
NEW = ["arr_copy_al_day", "n_arr_al_day", "arr_copy_day", "in_copy"]
OUT = ROOT / "research_660_rome_arrivals/results/rome_arrivals.json"


def arrivals(frame):
    a = frame[frame.PHASE_mvt.eq("ARR") & frame.ADES_mvt.eq("LIRF")].copy()
    blk, sched = pd.to_datetime(a.BLOCK_TIME_UTC_mvt, utc=True), pd.to_datetime(a.SCHED_TIME_UTC_mvt, utc=True)
    a["is_copy"] = ((blk - sched).dt.total_seconds().abs() <= 60).astype(float)
    a["date"] = R.local_day(a.SCHED_TIME_UTC_mvt)
    a["al"] = a.FLIGHT_mvt.astype(str).str[:3]
    a["mvt"] = pd.to_datetime(a.MVT_TIME_UTC_mvt, utc=True)
    return a


def arrival_inputs(dep, arr):
    """The four registered inputs for departure rows `dep`, from arrival rows `arr` (no departure label used)."""
    f = pd.DataFrame(index=dep.index)
    date, al = R.local_day(dep.SCHED_TIME_UTC_mvt), dep.FLIGHT_mvt.astype(str).str[:3]
    g = arr.groupby(["al", "date"]).is_copy.agg(["mean", "size"])
    key = pd.MultiIndex.from_arrays([al, date])
    f["arr_copy_al_day"] = g["mean"].reindex(key).to_numpy()
    f["n_arr_al_day"] = g["size"].reindex(key).fillna(0).to_numpy()
    f["arr_copy_day"] = arr.groupby("date").is_copy.mean().reindex(date).to_numpy()
    left = pd.DataFrame({"pos": np.arange(len(dep)), "stand": dep.STAND_mvt.astype(str).to_numpy(),
                         "t": pd.to_datetime(dep.SCHED_TIME_UTC_mvt, utc=True).to_numpy()}).sort_values("t")
    right = pd.DataFrame({"stand": arr.STAND_mvt.astype(str).to_numpy(), "t": arr.mvt.to_numpy(), "in_copy": arr.is_copy.to_numpy()}).sort_values("t")
    m = pd.merge_asof(left, right, on="t", by="stand", direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta(hours=18))
    f["in_copy"] = m.sort_values("pos").in_copy.to_numpy()
    return f


def fit_predict(train_x, train_k, test_x, num):
    x_tr, x_te = train_x.copy(), test_x.copy()
    for c in RC.CATS:
        levels = sorted(x_tr[c].unique())
        x_tr[c] = pd.Categorical(x_tr[c], categories=levels)
        x_te[c] = pd.Categorical(x_te[c], categories=levels)
    y = np.array([RC.CLASSES.index(k) for k in train_k])
    probs = []
    for s in RC.SEEDS:
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=200, learning_rate=.03, num_leaves=7,
                               min_child_samples=20, reg_lambda=10., colsample_bytree=.9, subsample=.8, subsample_freq=1,
                               random_state=s, n_jobs=4, verbosity=-1)
        m.fit(x_tr[num + RC.CATS], y, categorical_feature=RC.CATS)
        p = np.zeros((len(x_te), 3))
        p[:, m.classes_] = m.predict_proba(x_te[num + RC.CATS])
        probs.append(p)
    return np.mean(probs, axis=0)


def main():
    cols = R.COLS + [R.T, "FLIGHT_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "BLOCK_TIME_UTC_mvt"]
    raw = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted((ROOT / "data").glob("training_2025-*.parquet"))], ignore_index=True)
    arr = arrivals(raw)
    d = R.annotate(raw[raw.PHASE_mvt.eq("DEP") & raw.ADEP_mvt.eq("LIRF") & raw.AOBT_3_flt.isna()]).reset_index(drop=True)
    d = d[RC.cell(d)].reset_index(drop=True)
    d["kind"] = R.label_kind(d[R.T], d.delay)
    d["month"] = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True).dt.month
    shipped = pd.concat([pd.read_parquet(RC.PREDS / f"predictions_{p}.parquet", columns=[R.ID, "prod_anchor"])
                         for p in ("1_7", "2_8", "3_9", "4_10", "5_11", "6_12")]).set_index(R.ID).prod_anchor
    d["shipped"] = d[R.ID].map(shipped).astype(float)
    if d.shipped.isna().any():
        raise SystemExit("cell rows without a shipped value")
    x = pd.concat([RC.features(d), arrival_inputs(d, arr)], axis=1)
    probs = {"old": np.zeros((len(d), 3)), "new": np.zeros((len(d), 3))}
    ev = {"old": np.zeros(len(d)), "new": np.zeros(len(d))}
    for m in sorted(d.month.unique()):
        tr, te = d.month.ne(m).to_numpy(), d.month.eq(m).to_numpy()
        t = float(d.loc[tr & d.kind.eq("normal").to_numpy(), R.T].median())
        for name, num in (("old", RC.NUM), ("new", RC.NUM + NEW)):
            probs[name][te] = fit_predict(x[tr], d.kind[tr], x[te], num)
            ev[name][te] = RC.expected(probs[name][te], d.delay.to_numpy(float)[te], t)
    sl = (d.delay <= 8 * 3600).to_numpy()
    y, ship = d[R.T].to_numpy(float)[sl], d.shipped.to_numpy(float)[sl]
    ref, cand = (ship + ev["old"][sl]) / 2, (ship + ev["new"][sl]) / 2
    g = (ref - y) ** 2 - (cand - y) ** 2
    ds = d[sl].reset_index(drop=True)
    from sklearn.metrics import roc_auc_score
    is_copy = d.kind.eq("copy").to_numpy()
    ci, days = CL.day_bootstrap(g, R.local_day(ds.SCHED_TIME_UTC_mvt).astype(str).to_numpy())
    report = {"prereg": "research_660_rome_arrivals/PREREG_rome_arrivals.md", "cell_rows": int(len(d)), "slice_rows": int(sl.sum()),
              "auc_copy": {k: float(roc_auc_score(is_copy, probs[k][:, 0])) for k in probs},
              "auc_copy_slice": {k: float(roc_auc_score(is_copy[sl], probs[k][sl, 0])) for k in probs},
              "input_coverage": {c: float(x[c].notna().mean()) for c in NEW},
              "gain_sse": float(g.sum()), "gain_per_row": float(g.mean()), "bootstrap_95_per_row": ci, "days": days,
              "by_month": {int(m): float(g[ds.month.eq(m).to_numpy()].sum()) for m in sorted(ds.month.unique())},
              "by_kind": {k: float(g[ds.kind.eq(k).to_numpy()].sum()) for k in sorted(ds.kind.unique())},
              "rmse_slice": {"v70": float(np.sqrt(np.mean((ref - y) ** 2))), "candidate": float(np.sqrt(np.mean((cand - y) ** 2)))}}
    mse = ROWS_2026 * report["gain_per_row"] / N
    report["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    rk = pd.read_parquet(ROOT / "data/ranking.parquet", columns=cols[:-1] + ["BLOCK_TIME_UTC_mvt"] if False else [c for c in cols if c != R.T])
    report["arrival_copy_share"] = {"2025": float(arr.is_copy.mean()), "2026": float(arrivals(rk).is_copy.mean())}
    bm = report["by_month"]
    report["rules"] = {"pooled": report["gain_sse"] > 0, "bootstrap_lower_above_zero": ci[0] > 0,
                       "eight_of_twelve_months": sum(v > 0 for v in bm.values()) >= 8, "january_and_july": bm.get(1, 0) > 0 and bm.get(7, 0) > 0,
                       "planning_at_least_0_1_s": report["planning_s"] >= 0.1}
    report["verdict"] = "pass" if all(report["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

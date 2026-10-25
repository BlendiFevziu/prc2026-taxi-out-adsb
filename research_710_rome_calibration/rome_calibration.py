"""Test of research_710_rome_calibration/PREREG_rome_calibration.md. No upload.
Run from the repository root:  python3 -u -B research_710_rome_calibration/rome_calibration.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_500_midnight", ROOT / "research_510_rome_classifier", ROOT / "research_640_tail_rule"):
    sys.path.insert(0, str(p))
import rome_classifier as C  # noqa: E402  (main_model first)
import rule as R  # noqa: E402
from catboost_layers import day_bootstrap  # noqa: E402
ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 245.046
EXTRA = ["FLIGHT_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt"]
KINDS = ("copy", "normal", "shift24")
BANDS = [1, 2, 4, 8, 14, 1000]
OUT = ROOT / "research_710_rome_calibration/results/rome_calibration.json"


def load(files):
    d = pd.concat([pd.read_parquet(p, columns=R.COLS + [T] + EXTRA) for p in files], ignore_index=True)
    d = R.annotate(d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF") & d.AOBT_3_flt.isna()].drop_duplicates(ID))
    d = d[d.delay > 3600].reset_index(drop=True)
    d["band"] = pd.cut(d.delay / 3600, BANDS).astype(str)
    d["cross"] = d.crossing.clip(0, 1).astype(int)
    return d


def fit_predict(train_x, train_k, test_x):
    """rome_classifier.fit_predict, but robust to a training fold without one kind (LightGBM then still returns three
    columns while classes_ lists two)."""
    import lightgbm as lgb
    x_tr, x_te = train_x.copy(), test_x.copy()
    for c in C.CATS:
        levels = sorted(x_tr[c].unique())
        x_tr[c] = pd.Categorical(x_tr[c], categories=levels)
        x_te[c] = pd.Categorical(x_te[c], categories=levels)
    y = np.array([C.CLASSES.index(k) for k in train_k])
    probs = []
    for s in C.SEEDS:
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=200, learning_rate=.03, num_leaves=7,
                               min_child_samples=20, reg_lambda=10., colsample_bytree=.9, subsample=.8, subsample_freq=1,
                               random_state=s, n_jobs=4, verbosity=-1)
        m.fit(x_tr[C.NUM + C.CATS], y, categorical_feature=C.CATS)
        raw = m.predict_proba(x_te[C.NUM + C.CATS])
        p = np.zeros((len(x_te), 3))
        if raw.shape[1] == 3:
            p[:] = raw
        else:
            p[:, m.classes_] = raw
        probs.append(p)
    return np.mean(probs, axis=0)


def cell_shares(train, test):
    def shares(g):
        return np.array([(g.kind == k).mean() for k in KINDS])
    out = np.zeros((len(test), 3))
    for i, (b, c) in enumerate(zip(test.band, test.cross)):
        g = train[(train.band == b) & (train.cross == c)]
        out[i] = shares(g if len(g) else train[train.band == b])
    return out


def values(delay, t):
    return np.c_[delay, np.full(len(delay), t), np.full(len(delay), 86400.0 + t)]


def main():
    d = load(sorted((ROOT / "data").glob("training_2025-*.parquet")))
    d["kind"] = R.label_kind(d[T], d.delay)
    d["month"] = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True).dt.month
    d["day"] = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True).dt.tz_convert(R.TZ).dt.date.astype(str)
    anchor = pd.concat([pd.read_parquet(ROOT / f".cache/research_240/fallback_anchor_production/predictions_{p}.parquet", columns=[ID, "prod_anchor"])
                        for p in ("1_7", "2_8", "3_9", "4_10", "5_11", "6_12")]).drop_duplicates(ID).set_index(ID).prod_anchor
    d["anchor"] = d[ID].map(anchor).astype(float)
    if d.anchor.isna().any():
        raise SystemExit(f"{int(d.anchor.isna().sum())} rows without prod_anchor")
    rule_rows = (d.delay > R.MIN_DELAY_S) & (d.crossing == 0)
    cell_rows = (d.delay <= 8 * 3600) & ~rule_rows
    x = C.features(d)
    ref = d.anchor.to_numpy(float).copy()
    arm = {k: np.zeros(len(d)) for k in "ABC"}
    for m in sorted(d.month.unique()):
        tr, te = d.month.ne(m).to_numpy(), d.month.eq(m).to_numpy()
        t = float(d.loc[tr & d.kind.eq("normal").to_numpy(), T].median())
        v = values(d.delay.to_numpy(float)[te], t)
        # reference: v69 rule and v70 half average on cell rows (classifier fitted on the other months' cell rows)
        trc, tec = tr & cell_rows.to_numpy(), te & cell_rows.to_numpy()
        if tec.any():
            pc = fit_predict(x[trc], d.kind[trc], x[tec])
            ref[tec] = (d.anchor.to_numpy(float)[tec] + C.expected(pc, d.delay.to_numpy(float)[tec], t)) / 2
        ref[te & rule_rows.to_numpy()] = d.delay.to_numpy(float)[te & rule_rows.to_numpy()]
        qa = fit_predict(x[tr], d.kind[tr], x[te])
        qb = cell_shares(d[tr], d[te])
        arm["A"][te] = (qa * v).sum(1)
        arm["B"][te] = (qb * v).sum(1)
    arm["C"] = (arm["A"] + arm["B"]) / 2
    y = d[T].to_numpy(float)
    report = {"prereg": "research_710_rome_calibration/PREREG_rome_calibration.md", "rows": int(len(d)),
              "kinds": {k: int(v) for k, v in d.kind.value_counts().items()},
              "rmse": {"reference": float(np.sqrt(np.mean((ref - y) ** 2)))}, "arms": {}}
    for k, a in arm.items():
        g = (ref - y) ** 2 - (a - y) ** 2
        report["rmse"][k] = float(np.sqrt(np.mean((a - y) ** 2)))
        report["arms"][k] = {"gain_per_row": float(g.mean()), "gain_sse": float(g.sum()),
                             "by_month": {int(m): float(g[d.month.eq(m).to_numpy()].sum()) for m in sorted(d.month.unique())},
                             "by_cell": {f"{b}/{c}": float(g[((d.band == b) & (d.cross == c)).to_numpy()].sum())
                                         for b, c in sorted(set(zip(d.band, d.cross)))}}
    best = max(report["arms"], key=lambda k: report["arms"][k]["gain_sse"])
    g = (ref - y) ** 2 - (arm[best] - y) ** 2
    ci, days = day_bootstrap(g, d.day.to_numpy())
    te26 = load([ROOT / "data/ranking.parquet"])
    mse = len(te26) * float(g.mean()) / N
    planning = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    bm = report["arms"][best]["by_month"]
    report["selected"] = best
    report["bootstrap_95"], report["days"], report["rows_2026"], report["planning_s"] = ci, days, int(len(te26)), planning
    report["rules"] = {"pooled_positive": float(g.sum()) > 0, "bootstrap_lower_above_zero": ci[0] > 0,
                       "eight_of_twelve_and_jan_jul": sum(v > 0 for v in bm.values()) >= 8 and bm[1] > 0 and bm[7] > 0,
                       "planning_at_least_0_1_s": planning >= 0.1}
    report["verdict"] = "pass" if all(report["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

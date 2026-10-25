"""Leave-one-month-out test of research_510_rome_classifier/PREREG_rome_classifier.md. No upload.
Run from the repository root:  python3 -u -B research_510_rome_classifier/rome_classifier.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_500_midnight"))
import main_model  # noqa: F401,E402  (torch before tree libraries on macOS)
import lightgbm as lgb  # noqa: E402
import rule as R  # noqa: E402

N, REF, ROWS_2026 = 344841, 255.32, 338
SEEDS = (177, 277, 377)
CLASSES = ("copy", "normal", "shift24")
CATS = ["airline", "country", "stand_area", "runway"]
NUM = ["log_delay", "crossing", "sched_h", "mvt_h", "weekday"]
PREDS = ROOT / ".cache/research_240/fallback_anchor_production"
OUT = ROOT / "research_510_rome_classifier/results/rome_classifier.json"


def features(frame):
    f = pd.DataFrame(index=frame.index)
    sched = pd.to_datetime(frame.SCHED_TIME_UTC_mvt, utc=True).dt.tz_convert(R.TZ)
    mvt = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.tz_convert(R.TZ)
    f["log_delay"] = np.log(frame.delay.to_numpy(float))
    f["crossing"] = frame.crossing.to_numpy(float)
    f["sched_h"], f["mvt_h"], f["weekday"] = sched.dt.hour, mvt.dt.hour, sched.dt.weekday
    f["airline"] = frame.FLIGHT_mvt.astype(str).str[:3]
    f["country"] = frame.ADES_mvt.astype(str).str[:2]
    f["stand_area"] = frame.STAND_mvt.astype(str).str[:1]
    f["runway"] = frame.RUNWAY_mvt.astype(str)
    return f


def cell(frame):
    rule_rows = (frame.delay > R.MIN_DELAY_S) & (frame.crossing == 0)
    return (frame.PHASE_mvt.eq("DEP") & frame.ADEP_mvt.eq("LIRF") & frame.AOBT_3_flt.isna() & (frame.delay > 3600) & ~rule_rows).to_numpy()


def fit_predict(train_x, train_k, test_x):
    x_tr, x_te = train_x.copy(), test_x.copy()
    for c in CATS:
        levels = sorted(x_tr[c].unique())
        x_tr[c] = pd.Categorical(x_tr[c], categories=levels)
        x_te[c] = pd.Categorical(x_te[c], categories=levels)          # unseen levels become missing
    y = np.array([CLASSES.index(k) for k in train_k])
    probs = []
    for s in SEEDS:
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=200, learning_rate=.03, num_leaves=7,
                               min_child_samples=20, reg_lambda=10., colsample_bytree=.9, subsample=.8, subsample_freq=1,
                               random_state=s, n_jobs=4, verbosity=-1)
        m.fit(x_tr[NUM + CATS], y, categorical_feature=CATS)
        p = np.zeros((len(x_te), 3))
        p[:, m.classes_] = m.predict_proba(x_te[NUM + CATS])
        probs.append(p)
    return np.mean(probs, axis=0)


def expected(p, delay, t):
    return p[:, 0] * delay + p[:, 1] * t + p[:, 2] * (86400.0 + t)


def main():
    d = pd.concat([pd.read_parquet(p, columns=R.COLS + [R.T, "FLIGHT_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt"])
                   for p in sorted((ROOT / "data").glob("training_2025-*.parquet"))], ignore_index=True)
    d = R.annotate(d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF") & d.AOBT_3_flt.isna()]).reset_index(drop=True)
    d = d[cell(d)].reset_index(drop=True)
    d["kind"] = R.label_kind(d[R.T], d.delay)
    d["month"] = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True).dt.month
    shipped = pd.concat([pd.read_parquet(PREDS / f"predictions_{p}.parquet", columns=[R.ID, "prod_anchor"])
                         for p in ("1_7", "2_8", "3_9", "4_10", "5_11", "6_12")]).set_index(R.ID).prod_anchor
    d["shipped"] = d[R.ID].map(shipped).astype(float)
    if d.shipped.isna().any():
        raise SystemExit(f"{int(d.shipped.isna().sum())} cell rows without a shipped value")
    x = features(d)
    probs = np.zeros((len(d), 3))
    for m in sorted(d.month.unique()):
        tr, te = d.month.ne(m).to_numpy(), d.month.eq(m).to_numpy()
        probs[te] = fit_predict(x[tr], d.kind[tr], x[te])
        t = float(d.loc[tr & d.kind.eq("normal").to_numpy(), R.T].median())
        d.loc[te, "expected"] = expected(probs[te], d.delay.to_numpy(float)[te], t)
    y, ship = d[R.T].to_numpy(float), d.shipped.to_numpy(float)
    arms = {"A": (ship + d.expected.to_numpy(float)) / 2, "B": d.expected.to_numpy(float)}
    from sklearn.metrics import log_loss, roc_auc_score
    is_copy = d.kind.eq("copy").to_numpy()
    report = {"prereg": "research_510_rome_classifier/PREREG_rome_classifier.md", "rows": int(len(d)),
              "kinds": {k: int(v) for k, v in d.kind.value_counts().items()},
              "classifier": {"log_loss": float(log_loss([CLASSES.index(k) for k in d.kind], probs, labels=[0, 1, 2])),
                             "auc_copy": float(roc_auc_score(is_copy, probs[:, 0])),
                             "auc_copy_shipped_commitment": float(roc_auc_score(is_copy, ship / d.delay.to_numpy(float)))},
              "shipped_rmse": float(np.sqrt(np.mean((ship - y) ** 2)))}
    band = pd.cut(d.delay / 3600, [1, 2, 4, 8, 40]).astype(str)
    for name, arm in arms.items():
        g = (ship - y) ** 2 - (arm - y) ** 2
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-20:]] = False
        r = {"rmse": float(np.sqrt(np.mean((arm - y) ** 2))), "gain_sse": float(g.sum()), "gain_per_row": float(g.mean()),
             "without_best20": float(g[keep].sum()),
             "by_month": {int(m): float(g[d.month.eq(m).to_numpy()].sum()) for m in sorted(d.month.unique())},
             "by_crossing": {int(c): float(g[d.crossing.eq(c).to_numpy()].sum()) for c in sorted(d.crossing.unique())},
             "by_band": {b: float(g[(band == b).to_numpy()].sum()) for b in sorted(band.unique())}}
        mse = ROWS_2026 * r["gain_per_row"] / N
        r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
        report[f"arm_{name}"] = r
    a = report["arm_A"]
    report["rules"] = {"pooled_positive": a["gain_sse"] > 0, "without_best20_positive": a["without_best20"] > 0,
                       "eight_of_twelve_months": sum(v > 0 for v in a["by_month"].values()) >= 8,
                       "january_and_july": a["by_month"][1] > 0 and a["by_month"][7] > 0,
                       "planning_at_least_0_1_s": a["planning_s"] >= 0.1}
    report["verdict"] = "pass" if all(report["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

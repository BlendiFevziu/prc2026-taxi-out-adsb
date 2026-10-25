"""Test of research_520_rome_main/PREREG_rome_main.md on the held-out January/July 2025 Rome main rows. No upload.
Run from the repository root:  python3 -u -B research_520_rome_main/rome_main.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_360_integration"))
sys.path.insert(0, str(ROOT / "research_500_midnight"))
import main_model  # noqa: F401,E402  (torch before tree libraries on macOS)
import lightgbm as lgb  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402
import rule as R  # noqa: E402

N, REF, ROWS_2026 = 344841, 253.57, 26516
SEEDS = (177, 277, 377)
TIMES = ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]
CATS = ["airline", "AIRCRAFT_OPERATOR_flt", "country", "STAND_mvt", "stand_area", "RUNWAY_mvt", "AIRCRAFT_TYPE_flt",
        "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt"]
NUM = ["delay", "offblock_delay", "ersatz", "eobt_delay", "iobt_delay", "lobt_delay", "crossing", "sched_h", "mvt_h", "weekday"]
RAW = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt",
       "AIRCRAFT_TYPE_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt"] + TIMES[2:] + TIMES[:2]
OUT = ROOT / "research_520_rome_main/results/rome_main.json"


def rome_main(files, labelled=True):
    d = pd.concat([pd.read_parquet(p, columns=RAW + (["TAXITIME_SEC_mvt"] if labelled else [])) for p in files], ignore_index=True)
    d = d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF") & d.AOBT_3_flt.notna()].copy()
    return R.annotate(d).reset_index(drop=True)


def features(d):
    f = pd.DataFrame(index=d.index)
    t = {c: pd.to_datetime(d[c], utc=True) for c in TIMES}
    sec = lambda a, b: (t[a] - t[b]).dt.total_seconds()
    f["delay"], f["crossing"] = d.delay.to_numpy(float), d.crossing.to_numpy(float)
    f["offblock_delay"], f["ersatz"] = sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt"), sec("MVT_TIME_UTC_mvt", "AOBT_3_flt")
    f["eobt_delay"], f["iobt_delay"], f["lobt_delay"] = (sec(c, "SCHED_TIME_UTC_mvt") for c in ("EOBT_1_flt", "IOBT_flt", "LOBT_flt"))
    loc_s, loc_m = t["SCHED_TIME_UTC_mvt"].dt.tz_convert(R.TZ), t["MVT_TIME_UTC_mvt"].dt.tz_convert(R.TZ)
    f["sched_h"], f["mvt_h"], f["weekday"] = loc_s.dt.hour, loc_m.dt.hour, loc_s.dt.weekday
    f["airline"], f["country"] = d.FLIGHT_mvt.astype(str).str[:3], d.ADES_mvt.astype(str).str[:2]
    f["stand_area"] = d.STAND_mvt.astype(str).str[:1]
    for c in ("AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt"):
        f[c] = d[c].astype(str)
    return f


def fit_predict(x_tr, y_tr, x_te):
    x_tr, x_te = x_tr.copy(), x_te.copy()
    for c in CATS:
        levels = sorted(x_tr[c].unique())
        x_tr[c] = pd.Categorical(x_tr[c], categories=levels)
        x_te[c] = pd.Categorical(x_te[c], categories=levels)
    ps = []
    for s in SEEDS:
        m = lgb.LGBMClassifier(objective="binary", n_estimators=400, learning_rate=.03, num_leaves=31, min_child_samples=50,
                               reg_lambda=10., colsample_bytree=.9, subsample=.8, subsample_freq=1, random_state=s, n_jobs=4,
                               verbosity=-1)
        m.fit(x_tr[NUM + CATS], y_tr, categorical_feature=CATS)
        ps.append(m.predict_proba(x_te[NUM + CATS])[:, 1])
    return np.mean(ps, axis=0)


def arms(base, delay, p):
    mix = np.maximum(p * delay + (1 - p) * base, 300.0)
    return {"A": (base + mix) / 2, "B": mix}


def main():
    files = sorted((ROOT / "data").glob("training_2025-*.parquet"))
    d = rome_main(files)
    d["is_copy"] = (d.TAXITIME_SEC_mvt - d.delay).abs() <= 60
    month = pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True).dt.month
    comps = H.components_2025()
    comps = comps[comps.airport.eq("LIRF")]
    base = pd.Series(L.compose_base(comps), index=comps[H.ID].to_numpy())
    test_ids = set(comps[H.ID])
    train = d[~d[H.ID].isin(test_ids) & ~month.isin([1, 7])].reset_index(drop=True)
    test = d[d[H.ID].isin(test_ids)].reset_index(drop=True)
    if len(test) != len(comps):
        raise SystemExit(f"test rows {len(test)} vs held-out components {len(comps)}")
    p = fit_predict(features(train), train.is_copy.to_numpy(int), features(test))
    y = comps.set_index(H.ID).loc[test[H.ID], "y"].to_numpy(float)
    if not np.allclose(y, test.TAXITIME_SEC_mvt.to_numpy(float)):
        raise SystemExit("labels disagree between the components and the training files")
    b = base.loc[test[H.ID]].to_numpy(float)
    from sklearn.metrics import roc_auc_score
    gate = comps.set_index(H.ID).loc[test[H.ID], "gate"].to_numpy(float)
    report = {"prereg": "research_520_rome_main/PREREG_rome_main.md", "train_rows": int(len(train)), "test_rows": int(len(test)),
              "copy_share_train": float(train.is_copy.mean()), "copy_share_test": float(test.is_copy.mean()),
              "auc_new": float(roc_auc_score(test.is_copy, p)), "auc_shipped_gate": float(roc_auc_score(test.is_copy, gate)),
              "base_rmse": float(np.sqrt(np.mean((b - y) ** 2)))}
    tm = pd.to_datetime(test.MVT_TIME_UTC_mvt, utc=True)
    half = (tm.dt.month.astype(str) + np.where(tm.dt.day <= 15, "a", "b")).to_numpy()
    for name, v in arms(b, test.delay.to_numpy(float), p).items():
        g = (b - y) ** 2 - (v - y) ** 2
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-100:]] = False
        r = {"rmse": float(np.sqrt(np.mean((v - y) ** 2))), "gain_sse": float(g.sum()), "gain_per_row": float(g.mean()),
             "without_best100": float(g[keep].sum()),
             "by_month": {m: float(g[(tm.dt.month == int(m)).to_numpy()].sum()) for m in ("1", "7")},
             "by_half": {h: float(g[half == h].sum()) for h in sorted(set(half))}}
        mse = ROWS_2026 * r["gain_per_row"] / N
        r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
        report[f"arm_{name}"] = r
    a = report["arm_A"]
    report["rules"] = {"pooled_positive": a["gain_sse"] > 0, "without_best100_positive": a["without_best100"] > 0,
                       "january_and_july": min(a["by_month"].values()) > 0, "four_half_months": min(a["by_half"].values()) > 0,
                       "planning_at_least_0_2_s": a["planning_s"] >= 0.2}
    report["verdict"] = "pass" if all(report["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

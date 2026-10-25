"""Test of research_620_new_family/PREREG_extratrees.md. Fits one ExtraTrees model (saved). No upload.
Run from the repository root:  python3 -u -B research_620_new_family/extratrees.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_360_integration"))
import main_model as MM  # noqa: E402  (torch before tree libraries)
from research_250 import candidate_features as B  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, N, REF = "MVT_ID_mvt", 344841, 246.3837
DATA = ROOT / ".cache/clock_queue_release_20260912/data"
FEATURES = B.main_features(MM.MERKMALE_HAUPT)
CATS = [c for c in MM.KAT if c in FEATURES]
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)
CACHE = ROOT / ".cache/research_620"
OUT = ROOT / "research_620_new_family/results/extratrees.json"


def main_rows(name):
    cols = list(dict.fromkeys([ID, "hat_aobt", "y", "ersatz_s", "ADEP_mvt"] + FEATURES))       # no column twice
    f = pd.read_parquet(DATA / f"{name}.parquet", columns=cols)
    return f[f.hat_aobt.eq(1) & f.y.notna()].drop_duplicates(ID).reset_index(drop=True)


def encoder(train):
    return {c: {v: i for i, v in enumerate(sorted(train[c].astype(str).unique()))} for c in CATS}


def matrix(frame, enc):
    x = frame[FEATURES].copy()
    for c in CATS:
        x[c] = x[c].astype(str).map(enc[c]).fillna(-1).astype(float)
    return x.astype(float).fillna(-999.0).to_numpy(np.float32)


def fit(train, tag):
    path = CACHE / f"et_{tag}.pkl"
    if path.exists():
        return joblib.load(path)
    enc = encoder(train)
    target = np.clip(train.y.to_numpy(float) - train.ersatz_s.to_numpy(float), -3600, 3600)
    m = ExtraTreesRegressor(n_estimators=200, max_features=0.33, min_samples_leaf=40, bootstrap=True, max_samples=0.5,
                            random_state=0, n_jobs=8).fit(matrix(train, enc), target)
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "encoder": enc}, path)
    return {"model": m, "encoder": enc}


def predict(bundle, frame):
    return frame.ersatz_s.to_numpy(float) + bundle["model"].predict(matrix(frame, bundle["encoder"]))


def main():
    learn = main_rows("months_2_3_4_5_6_8_9_10_11_12")
    test = main_rows("months_1_7")
    bundle = fit(learn, "learning_months")
    comps = H.components_2025()
    comps["base"] = L.compose_base(comps)
    d = comps[[ID, "y", "base", "source_month", "airport"]].merge(test[list(dict.fromkeys([ID] + FEATURES + ["ersatz_s"]))], on=ID, how="inner", validate="one_to_one")
    if len(d) != len(comps):
        raise SystemExit(f"alignment: {len(d)} of {len(comps)} rows")
    d["et"] = predict(bundle, d)
    y, b, e = d.y.to_numpy(float), d.base.to_numpy(float), d.et.to_numpy(float)
    blend = lambda w: np.maximum((1 - w) * b + w * e, 300.0)
    report = {"prereg": "research_620_new_family/PREREG_extratrees.md", "rows": int(len(d)),
              "rmse": {"base": float(np.sqrt(np.mean((b - y) ** 2))), "extratrees": float(np.sqrt(np.mean((np.maximum(e, 300) - y) ** 2)))},
              "correlation_of_errors": float(np.corrcoef(b - y, e - y)[0, 1])}
    g_all = np.zeros(len(d))
    for pick_m, apply_m in ((1, 7), (7, 1)):
        pk, ap = (d.source_month == pick_m).to_numpy(), (d.source_month == apply_m).to_numpy()
        scores = {w: float(np.mean((blend(w)[pk] - y[pk]) ** 2)) for w in WEIGHTS}
        w = min(scores, key=scores.get)
        g = (b[ap] - y[ap]) ** 2 - (blend(w)[ap] - y[ap]) ** 2
        g_all[ap] = g
        keep = np.ones(len(g), bool)
        keep[np.argsort(g, kind="stable")[-100:]] = False
        report[f"pick_{pick_m}_apply_{apply_m}"] = {"w": w, "gain_per_row": float(g.mean()), "without_best100": float(g[keep].mean())}
    by_ap = {ap: float(g_all[d.airport.eq(ap).to_numpy()].mean()) for ap in sorted(d.airport.unique())}
    report["by_airport"] = by_ap
    v77 = pd.read_parquet(ROOT / ".cache/research_590/v77/candidate_v77.parquet").set_index(ID).TAXITIME_SEC_mvt
    base26 = H.current_base_2026()
    own = H.ownership_v62().set_index(ID)
    main26 = own[~own.owner.eq("fallback")].index
    final = main26[(v77.loc[main26].to_numpy() == base26.loc[main26].to_numpy().astype(v77.dtype))]
    rows26 = own.loc[final, "airport"].value_counts().to_dict()
    report["base_final_rows_2026"] = {k: int(v) for k, v in rows26.items()}
    mse = sum(by_ap.get(ap, 0.0) * n for ap, n in rows26.items()) / N
    report["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    a, b2 = report["pick_1_apply_7"], report["pick_7_apply_1"]
    report["rules"] = {"both_directions": min(a["gain_per_row"], b2["gain_per_row"]) > 0,
                       "without_best100_both": min(a["without_best100"], b2["without_best100"]) > 0,
                       "seven_of_ten_airports": sum(v > 0 for v in by_ap.values()) >= 7, "planning_at_least_0_1_s": report["planning_s"] >= 0.1}
    report["verdict"] = "pass" if all(report["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

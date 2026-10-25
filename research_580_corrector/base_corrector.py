"""Test of research_580_corrector/PREREG_base_corrector.md: January to July and back. No upload.
Run from the repository root:  python3 -u -B research_580_corrector/base_corrector.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_360_integration"))
import main_model  # noqa: F401,E402  (torch before tree libraries on macOS)
import lightgbm as lgb  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, N, REF = "MVT_ID_mvt", 344841, 249.3908
SEEDS = (177, 277, 377)
TZ = {"LIRF": "Europe/Rome", "LFPG": "Europe/Paris", "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul", "EGLL": "Europe/London",
      "EHAM": "Europe/Amsterdam", "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid"}
CATS = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
        "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt", "country"]
NUM = ["base", "ersatz", "delay", "aobt_eobt", "aobt_iobt", "aobt_lobt", "hour", "weekday"]
RAW = [ID, "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
       "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]
OUT = ROOT / "research_580_corrector/results/base_corrector.json"


def features(raw, base):
    f = pd.DataFrame(index=raw.index)
    t = {c: pd.to_datetime(raw[c], utc=True) for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt")}
    sec = lambda a, b: (t[a] - t[b]).dt.total_seconds()
    f["base"] = np.asarray(base, float)
    f["ersatz"], f["delay"] = sec("MVT_TIME_UTC_mvt", "AOBT_3_flt"), sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
    f["aobt_eobt"], f["aobt_iobt"], f["aobt_lobt"] = (sec("AOBT_3_flt", c) for c in ("EOBT_1_flt", "IOBT_flt", "LOBT_flt"))
    hour, wd = np.zeros(len(raw)), np.zeros(len(raw))
    for ap, tz in TZ.items():
        m = raw.ADEP_mvt.eq(ap).to_numpy()
        loc = t["MVT_TIME_UTC_mvt"][m].dt.tz_convert(tz)
        hour[m], wd[m] = loc.dt.hour, loc.dt.weekday
    f["hour"], f["weekday"] = hour, wd
    for c in CATS[:-1]:
        f[c] = raw[c].astype(str)
    f["country"] = raw.ADES_mvt.astype(str).str[:2]
    return f


def fit_predict(x_tr, y_tr, x_te):
    x_tr, x_te = x_tr.copy(), x_te.copy()
    for c in CATS:
        levels = sorted(x_tr[c].unique())
        x_tr[c] = pd.Categorical(x_tr[c], categories=levels)
        x_te[c] = pd.Categorical(x_te[c], categories=levels)
    preds = []
    for s in SEEDS:
        m = lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=.03, num_leaves=31, min_child_samples=200,
                              reg_lambda=100., colsample_bytree=.8, subsample=.8, subsample_freq=1, random_state=s, n_jobs=8, verbosity=-1)
        m.fit(x_tr[NUM + CATS], y_tr, categorical_feature=CATS)
        preds.append(m.predict(x_te[NUM + CATS]))
    return np.mean(preds, axis=0)


def main():
    comps = H.components_2025()
    comps["base"] = L.compose_base(comps)
    raw = pd.concat([pd.read_parquet(p, columns=RAW) for p in sorted((ROOT / "data").glob("training_2025-0[17]-*.parquet"))],
                    ignore_index=True).drop_duplicates(ID)
    d = comps[[ID, "y", "base", "source_month"]].merge(raw, on=ID, how="left", validate="one_to_one")
    if d.ADEP_mvt.isna().any():
        raise SystemExit("rows without raw columns")
    x = features(d, d.base)
    y = d.y.to_numpy(float)
    b = d.base.to_numpy(float)
    new = np.full(len(d), np.nan)
    for tr_m, te_m in ((1, 7), (7, 1)):
        tr, te = (d.source_month == tr_m).to_numpy(), (d.source_month == te_m).to_numpy()
        new[te] = np.maximum(b[te] + fit_predict(x[tr], (y - b)[tr], x[te]), 300.0)
        print("direction", tr_m, "->", te_m, flush=True)
    g = (b - y) ** 2 - (new - y) ** 2
    own = H.ownership_v62()
    base26 = own[own.owner.eq("base")].airport.value_counts()
    r = {"prereg": "research_580_corrector/PREREG_base_corrector.md", "rows": int(len(d)),
         "base_rmse": float(np.sqrt(np.mean((b - y) ** 2))), "corrected_rmse": float(np.sqrt(np.mean((new - y) ** 2)))}
    for m in (1, 7):
        k = (d.source_month == m).to_numpy()
        gm = g[k]
        keep = np.ones(len(gm), bool)
        keep[np.argsort(gm, kind="stable")[-100:]] = False
        r[f"test_month_{m}"] = {"rows": int(k.sum()), "gain_per_row": float(gm.mean()), "without_best100": float(gm[keep].mean())}
    by_ap = {ap: float(g[d.ADEP_mvt.eq(ap).to_numpy()].mean()) for ap in sorted(d.ADEP_mvt.unique())}
    r["by_airport"] = by_ap
    r["base_owned_2026"] = {ap: int(n) for ap, n in base26.items()}
    mse = sum(by_ap.get(ap, 0.0) * n for ap, n in base26.items() if by_ap.get(ap, 0.0) > 0) / N
    r["planning_s"] = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else 0.0
    r["rules"] = {"both_directions_positive": min(r["test_month_1"]["gain_per_row"], r["test_month_7"]["gain_per_row"]) > 0,
                  "without_best100_both": min(r["test_month_1"]["without_best100"], r["test_month_7"]["without_best100"]) > 0,
                  "seven_of_ten_airports": sum(v > 0 for v in by_ap.values()) >= 7, "planning_at_least_0_2_s": r["planning_s"] >= 0.2}
    r["verdict"] = "pass" if all(r["rules"].values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=2, default=str) + "\n")
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()

"""A second model family in the fallback lane, six pairs (research_460_fallback_family/PREREG_fallback_family.md).
No production change, no upload.
Run from the repository root:  python3 -u -B research_460_fallback_family/fallback_family.py
"""
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_420_fallback"))
import main_model as H  # noqa: E402  torch before tree libraries on macOS
import features_stand_rate as MSQ  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostClassifier, CatBoostRegressor  # noqa: E402
from research_300.heavy_lock import heavy_work  # noqa: E402
import rome_count as RC  # noqa: E402  (tables(), BASE, WITH_NT, saved predictions)

CACHE = ROOT / ".cache/research_460/fallback_family"
OUT = ROOT / "research_460_fallback_family/results/fallback_family.json"
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
N_FB, N, REF = 5290, 344841, 256.4921
PARAMS = dict(depth=6, learning_rate=0.05, l2_leaf_reg=10, random_seed=0, thread_count=4, verbose=0)


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def cat_frame(frame, cols):
    x = frame[cols].copy()
    cats = [c for c in cols if c in H.KAT]
    for c in cats:
        x[c] = x[c].astype(str)
    return x, cats


def cat_parts(key, found):
    path = CACHE / f"cat_{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    a, b = map(int, key.split("_"))
    train = pd.read_parquet(found[tuple(sorted(set(range(1, 13)) - {a, b}))]).reset_index(drop=True)
    valid = pd.read_parquet(found[(a, b)]).reset_index(drop=True)
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(train)
    tr, va = MSQ.ausser_faltung(train), MSQ.wende_an(mapping, valid)
    z0, y = tr.z.eq(0).to_numpy(), tr.y.to_numpy(float)
    xt, cats = cat_frame(tr, RC.BASE)
    xv, _ = cat_frame(va, RC.BASE)
    with heavy_work(f"research_460 fallback CatBoost {key}", log=log):
        p = CatBoostClassifier(iterations=600, cat_features=cats, **PARAMS).fit(xt, tr.z).predict_proba(xv)[:, 1]
        q = p ** H.HOCH
        height = ((1 - H.ANTEIL_HM) * va.versp_min.to_numpy(float) * 60
                  + H.ANTEIL_HM * CatBoostRegressor(iterations=300, cat_features=cats, **PARAMS).fit(xt[~z0], y[~z0]).predict(xv))
        normal = CatBoostRegressor(iterations=400, cat_features=cats, **PARAMS).fit(xt[z0], y[z0]).predict(xv)
        a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
        rows = z0 & np.isfinite(a_tr)
        xt2, cats2 = cat_frame(tr, RC.WITH_NT)
        xv2, _ = cat_frame(va, RC.WITH_NT)
        offset = CatBoostRegressor(iterations=400, cat_features=cats2, **PARAMS).fit(xt2[rows], y[rows] - a_tr[rows])
        has = np.isfinite(a_va)
        out = normal.copy()
        out[has] = a_va[has] + offset.predict(xv2[has])
        pred = np.maximum((1 - q) * out + q * height, 0)
    frame = pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "cat_pred": pred, "fold": key,
                          "month": pd.to_datetime(va.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()})
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    del train, valid, tr, va
    gc.collect()
    return frame


def main():
    found, frames = RC.tables(), []
    for a, b in PAIRS:
        key = f"{a}_{b}"
        c = cat_parts(key, found)
        s = pd.read_parquet(RC.SAVED / f"predictions_{key}.parquet", columns=["MVT_ID_mvt", "prod_anchor", "y"])
        m = s.merge(c, on="MVT_ID_mvt", validate="one_to_one")
        if len(m) != len(s):
            raise SystemExit(f"{key}: rows do not align")
        frames.append(m)
        log(key, "shipped", round(float(np.sqrt(np.mean((m.prod_anchor - m.y) ** 2))), 1), "catboost", round(float(np.sqrt(np.mean((m.cat_pred - m.y) ** 2))), 1))
    d = pd.concat(frames, ignore_index=True)
    y, ship = d.y.to_numpy(float), d.prod_anchor.to_numpy(float)
    arm = 0.5 * ship + 0.5 * d.cat_pred.to_numpy(float)
    gain = (ship - y) ** 2 - (arm - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    pairs = {k: float(gain[d.fold.eq(k).to_numpy()].mean()) for k in d.fold.unique()}
    m17 = d.fold.eq("1_7").to_numpy()
    months = {int(mo): float(gain[m17 & d.month.eq(mo).to_numpy()].mean()) for mo in (1, 7)}
    per_row = float(gain.mean())
    mse = per_row * N_FB / N
    official = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    rules = {"1_pooled_positive": per_row > 0, "2_without_best_100_positive": float(gain[keep].mean()) > 0,
             "3_four_of_six_pairs": sum(v > 0 for v in pairs.values()) >= 4,
             "4_january_and_july_1_7": all(v > 0 for v in months.values()), "5_official_at_least_0_1_s": official >= 0.1}
    rep = {"prereg": "research_460_fallback_family/PREREG_fallback_family.md", "rows": int(len(d)),
           "rmse_shipped": float(np.sqrt(((ship - y) ** 2).mean())), "rmse_catboost_alone": float(np.sqrt(((d.cat_pred - y) ** 2).mean())),
           "rmse_blend": float(np.sqrt(((arm - y) ** 2).mean())), "gain_per_row": per_row, "without_best_100": float(gain[keep].mean()),
           "pairs": pairs, "months_1_7": months, "official_scale_s": official, "rules": rules,
           "verdict": "pass" if all(rules.values()) else "fail"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

"""Fallback lane, part by part: LightGBM (the shipped recipe) against CatBoost (research_460 recipe), on the selection
pairs 1_7, 3_9 and 5_11 only. Descriptive; pairs 2_8, 4_10 and 6_12 are kept unexamined at part level so that any hybrid
chosen here can be confirmed on them. No upload.

Per part: classifier (labels above 1.5 h) AUC, log loss and Brier score; height regressor RMSE on the extreme rows;
normal regressor RMSE on the normal rows; anchored normal (anchor + offset) RMSE on the normal rows that have an anchor.
Also the full fallback RMSE with exactly one part swapped to CatBoost (four single-swap hybrids).
Run from the repository root:  python3 -u -B research_460_fallback_family/parts_diagnostic.py
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_420_fallback"))
sys.path.insert(0, str(ROOT / "research_460_fallback_family"))
import main_model as H  # noqa: E402
import features_stand_rate as MSQ  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostClassifier, CatBoostRegressor  # noqa: E402
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss  # noqa: E402
from research_300.heavy_lock import heavy_work  # noqa: E402
import rome_count as RC  # noqa: E402
import fallback_family as FF  # noqa: E402

SELECTION = [(1, 7), (3, 9), (5, 11)]
OUT = ROOT / "research_460_fallback_family/results/parts_diagnostic.json"


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def parts(key, found):
    cache = ROOT / f".cache/research_460/parts_{key}.npz"
    if cache.exists():
        return dict(np.load(cache))
    o = _parts(key, found)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **o)
    return o


def _parts(key, found):
    a, b = map(int, key.split("_"))
    train = pd.read_parquet(found[tuple(sorted(set(range(1, 13)) - {a, b}))]).reset_index(drop=True)
    valid = pd.read_parquet(found[(a, b)]).reset_index(drop=True)
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(train)
    tr, va = MSQ.ausser_faltung(train), MSQ.wende_an(mapping, valid)
    z0, y = tr.z.eq(0).to_numpy(), tr.y.to_numpy(float)
    rows = z0 & np.isfinite(tr.nt_anchor.to_numpy(float))
    a_tr = tr.nt_anchor.to_numpy(float)
    out = {"y": va.y.to_numpy(float), "versp": va.versp_min.to_numpy(float) * 60, "anchor": va.nt_anchor.to_numpy(float)}
    with heavy_work(f"research_460 parts diagnostic {key}", log=log):
        # LightGBM, the shipped recipe (research_240/fallback_anchor_production.parts / anchored)
        out["lgb_p"] = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31, min_child_samples=20,
                                          verbose=-1, n_jobs=4, random_state=0).fit(tr[RC.BASE], tr.z).predict_proba(va[RC.BASE])[:, 1]
        out["lgb_h"] = RC.reg(300, 15, 10).fit(tr.loc[~z0, RC.BASE], y[~z0]).predict(va[RC.BASE])
        out["lgb_n"] = RC.reg(400, 31, 20).fit(tr.loc[z0, RC.BASE], y[z0]).predict(va[RC.BASE])
        out["lgb_o"] = RC.reg(400, 31, 20).fit(tr.loc[rows, RC.WITH_NT], y[rows] - a_tr[rows]).predict(va[RC.WITH_NT])
        # CatBoost, the research_460 recipe
        xt, cats = FF.cat_frame(tr, RC.BASE)
        xv, _ = FF.cat_frame(va, RC.BASE)
        xt2, cats2 = FF.cat_frame(tr, RC.WITH_NT)
        xv2, _ = FF.cat_frame(va, RC.WITH_NT)
        out["cat_p"] = CatBoostClassifier(iterations=600, cat_features=cats, **FF.PARAMS).fit(xt, tr.z).predict_proba(xv)[:, 1]
        out["cat_h"] = CatBoostRegressor(iterations=300, cat_features=cats, **FF.PARAMS).fit(xt[~z0], y[~z0]).predict(xv)
        out["cat_n"] = CatBoostRegressor(iterations=400, cat_features=cats, **FF.PARAMS).fit(xt[z0], y[z0]).predict(xv)
        out["cat_o"] = CatBoostRegressor(iterations=400, cat_features=cats2, **FF.PARAMS).fit(xt2[rows], y[rows] - a_tr[rows]).predict(xv2)
    return out


def assemble(o, p, h, n, off):
    q = o[p] ** H.HOCH
    height = (1 - H.ANTEIL_HM) * o["versp"] + H.ANTEIL_HM * o[h]
    normal = o[n].copy()
    has = np.isfinite(o["anchor"])
    normal[has] = o["anchor"][has] + o[off][has]
    return np.maximum((1 - q) * normal + q * height, 0)


def main():
    found, per_pair = RC.tables(), {}
    pooled = {k: [] for k in ("y", "z", "lgb_p", "cat_p")}
    swaps = {s: [] for s in ("shipped", "swap_classifier", "swap_height", "swap_normal", "swap_offset", "all_catboost")}
    part_err = {k: [] for k in ("h_lgb", "h_cat", "n_lgb", "n_cat", "an_lgb", "an_cat")}
    for a, b in SELECTION:
        key = f"{a}_{b}"
        o = parts(key, found)
        y, z = o["y"], (o["y"] > H.SCHWELLE_EXTREM).astype(int)
        has = np.isfinite(o["anchor"])
        builds = {"shipped": ("lgb_p", "lgb_h", "lgb_n", "lgb_o"), "swap_classifier": ("cat_p", "lgb_h", "lgb_n", "lgb_o"),
                  "swap_height": ("lgb_p", "cat_h", "lgb_n", "lgb_o"), "swap_normal": ("lgb_p", "lgb_h", "cat_n", "lgb_o"),
                  "swap_offset": ("lgb_p", "lgb_h", "lgb_n", "cat_o"), "all_catboost": ("cat_p", "cat_h", "cat_n", "cat_o")}
        pair = {}
        for name, spec in builds.items():
            pred = assemble(o, *spec)
            swaps[name].append((pred - y) ** 2)
            pair[name] = float(np.sqrt(np.mean((pred - y) ** 2)))
        saved = pd.read_parquet(RC.SAVED / f"predictions_{key}.parquet", columns=["prod_anchor"]).prod_anchor.to_numpy(float)
        pair["shipped_equals_saved_max_diff_s"] = float(np.max(np.abs(assemble(o, *builds["shipped"]) - saved)))
        pair["auc"] = {"lightgbm": float(roc_auc_score(z, o["lgb_p"])), "catboost": float(roc_auc_score(z, o["cat_p"]))}
        per_pair[key] = pair
        log(key, json.dumps(pair))
        for k in ("lgb_p", "cat_p"):
            pooled[k].append(o[k])
        pooled["y"].append(y)
        pooled["z"].append(z)
        e = z == 1
        n = (z == 0)
        part_err["h_lgb"].append((o["lgb_h"][e] - y[e]) ** 2); part_err["h_cat"].append((o["cat_h"][e] - y[e]) ** 2)
        part_err["n_lgb"].append((o["lgb_n"][n] - y[n]) ** 2); part_err["n_cat"].append((o["cat_n"][n] - y[n]) ** 2)
        an = n & has
        part_err["an_lgb"].append((o["anchor"][an] + o["lgb_o"][an] - y[an]) ** 2)
        part_err["an_cat"].append((o["anchor"][an] + o["cat_o"][an] - y[an]) ** 2)
    z = np.concatenate(pooled["z"])
    p = {k: np.clip(np.concatenate(pooled[k]), 1e-6, 1 - 1e-6) for k in ("lgb_p", "cat_p")}
    rmse = lambda k: float(np.sqrt(np.concatenate(part_err[k]).mean()))
    report = {"selection_pairs": [f"{a}_{b}" for a, b in SELECTION], "per_pair": per_pair,
              "classifier": {fam: {"auc": float(roc_auc_score(z, p[f"{fam[:3]}_p"])), "log_loss": float(log_loss(z, p[f"{fam[:3]}_p"])),
                                   "brier": float(brier_score_loss(z, p[f"{fam[:3]}_p"]))} for fam in ("lgb", "cat")},
              "height_rmse_on_extreme_rows": {"lightgbm": rmse("h_lgb"), "catboost": rmse("h_cat")},
              "normal_rmse_on_normal_rows": {"lightgbm": rmse("n_lgb"), "catboost": rmse("n_cat")},
              "anchored_normal_rmse": {"lightgbm": rmse("an_lgb"), "catboost": rmse("an_cat")},
              "fallback_rmse_single_swaps": {k: float(np.sqrt(np.concatenate(v).mean())) for k, v in swaps.items()}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "per_pair"}, indent=1))


if __name__ == "__main__":
    main()

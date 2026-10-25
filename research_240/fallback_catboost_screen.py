"""Pre-registered six-pair fallback screen: model-class diversity, CatBoost averaged with the LightGBM fallback.

Pre-specified 13.09.2026, before any fit. The main lane gains from mixing model classes (main_model.py: LightGBM 229.69,
CatBoost 228.16, blend 226.08 on the January/July holdout); every fallback component has only ever been one LightGBM.
Earlier fallback screens changed features or blends, which moved few extreme rows; averaging two model classes can lower
variance across many rows. No new columns, no leaderboard feedback. Run outside the repository, then moved into research_240/.

Baseline: prod_anchor, the scored v46 fallback recipe with production encoding, rebuilt and gated against the saved
predictions of research_240/fallback_anchor_production.py (prod_v44 and prod_anchor within 1e-6 s).
CatBoost counterparts, fixed now: classifier CatBoostClassifier(600 iterations, learning rate 0.05, depth 6, Logloss);
normal and offset CatBoostRegressor(400, 0.05, depth 6, RMSE); height CatBoostRegressor(300, 0.05, depth 4, RMSE); all
random_seed 0, thread_count = --jobs; categorical inputs are the H.KAT columns among the features, as strings.
Blend weights fixed at 0.5 / 0.5, no tuning:
  C1_catboost_offset  only the anchored offset regressor is the average of LightGBM and CatBoost
  C2_catboost_all     classifier probability, height model, normal and offset regressors each averaged
Decision rule against prod_anchor, pooled over six pairs, as for the structural variants of
research_240/fallback_rescreen_production.py: (1) MSE gain positive, (2) positive without its 100 most favourable rows,
(3) at least 4 of 6 pairs gain, (4) January and July both gain in 1_7. No permutation control, since no column is added.
A variant advances only with all four; the candidate is the advancing variant with the larger pooled gain. Otherwise
nothing is built or uploaded. Fallback-only screen; not full ensemble; not a leaderboard forecast.

Run:
  python3 -B research_240/fallback_catboost_screen.py --project <repository> --output <folder>
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()
project, output = args.project.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import features_stand_rate as MSQ
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from research_240 import features_neighbour_taxi as NT

BASE = list(H.MERKMALE_RUECK)
OFFSET = BASE + [c for c in NT.MERKMALE if c not in BASE]
VARIANTS = {"C1_catboost_offset": ("offset",), "C2_catboost_all": ("classifier", "height", "normal", "offset")}
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
V46_OFFICIAL, FALLBACK_SHARE_2026 = 268.2394, 5290 / 344841
TABLES = project / ".cache/research_240/rescreen_production"
SAVED = project / ".cache/research_240/fallback_anchor_production"
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), project / "main_model.py", project / "features_stand_rate.py", Path(NT.__file__).resolve()]}
index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet") for s in TABLES.glob("rescreen_*.json")}
started = time.time()


def stand_rate(train, valid):
    train, valid = train.copy(), valid.copy()
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(train)
    train = MSQ.ausser_faltung(train)
    valid = MSQ.wende_an(mapping, valid)
    return train, valid


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                             min_child_samples=minimum, verbose=-1, n_jobs=args.jobs, random_state=0)


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=args.jobs, random_state=0)


def cat_columns(features):
    return [c for c in features if c in H.KAT]


def cat_frame(frame, features):
    x = frame[features].copy()
    for c in cat_columns(features):
        x[c] = x[c].astype(str)
    return x


def cb_classifier():
    return CatBoostClassifier(iterations=600, learning_rate=.05, depth=6, loss_function="Logloss", random_seed=0,
                              verbose=0, thread_count=args.jobs, allow_writing_files=False)


def cb_reg(iterations, depth):
    return CatBoostRegressor(iterations=iterations, learning_rate=.05, depth=depth, loss_function="RMSE", random_seed=0,
                             verbose=0, thread_count=args.jobs, allow_writing_files=False)


def recipe(tr, va, cat_parts=(), anchored=True):
    """research_240/fallback_weather_screen.py recipe; cat_parts averages the named components with CatBoost."""
    features = BASE
    z0 = tr.z.eq(0).to_numpy()
    y = tr.y.to_numpy(float)
    cats = cat_columns(features)
    c_tr, c_va = (cat_frame(tr, features), cat_frame(va, features)) if cat_parts else (None, None)
    p = classifier().fit(tr[features], tr.z).predict_proba(va[features])[:, 1]
    if "classifier" in cat_parts:
        p = .5 * p + .5 * cb_classifier().fit(c_tr, tr.z, cat_features=cats).predict_proba(c_va)[:, 1]
    height_model = reg(300, 15, 10).fit(tr.loc[~z0, features], y[~z0]).predict(va[features])
    if "height" in cat_parts:
        height_model = .5 * height_model + .5 * cb_reg(300, 4).fit(c_tr.loc[~z0], y[~z0], cat_features=cats).predict(c_va)
    height = (1-H.ANTEIL_HM)*va.versp_min.to_numpy(float)*60 + H.ANTEIL_HM*height_model
    normal = reg(400, 31, 20).fit(tr.loc[z0, features], y[z0]).predict(va[features])
    if "normal" in cat_parts:
        normal = .5 * normal + .5 * cb_reg(400, 6).fit(c_tr.loc[z0], y[z0], cat_features=cats).predict(c_va)
    if anchored:
        a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
        rows = z0 & np.isfinite(a_tr)
        has = np.isfinite(a_va)
        offset = reg(400, 31, 20).fit(tr.loc[rows, OFFSET], y[rows]-a_tr[rows]).predict(va.loc[has, OFFSET])
        if "offset" in cat_parts:
            o_tr, o_va = cat_frame(tr, OFFSET), cat_frame(va, OFFSET)
            offset = .5 * offset + .5 * cb_reg(400, 6).fit(o_tr.loc[rows], y[rows]-a_tr[rows], cat_features=cat_columns(OFFSET)).predict(o_va.loc[has])
        normal = normal.copy()
        normal[has] = a_va[has] + offset
    q = p ** H.HOCH
    return np.maximum((1-q)*normal + q*height, 0)


def load_pair(pair):
    learn = [m for m in range(1, 13) if m not in pair]
    train_raw = pd.read_parquet(index[tuple(learn)]).reset_index(drop=True)
    valid_raw = pd.read_parquet(index[tuple(sorted(pair))]).reset_index(drop=True)
    assert set(train_raw.MVT_ID_mvt).isdisjoint(set(valid_raw.MVT_ID_mvt))
    return stand_rate(train_raw, valid_raw)


def jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


frames, gates = [], []
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    before = time.time()
    tr, va = load_pair(pair)
    yv = va.y.to_numpy(float)
    frame = va[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "nt_anchor"]].copy()
    frame["ADEP_mvt"] = va.ADEP_mvt.astype(str).to_numpy()
    frame["prod_v44"] = recipe(tr, va, anchored=False)
    frame["prod_anchor"] = recipe(tr, va)
    saved = pd.read_parquet(SAVED / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt]
    gate = {"pair": key, "prod_v44_max_diff": float(np.abs(frame.prod_v44.to_numpy()-saved.prod_v44.to_numpy()).max()),
            "prod_anchor_max_diff": float(np.abs(frame.prod_anchor.to_numpy()-saved.prod_anchor.to_numpy()).max())}
    gate["passed"] = gate["prod_v44_max_diff"] <= 1e-6 and gate["prod_anchor_max_diff"] <= 1e-6
    gates.append(gate)
    print("GATE_B", json.dumps(gate), flush=True)
    if not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps(gates, indent=2))
        sys.exit(f"Replication gate failed on {key}")
    for name, parts in VARIANTS.items():
        frame[name] = recipe(tr, va, cat_parts=parts)
    print("PAIR", key, json.dumps({c: round(float(np.sqrt(np.mean((frame[c]-yv)**2))), 3) for c in ["prod_v44", "prod_anchor"] + list(VARIANTS)}),
          f"{time.time()-before:.0f}s", flush=True)
    frame["fold"] = key
    frame.to_parquet(output / f"predictions_{key}.parquet", index=False)
    frames.append(frame)
    del tr, va
    gc.collect()

pooled = pd.concat(frames, ignore_index=True)
y = pooled.y.to_numpy(float)
folds = pooled.fold.to_numpy()
stamps = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
month = np.array([m if str(m) in f.split("_") else m % 12 + 1 for m, f in zip(stamps.dt.month.to_numpy(), folds)])
_, inverse = np.unique((pooled.ADEP_mvt + "|" + stamps.dt.strftime("%Y-%m-%d")).to_numpy(), return_inverse=True)
cluster_rows = np.bincount(inverse)
draws = np.random.default_rng(20260913).integers(0, cluster_rows.size, size=(args.bootstrap, cluster_rows.size))
extreme = y > H.SCHWELLE_EXTREM


def compare(candidate):
    gain = (pooled.prod_anchor.to_numpy(float)-y)**2 - (np.asarray(candidate, float)-y)**2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    pair_gains = {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}
    boot = np.bincount(inverse, weights=gain)[draws].sum(axis=1)/cluster_rows[draws].sum(axis=1)
    out = {"pooled_mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
           "positive_sse_mass": float(gain[gain > 0].sum()), "negative_sse_mass": float(gain[gain < 0].sum()),
           "normal_rows_mse_gain_contribution": float(gain[~extreme].sum()/len(gain)), "extreme_rows_mse_gain_contribution": float(gain[extreme].sum()/len(gain)),
           "pair_gains": pair_gains, "pairs_positive": int(sum(g > 0 for g in pair_gains.values())),
           "january_gain_1_7": float(gain[(folds == "1_7") & (month == 1)].mean()), "july_gain_1_7": float(gain[(folds == "1_7") & (month == 7)].mean()),
           "airport_day_bootstrap_mse_gain_95pct": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
           "airport_contributions": {a: float(gain[pooled.ADEP_mvt.to_numpy() == a].sum()/len(gain)) for a in sorted(pooled.ADEP_mvt.unique())},
           "mechanical_overall_rmse_if_full_transfer_not_forecast": float(np.sqrt(max(V46_OFFICIAL**2 - FALLBACK_SHARE_2026*float(gain.mean()), 0)))}
    out["checks"] = {"1_pooled_gain_positive": out["pooled_mse_gain"] > 0, "2_survives_top100_removal": out["mse_gain_without_top100_rows"] > 0,
                     "3_at_least_4_of_6_pairs": out["pairs_positive"] >= 4,
                     "4_january_and_july_gain_in_1_7": out["january_gain_1_7"] > 0 and out["july_gain_1_7"] > 0}
    out["advance"] = bool(all(out["checks"].values()))
    return out


report = {"validation": "fallback-only six-pair screen against the scored v46 fallback recipe, production encoding; not full ensemble; not a leaderboard forecast",
          "replication_gates": gates, "code_sha256": CODE,
          "rmse": {c: float(np.sqrt(np.mean((pooled[c].to_numpy(float)-y)**2))) for c in ["prod_v44", "prod_anchor"] + list(VARIANTS)},
          "rmse_normal_rows": {c: float(np.sqrt(np.mean((pooled[c].to_numpy(float)[~extreme]-y[~extreme])**2))) for c in ["prod_anchor"] + list(VARIANTS)},
          "variants": {name: compare(pooled[name]) for name in VARIANTS}}
advancing = [n for n, e in report["variants"].items() if e["advance"]]
report["decision"] = {"advancing": advancing, "candidate_for_2026": max(advancing, key=lambda n: report["variants"][n]["pooled_mse_gain"]) if advancing else None}
report["seconds"] = time.time()-started
(output / "pooled.json").write_text(json.dumps(report, indent=2, default=jsonable))
print("SUMMARY", json.dumps({"rmse": report["rmse"], "rmse_normal_rows": report["rmse_normal_rows"], "decision": report["decision"],
                             "variants": {n: {k: e[k] for k in ("pooled_mse_gain", "mse_gain_without_top100_rows", "normal_rows_mse_gain_contribution",
                                                                "extreme_rows_mse_gain_contribution", "pairs_positive", "pair_gains", "january_gain_1_7",
                                                                "july_gain_1_7", "airport_day_bootstrap_mse_gain_95pct",
                                                                "mechanical_overall_rmse_if_full_transfer_not_forecast", "checks", "advance")}
                                          for n, e in report["variants"].items()}}, indent=2, default=jsonable), flush=True)

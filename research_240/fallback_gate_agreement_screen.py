"""Pre-registered six-pair fallback screen: an extreme gate that needs LightGBM and CatBoost classifiers to agree.

Pre-specified 13.09.2026, before any fit of this script. Follow-up to fallback_catboost_screen.py on the same six pairs:
averaging every component with CatBoost lowered the fallback RMSE on normal rows (TAXITIME <= 5,400 s) from 607.0 to
540.3 (+73,893 MSE) but lost 150,456 MSE on extremes, while averaging only the offset regressor moved normal rows by +283.
Hypothesis: the normal-row gain comes from a less confident extreme classifier; requiring both classifiers to agree keeps
true extremes gated and releases normal rows. Only the gate q changes; the height, normal and offset models are the v46
LightGBM ones. Because this design follows a result on the same pairs, the decision rule is unchanged and strict.
Run outside the repository, then moved into research_240/. No leaderboard feedback.

p_lgb = the v46 extreme classifier; p_cb = CatBoostClassifier(600 iterations, learning rate 0.05, depth 6, Logloss,
random_seed 0, thread_count = --jobs) on the same 35 inputs, categorical H.KAT columns as strings. HOCH = 2.5.
  G1_mean_gate       q = ((p_lgb + p_cb) / 2) ** HOCH     isolates the classifier part of C2_catboost_all
  G2_geometric_gate  q = (p_lgb * p_cb) ** (HOCH / 2)     low when either classifier doubts
  G3_min_gate        q = min(p_lgb, p_cb) ** HOCH         strict agreement
Prediction max((1 - q) * normal + q * height, 0) with the anchored normal branch, as in the v46 recipe.
Replication gate: prod_v44 and prod_anchor reproduce research_240/fallback_anchor_production.py predictions within 1e-6 s.
Decision rule against prod_anchor, pooled over six pairs: (1) MSE gain positive, (2) positive without its 100 most favourable
rows, (3) at least 4 of 6 pairs gain, (4) January and July both gain in 1_7. No column is added, so no permutation control.
The candidate is the advancing variant with the largest pooled gain; otherwise nothing is built or uploaded.
Fallback-only screen; not full ensemble; not a leaderboard forecast.

Run:
  python3 -B research_240/fallback_gate_agreement_screen.py --project <repository> --output <folder>
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
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from research_240 import features_neighbour_taxi as NT

BASE = list(H.MERKMALE_RUECK)
WITH_NT = BASE + NT.MERKMALE
GATES = {"G1_mean_gate": lambda a, b: ((a + b) / 2) ** H.HOCH, "G2_geometric_gate": lambda a, b: (a * b) ** (H.HOCH / 2),
         "G3_min_gate": lambda a, b: np.minimum(a, b) ** H.HOCH}
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


def parts(tr, va):
    """Components of research_240/fallback_anchor_production.py: p, height, v44 normal and anchored normal."""
    z0 = tr.z.eq(0).to_numpy()
    y = tr.y.to_numpy(float)
    p = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31, min_child_samples=20,
                           verbose=-1, n_jobs=args.jobs, random_state=0).fit(tr[BASE], tr.z).predict_proba(va[BASE])[:, 1]
    height = (1-H.ANTEIL_HM)*va.versp_min.to_numpy(float)*60 + H.ANTEIL_HM*reg(300, 15, 10).fit(tr.loc[~z0, BASE], y[~z0]).predict(va[BASE])
    normal = reg(400, 31, 20).fit(tr.loc[z0, BASE], y[z0]).predict(va[BASE])
    a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
    rows = z0 & np.isfinite(a_tr)
    offset = reg(400, 31, 20).fit(tr.loc[rows, WITH_NT], y[rows]-a_tr[rows])
    has = np.isfinite(a_va)
    anchored = normal.copy()
    anchored[has] = a_va[has] + offset.predict(va.loc[has, WITH_NT])
    return p, height, normal, anchored


def combine(normal, height, q):
    return np.maximum((1-q)*normal + q*height, 0)


def cat_classifier(tr, va):
    cats = [c for c in BASE if c in H.KAT]
    x_tr, x_va = tr[BASE].copy(), va[BASE].copy()
    for c in cats:
        x_tr[c], x_va[c] = x_tr[c].astype(str), x_va[c].astype(str)
    model = CatBoostClassifier(iterations=600, learning_rate=.05, depth=6, loss_function="Logloss", random_seed=0, verbose=0,
                               thread_count=args.jobs, allow_writing_files=False)
    return model.fit(x_tr, tr.z, cat_features=cats).predict_proba(x_va)[:, 1]


def auc(labels, scores):
    ranks = pd.Series(scores).rank().to_numpy()
    pos = labels.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


frames, gates = [], []
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    before = time.time()
    learn = [m for m in range(1, 13) if m not in pair]
    tr, va = stand_rate(pd.read_parquet(index[tuple(learn)]).reset_index(drop=True), pd.read_parquet(index[tuple(sorted(pair))]).reset_index(drop=True))
    p_lgb, height, normal, anchored = parts(tr, va)
    p_cb = cat_classifier(tr, va)
    yv = va.y.to_numpy(float)
    frame = va[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "nt_anchor"]].copy()
    frame["ADEP_mvt"] = va.ADEP_mvt.astype(str).to_numpy()
    frame["p_lgb"], frame["p_cb"] = p_lgb, p_cb
    frame["prod_v44"] = combine(normal, height, p_lgb ** H.HOCH)
    frame["prod_anchor"] = combine(anchored, height, p_lgb ** H.HOCH)
    saved = pd.read_parquet(SAVED / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt]
    gate = {"pair": key, "prod_v44_max_diff": float(np.abs(frame.prod_v44.to_numpy()-saved.prod_v44.to_numpy()).max()),
            "prod_anchor_max_diff": float(np.abs(frame.prod_anchor.to_numpy()-saved.prod_anchor.to_numpy()).max())}
    gate["passed"] = gate["prod_v44_max_diff"] <= 1e-6 and gate["prod_anchor_max_diff"] <= 1e-6
    gates.append(gate)
    print("GATE_B", json.dumps(gate), flush=True)
    if not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps(gates, indent=2))
        sys.exit(f"Replication gate failed on {key}")
    for name, rule in GATES.items():
        frame[name] = combine(anchored, height, rule(p_lgb, p_cb))
    z = (yv > H.SCHWELLE_EXTREM).astype(int)
    print("PAIR", key, json.dumps({c: round(float(np.sqrt(np.mean((frame[c]-yv)**2))), 3) for c in ["prod_anchor"] + list(GATES)}),
          "AUC lgb", round(auc(z, p_lgb), 4), "cb", round(auc(z, p_cb), 4), f"{time.time()-before:.0f}s", flush=True)
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
          "auc_pooled": {"lgb": auc(extreme.astype(int), pooled.p_lgb.to_numpy()), "cb": auc(extreme.astype(int), pooled.p_cb.to_numpy()),
                         "mean": auc(extreme.astype(int), (pooled.p_lgb.to_numpy() + pooled.p_cb.to_numpy()) / 2)},
          "mean_probability": {"extremes": {"lgb": float(pooled.p_lgb[extreme].mean()), "cb": float(pooled.p_cb[extreme].mean())},
                               "normal": {"lgb": float(pooled.p_lgb[~extreme].mean()), "cb": float(pooled.p_cb[~extreme].mean())}},
          "rmse": {c: float(np.sqrt(np.mean((pooled[c].to_numpy(float)-y)**2))) for c in ["prod_v44", "prod_anchor"] + list(GATES)},
          "rmse_normal_rows": {c: float(np.sqrt(np.mean((pooled[c].to_numpy(float)[~extreme]-y[~extreme])**2))) for c in ["prod_anchor"] + list(GATES)},
          "variants": {name: compare(pooled[name]) for name in GATES}}
advancing = [n for n, e in report["variants"].items() if e["advance"]]
report["decision"] = {"advancing": advancing, "candidate_for_2026": max(advancing, key=lambda n: report["variants"][n]["pooled_mse_gain"]) if advancing else None}
report["seconds"] = time.time()-started
(output / "pooled.json").write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("SUMMARY", json.dumps({k: report[k] for k in ("auc_pooled", "mean_probability", "rmse", "rmse_normal_rows", "decision")}, indent=2), flush=True)
for n, e in report["variants"].items():
    print("VARIANT", n, json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in e.items() if k in ("pooled_mse_gain", "mse_gain_without_top100_rows",
          "normal_rows_mse_gain_contribution", "extreme_rows_mse_gain_contribution", "pairs_positive", "january_gain_1_7", "july_gain_1_7", "advance")}), flush=True)

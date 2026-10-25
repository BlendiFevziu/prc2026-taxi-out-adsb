"""Production-faithful six-pair re-validation of the neighbour-taxi anchor.

Why: the research_250 fallback screens, and research_240's anchor screen built on them,
re-encode every categorical input from the fallback rows alone and turn missing values into
a "nan" category. The scored v44/v45 fallback keeps H.baue's full-split categories and real
missing values. Measured 13.09.2026 on identical rows and features: a production-style
all-year refit reproduces v45's official fallback predictions exactly (max 0.0 s), the
research-style refit differs by RMS 199.3 s. Earlier fallback CV numbers described a
different model than the one that was scored.

Pre-specified 13.09.2026, before any fit, on the cached six-pair tables (no builds):
  research_v44   research encoding; gate: research_250/results/fallback_six_pair.json
  prod_v44       production encoding, i.e. no categorical re-encoding: the scored recipe
  prod_anchor    production encoding; the normal branch predicts y - nt_anchor
  perm_<seed>    prod_anchor with the nt_ columns permuted jointly within (split, airport),
                 seeds 101, 202, 303, 404, 505; classifier and height model are unaffected

Decision rule for prod_anchor against prod_v44, fixed now. Advance only if, pooled over six
pairs: (1) MSE gain is positive, (2) it survives removing its 100 most favourable rows,
(3) at least 4 of 6 pairs gain, (4) January and July both gain inside pair 1_7, and
(5) the real pooled gain exceeds every permuted pooled gain.
prod_v44 against research_v44 is reported descriptively with the same statistics.
Fallback-only screen; not full ensemble; not a leaderboard forecast; no submission.
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
parser.add_argument("--screen", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--seeds", default="101,202,303,404,505")
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()
project, screen, output = args.project.resolve(), args.screen.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import features_stand_rate as MSQ
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_240 import features_neighbour_taxi as NT

BASE = list(H.MERKMALE_RUECK)
WITH_NT = BASE + NT.MERKMALE
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
SEEDS = [int(s) for s in args.seeds.split(",")]
V45_OFFICIAL = 269.4753
REFERENCE = {f["fold"]: f["baseline_rmse"] for f in json.loads(
    (project / "research_250/results/fallback_six_pair.json").read_text())["fallback"]["folds"]}
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).resolve(), Path(NT.__file__).resolve(),
                  project / "main_model.py", project / "features_stand_rate.py")}


def cached_tables():
    found = {}
    for side in sorted(screen.glob("fallback_nt_*.json")):
        months = tuple(sorted(json.loads(side.read_text())["source_months"]))
        if months in found:
            raise SystemExit(f"Two cached tables for months {months}")
        found[months] = side.with_suffix(".parquet")
    return found


def stand_rate(train, valid):
    """Shared by both encodings: extreme flag and out-of-month stand rate, as in endmodell."""
    train, valid = train.copy(), valid.copy()
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(train)
    train = MSQ.ausser_faltung(train)
    valid = MSQ.wende_an(mapping, valid)
    return train, valid


def research_encode(train, valid):
    """The research_250 prepare() re-encoding, applied after the stand rate as it was there."""
    for col in H.KAT:
        if col in train:
            cats = pd.unique(train[col].astype(str))
            train[col] = pd.Categorical(train[col].astype(str), categories=cats)
            valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
    return train, valid


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                             min_child_samples=minimum, verbose=-1, n_jobs=args.jobs, random_state=0)


def parts(tr, va):
    z0 = tr.z.eq(0).to_numpy()
    y = tr.y.to_numpy(float)
    q = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                           min_child_samples=20, verbose=-1, n_jobs=args.jobs,
                           random_state=0).fit(tr[BASE], tr.z).predict_proba(va[BASE])[:, 1] ** H.HOCH
    height = ((1-H.ANTEIL_HM)*va.versp_min.to_numpy(float)*60
              + H.ANTEIL_HM*reg(300, 15, 10).fit(tr.loc[~z0, BASE], y[~z0]).predict(va[BASE]))
    normal = reg(400, 31, 20).fit(tr.loc[z0, BASE], y[z0]).predict(va[BASE])
    return q, height, normal


def combine(normal, height, q):
    return np.maximum((1-q)*normal + q*height, 0)


def anchored(tr, va, q, height, normal):
    z0 = tr.z.eq(0).to_numpy()
    y = tr.y.to_numpy(float)
    a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
    rows = z0 & np.isfinite(a_tr)
    offset = reg(400, 31, 20).fit(tr.loc[rows, WITH_NT], y[rows]-a_tr[rows])
    has = np.isfinite(a_va)
    out = normal.copy()
    out[has] = a_va[has] + offset.predict(va.loc[has, WITH_NT])
    return combine(out, height, q)


def permute(frame, seed):
    rng = np.random.default_rng(seed)
    out = frame.copy()
    values = out[NT.MERKMALE].to_numpy(copy=True)
    airports = out.ADEP_mvt.astype(str).to_numpy()
    for airport in np.unique(airports):
        idx = np.flatnonzero(airports == airport)
        values[idx] = values[rng.permutation(idx)]
    out[NT.MERKMALE] = values
    return out


tables, frames, gates, started = cached_tables(), [], [], time.time()
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    train = pd.read_parquet(tables[tuple(sorted(set(range(1, 13))-set(pair)))]).reset_index(drop=True)
    valid = pd.read_parquet(tables[tuple(sorted(pair))]).reset_index(drop=True)
    tr_p, va_p = stand_rate(train, valid)
    tr_r, va_r = research_encode(*stand_rate(train, valid))
    assert np.array_equal(va_p.MVT_ID_mvt.to_numpy(), va_r.MVT_ID_mvt.to_numpy())
    yv = va_p.y.to_numpy(float)
    q_r, h_r, n_r = parts(tr_r, va_r)
    research_v44 = combine(n_r, h_r, q_r)
    gate = {"pair": key, "research_v44_rmse": float(np.sqrt(np.mean((research_v44-yv)**2))),
            "reference_rmse": REFERENCE[key]}
    gate["passed"] = abs(gate["research_v44_rmse"]-gate["reference_rmse"]) <= .01
    gates.append(gate)
    print("GATE", json.dumps(gate), flush=True)
    if not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps(gates, indent=2))
        sys.exit(f"Replication gate failed on {key}")
    q_p, h_p, n_p = parts(tr_p, va_p)
    frame = va_p[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "nt_anchor"]].copy()
    frame["ADEP_mvt"] = va_p.ADEP_mvt.astype(str).to_numpy()
    frame["research_v44"], frame["prod_v44"] = research_v44, combine(n_p, h_p, q_p)
    frame["prod_anchor"] = anchored(tr_p, va_p, q_p, h_p, n_p)
    for seed in SEEDS:
        frame[f"perm_{seed}"] = anchored(permute(tr_p, seed), permute(va_p, seed+1), q_p, h_p, n_p)
    frame["fold"] = key
    frame.to_parquet(output / f"predictions_{key}.parquet", index=False)
    print("PAIR", key, json.dumps({c: float(np.sqrt(np.mean((frame[c]-yv)**2)))
                                   for c in ("research_v44", "prod_v44", "prod_anchor")}), flush=True)
    frames.append(frame)
    del train, valid, tr_p, va_p, tr_r, va_r
    gc.collect()

pooled = pd.concat(frames, ignore_index=True)
y = pooled.y.to_numpy(float)
folds = pooled.fold.to_numpy()
stamps = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
month = np.array([m if str(m) in f.split("_") else m % 12 + 1 for m, f in zip(stamps.dt.month.to_numpy(), folds)])
day = (pooled.ADEP_mvt + "|" + stamps.dt.strftime("%Y-%m-%d")).to_numpy()
_, inverse = np.unique(day, return_inverse=True)
cluster_rows = np.bincount(inverse)
draws = np.random.default_rng(20260913).integers(0, cluster_rows.size, size=(args.bootstrap, cluster_rows.size))
disrupted = np.nan_to_num(pooled.nt_anchor.to_numpy(float), nan=-1.0) >= 1800


def compare(reference, candidate):
    gain = (pooled[reference].to_numpy(float)-y)**2 - (pooled[candidate].to_numpy(float)-y)**2
    order = np.argsort(gain)[::-1]
    keep = np.ones(len(gain), bool)
    keep[order[:100]] = False
    pair_gains = {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}
    boot = np.bincount(inverse, weights=gain)[draws].sum(axis=1)/cluster_rows[draws].sum(axis=1)
    return {"pooled_mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
            "positive_sse_mass": float(gain[gain > 0].sum()), "negative_sse_mass": float(gain[gain < 0].sum()),
            "pair_gains": pair_gains, "pairs_positive": int(sum(g > 0 for g in pair_gains.values())),
            "january_gain_1_7": float(gain[(folds == "1_7") & (month == 1)].mean()),
            "july_gain_1_7": float(gain[(folds == "1_7") & (month == 7)].mean()),
            "disrupted_rows": int(disrupted.sum()),
            "disrupted_net_gain_share": float(gain[disrupted].sum()/gain.sum()) if gain.sum() != 0 else None,
            "airport_day_bootstrap_mse_gain_95pct": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "airport_contributions": {a: float(gain[pooled.ADEP_mvt.to_numpy() == a].sum()/len(gain))
                                      for a in sorted(pooled.ADEP_mvt.unique())}}


anchor = compare("prod_v44", "prod_anchor")
permuted = {seed: compare("prod_v44", f"perm_{seed}")["pooled_mse_gain"] for seed in SEEDS}
checks = {"1_pooled_gain_positive": anchor["pooled_mse_gain"] > 0,
          "2_survives_top100_removal": anchor["mse_gain_without_top100_rows"] > 0,
          "3_at_least_4_of_6_pairs": anchor["pairs_positive"] >= 4,
          "4_january_and_july_gain_in_1_7": anchor["january_gain_1_7"] > 0 and anchor["july_gain_1_7"] > 0,
          "5_beats_every_permutation": all(anchor["pooled_mse_gain"] > g for g in permuted.values())}
report = {"validation": "fallback-only six-pair screen with production categorical encoding; not full ensemble; "
                        "not a leaderboard forecast",
          "replication_gates": gates, "seeds": SEEDS, "code_sha256": CODE,
          "rmse": {c: float(np.sqrt(np.mean((pooled[c].to_numpy(float)-y)**2)))
                   for c in ("research_v44", "prod_v44", "prod_anchor")},
          "prod_anchor_vs_prod_v44": {**anchor, "permuted_pooled_mse_gains": permuted,
                                      "checks": {k: bool(v) for k, v in checks.items()},
                                      "advance": bool(all(checks.values())),
                                      "mechanical_overall_rmse_if_transfer_from_v45_not_forecast":
                                          float(np.sqrt(max(V45_OFFICIAL**2 - (5290/344841)*anchor["pooled_mse_gain"], 0)))},
          "prod_v44_vs_research_v44_descriptive": compare("research_v44", "prod_v44"),
          "seconds": time.time()-started}
(output / "pooled.json").write_text(json.dumps(report, indent=2))
print("SUMMARY", json.dumps({"rmse": report["rmse"],
                             "anchor": {k: report["prod_anchor_vs_prod_v44"][k] for k in
                                        ("pooled_mse_gain", "mse_gain_without_top100_rows", "pairs_positive",
                                         "january_gain_1_7", "july_gain_1_7", "disrupted_net_gain_share",
                                         "airport_day_bootstrap_mse_gain_95pct", "permuted_pooled_mse_gains",
                                         "checks", "advance",
                                         "mechanical_overall_rmse_if_transfer_from_v45_not_forecast")},
                             "prod_vs_research": {k: report["prod_v44_vs_research_v44_descriptive"][k] for k in
                                                  ("pooled_mse_gain", "mse_gain_without_top100_rows", "pairs_positive",
                                                   "january_gain_1_7", "july_gain_1_7",
                                                   "airport_day_bootstrap_mse_gain_95pct")}}, indent=2), flush=True)

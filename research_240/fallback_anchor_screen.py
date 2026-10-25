"""Six-pair fallback screen: neighbour linked-departure taxi as features and as an anchor.

Pre-specified before any fit, 12.09.2026:

  v44_fallback      exact v44/v45 fallback recipe over H.MERKMALE_RUECK (35 inputs)
  nt_features       + NT.MERKMALE as plain inputs to classifier, normal and height model
  nt_anchor_normal  classifier and height model unchanged; the normal regressor predicts
                    y - nt_anchor where the anchor exists, inputs MERKMALE_RUECK + NT;
                    rows without an anchor keep the baseline normal prediction

Combination, the zero floor, HOCH and ANTEIL_HM are unchanged. All rows stay in every
metric; nothing is trimmed or clipped to improve a diagnostic.

Replication gate: before any variant is reported for a pair, the rebuilt v44_fallback
must match research_250/results/fallback_six_pair.json within 0.01 s RMSE. A miss stops
the run.

Decision rule, fixed before fitting. A variant advances to main-ensemble consideration
only if, pooled over all six pairs: (a) MSE gain versus v44_fallback is positive,
(b) the ordinary subset (nt_anchor < 1800 s or missing) loses at most 1 % of its baseline
MSE, and (c) the disrupted subset (nt_anchor >= 1800 s) improves. January/July alone
never promotes.

Fallback-only temporal screen. Not full-ensemble validation, not a leaderboard forecast.
No submission, no upload, no change to research_250 or the frozen v45 release.
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
parser.add_argument("--pairs", default="1,7;2,8;3,9;4,10;5,11;6,12")
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()
project, output = args.project.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import features_geometry as MG
import features_stand_rate as MSQ
from fid_gap_cv import month_of_file
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_240 import features_neighbour_taxi as NT

BASE = list(H.MERKMALE_RUECK)
WITH_NT = BASE + NT.MERKMALE
DISRUPTED_S = 1800.0
V45_OFFICIAL = 269.4753
REFERENCE = json.loads((project / "research_250/results/fallback_six_pair.json").read_text())
REF_RMSE = {f["fold"]: f["baseline_rmse"] for f in REFERENCE["fallback"]["folds"]}
pairs = [tuple(map(int, p.split(","))) for p in args.pairs.split(";")]
assert all(len(p) == 2 and len(set(p)) == 2 and all(1 <= m <= 12 for m in p) for p in pairs)
paths = sorted((project / "data").glob("training_2025-*.parquet"))
assert len(paths) == 12 and {month_of_file(p) for p in paths} == set(range(1, 13))
geometry = MG.alle()
SCREEN = Path(__file__).resolve()
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (SCREEN, Path(NT.__file__).resolve(), project / "main_model.py",
                  project / "features_stand_rate.py")}
KEEP = list(dict.fromkeys(
    [f for f in BASE if f not in MSQ.MERKMALE]
    + ["MVT_ID_mvt", "FLIGHT_ID_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
       "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "y", "monat", "hat_aobt"]))


def fingerprint(selected):
    """Tables depend on the project builders and NT only, not on this screen's reporting."""
    h = hashlib.sha256()
    for p in sorted(project.glob("*.py")) + [Path(NT.__file__).resolve()]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    for p in selected:
        h.update(p.name.encode())
        h.update(str((p.stat().st_size, p.stat().st_mtime_ns)).encode())
    return h.hexdigest()[:16]


def table(selected):
    key = fingerprint(selected)
    cached = output / f"fallback_nt_{key}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    started = time.time()
    months = [month_of_file(p) for p in selected]
    print("BUILD", months, flush=True)
    full = H.baue([str(p) for p in selected], geometry)
    mask = (full.hat_aobt.eq(0) & full.y.notna()).to_numpy()
    nt = NT.baue(full, query=mask)            # neighbours come from every linked departure
    small = full.loc[mask, KEEP].copy()
    for col in NT.MERKMALE:
        small[col] = nt[col].to_numpy()[mask]
    del full, nt
    gc.collect()
    small.to_parquet(cached, index=False)
    (output / f"fallback_nt_{key}.json").write_text(json.dumps(
        {"source_months": months, "source_files": [p.name for p in selected], "rows": len(small),
         "cache_fingerprint": key, "seconds": time.time()-started}, indent=2))
    print("BUILT", len(small), "rows", round(time.time()-started, 1), "s", flush=True)
    return small


def prepare(train, valid):
    """Identical to research_250/fallback250_cv.prepare: stand rate fitted out of month."""
    train, valid = train.copy(), valid.copy()
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(train)
    train = MSQ.ausser_faltung(train)
    valid = MSQ.wende_an(mapping, valid)
    for col in H.KAT:
        if col in train:
            cats = pd.unique(train[col].astype(str))
            train[col] = pd.Categorical(train[col].astype(str), categories=cats)
            valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
    return train, valid


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05,
                             num_leaves=leaves, min_child_samples=minimum, verbose=-1,
                             n_jobs=args.jobs, random_state=0)


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05,
                              num_leaves=31, min_child_samples=20, verbose=-1,
                              n_jobs=args.jobs, random_state=0)


def combine(normal, height, p):
    q = p ** H.HOCH
    return np.maximum((1-q)*normal + q*height, 0)


def summary(y, prediction):
    e = (prediction-y)**2
    order = np.argsort(e)[::-1]
    return {"rows": int(len(y)), "rmse": float(np.sqrt(e.mean())), "mse": float(e.mean()),
            "top5_sse_share": float(e[order[:5]].sum()/e.sum()),
            "rmse_without_5_largest": float(np.sqrt(np.delete(e, order[:5]).mean()))}


def subset(se_base, se_var, mask):
    if not mask.any():
        return {"rows": 0}
    b, v = float(se_base[mask].mean()), float(se_var[mask].mean())
    return {"rows": int(mask.sum()), "baseline_mse": b, "mse": v, "mse_gain": b-v,
            "relative_mse_change": (v-b)/b}


frames, gates, started = [], [], time.time()
for pair in pairs:
    key = f"{pair[0]}_{pair[1]}"
    train = table([p for p in paths if month_of_file(p) not in pair]).reset_index(drop=True)
    valid = table([p for p in paths if month_of_file(p) in pair]).reset_index(drop=True)
    assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
    print("FIT", pair, len(train), len(valid), flush=True)
    tr, va = prepare(train, valid)
    z0 = tr.z.eq(0).to_numpy()
    y = tr.y.to_numpy()
    versp_v = va.versp_min.to_numpy()*60

    def parts(features):
        p = classifier().fit(tr[features], tr.z).predict_proba(va[features])[:, 1]
        height_model = reg(300, 15, 10).fit(tr.loc[~z0, features], y[~z0])
        return p, (1-H.ANTEIL_HM)*versp_v + H.ANTEIL_HM*height_model.predict(va[features])

    p_base, h_base = parts(BASE)
    normal_base = reg(400, 31, 20).fit(tr.loc[z0, BASE], y[z0]).predict(va[BASE])
    predictions = {"v44_fallback": combine(normal_base, h_base, p_base)}

    got = float(np.sqrt(np.mean((predictions["v44_fallback"]-va.y.to_numpy())**2)))
    gate = {"pair": key, "rebuilt_rmse": got, "reference_rmse": REF_RMSE.get(key)}
    gate["passed"] = gate["reference_rmse"] is not None and abs(got-gate["reference_rmse"]) <= .01
    gates.append(gate)
    print("GATE", json.dumps(gate), flush=True)
    if not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps(gates, indent=2))
        sys.exit(f"Replication gate failed on {key}: {got} vs {gate['reference_rmse']}")

    p_nt, h_nt = parts(WITH_NT)
    normal_nt = reg(400, 31, 20).fit(tr.loc[z0, WITH_NT], y[z0]).predict(va[WITH_NT])
    predictions["nt_features"] = combine(normal_nt, h_nt, p_nt)

    anchor_tr, anchor_va = tr.nt_anchor.to_numpy(), va.nt_anchor.to_numpy()
    fit_rows = z0 & np.isfinite(anchor_tr)
    offset = reg(400, 31, 20).fit(tr.loc[fit_rows, WITH_NT], y[fit_rows]-anchor_tr[fit_rows])
    has = np.isfinite(anchor_va)
    normal_anchor = normal_base.copy()
    normal_anchor[has] = anchor_va[has] + offset.predict(va.loc[has, WITH_NT])
    predictions["nt_anchor_normal"] = combine(normal_anchor, h_base, p_base)

    frame = va[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "versp_min", "nt_anchor",
                "nt_med60", "nt_n60"]].copy()
    frame["ADEP_mvt"] = va.ADEP_mvt.astype(str).to_numpy()
    for name, prediction in predictions.items():
        frame[name] = prediction
    frame["fold"] = key
    frame.to_parquet(output / f"predictions_{key}.parquet", index=False)
    yv = va.y.to_numpy()
    disrupted = np.nan_to_num(anchor_va, nan=-1.0) >= DISRUPTED_S
    base_se = (predictions["v44_fallback"]-yv)**2
    result = {}
    for name, prediction in predictions.items():
        se = (prediction-yv)**2
        result[name] = summary(yv, prediction)
        result[name]["mse_gain_vs_v44"] = float(base_se.mean()-se.mean())
        result[name]["disrupted"] = subset(base_se, se, disrupted)
        result[name]["ordinary"] = subset(base_se, se, ~disrupted)
    (output / f"result_{key}.json").write_text(json.dumps(result, indent=2))
    print("RESULT", key, json.dumps({n: {k: result[n][k] for k in ("rmse", "mse_gain_vs_v44")}
                                     for n in result}), flush=True)
    frames.append(frame)
    del train, valid, tr, va
    gc.collect()

pooled = pd.concat(frames, ignore_index=True)
Y = pooled.y.to_numpy()
names = ["v44_fallback", "nt_features", "nt_anchor_normal"]
se = {n: (pooled[n].to_numpy()-Y)**2 for n in names}
disrupted = np.nan_to_num(pooled.nt_anchor.to_numpy(), nan=-1.0) >= DISRUPTED_S
cluster = (pooled.ADEP_mvt + "|" + pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
           .dt.strftime("%Y-%m-%d")).to_numpy()
_, inverse = np.unique(cluster, return_inverse=True)
cluster_rows = np.bincount(inverse)
rng = np.random.default_rng(20260912)
draws = rng.integers(0, cluster_rows.size, size=(args.bootstrap, cluster_rows.size))
complete = len(pairs) == 6
metrics = {}
for n in names:
    m = summary(Y, pooled[n].to_numpy())
    gain = float(se["v44_fallback"].mean()-se[n].mean())
    m["mse_gain_vs_v44"] = gain
    m["mechanical_overall_rmse_if_transfer_from_v45"] = float(
        np.sqrt(max(V45_OFFICIAL**2 - (5290/344841)*gain, 0)))
    m["disrupted"] = subset(se["v44_fallback"], se[n], disrupted)
    m["ordinary"] = subset(se["v44_fallback"], se[n], ~disrupted)
    m["folds"] = []
    for fold, g in pooled.groupby("fold", sort=False):
        idx = g.index.to_numpy()
        m["folds"].append({"fold": fold, "rows": int(idx.size),
                           "mse_gain": float(se["v44_fallback"][idx].mean()-se[n][idx].mean()),
                           "disrupted_rows": int(disrupted[idx].sum())})
    m["positive_folds"] = int(sum(f["mse_gain"] > 0 for f in m["folds"]))
    m["airports"] = [{"airport": a, "rows": int(g.index.size),
                      "mse_gain": float(se["v44_fallback"][g.index].mean()-se[n][g.index].mean())}
                     for a, g in pooled.groupby("ADEP_mvt", sort=True)]
    diff_sums = np.bincount(inverse, weights=se["v44_fallback"]-se[n])
    boot = diff_sums[draws].sum(axis=1)/cluster_rows[draws].sum(axis=1)
    m["airport_day_bootstrap_mse_gain_95pct"] = [float(np.quantile(boot, .025)),
                                                 float(np.quantile(boot, .975))]
    if n != "v44_fallback":
        a = gain > 0
        b = m["ordinary"]["relative_mse_change"] <= .01
        c = m["disrupted"].get("mse_gain", -1) > 0
        m["decision"] = {"a_pooled_gain_positive": bool(a), "b_ordinary_loss_within_1pct": bool(b),
                         "c_disrupted_improves": bool(c), "complete_six_pairs": complete,
                         "advance": bool(a and b and c and complete)}
    metrics[n] = m

report = {"validation": "fallback-only six source-file month-pair screen; not full ensemble; "
                        "mechanical score is NOT a leaderboard forecast",
          "pairs": [f"{a}_{b}" for a, b in pairs], "replication_gates": gates,
          "disrupted_threshold_seconds": DISRUPTED_S, "min_neighbours": NT.MIN_NEIGHBOURS,
          "windows_minutes": list(NT.WINDOWS_MIN), "baseline_features": BASE,
          "candidate_features": WITH_NT, "seed": 0, "jobs": args.jobs,
          "bootstrap_draws": args.bootstrap, "code_sha256": CODE, "metrics": metrics,
          "seconds": time.time()-started}
(output / "pooled.json").write_text(json.dumps(report, indent=2))
print("POOLED", json.dumps({n: {"rmse": metrics[n]["rmse"], "mse_gain": metrics[n]["mse_gain_vs_v44"],
                                "positive_folds": metrics[n]["positive_folds"],
                                "disrupted": metrics[n]["disrupted"],
                                "decision": metrics[n].get("decision")} for n in names}, indent=2),
      flush=True)

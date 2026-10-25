"""Pre-registered six-pair fallback screen: the blend re-tuned under the anchor, and arrival-copy
signals in the extreme classifier.

Pre-specified 13.09.2026, before any fit, after research_240/error_and_drift_diagnostics.py put 51 %
of the January/July squared error in the fallback lane. No leaderboard feedback is used.

Baseline: prod_anchor, the scored v46 fallback recipe (production encoding, anchored normal branch,
HOCH 2.5, ANTEIL_HM 0.3), on the cached production re-screen tables.
Variants:
  T1_blend_loo        HOCH in {1.5, 2, 2.5, 3, 4} x ANTEIL_HM in {0, 0.15, 0.3, 0.5}. For each held-out
                      pair the grid point with the lowest pooled MSE on the other five pairs' held-out
                      predictions is applied (nested selection); the models themselves are unchanged
  A1_copy_classifier  research_240/features_arrival_copy.MERKMALE added to the extreme classifier only;
                      reference time MVT_TIME - nt_anchor, or MVT_TIME - 1,200 s without an anchor
Replication gates: prod_v44 and prod_anchor reproduce the saved predictions of
research_240/fallback_anchor_production.py within 1e-6 s.
Decision rule against prod_anchor, pooled over the six pairs: (1) MSE gain positive, (2) survives
removing its 100 most favourable rows, (3) at least 4 of 6 pairs gain, (4) January and July both gain
inside pair 1_7. A1 must also (5) beat every pooled gain with the added columns permuted jointly within
(split, airport), seeds 101, 202, 303, 404, 505, and have ac_inbound_copied coverage of at least 80 % in
every held-out table and on the 2026 ranking fallback rows. A variant advances only if it passes all of
its checks; the 2026 candidate is the advancing variant with the largest pooled gain. If none advances,
nothing is built or uploaded. Fallback-only screen; not a leaderboard forecast.
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
parser.add_argument("--tables", default=Path(".cache/research_240/rescreen_production"), type=Path)
parser.add_argument("--reference", default=Path(".cache/research_240/fallback_anchor_production"), type=Path)
parser.add_argument("--ranking-tables", default=Path(".cache/research_240/anchor_preview"), type=Path)
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()
project, output = args.project.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
os.chdir(project)


def resolve(path):
    return path if path.is_absolute() else project / path


import main_model as H  # torch before tree libraries on macOS
import features_stand_rate as MSQ
from fid_gap_cv import month_of_file
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_240 import features_arrival_copy as AC
from research_240 import features_neighbour_taxi as NT

BASE = list(H.MERKMALE_RUECK)
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
SEEDS = [101, 202, 303, 404, 505]
GRID = [(h, m) for h in (1.5, 2.0, 2.5, 3.0, 4.0) for m in (0.0, 0.15, 0.3, 0.5)]
assert (H.HOCH, H.ANTEIL_HM) == (2.5, 0.3)
paths = sorted((project / "data").glob("training_2025-*.parquet"))
assert len(paths) == 12 and {month_of_file(p) for p in paths} == set(range(1, 13))
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), project / "main_model.py", project / "features_stand_rate.py",
                  Path(AC.__file__).resolve(), Path(NT.__file__).resolve()]}

index = {}
for sidecar in resolve(args.tables).glob("rescreen_*.json"):
    index[tuple(sorted(json.loads(sidecar.read_text())["source_months"]))] = sidecar.with_suffix(".parquet")


def add_copy(frame, raw_files):
    raw = pd.concat([pd.read_parquet(p, columns=AC.ARRIVAL_COLUMNS) for p in raw_files], ignore_index=True)
    anchor = frame.nt_anchor.to_numpy(float)
    departures = pd.DataFrame({"ADEP_mvt": frame.ADEP_mvt.astype(str).to_numpy(),
                               "STAND_mvt": frame.STAND_mvt.astype(object).to_numpy(),
                               "AIRCRAFT_OPERATOR_flt": frame.AIRCRAFT_OPERATOR_flt.astype(object).to_numpy(),
                               "ersatz_s": np.where(np.isfinite(anchor), anchor, 1200.0)})
    departures["MVT_TIME_UTC_mvt"] = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).reset_index(drop=True)
    extra = AC.features_for(departures, AC.arrivals_table(raw))
    frame = frame.copy()
    for col in AC.MERKMALE:
        frame[col] = extra[col].to_numpy()
    return frame


def table(months):
    key = tuple(sorted(months))
    if key not in index:
        sys.exit(f"No cached fallback table for source months {key}; run fallback_rescreen_production.py first")
    return add_copy(pd.read_parquet(index[key]), [p for p in paths if month_of_file(p) in key])


def stand_rate(train, valid):
    train, valid = train.copy(), valid.copy()
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(train)
    return MSQ.ausser_faltung(train), MSQ.wende_an(mapping, valid)


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                             min_child_samples=minimum, verbose=-1, n_jobs=args.jobs, random_state=0)


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=args.jobs, random_state=0)


def parts(tr, va):
    z0 = tr.z.eq(0).to_numpy()
    y = tr.y.to_numpy(float)
    out = {"p": classifier().fit(tr[BASE], tr.z).predict_proba(va[BASE])[:, 1],
           "h_model": reg(300, 15, 10).fit(tr.loc[~z0, BASE], y[~z0]).predict(va[BASE]),
           "normal": reg(400, 31, 20).fit(tr.loc[z0, BASE], y[z0]).predict(va[BASE]),
           "versp_min": va.versp_min.to_numpy(float)}
    offset_features = BASE + [c for c in NT.MERKMALE if c not in BASE]
    a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
    rows = z0 & np.isfinite(a_tr)
    offset = reg(400, 31, 20).fit(tr.loc[rows, offset_features], y[rows]-a_tr[rows])
    has = np.isfinite(a_va)
    out["anchored"] = out["normal"].copy()
    out["anchored"][has] = a_va[has] + offset.predict(va.loc[has, offset_features])
    return out


def blend(part, hoch, hm, normal_key="anchored", p=None):
    q = (part["p"] if p is None else p) ** hoch
    high = (1-hm)*part["versp_min"]*60 + hm*part["h_model"]
    return np.maximum((1-q)*part[normal_key] + q*high, 0)


def copy_classifier(tr, va):
    features = BASE + AC.MERKMALE
    return classifier().fit(tr[features], tr.z).predict_proba(va[features])[:, 1]


def permute(frame, columns, seed):
    rng = np.random.default_rng(seed)
    out = frame.copy()
    values = out[columns].to_numpy(copy=True)
    airports = out.ADEP_mvt.astype(str).to_numpy()
    for airport in np.unique(airports):
        idx = np.flatnonzero(airports == airport)
        values[idx] = values[rng.permutation(idx)]
    out[columns] = values
    return out


def load_pair(pair):
    train_raw = table([m for m in range(1, 13) if m not in pair]).reset_index(drop=True)
    valid_raw = table(list(pair)).reset_index(drop=True)
    assert set(train_raw.MVT_ID_mvt).isdisjoint(set(valid_raw.MVT_ID_mvt))
    tr, va = stand_rate(train_raw, valid_raw)
    return valid_raw, tr, va


started = time.time()
frames, gates, coverage, stored = [], [], {}, {}
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    valid_raw, tr, va = load_pair(pair)
    part = parts(tr, va)
    frame = va[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y"]].copy()
    frame["ADEP_mvt"] = va.ADEP_mvt.astype(str).to_numpy()
    frame["prod_v44"] = blend(part, 2.5, 0.3, "normal")
    frame["prod_anchor"] = blend(part, 2.5, 0.3)
    saved = pd.read_parquet(resolve(args.reference) / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt]
    gate = {"pair": key, "prod_v44_max_diff": float(np.abs(frame.prod_v44.to_numpy()-saved.prod_v44.to_numpy()).max()),
            "prod_anchor_max_diff": float(np.abs(frame.prod_anchor.to_numpy()-saved.prod_anchor.to_numpy()).max())}
    gate["passed"] = gate["prod_v44_max_diff"] <= 1e-6 and gate["prod_anchor_max_diff"] <= 1e-6
    gates.append(gate)
    coverage[key] = float(np.isfinite(valid_raw.ac_inbound_copied.to_numpy(float)).mean())
    print("GATE", json.dumps(gate), "COPY_COVERAGE", round(coverage[key], 4), flush=True)
    if not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps(gates, indent=2))
        sys.exit(f"Replication gate failed on {key}")
    part["p_copy"] = copy_classifier(tr, va)
    frame["A1_copy_classifier"] = blend(part, 2.5, 0.3, p=part["p_copy"])
    part["y"] = va.y.to_numpy(float)
    stored[key] = part
    frame["fold"] = key
    frames.append(frame)
    print("PAIR", key, f"anchor {np.sqrt(np.mean((frame.prod_anchor-frame.y)**2)):.3f}",
          f"A1 {np.sqrt(np.mean((frame.A1_copy_classifier-frame.y)**2)):.3f}", flush=True)
    del valid_raw, tr, va
    gc.collect()

sse = {key: {g: float(((blend(part, *g) - part["y"])**2).sum()) for g in GRID} for key, part in stored.items()}
selected = {}
for frame in frames:
    key = frame.fold.iloc[0]
    others = [k for k in stored if k != key]
    choice = min(GRID, key=lambda g: sum(sse[k][g] for k in others))
    selected[key] = {"hoch": choice[0], "anteil_hm": choice[1]}
    frame["T1_blend_loo"] = blend(stored[key], *choice)
print("SELECTED", json.dumps(selected), flush=True)

pooled = pd.concat(frames, ignore_index=True)
y = pooled.y.to_numpy(float)
folds = pooled.fold.to_numpy()
stamps = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
month = np.array([m if str(m) in f.split("_") else m % 12 + 1 for m, f in zip(stamps.dt.month.to_numpy(), folds)])
day = (pooled.ADEP_mvt + "|" + stamps.dt.strftime("%Y-%m-%d")).to_numpy()
_, inverse = np.unique(day, return_inverse=True)
cluster_rows = np.bincount(inverse)
draws = np.random.default_rng(20260913).integers(0, cluster_rows.size, size=(args.bootstrap, cluster_rows.size))


def compare(candidate):
    gain = (pooled.prod_anchor.to_numpy(float)-y)**2 - (np.asarray(candidate, float)-y)**2
    order = np.argsort(gain)[::-1]
    keep = np.ones(len(gain), bool)
    keep[order[:100]] = False
    pair_gains = {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}
    boot = np.bincount(inverse, weights=gain)[draws].sum(axis=1)/cluster_rows[draws].sum(axis=1)
    out = {"pooled_mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
           "pair_gains": pair_gains, "pairs_positive": int(sum(g > 0 for g in pair_gains.values())),
           "january_gain_1_7": float(gain[(folds == "1_7") & (month == 1)].mean()),
           "july_gain_1_7": float(gain[(folds == "1_7") & (month == 7)].mean()),
           "airport_day_bootstrap_mse_gain_95pct": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    out["checks"] = {"1_pooled_gain_positive": out["pooled_mse_gain"] > 0,
                     "2_survives_top100_removal": out["mse_gain_without_top100_rows"] > 0,
                     "3_at_least_4_of_6_pairs": out["pairs_positive"] >= 4,
                     "4_january_and_july_gain_in_1_7": out["january_gain_1_7"] > 0 and out["july_gain_1_7"] > 0}
    return out


report = {"validation": "fallback-only six-pair screen against the scored v46 fallback recipe; not a leaderboard forecast",
          "replication_gates": gates, "code_sha256": CODE, "grid": GRID, "selected_per_pair": selected,
          "rmse": {c: float(np.sqrt(np.mean((pooled[c].to_numpy(float)-y)**2)))
                   for c in ("prod_v44", "prod_anchor", "T1_blend_loo", "A1_copy_classifier")},
          "variants": {"T1_blend_loo": compare(pooled.T1_blend_loo), "A1_copy_classifier": compare(pooled.A1_copy_classifier)}}

rank_files = sorted(resolve(args.ranking_tables).glob("fallback_2026_ranking_*.parquet"))
assert len(rank_files) == 1, rank_files
ranking = pd.read_parquet(rank_files[0])
extra_cols = [c for c in ("STAND_mvt", "AIRCRAFT_OPERATOR_flt") if c not in ranking]
if extra_cols:
    ranking = ranking.merge(pd.read_parquet(project / "data/ranking.parquet", columns=["MVT_ID_mvt"] + extra_cols),
                            on="MVT_ID_mvt", how="left", validate="one_to_one")
ranking = add_copy(ranking, [project / "data/ranking.parquet"])
coverage_2026 = float(np.isfinite(ranking.ac_inbound_copied.to_numpy(float)).mean())
copy_coverage_ok = min(coverage.values()) >= .8 and coverage_2026 >= .8
report["copy_coverage"] = {"held_out_2025": coverage, "ranking_2026": coverage_2026, "passed": bool(copy_coverage_ok)}

a1 = report["variants"]["A1_copy_classifier"]
if all(a1["checks"].values()):
    permuted = {seed: [] for seed in SEEDS}
    for pair in PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        valid_raw, tr, va = load_pair(pair)
        rows = folds == key
        assert np.array_equal(pooled.loc[rows, "MVT_ID_mvt"].to_numpy(), va.MVT_ID_mvt.to_numpy())
        for seed in SEEDS:
            p = copy_classifier(permute(tr, AC.MERKMALE, seed), permute(va, AC.MERKMALE, seed+1))
            prediction = blend(stored[key], 2.5, 0.3, p=p)
            permuted[seed].append((pooled.loc[rows, "prod_anchor"].to_numpy(float)-y[rows])**2 - (prediction-y[rows])**2)
        del valid_raw, tr, va
        gc.collect()
    a1["permuted_pooled_mse_gains"] = {seed: float(np.concatenate(v).mean()) for seed, v in permuted.items()}
    a1["checks"]["5_beats_every_permutation"] = all(a1["pooled_mse_gain"] > g for g in a1["permuted_pooled_mse_gains"].values())
    print("PERMUTATION", json.dumps(a1["permuted_pooled_mse_gains"]), flush=True)
report["variants"]["T1_blend_loo"]["advance"] = bool(all(report["variants"]["T1_blend_loo"]["checks"].values()))
a1["advance"] = bool(all(a1["checks"].values()) and "5_beats_every_permutation" in a1["checks"] and copy_coverage_ok)
advancing = [n for n, e in report["variants"].items() if e["advance"]]
report["decision"] = {"advancing": advancing,
                      "candidate_for_2026": max(advancing, key=lambda n: report["variants"][n]["pooled_mse_gain"]) if advancing else None}
report["seconds"] = time.time()-started
(output / "pooled.json").write_text(json.dumps(report, indent=2, default=bool))
print("SUMMARY", json.dumps({"rmse": report["rmse"], "selected": selected, "copy_coverage": report["copy_coverage"],
                             "decision": report["decision"],
                             "variants": {n: {"gain": e["pooled_mse_gain"], "without_top100": e["mse_gain_without_top100_rows"],
                                              "checks": e["checks"], "advance": e["advance"]}
                                          for n, e in report["variants"].items()}}, indent=2, default=bool), flush=True)

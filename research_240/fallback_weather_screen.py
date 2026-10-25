"""Pre-registered six-pair screen: METAR weather and operating direction in the anchored fallback.

Pre-specified 13.09.2026, before any fit, after research_240/main_weather_diagnostic.py put the
main-lane weather bound at 211 total MSE (below its 500 threshold), so only the fallback lane is
screened. No leaderboard feedback is used.

Baseline: prod_anchor, the scored v46 fallback recipe (production categorical encoding,
neighbour-taxi anchored normal branch). Tables: the fallback tables cached by
research_240/fallback_rescreen_production.py, identified by their source months, plus
  weather   (features_weather.WEATHER): latest METAR at or before the runway time, at most 3 h old
  direction (features_weather.DIRECTION): arrival-runway direction of the hour from the same
            source table's arrivals, minutes since it changed, headwind component
Variants:
  W1_weather_offset    weather added to the anchor offset regressor only
  W2_weather_all       weather added to classifier, height, normal and offset
  O1_direction_offset  direction added to the anchor offset regressor only

Replication gates: prod_v44 and prod_anchor must reproduce the saved predictions of
research_240/fallback_anchor_production.py on every pair (max difference 1e-6 s).

Coverage gates (the silent-null join failure another team reported): weather present on at least
95 % and direction on at least 80 % of rows in every held-out table and in the 2026 ranking
fallback rows, and 2026 coverage within 5 points of the pooled 2025 held-out coverage.

Decision rule against prod_anchor, pooled over the six pairs:
  (1) MSE gain positive, (2) survives removing its 100 most favourable rows, (3) at least 4 of 6
  pairs gain, (4) January and July both gain inside pair 1_7, (5) the real pooled gain exceeds
  every pooled gain with the added columns permuted jointly within (split, airport), seeds 101,
  202, 303, 404, 505 (run only for variants passing 1-4).
A variant advances only if it passes (1)-(5) and the coverage gates. If several advance, the
2026 candidate is the one with the largest pooled MSE gain. If none advances, nothing is built
or uploaded. Descriptive: 2025 January/July held-out versus 2026 ranking fallback distributions.
Fallback-only screen; not full ensemble; not a leaderboard forecast.
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
parser.add_argument("--weather", default=Path(".cache/research_240/weather"), type=Path)
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
from research_240 import features_neighbour_taxi as NT
from research_240 import features_weather as W

BASE = list(H.MERKMALE_RUECK)
ADDED = {"W1_weather_offset": W.WEATHER, "W2_weather_all": W.WEATHER, "O1_direction_offset": W.DIRECTION}
VARIANTS = list(ADDED)
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
SEEDS = [101, 202, 303, 404, 505]
paths = sorted((project / "data").glob("training_2025-*.parquet"))
assert len(paths) == 12 and {month_of_file(p) for p in paths} == set(range(1, 13))
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), project / "main_model.py", project / "features_stand_rate.py",
                  Path(W.__file__).resolve(), Path(NT.__file__).resolve()]}
reports = W.report_features(W.load_metar(resolve(args.weather)))

index = {}
for sidecar in resolve(args.tables).glob("rescreen_*.json"):
    meta = json.loads(sidecar.read_text())
    index[tuple(sorted(meta["source_months"]))] = sidecar.with_suffix(".parquet")


def arrivals_from(files):
    raw = pd.concat([pd.read_parquet(p, columns=["PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
                     for p in files], ignore_index=True)
    return raw[raw.PHASE_mvt.astype(str).str.upper().str.startswith("ARR").to_numpy()]


def add_context(frame, arrivals):
    wx = W.weather_for(frame.ADEP_mvt, frame.MVT_TIME_UTC_mvt, reports)
    od = W.direction_for(frame.ADEP_mvt, frame.MVT_TIME_UTC_mvt, W.operating_direction(arrivals),
                         wx.wind_dir_deg, wx.wx_wind_kt)
    frame = frame.copy()
    for col in W.WEATHER:
        frame[col] = wx[col].to_numpy()
    for col in W.DIRECTION:
        frame[col] = od[col].to_numpy()
    return frame


def table(months):
    key = tuple(sorted(months))
    if key not in index:
        sys.exit(f"No cached fallback table for source months {key}; run fallback_rescreen_production.py first")
    return add_context(pd.read_parquet(index[key]), arrivals_from([p for p in paths if month_of_file(p) in key]))


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


def recipe(tr, va, features, anchored=False, offset_extra=()):
    z0 = tr.z.eq(0).to_numpy()
    y = tr.y.to_numpy(float)
    p = classifier().fit(tr[features], tr.z).predict_proba(va[features])[:, 1]
    height = ((1-H.ANTEIL_HM)*va.versp_min.to_numpy(float)*60
              + H.ANTEIL_HM*reg(300, 15, 10).fit(tr.loc[~z0, features], y[~z0]).predict(va[features]))
    normal = reg(400, 31, 20).fit(tr.loc[z0, features], y[z0]).predict(va[features])
    if anchored:
        offset_features = (features + [c for c in NT.MERKMALE if c not in features]
                           + [c for c in offset_extra if c not in features])
        a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
        rows = z0 & np.isfinite(a_tr)
        offset = reg(400, 31, 20).fit(tr.loc[rows, offset_features], y[rows]-a_tr[rows])
        has = np.isfinite(a_va)
        normal = normal.copy()
        normal[has] = a_va[has] + offset.predict(va.loc[has, offset_features])
    q = p ** H.HOCH
    return np.maximum((1-q)*normal + q*height, 0)


def variant(name, tr, va):
    if name == "W2_weather_all":
        return recipe(tr, va, BASE + list(W.WEATHER), anchored=True)
    return recipe(tr, va, BASE, anchored=True, offset_extra=ADDED[name])


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


def coverage(frame):
    return {"weather": float(np.isfinite(frame.wx_temp_c.to_numpy(float)).mean()),
            "direction": float(np.isfinite(frame.od_dir.to_numpy(float)).mean())}


def load_pair(pair):
    train_raw = table([m for m in range(1, 13) if m not in pair]).reset_index(drop=True)
    valid_raw = table(list(pair)).reset_index(drop=True)
    assert set(train_raw.MVT_ID_mvt).isdisjoint(set(valid_raw.MVT_ID_mvt))
    tr, va = stand_rate(train_raw, valid_raw)
    return valid_raw, tr, va


frames, gates, held_out_coverage, started = [], [], {}, time.time()
january_july = None
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    valid_raw, tr, va = load_pair(pair)
    held_out_coverage[key] = coverage(valid_raw)
    if pair == (1, 7):
        january_july = valid_raw[W.WEATHER + W.DIRECTION].copy()
    yv = va.y.to_numpy(float)
    frame = va[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "nt_anchor"]].copy()
    frame["ADEP_mvt"] = va.ADEP_mvt.astype(str).to_numpy()
    frame["prod_v44"] = recipe(tr, va, BASE)
    frame["prod_anchor"] = recipe(tr, va, BASE, anchored=True)
    saved = pd.read_parquet(resolve(args.reference) / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt]
    gate = {"pair": key,
            "prod_v44_max_diff": float(np.abs(frame.prod_v44.to_numpy()-saved.prod_v44.to_numpy()).max()),
            "prod_anchor_max_diff": float(np.abs(frame.prod_anchor.to_numpy()-saved.prod_anchor.to_numpy()).max())}
    gate["passed"] = gate["prod_v44_max_diff"] <= 1e-6 and gate["prod_anchor_max_diff"] <= 1e-6
    gates.append(gate)
    print("GATE", json.dumps(gate), "COVERAGE", json.dumps(held_out_coverage[key]), flush=True)
    if not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps(gates, indent=2))
        sys.exit(f"Replication gate failed on {key}")
    for name in VARIANTS:
        before = time.time()
        frame[name] = variant(name, tr, va)
        print("VARIANT", key, name, f"{np.sqrt(np.mean((frame[name]-yv)**2)):.3f}",
              f"anchor {np.sqrt(np.mean((frame.prod_anchor-yv)**2)):.3f}", f"{time.time()-before:.1f}s", flush=True)
    frame["fold"] = key
    frame.to_parquet(output / f"predictions_{key}.parquet", index=False)
    frames.append(frame)
    del valid_raw, tr, va
    gc.collect()

rank_files = sorted(resolve(args.ranking_tables).glob("fallback_2026_ranking_*.parquet"))
assert len(rank_files) == 1, rank_files
ranking = add_context(pd.read_parquet(rank_files[0]), arrivals_from([project / "data/ranking.parquet"]))
rows_2025 = sum(len(f) for f in frames)
pooled_2025 = {k: sum(held_out_coverage[f"{a}_{b}"][k]*len(fr) for (a, b), fr in zip(PAIRS, frames))/rows_2025
               for k in ("weather", "direction")}
coverage_report = {"held_out_2025": held_out_coverage, "pooled_2025": pooled_2025, "ranking_2026": coverage(ranking),
                   "ranking_2026_rows": int(len(ranking))}
weather_ok = (all(c["weather"] >= .95 for c in held_out_coverage.values()) and coverage_report["ranking_2026"]["weather"] >= .95
              and abs(coverage_report["ranking_2026"]["weather"] - pooled_2025["weather"]) <= .05)
direction_ok = (all(c["direction"] >= .80 for c in held_out_coverage.values()) and coverage_report["ranking_2026"]["direction"] >= .80
                and abs(coverage_report["ranking_2026"]["direction"] - pooled_2025["direction"]) <= .05)
coverage_report["weather_gate_passed"], coverage_report["direction_gate_passed"] = bool(weather_ok), bool(direction_ok)
distribution = {}
for col in W.WEATHER + W.DIRECTION:
    a, b = january_july[col].to_numpy(float), ranking[col].to_numpy(float)
    fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]
    sd = float(np.std(fa)) if fa.size else float("nan")
    distribution[col] = {"mean_2025_jan_jul": float(fa.mean()) if fa.size else None, "mean_2026": float(fb.mean()) if fb.size else None,
                         "standardized_difference": float((fb.mean()-fa.mean())/sd) if fa.size and fb.size and sd > 0 else None,
                         "missing_2025": float(1-fa.size/max(a.size, 1)), "missing_2026": float(1-fb.size/max(b.size, 1))}

pooled = pd.concat(frames, ignore_index=True)
y = pooled.y.to_numpy(float)
folds = pooled.fold.to_numpy()
stamps = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
month = np.array([m if str(m) in f.split("_") else m % 12 + 1 for m, f in zip(stamps.dt.month.to_numpy(), folds)])
day = (pooled.ADEP_mvt + "|" + stamps.dt.strftime("%Y-%m-%d")).to_numpy()
_, inverse = np.unique(day, return_inverse=True)
cluster_rows = np.bincount(inverse)
draws = np.random.default_rng(20260913).integers(0, cluster_rows.size, size=(args.bootstrap, cluster_rows.size))


def compare(reference_col, candidate):
    gain = (pooled[reference_col].to_numpy(float)-y)**2 - (np.asarray(candidate, float)-y)**2
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


report = {"validation": "fallback-only six-pair screen against the scored v46 fallback recipe, production categorical "
                        "encoding; not full ensemble; not a leaderboard forecast",
          "weather_source": "Iowa Environmental Mesonet ASOS/METAR archive (public domain), research_240/fetch_metar.py",
          "replication_gates": gates, "coverage": coverage_report, "distribution_2025_vs_2026": distribution,
          "code_sha256": CODE,
          "rmse": {c: float(np.sqrt(np.mean((pooled[c].to_numpy(float)-y)**2))) for c in ["prod_v44", "prod_anchor"] + VARIANTS},
          "variants": {}}
for name in VARIANTS:
    result = compare("prod_anchor", pooled[name])
    result["passes_1_to_4"] = all(result["checks"].values())
    report["variants"][name] = result

for name in VARIANTS:
    entry = report["variants"][name]
    if not entry["passes_1_to_4"]:
        continue
    columns = list(ADDED[name])
    permuted = {seed: [] for seed in SEEDS}
    for pair in PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        valid_raw, tr, va = load_pair(pair)
        rows = folds == key
        assert np.array_equal(pooled.loc[rows, "MVT_ID_mvt"].to_numpy(), va.MVT_ID_mvt.to_numpy())
        for seed in SEEDS:
            prediction = variant(name, permute(tr, columns, seed), permute(va, columns, seed+1))
            permuted[seed].append((pooled.loc[rows, "prod_anchor"].to_numpy(float)-va.y.to_numpy(float))**2
                                  - (prediction-va.y.to_numpy(float))**2)
        del valid_raw, tr, va
        gc.collect()
    entry["permuted_pooled_mse_gains"] = {seed: float(np.concatenate(v).mean()) for seed, v in permuted.items()}
    entry["checks"]["5_beats_every_permutation"] = all(entry["pooled_mse_gain"] > g for g in entry["permuted_pooled_mse_gains"].values())
    print("PERMUTATION", name, json.dumps(entry["permuted_pooled_mse_gains"]), flush=True)

for name, entry in report["variants"].items():
    gate_ok = weather_ok if name.startswith("W") else direction_ok
    entry["coverage_gate_passed"] = bool(gate_ok)
    entry["advance"] = bool(all(entry["checks"].values()) and "5_beats_every_permutation" in entry["checks"] and gate_ok)
advancing = [n for n, e in report["variants"].items() if e["advance"]]
report["decision"] = {"advancing": advancing,
                      "candidate_for_2026": max(advancing, key=lambda n: report["variants"][n]["pooled_mse_gain"]) if advancing else None}
report["seconds"] = time.time()-started
(output / "pooled.json").write_text(json.dumps(report, indent=2, default=bool))
print("SUMMARY", json.dumps({"rmse": report["rmse"], "coverage": coverage_report, "decision": report["decision"],
                             "variants": {n: {"gain": e["pooled_mse_gain"], "checks": e["checks"], "advance": e["advance"]}
                                          for n, e in report["variants"].items()}}, indent=2, default=bool), flush=True)

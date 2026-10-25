"""Pre-registered six-pair fallback screen: stand order, who else blocked in at the departure stand.

Pre-specified 13.09.2026, before any fit. Companion to main_stand_order_diagnostic.py. The fallback lane has no AOBT_3,
so the stand order is measured against MVT_TIME. BASE already holds um_seit_ank = MVT_TIME minus the in-block time t1 of
the last arrival at the same airport and STAND_mvt at or before MVT_TIME. New columns (SO):
  so_s2m      in-block time t2 of the second-to-last such arrival, minus MVT_TIME
  so_between  departures from the same stand with MVT_TIME strictly between t2 and the departure's MVT_TIME
  so_bound    um_seit_ank where so_between == 0, else missing. With one aircraft per stand, a last arrival that follows
              the arrival before it without any departure in between is most likely the next occupant: the block time
              then lies before t1 and the taxi-out time exceeds MVT_TIME - t1.
Computed separately on every source split from its own raw files. Arrival in-block times and stands are complete in
ranking.parquet, so the columns exist at rank time: the 2026 ranking rows use ranking.parquet, the all-year table all
twelve 2025 files.
Variants against prod_anchor (the scored v46 fallback recipe, production encoding):
  O1_stand_order_offset  SO added to the anchor offset regressor only
  O2_stand_order_all     SO added to classifier, height, normal and offset
Gates:
  A  t1 rebuilt from the raw files reproduces the cached um_seit_ank (same missing pattern, max 1e-6 s) on every table
     used. A failure on a six-pair table stops before any fit; on the all-year or 2026 table it blocks advancing.
     Stand keys follow features_turnaround.py, which runs on raw strings: a missing STAND_mvt is the key 'None'.
     The first run (13.09.2026) stopped at this gate before any fit: its table keys used the categorical 'nan'
     (7 all-year departures without a stand). Keys fixed; nothing else changed.
  B  prod_v44 and prod_anchor reproduce research_240/fallback_anchor_production.py predictions within 1e-6 s.
  C  so_between present on at least 95 % of the rows of every held-out table and of the 2026 ranking fallback rows,
     and the 2026 share within 5 points of the pooled 2025 held-out share.
Decision rule, pooled over six pairs, each variant against prod_anchor: (1) MSE gain positive, (2) positive without its
100 most favourable rows, (3) at least 4 of 6 pairs gain, (4) January and July both gain in 1_7, (5) the real pooled
gain exceeds every pooled gain with SO permuted jointly within (split, airport), seeds 101, 202, 303, 404, 505 (run only
for variants passing 1-4). A variant advances only with (1)-(5), gate C and gate A on both build tables; the candidate
is the advancing variant with the larger pooled gain. Otherwise nothing is built or uploaded.
Fallback-only screen; not full ensemble; not a leaderboard forecast.

Run:
  python3 -B research_240/fallback_stand_order_screen.py --project <repository> --output <folder>
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
from fid_gap_cv import month_of_file
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_240 import features_neighbour_taxi as NT

BASE = list(H.MERKMALE_RUECK)
SO = ["so_s2m", "so_between", "so_bound"]
VARIANTS = ["O1_stand_order_offset", "O2_stand_order_all"]
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
SEEDS = [101, 202, 303, 404, 505]
V46_OFFICIAL, FALLBACK_SHARE_2026 = 268.2394, 5290 / 344841
RAW = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
ORIGIN = pd.Timestamp("2025-01-01", tz="UTC")
TABLES = project / ".cache/research_240/rescreen_production"
SAVED = project / ".cache/research_240/fallback_anchor_production"
PREVIEW = project / ".cache/research_240/anchor_preview"
paths = sorted((project / "data").glob("training_2025-*.parquet"))
assert len(paths) == 12 and {month_of_file(p) for p in paths} == set(range(1, 13))
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), project / "main_model.py", project / "features_stand_rate.py",
                  project / "features_turnaround.py", Path(NT.__file__).resolve()]}
index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet")
         for s in TABLES.glob("rescreen_*.json")}
assert "um_seit_ank" in BASE
started = time.time()


def seconds(values):
    return (pd.to_datetime(values, utc=True) - ORIGIN).dt.total_seconds().to_numpy(float)


def raw_text(values):
    """Strings as features_turnaround.py sees raw parquet values: a missing value becomes 'None'."""
    s = pd.Series(values).astype(object)
    return np.where(s.notna().to_numpy(), s.astype(str).to_numpy(), "None").astype(object)


def events_of(path):
    """Arrival in-block and departure movement times per airport|stand key, from one raw file."""
    raw = pd.read_parquet(path, columns=RAW)
    phase = raw.PHASE_mvt.astype(str).str.upper()
    arr, dep = raw[phase.str.startswith("ARR").to_numpy()], raw[phase.str.startswith("DEP").to_numpy()]
    return (pd.DataFrame({"k": (arr.ADES_mvt.astype(str) + "|" + arr.STAND_mvt.astype(str)).to_numpy(), "t": seconds(arr.BLOCK_TIME_UTC_mvt)}),
            pd.DataFrame({"k": (dep.ADEP_mvt.astype(str) + "|" + dep.STAND_mvt.astype(str)).to_numpy(), "t": seconds(dep.MVT_TIME_UTC_mvt)}))


def sorted_events(parts):
    out = []
    for i in (0, 1):
        frame = pd.concat([p[i] for p in parts], ignore_index=True).dropna().sort_values(["k", "t"], kind="stable")
        out.append({k: g.t.to_numpy() for k, g in frame.groupby("k", sort=False)})
    return out


MONTH_EVENTS = {month_of_file(p): events_of(p) for p in paths}
EVENT_CACHE, ENRICHED, GATE_A = {}, {}, []
print("LOADED EVENTS", f"{time.time()-started:.0f}s", flush=True)


def events(months):
    key = tuple(sorted(months))
    if key not in EVENT_CACHE:
        EVENT_CACHE[key] = sorted_events([MONTH_EVENTS[m] for m in key])
    return EVENT_CACHE[key]


def stand_order(table, split_events):
    arrivals, departures = split_events
    keys = (pd.Series(raw_text(table.ADEP_mvt)) + "|" + pd.Series(raw_text(table.STAND_mvt))).to_numpy()
    mvt = seconds(table.MVT_TIME_UTC_mvt)
    n = len(table)
    t1, t2, between = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for key, idx in pd.Series(np.arange(n)).groupby(keys).indices.items():
        a = arrivals.get(key)
        if a is None:
            continue
        t = mvt[idx]
        pos = np.searchsorted(a, t, side="right")
        t1[idx] = np.where(pos >= 1, a[np.maximum(pos - 1, 0)], np.nan)
        second = np.where(pos >= 2, a[np.maximum(pos - 2, 0)], np.nan)
        t2[idx] = second
        d = departures.get(key, np.array([]))
        lo = np.searchsorted(d, np.nan_to_num(second, nan=np.inf), side="right")
        hi = np.searchsorted(d, t, side="left")
        between[idx] = np.where(pos >= 2, hi - lo, np.nan)
    since = mvt - t1
    return pd.DataFrame({"um_seit_ank_rebuilt": since, "so_s2m": t2 - mvt, "so_between": between,
                         "so_bound": np.where(between == 0, since, np.nan)})


def enrich(table, split_events, label, stop=True):
    cols = stand_order(table, split_events)
    a, b = table.um_seit_ank.to_numpy(float), cols.um_seit_ank_rebuilt.to_numpy(float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    gate = {"table": label, "rows": int(len(table)), "missing_pattern_equal": bool(np.array_equal(fa, fb)),
            "max_abs_diff_s": float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0}
    gate["passed"] = gate["missing_pattern_equal"] and gate["max_abs_diff_s"] <= 1e-6
    GATE_A.append(gate)
    print("GATE_A", json.dumps(gate), flush=True)
    if stop and not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps({"gate_a": GATE_A}, indent=2))
        sys.exit(f"Gate A failed on {label}: rebuilt t1 does not reproduce um_seit_ank")
    out = table.copy()
    for c in SO:
        out[c] = cols[c].to_numpy()
    return out, gate


def enriched(months):
    key = tuple(sorted(months))
    if key not in ENRICHED:
        ENRICHED[key] = enrich(pd.read_parquet(index[key]).reset_index(drop=True), events(key), f"months {list(key)}")[0]
    return ENRICHED[key]


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
    """research_240/fallback_weather_screen.py recipe, unchanged."""
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
    if name == "O1_stand_order_offset":
        return recipe(tr, va, BASE, anchored=True, offset_extra=SO)
    if name == "O2_stand_order_all":
        return recipe(tr, va, BASE + SO, anchored=True)
    raise KeyError(name)


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
    learn = [m for m in range(1, 13) if m not in pair]
    train_raw, valid_raw = enriched(learn), enriched(list(pair))
    assert set(train_raw.MVT_ID_mvt).isdisjoint(set(valid_raw.MVT_ID_mvt))
    tr, va = stand_rate(train_raw, valid_raw)
    return valid_raw, tr, va


def jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


# Build tables first: gate A there blocks advancing but not the screen.
preview_all = sorted(PREVIEW.glob("fallback_2025_all_*.parquet"))
preview_rank = sorted(PREVIEW.glob("fallback_2026_ranking_*.parquet"))
assert len(preview_all) == 1 and len(preview_rank) == 1, (preview_all, preview_rank)
table_all, gate_all = enrich(pd.read_parquet(preview_all[0]).reset_index(drop=True), events(range(1, 13)), "2025 all months", stop=False)
table_2026, gate_2026 = enrich(pd.read_parquet(preview_rank[0]).reset_index(drop=True),
                               sorted_events([events_of(project / "data/ranking.parquet")]), "2026 ranking", stop=False)
table_all.to_parquet(output / "stand_order_table_2025_all.parquet", index=False)
table_2026.to_parquet(output / "stand_order_table_2026_ranking.parquet", index=False)
build_gates_ok = bool(gate_all["passed"] and gate_2026["passed"])
print("BUILD TABLES", len(table_all), len(table_2026), "gates", build_gates_ok, f"{time.time()-started:.0f}s", flush=True)

frames, gates_b, held_coverage = [], [], {}
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    before = time.time()
    valid_raw, tr, va = load_pair(pair)
    held_coverage[key] = {"rows": int(len(valid_raw)), "so_between": float(np.isfinite(valid_raw.so_between.to_numpy(float)).mean()),
                          "so_bound": float(np.isfinite(valid_raw.so_bound.to_numpy(float)).mean())}
    yv = va.y.to_numpy(float)
    frame = va[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "nt_anchor", "um_seit_ank"] + SO].copy()
    frame["ADEP_mvt"] = va.ADEP_mvt.astype(str).to_numpy()
    frame["prod_v44"] = recipe(tr, va, BASE)
    frame["prod_anchor"] = recipe(tr, va, BASE, anchored=True)
    saved = pd.read_parquet(SAVED / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt]
    gate = {"pair": key,
            "prod_v44_max_diff": float(np.abs(frame.prod_v44.to_numpy()-saved.prod_v44.to_numpy()).max()),
            "prod_anchor_max_diff": float(np.abs(frame.prod_anchor.to_numpy()-saved.prod_anchor.to_numpy()).max())}
    gate["passed"] = gate["prod_v44_max_diff"] <= 1e-6 and gate["prod_anchor_max_diff"] <= 1e-6
    gates_b.append(gate)
    print("GATE_B", json.dumps(gate), "COVERAGE", json.dumps(held_coverage[key]), flush=True)
    if not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps({"gate_a": GATE_A, "gate_b": gates_b}, indent=2))
        sys.exit(f"Gate B failed on {key}")
    for name in VARIANTS:
        frame[name] = variant(name, tr, va)
    print("PAIR", key, json.dumps({c: round(float(np.sqrt(np.mean((frame[c]-yv)**2))), 3)
                                   for c in ["prod_v44", "prod_anchor"] + VARIANTS}), f"{time.time()-before:.0f}s", flush=True)
    frame["fold"] = key
    frame.to_parquet(output / f"predictions_{key}.parquet", index=False)
    frames.append(frame)
    del valid_raw, tr, va
    gc.collect()

rows_2025 = sum(c["rows"] for c in held_coverage.values())
pooled_between = sum(c["so_between"]*c["rows"] for c in held_coverage.values()) / rows_2025
coverage_2026 = {"rows": int(len(table_2026)), "so_between": float(np.isfinite(table_2026.so_between.to_numpy(float)).mean()),
                 "so_bound": float(np.isfinite(table_2026.so_bound.to_numpy(float)).mean())}
coverage_ok = bool(all(c["so_between"] >= .95 for c in held_coverage.values()) and coverage_2026["so_between"] >= .95
                   and abs(coverage_2026["so_between"] - pooled_between) <= .05)

pooled = pd.concat(frames, ignore_index=True)
y = pooled.y.to_numpy(float)
folds = pooled.fold.to_numpy()
stamps = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
month = np.array([m if str(m) in f.split("_") else m % 12 + 1 for m, f in zip(stamps.dt.month.to_numpy(), folds)])
_, inverse = np.unique((pooled.ADEP_mvt + "|" + stamps.dt.strftime("%Y-%m-%d")).to_numpy(), return_inverse=True)
cluster_rows = np.bincount(inverse)
draws = np.random.default_rng(20260913).integers(0, cluster_rows.size, size=(args.bootstrap, cluster_rows.size))
disrupted = np.nan_to_num(pooled.nt_anchor.to_numpy(float), nan=-1.0) >= 1800


def compare(candidate):
    gain = (pooled.prod_anchor.to_numpy(float)-y)**2 - (np.asarray(candidate, float)-y)**2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    pair_gains = {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}
    boot = np.bincount(inverse, weights=gain)[draws].sum(axis=1)/cluster_rows[draws].sum(axis=1)
    out = {"pooled_mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
           "positive_sse_mass": float(gain[gain > 0].sum()), "negative_sse_mass": float(gain[gain < 0].sum()),
           "pair_gains": pair_gains, "pairs_positive": int(sum(g > 0 for g in pair_gains.values())),
           "january_gain_1_7": float(gain[(folds == "1_7") & (month == 1)].mean()),
           "july_gain_1_7": float(gain[(folds == "1_7") & (month == 7)].mean()),
           "disrupted_rows": int(disrupted.sum()),
           "disrupted_net_gain_share": float(gain[disrupted].sum()/gain.sum()) if gain.sum() != 0 else None,
           "airport_day_bootstrap_mse_gain_95pct": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
           "airport_contributions": {a: float(gain[pooled.ADEP_mvt.to_numpy() == a].sum()/len(gain))
                                     for a in sorted(pooled.ADEP_mvt.unique())},
           "mechanical_overall_rmse_if_full_transfer_not_forecast":
               float(np.sqrt(max(V46_OFFICIAL**2 - FALLBACK_SHARE_2026*float(gain.mean()), 0)))}
    out["checks"] = {"1_pooled_gain_positive": out["pooled_mse_gain"] > 0,
                     "2_survives_top100_removal": out["mse_gain_without_top100_rows"] > 0,
                     "3_at_least_4_of_6_pairs": out["pairs_positive"] >= 4,
                     "4_january_and_july_gain_in_1_7": out["january_gain_1_7"] > 0 and out["july_gain_1_7"] > 0}
    return out


bound = pooled.so_bound.to_numpy(float)
has_bound = np.isfinite(bound)
report = {"validation": "fallback-only six-pair screen against the scored v46 fallback recipe, production categorical "
                        "encoding; not full ensemble; not a leaderboard forecast",
          "gate_a_rebuilt_t1": GATE_A, "gate_b_replication": gates_b,
          "gate_c_coverage": {"held_out_2025": held_coverage, "pooled_2025_so_between": pooled_between,
                              "ranking_2026": coverage_2026, "passed": coverage_ok},
          "build_tables_gate_a_passed": build_gates_ok, "code_sha256": CODE,
          "descriptive_held_out": {"rows_with_bound": int(has_bound.sum()),
                                   "share_y_above_bound": float(np.mean(y[has_bound] > bound[has_bound])) if has_bound.any() else None,
                                   "share_nt_anchor_below_bound": float(np.mean(pooled.nt_anchor.to_numpy(float)[has_bound] < bound[has_bound])) if has_bound.any() else None},
          "rmse": {c: float(np.sqrt(np.mean((pooled[c].to_numpy(float)-y)**2))) for c in ["prod_v44", "prod_anchor"] + VARIANTS},
          "variants": {name: compare(pooled[name]) for name in VARIANTS}}
print("SCREEN", json.dumps({n: {"gain": e["pooled_mse_gain"], "without_top100": e["mse_gain_without_top100_rows"],
                                "pairs": e["pairs_positive"], "jan": e["january_gain_1_7"], "jul": e["july_gain_1_7"],
                                "checks": e["checks"]} for n, e in report["variants"].items()}, default=jsonable), flush=True)

for name in VARIANTS:
    entry = report["variants"][name]
    if not all(entry["checks"].values()):
        continue
    permuted = {seed: [] for seed in SEEDS}
    for pair in PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        valid_raw, tr, va = load_pair(pair)
        rows = folds == key
        assert np.array_equal(pooled.loc[rows, "MVT_ID_mvt"].to_numpy(), va.MVT_ID_mvt.to_numpy())
        yv = va.y.to_numpy(float)
        base_error = (pooled.loc[rows, "prod_anchor"].to_numpy(float)-yv)**2
        for seed in SEEDS:
            prediction = variant(name, permute(tr, SO, seed), permute(va, SO, seed+1))
            permuted[seed].append(base_error - (prediction-yv)**2)
        del valid_raw, tr, va
        gc.collect()
    entry["permuted_pooled_mse_gains"] = {str(seed): float(np.concatenate(v).mean()) for seed, v in permuted.items()}
    entry["checks"]["5_beats_every_permutation"] = all(entry["pooled_mse_gain"] > g for g in entry["permuted_pooled_mse_gains"].values())
    print("PERMUTATION", name, json.dumps(entry["permuted_pooled_mse_gains"]), flush=True)

for entry in report["variants"].values():
    entry["advance"] = bool(all(entry["checks"].values()) and "5_beats_every_permutation" in entry["checks"]
                            and coverage_ok and build_gates_ok)
advancing = [n for n, e in report["variants"].items() if e["advance"]]
report["decision"] = {"advancing": advancing,
                      "candidate_for_2026": max(advancing, key=lambda n: report["variants"][n]["pooled_mse_gain"]) if advancing else None}
report["seconds"] = time.time()-started
(output / "pooled.json").write_text(json.dumps(report, indent=2, default=jsonable))
print("SUMMARY", json.dumps({"rmse": report["rmse"], "gate_c": report["gate_c_coverage"], "build_gates": build_gates_ok,
                             "descriptive": report["descriptive_held_out"], "decision": report["decision"],
                             "variants": {n: {"gain": e["pooled_mse_gain"], "without_top100": e["mse_gain_without_top100_rows"],
                                              "pairs": e["pairs_positive"], "jan": e["january_gain_1_7"], "jul": e["july_gain_1_7"],
                                              "bootstrap": e["airport_day_bootstrap_mse_gain_95pct"],
                                              "mechanical_rmse": e["mechanical_overall_rmse_if_full_transfer_not_forecast"],
                                              "airports": {a: round(v) for a, v in e["airport_contributions"].items()},
                                              "permuted": e.get("permuted_pooled_mse_gains"), "checks": e["checks"],
                                              "advance": e["advance"]} for n, e in report["variants"].items()}},
                            indent=2, default=jsonable), flush=True)

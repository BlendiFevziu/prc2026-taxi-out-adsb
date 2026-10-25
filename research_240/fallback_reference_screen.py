"""Pre-registered six-pair fallback screen: stage references and a stage-adjusted neighbour anchor.

Pre-specified 13.09.2026, before any fit, as the follow-up to the phase decomposition and the stage models
(research_240/phase_decomposition.py, research_240/stage_models.py). No leaderboard feedback. Run outside the
repository while another agent worked in it, then moved into research_240/.

A fallback departure has no AOBT_3, so its own NM taxi time p = MVT_TIME - AOBT_3 is unknown. Stage view:
  reference  stand-to-take-off time without the day's queue, from linked departures with 0 < p <= 120 min:
             10th percentile and median of p per airport, stand and runway when at least 10 flights lie at
             or below the 10th percentile (EUROCONTROL rule), else per airport and runway, else per airport.
  queue      the neighbour excess: median of p - reference median over the SAME linked neighbours that
             research_240/features_neighbour_taxi.py uses for nt_anchor (same airport, +/-60 min with at
             least 5, else +/-180 min with at least 5). nt_anchor mixes neighbours at near and far stands;
             stage anchor = own reference median + neighbour excess removes that mix.
Columns added to every fallback table; references always come from the learning months of the pair (for the
2026 ranking rows and the all-year training table: all twelve 2025 months, as a final fit applies them):
  REFERENCE  ref_p10, ref_p50, ref_log_n (log 1 + flights in the group used), ref_level (2 stand, 1 runway, 0 airport)
  STAGE      sa_excess60, sa_excess180 (neighbour excess medians), sa_excess (chosen by the nt_anchor rule)
Variants, each against prod_anchor (the scored v46 fallback recipe, production encoding):
  R1_reference_offset  REFERENCE added to the anchor offset regressor only
  R2_reference_all     REFERENCE added to classifier, height, normal and offset
  S1_stage_anchor      anchor = ref_p50 + sa_excess (nt_anchor where undefined); offset regressor on
                       BASE + nt_ columns + REFERENCE + STAGE
Gates:
  A  rebuilt neighbours: nt_med60, nt_med180, nt_n60, nt_n180 and nt_anchor recomputed from the raw files equal
     the cached columns exactly (same missing pattern, max difference 1e-6). Failing on a six-pair table stops
     the screen before any fit; failing on the all-year or 2026 table blocks advancing.
  B  prod_v44 and prod_anchor reproduce research_240/fallback_anchor_production.py predictions within 1e-6 s.
  C  coverage: stand or runway reference (ref_level >= 1) on at least 95 % of the rows of every held-out table
     and of the 2026 ranking fallback rows, and the 2026 stand-level share within 5 points of the pooled 2025
     held-out share.
Decision rule, pooled over six pairs, each variant against prod_anchor:
  (1) MSE gain positive, (2) survives removing its 100 most favourable rows, (3) at least 4 of 6 pairs gain,
  (4) January and July both gain inside pair 1_7, (5) the real pooled gain exceeds every pooled gain with
  REFERENCE permuted jointly within (split, airport), seeds 101, 202, 303, 404, 505 (for S1 this moves the
  stand part of the anchor); S1 must also exceed every pooled gain with REFERENCE and STAGE permuted jointly.
  (5) runs only for variants passing (1)-(4).
A variant advances only if it passes (1)-(5), gate C and the all-year and 2026 gate A. If several advance,
the 2026 candidate is the one with the largest pooled MSE gain. If none advances, nothing is built or
uploaded. Fallback-only screen; not full ensemble; not a leaderboard forecast.

Run:
  python3 -B research_240/fallback_reference_screen.py --project <repository> --output <folder>
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
REFERENCE = ["ref_p10", "ref_p50", "ref_log_n", "ref_level"]
STAGE = ["sa_excess60", "sa_excess180", "sa_excess"]
REBUILT = ["nt_med60", "nt_med180", "nt_n60", "nt_n180", "nt_anchor"]
VARIANTS = ["R1_reference_offset", "R2_reference_all", "S1_stage_anchor"]
CONTROLS = {"R1_reference_offset": {"REFERENCE": REFERENCE}, "R2_reference_all": {"REFERENCE": REFERENCE},
            "S1_stage_anchor": {"REFERENCE": REFERENCE, "REFERENCE+STAGE": REFERENCE + STAGE}}
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
SEEDS = [101, 202, 303, 404, 505]
LEVELS = (("stand", ["airport", "stand", "runway"], 2), ("runway", ["airport", "runway"], 1), ("airport", ["airport"], 0))
MIN_BELOW = 10
V46_OFFICIAL, FALLBACK_SHARE_2026 = 268.2394, 5290 / 344841
RAW = ["PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
TABLES = project / ".cache/research_240/rescreen_production"
SAVED = project / ".cache/research_240/fallback_anchor_production"
PREVIEW = project / ".cache/research_240/anchor_preview"
paths = sorted((project / "data").glob("training_2025-*.parquet"))
assert len(paths) == 12 and {month_of_file(p) for p in paths} == set(range(1, 13))
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), project / "main_model.py", project / "features_stand_rate.py",
                  Path(NT.__file__).resolve()]}
index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet")
         for s in TABLES.glob("rescreen_*.json")}
started = time.time()


def text(values):
    s = pd.Series(values).reset_index(drop=True)
    return np.where(s.notna().to_numpy(), s.astype(str).to_numpy(), "").astype(object)


def departures(path):
    """All departures of one raw file, as H.baue keeps them, with p = MVT_TIME - AOBT_3 in seconds."""
    raw = pd.read_parquet(path, columns=RAW)
    dep = raw[raw.PHASE_mvt.astype(str).str.upper().str.startswith("DEP").to_numpy()].reset_index(drop=True)
    mvt = pd.to_datetime(dep.MVT_TIME_UTC_mvt, errors="coerce")
    aobt = pd.to_datetime(dep.AOBT_3_flt, errors="coerce")
    t, t_ok = NT._nanoseconds(mvt)
    return pd.DataFrame({"airport": dep.ADEP_mvt.astype(str).to_numpy(), "stand": text(dep.STAND_mvt),
                         "runway": text(dep.RUNWAY_mvt), "t": t, "t_ok": t_ok,
                         "p": (mvt - aobt).dt.total_seconds().to_numpy(float), "linked": aobt.notna().to_numpy()})


DEPS = {month_of_file(p): departures(p) for p in paths}
REF_CACHE, ENRICHED, GATE_A = {}, {}, []
print("LOADED", {m: len(d) for m, d in sorted(DEPS.items())}, f"{time.time()-started:.0f}s", flush=True)


def reference_tables(deps):
    p = deps.p.to_numpy()
    base = deps[deps.linked.to_numpy() & (p > 0) & (p <= 7200)]
    tables = {}
    for name, keys, _ in LEVELS:
        data = base
        for k in keys[1:]:
            data = data[data[k].to_numpy() != ""]
        grouped = data.groupby(keys, sort=False).p
        table = pd.DataFrame({"p10": grouped.quantile(.1), "p50": grouped.median(), "n": grouped.size()})
        if name == "stand":
            joined = data.join(table.p10.rename("q"), on=keys)
            below = joined[joined.p.to_numpy() <= joined.q.to_numpy()].groupby(keys, sort=False).size()
            table = table[below.reindex(table.index).fillna(0).to_numpy() >= MIN_BELOW]
        tables[name] = table
    return tables


def refs(months):
    key = tuple(sorted(months))
    if key not in REF_CACHE:
        REF_CACHE[key] = reference_tables(pd.concat([DEPS[m] for m in key], ignore_index=True))
    return REF_CACHE[key]


def lookup(airport, stand, runway, tables):
    keys = pd.DataFrame({"airport": np.asarray(airport, object), "stand": np.asarray(stand, object),
                         "runway": np.asarray(runway, object)})
    out = np.full((len(keys), 4), np.nan)
    for name, cols, level in LEVELS:
        joined = keys.join(tables[name], on=cols)
        fill = np.isnan(out[:, 1]) & joined.p50.notna().to_numpy()
        out[fill, 0] = joined.p10.to_numpy(float)[fill]
        out[fill, 1] = joined.p50.to_numpy(float)[fill]
        out[fill, 2] = np.log1p(joined.n.to_numpy(float)[fill])
        out[fill, 3] = level
    return pd.DataFrame(out, columns=REFERENCE)


def stage_columns(table, deps, tables):
    """Rebuild the nt_ neighbour columns of NT.baue and add reference and neighbour-excess columns."""
    ref = lookup(table.ADEP_mvt.astype(str).to_numpy(), text(table.STAND_mvt), text(table.RUNWAY_mvt), tables)
    keep = deps.linked.to_numpy() & deps.t_ok.to_numpy() & np.isfinite(deps.p.to_numpy())
    src = deps.loc[keep].reset_index(drop=True)
    src_ref = lookup(src.airport.to_numpy(), src.stand.to_numpy(), src.runway.to_numpy(), tables).ref_p50.to_numpy()
    src_t, src_p, src_airport = src.t.to_numpy(), src.p.to_numpy(), src.airport.to_numpy()
    src_e = src_p - src_ref
    t, t_ok = NT._nanoseconds(table.MVT_TIME_UTC_mvt)
    airports = table.ADEP_mvt.astype(str).to_numpy()
    out = {c: np.full(len(table), np.nan) for c in ("nt_med60", "nt_med180", "nt_n60", "nt_n180", "sa_excess60", "sa_excess180")}
    for airport in np.unique(airports[t_ok]):
        qry = np.flatnonzero(t_ok & (airports == airport))
        at = np.flatnonzero(src_airport == airport)
        at = at[np.argsort(src_t[at], kind="stable")]
        at_t, at_p, at_e = src_t[at], src_p[at], src_e[at]
        for minutes in (60, 180):
            radius = np.int64(minutes) * 60_000_000_000
            left = np.searchsorted(at_t, t[qry] - radius, side="left")
            right = np.searchsorted(at_t, t[qry] + radius, side="right")
            out[f"nt_n{minutes}"][qry] = right - left
            for k, row in enumerate(qry):
                if right[k] > left[k]:
                    out[f"nt_med{minutes}"][row] = np.median(at_p[left[k]:right[k]])
                    out[f"sa_excess{minutes}"][row] = np.median(at_e[left[k]:right[k]])
    use60 = np.nan_to_num(out["nt_n60"], nan=-1) >= NT.MIN_NEIGHBOURS
    use180 = ~use60 & (np.nan_to_num(out["nt_n180"], nan=-1) >= NT.MIN_NEIGHBOURS)
    out["nt_anchor"] = np.where(use60, out["nt_med60"], np.where(use180, out["nt_med180"], np.nan))
    out["sa_excess"] = np.where(use60, out["sa_excess60"], np.where(use180, out["sa_excess180"], np.nan))
    frame = pd.DataFrame(out)
    for c in REFERENCE:
        frame[c] = ref[c].to_numpy()
    return frame


def enrich(table, deps, tables, label, stop=True):
    cols = stage_columns(table, deps, tables)
    gate = {"table": label, "rows": int(len(table))}
    for c in REBUILT:
        a, b = table[c].to_numpy(float), cols[c].to_numpy(float)
        fa, fb = np.isfinite(a), np.isfinite(b)
        both = fa & fb
        gate[c] = {"missing_pattern_equal": bool(np.array_equal(fa, fb)),
                   "max_abs_diff": float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0}
    gate["passed"] = all(gate[c]["missing_pattern_equal"] and gate[c]["max_abs_diff"] <= 1e-6 for c in REBUILT)
    GATE_A.append(gate)
    print("GATE_A", json.dumps(gate), flush=True)
    if stop and not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps({"gate_a": GATE_A}, indent=2))
        sys.exit(f"Gate A failed on {label}: rebuilt neighbour columns differ from the cached table")
    out = table.copy()
    for c in REFERENCE + STAGE:
        out[c] = cols[c].to_numpy()
    return out, gate


def enriched(months, reference_months):
    key = (tuple(sorted(months)), tuple(sorted(reference_months)))
    if key not in ENRICHED:
        deps = pd.concat([DEPS[m] for m in key[0]], ignore_index=True)
        ENRICHED[key] = enrich(pd.read_parquet(index[key[0]]).reset_index(drop=True), deps, refs(key[1]),
                               f"months {list(key[0])}, references {list(key[1])}")[0]
        del deps
        gc.collect()
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


def anchor_of(frame, stage):
    nt = frame.nt_anchor.to_numpy(float)
    if not stage:
        return nt
    sa = frame.ref_p50.to_numpy(float) + frame.sa_excess.to_numpy(float)
    return np.where(np.isfinite(sa), sa, nt)


def recipe(tr, va, features, anchored=False, offset_extra=(), stage_anchor=False):
    """research_240/fallback_weather_screen.py recipe; stage_anchor swaps the anchor only."""
    z0 = tr.z.eq(0).to_numpy()
    y = tr.y.to_numpy(float)
    p = classifier().fit(tr[features], tr.z).predict_proba(va[features])[:, 1]
    height = ((1-H.ANTEIL_HM)*va.versp_min.to_numpy(float)*60
              + H.ANTEIL_HM*reg(300, 15, 10).fit(tr.loc[~z0, features], y[~z0]).predict(va[features]))
    normal = reg(400, 31, 20).fit(tr.loc[z0, features], y[z0]).predict(va[features])
    if anchored:
        offset_features = (features + [c for c in NT.MERKMALE if c not in features]
                           + [c for c in offset_extra if c not in features])
        a_tr, a_va = anchor_of(tr, stage_anchor), anchor_of(va, stage_anchor)
        rows = z0 & np.isfinite(a_tr)
        offset = reg(400, 31, 20).fit(tr.loc[rows, offset_features], y[rows]-a_tr[rows])
        has = np.isfinite(a_va)
        normal = normal.copy()
        normal[has] = a_va[has] + offset.predict(va.loc[has, offset_features])
    q = p ** H.HOCH
    return np.maximum((1-q)*normal + q*height, 0)


def variant(name, tr, va):
    if name == "R1_reference_offset":
        return recipe(tr, va, BASE, anchored=True, offset_extra=REFERENCE)
    if name == "R2_reference_all":
        return recipe(tr, va, BASE + REFERENCE, anchored=True)
    if name == "S1_stage_anchor":
        return recipe(tr, va, BASE, anchored=True, offset_extra=REFERENCE + STAGE, stage_anchor=True)
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
    train_raw, valid_raw = enriched(learn, learn), enriched(list(pair), learn)
    assert set(train_raw.MVT_ID_mvt).isdisjoint(set(valid_raw.MVT_ID_mvt))
    tr, va = stand_rate(train_raw, valid_raw)
    return valid_raw, tr, va


def jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


# All-year 2025 and 2026 ranking tables, for a later build; gate A here blocks advancing but not the screen.
preview_all = sorted(PREVIEW.glob("fallback_2025_all_*.parquet"))
preview_rank = sorted(PREVIEW.glob("fallback_2026_ranking_*.parquet"))
assert len(preview_all) == 1 and len(preview_rank) == 1, (preview_all, preview_rank)
ALL = list(range(1, 13))
deps_all = pd.concat([DEPS[m] for m in ALL], ignore_index=True)
table_all, gate_all = enrich(pd.read_parquet(preview_all[0]).reset_index(drop=True), deps_all, refs(ALL),
                             "2025 all months", stop=False)
del deps_all
gc.collect()
table_2026, gate_2026 = enrich(pd.read_parquet(preview_rank[0]).reset_index(drop=True),
                               departures(project / "data/ranking.parquet"), refs(ALL), "2026 ranking", stop=False)
table_all.to_parquet(output / "stage_table_2025_all.parquet", index=False)
table_2026.to_parquet(output / "stage_table_2026_ranking.parquet", index=False)
build_gates_ok = bool(gate_all["passed"] and gate_2026["passed"])
print("PREVIEW TABLES", f"all-year {len(table_all)}", f"2026 {len(table_2026)}", "gates", build_gates_ok,
      f"{time.time()-started:.0f}s", flush=True)

frames, gates_b, held_coverage = [], [], {}
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    before = time.time()
    valid_raw, tr, va = load_pair(pair)
    held_coverage[key] = {"rows": int(len(valid_raw)), "stand_or_runway": float((valid_raw.ref_level >= 1).mean()),
                          "stand": float((valid_raw.ref_level == 2).mean())}
    print("TABLES", key, json.dumps(held_coverage[key]), f"{time.time()-before:.0f}s", flush=True)
    yv = va.y.to_numpy(float)
    frame = va[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "nt_anchor", "strecke_m", "ref_p50", "ref_level", "sa_excess"]].copy()
    frame["ADEP_mvt"] = va.ADEP_mvt.astype(str).to_numpy()
    frame["prod_v44"] = recipe(tr, va, BASE)
    frame["prod_anchor"] = recipe(tr, va, BASE, anchored=True)
    saved = pd.read_parquet(SAVED / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt]
    gate = {"pair": key,
            "prod_v44_max_diff": float(np.abs(frame.prod_v44.to_numpy()-saved.prod_v44.to_numpy()).max()),
            "prod_anchor_max_diff": float(np.abs(frame.prod_anchor.to_numpy()-saved.prod_anchor.to_numpy()).max())}
    gate["passed"] = gate["prod_v44_max_diff"] <= 1e-6 and gate["prod_anchor_max_diff"] <= 1e-6
    gates_b.append(gate)
    print("GATE_B", json.dumps(gate), flush=True)
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
pooled_stand_share = sum(c["stand"]*c["rows"] for c in held_coverage.values()) / rows_2025
coverage_2026 = {"rows": int(len(table_2026)), "stand_or_runway": float((table_2026.ref_level >= 1).mean()),
                 "stand": float((table_2026.ref_level == 2).mean())}
coverage_ok = bool(all(c["stand_or_runway"] >= .95 for c in held_coverage.values())
                   and coverage_2026["stand_or_runway"] >= .95 and abs(coverage_2026["stand"]-pooled_stand_share) <= .05)

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


difference = pooled.ref_p50.to_numpy(float) + pooled.sa_excess.to_numpy(float) - pooled.nt_anchor.to_numpy(float)
finite = np.isfinite(difference)
spearman = {}
for airport, g in pooled.groupby("ADEP_mvt"):
    ok = np.isfinite(g.ref_p50.to_numpy(float)) & np.isfinite(g.strecke_m.to_numpy(float))
    spearman[airport] = (float(np.corrcoef(g.ref_p50[ok].rank(), g.strecke_m[ok].rank())[0, 1]) if ok.sum() >= 100 else None)
report = {"validation": "fallback-only six-pair screen against the scored v46 fallback recipe, production categorical "
                        "encoding; not full ensemble; not a leaderboard forecast",
          "gate_a_rebuilt_neighbours": GATE_A, "gate_b_replication": gates_b,
          "gate_c_coverage": {"held_out_2025": held_coverage, "pooled_2025_stand_share": pooled_stand_share,
                              "ranking_2026": coverage_2026, "passed": coverage_ok},
          "build_tables_gate_a_passed": build_gates_ok, "code_sha256": CODE,
          "descriptive": {"stage_anchor_minus_nt_anchor_s": {
                              "rows": int(finite.sum()), "median_abs": float(np.median(np.abs(difference[finite]))),
                              "p90_abs": float(np.quantile(np.abs(difference[finite]), .9)),
                              "share_over_120s": float((np.abs(difference[finite]) > 120).mean())},
                          "spearman_ref_p50_vs_strecke_m_by_airport": spearman},
          "rmse": {c: float(np.sqrt(np.mean((pooled[c].to_numpy(float)-y)**2))) for c in ["prod_v44", "prod_anchor"] + VARIANTS},
          "variants": {name: compare(pooled[name]) for name in VARIANTS}}
print("SCREEN", json.dumps({n: {"gain": e["pooled_mse_gain"], "without_top100": e["mse_gain_without_top100_rows"],
                                "pairs": e["pairs_positive"], "jan": e["january_gain_1_7"], "jul": e["july_gain_1_7"],
                                "checks": e["checks"]} for n, e in report["variants"].items()}, default=jsonable), flush=True)

for name in VARIANTS:
    entry = report["variants"][name]
    if not all(entry["checks"].values()):
        continue
    entry["permuted_pooled_mse_gains"] = {}
    for label, columns in CONTROLS[name].items():
        permuted = {seed: [] for seed in SEEDS}
        for pair in PAIRS:
            key = f"{pair[0]}_{pair[1]}"
            valid_raw, tr, va = load_pair(pair)
            rows = folds == key
            assert np.array_equal(pooled.loc[rows, "MVT_ID_mvt"].to_numpy(), va.MVT_ID_mvt.to_numpy())
            yv = va.y.to_numpy(float)
            base_error = (pooled.loc[rows, "prod_anchor"].to_numpy(float)-yv)**2
            for seed in SEEDS:
                prediction = variant(name, permute(tr, columns, seed), permute(va, columns, seed+1))
                permuted[seed].append(base_error - (prediction-yv)**2)
            del valid_raw, tr, va
            gc.collect()
        entry["permuted_pooled_mse_gains"][label] = {str(seed): float(np.concatenate(v).mean()) for seed, v in permuted.items()}
        print("PERMUTATION", name, label, json.dumps(entry["permuted_pooled_mse_gains"][label]), flush=True)
    entry["checks"]["5_beats_every_permutation"] = all(entry["pooled_mse_gain"] > g for gains in
                                                       entry["permuted_pooled_mse_gains"].values() for g in gains.values())

for entry in report["variants"].values():
    entry["advance"] = bool(all(entry["checks"].values()) and "5_beats_every_permutation" in entry["checks"]
                            and coverage_ok and build_gates_ok)
advancing = [n for n, e in report["variants"].items() if e["advance"]]
report["decision"] = {"advancing": advancing,
                      "candidate_for_2026": max(advancing, key=lambda n: report["variants"][n]["pooled_mse_gain"]) if advancing else None}
report["seconds"] = time.time()-started
(output / "pooled.json").write_text(json.dumps(report, indent=2, default=jsonable))
print("SUMMARY", json.dumps({"rmse": report["rmse"], "gate_c": report["gate_c_coverage"], "build_gates": build_gates_ok,
                             "descriptive": report["descriptive"], "decision": report["decision"],
                             "variants": {n: {"gain": e["pooled_mse_gain"], "without_top100": e["mse_gain_without_top100_rows"],
                                              "pairs": e["pairs_positive"], "jan": e["january_gain_1_7"], "jul": e["july_gain_1_7"],
                                              "bootstrap": e["airport_day_bootstrap_mse_gain_95pct"],
                                              "mechanical_rmse": e["mechanical_overall_rmse_if_full_transfer_not_forecast"],
                                              "permuted": e.get("permuted_pooled_mse_gains"), "checks": e["checks"],
                                              "advance": e["advance"]} for n, e in report["variants"].items()}},
                            indent=2, default=jsonable), flush=True)

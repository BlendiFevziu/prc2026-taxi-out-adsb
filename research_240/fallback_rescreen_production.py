"""Production-encoded re-screen of earlier fallback rejections, plus a month-clipped count.

Why: research_250's fallback experiments, and research_240's, were measured with a harness
that re-encodes categorical inputs from the fallback rows alone; that model differs from the
scored fallback by RMS 199.3 s on 2026 rows. This screen repeats them with production encoding
(H.baue category dtypes and real missing values, as main_model.py --alle), on tables rebuilt
with every column the variants need. No leaderboard feedback is used.

Pre-specified 13.09.2026, before any fit.

Tables, one per source split (six ten-month learning and six two-month held-out splits), built
on the full departure table of that split: fallback rows with the 35 production inputs, the six
nt_ neighbour-taxi columns, tu_auslast, the twelve rc_ reporting-context columns, the four rt_
record-time columns, auslast_rate, and auslast_month (legacy auslast window, +/-1,800,000 s,
counted only among departures in the row's calendar month).

Variants, each compared against prod_v44 (the scored fallback recipe):
  R1  tu_added             + tu_auslast                        research_250 corrected count added
  R2  tu_replaces          auslast replaced by tu_auslast      research_250 replacement
  R3  reporting_context    + tu_auslast + 12 rc_               research_250 reporting context
  R4  missingness_pair     + tu_auslast + 2 rc_ missingness    research_250 narrow ablation
  R5  record_time          + tu_auslast + 4 rt_                research_250 record-time
  R6  record_missingness   + tu_auslast + 4 rt_ + 2 rc_        research_250 combined
  R7  capped_excess        E[min(y,5400)] + E[max(y-5400,0)] on R3 inputs   research_250
  R8  bootstrap_ensemble   original fit and five (month, extreme)-stratified bootstrap replicates,
                           seeds 11, 23, 37, 53, 71, equal weights          research_250
  R9  crossfit_gate        inner month-pair cross-fitted squared-error gate on R3 inputs
  R9b half_context_gate    0.5 * R3 + 0.5 * R9                             research_250
  R10 nt_features          + six nt_ columns in every model                research_240
  R11 rate                 auslast replaced by auslast_rate                research_240
  N1  month_clipped        auslast replaced by auslast_month               new
Not re-screened: same-flight arrival relinking; its limit (775 of 22,470 rows covered, at most
11,665 fallback MSE even with perfect predictions) does not depend on the encoding.

Replication gates: prod_v44 and the anchored recipe must reproduce the saved predictions of
research_240/fallback_anchor_production.py on every pair (max difference 1e-6 s).

Decision rule, fixed now, each variant against prod_v44, pooled over six pairs:
  (1) MSE gain positive, (2) survives removing its 100 most favourable rows, (3) at least 4 of 6
  pairs gain, (4) January and July both gain inside pair 1_7.
  Variants that add or replace columns (R1-R6, R10, R11, N1) and pass (1)-(4) must also pass
  (5): the real pooled gain exceeds every pooled gain with those columns permuted jointly within
  (split, airport), seeds 101, 202, 303, 404, 505.
  A variant that advances is also fitted on top of the neighbour-taxi anchor and reported against
  the anchor with checks (1)-(4), descriptively.
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
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--reference", default=Path(".cache/research_240/fallback_anchor_production"), type=Path)
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()
project, output = args.project.resolve(), args.output.resolve()
reference = args.reference if args.reference.is_absolute() else project / args.reference
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import features_geometry as MG
import features_stand_rate as MSQ
from fid_gap_cv import month_of_file
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_240 import features_auslast_rate as AR
from research_240 import features_neighbour_taxi as NT
from research_250 import features_record_time as R
from research_250 import features_reporting_context as C
from research_250 import features_time_units as U

BASE = list(H.MERKMALE_RUECK)
MISSING = ["rc_missing_share_15m", "rc_missing_run_size"]
UNIT = BASE + ["tu_auslast"]
CONTEXT = UNIT + C.MERKMALE
ADD = {"R1_tu_added": UNIT, "R3_reporting_context": CONTEXT, "R4_missingness_pair": UNIT + MISSING,
       "R5_record_time": UNIT + R.MERKMALE, "R6_record_missingness": UNIT + R.MERKMALE + MISSING,
       "R10_nt_features": BASE + NT.MERKMALE}
REPLACE = {"R2_tu_replaces": "tu_auslast", "R11_rate": "auslast_rate", "N1_month_clipped": "auslast_month"}
PERMUTED = {"R1_tu_added": ["tu_auslast"], "R3_reporting_context": ["tu_auslast"] + C.MERKMALE,
            "R4_missingness_pair": ["tu_auslast"] + MISSING, "R5_record_time": ["tu_auslast"] + R.MERKMALE,
            "R6_record_missingness": ["tu_auslast"] + R.MERKMALE + MISSING, "R10_nt_features": NT.MERKMALE,
            "R2_tu_replaces": ["auslast"], "R11_rate": ["auslast"], "N1_month_clipped": ["auslast"]}
STRUCTURAL = ["R7_capped_excess", "R8_bootstrap_ensemble", "R9_crossfit_gate", "R9b_half_context_gate"]
VARIANTS = list(ADD) + list(REPLACE) + STRUCTURAL
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
SEEDS = [101, 202, 303, 404, 505]
KEEP = list(dict.fromkeys([f for f in BASE if f not in MSQ.MERKMALE]
                          + ["MVT_ID_mvt", "FLIGHT_ID_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                             "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "y", "monat", "hat_aobt"]))
EXTRA = NT.MERKMALE + ["tu_auslast"] + C.MERKMALE + R.MERKMALE + ["auslast_month", "auslast_rate"]
paths = sorted((project / "data").glob("training_2025-*.parquet"))
assert len(paths) == 12 and {month_of_file(p) for p in paths} == set(range(1, 13))
geometry = MG.alle()
BUILDERS = [Path(m.__file__).resolve() for m in (NT, AR, R, C, U)]
CODE = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), project / "main_model.py", project / "features_stand_rate.py"] + BUILDERS}


def fingerprint(selected):
    h = hashlib.sha256()
    for p in sorted(project.glob("*.py")) + BUILDERS:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    for p in selected:
        h.update(p.name.encode())
        h.update(str((p.stat().st_size, p.stat().st_mtime_ns)).encode())
    return h.hexdigest()[:16]


def month_clipped_count(full):
    stamps = pd.to_datetime(full.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    naive = stamps.dt.tz_convert(None).dt.as_unit("ns").to_numpy(dtype="datetime64[ns]")
    ok = ~np.isnat(naive)
    seconds = np.where(ok, naive.astype(np.int64) // 1_000_000_000, 0)
    key = (full.ADEP_mvt.astype(str) + "|" + stamps.dt.strftime("%Y-%m")).to_numpy()
    frame = pd.DataFrame({"k": key, "s": seconds, "i": np.arange(len(full))})[ok]
    out = np.full(len(full), np.nan)
    for _, g in frame.groupby("k", sort=False):
        s = np.sort(g.s.to_numpy())
        q = g.s.to_numpy()
        out[g.i.to_numpy()] = np.searchsorted(s, q + 1_800_000, "right") - np.searchsorted(s, q - 1_800_000, "left") - 1
    return out


def table(selected):
    key = fingerprint(selected)
    cached = output / f"rescreen_{key}.parquet"
    months = [month_of_file(p) for p in selected]
    if cached.exists():
        return pd.read_parquet(cached)
    started = time.time()
    print("BUILD", months, flush=True)
    full = H.baue([str(p) for p in selected], geometry)
    mask = (full.hat_aobt.eq(0) & full.y.notna()).to_numpy()
    nt = NT.baue(full, query=mask)
    raw = pd.concat([pd.read_parquet(p, columns=U.RAW_COLUMNS) for p in selected], ignore_index=True)
    fixed = U.corrected_values(raw[U.RAW_COLUMNS], full[U.DEP_COLUMNS])
    del raw
    context = C.baue(full[C.RAW_COLUMNS])
    record = R.baue(full[R.RAW_COLUMNS])
    assert context.index.equals(full.index) and record.index.equals(full.index)
    clipped = month_clipped_count(full)
    small = full.loc[mask, KEEP].copy()
    for col in NT.MERKMALE:
        small[col] = nt[col].to_numpy()[mask]
    small["tu_auslast"] = fixed["auslast"].to_numpy()[mask]
    for col in C.MERKMALE:
        small[col] = context[col].to_numpy()[mask]
    for col in R.MERKMALE:
        small[col] = record[col].to_numpy()[mask]
    small["auslast_month"] = clipped[mask]
    small["auslast_rate"] = AR.rate(small.auslast.to_numpy(float), small.MVT_TIME_UTC_mvt,
                                    AR.month_spans([(2025, m) for m in months]))
    del full, nt, fixed, context, record
    gc.collect()
    small.to_parquet(cached, index=False)
    (output / f"rescreen_{key}.json").write_text(json.dumps({"source_months": months, "rows": len(small),
                                                             "seconds": time.time()-started}, indent=2))
    print("BUILT", months, len(small), "rows", round(time.time()-started, 1), "s", flush=True)
    return small


def stand_rate(train, valid):
    train, valid = train.copy(), valid.copy()
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(train)
    train = MSQ.ausser_faltung(train)
    valid = MSQ.wende_an(mapping, valid)
    return train, valid


def reg(rounds, leaves, minimum, **kw):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                             min_child_samples=minimum, verbose=-1, n_jobs=args.jobs, random_state=0, **kw)


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=args.jobs, random_state=0)


def parts(tr, va, features):
    """Classifier probability, height and normal prediction on already stand-rated tables."""
    z0 = tr.z.eq(0).to_numpy()
    y = tr.y.to_numpy(float)
    p = classifier().fit(tr[features], tr.z).predict_proba(va[features])[:, 1]
    height = ((1-H.ANTEIL_HM)*va.versp_min.to_numpy(float)*60
              + H.ANTEIL_HM*reg(300, 15, 10).fit(tr.loc[~z0, features], y[~z0]).predict(va[features]))
    normal = reg(400, 31, 20).fit(tr.loc[z0, features], y[z0]).predict(va[features])
    return p, height, normal


def combine(p, height, normal):
    q = p ** H.HOCH
    return np.maximum((1-q)*normal + q*height, 0)


def recipe(tr, va, features, anchored=False):
    p, height, normal = parts(tr, va, features)
    if anchored:
        z0 = tr.z.eq(0).to_numpy()
        y = tr.y.to_numpy(float)
        offset_features = features + [c for c in NT.MERKMALE if c not in features]
        a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
        rows = z0 & np.isfinite(a_tr)
        offset = reg(400, 31, 20).fit(tr.loc[rows, offset_features], y[rows]-a_tr[rows])
        has = np.isfinite(a_va)
        normal = normal.copy()
        normal[has] = a_va[has] + offset.predict(va.loc[has, offset_features])
    return combine(p, height, normal)


def replaced(frame, column):
    out = frame.copy()
    out["auslast"] = out[column].to_numpy(float)
    return out


def capped_excess(tr, va, features):
    y = tr.y.to_numpy(float)
    cap = float(H.SCHWELLE_EXTREM)
    lower = reg(400, 31, 50).fit(tr[features], np.minimum(y, cap))
    upper = reg(300, 15, 50).fit(tr[features], np.maximum(y-cap, 0))
    return np.maximum(lower.predict(va[features]) + np.maximum(upper.predict(va[features]), 0), 0)


def bootstrap_ensemble(tr, va):
    members = [recipe(tr, va, BASE)]
    groups = [np.asarray(idx) for idx in tr.groupby(["monat", "z"], observed=True).indices.values()]
    for seed in (11, 23, 37, 53, 71):
        rng = np.random.default_rng(seed)
        indices = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        rng.shuffle(indices)
        members.append(recipe(tr.iloc[indices].reset_index(drop=True), va, BASE))
    return np.mean(members, axis=0)


def gate_frame(raw, frame_parts):
    cols = ["ADEP_mvt", "versp_min", "stunde", "tag_versch", "id_abw", "id_abw2",
            "ank_med_30min", "ank_med_90min", "tu_auslast"] + C.MERKMALE
    frame = raw[cols].copy()
    for col in ("normal", "height", "p"):
        frame["gate_" + col] = frame_parts[col].to_numpy()
    frame["gate_delta"] = frame_parts.height.to_numpy() - frame_parts.normal.to_numpy()
    return frame


def crossfit_gate(train_raw, valid_raw, valid_parts, features):
    """research_250 predict_gate with production encoding; inner splits re-fit the stand rate."""
    oof = pd.DataFrame(index=train_raw.index, columns=["normal", "height", "p"], dtype=float)
    available = set(train_raw.monat.unique())
    for first in range(1, 7):
        months = available & {first, first+6}
        if not months:
            continue
        hold = train_raw.monat.isin(months).to_numpy()
        tr_i, va_i = stand_rate(train_raw.loc[~hold], train_raw.loc[hold])
        p, height, normal = parts(tr_i, va_i, features)
        oof.loc[va_i.index, ["normal", "height", "p"]] = np.column_stack([normal, height, p])
    assert oof.notna().all().all()
    x, xv = gate_frame(train_raw, oof), gate_frame(valid_raw, valid_parts)
    delta = oof.height.to_numpy()-oof.normal.to_numpy()
    useful = np.abs(delta) >= 60
    target = (train_raw.y.to_numpy(float)[useful]-oof.normal.to_numpy()[useful])/delta[useful]
    weight = delta[useful]**2
    weight /= weight.mean()
    model = reg(120, 7, 200, reg_lambda=100).fit(x.loc[useful], target, sample_weight=weight)
    q = np.clip(model.predict(xv), 0, 1)
    return np.maximum(valid_parts.normal.to_numpy() + q*(valid_parts.height.to_numpy()-valid_parts.normal.to_numpy()), 0)


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


def variant_prediction(name, tr, va, train_raw, valid_raw, cache, anchored=False):
    if name in ADD:
        return recipe(tr, va, ADD[name], anchored)
    if name in REPLACE:
        return recipe(replaced(tr, REPLACE[name]), replaced(va, REPLACE[name]), BASE, anchored)
    if name == "R7_capped_excess":
        return capped_excess(tr, va, CONTEXT)
    if name == "R8_bootstrap_ensemble":
        return bootstrap_ensemble(tr, va)
    if name in ("R9_crossfit_gate", "R9b_half_context_gate"):
        if "gate" not in cache:
            p, height, normal = parts(tr, va, CONTEXT)
            context_parts = pd.DataFrame({"normal": normal, "height": height, "p": p}, index=va.index)
            cache["context"] = combine(p, height, normal)
            cache["gate"] = crossfit_gate(train_raw, valid_raw, context_parts, CONTEXT)
        return cache["gate"] if name == "R9_crossfit_gate" else .5*cache["context"] + .5*cache["gate"]
    raise KeyError(name)


def load_pair(pair):
    train_raw = table([p for p in paths if month_of_file(p) not in pair]).reset_index(drop=True)
    valid_raw = table([p for p in paths if month_of_file(p) in pair]).reset_index(drop=True)
    assert set(train_raw.MVT_ID_mvt).isdisjoint(set(valid_raw.MVT_ID_mvt))
    tr, va = stand_rate(train_raw, valid_raw)
    return train_raw, valid_raw, tr, va


frames, gates, started = [], [], time.time()
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    train_raw, valid_raw, tr, va = load_pair(pair)
    yv = va.y.to_numpy(float)
    frame = va[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y", "nt_anchor"]].copy()
    frame["ADEP_mvt"] = va.ADEP_mvt.astype(str).to_numpy()
    frame["prod_v44"] = recipe(tr, va, BASE)
    frame["prod_anchor"] = recipe(tr, va, BASE, anchored=True)
    saved = pd.read_parquet(reference / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt]
    gate = {"pair": key,
            "prod_v44_max_diff": float(np.abs(frame.prod_v44.to_numpy()-saved.prod_v44.to_numpy()).max()),
            "prod_anchor_max_diff": float(np.abs(frame.prod_anchor.to_numpy()-saved.prod_anchor.to_numpy()).max())}
    gate["passed"] = gate["prod_v44_max_diff"] <= 1e-6 and gate["prod_anchor_max_diff"] <= 1e-6
    gates.append(gate)
    print("GATE", json.dumps(gate), flush=True)
    if not gate["passed"]:
        (output / "replication_failed.json").write_text(json.dumps(gates, indent=2))
        sys.exit(f"Replication gate failed on {key}")
    cache = {}
    for name in VARIANTS:
        before = time.time()
        frame[name] = variant_prediction(name, tr, va, train_raw, valid_raw, cache)
        print("VARIANT", key, name, f"{np.sqrt(np.mean((frame[name]-yv)**2)):.3f}", f"{time.time()-before:.1f}s", flush=True)
    frame["fold"] = key
    frame.to_parquet(output / f"predictions_{key}.parquet", index=False)
    frames.append(frame)
    del train_raw, valid_raw, tr, va
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


report = {"validation": "fallback-only six-pair re-screen with production categorical encoding; not full ensemble; "
                        "not a leaderboard forecast", "replication_gates": gates, "code_sha256": CODE,
          "rmse": {c: float(np.sqrt(np.mean((pooled[c].to_numpy(float)-y)**2))) for c in ["prod_v44", "prod_anchor"] + VARIANTS},
          "variants": {}}
for name in VARIANTS:
    result = compare("prod_v44", pooled[name])
    result["passes_1_to_4"] = all(result["checks"].values())
    report["variants"][name] = result

for name, columns in PERMUTED.items():
    entry = report["variants"][name]
    if not entry["passes_1_to_4"]:
        continue
    permuted_gains = {seed: [] for seed in SEEDS}
    for pair in PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        train_raw, valid_raw, tr, va = load_pair(pair)
        rows = pooled.fold.to_numpy() == key
        assert np.array_equal(pooled.loc[rows, "MVT_ID_mvt"].to_numpy(), va.MVT_ID_mvt.to_numpy())
        for seed in SEEDS:
            tr_s, va_s = (replaced(tr, REPLACE[name]), replaced(va, REPLACE[name])) if name in REPLACE else (tr, va)
            tr_s, va_s = permute(tr_s, columns, seed), permute(va_s, columns, seed+1)
            prediction = recipe(tr_s, va_s, ADD.get(name, BASE))
            permuted_gains[seed].append(((pooled.loc[rows, "prod_v44"].to_numpy(float)-va.y.to_numpy(float))**2
                                         - (prediction-va.y.to_numpy(float))**2))
        del train_raw, valid_raw, tr, va
        gc.collect()
    pooled_permuted = {seed: float(np.concatenate(v).mean()) for seed, v in permuted_gains.items()}
    entry["permuted_pooled_mse_gains"] = pooled_permuted
    entry["checks"]["5_beats_every_permutation"] = all(entry["pooled_mse_gain"] > g for g in pooled_permuted.values())
    print("PERMUTATION", name, json.dumps(pooled_permuted), flush=True)

for name, entry in report["variants"].items():
    entry["advance"] = bool(all(entry["checks"].values()))
    if entry["advance"]:
        anchored_predictions = []
        for pair in PAIRS:
            train_raw, valid_raw, tr, va = load_pair(pair)
            anchored_predictions.append(variant_prediction(name, tr, va, train_raw, valid_raw, {}, anchored=True)
                                        if name in ADD or name in REPLACE else None)
            del train_raw, valid_raw, tr, va
            gc.collect()
        if all(p is not None for p in anchored_predictions):
            entry["on_top_of_anchor_descriptive"] = compare("prod_anchor", np.concatenate(anchored_predictions))

report["seconds"] = time.time()-started
(output / "pooled.json").write_text(json.dumps(report, indent=2, default=bool))
summary = {name: {"rmse": report["rmse"][name], "gain": e["pooled_mse_gain"], "checks": e["checks"], "advance": e["advance"]}
           for name, e in report["variants"].items()}
print("SUMMARY", json.dumps({"prod_v44": report["rmse"]["prod_v44"], "prod_anchor": report["rmse"]["prod_anchor"],
                             "variants": summary}, indent=2, default=bool), flush=True)

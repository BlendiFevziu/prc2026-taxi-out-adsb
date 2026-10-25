"""Permutation control for the research_240 fallback anchor screen.

Recommended for classifier-based changes by another team on the competition Discord
(another participant, 12.09.2026) and adopted here before any promotion. The six neighbour-taxi
columns are permuted jointly within each (source split, airport) group, in the learning
and the held-out table alike. That keeps each airport's marginal distribution and
destroys the link to the actual hour. Both variants are refitted on the permuted columns.
If permuted columns reproduce a variant's pooled gain, that gain is no evidence for the
neighbour mechanism.

Builds nothing: the screen's cached tables are found through their JSON sidecars. The real
variants are refitted first and must reproduce result_<pair>.json, otherwise the run stops.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--screen", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--seeds", default="101,202,303,404,505")
parser.add_argument("--jobs", type=int, default=4)
args = parser.parse_args()
project, screen = args.project.resolve(), args.screen.resolve()
output = args.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
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
VARIANTS = ("nt_features", "nt_anchor_normal")
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
seeds = [int(s) for s in args.seeds.split(",")]


def cached_tables():
    found = {}
    for side in sorted(screen.glob("fallback_nt_*.json")):
        months = tuple(sorted(json.loads(side.read_text())["source_months"]))
        if months in found:
            raise SystemExit(f"Two cached tables for months {months}; remove the stale one")
        found[months] = side.with_suffix(".parquet")
    return found


def prepare(train, valid):
    """Identical to fallback_anchor_screen.prepare."""
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


def fit_variants(train, valid, with_baseline):
    """Same fits as the screen. The baseline ignores the neighbour columns by design."""
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
    out = {"v44_fallback": combine(normal_base, h_base, p_base)} if with_baseline else {}
    p_nt, h_nt = parts(WITH_NT)
    normal_nt = reg(400, 31, 20).fit(tr.loc[z0, WITH_NT], y[z0]).predict(va[WITH_NT])
    out["nt_features"] = combine(normal_nt, h_nt, p_nt)
    anchor_tr, anchor_va = tr.nt_anchor.to_numpy(), va.nt_anchor.to_numpy()
    fit_rows = z0 & np.isfinite(anchor_tr)
    offset = reg(400, 31, 20).fit(tr.loc[fit_rows, WITH_NT], y[fit_rows]-anchor_tr[fit_rows])
    has = np.isfinite(anchor_va)
    normal_anchor = normal_base.copy()
    normal_anchor[has] = anchor_va[has] + offset.predict(va.loc[has, WITH_NT])
    out["nt_anchor_normal"] = combine(normal_anchor, h_base, p_base)
    return out, va.y.to_numpy(float)


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


tables = cached_tables()
records, started = {}, time.time()
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    train = pd.read_parquet(tables[tuple(sorted(set(range(1, 13))-set(pair)))]).reset_index(drop=True)
    valid = pd.read_parquet(tables[tuple(sorted(pair))]).reset_index(drop=True)
    real, yv = fit_variants(train, valid, with_baseline=True)
    base_mse = float(((real["v44_fallback"]-yv)**2).mean())
    reference = json.loads((screen / f"result_{key}.json").read_text())
    record = {"rows": int(len(yv)), "baseline_mse": base_mse, "real": {}, "permuted": {v: [] for v in VARIANTS}}
    for variant in VARIANTS:
        gain = base_mse - float(((real[variant]-yv)**2).mean())
        expected = reference[variant]["mse_gain_vs_v44"]
        if abs(gain-expected) > 1e-6*max(1.0, abs(expected)):
            raise SystemExit(f"Refit of {variant} on {key} does not reproduce the screen: {gain} vs {expected}")
        record["real"][variant] = gain
    for seed in seeds:
        permuted, _ = fit_variants(permute(train, seed), permute(valid, seed+1), with_baseline=False)
        for variant in VARIANTS:
            record["permuted"][variant].append(base_mse - float(((permuted[variant]-yv)**2).mean()))
        gc.collect()
    print("PAIR", key, json.dumps({v: {"real": record["real"][v], "permuted": record["permuted"][v]}
                                   for v in VARIANTS}), flush=True)
    records[key] = record
    del train, valid
    gc.collect()

rows = np.array([records[f"{a}_{b}"]["rows"] for a, b in PAIRS], dtype=float)
summary = {}
for variant in VARIANTS:
    real = np.array([records[f"{a}_{b}"]["real"][variant] for a, b in PAIRS])
    perm = np.array([records[f"{a}_{b}"]["permuted"][variant] for a, b in PAIRS])  # pairs x seeds
    pooled_real = float((real*rows).sum()/rows.sum())
    pooled_perm = (perm*rows[:, None]).sum(axis=0)/rows.sum()
    summary[variant] = {"pooled_real_mse_gain": pooled_real,
                        "pooled_permuted_mse_gains": [float(x) for x in pooled_perm],
                        "permutations_reaching_real_gain": int((pooled_perm >= pooled_real).sum()),
                        "pairs_where_real_beats_every_permutation": int((real[:, None] > perm).all(axis=1).sum())}
report = {"control": "neighbour-taxi columns permuted jointly within (source split, airport)",
          "seeds": seeds, "pairs": records, "summary": summary, "seconds": time.time()-started}
output.write_text(json.dumps(report, indent=2))
print("SUMMARY", json.dumps(summary, indent=2), flush=True)

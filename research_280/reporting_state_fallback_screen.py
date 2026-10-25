"""Pre-registered six-pair fallback screen: the airport's reporting state as fallback inputs. Fixed 16.09.2026.

Same six columns as the main-lane screen (features_reporting_state.py), here in the lane where the measured
association is strongest. Over all 2025 fallback departures, grouped by the arrival copy rate of the same airport in
a two-hour bucket, the fallback copy rate runs 2.93 % -> 14.33 % across quintiles and the extreme rate
1.94 % -> 10.42 %. Grouped by the share of arrival block times carrying a seconds field, the copy rate runs
8.67 % -> 4.90 % and the extreme rate 6.14 % -> 1.37 %.

Distinct from two rejected things. research_270's own-arrival copy flag asked whether the SAME flight's arrival was a
copy (5.87 % against 5.60 %, dead); this is the airport's state in the window. And today's day-level disruption
screen (research_280/fallback_disruption_screen.py) measured delay and volume, which the model already carries
through auslast and versp_min; it was rejected at 1486.082 -> 1498.210. Reporting state is a different quantity: how
the timestamps are written, not how busy the airport is.

Baseline prod_anchor: the scored v46 fallback recipe rebuilt per pair, GATE reproduces the saved predictions within
1e-6 s. Candidate: the identical recipe with the six columns appended to the fallback inputs and to the anchored
offset. No other change, no hyperparameter search.

Decision rule, pooled over six pairs:
  1 pooled MSE gain positive
  2 positive without its 100 most favourable rows
  3 at least 4 of 6 pairs gain
  4 January and July both gain inside pair 1_7
  5 beats the permutation control, which shuffles the six columns across airport-days
  6 mean gain on ordinary rows (label <= 5,400 s) >= 0
Reported either way: per-regime and per-airport contributions and an airport-day bootstrap. Nothing is built or
uploaded here.

Run from the repository root:  python3 -u -B research_280/reporting_state_fallback_screen.py
"""
import argparse
import glob
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, default=Path("."))
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()
project = args.project.resolve()
sys.path.insert(0, str(project))

import main_model as H  # torch before tree libraries on macOS
import features_stand_rate as MSQ
import features_reporting_state as RS
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from research_240 import features_neighbour_taxi as NT

BASE = list(H.MERKMALE_RUECK)
OFFSET = BASE + [c for c in NT.MERKMALE if c not in BASE]
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
TABLES = project / ".cache/research_240/rescreen_production"
SAVED = project / ".cache/research_240/fallback_anchor_production"
OUTPUT = project / "research_280/results/reporting_state_fallback_screen.json"
V48, SHARE_2026 = 266.4508, 5290 / 344841
ORDINARY = 5400.0
index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet")
         for s in TABLES.glob("rescreen_*.json")}


def reporting_state():
    frames = [pq.ParquetFile(p).read(columns=RS.RAW_COLUMNS + ["MVT_ID_mvt"], use_threads=False).to_pandas()
              for p in sorted(glob.glob(str(project / "data/training_2025-*.parquet")))]
    raw = pd.concat(frames, ignore_index=True)
    dep = raw.loc[raw.PHASE_mvt.astype(str).str.upper().str.startswith("DEP")].reset_index(drop=True)
    built = RS.baue(raw, dep)
    built["MVT_ID_mvt"] = dep.MVT_ID_mvt.to_numpy()
    return built.set_index("MVT_ID_mvt")


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                             min_child_samples=minimum, verbose=-1, n_jobs=args.jobs, random_state=0)


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=args.jobs, random_state=0)


def recipe(tr, va, base_columns, offset_columns):
    z0, ytr = tr.z.eq(0).to_numpy(), tr.y.to_numpy(float)
    p = classifier().fit(tr[base_columns], tr.z).predict_proba(va[base_columns])[:, 1]
    delay = 60 * va.versp_min.to_numpy(float)
    height = (1 - H.ANTEIL_HM) * delay + H.ANTEIL_HM * reg(300, 15, 10).fit(tr.loc[~z0, base_columns], ytr[~z0]).predict(va[base_columns])
    normal = reg(400, 31, 20).fit(tr.loc[z0, base_columns], ytr[z0]).predict(va[base_columns])
    a_tr, a_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
    rows = z0 & np.isfinite(a_tr)
    offset = reg(400, 31, 20).fit(tr.loc[rows, offset_columns], ytr[rows] - a_tr[rows])
    has = np.isfinite(a_va)
    anchored = normal.copy()
    anchored[has] = a_va[has] + offset.predict(va.loc[has, offset_columns])
    q = p ** H.HOCH
    return np.maximum((1 - q) * anchored + q * height, 0)


print("building the reporting state", flush=True)
state = reporting_state()
print(f"  {len(state):,} departures covered", flush=True)
rng = np.random.default_rng(20260916)
frames = []
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    learn = [m for m in range(1, 13) if m not in pair]
    tr = pd.read_parquet(index[tuple(learn)]).reset_index(drop=True)
    va = pd.read_parquet(index[tuple(sorted(pair))]).reset_index(drop=True)
    for frame in (tr, va):
        frame["z"] = (frame.y > H.SCHWELLE_EXTREM).astype(int)
        for column in RS.MERKMALE:
            frame[column] = frame.MVT_ID_mvt.map(state[column]).to_numpy(float)
    tr, va = MSQ.ausser_faltung(tr), MSQ.wende_an(MSQ.passe_an(tr), va)
    base = recipe(tr, va, BASE, OFFSET)
    saved = pd.read_parquet(SAVED / f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[va.MVT_ID_mvt, "prod_anchor"].to_numpy(float)
    gap = float(np.abs(base - saved).max())
    print("GATE", key, gap, flush=True)
    if gap > 1e-6:
        raise SystemExit(f"rebuild does not reproduce prod_anchor on {key}")
    frame = pd.DataFrame({"fold": key, "MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "MVT_TIME_UTC_mvt": va.MVT_TIME_UTC_mvt.to_numpy(),
                          "airport": va.ADEP_mvt.astype(str).to_numpy(), "y": va.y.to_numpy(float),
                          "delay": 60 * va.versp_min.to_numpy(float), "prod_anchor": base})
    for name in ("real", "permuted"):
        tr2, va2 = tr.copy(), va.copy()
        if name == "permuted":
            for side in (tr2, va2):
                keys = side.ADEP_mvt.astype(str) + "|" + pd.to_datetime(side.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
                table = side.groupby(keys, observed=True)[RS.MERKMALE].first()
                shuffled = table.sample(frac=1, random_state=int(rng.integers(1 << 30))).set_index(table.index)
                for column in RS.MERKMALE:
                    side[column] = keys.map(shuffled[column]).to_numpy(float)
        frame[name] = recipe(tr2, va2, BASE + RS.MERKMALE, OFFSET + RS.MERKMALE)
    yv = frame.y.to_numpy()
    print("PAIR", key, json.dumps({k: round(float(np.sqrt(np.mean((frame[k].to_numpy() - yv) ** 2))), 3)
                                   for k in ("prod_anchor", "real", "permuted")}), flush=True)
    frames.append(frame)

pooled = pd.concat(frames, ignore_index=True)
y, delay, base = pooled.y.to_numpy(), pooled.delay.to_numpy(), pooled.prod_anchor.to_numpy()
folds = pooled.fold.to_numpy()
stamps = pd.to_datetime(pooled.MVT_TIME_UTC_mvt, utc=True)
month = np.array([m if str(m) in f.split("_") else m % 12 + 1 for m, f in zip(stamps.dt.month.to_numpy(), folds)])
_, inverse = np.unique((pooled.airport + "|" + stamps.dt.strftime("%Y-%m-%d")).to_numpy(), return_inverse=True)
cluster_rows = np.bincount(inverse)
draws = np.random.default_rng(719).integers(0, cluster_rows.size, size=(args.bootstrap, cluster_rows.size))
copy = np.abs(y - delay) <= 60
extreme = y > H.SCHWELLE_EXTREM
shift = ~copy & (y >= 80000) & (y <= 93600)
normal = ~copy & (y >= -600) & (y <= 5400)
regimes = {"copy_extreme": copy & extreme, "copy_nonextreme": copy & ~extreme, "day_shift": shift,
           "normal": normal, "other": ~copy & ~shift & ~normal}
ordinary = y <= ORDINARY


def compare(candidate):
    gain = (base - y) ** 2 - (candidate - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    pair_gains = {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}
    boot = np.bincount(inverse, weights=gain)[draws].sum(axis=1) / cluster_rows[draws].sum(axis=1)
    return {"pooled_mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
            "pair_gains": pair_gains, "pairs_positive": int(sum(g > 0 for g in pair_gains.values())),
            "january_gain_1_7": float(gain[(folds == "1_7") & (month == 1)].mean()),
            "july_gain_1_7": float(gain[(folds == "1_7") & (month == 7)].mean()),
            "ordinary_rows_mse_gain": float(gain[ordinary].mean()),
            "airport_day_bootstrap_mse_gain_95pct": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "regime_contributions": {r: float(gain[m].sum() / len(gain)) for r, m in regimes.items()},
            "airport_contributions": {a: float(gain[pooled.airport.to_numpy() == a].sum() / len(gain))
                                      for a in sorted(pooled.airport.unique())},
            "rows_changed_over_60s": int(np.sum(np.abs(candidate - base) > 60)),
            "mechanical_leaderboard_rmse_if_full_transfer_not_forecast":
                float(np.sqrt(max(V48 ** 2 - SHARE_2026 * float(gain.mean()), 0)))}


report = {"validation": "fallback-only six-pair screen against the scored v46 fallback recipe; not a leaderboard forecast",
          "columns": RS.MERKMALE,
          "rmse": {k: float(np.sqrt(np.mean((pooled[k].to_numpy() - y) ** 2))) for k in ("prod_anchor", "real", "permuted")},
          "real": compare(pooled.real.to_numpy()), "permuted": compare(pooled.permuted.to_numpy())}
checks = {"1_pooled_gain_positive": report["real"]["pooled_mse_gain"] > 0,
          "2_survives_top100_removal": report["real"]["mse_gain_without_top100_rows"] > 0,
          "3_at_least_4_of_6_pairs": report["real"]["pairs_positive"] >= 4,
          "4_january_and_july_gain_in_1_7": report["real"]["january_gain_1_7"] > 0 and report["real"]["july_gain_1_7"] > 0,
          "5_beats_permutation_control": report["real"]["pooled_mse_gain"] > report["permuted"]["pooled_mse_gain"],
          "6_ordinary_rows_not_worse": report["real"]["ordinary_rows_mse_gain"] >= 0}
report["checks"], report["advance"] = checks, bool(all(checks.values()))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print("\nRMSE", {k: round(v, 3) for k, v in report["rmse"].items()})
for name in ("real", "permuted"):
    e = report[name]
    print(name, json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in e.items()
                            if k in ("pooled_mse_gain", "mse_gain_without_top100_rows", "pairs_positive",
                                     "january_gain_1_7", "july_gain_1_7", "ordinary_rows_mse_gain",
                                     "rows_changed_over_60s")}))
    print("   regimes", {r: round(v) for r, v in e["regime_contributions"].items()},
          "| boot", [round(x) for x in e["airport_day_bootstrap_mse_gain_95pct"]])
print("AIRPORTS", {a: round(v) for a, v in report["real"]["airport_contributions"].items()})
print("CHECKS", json.dumps(checks), "ADVANCE", report["advance"])

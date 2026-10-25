"""Pre-registered six-pair fallback screen: the operator copy rate as a fallback input. Rules fixed 16.09.2026.

Why. The fallback lane carries 37 % of squared error, and Rome carries most of the mechanisms inside it: 722 of its
1,488 fallback rows in 2025 are schedule copies, where the block time is the scheduled time and the label is exactly
the delay, with a median of 7,534 s against 1,070 s on its non-copies. Splitting those two groups is worth more than
any regression improvement on the group as a whole.

What is new here, and it is one column. On fallback rows every `_flt` column is absent, so the model never sees the
operator. `FLIGHT_mvt` survives, and its letter prefix is the operator. Measured 16.09.2026
(research_280/rome_identity_copy_rate.py): a copy rate per (airport, prefix) learned on months 1-6 separates months
7-12 with AUC 0.807, and 0.790 in the reverse direction, against 0.656 and 0.684 for schedule delay alone and 0.570
and 0.558 for hour alone. Top quintile 84 % copies, bottom quintile 5.7 %, about 10 % of held-out rows carry an
unseen prefix. The rate is label-derived, so it is computed out of fold exactly as features_stand_rate.py computes
the stand extreme rate: per learning month from the other learning months, and the full learning table for the pair.

Already rejected, not repeated: a dedicated copy classifier on the existing inputs (research_270), the NM taxi-time
source as a copy marker (refuted 16.09.2026), the own-arrival copy flag, the inbound-stand flag, the day-shift
component and the soft copy gate (research_260).

Baseline prod_anchor: the scored v46 fallback recipe rebuilt per pair; GATE reproduces
research_240/fallback_anchor_production.py within 1e-6 s. Candidate: the identical recipe with `q_opcopy` and
`q_opcopy_n` appended to the fallback inputs, so the existing extreme classifier, height model, normal model and
anchored offset all see the operator's copy history. No other change, no hyperparameter search.

Decision rule against prod_anchor, pooled over six pairs:
  1. pooled MSE gain positive
  2. positive without its 100 most favourable rows
  3. at least 4 of 6 pairs gain
  4. January and July both gain inside pair 1_7
  5. beats the permutation control, which shuffles the prefix within airport and month before the rate is built
An airport-day bootstrap (2,000 resamples, seed 719) and the per-regime and per-airport contributions are reported
either way. Fallback-only; not a leaderboard forecast, and nothing is built or uploaded here.

Run from the repository root:  python3 -u -B research_280/rome_operator_copy_screen.py
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
import features_operator_copy as OPC
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
OUTPUT = project / "research_280/results/rome_operator_copy_screen.json"
V48, SHARE_2026 = 266.4508, 5290 / 344841
index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet")
         for s in TABLES.glob("rescreen_*.json")}


def flight_numbers():
    frames = []
    for path in sorted(glob.glob(str(project / "data/training_2025-*.parquet"))):
        frames.append(pq.ParquetFile(path).read(columns=["MVT_ID_mvt", "FLIGHT_mvt"], use_threads=False).to_pandas())
    table = pd.concat(frames, ignore_index=True).drop_duplicates("MVT_ID_mvt")
    return table.set_index("MVT_ID_mvt").FLIGHT_mvt


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                             min_child_samples=minimum, verbose=-1, n_jobs=args.jobs, random_state=0)


def classifier():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=args.jobs, random_state=0)


def recipe(tr, va, base_columns, offset_columns):
    """The v46 fallback recipe, unchanged apart from which columns the four models see."""
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


numbers = flight_numbers()
rng = np.random.default_rng(20260916)
frames = []
for pair in PAIRS:
    key = f"{pair[0]}_{pair[1]}"
    learn = [m for m in range(1, 13) if m not in pair]
    tr = pd.read_parquet(index[tuple(learn)]).reset_index(drop=True)
    va = pd.read_parquet(index[tuple(sorted(pair))]).reset_index(drop=True)
    for frame in (tr, va):
        frame["z"] = (frame.y > H.SCHWELLE_EXTREM).astype(int)
        frame["zk"] = (np.abs(frame.y.to_numpy(float) - 60 * frame.versp_min.to_numpy(float)) <= 60).astype(int)
        frame["op_praefix"] = OPC.praefix(frame.MVT_ID_mvt.map(numbers)).to_numpy()
    mapping = MSQ.passe_an(tr)
    tr, va = MSQ.ausser_faltung(tr), MSQ.wende_an(mapping, va)
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
        learn_rows = tr.copy()
        if name == "permuted":
            shuffled = learn_rows.groupby(["ADEP_mvt", "monat"], observed=True).op_praefix.transform(
                lambda s: rng.permutation(s.to_numpy()))
            learn_rows["op_praefix"] = shuffled.to_numpy()
        model = OPC.passe_an(learn_rows)
        learn_rows = OPC.ausser_faltung(learn_rows)
        held = OPC.wende_an(model, va)
        frame[name] = recipe(learn_rows, held, BASE + OPC.MERKMALE, OFFSET + OPC.MERKMALE)
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


def compare(candidate):
    gain = (base - y) ** 2 - (candidate - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    pair_gains = {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}
    boot = np.bincount(inverse, weights=gain)[draws].sum(axis=1) / cluster_rows[draws].sum(axis=1)
    out = {"pooled_mse_gain": float(gain.mean()), "mse_gain_without_top100_rows": float(gain[keep].mean()),
           "pair_gains": pair_gains, "pairs_positive": int(sum(g > 0 for g in pair_gains.values())),
           "january_gain_1_7": float(gain[(folds == "1_7") & (month == 1)].mean()),
           "july_gain_1_7": float(gain[(folds == "1_7") & (month == 7)].mean()),
           "airport_day_bootstrap_mse_gain_95pct": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
           "regime_contributions": {r: float(gain[m].sum() / len(gain)) for r, m in regimes.items()},
           "airport_contributions": {a: float(gain[pooled.airport.to_numpy() == a].sum() / len(gain))
                                     for a in sorted(pooled.airport.unique())},
           "rows_changed_over_60s": int(np.sum(np.abs(candidate - base) > 60)),
           "mechanical_leaderboard_rmse_if_full_transfer_not_forecast":
               float(np.sqrt(max(V48 ** 2 - SHARE_2026 * float(gain.mean()), 0)))}
    return out


report = {"validation": "fallback-only six-pair screen against the scored v46 fallback recipe; not a leaderboard forecast",
          "rmse": {k: float(np.sqrt(np.mean((pooled[k].to_numpy() - y) ** 2))) for k in ("prod_anchor", "real", "permuted")},
          "real": compare(pooled.real.to_numpy()), "permuted": compare(pooled.permuted.to_numpy())}
checks = {"1_pooled_gain_positive": report["real"]["pooled_mse_gain"] > 0,
          "2_survives_top100_removal": report["real"]["mse_gain_without_top100_rows"] > 0,
          "3_at_least_4_of_6_pairs": report["real"]["pairs_positive"] >= 4,
          "4_january_and_july_gain_in_1_7": report["real"]["january_gain_1_7"] > 0 and report["real"]["july_gain_1_7"] > 0,
          "5_beats_permutation_control": report["real"]["pooled_mse_gain"] > report["permuted"]["pooled_mse_gain"]}
report["checks"] = checks
report["advance"] = bool(all(checks.values()))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print("\nRMSE", {k: round(v, 3) for k, v in report["rmse"].items()})
for name in ("real", "permuted"):
    e = report[name]
    print(name, json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in e.items()
                            if k in ("pooled_mse_gain", "mse_gain_without_top100_rows", "pairs_positive",
                                     "january_gain_1_7", "july_gain_1_7", "rows_changed_over_60s")}),
          "| regimes", {r: round(v) for r, v in e["regime_contributions"].items()},
          "| boot", [round(x) for x in e["airport_day_bootstrap_mse_gain_95pct"]])
print("AIRPORTS", {a: round(v) for a, v in report["real"]["airport_contributions"].items()})
print("CHECKS", json.dumps(checks), "ADVANCE", report["advance"])

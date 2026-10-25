"""Pilot 6 (PREREG_stacker_ensemble.md): the ADS-B corrector on top of the held-out v45/v47 ensemble predictions.
Run from the repository root:  python3 -u -B research_730_adsb_learner/pilot6_screen.py
"""
import json
from pathlib import Path
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from research_730_adsb_learner import pilot_screen as P
from research_730_adsb_learner import pilot5_screen as S
from research_300 import incremental as I

CACHE, ROOT = P.CACHE, P.ROOT
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot6_screen.json"
ENSEMBLE = ROOT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
PILOT5_GAIN = 17.63


def main():
    table = pd.read_parquet(CACHE / "table_1_7_stand.parquet")
    e = pd.read_parquet(ENSEMBLE, columns=["MVT_ID_mvt", "y", "baseline"])
    assert np.array_equal(e.MVT_ID_mvt.to_numpy(), table.MVT_ID_mvt.to_numpy()) and np.allclose(e.y.to_numpy(), table.y.to_numpy())
    table["p1"] = e.baseline.to_numpy(float)
    date = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    fold = date.map(lambda d: zlib.crc32(d.encode()) % P.FOLDS).to_numpy()
    table["_day"] = table.ADEP_mvt.astype(str) + "|" + date
    airport = table.ADEP_mvt.astype(str).to_numpy()
    month = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    y = table.y.to_numpy(float)
    preds = {}
    for seed in P.SEEDS:
        preds[("base", seed)] = table.p1.to_numpy().copy()
        preds[("stack", seed)] = S.stage2(table, fold, S.STAGE2_BASE + S.AD18, seed, "ens_stack")
        preds[("stack0", seed)] = S.stage2(table, fold, S.STAGE2_BASE, seed, "ens_stack0")
        for k in P.CONTROLS:
            ctrl = P.permute_within(table, S.AD18, table._day, 11000 + k)
            preds[(f"control_{k}", seed)] = S.stage2(ctrl, fold, S.STAGE2_BASE + S.AD18, seed, f"ens_control_{k}")
    loss = {key: (p - y) ** 2 for key, p in preds.items()}
    mean = lambda name: np.mean([loss[(name, s)] for s in P.SEEDS], axis=0)
    stack, stack0, base = mean("stack"), mean("stack0"), mean("base")
    ctrl = np.mean([loss[(f"control_{k}", s)] for s in P.SEEDS for k in P.CONTROLS], axis=0)
    diff = ctrl - stack
    rmse = lambda l: float(np.sqrt(l.mean()))
    per_seed = {s: float(np.mean([loss[(f"control_{k}", s)].mean() for k in P.CONTROLS]) - loss[("stack", s)].mean()) for s in P.SEEDS}
    report = {"prereg": "research_730_adsb_learner/PREREG_stacker_ensemble.md", "rows": int(len(y)),
              "rmse": {"stack": rmse(stack), "mean_control": rmse(ctrl), "stack0": rmse(stack0), "base_ensemble": rmse(base)},
              "rmse_gain_vs_control_s": rmse(ctrl) - rmse(stack), "rmse_gain_vs_base_s": rmse(base) - rmse(stack),
              "share_of_pilot5_gain": (rmse(base) - rmse(stack)) / PILOT5_GAIN,
              "per_seed_mse_gain": per_seed, "months": {str(m): float(diff[month == m].mean()) for m in (1, 7)},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, table._day.to_numpy()),
              "airports": {a: {"rows": int((airport == a).sum()), "vs_control": float(diff[airport == a].mean()),
                               "vs_base": float((base - stack)[airport == a].mean())} for a in sorted(set(airport))}}
    checks = {"1_every_seed_beats_mean_control": all(v > 0 for v in per_seed.values()),
              "2_both_months_positive": all(v > 0 for v in report["months"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_rmse_gain_vs_control_at_least_0_3_s": report["rmse_gain_vs_control_s"] >= 0.3,
              "6_stack_beats_base": report["rmse_gain_vs_base_s"] > 0}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

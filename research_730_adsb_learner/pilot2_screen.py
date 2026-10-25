"""Pilot 2 (PREREG_stand.md): the stand readout on top of the per-flight ADS-B features.

Same rows, folds, seeds and hyperparameters as pilot_screen.py; its cached table gets the six sd_ columns, and its cached
out-of-fold predictions of the 'real' arm (91 + 12 ADS-B) are the comparison. New arms: real2 (real + sd_), control2_k (real +
sd_ permuted jointly within airport and day). Run from the repository root:  python3 -u -B research_730_adsb_learner/pilot2_screen.py
"""
import glob
import json
from pathlib import Path
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from research_730_adsb_learner import features_adsb as F
from research_730_adsb_learner import pilot_screen as P
from research_300 import incremental as I

CACHE, TRACKS = P.CACHE, P.TRACKS
OUTPUT = P.ROOT / "research_730_adsb_learner/results/pilot2_screen.json"
SD = list(F.STAND_MERKMALE)


def build_table():
    cached = CACHE / "table_1_7_stand.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    table = pd.read_parquet(CACHE / "table_1_7.parquet")
    files = sorted(glob.glob(str(TRACKS / "2025-01-??.parquet")) + glob.glob(str(TRACKS / "2025-07-??.parquet")))
    tracks = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = F.compute(table, tracks, stands=True)
    assert np.array_equal(out.index.to_numpy(), table.MVT_ID_mvt.to_numpy())
    for c in SD:
        table[c] = out[c].to_numpy()
    table.to_parquet(cached, index=False)
    return table


def main():
    table = build_table()
    date = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    fold = date.map(lambda d: zlib.crc32(d.encode()) % P.FOLDS).to_numpy()
    table["_day"] = table.ADEP_mvt.astype(str) + "|" + date
    airport = table.ADEP_mvt.astype(str).to_numpy()
    month = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    ok = table.sd_status.eq(1).to_numpy()
    cover = {a: {"rows": int((airport == a).sum()), "sd_ok": float(ok[airport == a].mean())} for a in sorted(set(airport))}
    status = table.sd_status.value_counts().sort_index().to_dict()
    P.log("sd_ok share", round(float(ok.mean()), 3), "status counts", json.dumps({int(k): int(v) for k, v in status.items()}), json.dumps(cover))
    preds, y = {}, table.y.to_numpy(float)
    for seed in P.SEEDS:
        preds[("real", seed)] = np.load(CACHE / f"oof_real_s{seed}.npy")
        preds[("real2", seed)] = P.fit_predict(table, fold, P.BASE + P.AD + SD, seed, "real2")
        for k in P.CONTROLS:
            ctrl = P.permute_within(table, SD, table._day, 3000 + k)
            preds[(f"control2_{k}", seed)] = P.fit_predict(ctrl, fold, P.BASE + P.AD + SD, seed, f"control2_{k}")
    loss = {key: (p - y) ** 2 for key, p in preds.items()}
    real2 = np.mean([loss[("real2", s)] for s in P.SEEDS], axis=0)
    real = np.mean([loss[("real", s)] for s in P.SEEDS], axis=0)
    ctrl = np.mean([loss[(f"control2_{k}", s)] for s in P.SEEDS for k in P.CONTROLS], axis=0)
    diff = ctrl - real2
    rmse = lambda l: float(np.sqrt(l.mean()))
    per_seed = {s: float(np.mean([loss[(f"control2_{k}", s)].mean() for k in P.CONTROLS]) - loss[("real2", s)].mean()) for s in P.SEEDS}
    report = {"prereg": "research_730_adsb_learner/PREREG_stand.md", "rows": int(len(y)), "sd_ok_share": float(ok.mean()),
              "status_counts": {int(k): int(v) for k, v in status.items()}, "coverage": cover,
              "rmse": {"real2": rmse(real2), "mean_control2": rmse(ctrl), "real": rmse(real)},
              "rmse_gain_vs_control_s": rmse(ctrl) - rmse(real2), "rmse_gain_vs_real_s": rmse(real) - rmse(real2),
              "mse_gain_per_row": float(diff.mean()), "per_seed_mse_gain": per_seed,
              "months": {str(m): float(diff[month == m].mean()) for m in (1, 7)},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, table._day.to_numpy()),
              "airports": {a: {"rows": int((airport == a).sum()), "vs_control": float(diff[airport == a].mean()),
                               "vs_real": float((real - real2)[airport == a].mean())} for a in sorted(set(airport))},
              "sd_ok_rows": {"rows": int(ok.sum()), "vs_control": float(diff[ok].mean()), "vs_real": float((real - real2)[ok].mean())},
              "other_rows": {"rows": int((~ok).sum()), "vs_control": float(diff[~ok].mean()), "vs_real": float((real - real2)[~ok].mean())}}
    checks = {"1_every_seed_beats_mean_control": all(v > 0 for v in per_seed.values()),
              "2_both_months_positive": all(v > 0 for v in report["months"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_rmse_gain_vs_control_at_least_0_3_s": report["rmse_gain_vs_control_s"] >= 0.3,
              "6_real2_beats_real": report["rmse_gain_vs_real_s"] > 0}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

"""Pilot 3 (PREREG_arrival_stands.md): stand positions learned from arriving aircraft.

Two stages so the learned-stand coverage is seen before any fit:
  python3 -u -B research_730_adsb_learner/pilot3_screen.py prepare   learn the table, build the sl_ columns, print coverage, no fit
  python3 -u -B research_730_adsb_learner/pilot3_screen.py fit       the 14 arm/seed runs and the verdict
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
from research_730_adsb_learner import features_stand_learned as L
from research_730_adsb_learner import pilot_screen as P
from research_300 import incremental as I

CACHE, TRACKS, ROOT = P.CACHE, P.TRACKS, P.ROOT
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot3_screen.json"
SD = list(F.STAND_MERKMALE)
SL = [c.replace("sd_", "sl_") for c in SD]


def tracks_1_7():
    files = sorted(glob.glob(str(TRACKS / "2025-01-??.parquet")) + glob.glob(str(TRACKS / "2025-07-??.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def prepare():
    cached = CACHE / "table_1_7_stand_learned.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    sd = _stand_detector()
    tracks = tracks_1_7()
    arr = pd.concat([pd.read_parquet(p, columns=["PHASE_mvt", "ADES_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "STAND_mvt"])
                     for p in sorted(glob.glob(str(ROOT / "data/training_2025-0[17]-01_*.parquet")))], ignore_index=True)
    arr = arr[arr.PHASE_mvt.eq("ARR")]
    points = L.parked_positions(arr, tracks)
    learned = L.learn_table(points, sd.FG.stand_schluessel)
    P.log(f"{len(arr):,} arrivals, {len(points):,} with a parked position")
    per = {}
    for airport in sorted(set(points.airport)):
        osm = len(sd.stand_table(airport)) if (sd.GEOMETRY / f"{airport.lower()}_osm.json").exists() else 0
        seen = points[points.airport == airport].stand.map(sd.FG.stand_schluessel).nunique()
        per[airport] = {"stands_seen": int(seen), "learned": len(learned.get(airport, {})), "osm_keys": int(osm)}
    P.log("learned stands per airport", json.dumps(per))
    (CACHE / "learned_stands.json").write_text(json.dumps(per, indent=1))
    table = pd.read_parquet(CACHE / "table_1_7_stand.parquet")
    out = F.compute(table, tracks, stands=True, stand_tables=learned)
    assert np.array_equal(out.index.to_numpy(), table.MVT_ID_mvt.to_numpy())
    for c, n in zip(SD, SL):
        table[n] = out[c].to_numpy()
    table.to_parquet(cached, index=False)
    return table


def _stand_detector():
    return F._stand_module()


def fit():
    table = prepare()
    date = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    fold = date.map(lambda d: zlib.crc32(d.encode()) % P.FOLDS).to_numpy()
    table["_day"] = table.ADEP_mvt.astype(str) + "|" + date
    airport = table.ADEP_mvt.astype(str).to_numpy()
    month = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    ok, osm_ok = table.sl_status.eq(1).to_numpy(), table.sd_status.eq(1).to_numpy()
    P.log("sl_ ok share", round(float(ok.mean()), 3), "| osm ok share", round(float(osm_ok.mean()), 3), "| status", json.dumps({int(k): int(v) for k, v in table.sl_status.value_counts().sort_index().items()}))
    preds, y = {}, table.y.to_numpy(float)
    base_features = P.BASE + P.AD + SD
    for seed in P.SEEDS:
        preds[("real2", seed)] = np.load(CACHE / f"oof_real2_s{seed}.npy")
        preds[("real3", seed)] = P.fit_predict(table, fold, base_features + SL, seed, "real3")
        for k in P.CONTROLS:
            ctrl = P.permute_within(table, SL, table._day, 5000 + k)
            preds[(f"control3_{k}", seed)] = P.fit_predict(ctrl, fold, base_features + SL, seed, f"control3_{k}")
    loss = {key: (p - y) ** 2 for key, p in preds.items()}
    real3 = np.mean([loss[("real3", s)] for s in P.SEEDS], axis=0)
    real2 = np.mean([loss[("real2", s)] for s in P.SEEDS], axis=0)
    ctrl = np.mean([loss[(f"control3_{k}", s)] for s in P.SEEDS for k in P.CONTROLS], axis=0)
    diff = ctrl - real3
    rmse = lambda l: float(np.sqrt(l.mean()))
    per_seed = {s: float(np.mean([loss[(f"control3_{k}", s)].mean() for k in P.CONTROLS]) - loss[("real3", s)].mean()) for s in P.SEEDS}
    abstain = ~osm_ok
    report = {"prereg": "research_730_adsb_learner/PREREG_arrival_stands.md", "rows": int(len(y)), "sl_ok_share": float(ok.mean()), "osm_ok_share": float(osm_ok.mean()),
              "learned_stands": json.loads((CACHE / "learned_stands.json").read_text()),
              "rmse": {"real3": rmse(real3), "mean_control3": rmse(ctrl), "real2": rmse(real2)},
              "rmse_gain_vs_control_s": rmse(ctrl) - rmse(real3), "rmse_gain_vs_real2_s": rmse(real2) - rmse(real3),
              "mse_gain_per_row": float(diff.mean()), "per_seed_mse_gain": per_seed,
              "months": {str(m): float(diff[month == m].mean()) for m in (1, 7)},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, table._day.to_numpy()),
              "airports": {a: {"rows": int((airport == a).sum()), "sl_ok": float(ok[airport == a].mean()), "vs_control": float(diff[airport == a].mean()),
                               "vs_real2": float((real2 - real3)[airport == a].mean())} for a in sorted(set(airport))},
              "osm_abstains": {"rows": int(abstain.sum()), "vs_control": float(diff[abstain].mean()), "vs_real2": float((real2 - real3)[abstain].mean())}}
    checks = {"1_every_seed_beats_mean_control": all(v > 0 for v in per_seed.values()),
              "2_both_months_positive": all(v > 0 for v in report["months"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_rmse_gain_vs_control_at_least_0_3_s": report["rmse_gain_vs_control_s"] >= 0.3,
              "6_real3_beats_real2": report["rmse_gain_vs_real2_s"] > 0}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "fit": fit}[sys.argv[1]]()

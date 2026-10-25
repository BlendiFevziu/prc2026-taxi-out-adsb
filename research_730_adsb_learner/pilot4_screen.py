"""Pilot 4 (PREREG_allrows.md): the ADS-B and stand columns on all rows of January and July 2025, fallback lane included.
Run from the repository root:  python3 -u -B research_730_adsb_learner/pilot4_screen.py
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

CACHE, TRACKS, ROOT = P.CACHE, P.TRACKS, P.ROOT
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot4_screen.json"
SD = list(F.STAND_MERKMALE)
AD18 = P.AD + SD
LABEL_CAP = 20000.0


def prefix_map(dep):
    """Two-letter airline code -> most frequent ICAO prefix, from departures whose callsign ends with the flight digits (PREREG addendum A)."""
    d = dep[dep.CALLSIGN_flt.notna() & dep.FLIGHT_mvt.notna()].reset_index(drop=True)
    same = np.array([c.endswith(f[2:]) for c, f in zip(d.CALLSIGN_flt, d.FLIGHT_mvt)])
    d = d[same]
    pair = pd.DataFrame({"iata": d.FLIGHT_mvt.str[:2], "icao": d.CALLSIGN_flt.str[:3]})
    return pair.groupby("iata").icao.agg(lambda s: s.value_counts().index[0]).to_dict()


def derive_callsign(frame, prefix):
    """CALLSIGN_flt where present, else ICAO prefix of the two-letter code plus the flight digits; NaN when the code is unknown."""
    flight = frame.FLIGHT_mvt.astype(object)
    derived = [prefix[f[:2]] + f[2:] if isinstance(f, str) and f[:2] in prefix else np.nan for f in flight]
    return frame.CALLSIGN_flt.where(frame.CALLSIGN_flt.notna(), pd.Series(derived, index=frame.index))


def build_table():
    cached = CACHE / "table_1_7_all.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    frame = pd.read_parquet(P.RELEASE, columns=list(dict.fromkeys(P.BASE + P.EXTRA)))
    frame = frame[frame.y.notna()].reset_index(drop=True)
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "PHASE_mvt", "CALLSIGN_flt", "FLIGHT_mvt"])
                     for p in sorted(glob.glob(str(ROOT / "data/training_2025-*.parquet")))], ignore_index=True)
    prefix = prefix_map(raw[raw.PHASE_mvt.eq("DEP")])
    raw = raw.drop_duplicates("MVT_ID_mvt")
    frame = frame.merge(raw[["MVT_ID_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]], on="MVT_ID_mvt", how="left")
    frame["CALLSIGN_flt"] = derive_callsign(frame, prefix)
    files = sorted(glob.glob(str(TRACKS / "2025-01-??.parquet")) + glob.glob(str(TRACKS / "2025-07-??.parquet")))
    tracks = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    P.log(f"{len(frame):,} departures (main lane {int(frame.hat_aobt.eq(1).sum()):,}, fallback {int(frame.hat_aobt.eq(0).sum()):,})")
    out = F.compute(frame, tracks, stands=True)
    assert np.array_equal(out.index.to_numpy(), frame.MVT_ID_mvt.to_numpy())
    for c in AD18:
        frame[c] = out[c].to_numpy()
    frame.to_parquet(cached, index=False)
    return frame


def stats(loss_real, loss_ctrl, loss_base, mask):
    d = loss_ctrl[mask] - loss_real[mask]
    rm = lambda l: float(np.sqrt(l[mask].mean()))
    return {"rows": int(mask.sum()), "rmse_real": rm(loss_real), "rmse_control": rm(loss_ctrl), "rmse_base": rm(loss_base),
            "gain_vs_control_s": rm(loss_ctrl) - rm(loss_real), "gain_vs_base_s": rm(loss_base) - rm(loss_real), "mse_gain_vs_control": float(d.mean())}


def main():
    table = build_table()
    date = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    fold = date.map(lambda d: zlib.crc32(d.encode()) % P.FOLDS).to_numpy()
    table["_day"] = table.ADEP_mvt.astype(str) + "|" + date
    airport = table.ADEP_mvt.astype(str).to_numpy()
    month = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    fallback = table.hat_aobt.eq(0).to_numpy()
    y = table.y.to_numpy(float)
    P.log("rows", len(table), "fallback", int(fallback.sum()), "matched share main/fallback",
          round(float(table.ad_matched[~fallback].mean()), 3), round(float(table.ad_matched[fallback].mean()), 3))
    preds = {}
    for seed in P.SEEDS:
        preds[("base", seed)] = P.fit_predict(table, fold, P.BASE, seed, "all_base")
        preds[("real", seed)] = P.fit_predict(table, fold, P.BASE + AD18, seed, "all_real")
        for k in P.CONTROLS:
            ctrl = P.permute_within(table, AD18, table._day, 7000 + k)
            preds[(f"control_{k}", seed)] = P.fit_predict(ctrl, fold, P.BASE + AD18, seed, f"all_control_{k}")
    loss = {key: (p - y) ** 2 for key, p in preds.items()}
    real = np.mean([loss[("real", s)] for s in P.SEEDS], axis=0)
    base = np.mean([loss[("base", s)] for s in P.SEEDS], axis=0)
    ctrl = np.mean([loss[(f"control_{k}", s)] for s in P.SEEDS for k in P.CONTROLS], axis=0)
    judged = y <= LABEL_CAP
    diff = (ctrl - real)[judged]
    per_seed = {s: float(np.mean([loss[(f"control_{k}", s)][judged].mean() for k in P.CONTROLS]) - loss[("real", s)][judged].mean()) for s in P.SEEDS}
    rm = lambda l, m: float(np.sqrt(l[m].mean()))
    report = {"prereg": "research_730_adsb_learner/PREREG_allrows.md", "rows": int(len(y)), "judged_rows": int(judged.sum()), "label_cap_s": LABEL_CAP,
              "all_rows": stats(real, ctrl, base, np.ones(len(y), bool)), "judged": stats(real, ctrl, base, judged),
              "fallback_judged": stats(real, ctrl, base, judged & fallback), "main_judged": stats(real, ctrl, base, judged & ~fallback),
              "fallback_all_labels": stats(real, ctrl, base, fallback),
              "per_seed_mse_gain": per_seed,
              "months": {str(m): float((ctrl - real)[judged & (month == m)].mean()) for m in (1, 7)},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, table._day.to_numpy()[judged]),
              "airports_judged": {a: stats(real, ctrl, base, judged & (airport == a)) for a in sorted(set(airport))}}
    checks = {"1_every_seed_beats_mean_control": all(v > 0 for v in per_seed.values()),
              "2_both_months_positive": all(v > 0 for v in report["months"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_rmse_gain_vs_control_at_least_0_3_s": report["judged"]["gain_vs_control_s"] >= 0.3,
              "6_real_beats_the_base": report["judged"]["gain_vs_base_s"] > 0}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "airports_judged"}, indent=1))


if __name__ == "__main__":
    main()

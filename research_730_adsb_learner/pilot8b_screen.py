"""Pilot 8b (PREREG_daily_reconcile_v2.md): daily reconciliation without LIRF, on February and June 2025, over pilot 7's
cross-month corrector (direction 1_7 -> 2_6, mean of its two seeds). Same method as pilot8_screen.py; LIRF rows never move.
Run from the repository root after pilot 7:  python3 -u -B research_730_adsb_learner/pilot8b_screen.py
"""
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from research_300 import incremental as I
from research_730_adsb_learner.pilot8_screen import shifts, LAMBDAS, CAP_S, HEAVY_S

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/research_730"
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot8b_screen.json"
MONTHS = (2, 6)


def main():
    t = pd.read_parquet(CACHE / "table_2_6_stand.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y"])
    t["p"] = np.mean([np.load(CACHE / f"transfer_17to26_stack_s{s}.npy") for s in (177, 277)], axis=0)
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
                     for p in sorted(glob.glob(str(ROOT / "data/training_2025-0[26]-01_*.parquet")))], ignore_index=True)
    dep = raw[raw.PHASE_mvt.eq("DEP") & raw.TAXITIME_SEC_mvt.notna()].drop_duplicates("MVT_ID_mvt").copy()
    mvt, sched = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True), pd.to_datetime(dep.SCHED_TIME_UTC_mvt, utc=True)
    dep["G"] = (mvt - sched).dt.total_seconds()
    dep["date"], dep["month"] = mvt.dt.strftime("%Y-%m-%d").to_numpy(), mvt.dt.month.to_numpy()
    dep["key"] = dep.ADEP_mvt.astype(str) + "|" + dep.date
    main_p = t.set_index("MVT_ID_mvt").p
    dep["main"] = dep.MVT_ID_mvt.isin(main_p.index)
    dep["p"] = dep.MVT_ID_mvt.map(main_p)
    med = dep[dep.main].groupby(["ADEP_mvt", "month"]).p.median()
    dep["p"] = dep.p.fillna(pd.Series([med.get((a, m), np.nan) for a, m in zip(dep.ADEP_mvt, dep.month)], index=dep.index))
    dep = dep[dep.p.notna() & dep.G.notna()]
    dep["movable"] = dep.main & dep.p.le(HEAVY_S) & dep.ADEP_mvt.ne("LIRF")          # the one change against pilot 8
    daily = pd.read_parquet(ROOT / ".cache/research_390_aiu/daily.parquet")
    pub = {f"{a}|{d}": 60.0 * v for a, d, v in zip(daily.airport, daily.date, daily["DLY_ALL_PRE_2__All_Pre-Departure_Delay"])}
    rows = dep[dep.main].set_index("MVT_ID_mvt").loc[t.MVT_ID_mvt]
    y, month = t.y.to_numpy(float), rows.month.to_numpy()
    rmse = lambda p, m=None: float(np.sqrt(np.mean(((p - y) ** 2) if m is None else ((p - y) ** 2)[m])))

    def run(published):
        star = rows.key.map(shifts(dep, published)).to_numpy(float)
        move = rows.movable.to_numpy()
        preds = {lam: np.maximum(rows.p.to_numpy() + np.where(move & np.isfinite(star), np.clip(lam * star, -CAP_S, CAP_S), 0.0), 300.0) for lam in LAMBDAS}
        chosen, out = {}, rows.p.to_numpy().copy()
        for m, other in ((2, 6), (6, 2)):
            best = min(LAMBDAS, key=lambda lam: rmse(preds[lam], month == other))
            chosen[m] = best
            out[month == m] = preds[best][month == m]
        return out, chosen

    real, chosen = run(pub)
    keys_by_am = {}
    for k in pub:
        a, d = k.split("|")
        keys_by_am.setdefault((a, d[:7]), []).append(k)
    ctrl_preds, ctrl_lams = [], []
    for c in (1, 2, 3):
        rng, perm = np.random.default_rng(16000 + c), {}
        for ks in keys_by_am.values():
            for k, v in zip(ks, rng.permutation([pub[k] for k in ks])):
                perm[k] = v
        p_c, l_c = run(perm)
        ctrl_preds.append(p_c); ctrl_lams.append(l_c)
    base = t.p.to_numpy()
    loss_real, loss_base = (real - y) ** 2, (base - y) ** 2
    loss_ctrl = np.mean([(p - y) ** 2 for p in ctrl_preds], axis=0)
    diff = loss_base - loss_real
    day = (t.ADEP_mvt.astype(str) + "|" + rows.date.to_numpy()).to_numpy()
    airport = t.ADEP_mvt.astype(str).to_numpy()
    report = {"prereg": "research_730_adsb_learner/PREREG_daily_reconcile_v2.md", "rows": int(len(y)),
              "rmse": {"reconciled": rmse(real), "mean_control": float(np.sqrt(loss_ctrl.mean())), "corrector": rmse(base)},
              "rmse_gain_vs_corrector_s": rmse(base) - rmse(real), "rmse_gain_vs_control_s": float(np.sqrt(loss_ctrl.mean())) - rmse(real),
              "lambda_chosen": {str(k): v for k, v in chosen.items()}, "control_lambdas": [{str(k): v for k, v in l.items()} for l in ctrl_lams],
              "months_gain": {str(m): float(diff[month == m].mean()) for m in MONTHS},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, day),
              "airports_gain": {a: float(diff[airport == a].mean()) for a in sorted(set(airport))}}
    checks = {"1_beats_mean_control": report["rmse_gain_vs_control_s"] > 0,
              "2_both_months_beat_corrector": all(v > 0 for v in report["months_gain"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_gain_at_least_0_3_s": report["rmse_gain_vs_corrector_s"] >= 0.3,
              "6_lambda_above_zero_both_months": all(v > 0 for v in chosen.values())}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

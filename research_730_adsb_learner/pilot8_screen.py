"""Pilot 8 (PREREG_daily_reconcile.md): reconcile each airport-day with EUROCONTROL's published all-causes pre-departure delay.
Run from the repository root:  python3 -u -B research_730_adsb_learner/pilot8_screen.py
"""
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from research_300 import incremental as I

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/research_730"
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot8_screen.json"
LAMBDAS, CAP_S, BOUND_S, HEAVY_S = (0.0, 0.25, 0.5, 0.75, 1.0), 300.0, 1800.0, 3600.0


def solve_shift(G, p, movable, D):
    """s* with sum max(0, G - p - s * movable) = D, by bisection on [-BOUND_S, BOUND_S] (clamped at the ends)."""
    f = lambda s: np.maximum(0.0, G - p - s * movable).sum() - D
    lo, hi = -BOUND_S, BOUND_S
    if f(lo) <= 0:
        return lo
    if f(hi) >= 0:
        return hi
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def shifts(frame, published):
    """s* per (airport, date) for the rows of frame (columns key, G, p, movable); NaN where no published figure."""
    out = {}
    for key, g in frame.groupby("key", sort=False):
        D = published.get(key)
        if D is None or not np.isfinite(D):
            continue
        out[key] = solve_shift(g.G.to_numpy(), g.p.to_numpy(), g.movable.to_numpy(float), D)
    return out


def main():
    t = pd.read_parquet(CACHE / "table_1_7_stand.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y"])
    stack = np.mean([np.load(CACHE / f"stack_ens_stack_s{s}.npy") for s in (177, 277)], axis=0)
    t["p"] = stack
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
                     for p in sorted(glob.glob(str(ROOT / "data/training_2025-0[17]-01_*.parquet")))], ignore_index=True)
    dep = raw[raw.PHASE_mvt.eq("DEP") & raw.TAXITIME_SEC_mvt.notna()].drop_duplicates("MVT_ID_mvt").copy()
    mvt, sched = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True), pd.to_datetime(dep.SCHED_TIME_UTC_mvt, utc=True)
    dep["G"] = (mvt - sched).dt.total_seconds()
    dep["date"], dep["month"] = mvt.dt.strftime("%Y-%m-%d").to_numpy(), mvt.dt.month.to_numpy()
    dep["key"] = dep.ADEP_mvt.astype(str) + "|" + dep.date
    main_p = t.set_index("MVT_ID_mvt").p
    dep["main"] = dep.MVT_ID_mvt.isin(main_p.index)
    dep["p"] = dep.MVT_ID_mvt.map(main_p)
    med = dep[dep.main].groupby(["ADEP_mvt", "month"]).p.median()
    fill = pd.Series([med.get((a, m), np.nan) for a, m in zip(dep.ADEP_mvt, dep.month)], index=dep.index)
    dep["p"] = dep.p.fillna(fill)
    dep = dep[dep.p.notna() & dep.G.notna()]
    dep["movable"] = dep.main & dep.p.le(HEAVY_S)
    daily = pd.read_parquet(ROOT / ".cache/research_390_aiu/daily.parquet")
    pub = {f"{a}|{d}": 60.0 * v for a, d, v in zip(daily.airport, daily.date, daily["DLY_ALL_PRE_2__All_Pre-Departure_Delay"])}
    print(f"departures {len(dep):,} (main {int(dep.main.sum()):,}); airport-days with a published figure {sum(k in pub for k in dep.key.unique())} of {dep.key.nunique()}", flush=True)

    rows = dep[dep.main].set_index("MVT_ID_mvt").loc[t.MVT_ID_mvt]
    y = t.y.to_numpy(float)
    month = rows.month.to_numpy()
    rmse = lambda p, m=None: float(np.sqrt(np.mean(((p - y) ** 2) if m is None else ((p - y) ** 2)[m])))

    def run(published):
        s = shifts(dep, published)
        star = rows.key.map(s).to_numpy(float)
        move = rows.movable.to_numpy()
        preds = {}
        for lam in LAMBDAS:
            shift = np.where(move & np.isfinite(star), np.clip(lam * star, -CAP_S, CAP_S), 0.0)
            preds[lam] = np.maximum(rows.p.to_numpy() + shift, 300.0)
        chosen, out = {}, rows.p.to_numpy().copy()
        for m, other in ((1, 7), (7, 1)):
            best = min(LAMBDAS, key=lambda lam: rmse(preds[lam], month == other))
            chosen[m] = best
            out[month == m] = preds[best][month == m]
        return out, chosen, star

    real, chosen, star = run(pub)
    ctrl_preds, ctrl_lams = [], []
    keys_by_am = {}
    for k in pub:
        a, d = k.split("|")
        keys_by_am.setdefault((a, d[:7]), []).append(k)
    for c in (1, 2, 3):
        rng, perm = np.random.default_rng(15000 + c), {}
        for ks in keys_by_am.values():
            vals = [pub[k] for k in ks]
            for k, v in zip(ks, rng.permutation(vals)):
                perm[k] = v
        p_c, l_c, _ = run(perm)
        ctrl_preds.append(p_c); ctrl_lams.append(l_c)
    base = t.p.to_numpy()
    loss_real, loss_base = (real - y) ** 2, (base - y) ** 2
    loss_ctrl = np.mean([(p - y) ** 2 for p in ctrl_preds], axis=0)
    diff = loss_base - loss_real
    day = (t.ADEP_mvt.astype(str) + "|" + rows.date.to_numpy()).to_numpy()
    airport = t.ADEP_mvt.astype(str).to_numpy()
    s_used = star[np.isfinite(star)]
    report = {"prereg": "research_730_adsb_learner/PREREG_daily_reconcile.md", "rows": int(len(y)),
              "rmse": {"reconciled": rmse(real), "mean_control": float(np.sqrt(loss_ctrl.mean())), "stack": rmse(base)},
              "rmse_gain_vs_stack_s": rmse(base) - rmse(real), "rmse_gain_vs_control_s": float(np.sqrt(loss_ctrl.mean())) - rmse(real),
              "lambda_chosen": {str(k): v for k, v in chosen.items()}, "control_lambdas": [{str(k): v for k, v in l.items()} for l in ctrl_lams],
              "months_gain_vs_stack": {str(m): float(diff[month == m].mean()) for m in (1, 7)},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, day),
              "shift_quantiles_s": [float(x) for x in np.quantile(s_used, [.05, .25, .5, .75, .95])],
              "airports_gain_vs_stack": {a: float(diff[airport == a].mean()) for a in sorted(set(airport))}}
    checks = {"1_beats_mean_control": report["rmse_gain_vs_control_s"] > 0,
              "2_both_months_beat_stack": all(v > 0 for v in report["months_gain_vs_stack"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_gain_vs_stack_at_least_0_3_s": report["rmse_gain_vs_stack_s"] >= 0.3,
              "6_lambda_above_zero_both_months": all(v > 0 for v in chosen.values())}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

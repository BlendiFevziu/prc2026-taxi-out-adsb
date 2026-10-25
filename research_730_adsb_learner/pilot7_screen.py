"""Pilot 7 (PREREG_transfer.md): the ADS-B corrector trained on one month pair of 2025 and applied to another.

  python3 -u -B research_730_adsb_learner/pilot7_screen.py stage1   leave-pair-out base predictions for (1, 7) and (2, 6); no tracks needed
  python3 -u -B research_730_adsb_learner/pilot7_screen.py table    ADS-B and stand columns for February and June (refuses until all 58 days exist)
  python3 -u -B research_730_adsb_learner/pilot7_screen.py fit      the cross-month corrector, controls and the verdict
"""
import glob
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main_model as H  # torch before tree libraries on macOS
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_730_adsb_learner import features_adsb as F
from research_730_adsb_learner import pilot_screen as P
from research_730_adsb_learner import pilot5_screen as S
from research_300 import incremental as I
from research_300.heavy_lock import heavy_work

CACHE, ROOT, TRACKS = P.CACHE, P.ROOT, P.TRACKS
ALL = ROOT / ".cache/clock_queue_release_20260912/data/months_1_2_3_4_5_6_7_8_9_10_11_12.parquet"
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot7_screen.json"
PAIRS = {"1_7": (1, 7), "2_6": (2, 6)}
STAGE1 = CACHE / "stage1_leave_pair_out.parquet"


def month_of(frame):
    return pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()


def stage1():
    if STAGE1.exists():
        P.log("stage 1 already cached"); return
    frame = pd.read_parquet(ALL, columns=list(dict.fromkeys(P.BASE + P.EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    month = month_of(frame)
    out = []
    for name, pair in PAIRS.items():
        held_mask = np.isin(month, pair)
        learn, held = frame[~held_mask].copy(), frame[held_mask].copy()
        for column in H.KAT:
            if column in P.BASE:
                cats = pd.unique(learn[column].astype(str))
                learn[column] = pd.Categorical(learn[column].astype(str), categories=cats)
                held[column] = pd.Categorical(held[column].astype(str), categories=cats)
        res = pd.DataFrame({"MVT_ID_mvt": held.MVT_ID_mvt.to_numpy(), "pair": name})
        for seed in P.SEEDS:
            with heavy_work(f"research_730 pilot 7 stage 1 {name} seed {seed}", log=P.log):
                t0 = time.time()
                model = lgb.LGBMRegressor(objective="l2", n_estimators=P.ROUNDS, learning_rate=.03, num_leaves=255, min_child_samples=50,
                                          subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)
                model.fit(learn[P.BASE], (learn.y - learn.ersatz_s).to_numpy())
                res[f"p1_s{seed}"] = np.maximum(held.ersatz_s.to_numpy(float) + model.predict(held[P.BASE]), 300)
            P.log(f"stage 1 {name} seed {seed}: {time.time() - t0:.0f} s, RMSE {np.sqrt(np.mean((res[f'p1_s{seed}'] - held.y.to_numpy()) ** 2)):.3f}")
        out.append(res)
    pd.concat(out, ignore_index=True).to_parquet(STAGE1, index=False)


def table():
    cached = CACHE / "table_2_6_stand.parquet"
    if cached.exists():
        P.log("table already cached"); return
    files = sorted(glob.glob(str(TRACKS / "2025-02-??.parquet")) + glob.glob(str(TRACKS / "2025-06-??.parquet")))
    if len(files) != 58:
        raise SystemExit(f"only {len(files)} of 58 February/June 2025 track days; wait for the downloads")
    frame = pd.read_parquet(ALL, columns=list(dict.fromkeys(P.BASE + P.EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna() & np.isin(month_of(frame), (2, 6))].reset_index(drop=True)
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "CALLSIGN_flt"])
                     for p in sorted(glob.glob(str(ROOT / "data/training_2025-0[26]-01_*.parquet")))], ignore_index=True).drop_duplicates("MVT_ID_mvt")
    frame = frame.merge(raw, on="MVT_ID_mvt", how="left")
    tracks = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = F.compute(frame, tracks, stands=True)
    assert np.array_equal(out.index.to_numpy(), frame.MVT_ID_mvt.to_numpy())
    for c in S.AD18:
        frame[c] = out[c].to_numpy()
    P.log(f"February/June table: {len(frame):,} rows, matched {frame.ad_matched.mean():.3f}, stand ok {frame.sd_status.eq(1).mean():.3f}")
    frame.to_parquet(cached, index=False)


def fit_apply(train, apply, features, seed, label):
    cache = CACHE / f"transfer_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    train, apply = train.copy(), apply.copy()
    cats = pd.unique(train.ADEP_mvt.astype(str))
    train["ADEP_mvt"] = pd.Categorical(train.ADEP_mvt.astype(str), categories=cats)
    apply["ADEP_mvt"] = pd.Categorical(apply.ADEP_mvt.astype(str), categories=cats)
    with heavy_work(f"research_730 pilot 7 {label} seed {seed}", log=P.log):
        model = lgb.LGBMRegressor(objective="l2", n_estimators=S.ROUNDS, learning_rate=.03, num_leaves=63, min_child_samples=100,
                                  subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)
        model.fit(train[features], (train.y - train.p1).to_numpy())
        pred = np.maximum(apply.p1.to_numpy(float) + model.predict(apply[features]), 300)
    np.save(cache, pred)
    return pred


def fit():
    t17 = pd.read_parquet(CACHE / "table_1_7_stand.parquet")
    t26 = pd.read_parquet(CACHE / "table_2_6_stand.parquet")
    s1 = pd.read_parquet(STAGE1).set_index("MVT_ID_mvt")
    tables = {"1_7": t17, "2_6": t26}
    for t in tables.values():
        t["_day"] = t.ADEP_mvt.astype(str) + "|" + pd.to_datetime(t.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
        assert t.MVT_ID_mvt.isin(s1.index).all()
    held = pd.concat([t26.assign(_dir="1_7->2_6"), t17.assign(_dir="2_6->1_7")], ignore_index=True)
    y, month, airport = held.y.to_numpy(float), month_of(held), held.ADEP_mvt.astype(str).to_numpy()
    preds = {}
    for seed in P.SEEDS:
        for t in tables.values():
            t["p1"] = s1.loc[t.MVT_ID_mvt, f"p1_s{seed}"].to_numpy()
        base = np.concatenate([t26.p1.to_numpy(), t17.p1.to_numpy()])
        preds[("base", seed)] = base
        feats = S.STAGE2_BASE + S.AD18
        preds[("stack", seed)] = np.concatenate([fit_apply(t17, t26, feats, seed, "17to26_stack"), fit_apply(t26, t17, feats, seed, "26to17_stack")])
        for k in P.CONTROLS:
            c17 = P.permute_within(t17, S.AD18, t17._day, 13000 + k)
            c26 = P.permute_within(t26, S.AD18, t26._day, 14000 + k)
            preds[(f"control_{k}", seed)] = np.concatenate([fit_apply(c17, c26, feats, seed, f"17to26_control_{k}"),
                                                            fit_apply(c26, c17, feats, seed, f"26to17_control_{k}")])
        P.log(f"seed {seed}: base {np.sqrt(np.mean((base - y) ** 2)):.3f}, stack {np.sqrt(np.mean((preds[('stack', seed)] - y) ** 2)):.3f}")
    loss = {key: (p - y) ** 2 for key, p in preds.items()}
    mean = lambda name: np.mean([loss[(name, s)] for s in P.SEEDS], axis=0)
    stack, base = mean("stack"), mean("base")
    ctrl = np.mean([loss[(f"control_{k}", s)] for s in P.SEEDS for k in P.CONTROLS], axis=0)
    diff = ctrl - stack
    rmse = lambda l, m=None: float(np.sqrt(l.mean() if m is None else l[m].mean()))
    per_seed = {s: float(np.mean([loss[(f"control_{k}", s)].mean() for k in P.CONTROLS]) - loss[("stack", s)].mean()) for s in P.SEEDS}
    directions = {d: {"rows": int((held._dir == d).sum()), "stack": rmse(stack, (held._dir == d).to_numpy()), "control": rmse(ctrl, (held._dir == d).to_numpy()),
                      "base": rmse(base, (held._dir == d).to_numpy())} for d in ("1_7->2_6", "2_6->1_7")}
    report = {"prereg": "research_730_adsb_learner/PREREG_transfer.md", "rows": int(len(y)),
              "rmse": {"stack": rmse(stack), "mean_control": rmse(ctrl), "base": rmse(base)},
              "rmse_gain_vs_control_s": rmse(ctrl) - rmse(stack), "rmse_gain_vs_base_s": rmse(base) - rmse(stack),
              "directions": directions, "per_seed_mse_gain": per_seed,
              "months": {str(m): float(diff[month == m].mean()) for m in (1, 2, 6, 7)},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, held._day.to_numpy()),
              "airports": {a: {"rows": int((airport == a).sum()), "vs_base": float((base - stack)[airport == a].mean())} for a in sorted(set(airport))}}
    checks = {"1_every_seed_beats_mean_control": all(v > 0 for v in per_seed.values()),
              "2_each_of_four_months_positive": all(v > 0 for v in report["months"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_rmse_gain_vs_control_at_least_0_3_s": report["rmse_gain_vs_control_s"] >= 0.3,
              "6_stack_beats_base": report["rmse_gain_vs_base_s"] > 0}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"stage1": stage1, "table": table, "fit": fit}[sys.argv[1]]()

"""Pilot 9 (PREREG_more_months.md): the ADS-B corrector trained on five months against two.

  python3 -u -B research_730_adsb_learner/pilot9_screen.py stage1   leave-group-out base predictions for March, May, December; no tracks needed
  python3 -u -B research_730_adsb_learner/pilot9_screen.py table    ADS-B and stand columns for March, May, December (2025-12-31 dropped)
  python3 -u -B research_730_adsb_learner/pilot9_screen.py fit      parts A and B and the verdicts
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
from research_730_adsb_learner import pilot7_screen as Q
from research_300 import incremental as I
from research_300.heavy_lock import heavy_work

CACHE, ROOT, TRACKS = P.CACHE, P.ROOT, P.TRACKS
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot9_screen.json"
EXTRA_MONTHS = (3, 5, 12)
STAGE1 = CACHE / "stage1_leave_3_5_12.parquet"
TABLE = CACHE / "table_3_5_12_stand.parquet"
FEATURES = S.STAGE2_BASE + S.AD18


def stage1():
    if STAGE1.exists():
        P.log("stage 1 already cached"); return
    frame = pd.read_parquet(Q.ALL, columns=list(dict.fromkeys(P.BASE + P.EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna()].reset_index(drop=True)
    held_mask = np.isin(Q.month_of(frame), EXTRA_MONTHS)
    learn, held = frame[~held_mask].copy(), frame[held_mask].copy()
    for column in H.KAT:
        if column in P.BASE:
            cats = pd.unique(learn[column].astype(str))
            learn[column] = pd.Categorical(learn[column].astype(str), categories=cats)
            held[column] = pd.Categorical(held[column].astype(str), categories=cats)
    res = pd.DataFrame({"MVT_ID_mvt": held.MVT_ID_mvt.to_numpy()})
    for seed in P.SEEDS:
        with heavy_work(f"research_730 pilot 9 stage 1 seed {seed}", log=P.log):
            t0 = time.time()
            model = lgb.LGBMRegressor(objective="l2", n_estimators=P.ROUNDS, learning_rate=.03, num_leaves=255, min_child_samples=50,
                                      subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)
            model.fit(learn[P.BASE], (learn.y - learn.ersatz_s).to_numpy())
            res[f"p1_s{seed}"] = np.maximum(held.ersatz_s.to_numpy(float) + model.predict(held[P.BASE]), 300)
        P.log(f"stage 1 3_5_12 seed {seed}: {time.time() - t0:.0f} s, RMSE {np.sqrt(np.mean((res[f'p1_s{seed}'] - held.y.to_numpy()) ** 2)):.3f}")
    res.to_parquet(STAGE1, index=False)


def table():
    if TABLE.exists():
        P.log("table already cached"); return
    frame = pd.read_parquet(Q.ALL, columns=list(dict.fromkeys(P.BASE + P.EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna() & np.isin(Q.month_of(frame), EXTRA_MONTHS)].reset_index(drop=True)
    day = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    frame = frame[day.ne("2025-12-31").to_numpy()].reset_index(drop=True)
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "CALLSIGN_flt"])
                     for p in sorted(glob.glob(str(ROOT / "data/training_2025-0[35]-01_*.parquet")) + glob.glob(str(ROOT / "data/training_2025-12-01_*.parquet")))],
                    ignore_index=True).drop_duplicates("MVT_ID_mvt")
    frame = frame.merge(raw, on="MVT_ID_mvt", how="left")
    month = Q.month_of(frame)
    parts = []
    for m in EXTRA_MONTHS:
        files = sorted(glob.glob(str(TRACKS / f"2025-{m:02d}-??.parquet")))
        rows = frame[month == m].reset_index(drop=True)
        tracks = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        out = F.compute(rows, tracks, stands=True)
        assert np.array_equal(out.index.to_numpy(), rows.MVT_ID_mvt.to_numpy())
        for c in S.AD18:
            rows[c] = out[c].to_numpy()
        P.log(f"2025-{m:02d}: {len(rows):,} rows, {len(files)} track files, matched {rows.ad_matched.mean():.3f}, stand ok {rows.sd_status.eq(1).mean():.3f}")
        parts.append(rows)
        del tracks
    pd.concat(parts, ignore_index=True).to_parquet(TABLE, index=False)


def fit_apply(train, apply, seed, label):
    cache = CACHE / f"more_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    train, apply = train.copy(), apply.copy()
    cats = pd.unique(train.ADEP_mvt.astype(str))
    train["ADEP_mvt"] = pd.Categorical(train.ADEP_mvt.astype(str), categories=cats)
    apply["ADEP_mvt"] = pd.Categorical(apply.ADEP_mvt.astype(str), categories=cats)
    with heavy_work(f"research_730 pilot 9 {label} seed {seed}", log=P.log):
        model = lgb.LGBMRegressor(objective="l2", n_estimators=S.ROUNDS, learning_rate=.03, num_leaves=63, min_child_samples=100,
                                  subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)
        model.fit(train[FEATURES], (train.y - train.p1).to_numpy())
        pred = np.maximum(apply.p1.to_numpy(float) + model.predict(apply[FEATURES]), 300)
    np.save(cache, pred)
    return pred


def judge(y, month, day, new, old, base, seed_loss_new, seed_loss_old, months):
    diff = old - new
    rmse = lambda l: float(np.sqrt(l.mean()))
    per_seed = {s: float(seed_loss_old[s].mean() - seed_loss_new[s].mean()) for s in P.SEEDS}
    report = {"rows": int(len(y)), "rmse": {"new": rmse(new), "old": rmse(old), "base": rmse(base)},
              "rmse_gain_vs_old_s": rmse(old) - rmse(new), "rmse_gain_vs_base_s": rmse(base) - rmse(new), "per_seed_mse_gain": per_seed,
              "months": {str(m): float(diff[month == m].mean()) for m in months},
              "months_rmse": {str(m): {"new": rmse(new[month == m]), "old": rmse(old[month == m]), "base": rmse(base[month == m])} for m in months},
              "without_best_100": float(np.sort(diff)[:-100].mean()),
              "airport_day_bootstrap_95": I.block_bootstrap(diff, day)}
    checks = {"1_every_seed_new_beats_old": all(v > 0 for v in per_seed.values()),
              "2_each_month_positive": all(v > 0 for v in report["months"].values()),
              "3_without_best_100_positive": report["without_best_100"] > 0,
              "4_airport_day_interval_above_zero": report["airport_day_bootstrap_95"][0] > 0,
              "5_rmse_gain_at_least_0_3_s": report["rmse_gain_vs_old_s"] >= 0.3,
              "6_new_beats_base": report["rmse_gain_vs_base_s"] > 0}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    return report


def fit():
    t17 = pd.read_parquet(CACHE / "table_1_7_stand.parquet")
    t26 = pd.read_parquet(CACHE / "table_2_6_stand.parquet")
    tx = pd.read_parquet(TABLE)
    s1 = pd.concat([pd.read_parquet(Q.STAGE1).drop(columns="pair"), pd.read_parquet(STAGE1)]).set_index("MVT_ID_mvt")
    for t in (t17, t26, tx):
        t["_day"] = t.ADEP_mvt.astype(str) + "|" + pd.to_datetime(t.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
        assert t.MVT_ID_mvt.isin(s1.index).all()
    cols = list(dict.fromkeys([c for c in FEATURES if c != "p1"] + ["y", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "_day"]))
    t17, t26, tx = t17[cols], t26[cols], tx[cols]

    # part A: held-out pairs, stage-1 LightGBM p1 on both sides
    held = pd.concat([t26, t17], ignore_index=True)
    yA, monthA, dayA = held.y.to_numpy(float), Q.month_of(held), held._day.to_numpy()
    loss = {}
    for seed in P.SEEDS:
        for t in (t17, t26, tx):
            t["p1"] = s1.loc[t.MVT_ID_mvt, f"p1_s{seed}"].to_numpy()
        base = np.concatenate([t26.p1.to_numpy(), t17.p1.to_numpy()])
        two = np.concatenate([np.load(CACHE / f"transfer_17to26_stack_s{seed}.npy"), np.load(CACHE / f"transfer_26to17_stack_s{seed}.npy")])
        five = np.concatenate([fit_apply(pd.concat([t17, tx], ignore_index=True), t26, seed, "A_17x_to_26"),
                               fit_apply(pd.concat([t26, tx], ignore_index=True), t17, seed, "A_26x_to_17")])
        loss[("base", seed)], loss[("two", seed)], loss[("five", seed)] = (base - yA) ** 2, (two - yA) ** 2, (five - yA) ** 2
        P.log(f"part A seed {seed}: base {np.sqrt(loss[('base', seed)].mean()):.3f}, two {np.sqrt(loss[('two', seed)].mean()):.3f}, five {np.sqrt(loss[('five', seed)].mean()):.3f}")
    mean = lambda name: np.mean([loss[(name, s)] for s in P.SEEDS], axis=0)
    partA = judge(yA, monthA, dayA, mean("five"), mean("two"), mean("base"),
                  {s: loss[("five", s)] for s in P.SEEDS}, {s: loss[("two", s)] for s in P.SEEDS}, (1, 2, 6, 7))

    # part B: January and July with the ensemble's held-out p1, corrector trained on stage-1 LightGBM p1
    e = pd.read_parquet(ROOT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet", columns=["MVT_ID_mvt", "y", "baseline"])
    assert np.array_equal(e.MVT_ID_mvt.to_numpy(), t17.MVT_ID_mvt.to_numpy()) and np.allclose(e.y.to_numpy(), t17.y.to_numpy())
    apply = t17.copy()
    apply["p1"] = e.baseline.to_numpy(float)
    yB, monthB, dayB = apply.y.to_numpy(float), Q.month_of(apply), apply._day.to_numpy()
    baseB = (apply.p1.to_numpy() - yB) ** 2
    lossB = {}
    for seed in P.SEEDS:
        for t in (t26, tx):
            t["p1"] = s1.loc[t.MVT_ID_mvt, f"p1_s{seed}"].to_numpy()
        lossB[("two", seed)] = (fit_apply(t26, apply, seed, "B_26_to_ens17") - yB) ** 2
        lossB[("five", seed)] = (fit_apply(pd.concat([t26, tx], ignore_index=True), apply, seed, "B_26x_to_ens17") - yB) ** 2
        P.log(f"part B seed {seed}: base {np.sqrt(baseB.mean()):.3f}, two_ens {np.sqrt(lossB[('two', seed)].mean()):.3f}, five_ens {np.sqrt(lossB[('five', seed)].mean()):.3f}")
    meanB = lambda name: np.mean([lossB[(name, s)] for s in P.SEEDS], axis=0)
    partB = judge(yB, monthB, dayB, meanB("five"), meanB("two"), baseB,
                  {s: lossB[("five", s)] for s in P.SEEDS}, {s: lossB[("two", s)] for s in P.SEEDS}, (1, 7))
    partB["pilot6_stack_reference"] = 200.99213047248654
    partB["five_ens_below_pilot6"] = partB["rmse"]["new"] < partB["pilot6_stack_reference"]

    report = {"prereg": "research_730_adsb_learner/PREREG_more_months.md", "extra_table_rows": int(len(tx)),
              "extra_matched": float(tx.ad_matched.mean()) if "ad_matched" in tx else None,
              "part_A_five_vs_two": partA, "part_B_ensemble_form": partB,
              "rebuild_2026_indicated": bool(partB["verdict"] == "pass" and partB["five_ens_below_pilot6"]),
              "fetch_remaining_months_indicated": bool(partA["verdict"] == "pass")}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"stage1": stage1, "table": table, "fit": fit}[sys.argv[1]]()

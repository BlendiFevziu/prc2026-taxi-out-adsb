"""Pilot 13 (PREREG_neighbour_months.md): November/December corrector trained on its own pair, plus January (neighbour), plus January/February/June/July (all).
  python3 -u -B research_730_adsb_learner/pilot13_screen.py table   November ADS-B and stand columns (refuses until all 30 November track days exist)
  python3 -u -B research_730_adsb_learner/pilot13_screen.py fit     the three arms and both verdicts (needs .cache/ensemble_holdout_11_12)
"""
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_730_adsb_learner import pilot9_screen as N  # imports torch before the tree libraries
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_730_adsb_learner import features_adsb as F
from research_730_adsb_learner import pilot_screen as P
from research_730_adsb_learner import pilot5_screen as S
from research_730_adsb_learner import pilot12_screen as T
from research_300.heavy_lock import heavy_work

CACHE, ROOT, TRACKS = P.CACHE, P.ROOT, P.TRACKS
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot13_screen.json"
TABLE_11 = CACHE / "table_11_stand.parquet"
ENSEMBLE_1112 = ROOT / ".cache/ensemble_holdout_11_12/candidate/holdout_predictions.parquet"
FEATURES = S.STAGE2_BASE + S.AD18
COLS = list(dict.fromkeys([c for c in FEATURES if c != "p1"] + ["y", "MVT_ID_mvt", "MVT_TIME_UTC_mvt"]))


def table():
    if TABLE_11.exists():
        P.log("table already cached"); return
    files = sorted(glob.glob(str(TRACKS / "2025-11-??.parquet")))
    if len(files) != 30:
        raise SystemExit(f"only {len(files)} of 30 November 2025 track days; wait for the download")
    frame = pd.read_parquet(N.Q.ALL, columns=list(dict.fromkeys(P.BASE + P.EXTRA)))
    frame = frame[frame.hat_aobt.eq(1) & frame.y.notna() & (N.Q.month_of(frame) == 11)].reset_index(drop=True)
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "CALLSIGN_flt"]) for p in sorted(glob.glob(str(ROOT / "data/training_2025-11-01_*.parquet")))],
                    ignore_index=True).drop_duplicates("MVT_ID_mvt")
    frame = frame.merge(raw, on="MVT_ID_mvt", how="left")
    tracks = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = F.compute(frame, tracks, stands=True)
    assert np.array_equal(out.index.to_numpy(), frame.MVT_ID_mvt.to_numpy())
    for c in S.AD18:
        frame[c] = out[c].to_numpy()
    P.log(f"2025-11: {len(frame):,} rows, {len(files)} track files, matched {frame.ad_matched.mean():.3f}, stand ok {frame.sd_status.eq(1).mean():.3f}")
    frame.to_parquet(TABLE_11, index=False)


def with_ensemble(table, path):
    e = pd.read_parquet(path).set_index("MVT_ID_mvt", verify_integrity=True)
    assert table.MVT_ID_mvt.isin(e.index).all() and e.loc[table.MVT_ID_mvt, "hat_aobt"].eq(1).all()
    assert np.allclose(e.loc[table.MVT_ID_mvt, "y"].to_numpy(float), table.y.to_numpy(float))
    table["p1"] = e.loc[table.MVT_ID_mvt, "prediction"].to_numpy(float)
    return table


def corrector(table, fold, extra, seed, label):
    cache = CACHE / f"neighbour_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    pred = np.zeros(len(table))
    for k in range(P.FOLDS):
        learn = pd.concat([table[fold != k]] + ([extra] if extra is not None else []), ignore_index=True)
        held = table[fold == k].copy()
        cats = pd.unique(learn.ADEP_mvt.astype(str))
        learn["ADEP_mvt"] = pd.Categorical(learn.ADEP_mvt.astype(str), categories=cats)
        held["ADEP_mvt"] = pd.Categorical(held.ADEP_mvt.astype(str), categories=cats)
        with heavy_work(f"research_730 pilot 13 {label} seed {seed} fold {k}", log=P.log):
            model = lgb.LGBMRegressor(objective="l2", n_estimators=S.ROUNDS, learning_rate=.03, num_leaves=63, min_child_samples=100,
                                      subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)
            model.fit(learn[FEATURES], (learn.y - learn.p1).to_numpy())
            pred[fold == k] = np.maximum(held.p1.to_numpy(float) + model.predict(held[FEATURES]), 300)
    np.save(cache, pred)
    P.log(f"{label} seed {seed}: RMSE {np.sqrt(np.mean((pred - table.y.to_numpy()) ** 2)):.3f}")
    return pred


def fit():
    t11 = pd.read_parquet(TABLE_11, columns=COLS)
    t12 = pd.read_parquet(N.TABLE, columns=COLS)
    t12 = t12[N.Q.month_of(t12) == 12].reset_index(drop=True)
    pair = with_ensemble(pd.concat([t11, t12], ignore_index=True), ENSEMBLE_1112)
    t17 = pd.read_parquet(CACHE / "table_1_7_stand.parquet", columns=COLS)
    e17 = pd.read_parquet(T.ENSEMBLE_17, columns=["MVT_ID_mvt", "y", "baseline"])
    assert np.array_equal(e17.MVT_ID_mvt.to_numpy(), t17.MVT_ID_mvt.to_numpy()) and np.allclose(e17.y.to_numpy(), t17.y.to_numpy())
    t17["p1"] = e17.baseline.to_numpy(float)
    t26 = with_ensemble(pd.read_parquet(CACHE / "table_2_6_stand.parquet", columns=COLS), T.ENSEMBLE_26)
    january = t17[N.Q.month_of(t17) == 1]
    every = pd.concat([t17, t26], ignore_index=True)
    fold, day = T.folds(pair)
    y, month = pair.y.to_numpy(float), N.Q.month_of(pair)
    base = (pair.p1.to_numpy() - y) ** 2
    loss = {}
    for seed in P.SEEDS:
        loss[("own", seed)] = (corrector(pair, fold, None, seed, "own") - y) ** 2
        loss[("neighbour", seed)] = (corrector(pair, fold, january, seed, "plus_jan") - y) ** 2
        loss[("all", seed)] = (corrector(pair, fold, every, seed, "plus_1267") - y) ** 2
    mean = lambda name: np.mean([loss[(name, s)] for s in P.SEEDS], axis=0)
    per = lambda name: {s: loss[(name, s)] for s in P.SEEDS}
    airport = pair.ADEP_mvt.astype(str).to_numpy()
    reports = {}
    for name in ("neighbour", "all"):
        r = N.judge(y, month, day, mean(name), mean("own"), base, per(name), per("own"), (11, 12))
        r["airports_vs_own"] = {a: float((mean("own") - mean(name))[airport == a].mean()) for a in sorted(set(airport))}
        reports[name] = r
    report = {"prereg": "research_730_adsb_learner/PREREG_neighbour_months.md", "rows": int(len(y)),
              "rows_by_month": {str(m): int((month == m).sum()) for m in (11, 12)}, "ensemble_rmse": float(np.sqrt(base.mean())),
              "B_neighbour_vs_own_judged": reports["neighbour"], "A_all_vs_own_reported": reports["all"]}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"table": table, "fit": fit}[sys.argv[1]]()

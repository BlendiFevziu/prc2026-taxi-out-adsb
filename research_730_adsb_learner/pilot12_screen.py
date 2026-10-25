"""Pilot 12 (PREREG_four_month_ens.md): the corrector over the ensemble's own held-out predictions, four months pooled against each pair alone.
Run from the repository root after research_740_ensemble_holdout (February/June candidate stage):  python3 -u -B research_730_adsb_learner/pilot12_screen.py
"""
import json
from pathlib import Path
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_730_adsb_learner import pilot9_screen as N  # imports torch before the tree libraries
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_730_adsb_learner import pilot_screen as P
from research_730_adsb_learner import pilot5_screen as S
from research_300.heavy_lock import heavy_work

CACHE, ROOT = P.CACHE, P.ROOT
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot12_screen.json"
ENSEMBLE_17 = ROOT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
ENSEMBLE_26 = ROOT / ".cache/ensemble_holdout_2_6/candidate/holdout_predictions.parquet"
FEATURES = S.STAGE2_BASE + S.AD18


def folds(table):
    date = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    return date.map(lambda d: zlib.crc32(d.encode()) % P.FOLDS).to_numpy(), (table.ADEP_mvt.astype(str) + "|" + date).to_numpy()


def corrector(table, fold, seed, label):
    cache = CACHE / f"four_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    pred = np.zeros(len(table))
    for k in range(P.FOLDS):
        learn, held = table[fold != k].copy(), table[fold == k].copy()
        cats = pd.unique(learn.ADEP_mvt.astype(str))
        learn["ADEP_mvt"] = pd.Categorical(learn.ADEP_mvt.astype(str), categories=cats)
        held["ADEP_mvt"] = pd.Categorical(held.ADEP_mvt.astype(str), categories=cats)
        with heavy_work(f"research_730 pilot 12 {label} seed {seed} fold {k}", log=P.log):
            model = lgb.LGBMRegressor(objective="l2", n_estimators=S.ROUNDS, learning_rate=.03, num_leaves=63, min_child_samples=100,
                                      subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)
            model.fit(learn[FEATURES], (learn.y - learn.p1).to_numpy())
            pred[fold == k] = np.maximum(held.p1.to_numpy(float) + model.predict(held[FEATURES]), 300)
    np.save(cache, pred)
    P.log(f"{label} seed {seed}: RMSE {np.sqrt(np.mean((pred - table.y.to_numpy()) ** 2)):.3f}")
    return pred


def main():
    cols = list(dict.fromkeys([c for c in FEATURES if c != "p1"] + ["y", "MVT_ID_mvt", "MVT_TIME_UTC_mvt"]))
    t17 = pd.read_parquet(CACHE / "table_1_7_stand.parquet", columns=cols)
    e17 = pd.read_parquet(ENSEMBLE_17, columns=["MVT_ID_mvt", "y", "baseline"])
    assert np.array_equal(e17.MVT_ID_mvt.to_numpy(), t17.MVT_ID_mvt.to_numpy()) and np.allclose(e17.y.to_numpy(), t17.y.to_numpy())
    t17["p1"] = e17.baseline.to_numpy(float)
    t26 = pd.read_parquet(CACHE / "table_2_6_stand.parquet", columns=cols)
    e26 = pd.read_parquet(ENSEMBLE_26).set_index("MVT_ID_mvt", verify_integrity=True)
    assert t26.MVT_ID_mvt.isin(e26.index).all() and e26.loc[t26.MVT_ID_mvt, "hat_aobt"].eq(1).all()
    assert np.allclose(e26.loc[t26.MVT_ID_mvt, "y"].to_numpy(float), t26.y.to_numpy(float))
    t26["p1"] = e26.loc[t26.MVT_ID_mvt, "prediction"].to_numpy(float)
    f17, d17 = folds(t17)
    f26, d26 = folds(t26)
    four = pd.concat([t17, t26], ignore_index=True)
    f4 = np.concatenate([f17, f26])
    y = four.y.to_numpy(float)
    month = N.Q.month_of(four)
    day = np.concatenate([d17, d26])
    base = (four.p1.to_numpy() - y) ** 2
    loss = {}
    for seed in P.SEEDS:
        pair = np.concatenate([np.load(CACHE / f"stack_ens_stack_s{seed}.npy"), corrector(t26, f26, seed, "pair_26")])
        loss[("pair", seed)] = (pair - y) ** 2
        loss[("four", seed)] = (corrector(four, f4, seed, "pooled") - y) ** 2
    mean = lambda name: np.mean([loss[(name, s)] for s in P.SEEDS], axis=0)
    report = N.judge(y, month, day, mean("four"), mean("pair"), base,
                     {s: loss[("four", s)] for s in P.SEEDS}, {s: loss[("pair", s)] for s in P.SEEDS}, (1, 2, 6, 7))
    airport = four.ADEP_mvt.astype(str).to_numpy()
    report["airports_vs_pair"] = {a: float((mean("pair") - mean("four"))[airport == a].mean()) for a in sorted(set(airport))}
    report = {"prereg": "research_730_adsb_learner/PREREG_four_month_ens.md",
              "ensemble_rmse": {"1_7": float(np.sqrt(np.mean((t17.p1 - t17.y) ** 2))), "2_6": float(np.sqrt(np.mean((t26.p1 - t26.y) ** 2)))}, **report}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

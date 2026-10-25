"""Pilot 11 (PREREG_more_months_ens.md): pilot 6's corrector (ensemble p1, four day folds on January/July) with five extra training months.
Run from the repository root after pilot 9:  python3 -u -B research_730_adsb_learner/pilot11_screen.py
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
from research_300 import incremental as I
from research_300.heavy_lock import heavy_work

CACHE, ROOT = P.CACHE, P.ROOT
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot11_screen.json"
ENSEMBLE = ROOT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
FEATURES = S.STAGE2_BASE + S.AD18


def plus5(table, fold, extra, seed):
    cache = CACHE / f"stack_ens_plus5_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    pred = np.zeros(len(table))
    for k in range(P.FOLDS):
        learn = pd.concat([table[fold != k], extra], ignore_index=True)
        held = table[fold == k].copy()
        cats = pd.unique(learn.ADEP_mvt.astype(str))
        learn["ADEP_mvt"] = pd.Categorical(learn.ADEP_mvt.astype(str), categories=cats)
        held["ADEP_mvt"] = pd.Categorical(held.ADEP_mvt.astype(str), categories=cats)
        with heavy_work(f"research_730 pilot 11 plus5 seed {seed} fold {k}", log=P.log):
            model = lgb.LGBMRegressor(objective="l2", n_estimators=S.ROUNDS, learning_rate=.03, num_leaves=63, min_child_samples=100,
                                      subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)
            model.fit(learn[FEATURES], (learn.y - learn.p1).to_numpy())
            pred[fold == k] = np.maximum(held.p1.to_numpy(float) + model.predict(held[FEATURES]), 300)
    np.save(cache, pred)
    P.log(f"plus5 seed {seed}: RMSE {np.sqrt(np.mean((pred - table.y) ** 2)):.3f}")
    return pred


def main():
    table = pd.read_parquet(CACHE / "table_1_7_stand.parquet")
    e = pd.read_parquet(ENSEMBLE, columns=["MVT_ID_mvt", "y", "baseline"])
    assert np.array_equal(e.MVT_ID_mvt.to_numpy(), table.MVT_ID_mvt.to_numpy()) and np.allclose(e.y.to_numpy(), table.y.to_numpy())
    table["p1"] = e.baseline.to_numpy(float)
    date = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    fold = date.map(lambda d: zlib.crc32(d.encode()) % P.FOLDS).to_numpy()          # pilot 6's folds
    day = (table.ADEP_mvt.astype(str) + "|" + date).to_numpy()
    month = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    y = table.y.to_numpy(float)
    cols = list(dict.fromkeys([c for c in FEATURES if c != "p1"] + ["y", "MVT_ID_mvt"]))
    extra = pd.concat([pd.read_parquet(CACHE / "table_2_6_stand.parquet", columns=cols), pd.read_parquet(N.TABLE, columns=cols)], ignore_index=True)
    s1 = pd.concat([pd.read_parquet(N.Q.STAGE1).drop(columns="pair"), pd.read_parquet(N.STAGE1)]).set_index("MVT_ID_mvt")
    assert extra.MVT_ID_mvt.isin(s1.index).all() and not extra.MVT_ID_mvt.isin(table.MVT_ID_mvt).any()
    table = table[cols + ["p1"]]
    loss = {}
    for seed in P.SEEDS:
        extra["p1"] = s1.loc[extra.MVT_ID_mvt, f"p1_s{seed}"].to_numpy()
        loss[("pilot6", seed)] = (np.load(CACHE / f"stack_ens_stack_s{seed}.npy") - y) ** 2
        loss[("plus5", seed)] = (plus5(table, fold, extra, seed) - y) ** 2
    base = (table.p1.to_numpy() - y) ** 2
    mean = lambda name: np.mean([loss[(name, s)] for s in P.SEEDS], axis=0)
    report = N.judge(y, month, day, mean("plus5"), mean("pilot6"), base,
                     {s: loss[("plus5", s)] for s in P.SEEDS}, {s: loss[("pilot6", s)] for s in P.SEEDS}, (1, 7))
    airport = table.ADEP_mvt.astype(str).to_numpy()
    report["airports_vs_pilot6"] = {a: float((mean("pilot6") - mean("plus5"))[airport == a].mean()) for a in sorted(set(airport))}
    report = {"prereg": "research_730_adsb_learner/PREREG_more_months_ens.md", "extra_rows": int(len(extra)), **report}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

"""Pilot 14 (PREREG_matched_months.md): February/June corrected by a February/June corrector (week-block folds) against the January/July one.
  python3 -u -B research_730_adsb_learner/pilot14_screen.py test    the arms and the verdict
  python3 -u -B research_730_adsb_learner/pilot14_screen.py build   only after a pass: the second 2026 file
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_730_adsb_learner import pilot9_screen as N  # imports torch before the tree libraries
import lightgbm as lgb
import numpy as np
import pandas as pd
import submission as SUB
from research_730_adsb_learner import pilot_screen as P
from research_730_adsb_learner import pilot5_screen as S
from research_730_adsb_learner import pilot12_screen as T
from research_730_adsb_learner import pilot13_screen as U
from research_300.heavy_lock import heavy_work

CACHE, ROOT = P.CACHE, P.ROOT
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot14_screen.json"
REBUILD = CACHE / "rebuild_2026"
FEATURES = S.STAGE2_BASE + S.AD18
COLS = U.COLS


def model(seed):
    return lgb.LGBMRegressor(objective="l2", n_estimators=S.ROUNDS, learning_rate=.03, num_leaves=63, min_child_samples=100,
                             subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)


def categorical(learn, apply):
    learn, apply = learn.copy(), apply.copy()
    cats = pd.unique(learn.ADEP_mvt.astype(str))
    learn["ADEP_mvt"] = pd.Categorical(learn.ADEP_mvt.astype(str), categories=cats)
    apply["ADEP_mvt"] = pd.Categorical(apply.ADEP_mvt.astype(str), categories=cats)
    return learn, apply


def cross(train, apply, seed, label):
    cache = CACHE / f"matched_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    learn, held = categorical(train, apply)
    with heavy_work(f"research_730 pilot 14 {label} seed {seed}", log=P.log):
        m = model(seed)
        m.fit(learn[FEATURES], (learn.y - learn.p1).to_numpy())
        pred = np.maximum(held.p1.to_numpy(float) + m.predict(held[FEATURES]), 300)
    np.save(cache, pred)
    P.log(f"{label} seed {seed}: RMSE {np.sqrt(np.mean((pred - apply.y.to_numpy()) ** 2)):.3f}")
    return pred


def week_blocks(table):
    day = pd.to_datetime(table.MVT_TIME_UTC_mvt, utc=True).dt.day.to_numpy()
    return np.minimum((day - 1) // 7, 3)


def matched(table, seed, label):
    cache = CACHE / f"matched_{label}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    fold, pred = week_blocks(table), np.zeros(len(table))
    for k in range(4):
        learn, held = categorical(table[fold != k], table[fold == k])
        with heavy_work(f"research_730 pilot 14 {label} seed {seed} block {k}", log=P.log):
            m = model(seed)
            m.fit(learn[FEATURES], (learn.y - learn.p1).to_numpy())
            pred[fold == k] = np.maximum(held.p1.to_numpy(float) + m.predict(held[FEATURES]), 300)
    np.save(cache, pred)
    P.log(f"{label} seed {seed}: RMSE {np.sqrt(np.mean((pred - table.y.to_numpy()) ** 2)):.3f}")
    return pred


def tables():
    t17 = pd.read_parquet(CACHE / "table_1_7_stand.parquet", columns=COLS)
    e17 = pd.read_parquet(T.ENSEMBLE_17, columns=["MVT_ID_mvt", "y", "baseline"])
    assert np.array_equal(e17.MVT_ID_mvt.to_numpy(), t17.MVT_ID_mvt.to_numpy()) and np.allclose(e17.y.to_numpy(), t17.y.to_numpy())
    t17["p1"] = e17.baseline.to_numpy(float)
    t26 = U.with_ensemble(pd.read_parquet(CACHE / "table_2_6_stand.parquet", columns=COLS), T.ENSEMBLE_26)
    return t17, t26


def test():
    t17, t26 = tables()
    report = {"prereg": "research_730_adsb_learner/PREREG_matched_months.md"}
    for name, target, other, months in (("feb_jun_judged", t26, t17, (2, 6)), ("jan_jul_mirror_reported", t17, t26, (1, 7))):
        y, month = target.y.to_numpy(float), N.Q.month_of(target)
        _, day = T.folds(target)
        base = (target.p1.to_numpy() - y) ** 2
        loss = {}
        for seed in P.SEEDS:
            loss[("cross", seed)] = (cross(other, target, seed, f"{name}_cross") - y) ** 2
            loss[("matched", seed)] = (matched(target, seed, f"{name}_weeks") - y) ** 2
        mean = lambda a: np.mean([loss[(a, s)] for s in P.SEEDS], axis=0)
        r = N.judge(y, month, day, mean("matched"), mean("cross"), base,
                    {s: loss[("matched", s)] for s in P.SEEDS}, {s: loss[("cross", s)] for s in P.SEEDS}, months)
        airport = target.ADEP_mvt.astype(str).to_numpy()
        r["airports_vs_cross"] = {a: float((mean("cross") - mean("matched"))[airport == a].mean()) for a in sorted(set(airport))}
        if name == "feb_jun_judged":
            crc = np.mean([(np.load(CACHE / f"four_pair_26_s{s}.npy") - y) ** 2 for s in P.SEEDS], axis=0)
            r["matched_crc32_day_folds_rmse_reported"] = float(np.sqrt(crc.mean()))
        report[name] = r
    report["verdict"] = report["feb_jun_judged"]["verdict"]
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


def build():
    assert json.loads(OUTPUT.read_text())["verdict"] == "pass", "pilot 14 did not pass; the current file stays"
    _, t26 = tables()
    a26 = pd.read_parquet(REBUILD / "table_2026_final.parquet")
    apply = a26[a26.month.isin(["2026-02", "2026-06"])].reset_index(drop=True)
    corrections = []
    for seed in P.SEEDS:
        learn, held = categorical(t26, apply)
        with heavy_work(f"research_730 pilot 14 build seed {seed}", log=P.log):
            m = model(seed)
            m.fit(learn[FEATURES], (learn.y - learn.p1).to_numpy())
            corrections.append(m.predict(held[FEATURES]))
            m.booster_.save_model(str(REBUILD / f"corrector_feb_jun_s{seed}.txt"))
    corr = np.mean(corrections, axis=0)
    new_rows = pd.Series(np.maximum(apply.p1.to_numpy() + corr, 300.0), index=apply.MVT_ID_mvt.to_numpy())
    current = pd.read_parquet(REBUILD / "final_rebuild_2026.parquet")
    template = pd.read_parquet(ROOT / "data/final_submitting.parquet")
    t = "TAXITIME_SEC_mvt"
    cand = current.copy()
    hit = cand.MVT_ID_mvt.isin(new_rows.index).to_numpy()
    cand.loc[hit, t] = cand.loc[hit, "MVT_ID_mvt"].map(new_rows).to_numpy()
    cand[t] = cand[t].astype(template[t].dtype)
    errors = SUB.pruefe(cand, template, None)
    checks = {"pruefe_errors": [str(x) for x in errors], "rows": int(len(cand)), "rows_equal_template": bool(cand.MVT_ID_mvt.equals(template.MVT_ID_mvt)),
              "all_finite": bool(np.isfinite(cand[t]).all()), "rows_changed_scope": int(hit.sum()), "feb_jun_main_rows": int(len(apply)),
              "other_rows_identical": bool(np.array_equal(cand[t].to_numpy()[~hit], current[t].to_numpy()[~hit]))}
    path = REBUILD / "final_rebuild_2026_matched.parquet"
    cand.to_parquet(path, index=False)
    report = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "checks": checks,
              "change_vs_current_s": {"mean": float((cand[t] - current[t])[hit].mean()), "mean_abs": float((cand[t] - current[t])[hit].abs().mean())},
              "correction_s": {"mean": float(corr.mean()), "mean_abs": float(np.abs(corr).mean())},
              "validated_by": "PREREG_matched_months.md (pilot 14)", "uploaded": False}
    (REBUILD / "build_report_matched.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"test": test, "build": build}[sys.argv[1]]()

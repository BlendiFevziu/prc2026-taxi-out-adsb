"""Pilot 15 (PREREG_corrector_capacity.md): corrector rounds x leaves, selected on January/July, judged on February/June and November/December.
  python3 -u -B research_730_adsb_learner/pilot15_screen.py test    selection, judgement, verdict
  python3 -u -B research_730_adsb_learner/pilot15_screen.py build   only after a pass: the tuned 2026 file
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
from research_730_adsb_learner import pilot14_screen as V
from research_300.heavy_lock import heavy_work

CACHE, ROOT = P.CACHE, P.ROOT
OUTPUT = ROOT / "research_730_adsb_learner/results/pilot15_screen.json"
REBUILD = CACHE / "rebuild_2026"
FEATURES = S.STAGE2_BASE + S.AD18
GRID = [(r, l) for r in (600, 1200, 2400) for l in (63, 255)]          # sorted small to large: the tie rule
DEFAULT = (600, 63)
PREREG = "research_730_adsb_learner/PREREG_corrector_capacity.md"


def model(seed, rounds, leaves):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.03, num_leaves=leaves, min_child_samples=100,
                             subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)


def folded(table, fold, seed, rounds, leaves, label):
    cache = CACHE / f"capacity_{label}_{rounds}x{leaves}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    pred = np.zeros(len(table))
    for k in range(P.FOLDS):
        learn, held = V.categorical(table[fold != k], table[fold == k])
        with heavy_work(f"research_730 pilot 15 {label} {rounds}x{leaves} seed {seed} fold {k}", log=P.log):
            m = model(seed, rounds, leaves)
            m.fit(learn[FEATURES], (learn.y - learn.p1).to_numpy())
            pred[fold == k] = np.maximum(held.p1.to_numpy(float) + m.predict(held[FEATURES]), 300)
    np.save(cache, pred)
    P.log(f"{label} {rounds}x{leaves} seed {seed}: RMSE {np.sqrt(np.mean((pred - table.y.to_numpy()) ** 2)):.3f}")
    return pred


def tables():
    t17, t26 = V.tables()
    t11 = pd.read_parquet(U.TABLE_11, columns=U.COLS)
    t12 = pd.read_parquet(N.TABLE, columns=U.COLS)
    t12 = t12[N.Q.month_of(t12) == 12].reset_index(drop=True)
    t1112 = U.with_ensemble(pd.concat([t11, t12], ignore_index=True), U.ENSEMBLE_1112)
    return t17, t26, t1112


def test():
    t17, t26, t1112 = tables()
    f17, _ = T.folds(t17)
    y17 = t17.y.to_numpy(float)
    selection = {}
    for rounds, leaves in GRID:
        if (rounds, leaves) == DEFAULT:
            preds = [np.load(CACHE / f"stack_ens_stack_s{s}.npy") for s in P.SEEDS]
        else:
            preds = [folded(t17, f17, s, rounds, leaves, "sel17") for s in P.SEEDS]
        selection[f"{rounds}x{leaves}"] = float(np.sqrt(np.mean([(p - y17) ** 2 for p in preds], axis=0).mean()))
    best = min(selection.values())
    chosen = next((r, l) for r, l in GRID if selection[f"{r}x{l}"] <= best + 0.05)
    report = {"prereg": PREREG, "selection_jan_jul_rmse": selection, "chosen": f"{chosen[0]}x{chosen[1]}"}
    if chosen == DEFAULT:
        report["verdict"] = "default selected, nothing judged"
    else:
        held = pd.concat([t26, t1112], ignore_index=True)
        y, month = held.y.to_numpy(float), N.Q.month_of(held)
        f26, d26 = T.folds(t26)
        f1112, d1112 = T.folds(t1112)
        day = np.concatenate([d26, d1112])
        base = (held.p1.to_numpy() - y) ** 2
        loss = {}
        for s in P.SEEDS:
            default = np.concatenate([np.load(CACHE / f"four_pair_26_s{s}.npy"), np.load(CACHE / f"neighbour_own_s{s}.npy")])
            new = np.concatenate([folded(t26, f26, s, *chosen, "judge26"), folded(t1112, f1112, s, *chosen, "judge1112")])
            loss[("default", s)], loss[("new", s)] = (default - y) ** 2, (new - y) ** 2
        mean = lambda a: np.mean([loss[(a, s)] for s in P.SEEDS], axis=0)
        r = N.judge(y, month, day, mean("new"), mean("default"), base,
                    {s: loss[("new", s)] for s in P.SEEDS}, {s: loss[("default", s)] for s in P.SEEDS}, (2, 6, 11, 12))
        airport = held.ADEP_mvt.astype(str).to_numpy()
        r["airports_vs_default"] = {a: float((mean("default") - mean("new"))[airport == a].mean()) for a in sorted(set(airport))}
        report["judgement"] = r
        report["verdict"] = r["verdict"]
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


def build():
    report = json.loads(OUTPUT.read_text())
    assert report["verdict"] == "pass", "pilot 15 did not pass; the current file stays"
    rounds, leaves = map(int, report["chosen"].split("x"))
    t17, _, _ = tables()
    a26 = pd.read_parquet(REBUILD / "table_2026_final.parquet")
    corrections = []
    for seed in P.SEEDS:
        learn, held = V.categorical(t17, a26)
        with heavy_work(f"research_730 pilot 15 build seed {seed}", log=P.log):
            m = model(seed, rounds, leaves)
            m.fit(learn[FEATURES], (learn.y - learn.p1).to_numpy())
            corrections.append(m.predict(held[FEATURES]))
            m.booster_.save_model(str(REBUILD / f"corrector_tuned_{rounds}x{leaves}_s{seed}.txt"))
    corr = np.mean(corrections, axis=0)
    new_main = pd.Series(np.maximum(a26.p1.to_numpy() + corr, 300.0), index=a26.MVT_ID_mvt.to_numpy())
    current = pd.read_parquet(REBUILD / "final_rebuild_2026.parquet")
    template = pd.read_parquet(ROOT / "data/final_submitting.parquet")
    t = "TAXITIME_SEC_mvt"
    cand = current.copy()
    hit = cand.MVT_ID_mvt.isin(new_main.index).to_numpy()
    cand.loc[hit, t] = cand.loc[hit, "MVT_ID_mvt"].map(new_main).to_numpy()
    cand[t] = cand[t].astype(template[t].dtype)
    errors = SUB.pruefe(cand, template, None)
    checks = {"pruefe_errors": [str(x) for x in errors], "rows": int(len(cand)), "rows_equal_template": bool(cand.MVT_ID_mvt.equals(template.MVT_ID_mvt)),
              "all_finite": bool(np.isfinite(cand[t]).all()), "main_rows_corrected": int(hit.sum()),
              "fallback_rows_identical": bool(np.array_equal(cand[t].to_numpy()[~hit], current[t].to_numpy()[~hit]))}
    path = REBUILD / "final_rebuild_2026_tuned.parquet"
    cand.to_parquet(path, index=False)
    out = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "configuration": report["chosen"], "checks": checks,
           "change_vs_current_s": {"mean": float((cand[t] - current[t])[hit].mean()), "mean_abs": float((cand[t] - current[t])[hit].abs().mean())},
           "validated_by": "PREREG_corrector_capacity.md (pilot 15)", "uploaded": False}
    (REBUILD / "build_report_tuned.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"test": test, "build": build}[sys.argv[1]]()

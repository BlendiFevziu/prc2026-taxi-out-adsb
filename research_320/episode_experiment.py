"""Pre-registered episode experiment (research_320/PREREG_episodes.md): fit once, then screen, then confirm. No upload.

Stages: "screen" (v56's replication dates) and "confirm" (fresh dates). The eighteen models (episodes and five
same-width controls, three seeds) are fitted once on v56's training rows before the screen and reused unchanged.
Before scoring the screen, the extraction plus the rebuilt v56 models must reproduce Codex's recorded replication
(research_310/results/multiairport_confirm.json at 9884150) exactly, so the screen rows are v56's own.
Run from the repository root:  python3 -u -B research_320/episode_experiment.py {screen,confirm}
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import extract_episodes as X  # noqa: E402
from research_320.episodes import EPISODE_FEATURES  # noqa: E402

MS = X.v56_modules()
R = MS.R
SEEDS, CONTROLS = (177, 277, 377), (3201, 3202, 3203, 3204, 3205)
TRAIN = tuple(MS.TRAIN)
STAGES = {"screen": ("2025-01-14", "2025-01-21", "2025-01-26", "2025-07-13", "2025-07-14", "2025-07-25"),
          "confirm": ("2025-01-02", "2025-01-18", "2025-01-06", "2025-01-01",
                      "2025-07-12", "2025-07-31", "2025-07-26", "2025-07-07")}
DEPLOYED = {("EDDF", "01"), ("EGLL", "01"), ("EHAM", "07"), ("LEBL", "01"), ("LEBL", "07"), ("LSZH", "01"), ("LSZH", "07")}
V56COLS = list(MS.COLS)
EPCOLS = V56COLS + list(EPISODE_FEATURES)
REBUILD = HERE / ".cache/research_320/v56_rebuild"
MODELS = HERE / ".cache/research_320/models"
REPLICATION = {"rows": 15909, "changed": 9826, "base_rmse": 185.66812426781576, "rmse": 177.09363856021687}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lgbm(seed):
    return R.lgb.LGBMRegressor(objective="regression", n_estimators=200, learning_rate=.03, num_leaves=7,
                               min_child_samples=80, reg_lambda=100., subsample=.8, subsample_freq=1,
                               colsample_bytree=.9, max_bin=63, n_jobs=4, random_state=seed, verbosity=-1)


def inputs(frame, arm):
    x = frame[EPCOLS].copy()
    if arm != "episodes":
        rng = np.random.default_rng(int(arm.split("_")[1]))
        cols = x.columns.get_indexer(list(EPISODE_FEATURES))
        for pos in frame.groupby(["airport", "date"], sort=True).indices.values():
            x.iloc[pos, cols] = x.iloc[rng.permutation(pos), cols].to_numpy()
    return x


def arms():
    return ["episodes"] + [f"control_{c}" for c in CONTROLS]


def fit_models():
    manifest = MODELS / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        if any(sha(MODELS / name) != h for name, h in m["model_hashes"].items()):
            raise SystemExit("a saved episode model changed since it was fitted")
        return m
    train = X.extract(TRAIN, "train")
    ref = pd.read_parquet(REBUILD / "all_features.parquet")
    both = train[["MVT_ID_mvt"] + V56COLS].merge(ref[["MVT_ID_mvt"] + V56COLS], on="MVT_ID_mvt", suffixes=("", "_r"))
    parity = len(both) == len(ref[ref.date.isin(TRAIN)]) and all(
        np.array_equal(both[c].to_numpy(float), both[c + "_r"].to_numpy(float), equal_nan=True) for c in V56COLS)
    if not parity:
        raise SystemExit("training rows differ from v56's")
    train = train[train.category.eq("partial_ground")].reset_index(drop=True)
    target = (train.y - train.comparator).to_numpy(float)
    MODELS.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for arm in arms():
        x = inputs(train, arm)
        for seed in SEEDS:
            path = MODELS / f"{arm}_{seed}.pkl"
            R.joblib.dump(lgbm(seed).fit(x, target), path)
            hashes[path.name] = sha(path)
    m = {"train_rows": len(train), "v56_parity": parity, "model_hashes": hashes}
    manifest.write_text(json.dumps(m, indent=2) + "\n")
    return m


def corrections(frame, arm):
    if arm == "v56":
        rebuilt = json.loads((REBUILD / "multiairport_screen_rebuilt.json").read_text())["model_hashes"]
        paths = [REBUILD / f"real_{s}.pkl" for s in SEEDS]
        if any(sha(p) != rebuilt[p.name] for p in paths):
            raise SystemExit("rebuilt v56 model changed")
        return [R.joblib.load(p).predict(frame[V56COLS]) for p in paths]
    x = inputs(frame, arm)
    return [R.joblib.load(MODELS / f"{arm}_{s}.pkl").predict(x) for s in SEEDS]


def corrected(frame, corr):
    return np.maximum(frame.comparator.to_numpy(float) + .5 * corr, 300.)


def detector_ids(dates):
    from research_300 import tow_guard_test as T
    from research_300 import geometry_compare as GC
    maps, ids = (R.SD.stand_table("EHAM"), GC.xplane_table("EHAM")[0]), set()
    mapping = R.AT.OB.callsign_mapping()
    for date in [d for d in dates if d[5:7] == "07"]:
        est = T.estimates(date, "EHAM", mapping, maps)
        ids |= set(est.loc[est.guarded.notna(), "MVT_ID_mvt"])
    return ids


def describe(frame, base, pred):
    y = frame.y.to_numpy(float)
    gain = (base - y) ** 2 - (pred - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    cells = {f"{a}-{m}": float(gain[(frame.airport.eq(a) & frame.date.str[5:7].eq(m)).to_numpy()].mean())
             if (frame.airport.eq(a) & frame.date.str[5:7].eq(m)).any() else None for a, m in sorted(DEPLOYED)}
    return {"rows": int(len(frame)), "v56_rmse": float(np.sqrt(np.mean((base - y) ** 2))),
            "arm_rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "gain_mse": float(gain.mean()),
            "gain_without_best100": float(gain[keep].mean()),
            "airport_day_ci": list(map(float, R.block_bootstrap(gain, (frame.airport + "/" + frame.date).to_numpy()))),
            "by_month": {m: float(gain[frame.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")},
            "by_cell": cells, "by_date": {d: float(gain[frame.date.eq(d).to_numpy()].mean()) for d in sorted(frame.date.unique())}}


def run(stage):
    models = fit_models()
    dates = STAGES[stage]
    frame = X.extract(dates, stage)
    report = {"prereg": "research_320/PREREG_episodes.md", "stage": stage, "dates": list(dates), "models": models}
    if stage == "screen":
        allrows = frame.copy()
        hit = allrows.category.eq("partial_ground").to_numpy()
        pred = allrows.comparator.to_numpy(float).copy()
        pred[hit] = corrected(allrows[hit], np.mean(corrections(allrows[hit], "v56"), axis=0))
        base, y = allrows.comparator.to_numpy(float), allrows.y.to_numpy(float)
        got = {"rows": len(allrows), "changed": int(np.count_nonzero(pred != base)),
               "base_rmse": float(np.sqrt(np.mean((base - y) ** 2))), "rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}
        report["replication_parity"] = got
        if not (got["rows"] == REPLICATION["rows"] and got["changed"] == REPLICATION["changed"]
                and abs(got["base_rmse"] - REPLICATION["base_rmse"]) < 1e-9 and abs(got["rmse"] - REPLICATION["rmse"]) < 1e-9):
            raise SystemExit(f"the screen rows do not reproduce Codex's replication: {got}")
    in_cell = np.array([(a, d[5:7]) in DEPLOYED for a, d in zip(frame.airport, frame.date)])
    detector = detector_ids(dates)
    cohort = frame[in_cell & frame.category.eq("partial_ground").to_numpy() & ~frame.MVT_ID_mvt.isin(detector).to_numpy()]
    cohort = cohort.reset_index(drop=True)
    report["cohort_rows"], report["detector_rows_excluded"] = len(cohort), int(frame.MVT_ID_mvt.isin(detector).sum())
    v56 = corrections(cohort, "v56")
    base = corrected(cohort, np.mean(v56, axis=0))
    results, seeds = {}, {}
    for arm in arms():
        corr = corrections(cohort, arm)
        results[arm] = describe(cohort, base, corrected(cohort, np.mean(corr, axis=0)))
        if arm == "episodes":
            seeds = {s: describe(cohort, corrected(cohort, v), corrected(cohort, c))["gain_mse"]
                     for s, v, c in zip(SEEDS, v56, corr)}
            outside = ~cohort.date.str[8:10].isin(["06", "13", "20"]).to_numpy()
            report["episodes_outside_nm_days"] = describe(cohort[outside].reset_index(drop=True), base[outside],
                                                          corrected(cohort, np.mean(corr, axis=0))[outside])
    e = results["episodes"]
    rules = {"pooled_positive": e["gain_mse"] > 0,
             "beats_all_controls": all(e["gain_mse"] > results[a]["gain_mse"] for a in arms() if a != "episodes"),
             "both_months_positive": min(e["by_month"].values()) > 0,
             "best100_removed_positive": e["gain_without_best100"] > 0,
             "airport_day_ci_positive": e["airport_day_ci"][0] > 0,
             "all_seeds_positive": all(v > 0 for v in seeds.values()),
             "five_of_seven_cells_positive": sum((v or 0) > 0 for v in e["by_cell"].values()) >= 5}
    report.update(arms=results, seeds=seeds, rules=rules, verdict="pass" if all(rules.values()) else "fail")
    (HERE / f"research_320/results/episode_{stage}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("stage", "cohort_rows", "verdict", "rules", "seeds")}, indent=1, default=str))
    print("episodes", json.dumps({k: e[k] for k in ("v56_rmse", "arm_rmse", "gain_mse", "gain_without_best100",
                                                    "airport_day_ci", "by_month", "by_cell")}, default=str))
    print("controls", {a: round(r["gain_mse"], 1) for a, r in results.items() if a != "episodes"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    run(parser.parse_args().stage)

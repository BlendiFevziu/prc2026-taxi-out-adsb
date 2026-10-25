"""Pre-registered stand test (research_320/PREREG_stand.md): clean stand observations on top of v56. No upload.

Stages: "development" (fit once on v56's training dates, evaluate on 14 used dates) and "confirm" (the eight unread
reserve dates, allowed only after a development pass). Rows and 35 of the 36 columns come from Codex's own
multiairport_screen.extract() in the frozen checkout of 9884150; the 36th, detector_taxi_s, is the validated stand
detector with its guards, computed inside the same loop on the same matched visit and stand.
Run from the repository root:  python3 -u -B research_320/stand_experiment.py {development,confirm}
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import extract_episodes as X  # noqa: E402
from research_320.episodes import ground_mask  # noqa: E402

MS = X.v56_modules()
R = MS.R
from research_300 import tow_guard as TG  # noqa: E402  (the frozen checkout's copy, identical to fa2e2ab)

SEEDS, CONTROLS, SHRINK = (177, 277, 377), (4101, 4102, 4103, 4104, 4105), 0.5
TRAIN = tuple(MS.TRAIN)
STAGES = {"development": ("2025-01-04", "2025-01-13", "2025-01-14", "2025-01-20", "2025-01-21", "2025-01-26", "2025-01-28",
                          "2025-07-09", "2025-07-11", "2025-07-13", "2025-07-14", "2025-07-16", "2025-07-25", "2025-07-27"),
          "confirm": ("2025-01-01", "2025-01-02", "2025-01-06", "2025-01-18", "2025-07-07", "2025-07-12", "2025-07-26", "2025-07-31")}
CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07")
REPORTED_ONLY = ("EGLL/07",)
AIRPORTS = ("EDDF", "EGLL", "LEBL", "LSZH")
TRACK = list(MS.FEATURES) + ["detector_taxi_s"]
COLS = list(MS.COLS) + ["detector_taxi_s"]
CACHE = HERE / ".cache/research_320/stand"
V56, N = 261.2616, 344841


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detector_taxi(points, stand, takeoff):
    if stand is None or not len(points):
        return np.nan
    ts, lat, lon = (points[c].to_numpy(float) for c in ("ts", "lat", "lon"))
    gs = pd.to_numeric(points.gs, errors="coerce").to_numpy(float)
    ground = ground_mask(points.on_ground)[0]
    o = R.AT.guarded(ts, lat, lon, gs, stand, takeoff)
    g = TG.check(ts, lat, lon, gs, ground, stand, o, takeoff)
    return float(takeoff - (g["t_before"] + g["t_after"]) / 2) if g["status"] == "ok" else np.nan


def extract(dates, tag):
    target = CACHE / f"{tag}.parquet"
    if target.exists():
        return pd.read_parquet(target)
    CACHE.mkdir(parents=True, exist_ok=True)
    saved = (MS.TRAIN, MS.TEST, MS.CACHE, MS.summarise)
    original = MS.summarise
    MS.TRAIN, MS.TEST, MS.CACHE = tuple(dates), (), CACHE / f"days_{tag}"
    MS.CACHE.mkdir(parents=True, exist_ok=True)
    MS.summarise = lambda p, stand, takeoff: {**original(p, stand, takeoff), "detector_taxi_s": detector_taxi(p, stand, takeoff)}
    try:
        frame, _, _ = MS.extract()
    finally:
        MS.TRAIN, MS.TEST, MS.CACHE, MS.summarise = saved
    frame.to_parquet(target, index=False)
    return frame


def inputs(frame, arm):
    x = frame[COLS].copy()
    if arm == "context":
        x.loc[:, TRACK] = 0.0
    elif arm.startswith("shuffle_"):
        rng = np.random.default_rng(int(arm.split("_")[1]))
        cols = x.columns.get_indexer(TRACK)
        for pos in frame.groupby(["airport", "date"], sort=True).indices.values():
            x.iloc[pos, cols] = x.iloc[rng.permutation(pos), cols].to_numpy()
    return x


def arms():
    return ["real", "context"] + [f"shuffle_{c}" for c in CONTROLS]


def models():
    manifest = CACHE / "models/manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        if any(sha(CACHE / "models" / k) != h for k, h in m["model_hashes"].items()):
            raise SystemExit("a saved stand model changed since it was fitted")
        return m
    train = extract(TRAIN, "train")
    train = train[train.category.eq("observed_at_assigned_stand")].reset_index(drop=True)
    target = (train.y - train.comparator).to_numpy(float)
    (CACHE / "models").mkdir(parents=True, exist_ok=True)
    hashes = {}
    for arm in arms():
        x = inputs(train, arm)
        for seed in SEEDS:
            model = R.lgb.LGBMRegressor(objective="regression", n_estimators=200, learning_rate=.03, num_leaves=7,
                                        min_child_samples=80, reg_lambda=100., subsample=.8, subsample_freq=1,
                                        colsample_bytree=.9, max_bin=63, n_jobs=4, random_state=seed, verbosity=-1)
            path = CACHE / "models" / f"{arm}_{seed}.pkl"
            R.joblib.dump(model.fit(x, target), path)
            hashes[path.name] = sha(path)
    m = {"train_rows": int(len(train)), "train_rows_by_airport": train.airport.value_counts().to_dict(),
         "detector_estimates_in_training": int(train.detector_taxi_s.notna().sum()), "model_hashes": hashes}
    manifest.write_text(json.dumps(m, indent=2) + "\n")
    return m


def predict(frame, arm, seed=None):
    x = inputs(frame, arm)
    seeds = SEEDS if seed is None else (seed,)
    corr = np.mean([R.joblib.load(CACHE / "models" / f"{arm}_{s}.pkl").predict(x) for s in seeds], axis=0)
    return np.maximum(frame.comparator.to_numpy(float) + SHRINK * corr, 300.0)


def describe(frame, pred):
    y, base = frame.y.to_numpy(float), frame.comparator.to_numpy(float)
    gain = (base - y) ** 2 - (pred - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    cell = (frame.airport + "/" + frame.date.str[5:7]).to_numpy()
    return {"rows": int(len(frame)), "base_rmse": float(np.sqrt(np.mean((base - y) ** 2))),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "gain_mse": float(gain.mean()),
            "gain_without_best100": float(gain[keep].mean()),
            "airport_day_ci": list(map(float, R.block_bootstrap(gain, (frame.airport + "/" + frame.date).to_numpy()))),
            "by_month": {m: float(gain[frame.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")},
            "by_airport": {a: float(gain[frame.airport.eq(a).to_numpy()].mean()) for a in AIRPORTS if frame.airport.eq(a).any()},
            "by_cell": {c: {"rows": int((cell == c).sum()), "gain_mse": float(gain[cell == c].mean())} for c in sorted(set(cell))},
            "by_date": {d: float(gain[frame.date.eq(d).to_numpy()].mean()) for d in sorted(frame.date.unique())}}


def run(stage):
    if stage == "confirm":
        dev = HERE / "research_320/results/stand_development.json"
        if not dev.exists() or json.loads(dev.read_text())["verdict"] != "pass":
            raise SystemExit("the confirmation is allowed only after a development pass")
    m = models()
    frame = extract(STAGES[stage], stage)
    stand = frame[frame.category.eq("observed_at_assigned_stand")].copy()
    stand["cell"] = stand.airport + "/" + stand.date.str[5:7]
    cohort = stand[stand.cell.isin(CELLS)].reset_index(drop=True)
    report = {"prereg": "research_320/PREREG_stand.md", "stage": stage, "dates": list(STAGES[stage]), "models": m,
              "cohort_rows": int(len(cohort)), "detector_estimates": int(cohort.detector_taxi_s.notna().sum()), "arms": {}}
    for arm in arms():
        report["arms"][arm] = describe(cohort, predict(cohort, arm))
    report["seeds"] = {s: describe(cohort, predict(cohort, "real", s))["gain_mse"] for s in SEEDS}
    outside = ~cohort.date.str[8:10].isin(["06", "13", "20"]).to_numpy()
    report["real_outside_days_6_13_20"] = describe(cohort[outside].reset_index(drop=True), predict(cohort, "real")[outside])
    for c in REPORTED_ONLY:
        g = stand[stand.cell.eq(c)].reset_index(drop=True)
        report[f"reported_only_{c}"] = describe(g, predict(g, "real")) if len(g) else {"rows": 0}
    real = report["arms"]["real"]
    rules = {"pooled_positive": real["gain_mse"] > 0,
             "beats_context_and_controls": all(real["gain_mse"] > report["arms"][a]["gain_mse"] for a in arms() if a != "real"),
             "both_months_positive": min(real["by_month"].values()) > 0,
             "best100_removed_positive": real["gain_without_best100"] > 0,
             "airport_day_ci_positive": real["airport_day_ci"][0] > 0,
             "all_seeds_positive": all(v > 0 for v in report["seeds"].values()),
             "three_of_four_airports_positive": sum(v > 0 for v in real["by_airport"].values()) >= 3}
    if stage == "development":
        cov = json.loads(subprocess.run(["git", "show", "9884150:research_310/results/full_coverage_2026.json"], cwd=HERE,
                                        capture_output=True, text=True, check=True).stdout)["cells"]
        rows2026 = {c: cov[c]["category_counts"].get("observed_at_assigned_stand", 0) * 56 / 62 for c in CELLS}
        mse = sum(rows2026[c] * real["by_cell"].get(c, {"gain_mse": 0.0})["gain_mse"] for c in CELLS) / N
        report["planning"] = {"rows_2026_after_nm_allowance": rows2026, "mse_per_scored_row": mse,
                              "full_transfer_s": float(V56 - np.sqrt(V56 ** 2 - mse)) if mse > 0 else float(mse / (2 * V56)),
                              "half_transfer_s": float(V56 - np.sqrt(V56 ** 2 - mse / 2)) if mse > 0 else float(mse / (4 * V56))}
        rules["minimum_meaningful_0_3_s"] = report["planning"]["full_transfer_s"] >= 0.3
    report.update(rules=rules, verdict="pass" if all(rules.values()) else "fail")
    (HERE / f"research_320/results/stand_{stage}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report.get(k) for k in ("stage", "cohort_rows", "detector_estimates", "verdict", "rules", "seeds", "planning")},
                     indent=1, default=str))
    print("real", json.dumps({k: real[k] for k in ("base_rmse", "rmse", "gain_mse", "gain_without_best100", "airport_day_ci",
                                                   "by_month", "by_airport", "by_cell")}, default=str))
    print("other arms", {a: round(r["gain_mse"], 1) for a, r in report["arms"].items() if a != "real"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    run(parser.parse_args().stage)

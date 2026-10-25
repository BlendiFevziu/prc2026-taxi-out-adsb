"""Pre-registered training-volume and capacity experiment for v56's partial-track correction
(research_320/PREREG_volume_capacity.md with its addendum A). No candidate, no upload.

Arms (s = 1.0, floor 300, v56's 35 columns): 1 = S x current (must reproduce the deployed models), 2 = E x current,
3 = S x larger, 4 = E x larger; E-fixed descriptive. Development on B1, controls for survivors, replication on B2 only
for arms that survive both. Rows and features from Codex's frozen code (research_320 extraction caches). The
extraction, the Amsterdam detector replay and each arm's fits hold the shared lock, released for 35 s in between.
Run from the repository root:  python3 -u -B research_320/volume_capacity.py
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import stand_experiment as S  # noqa: E402  (frozen v56 modules, extraction, caches)
from research_320.episode_experiment import detector_ids  # noqa: E402
from research_300.heavy_lock import heavy_work  # noqa: E402

MS, R = S.MS, S.R
from research_310 import partial_coverage_2026 as P  # noqa: E402  (the frozen checkout's drift rule)

SEEDS, CONTROLS = (177, 277, 377), (3101, 3102, 3103, 3104, 3105)
S_DATES = tuple(MS.TRAIN)
E_EXTRA = ("2025-01-16", "2025-01-23", "2025-07-03", "2025-07-21", "2025-07-22", "2025-07-29")
PARITY = ("2025-01-23", "2025-07-21", "2025-07-29")          # extracted before for the s = 1.0 confirmation
B1, B2 = tuple(S.STAGES["development"]), tuple(S.STAGES["confirm"])
CELLS = ("EDDF/01", "EGLL/01", "EHAM/07", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07")
ALL_CELLS = tuple(f"{a}/{m}" for a in MS.AIRPORTS for m in ("01", "07"))
CAP = {"current": (200, 7), "larger": (400, 15)}
ARMS = {"arm2": ("E", "current"), "arm3": ("S", "larger"), "arm4": ("E", "larger")}
DESCRIPTIVE = {"e_fixed": ("E_fixed", "current")}
ALPHA_DEV = 0.05 / 4                                          # addendum A: four tested arms
GAP_S = 35                                                    # longer than the shared helper's 30 s poll
REBUILD = HERE / ".cache/research_320/v56_rebuild"
CACHE = HERE / ".cache/research_320/volume"
OUT = HERE / "research_320/results/volume_capacity.json"
V59, N = 259.4958, 344841
ID = "MVT_ID_mvt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_path(arm, inputs_arm, seed):
    return CACHE / "models" / f"{arm}_{inputs_arm}_{seed}.pkl"


def lgbm(cap, seed):
    trees, leaves = CAP[cap]
    return R.lgb.LGBMRegressor(objective="regression", n_estimators=trees, learning_rate=.03, num_leaves=leaves,
                               min_child_samples=80, reg_lambda=100., subsample=.8, subsample_freq=1,
                               colsample_bytree=.9, max_bin=63, n_jobs=4, random_state=seed, verbosity=-1)


def fit_arm(train, arm, cap, inputs_arms=("real",)):
    """Every seed (and input arm) of one arm inside one hold of the lock; saved models are reused, never refitted."""
    todo = [(a, s) for a in inputs_arms for s in SEEDS if not model_path(arm, a, s).exists()]
    if not todo:
        return
    with heavy_work(f"volume/capacity {arm} ({cap}, {'+'.join(inputs_arms)})", poll_s=10.0):
        target = (train.y - train.comparator).to_numpy(float)
        for a, s in todo:
            model = lgbm(cap, s).fit(MS.inputs(train, a), target)
            model_path(arm, a, s).parent.mkdir(parents=True, exist_ok=True)
            R.joblib.dump(model, model_path(arm, a, s))
            print("fitted", arm, a, s, flush=True)
    time.sleep(GAP_S)


def seed_corrections(frame, arm, inputs_arm="real", folder=None):
    x = MS.inputs(frame, inputs_arm)
    out = {}
    for s in SEEDS:
        path = (folder / f"real_{s}.pkl") if folder else model_path(arm, inputs_arm, s)
        m = R.joblib.load(path)
        p = m.predict(x)
        if not np.array_equal(m.predict(x.iloc[::-1])[::-1], p):
            raise SystemExit(f"reverse-order replay failed for {path.name}")
        out[s] = p
    return out


def pred(frame, corr):
    return np.maximum(frame.comparator.to_numpy(float) + 1.0 * corr, 300.0)


def bootstrap(gain, blocks, alpha, draws=2000, seed=0):
    """research_300.incremental.block_bootstrap at a two-sided level of 1 - alpha (same draws, same estimator)."""
    gain, blocks = np.asarray(gain, float), np.asarray(blocks)
    keys = np.unique(blocks)
    sums = np.array([gain[blocks == k].sum() for k in keys])
    counts = np.array([(blocks == k).sum() for k in keys])
    pick = np.random.default_rng(seed).integers(0, len(keys), size=(draws, len(keys)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return [float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))]


def judge(frame, ref_se, arm_se, seed_gains, alpha):
    gain = ref_se - arm_se
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    cell = (frame.airport + "/" + frame.date.str[5:7]).to_numpy()
    date = frame.date.to_numpy()
    r = {"rows": int(len(gain)), "reference_rmse": float(np.sqrt(ref_se.mean())), "arm_rmse": float(np.sqrt(arm_se.mean())),
         "gain_mse": float(gain.mean()), "without_best100": float(gain[keep].mean()),
         "interval": bootstrap(gain, (frame.airport + "/" + frame.date).to_numpy(), alpha), "interval_level": 1 - alpha,
         "by_month": {m: float(gain[frame.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")},
         "by_cell": {c: float(gain[cell == c].mean()) for c in CELLS if (cell == c).any()},
         "cell_rows": {c: int((cell == c).sum()) for c in CELLS},
         "by_date": {d: float(gain[date == d].mean()) for d in sorted(set(date))},
         "by_cell_date": {f"{c} {d}": {"rows": int(((cell == c) & (date == d)).sum()),
                                       "gain_mse": float(gain[(cell == c) & (date == d)].mean())}
                          for c in CELLS for d in sorted(set(date)) if ((cell == c) & (date == d)).any()},
         "seeds": seed_gains}
    r["rules"] = {"pooled_positive": r["gain_mse"] > 0, "both_months_positive": min(r["by_month"].values()) > 0,
                  "best100_removed_positive": r["without_best100"] > 0, "adjusted_interval_positive": r["interval"][0] > 0,
                  "every_seed_positive": all(v > 0 for v in seed_gains.values()),
                  "five_of_seven_cells_positive": sum(v > 0 for v in r["by_cell"].values()) >= 5}
    r["pass"] = all(r["rules"].values())
    return r


def eligible(frame, detector):
    f = frame.copy()
    f["cell"] = f.airport + "/" + f.date.str[5:7]
    return f[f.cell.isin(CELLS) & f.category.eq("partial_ground") & ~f[ID].isin(detector)].reset_index(drop=True)


def drift(reference, rows2026):
    """v56's release gate per airport/month (research_310/RELEASE_PROTOCOL.md step 2, Codex's own drift function)."""
    cells = {}
    for c in ALL_CELLS:
        ap, m = c.split("/")
        r = reference[reference.airport.eq(ap) & reference.date.str[5:7].eq(m)]
        p = rows2026[rows2026.airport.eq(ap) & rows2026.month.eq(m) & rows2026.category.eq("partial_ground")]
        stats = P.drift(r, p)
        constant = []
        for feature in MS.FEATURES:
            a, b = r[feature].dropna(), p[feature].dropna()
            if len(a) and a.nunique() == 1 and len(b) and not b.eq(a.iloc[0]).all():
                constant.append(feature)
        flags = [f for f, v in stats.items() if v["diagnostic_flag"]]
        cells[c] = {"reference_rows": int(len(r)), "rows_2026": int(len(p)), "flagged": flags, "constant_changed": constant,
                    "allowed": not flags and not constant and len(p) > 0}
    return cells


def same_rows(a, b, columns):
    a, b = a.sort_values(ID).reset_index(drop=True), b.sort_values(ID).reset_index(drop=True)
    return bool(a[ID].equals(b[ID]) and a.category.equals(b.category)
                and np.array_equal(a[columns].to_numpy(float), b[columns].to_numpy(float), equal_nan=True))


def main():
    report = {"prereg": "research_320/PREREG_volume_capacity.md", "addendum": "A", "alpha_development": ALPHA_DEV}
    day = lambda d: pd.Timestamp(d)
    e_dates = S_DATES + E_EXTRA
    report["min_embargo_days"] = min(abs((day(t) - day(e)).days) for t in e_dates for e in B1 + B2)
    if report["min_embargo_days"] < 2 or not P.__file__.startswith(str(S.X.V56)):
        raise SystemExit("embargo or module origin check failed")
    gate = {}
    train_all = S.extract(S_DATES, "train")
    train_s = train_all[train_all.category.eq("partial_ground")].reset_index(drop=True)
    rebuilt = json.loads((REBUILD / "multiairport_screen_rebuilt.json").read_text())
    gate["train_rows_equal_v56"] = len(train_s) == rebuilt["train_rows"]
    with heavy_work("volume/capacity: extract six E days", poll_s=10.0):
        extra_all = S.extract(E_EXTRA, "volume_extra")
    time.sleep(GAP_S)
    old = pd.read_parquet(S.CACHE / "shrink_confirm.parquet")
    gate["parity_three_reextracted_days"] = same_rows(extra_all[extra_all.date.isin(PARITY)], old[old.date.isin(PARITY)],
                                                      list(MS.COLS) + ["y"])
    train_e = pd.concat([train_s, extra_all[extra_all.category.eq("partial_ground")]], ignore_index=True)
    fixed = train_e.sample(n=len(train_s), random_state=4242).reset_index(drop=True)
    data = {"S": train_s, "E": train_e, "E_fixed": fixed}
    report["training_rows"] = {k: int(len(v)) for k, v in data.items()}
    report["training_rows_by_cell"] = {k: {c: int(n) for c, n in v.groupby(v.airport + "/" + v.date.str[5:7]).size().items()}
                                       for k, v in data.items()}
    report["training_dates"] = {"S": list(S_DATES), "E": list(e_dates)}

    # arm 1 first; it must reproduce the deployed models and they Codex's saved predictions, before any label is read
    fit_arm(train_s, "arm1", "current")
    b1_all = S.extract(B1, "development")
    saved = pd.read_parquet(REBUILD / "predictions.parquet")
    saved = saved[saved.category.eq("partial_ground")].reset_index(drop=True)
    mine = b1_all[b1_all.date.isin(MS.TEST) & b1_all.category.eq("partial_ground")]
    gate["b1_rows_equal_codex_saved_rows"] = same_rows(mine, saved, list(MS.COLS))
    dep_saved = seed_corrections(saved, None, folder=REBUILD)
    gate["deployed_reproduce_codex_saved_predictions"] = bool(np.array_equal(
        np.maximum(saved.comparator.to_numpy(float) + .5 * np.mean([dep_saved[s] for s in SEEDS], axis=0), 300.),
        saved.real.to_numpy(float)))
    with heavy_work("volume/capacity: Amsterdam detector replay, B1 July days", poll_s=10.0):
        det1 = detector_ids(B1)
    time.sleep(GAP_S)
    b1 = eligible(b1_all, det1)
    deployed = seed_corrections(b1, None, folder=REBUILD)
    corr = {"arm1": seed_corrections(b1, "arm1")}
    gate["arm1_equals_deployed_every_seed"] = all(np.array_equal(corr["arm1"][s], deployed[s]) for s in SEEDS)
    gate["arm1_model_bytes_equal_deployed"] = all(sha(model_path("arm1", "real", s)) == sha(REBUILD / f"real_{s}.pkl") for s in SEEDS)
    report["gate"] = gate
    report["b1_rows"], report["b1_detector_rows_excluded"] = int(len(b1)), int(b1_all[ID].isin(det1).sum())
    if not all(v for k, v in gate.items() if k != "arm1_model_bytes_equal_deployed"):
        report["verdict"] = "gate_failed"
        OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
        raise SystemExit(f"a reproduction gate failed: {gate}")
    print("gates", gate, flush=True)

    for arm, (d, cap) in {**ARMS, **DESCRIPTIVE}.items():
        fit_arm(data[d], arm, cap)
        corr[arm] = seed_corrections(b1, arm)
    y1 = b1.y.to_numpy(float)
    se = lambda p: (p - y1) ** 2
    mean = lambda c: np.mean([c[s] for s in SEEDS], axis=0)
    ref = se(pred(b1, mean(corr["arm1"])))
    blocks = (b1.airport + "/" + b1.date).to_numpy()
    probe = np.random.default_rng(1).normal(size=len(ref))
    if bootstrap(probe, blocks, .05) != list(map(float, R.block_bootstrap(probe, blocks))):
        raise SystemExit("the adjusted bootstrap does not match the shared one at 95 %")
    report["arm1_vs_pre_ads_b_comparator_b1"] = float((se(b1.comparator.to_numpy(float)) - ref).mean())
    dev = {}
    for arm in list(ARMS) + list(DESCRIPTIVE):
        seeds = {s: float((se(pred(b1, corr["arm1"][s])) - se(pred(b1, corr[arm][s]))).mean()) for s in SEEDS}
        dev[arm] = judge(b1, ref, se(pred(b1, mean(corr[arm]))), seeds, ALPHA_DEV)
        dev[arm]["vs_pre_ads_b_comparator"] = float((se(b1.comparator.to_numpy(float)) - se(pred(b1, mean(corr[arm])))).mean())
        print(arm, "PASS" if dev[arm]["pass"] else "fail",
              json.dumps({k: dev[arm][k] for k in ("gain_mse", "without_best100", "interval", "by_month", "by_cell", "seeds")}), flush=True)
    report["development"] = dev
    survivors = [a for a in ARMS if dev[a]["pass"]]
    report["survivors_development"] = survivors

    controls = {}
    for arm in survivors:
        d, cap = ARMS[arm]
        names = ["context"] + [f"shuffle_{c}" for c in CONTROLS]
        fit_arm(data[d], arm, cap, inputs_arms=names)
        gains = {c: float((ref - se(pred(b1, mean(seed_corrections(b1, arm, c))))).mean()) for c in names}
        controls[arm] = {"control_gains_vs_arm1": gains, "beats_all": all(dev[arm]["gain_mse"] > v for v in gains.values())}
        print("controls", arm, {k: round(v, 1) for k, v in gains.items()}, "beats all:", controls[arm]["beats_all"], flush=True)
    report["controls"] = controls
    passing = [a for a in survivors if controls[a]["beats_all"]]
    report["survivors_controls"] = passing

    rows2026 = pd.read_parquet(HERE / ".cache/codex_v56/full_coverage_2026/features.parquet")
    cov = json.loads(subprocess.run(["git", "show", "9884150:research_310/results/full_coverage_2026.json"], cwd=HERE,
                                    capture_output=True, text=True, check=True).stdout)
    arm1_allowed = [c for c, v in drift(train_s, rows2026).items() if v["allowed"]]
    report["drift_replay_arm1"] = {"allowed": arm1_allowed, "equals_codex_allowed_cells": arm1_allowed == cov["allowed_cells"],
                                   "features_2026_sha_equal": sha(HERE / ".cache/codex_v56/full_coverage_2026/features.parquet") == cov["features_sha256"]}
    saved_pred = {ID: b1[ID], "date": b1.date, "airport": b1.airport, "y": y1,
                  **{f"{a}_corr{s}": corr[a][s] for a in corr for s in SEEDS}, **{f"deployed_corr{s}": deployed[s] for s in SEEDS}}
    if passing:
        with heavy_work("volume/capacity: Amsterdam detector replay, B2 July days", poll_s=10.0):
            det2 = detector_ids(B2)
        b2 = eligible(S.extract(B2, "confirm"), det2)
        y2 = b2.y.to_numpy(float)
        se2 = lambda p: (p - y2) ** 2
        c2 = {a: seed_corrections(b2, a) for a in ["arm1"] + passing}
        ref2 = se2(pred(b2, mean(c2["arm1"])))
        rep = {}
        for arm in passing:
            seeds = {s: float((se2(pred(b2, c2["arm1"][s])) - se2(pred(b2, c2[arm][s]))).mean()) for s in SEEDS}
            rep[arm] = judge(b2, ref2, se2(pred(b2, mean(c2[arm]))), seeds, 0.05 / len(passing))
            print("replication", arm, "PASS" if rep[arm]["pass"] else "fail",
                  json.dumps({k: rep[arm][k] for k in ("gain_mse", "without_best100", "interval", "by_month", "by_cell", "seeds")}), flush=True)
        report["b2_rows"], report["replication"] = int(len(b2)), rep
        pd.DataFrame({ID: b2[ID], "date": b2.date, "airport": b2.airport, "y": y2,
                      **{f"{a}_corr{s}": c2[a][s] for a in c2 for s in SEEDS}}).to_parquet(CACHE / "b2_predictions.parquet", index=False)
        final = [a for a in passing if rep[a]["pass"]]
        report["passing_both"] = final
        eligible_2026 = json.loads(subprocess.run(["git", "show", "9884150:research_310/results/partial_candidate_v55base.json"], cwd=HERE,
                                                  capture_output=True, text=True, check=True).stdout)["cells"]
        report["drift"], report["planning"] = {}, {}
        for arm in final:
            cells = drift(data[ARMS[arm][0]], rows2026)
            allowed = [c for c in CELLS if cells[c]["allowed"]]
            mse = sum(eligible_2026[c]["scored_eligible"] * rep[arm]["by_cell"].get(c, 0.0) for c in allowed) / N
            report["drift"][arm] = cells
            report["planning"][arm] = {"cells_allowed": allowed, "mse_per_scored_row": mse,
                                       "full_s": float(V59 - np.sqrt(V59 ** 2 - mse)) if mse > 0 else float(mse / (2 * V59)),
                                       "half_s": float(V59 - np.sqrt(V59 ** 2 - mse / 2)) if mse > 0 else float(mse / (4 * V59)),
                                       "minimum_0_1_s": bool(mse > 0 and V59 - np.sqrt(V59 ** 2 - mse) >= 0.1)}
    pd.DataFrame(saved_pred).to_parquet(CACHE / "b1_predictions.parquet", index=False)
    report["input_hashes"] = {p.name: sha(p) for p in (S.CACHE / "train.parquet", S.CACHE / "development.parquet",
                                                        S.CACHE / "volume_extra.parquet", REBUILD / "predictions.parquet")}
    report["e_fixed_ids_sha256"] = hashlib.sha256("\n".join(map(str, fixed[ID])).encode()).hexdigest()
    report["model_hashes"] = {p.name: sha(p) for p in sorted((CACHE / "models").glob("*.pkl"))}
    report["predictions_sha256"] = {p.name: sha(p) for p in sorted(CACHE.glob("b*_predictions.parquet"))}
    report["verdict"] = "pass" if report.get("passing_both") else "no_arm_passes"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("verdict", report["verdict"], "| planning", report.get("planning"), "| drift replay", report["drift_replay_arm1"], flush=True)


if __name__ == "__main__":
    main()

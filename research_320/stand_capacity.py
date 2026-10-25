"""Weight and capacity for the stand model (research_320/PREREG_stand_capacity.md). No candidate, no upload.

Rows, 36 columns and controls from research_320/stand_experiment.py; the reference is its saved deployed model at
s = 0.5. Every fit runs under the shared lock, one arm per hold.
Run from the repository root:  python3 -u -B research_320/stand_capacity.py
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import stand_experiment as S  # noqa: E402
from research_320 import volume_capacity as V  # noqa: E402  (bootstrap, sha)
from research_300.heavy_lock import heavy_work  # noqa: E402

R = S.R
SEEDS, CONTROLS = S.SEEDS, S.CONTROLS
CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07")
ARMS = {"k1": (200, 7), "k2": (400, 15), "k3": (800, 31)}
ALPHA = 0.05 / 3
CACHE = HERE / ".cache/research_320/stand_capacity"
OUT = HERE / "research_320/results/stand_capacity.json"
V61, N = 258.9401, 344841


def path(arm, inputs_arm, seed):
    return CACHE / "models" / f"{arm}_{inputs_arm}_{seed}.pkl"


def fit(train, arm, names=("real",)):
    todo = [(a, s) for a in names for s in SEEDS if not path(arm, a, s).exists()]
    if not todo:
        return
    trees, leaves = ARMS[arm]
    with heavy_work(f"stand capacity {arm} {'+'.join(names)}", poll_s=10.0):
        target = (train.y - train.comparator).to_numpy(float)
        for a, s in todo:
            m = R.lgb.LGBMRegressor(objective="regression", n_estimators=trees, learning_rate=.03, num_leaves=leaves,
                                    min_child_samples=80, reg_lambda=100., subsample=.8, subsample_freq=1, colsample_bytree=.9,
                                    max_bin=63, n_jobs=4, random_state=s, verbosity=-1).fit(S.inputs(train, a), target)
            path(arm, a, s).parent.mkdir(parents=True, exist_ok=True)
            R.joblib.dump(m, path(arm, a, s))
    time.sleep(35)


def corrections(frame, arm, inputs_arm="real"):
    x = S.inputs(frame, inputs_arm)
    folder = (lambda s: S.CACHE / "models" / f"real_{s}.pkl") if arm == "ref" else (lambda s: path(arm, inputs_arm, s))
    return {s: R.joblib.load(folder(s)).predict(x) for s in SEEDS}


def pred(frame, corr, weight):
    return np.maximum(frame.comparator.to_numpy(float) + weight * corr, 300.0)


def judge(frame, ref_se, arm_se, seeds, alpha):
    gain = ref_se - arm_se
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    cell = frame.cell.to_numpy()
    r = {"rows": int(len(gain)), "gain_mse": float(gain.mean()), "without_best100": float(gain[keep].mean()),
         "interval": V.bootstrap(gain, (frame.airport + "/" + frame.date).to_numpy(), alpha), "interval_level": 1 - alpha,
         "by_month": {m: float(gain[frame.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")},
         "by_cell": {c: float(gain[cell == c].mean()) for c in CELLS if (cell == c).any()},
         "cell_rows": {c: int((cell == c).sum()) for c in CELLS}, "seeds": seeds}
    r["rules"] = {"pooled_positive": r["gain_mse"] > 0, "both_months_positive": min(r["by_month"].values()) > 0,
                  "best100_removed_positive": r["without_best100"] > 0, "adjusted_interval_positive": r["interval"][0] > 0,
                  "every_seed_positive": all(v > 0 for v in seeds.values()),
                  "three_of_five_cells_positive": sum(v > 0 for v in r["by_cell"].values()) >= 3}
    r["pass"] = all(r["rules"].values())
    return r


def rows(tag, dates):
    f = S.extract(dates, tag)
    f["cell"] = f.airport + "/" + f.date.str[5:7]
    return f[f.category.eq("observed_at_assigned_stand") & f.cell.isin(CELLS)].reset_index(drop=True)


def evaluate(frame, arms, alpha):
    y = frame.y.to_numpy(float)
    se = lambda p: (p - y) ** 2
    mean = lambda c: np.mean([c[s] for s in SEEDS], axis=0)
    rc = corrections(frame, "ref")
    ref = se(pred(frame, mean(rc), 0.5))
    out = {}
    for arm in arms:
        c = corrections(frame, arm)
        seeds = {s: float((se(pred(frame, rc[s], 0.5)) - se(pred(frame, c[s], 1.0))).mean()) for s in SEEDS}
        out[arm] = judge(frame, ref, se(pred(frame, mean(c), 1.0)), seeds, alpha)
        print(arm, "PASS" if out[arm]["pass"] else "fail", json.dumps({k: out[arm][k] for k in ("gain_mse", "without_best100", "interval", "by_month", "by_cell", "seeds")}), flush=True)
    return out, ref, se, mean


def main():
    S.models()
    s_rows = S.extract(S.TRAIN, "train")
    extra = S.extract(V.E_EXTRA, "volume_extra")
    train = pd.concat([s_rows[s_rows.category.eq("observed_at_assigned_stand")],
                       extra[extra.category.eq("observed_at_assigned_stand")]], ignore_index=True)
    report = {"prereg": "research_320/PREREG_stand_capacity.md", "training_rows": int(len(train)), "alpha_development": ALPHA}
    b1 = rows("development", S.STAGES["development"])
    report["b1_rows"] = int(len(b1))
    for arm in ARMS:
        fit(train, arm)
    dev, ref, se, mean = evaluate(b1, ARMS, ALPHA)
    report["development"] = dev
    survivors = [a for a in ARMS if dev[a]["pass"]]
    controls = {}
    for arm in survivors:
        names = ["context"] + [f"shuffle_{c}" for c in CONTROLS]
        fit(train, arm, names)
        gains = {n: float((ref - se(pred(b1, mean(corrections(b1, arm, n)), 1.0))).mean()) for n in names}
        controls[arm] = {"control_gains_vs_reference": gains, "beats_all": all(dev[arm]["gain_mse"] > v for v in gains.values())}
    report["controls"] = controls
    passing = [a for a in survivors if controls[a]["beats_all"]]
    report["survivors"] = passing
    if passing:
        b2 = rows("confirm", S.STAGES["confirm"])
        report["b2_rows"] = int(len(b2))
        rep, _, _, _ = evaluate(b2, passing, 0.05 / len(passing))
        report["replication"] = rep
        final = [a for a in passing if rep[a]["pass"]]
        report["passing_both"] = final
        if final:
            chosen = max(final, key=lambda a: dev[a]["gain_mse"])
            elig = json.loads((HERE / "research_320/results/stand_candidate.json").read_text())["eligible_rows_by_cell"]
            nm = json.loads((HERE / "research_320/results/nmfree_plus.json").read_text())["cells"]["stand"]
            n26 = {c: elig.get(c, 0) + nm.get(c, 0) for c in CELLS}
            mse = sum(n26[c] * rep[chosen]["by_cell"].get(c, 0.0) for c in CELLS) / N
            report["planning"] = {"chosen": chosen, "rows_2026": n26, "mse_per_scored_row": mse,
                                  "full_s": float(V61 - np.sqrt(V61 ** 2 - mse)) if mse > 0 else float(mse / (2 * V61)),
                                  "minimum_0_1_s": bool(mse > 0 and V61 - np.sqrt(V61 ** 2 - mse) >= 0.1)}
    report["model_hashes"] = {p.name: V.sha(p) for p in sorted((CACHE / "models").glob("*.pkl"))} if (CACHE / "models").exists() else {}
    report["verdict"] = "pass" if report.get("planning", {}).get("minimum_0_1_s") else "no_arm_passes"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("verdict", report["verdict"], "| planning", report.get("planning"), flush=True)


if __name__ == "__main__":
    main()

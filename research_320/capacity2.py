"""Second capacity step for the partial-track correction (research_320/PREREG_capacity2.md). No candidate, no upload.

Reuses research_320/volume_capacity.py unchanged: same training set E, B1/B2 rows, fitting, judging and bootstrap;
the reference is its saved arm 4. Every fit runs under the shared lock, one arm per hold.
Run from the repository root:  python3 -u -B research_320/capacity2.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import volume_capacity as V  # noqa: E402

S, MS, R = V.S, V.MS, V.R
V.CAP.update({"c800": (800, 31), "c1600": (1600, 63)})
ARMS = {"c1": "c800", "c2": "c1600"}
ALPHA = 0.05 / 2
OUT = HERE / "research_320/results/capacity2.json"
V61, N = 258.9401, 344841


def cohort(tag, dates):
    with V.heavy_work(f"capacity2: detector replay {tag}", poll_s=10.0):
        det = V.detector_ids(dates)
    return V.eligible(S.extract(dates, tag), det)


def main():
    vol = json.loads((HERE / "research_320/results/volume_capacity.json").read_text())
    if any(V.sha(V.model_path("arm4", "real", s)) != vol["model_hashes"][f"arm4_real_{s}.pkl"] for s in V.SEEDS):
        raise SystemExit("the saved arm 4 models changed")
    train_s = S.extract(V.S_DATES, "train")
    train_s = train_s[train_s.category.eq("partial_ground")].reset_index(drop=True)
    extra = S.extract(V.E_EXTRA, "volume_extra")
    train_e = pd.concat([train_s, extra[extra.category.eq("partial_ground")]], ignore_index=True)
    report = {"prereg": "research_320/PREREG_capacity2.md", "training_rows": int(len(train_e)), "alpha_development": ALPHA}
    if len(train_e) != vol["training_rows"]["E"]:
        raise SystemExit("training set E differs from the first study")
    b1 = cohort("development", V.B1)
    if len(b1) != vol["b1_rows"]:
        raise SystemExit("B1 rows differ from the first study")
    for arm, cap in ARMS.items():
        V.fit_arm(train_e, arm, cap)
    y = b1.y.to_numpy(float)
    se = lambda p: (p - y) ** 2
    mean = lambda c: np.mean([c[s] for s in V.SEEDS], axis=0)
    ref_c = V.seed_corrections(b1, "arm4")
    ref = se(V.pred(b1, mean(ref_c)))
    report["arm4_vs_comparator_b1"] = float((se(b1.comparator.to_numpy(float)) - ref).mean())
    dev, corr = {}, {}
    for arm in ARMS:
        corr[arm] = V.seed_corrections(b1, arm)
        seeds = {s: float((se(V.pred(b1, ref_c[s])) - se(V.pred(b1, corr[arm][s]))).mean()) for s in V.SEEDS}
        dev[arm] = V.judge(b1, ref, se(V.pred(b1, mean(corr[arm]))), seeds, ALPHA)
        print(arm, "PASS" if dev[arm]["pass"] else "fail", json.dumps({k: dev[arm][k] for k in ("gain_mse", "without_best100", "interval", "by_month", "by_cell", "seeds")}), flush=True)
    report["development"] = dev
    survivors = [a for a in ARMS if dev[a]["pass"]]
    controls = {}
    for arm in survivors:
        names = ["context"] + [f"shuffle_{c}" for c in V.CONTROLS]
        V.fit_arm(train_e, arm, ARMS[arm], inputs_arms=names)
        gains = {c: float((ref - se(V.pred(b1, mean(V.seed_corrections(b1, arm, c))))).mean()) for c in names}
        controls[arm] = {"control_gains_vs_arm4": gains, "beats_all": all(dev[arm]["gain_mse"] > v for v in gains.values())}
    report["controls"] = controls
    passing = [a for a in survivors if controls[a]["beats_all"]]
    report["survivors"] = passing
    if passing:
        b2 = cohort("confirm", V.B2)
        y2 = b2.y.to_numpy(float)
        se2 = lambda p: (p - y2) ** 2
        ref2_c = V.seed_corrections(b2, "arm4")
        ref2 = se2(V.pred(b2, mean(ref2_c)))
        rep = {}
        for arm in passing:
            c2 = V.seed_corrections(b2, arm)
            seeds = {s: float((se2(V.pred(b2, ref2_c[s])) - se2(V.pred(b2, c2[s]))).mean()) for s in V.SEEDS}
            rep[arm] = V.judge(b2, ref2, se2(V.pred(b2, mean(c2))), seeds, 0.05 / len(passing))
            print("replication", arm, "PASS" if rep[arm]["pass"] else "fail", json.dumps({k: rep[arm][k] for k in ("gain_mse", "without_best100", "interval", "by_month", "by_cell", "seeds")}), flush=True)
        report["replication"] = rep
        final = [a for a in passing if rep[a]["pass"]]
        report["passing_both"] = final
        if final:
            chosen = max(final, key=lambda a: dev[a]["gain_mse"])     # B1 decides the choice, B2 only pass or fail
            cells = json.loads(V.subprocess.run(["git", "show", "9884150:research_310/results/partial_candidate_v55base.json"], cwd=HERE,
                                                capture_output=True, text=True, check=True).stdout)["cells"]
            nm = json.loads((HERE / "research_320/results/nmfree_plus.json").read_text())["cells"]["partial"]
            rows = {c: cells[c]["scored_eligible"] + nm.get(c, 0) for c in V.CELLS}
            mse = sum(rows[c] * rep[chosen]["by_cell"].get(c, 0.0) for c in V.CELLS) / N
            report["planning"] = {"chosen": chosen, "rows_2026": rows, "mse_per_scored_row": mse,
                                  "full_s": float(V61 - np.sqrt(V61 ** 2 - mse)) if mse > 0 else float(mse / (2 * V61)),
                                  "at_0_88_s": float(V61 - np.sqrt(V61 ** 2 - 0.88 * mse)) if mse > 0 else 0.0,
                                  "minimum_0_1_s": bool(mse > 0 and V61 - np.sqrt(V61 ** 2 - mse) >= 0.1)}
    report["model_hashes"] = {p.name: V.sha(p) for p in sorted((V.CACHE / "models").glob("c[12]_*.pkl"))}
    report["verdict"] = "pass" if report.get("planning", {}).get("minimum_0_1_s") else "no_arm_passes"
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("verdict", report["verdict"], "| planning", report.get("planning"), flush=True)


if __name__ == "__main__":
    main()

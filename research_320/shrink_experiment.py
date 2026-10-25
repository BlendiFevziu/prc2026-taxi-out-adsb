"""Pre-registered shrink confirmation (research_320/PREREG_shrink.md): s = 1.0 against 0.5 on fresh dates. No fitting.

Component A: v56's partial-track correction (rebuilt real models, byte-identical to Codex's; controls = v56's own
permutation-control models with v56's own permutation, multiairport_screen.inputs). Component B: v57's stand correction
(research_320 stand models and controls). Rows from Codex's frozen extraction of the fresh dates. Frozen models only.
Run from the repository root:  python3 -u -B research_320/shrink_experiment.py
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import stand_experiment as S  # noqa: E402
from research_320.episode_experiment import detector_ids  # noqa: E402

MS, R = S.MS, S.R
DATES = ("2025-01-05", "2025-01-19", "2025-01-23", "2025-01-07", "2025-07-15", "2025-07-29", "2025-07-21", "2025-07-17")
REBUILD = HERE / ".cache/research_320/v56_rebuild"
A_CELLS = ("EDDF/01", "EGLL/01", "EHAM/07", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07")
B_CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07")
OUT = HERE / "research_320/results/shrink_confirm.json"


def corrections(component, frame, arm):
    if component == "A":
        x = MS.inputs(frame, "real" if arm == "real" else f"shuffle_{arm}")
        return [R.joblib.load(REBUILD / f"{'real' if arm == 'real' else 'shuffle_' + str(arm)}_{s}.pkl").predict(x) for s in S.SEEDS]
    x = S.inputs(frame, "real" if arm == "real" else f"shuffle_{arm}")
    return [R.joblib.load(S.CACHE / "models" / f"{'real' if arm == 'real' else 'shuffle_' + str(arm)}_{s}.pkl").predict(x) for s in S.SEEDS]


def gain(frame, corr):
    y, base = frame.y.to_numpy(float), frame.comparator.to_numpy(float)
    return (np.maximum(base + .5 * corr, 300.) - y) ** 2 - (np.maximum(base + 1. * corr, 300.) - y) ** 2


def describe(frame, g, cells):
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-100:]] = False
    cell = (frame.airport + "/" + frame.date.str[5:7]).to_numpy()
    return {"rows": int(len(g)), "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
            "airport_day_ci": list(map(float, R.block_bootstrap(g, (frame.airport + "/" + frame.date).to_numpy()))),
            "by_month": {m: float(g[frame.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")},
            "by_cell": {c: float(g[cell == c].mean()) if (cell == c).any() else None for c in cells}}


def component(name, frame, cells, controls):
    frame = frame.reset_index(drop=True)
    real = corrections(name, frame, "real")
    res = {"real": describe(frame, gain(frame, np.mean(real, axis=0)), cells),
           "seeds": {s: float(gain(frame, c).mean()) for s, c in zip(S.SEEDS, real)},
           "controls": {c: float(gain(frame, np.mean(corrections(name, frame, c), axis=0)).mean()) for c in controls}}
    r = res["real"]
    res["rules"] = {"pooled_positive": r["gain_mse"] > 0,
                    "beats_all_controls": all(r["gain_mse"] > v for v in res["controls"].values()),
                    "both_months_positive": min(r["by_month"].values()) > 0,
                    "best100_removed_positive": r["without_best100"] > 0,
                    "airport_day_ci_positive": r["airport_day_ci"][0] > 0,
                    "every_seed_positive": all(v > 0 for v in res["seeds"].values()),
                    "cells_positive": sum((v or 0) > 0 for v in r["by_cell"].values()) >= (5 if name == "A" else 3)}
    res["verdict"] = "pass" if all(res["rules"].values()) else "fail"
    return res


def main():
    frame = S.extract(DATES, "shrink_confirm")
    frame["cell"] = frame.airport + "/" + frame.date.str[5:7]
    det = detector_ids(DATES)
    a = frame[frame.cell.isin(A_CELLS) & frame.category.eq("partial_ground") & ~frame.MVT_ID_mvt.isin(det)]
    b = frame[frame.cell.isin(B_CELLS) & frame.category.eq("observed_at_assigned_stand")]
    report = {"prereg": "research_320/PREREG_shrink.md", "dates": list(DATES), "detector_rows_excluded": int(frame.MVT_ID_mvt.isin(det).sum()),
              "A_v56_partial": component("A", a, A_CELLS, (3101, 3102, 3103, 3104, 3105)),
              "B_v57_stand": component("B", b, B_CELLS, S.CONTROLS)}
    cand = json.loads(subprocess.run(["git", "show", "9884150:research_310/results/partial_candidate_v55base.json"], cwd=HERE,
                                     capture_output=True, text=True, check=True).stdout)["cells"]
    rows_a = {c.replace("/", "/"): cand[c]["scored_eligible"] for c in A_CELLS}
    rows_b = json.loads((HERE / "research_320/results/stand_candidate.json").read_text())["eligible_rows_by_cell"]
    mse = 0.0
    for key, rows in (("A_v56_partial", rows_a), ("B_v57_stand", rows_b)):
        if report[key]["verdict"] == "pass":
            mse += sum(rows.get(c, 0) * (report[key]["real"]["by_cell"].get(c) or 0.0) for c in rows) / S.N
    v57 = 260.9292
    report["planning"] = {"mse_per_scored_row": mse, "full_s": float(v57 - np.sqrt(v57 ** 2 - mse)) if mse > 0 else 0.0,
                          "half_s": float(v57 - np.sqrt(v57 ** 2 - mse / 2)) if mse > 0 else 0.0}
    report["minimum_meaningful_0_3_s"] = report["planning"]["full_s"] >= 0.3
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    for key in ("A_v56_partial", "B_v57_stand"):
        c = report[key]
        print(key, c["verdict"], json.dumps(c["rules"]))
        print("   real", json.dumps({k: c["real"][k] for k in ("rows", "gain_mse", "without_best100", "airport_day_ci", "by_month", "by_cell")}))
        print("   seeds", c["seeds"], "controls", {k: round(v, 1) for k, v in c["controls"].items()})
    print("planning", report["planning"], "minimum met:", report["minimum_meaningful_0_3_s"])


if __name__ == "__main__":
    main()

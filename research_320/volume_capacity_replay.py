"""Independent replay of the volume/capacity experiment (research_320/volume_capacity.py), in a fresh process.

Reloads every saved model and the cached input tables, recomputes each arm's per-seed corrections on the saved B1
(and, if it exists, B2) rows, requires them to equal the saved aligned predictions exactly, and recomputes the
headline numbers with its own few lines (no helper of the experiment) against research_320/results/volume_capacity.json.
No fit and no label beyond those in the saved files.
Run from the repository root:  python3 -u -B research_320/volume_capacity_replay.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
R = MS.R
CACHE = HERE / ".cache/research_320/volume"
TABLES = HERE / ".cache/research_320/stand"
REPORT = HERE / "research_320/results/volume_capacity.json"
OUT = HERE / "research_320/results/volume_capacity_replay.json"
SEEDS, ID = (177, 277, 377), "MVT_ID_mvt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    return bool(abs(a - b) <= 1e-9 * max(1.0, abs(b)))


def replay(saved, table, arms, recorded):
    rows = saved[[ID]].merge(table, on=ID, how="left", validate="one_to_one")
    out = {"rows": int(len(rows)), "labels_equal": bool(np.array_equal(rows.y.to_numpy(float), saved.y.to_numpy(float)))}
    x = MS.inputs(rows, "real")
    exact = {}
    for arm in arms:
        for s in SEEDS:
            exact[f"{arm}_{s}"] = bool(np.array_equal(R.joblib.load(CACHE / "models" / f"{arm}_real_{s}.pkl").predict(x),
                                                      saved[f"{arm}_corr{s}"].to_numpy(float)))
    out["corrections_exact"] = all(exact.values())
    out["not_exact"] = [k for k, v in exact.items() if not v]
    y, comp = saved.y.to_numpy(float), rows.comparator.to_numpy(float)
    err = {a: (np.maximum(comp + np.mean([saved[f"{a}_corr{s}"].to_numpy(float) for s in SEEDS], axis=0), 300.) - y) ** 2 for a in arms}
    checks = {}
    for arm in arms:
        if arm == "arm1" or arm not in recorded:
            continue
        gain = err["arm1"] - err[arm]
        top = np.argsort(gain, kind="stable")[-100:]
        rest = np.delete(gain, top)
        jan = rows.date.str[5:7].eq("01").to_numpy()
        seeds = {s: float(((np.maximum(comp + saved[f"arm1_corr{s}"].to_numpy(float), 300.) - y) ** 2
                           - (np.maximum(comp + saved[f"{arm}_corr{s}"].to_numpy(float), 300.) - y) ** 2).mean()) for s in SEEDS}
        r = recorded[arm]
        checks[arm] = {"gain_mse": close(gain.mean(), r["gain_mse"]), "without_best100": close(rest.mean(), r["without_best100"]),
                       "january": close(gain[jan].mean(), r["by_month"]["01"]), "july": close(gain[~jan].mean(), r["by_month"]["07"]),
                       "seeds": all(close(seeds[s], r["seeds"][str(s)] if str(s) in r["seeds"] else r["seeds"][s]) for s in SEEDS)}
    out["headline_numbers_equal"] = checks
    out["all_equal"] = out["labels_equal"] and out["corrections_exact"] and all(all(v.values()) for v in checks.values())
    return out


def main():
    report = json.loads(REPORT.read_text())
    result = {"model_hashes_equal": all(sha(CACHE / "models" / k) == v for k, v in report["model_hashes"].items()),
              "prediction_files_equal": all(sha(CACHE / k) == v for k, v in report["predictions_sha256"].items())}
    b1 = pd.read_parquet(CACHE / "b1_predictions.parquet")
    dev_table = pd.read_parquet(TABLES / "development.parquet")
    arms = ["arm1", "arm2", "arm3", "arm4", "e_fixed"]
    result["b1"] = replay(b1, dev_table, arms, report["development"])
    result["b1"]["deployed_equals_arm1"] = all(np.array_equal(b1[f"deployed_corr{s}"], b1[f"arm1_corr{s}"]) for s in SEEDS)
    if (CACHE / "b2_predictions.parquet").exists():
        b2 = pd.read_parquet(CACHE / "b2_predictions.parquet")
        present = [a for a in arms if f"{a}_corr177" in b2]
        result["b2"] = replay(b2, pd.read_parquet(TABLES / "confirm.parquet"), present, report.get("replication", {}))
    result["verdict"] = "replayed" if (result["model_hashes_equal"] and result["prediction_files_equal"]
                                       and result["b1"]["all_equal"] and result["b1"]["deployed_equals_arm1"]
                                       and result.get("b2", {"all_equal": True})["all_equal"]) else "mismatch"
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

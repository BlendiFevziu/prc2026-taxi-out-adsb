"""Checks after the volume/capacity experiment passed (research_320/PREREG_volume_capacity.md); descriptive, no fit.

1. Label-free 2026 support: the frozen arm 3 and arm 4 models on v56's 66,292 scored overlay rows (Codex's 2026
   context table), against the same models on the B1 and B2 rows. First the deployed models must reproduce Codex's
   2026 corrections exactly, which proves the 2026 inputs are his. Compared: correction quantiles, the change against
   the deployed correction, its correlation, and how often the 300 s floor binds.
2. Not pre-registered, reported as such: arm 4 against arm 3, paired, on B1 and B2 (95 % airport-day interval).
Run from the repository root:  python3 -u -B research_320/volume_capacity_support.py
"""
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
REBUILD = HERE / ".cache/research_320/v56_rebuild"
TABLES = HERE / ".cache/research_320/stand"
OUT = HERE / "research_320/results/volume_capacity_support.json"
SEEDS, ID = (177, 277, 377), "MVT_ID_mvt"


def corr(x, prefix, folder=CACHE / "models"):
    return np.mean([R.joblib.load(folder / f"{prefix}{s}.pkl").predict(x) for s in SEEDS], axis=0)


def profile(comp, dep, new):
    q = lambda v: [round(float(t), 1) for t in np.quantile(v, [.01, .1, .5, .9, .99])]
    return {"rows": int(len(comp)), "deployed_p1_p10_p50_p90_p99": q(dep), "new_p1_p10_p50_p90_p99": q(new),
            "rms_change_s": float(np.sqrt(np.mean((new - dep) ** 2))), "correlation_with_deployed": float(np.corrcoef(dep, new)[0, 1]),
            "floor_binds_deployed": float(np.mean(comp + dep < 300)), "floor_binds_new": float(np.mean(comp + new < 300))}


def main():
    upd = pd.read_parquet(HERE / ".cache/codex_v56/partial_release_v55/updates.parquet")
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    rows26 = upd[[ID, "residual"]].merge(ctx, on=ID, how="left", validate="one_to_one")
    x26 = MS.inputs(rows26, "real")
    dep26 = corr(x26, "real_", REBUILD)
    out = {"rows_2026": int(len(rows26)), "deployed_reproduce_codex_2026_corrections": bool(np.array_equal(dep26, rows26.residual.to_numpy(float)))}
    if not out["deployed_reproduce_codex_2026_corrections"]:
        raise SystemExit("the 2026 inputs are not Codex's")
    b1 = pd.read_parquet(CACHE / "b1_predictions.parquet").merge(pd.read_parquet(TABLES / "development.parquet")[[ID, "comparator"]], on=ID)
    b2 = pd.read_parquet(CACHE / "b2_predictions.parquet").merge(pd.read_parquet(TABLES / "confirm.parquet")[[ID, "comparator"]], on=ID)
    cell26 = (rows26.airport + "/" + rows26.date.str[5:7]).to_numpy()
    for arm in ("arm3", "arm4"):
        new26 = corr(x26, f"{arm}_real_")
        comp26 = rows26.comparator.to_numpy(float)
        res = {"2026": profile(comp26, dep26, new26)}
        for tag, b in (("B1", b1), ("B2", b2)):
            dep = np.mean([b[f"arm1_corr{s}"] for s in SEEDS], axis=0)
            new = np.mean([b[f"{arm}_corr{s}"] for s in SEEDS], axis=0)
            res[tag] = profile(b.comparator.to_numpy(float), dep, new)
        res["2026_rms_change_by_cell"] = {c: float(np.sqrt(np.mean((new26 - dep26)[cell26 == c] ** 2))) for c in sorted(set(cell26))}
        out[arm] = res
    out["not_preregistered_arm4_vs_arm3"] = {}
    for tag, b in (("B1", b1), ("B2", b2)):
        y, comp = b.y.to_numpy(float), b.comparator.to_numpy(float)
        err = lambda a, seeds=SEEDS: (np.maximum(comp + np.mean([b[f"{a}_corr{s}"] for s in seeds], axis=0), 300.) - y) ** 2
        gain = err("arm3") - err("arm4")
        out["not_preregistered_arm4_vs_arm3"][tag] = {
            "gain_mse": float(gain.mean()), "airport_day_ci_95": list(map(float, R.block_bootstrap(gain, (b.airport + "/" + b.date).to_numpy()))),
            "by_month": {m: float(gain[b.date.str[5:7].eq(m).to_numpy()].mean()) for m in ("01", "07")},
            "seeds": {s: float((err("arm3", (s,)) - err("arm4", (s,))).mean()) for s in SEEDS},
            "cells_positive": f"{sum(gain[(b.airport + '/' + b.date.str[5:7]).eq(c).to_numpy()].mean() > 0 for c in sorted(set(b.airport + '/' + b.date.str[5:7])))} of 7"}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

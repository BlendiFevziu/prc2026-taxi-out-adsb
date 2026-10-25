"""Control-referenced reading of research_280/nm_proxy_screen.py. Added after the column-count artefact was found.

Why. Every arm with 94 columns, the permutation controls included, shows January negative and July positive
against the 91-column baseline (January -129.8 MSE, July +73.5, 10 of 10 arms at the time it was noticed): with
colsample_bytree 0.8 and a fixed seed, changing the column count changes which columns each tree may use. The
baseline-referenced rules R1-R3 absorb that; R4 does not. See the dated addendum in NM_PROXY_INVENTORY.md.

This script is INTERPRETATION, not a new gate; the pre-registered verdict (all six rules) stands as written.
It compares each candidate with its own five controls, which share its column count and seed, row by row:

    d_row = mean_k (control_k - y)^2  -  (candidate - y)^2          positive: the candidate beats its controls

and reports the mean of d, January and July separately, d without its 100 most favourable rows, d per airport, a
2,000-draw airport-day cluster bootstrap of d, and the candidate's z-score within the control distribution.

Reads only the checkpoints in .cache/nm_proxy_screen/ and the held-out labels; fits nothing.
Run from the repository root:  python3 -u -B research_280/nm_proxy_control_referenced.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT / ".cache/nm_proxy_screen"
OUTPUT = PROJECT / "research_280/results/nm_proxy_control_referenced.json"
FAMILIES = ("H1", "H2", "H3")
SEEDS = (1, 2, 3, 4, 5)
DRAWS = 2000


def main():
    v = pd.read_parquet(PROJECT / ".cache/clock_queue_release_20260912/data/months_1_7.parquet",
                        columns=["y", "hat_aobt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt", "ADEP_mvt"])
    v = v[v.hat_aobt.eq(1) & v.y.notna()].reset_index(drop=True)
    y = v.y.to_numpy(float)
    ids_sha = __import__("hashlib").sha256(np.ascontiguousarray(v.MVT_ID_mvt.to_numpy()).tobytes()).hexdigest()
    base_meta = json.loads((CACHE / "pred_baseline.json").read_text())
    if base_meta["valid_ids_sha256"] != ids_sha:
        raise SystemExit("row alignment failed: held-out IDs differ from the screen's checkpoints")
    stamp = pd.to_datetime(v.MVT_TIME_UTC_mvt, utc=True)
    month = stamp.dt.month.to_numpy()
    airport = v.ADEP_mvt.astype(str).to_numpy()
    cluster = pd.DataFrame({"a": airport, "d": stamp.dt.normalize().to_numpy()}).groupby(
        ["a", "d"], sort=False).ngroup().to_numpy()
    base = np.load(CACHE / "pred_baseline.npy")
    base_se = (base - y) ** 2
    rng = np.random.default_rng(0)
    draws = rng.integers(0, cluster.max() + 1, size=(DRAWS, cluster.max() + 1))

    out = {"note": "interpretation added after the column-count artefact; not a gate, the pre-registered verdict "
                   "stands", "rows": int(len(y))}
    for family in FAMILIES:
        cand = np.load(CACHE / f"pred_{family}_candidate.npy")
        ctrls = np.stack([np.load(CACHE / f"pred_{family}_perm{s}.npy") for s in SEEDS])
        cand_se = (cand - y) ** 2
        ctrl_se = (ctrls - y) ** 2
        d = ctrl_se.mean(axis=0) - cand_se
        keep = np.ones(len(d), bool)
        keep[np.argsort(d)[::-1][:100]] = False
        c_sum = np.bincount(cluster, weights=d)
        c_n = np.bincount(cluster).astype(float)
        boot = c_sum[draws].sum(axis=1) / c_n[draws].sum(axis=1)
        cand_gain = float((base_se - cand_se).mean())
        ctrl_gain = (base_se[None, :] - ctrl_se).mean(axis=1)
        # the same comparison with the column-count artefact removed month by month
        month_art = {m: float((base_se[None, month == m] - ctrl_se[:, month == m]).mean()) for m in (1, 7)}
        out[family] = {
            "candidate_gain_vs_baseline": cand_gain,
            "control_gains_vs_baseline": [float(g) for g in ctrl_gain],
            "control_mean": float(ctrl_gain.mean()), "control_sd": float(ctrl_gain.std(ddof=1)),
            "z_candidate_within_controls": float((cand_gain - ctrl_gain.mean()) / ctrl_gain.std(ddof=1)),
            "d_mean": float(d.mean()),
            "d_january": float(d[month == 1].mean()), "d_july": float(d[month == 7].mean()),
            "d_without_top100": float(d[keep].mean()),
            "d_bootstrap_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "column_count_artefact_by_month": month_art,
            "d_by_airport": {a: float(d[airport == a].mean()) for a in sorted(np.unique(airport))},
            "airports_positive": int(sum(d[airport == a].mean() > 0 for a in np.unique(airport))),
        }
        e = out[family]
        print(f"{family}: vs baseline {cand_gain:+7.1f} | controls mean {e['control_mean']:+6.1f} sd {e['control_sd']:5.1f}"
              f" | z {e['z_candidate_within_controls']:+5.2f} | d {e['d_mean']:+6.1f}  Jan {e['d_january']:+6.1f}"
              f"  Jul {e['d_july']:+6.1f}  top100-removed {e['d_without_top100']:+7.1f}"
              f"  boot [{e['d_bootstrap_95'][0]:+.1f}, {e['d_bootstrap_95'][1]:+.1f}]"
              f"  airports + {e['airports_positive']}/10")
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()

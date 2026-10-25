"""Confirmation of research_360_integration/PREREG_eddf_july.md on 6, 8 and 10 July 2025. No fit, no upload.

Extraction with the frozen v56 code (stand_experiment.extract, under the shared lock); base checked bit for bit
against the recomposed held-out comparator; the frozen C1 model applied to EDDF July partial rows; the five
registered rules evaluated. Labels of these days are read here for the first time.
Run from the repository root:  python3 -u -B research_360_integration/eddf_july_confirm.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_320 import stand_experiment as S  # noqa: E402  (frozen v56 extraction)
from research_300.heavy_lock import heavy_work  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402
from error_by_track_quality import stratum  # noqa: E402

ID = "MVT_ID_mvt"
DAYS = ("2025-07-06", "2025-07-08", "2025-07-10")
TRAIN_JULY = ("2025-07-02", "2025-07-03", "2025-07-04", "2025-07-18", "2025-07-21", "2025-07-22", "2025-07-23", "2025-07-29")
SHARE_DEGRADED_2026, ROWS_2026, V62, N = 0.7298, 11993, 258.5995, 344841
OUT = HERE / "research_360_integration/results/eddf_july_confirm.json"


def main():
    if min(abs((pd.Timestamp(a) - pd.Timestamp(b)).days) for a in DAYS for b in TRAIN_JULY) < 2:
        raise SystemExit("embargo violated")
    missing = [d for d in DAYS if not (S.MS.R.MD.ADSB / f"tracks/{d}.parquet").exists()]
    if missing:
        raise SystemExit(f"tracks missing: {missing}")
    with heavy_work("research_360: extraction of 6/8/10 July 2025", poll_s=10.0):
        ext = S.extract(DAYS, "eddf_july_confirm")
    comps = H.components_2025()
    base = pd.Series(L.compose_base(comps), index=comps[ID].to_numpy())
    checks = {"comparator_bit_identical": bool(np.array_equal(base.loc[ext[ID]].to_numpy(float), ext.comparator.to_numpy(float)))}
    f = ext[ext.airport.eq("EDDF") & ext.date.str[5:7].eq("07") & ext.category.eq("partial_ground")].reset_index(drop=True)
    checks["rows"] = int(len(f))
    if not checks["comparator_bit_identical"] or not len(f):
        OUT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2) + "\n")
        raise SystemExit(f"precondition failed: {checks}")
    y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
    p = L.apply(f, L.correction("partial", f), 1.0)
    gain = (b - y) ** 2 - (p - y) ** 2
    st = pd.Series(stratum(f).to_numpy())
    degraded = (st.str.startswith("sparse") | st.str.endswith("/late")).to_numpy()
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    by_day = {d: float(gain[f.date.eq(d).to_numpy()].mean()) for d in DAYS}
    g_deg, g_well = float(gain[degraded].mean()), float(gain[~degraded].mean())
    reweighted = SHARE_DEGRADED_2026 * g_deg + (1 - SHARE_DEGRADED_2026) * g_well
    mse = ROWS_2026 * reweighted / N
    full_s = float(V62 - np.sqrt(V62 ** 2 - mse)) if mse > 0 else float(mse / (2 * V62))
    rules = {"1_pooled_positive": float(gain.mean()) > 0, "2_best100_removed_positive": float(gain[keep].mean()) > 0,
             "3_each_day_positive": all(v > 0 for v in by_day.values()),
             "4_degraded_and_well_observed_positive": g_deg > 0 and g_well > 0,
             "5_reweighted_positive_and_at_least_0_1_s": reweighted > 0 and full_s >= 0.1}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):                                   # descriptive: flights resampled within each day
        idx = np.concatenate([rng.choice(np.flatnonzero(f.date.eq(d).to_numpy()), int(f.date.eq(d).sum())) for d in DAYS])
        boot.append(gain[idx].mean())
    report = {"prereg": "research_360_integration/PREREG_eddf_july.md", "days": list(DAYS), "checks": checks,
              "rows_by_day": {d: int(f.date.eq(d).sum()) for d in DAYS}, "base_rmse": float(np.sqrt(((b - y) ** 2).mean())),
              "c1_rmse": float(np.sqrt(((p - y) ** 2).mean())), "gain_mse": float(gain.mean()), "without_best100": float(gain[keep].mean()),
              "by_day": by_day, "degraded": {"rows": int(degraded.sum()), "share_2025_days": float(degraded.mean()), "gain_mse": g_deg},
              "well_observed": {"rows": int((~degraded).sum()), "gain_mse": g_well},
              "by_stratum": {s: {"rows": int(st.eq(s).sum()), "gain_mse": float(gain[st.eq(s).to_numpy()].mean())} for s in sorted(st.unique())},
              "reweighted_to_2026_mix": reweighted, "planning": {"mse_per_scored_row": mse, "full_s": full_s,
              "half_s": float(V62 - np.sqrt(V62 ** 2 - mse / 2)) if mse > 0 else float(mse / (4 * V62))},
              "flight_bootstrap_within_day_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
              "rules": rules}
    report["verdict"] = "pass" if all(rules.values()) else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

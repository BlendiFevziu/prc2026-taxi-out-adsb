"""Rome stage 2: the frozen C1 model at LIRF, unchanged, on 16 fresh July 2025 days (research_380_rome/PREREG_stage2.md).
No fit, no upload. Rows and track summaries from stage 1's frozen extraction; comparator, ersatz_s, schedule and gate
from the held-out 2025 components (harness.components_2025, layers.compose_base); the hour phase as v56 computes it.
Run from the repository root:  python3 -u -B research_380_rome/stage2_c1_transfer.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
sys.path.insert(0, str(HERE / "research_380_rome"))
import stage1_coverage as S1  # noqa: E402  (frozen extraction, v56 modules loaded there)
import harness as H  # noqa: E402
import layers as L  # noqa: E402
from error_by_track_quality import stratum  # noqa: E402

MS, R = S1.MS, S1.R
ID, SEEDS = "MVT_ID_mvt", (177, 277, 377)
FRESH = ("2025-07-03", "2025-07-06", "2025-07-07", "2025-07-08", "2025-07-10", "2025-07-12", "2025-07-13", "2025-07-14",
         "2025-07-15", "2025-07-17", "2025-07-21", "2025-07-22", "2025-07-25", "2025-07-26", "2025-07-29", "2025-07-31")
CODEX = ("2025-07-02", "2025-07-04", "2025-07-18", "2025-07-23", "2025-07-09", "2025-07-11", "2025-07-16", "2025-07-27")
DEGRADED_2026, ROWS_2026, N, REFERENCE = 0.6914, 6767, 344841, 258.24
MODELS = HERE / ".cache/research_320/volume/models"
OUT = HERE / "research_380_rome/results/stage2_c1_transfer.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frame(days):
    rows = S1.extract_2025(S1.departures_2025(list(days)), list(days))
    rows = rows[rows.airport.eq("LIRF") & rows.category.eq("partial_ground")]
    comps = H.components_2025()
    comps["comparator"] = L.compose_base(comps)
    f = rows.merge(comps[[ID, "y", "comparator", "ersatz_s", "schedule", "gate", "MVT_TIME_UTC_mvt"]], on=ID, how="left",
                   validate="one_to_one")
    if f.y.isna().any():
        raise SystemExit(f"{int(f.y.isna().sum())} LIRF partial rows lack a held-out component row")
    epoch = (pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True) - R.MD.EPOCH0).dt.total_seconds()
    phase = 2 * np.pi * (epoch % 86400) / 86400
    f["hour_sin"], f["hour_cos"] = np.sin(phase), np.cos(phase)
    for ap in ("EDDF", "EGLL", "EHAM", "LEBL", "LSZH"):             # C1 never saw LIRF: all five flags zero
        f[f"airport_{ap}"] = 0.0
    f["is_july"] = 1.0
    return f.reset_index(drop=True)


def evaluate(f):
    x = MS.inputs(f, "real")
    c1 = np.mean([R.joblib.load(MODELS / f"c1_real_{s}.pkl").predict(x) for s in SEEDS], axis=0)
    rev = np.mean([R.joblib.load(MODELS / f"c1_real_{s}.pkl").predict(x.iloc[::-1]) for s in SEEDS], axis=0)[::-1]
    y, b = f.y.to_numpy(float), f.comparator.to_numpy(float)
    p = np.maximum(b + c1, 300.0)
    return (b - y) ** 2 - (p - y) ** 2, bool(np.array_equal(c1, rev)), p


def main():
    cap = json.loads((HERE / "research_320/results/capacity2.json").read_text())
    checks = {"c1_models_as_tested": all(sha(MODELS / f"c1_real_{s}.pkl") == cap["model_hashes"][f"c1_real_{s}.pkl"] for s in SEEDS),
              "airport_flags_as_v56": list(MS.COLS[-6:]) == [f"airport_{a}" for a in ("EDDF", "EGLL", "EHAM", "LEBL", "LSZH")] + ["is_july"]}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    f = frame(FRESH)
    gain, checks["c1_reverse_replay"], pred = evaluate(f)
    st = stratum(f).to_numpy()
    degraded = pd.Series(st).str.startswith("sparse").to_numpy() | pd.Series(st).str.endswith("/late").to_numpy()
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    first, second = f.date.isin(FRESH[:8]).to_numpy(), f.date.isin(FRESH[8:]).to_numpy()
    g_deg, g_well = float(gain[degraded].mean()), float(gain[~degraded].mean())
    reweighted = DEGRADED_2026 * g_deg + (1 - DEGRADED_2026) * g_well
    mse = ROWS_2026 * reweighted / N
    full_s = float(REFERENCE - np.sqrt(REFERENCE ** 2 - mse)) if mse > 0 else float(mse / (2 * REFERENCE))
    rules = {"1_pooled_positive": float(gain.mean()) > 0, "2_best100_removed_positive": float(gain[keep].mean()) > 0,
             "3_each_half_positive": float(gain[first].mean()) > 0 and float(gain[second].mean()) > 0,
             "4_degraded_and_well_observed_positive": g_deg > 0 and g_well > 0,
             "5_reweighted_positive_and_at_least_0_1_s": reweighted > 0 and full_s >= 0.1}
    rng = np.random.default_rng(0)
    boot = [gain[np.concatenate([rng.choice(np.flatnonzero(f.date.eq(d).to_numpy()), int(f.date.eq(d).sum())) for d in FRESH if f.date.eq(d).any()])].mean()
            for _ in range(2000)]
    codex = frame(CODEX)
    cg, _, _ = evaluate(codex)
    ck = np.ones(len(cg), bool)
    ck[np.argsort(cg, kind="stable")[-100:]] = False
    report = {"prereg": "research_380_rome/PREREG_stage2.md", "checks": checks, "rows": int(len(f)),
              "base_rmse": float(np.sqrt(np.mean((f.comparator - f.y) ** 2))), "c1_rmse": float(np.sqrt(np.mean((pred - f.y) ** 2))),
              "gain_mse": float(gain.mean()), "without_best100": float(gain[keep].mean()),
              "share_of_gain_from_best100": float(np.sort(gain)[-100:].sum() / gain.sum()) if gain.sum() > 0 else None,
              "halves": {"first": float(gain[first].mean()), "second": float(gain[second].mean())},
              "by_day": {d: float(gain[f.date.eq(d).to_numpy()].mean()) for d in FRESH if f.date.eq(d).any()},
              "degraded": {"rows": int(degraded.sum()), "gain_mse": g_deg}, "well_observed": {"rows": int((~degraded).sum()), "gain_mse": g_well},
              "by_stratum": {s: {"rows": int((st == s).sum()), "gain_mse": float(gain[st == s].mean())} for s in sorted(set(st))},
              "reweighted_to_jan_2026": reweighted, "planning": {"mse_per_scored_row": mse, "full_s": full_s},
              "flight_bootstrap_within_day_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
              "codex_days_descriptive": {"rows": int(len(codex)), "gain_mse": float(cg.mean()), "without_best100": float(cg[ck].mean())},
              "rules": rules}
    report["verdict"] = "pass" if all(rules.values()) and checks["c1_reverse_replay"] else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

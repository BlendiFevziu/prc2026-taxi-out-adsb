"""Rebuild for the final-phase file (January, February, June, July 2026): four-month ensemble base + ADS-B corrector.

Validated design (research_730_adsb_learner): pilot 6 (corrector over this ensemble's held-out predictions, +15.8 s on January/July 2025)
and pilot 7 (corrector trained on one month pair transfers to another, +15.2 s on four 2025 months).

  stage 1  `.cache/research_800/base_final_predictions.parquet`: the frozen four-month base (v45 all-year ensemble + sequence nets,
           anchored fallback, copy gate, Rome rules; main lane equal to the scored v47 recipe), 670,790 rows, lane main/fallback.
  stage 2  the pilot-5 corrector (residual LightGBM, learning rate 0.03, 63 leaves, min_child_samples 100, subsample/colsample 0.8,
           600 rounds; features p1, ersatz_s, stunde, ADEP_mvt + 12 ADS-B + 6 stand columns), trained on all 339,046 January/July
           2025 main-lane rows with p1 = the same recipe's held-out predictions (`.cache/sequence_release_v47/validate`),
           seeds 177 and 277, predictions averaged. Applied to every main-lane row of the four 2026 months; fallback rows keep the base.

No leaderboard, no upload, no probe-derived value. Run from the repository root:
  python3 -u -B research_730_adsb_learner/build_2026_final.py table     2026 ADS-B and stand columns, month by month (label-free)
  python3 -u -B research_730_adsb_learner/build_2026_final.py build     train the corrector, apply it, check and write the file
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightgbm as lgb
import numpy as np
import pandas as pd
import submission as SUB
from research_730_adsb_learner import features_adsb as F
from research_730_adsb_learner import pilot5_screen as S
from research_300.heavy_lock import heavy_work

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/research_730"
OUT = CACHE / "rebuild_2026"
TRACKS = ROOT.parent / "adsb-lol/tracks"
BASE = ROOT / ".cache/research_800/base_final_predictions.parquet"
FEATURES = S.STAGE2_BASE + S.AD18
SEEDS = (177, 277)
T = "TAXITIME_SEC_mvt"


def log(*a):
    print(*a, flush=True)


def table():
    target = OUT / "table_2026_final.parquet"
    if target.exists():
        log("table cached"); return
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASE)
    raw = pd.read_parquet(ROOT / "data/final_ranking.parquet",
                          columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "CALLSIGN_flt", "STAND_mvt"])
    dep = raw[raw.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt")
    main = base[base.lane.eq("main")][["MVT_ID_mvt", T]].rename(columns={T: "p1"}).merge(dep, on="MVT_ID_mvt", how="left")
    mvt = pd.to_datetime(main.MVT_TIME_UTC_mvt, utc=True)
    main["ersatz_s"] = (mvt - pd.to_datetime(main.AOBT_3_flt, utc=True)).dt.total_seconds()
    main["stunde"] = mvt.dt.hour
    main["month"] = mvt.dt.strftime("%Y-%m")
    assert main.ersatz_s.notna().all(), "a main-lane row without AOBT_3"
    parts = []
    for month, rows in main.groupby("month"):
        first = pd.Timestamp(month + "-01")
        days = [first - pd.Timedelta(days=1)] + list(pd.date_range(first, first + pd.offsets.MonthEnd(0)))
        files = [TRACKS / f"{d:%Y-%m-%d}.parquet" for d in days if (TRACKS / f"{d:%Y-%m-%d}.parquet").exists()]
        tracks = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        out = F.compute(rows, tracks, stands=True)
        assert np.array_equal(out.index.to_numpy(), rows.MVT_ID_mvt.to_numpy())
        rows = rows.copy()
        for c in S.AD18:
            rows[c] = out[c].to_numpy()
        log(f"{month}: {len(rows):,} main-lane rows, {len(files)} track files, matched {rows.ad_matched.mean():.3f}, stand ok {rows.sd_status.eq(1).mean():.3f}")
        parts.append(rows)
        del tracks
    pd.concat(parts, ignore_index=True).to_parquet(target, index=False)


def build():
    t25 = pd.read_parquet(CACHE / "table_1_7_stand.parquet")
    e = pd.read_parquet(ROOT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet", columns=["MVT_ID_mvt", "y", "baseline"])
    assert np.array_equal(e.MVT_ID_mvt.to_numpy(), t25.MVT_ID_mvt.to_numpy()) and np.allclose(e.y.to_numpy(), t25.y.to_numpy())
    t25["p1"] = e.baseline.to_numpy(float)
    t26 = pd.read_parquet(OUT / "table_2026_final.parquet")
    cats = pd.unique(t25.ADEP_mvt.astype(str))
    t25["ADEP_mvt"] = pd.Categorical(t25.ADEP_mvt.astype(str), categories=cats)
    apply = t26.copy()
    apply["ADEP_mvt"] = pd.Categorical(apply.ADEP_mvt.astype(str), categories=cats)
    corrections = []
    for seed in SEEDS:
        with heavy_work(f"research_730 rebuild 2026 corrector seed {seed}", log=log):
            model = lgb.LGBMRegressor(objective="l2", n_estimators=S.ROUNDS, learning_rate=.03, num_leaves=63, min_child_samples=100,
                                      subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=8, random_state=seed)
            model.fit(t25[FEATURES], (t25.y - t25.p1).to_numpy())
            corrections.append(model.predict(apply[FEATURES]))
            model.booster_.save_model(str(OUT / f"corrector_s{seed}.txt"))
    corr = np.mean(corrections, axis=0)
    new_main = pd.Series(np.maximum(t26.p1.to_numpy() + corr, 300.0), index=t26.MVT_ID_mvt.to_numpy())
    base = pd.read_parquet(BASE)
    template = pd.read_parquet(ROOT / "data/final_submitting.parquet")
    cand = template[["MVT_ID_mvt"]].merge(base[["MVT_ID_mvt", T, "lane"]], on="MVT_ID_mvt", how="left")
    is_main = cand.lane.eq("main").to_numpy()
    cand.loc[is_main, T] = cand.loc[is_main, "MVT_ID_mvt"].map(new_main).to_numpy()
    cand = cand[["MVT_ID_mvt", T]]
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    checks = {"pruefe_errors": [str(x) for x in errors], "rows": int(len(cand)), "rows_equal_template": bool(cand.MVT_ID_mvt.equals(template.MVT_ID_mvt)),
              "all_finite": bool(np.isfinite(cand[T]).all()), "main_rows_corrected": int(is_main.sum()),
              "fallback_rows_equal_base": bool(np.allclose(cand[T].to_numpy()[~is_main], base.set_index("MVT_ID_mvt").loc[cand.MVT_ID_mvt[~is_main], T].to_numpy()))}
    path = OUT / "final_rebuild_2026.parquet"
    cand.to_parquet(path, index=False)
    t26["correction"] = corr
    t26["matched"] = t26.ad_matched.eq(1)
    by = t26.groupby(["month", "ADEP_mvt"]).agg(rows=("correction", "size"), matched=("matched", "mean"),
                                                 mean_corr=("correction", "mean"), mean_abs_corr=("correction", lambda c: float(np.abs(c).mean())))
    report = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "checks": checks,
              "correction_s": {"mean": float(corr.mean()), "mean_abs": float(np.abs(corr).mean()),
                               "quantiles_5_25_50_75_95": [float(x) for x in np.quantile(corr, [.05, .25, .5, .75, .95])],
                               "matched_rows_mean_abs": float(np.abs(corr[t26.matched]).mean()), "unmatched_rows_mean_abs": float(np.abs(corr[~t26.matched]).mean())},
              "by_month_airport": {f"{m}|{a}": {k: (round(float(v), 3) if isinstance(v, float) else int(v)) for k, v in r.items()} for (m, a), r in by.iterrows()},
              "validated_by": ["PREREG_stacker_ensemble.md (pilot 6)", "PREREG_transfer.md (pilot 7)"], "uploaded": False}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps({k: v for k, v in report.items() if k != "by_month_airport"}, indent=1))


if __name__ == "__main__":
    {"table": table, "build": build}[sys.argv[1]]()

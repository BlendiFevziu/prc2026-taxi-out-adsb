"""Validation of the FROZEN stand detector on new labelled dates: Munich, 3, 10, 17 and 20 December 2025.

Rules, comparator and decision are fixed in research_290/PREREG_stand_detector.md, committed (338885a) before this
script first read any validation label. Nothing here is tuned: stand_detector.detect_onset runs with its frozen
constants and the midpoint estimate. The comparator is research_290/december_model.py (single LightGBM on source
months 1-11, fixed 1,200 rounds), which is WEAKER than production; any advantage over it is overstated.

  V1 accuracy transfers        MAE against the recorded block <= 60 s and >= 75 % within 60 s
  V2 beats the comparator      detector RMSE < comparator RMSE on identical main-lane rows, on every date
  V3 not concentrated          squared-error gain > 0 after removing the 5 % most favourable rows
  V4 interval                  2,000-draw flight bootstrap 95 % interval of the gain above zero

Row-level outputs go to .cache/munich_stand_detector/ (never committed); aggregates to research_290/results/.
Run from the repository root:  python3 -u -B research_290/munich_validate.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402

DATES = ("2025-12-03", "2025-12-10", "2025-12-17", "2025-12-20")
AIRPORT = "EDDM"
CACHE = ROOT / ".cache/munich_stand_detector"
OUTPUT = ROOT / "research_290/results/munich_validate.json"
OFFICIAL, N_SCORED, N_MAIN = 265.2839, 344841, 339551


def detections(date, detect=SD.detect_onset):
    tracks, dep, joined = MD.matched_set(date, AIRPORT)
    table = SD.stand_table(AIRPORT)
    rows = []
    for row in joined[joined.unique].itertuples(index=False):
        vertices, stand_status = SD.lookup(table, row.STAND_mvt)
        rec = {"date": date, "MVT_ID_mvt": row.MVT_ID_mvt, "lane": row.lane, "stand_status": stand_status,
               "block_epoch": row.block_epoch, "mvt_epoch": row.mvt_epoch}
        if vertices is None:
            rec["status"] = stand_status
        else:
            ts, lat, lon, gs, _ = MD.points_of(tracks, row)
            out = detect(ts, lat, lon, gs, vertices, row.mvt_epoch, SD.R_PARK_M)
            rec["status"] = out["status"]
            if out["status"] == "ok":
                rec["estimate"] = (out["t_before"] + out["t_after"]) / 2
        rows.append(rec)
    return len(dep), pd.DataFrame(rows)


def main(dates=DATES, detect=SD.detect_onset, output=OUTPUT, rows_file="validation_rows.parquet",
         preregistration="research_290/PREREG_stand_detector.md (commit 338885a)", comparator=None,
         comparator_name=None):
    """Defaults are the v1 run exactly; munich_validate_v2.py passes its own detector, dates and outputs, and
    munich_validate_jan2025.py also its own comparator: a frame with MVT_ID_mvt, pred, ersatz_s and y."""
    frames, departures = [], {}
    for date in dates:
        n, frame = detections(date, detect)
        departures[date] = n
        frames.append(frame)
        print(f"{date}: {n} departures, {len(frame)} matched, {int(frame.status.eq('ok').sum())} estimates", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    ok = rows[rows.status.eq("ok")].copy()
    ok["err"] = ok.estimate - ok.block_epoch
    e = ok.err.to_numpy()
    accuracy = {"n": int(len(e)), "median_s": float(np.median(e)), "mae_s": float(np.abs(e).mean()),
                "rmse_s": float(np.sqrt((e ** 2).mean())), "within_60": int((np.abs(e) <= 60).sum()),
                "within_60_share": float((np.abs(e) <= 60).mean()), "within_120": int((np.abs(e) <= 120).sum()),
                "worst": [float(e.min()), float(e.max())],
                "by_date": {d: {"n": int(len(g)), "median_s": float(np.median(g.err)), "mae_s": float(g.err.abs().mean()),
                                "within_60": int((g.err.abs() <= 60).sum())} for d, g in ok.groupby("date")}}

    model = comparator if comparator is not None else pd.read_parquet(CACHE / "december_model.parquet")
    main_rows = ok[ok.lane.eq("main")].merge(model, on="MVT_ID_mvt", how="inner")
    if len(main_rows) != int(ok.lane.eq("main").sum()):
        raise SystemExit("row alignment failed: covered main-lane rows missing from the comparator predictions")
    y = main_rows.y.to_numpy(float)
    det = main_rows.mvt_epoch.to_numpy() - main_rows.estimate.to_numpy()
    mod = main_rows.pred.to_numpy(float)
    ers = main_rows.ersatz_s.to_numpy(float)
    label_check = float(np.abs((main_rows.mvt_epoch - main_rows.block_epoch).to_numpy() - y).max())
    gain = (mod - y) ** 2 - (det - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:int(np.ceil(0.05 * len(gain)))]] = False
    rng = np.random.default_rng(0)
    boot = gain[rng.integers(0, len(gain), size=(2000, len(gain)))].mean(axis=1)
    rmse = lambda p, t: float(np.sqrt(np.mean((p - t) ** 2)))
    per_date = {}
    for d in dates:
        m = main_rows.date.eq(d).to_numpy()
        per_date[d] = {"rows": int(m.sum()), "detector_rmse": rmse(det[m], y[m]), "comparator_rmse": rmse(mod[m], y[m]),
                       "ersatz_rmse": rmse(ers[m], y[m])}
    comparison = {"rows": int(len(y)), "label_check_max_diff_s": label_check,
                  "detector_rmse": rmse(det, y), "comparator_rmse": rmse(mod, y), "ersatz_rmse": rmse(ers, y),
                  "blend_half_rmse_descriptive": rmse((det + mod) / 2, y),
                  "mse_gain_per_row": float(gain.mean()), "mse_gain_without_top5pct": float(gain[keep].mean()),
                  "bootstrap_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                  "per_date": per_date}

    rank = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"])
    scored_main = rank[rank.PHASE_mvt.eq("DEP") & rank.TAXITIME_SEC_mvt.isna() & rank.AOBT_3_flt.notna()]
    munich_share = float(scored_main.ADEP_mvt.astype(str).eq(AIRPORT).mean())
    coverage = float(len(y) / max(sum(departures.values()), 1))
    implied = OFFICIAL - np.sqrt(max(OFFICIAL ** 2 - (N_MAIN / N_SCORED) * munich_share * coverage * gain.mean(), 0))
    context = {"departures_by_date": departures, "estimates_share_of_departures": coverage,
               "munich_share_of_scored_main_rows": munich_share,
               "official_scale_upper_bound_s": float(implied),
               "why_an_upper_bound": "comparator is weaker than production; July 2026 detector coverage not measured; "
                                     "January 2026 coverage not checked; no transfer discount applied",
               "fallback_rows": {"n": int(ok.lane.eq('fallback').sum()),
                                 "mae_s": float(ok.loc[ok.lane.eq('fallback'), 'err'].abs().mean())
                                 if ok.lane.eq('fallback').any() else None},
               "status_counts": rows.status.value_counts().to_dict()}

    checks = {"V1_accuracy_transfers": accuracy["mae_s"] <= 60 and accuracy["within_60_share"] >= .75,
              "V2_beats_comparator_every_date": all(v["detector_rmse"] < v["comparator_rmse"] for v in per_date.values()),
              "V3_not_concentrated": comparison["mse_gain_without_top5pct"] > 0,
              "V4_interval_above_zero": comparison["bootstrap_95"][0] > 0}
    report = {"preregistration": preregistration, **({"comparator": comparator_name} if comparator_name else {}),
              "dates": list(dates),
              "accuracy_vs_block": accuracy, "comparison_on_identical_main_rows": comparison, "context": context,
              "checks": checks, "justifies_further_work": bool(all(checks.values()))}
    CACHE.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(CACHE / rows_file, index=False)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

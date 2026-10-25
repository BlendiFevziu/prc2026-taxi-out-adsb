"""Can ANY external data source reach the fallback lane's error? Measured, not assumed.

Motivation. adsb.lol (21.09) and OPDI (22.09) were each closed on economics: coverage too thin,
gain 0.14 to 1.9 s. Both arguments were about how many rows a source can see. This tool asks the
prior question: on the rows that actually carry the fallback error, is the target a physical event
at all? If it is not, coverage is beside the point and no combination of sources can help.

Method. Training departures 2025, labels present, no model, no external data, nothing uploaded.
Fallback rows are those where the flight-plan join failed (AOBT_3_flt missing); on them every _flt
column is absent. At rank time such a row gives SCHED and MVT but never BLOCK, so
  versp = MVT - SCHED   is observable
  y     = MVT - BLOCK   is not
and y == versp exactly when the airport wrote the scheduled time into the block field.

Run from the repository root:
  python3 -u -B research_280/fallback_reachability.py
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "research_280/results/fallback_reachability.json"
COLS = ["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"]
TOL = 60          # |BLOCK - SCHED| at or below this counts as a schedule copy
EXTREME = 7200    # a label above two hours is not a taxi time
N, NM, NF, OFF = 344841, 339551, 5290, 265.2839   # scored composition and our v52 official score


def main():
    d = pd.concat([pd.read_parquet(p, columns=COLS) for p in sorted(glob.glob("data/training_*.parquet"))],
                  ignore_index=True)
    d = d[d.PHASE_mvt.eq("DEP")]
    fb = d[d.AOBT_3_flt.isna()].copy()
    fb["versp"] = (fb.MVT_TIME_UTC_mvt - fb.SCHED_TIME_UTC_mvt).dt.total_seconds()
    fb["bs"] = (fb.BLOCK_TIME_UTC_mvt - fb.SCHED_TIME_UTC_mvt).dt.total_seconds()
    y = fb.TAXITIME_SEC_mvt
    med = fb.groupby("ADEP_mvt").TAXITIME_SEC_mvt.transform("median")
    sse = (y - med) ** 2
    tot = float(sse.sum())

    copy, ext = fb.bs.abs() <= TOL, y > EXTREME
    rmse = lambda p: float(np.sqrt(((y - p) ** 2).mean()))
    oracle = rmse(np.where(copy, fb.versp, med))
    flat = rmse(med)

    res = {
        "note": "diagnostic only; training labels, no model, no external data, nothing uploaded",
        "fallback_rows": int(len(fb)), "total_sse_about_airport_median": tot,
        "sse_share_by_label_threshold": {
            str(t): {"rows": int((y > t).sum()), "sse_share": float(sse[y > t].sum() / tot)}
            for t in (1800, 3600, 7200, 21600, 43200, 80000)},
        "copy_rate_by_tolerance": {
            str(t): {"rows": int((fb.bs.abs() <= t).sum()),
                     "sse_share": float(sse[fb.bs.abs() <= t].sum() / tot)}
            for t in (0, 5, 30, 60, 300, 900)},
        "extreme_rows": {
            "threshold_s": EXTREME, "rows": int(ext.sum()), "sse_share": float(sse[ext].sum() / tot),
            "copy": {"rows": int((ext & copy).sum()), "sse_share": float(sse[ext & copy].sum() / tot),
                     "by_airport": fb[ext & copy].ADEP_mvt.value_counts().to_dict(),
                     "comment": "y == versp, and both terms are given at scoring time"},
            "not_copy": {"rows": int((ext & ~copy).sum()), "sse_share": float(sse[ext & ~copy].sum() / tot),
                         "by_airport": fb[ext & ~copy].ADEP_mvt.value_counts().to_dict(),
                         "comment": "y rests on a BLOCK stamp hours from anything observable"}},
        "oracle_ladder_fallback_rmse": {
            "per_airport_median": flat, "perfect_copy_detection": oracle,
            "ratio": oracle / flat,
            "versp_for_every_row": rmse(fb.versp),
            "comment": "predicting the physically-true delay everywhere is far WORSE than a flat median; "
                       "the copy value is a classification problem, never a measurement problem"},
        "max_abs_y_minus_versp_on_copy_rows": float((y - fb.versp)[copy].abs().max()),
    }

    scen = {}
    for m in (200, 210, 216.773, 225, 235):
        fb_sse = OFF ** 2 * N - m ** 2 * NM
        if fb_sse <= 0:
            continue
        r = np.sqrt(fb_sse / NF)
        scen[str(m)] = {"implied_fallback_rmse": float(r), "fallback_share_of_sse": float(fb_sse / (OFF ** 2 * N)),
                        "gain_s_at_oracle_ratio": float(OFF - np.sqrt((m ** 2 * NM + (r * oracle / flat) ** 2 * NF) / N)),
                        "gain_s_if_fallback_perfect": float(OFF - np.sqrt(m ** 2 * NM / N))}
    res["official_scale_value"] = {"our_official": OFF, "scenarios_by_assumed_main_rmse": scen}

    OUT.write_text(json.dumps(res, indent=2) + "\n")
    e = res["extreme_rows"]
    print(f"fallback rows {len(fb):,}")
    print(f"rows above {EXTREME} s: {e['rows']:,} ({e['rows']/len(fb)*100:.2f} %) carry {e['sse_share']*100:.2f} % of fallback SSE")
    print(f"  of them copy     {e['copy']['rows']:>4,} rows -> {e['copy']['sse_share']*100:5.2f} % of SSE, answer already observable")
    print(f"  of them not copy {e['not_copy']['rows']:>4,} rows -> {e['not_copy']['sse_share']*100:5.2f} % of SSE, answer unobservable")
    print(f"fallback RMSE: flat median {flat:.1f} -> perfect copy detection {oracle:.1f} (ratio {oracle/flat:.3f})")
    print(f"  predicting versp everywhere: {res['oracle_ladder_fallback_rmse']['versp_for_every_row']:.1f}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()

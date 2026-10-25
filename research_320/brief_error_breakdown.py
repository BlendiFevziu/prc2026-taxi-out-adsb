"""Aggregates for the one-page project brief (error breakdown and 2026 coverage). Descriptive, no fit, no upload.

A. Held-out January and July 2025 (the v45 ensemble, .cache/sequence_release_v47/validate): main-lane error by airport
   and month, tail concentration, and the fallback lane's share of squared error.
B. The 2026 scored set, label-free: rows per airport, fallback rows, and which rows the NM add-on and the open ADS-B
   layers change in the scored v60 (lineage files hash-checked against their receipts).
C. A rough current split: fallback squared error at its 2025 out-of-sample RMSE against v60's official total.
Run from the repository root:  python3 -u -B research_320/brief_error_breakdown.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
VAL = HERE / ".cache/sequence_release_v47/validate"
OUT = HERE / "research_320/results/brief_error_breakdown.json"
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
FILES = {"v51": ".cache/research_280/release/candidate_v51_combined.parquet",
         "v52": ".cache/research_280/release/candidate_v52_combined.parquet",
         "v60": ".cache/research_320/volume_candidate/candidate.parquet"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = {}
    summary = json.loads((VAL / "ensemble_baseline.json").read_text())
    b = pd.read_parquet(VAL / "ensemble_baseline.parquet", columns=[ID, "MVT_TIME_UTC_mvt", "ADEP_mvt", "y", "baseline"])
    se = (b.baseline - b.y).to_numpy(float) ** 2
    main_sse, fb_sse = float(se.sum()), float(summary["fallback_sse"])
    assert abs(np.sqrt(se.mean()) - summary["main_rmse"]) < 1e-6
    month = pd.to_datetime(b.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%m")
    top = np.sort(se)[::-1]
    out["holdout_2025"] = {
        "main_rows": int(len(b)), "main_rmse": float(np.sqrt(se.mean())), "fallback_rows": summary["fallback_rows"],
        "fallback_rmse": float(np.sqrt(fb_sse / summary["fallback_rows"])), "overall_rmse": summary["overall_rmse"],
        "fallback_share_of_rows": summary["fallback_rows"] / summary["rows"], "fallback_share_of_sse": fb_sse / (fb_sse + main_sse),
        "main_by_month_rmse": {m: float(np.sqrt(se[month.eq(m).to_numpy()].mean())) for m in ("01", "07")},
        "main_top1pct_share_of_main_sse": float(top[: len(top) // 100].sum() / main_sse),
        "main_rows_over_30min_error": int((np.sqrt(se) > 1800).sum()),
        "main_by_airport": {ap: {"rows": int(m.sum()), "rmse": float(np.sqrt(se[m].mean())), "share_of_main_sse": float(se[m].sum() / main_sse)}
                            for ap in sorted(b.ADEP_mvt.unique()) for m in [b.ADEP_mvt.eq(ap).to_numpy()]}}
    fh = json.loads((HERE / "research_300/results/fallback_headroom.json").read_text())
    out["fallback_six_pair"] = {"rows": fh["fallback_rows_2025_out_of_sample"], "rmse": fh["shipped_rmse"],
                                "largest_rows_share": fh["largest_rows_share_of_shipped_sse"],
                                "classes": {k: {"rows": v["rows"], "share_of_sse": v["share_of_shipped_sse"]} for k, v in fh["classes"].items()},
                                "by_airport_share": {a: v["share_of_shipped_sse"] for a, v in fh["airports"].items()}}
    receipts = {v: json.loads((HERE / f"results/upbeat-goblin_{v}.json").read_text()) for v in FILES}
    lineage = {}
    for v, p in FILES.items():
        assert sha(HERE / p) == (receipts[v].get("file_sha256") or receipts[v].get("candidate_sha256")), v
        lineage[v] = pd.read_parquet(HERE / p)
    tmpl = pd.read_parquet(HERE / "data/submitting.parquet", columns=[ID])
    rank = pd.read_parquet(HERE / "data/ranking.parquet", columns=[ID, "ADEP_mvt", "AOBT_3_flt"])
    info = tmpl.merge(rank, on=ID, how="left", validate="one_to_one")
    assert all(lineage[v][ID].equals(tmpl[ID]) for v in FILES)
    ap, fb = info.ADEP_mvt.to_numpy(), info.AOBT_3_flt.isna().to_numpy()
    nm = lineage["v52"][T].to_numpy() != lineage["v51"][T].to_numpy()
    adsb = lineage["v60"][T].to_numpy() != lineage["v52"][T].to_numpy()
    out["scored_2026"] = {"rows": int(len(info)), "fallback_rows": int(fb.sum()), "nm_changed_rows": int(nm.sum()),
                          "open_adsb_changed_rows_vs_v52": int(adsb.sum()), "either": int((nm | adsb).sum()),
                          "main_rows_untouched_by_both": int((~fb & ~nm & ~adsb).sum()),
                          "by_airport": {a: {"rows": int(m.sum()), "fallback": int((m & fb).sum()), "nm": int((m & nm).sum()),
                                             "open_adsb": int((m & adsb).sum())} for a in sorted(set(ap)) for m in [ap == a]}}
    total = receipts["v60"]["score"] ** 2 * len(info)
    est = {k: rmse ** 2 * fb.sum() / total for k, rmse in (("at_jan_jul_2025_rmse", out["holdout_2025"]["fallback_rmse"]),
                                                            ("at_six_pair_rmse", fh["shipped_rmse"]))}
    out["current_split_estimate"] = {"v60_official_rmse": receipts["v60"]["score"], "fallback_share_of_sse": est,
                                     "note": "Fallback unchanged since v50; its 2025 out-of-sample RMSE applied to 2026 fallback rows. Estimate, not measured."}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

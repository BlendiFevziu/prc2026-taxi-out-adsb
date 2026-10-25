"""Does OPDI's physical off-block add anything our model does not already have?

The 6 September check measured OPDI's own spread against the reported block time (180 s at LSZH) and estimated
0.8 s of value against a model that then scored 226 s on the main lane. That model now scores 216.8 overall and
170.8 at LSZH, so OPDI's estimate is no longer clearly better than ours on the rows it can see.

The question is therefore not "how good is OPDI" but "how much of OPDI's information is independent of ours".
Both quantities are measured against the airport's WRITTEN block time, so both carry its writing noise; if that
shared component dominates, combining them buys nothing. The ADS-B work reached exactly that wall.

This measures it directly, on the one 10-day file where OPDI has real coverage:
  e_model = baseline - y            our ensemble's error on held-out January 2025 rows
  e_opdi  = (MVT - T_offblock) - y  the error of taking OPDI's parking exit as the off-block time
and reports their correlation and what a least-squares combination of the two actually achieves.

Matching is label-free: an OPDI exit-parking event is attached to a departure of the same airport and callsign
whose take-off time follows it by 0 to 60 minutes. The block time is never used to match.

  python3 -B research_280/opdi_decisive_test.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
EVENTS = PROJECT / ".cache/opdi/exit_parking_20250105_parsed.parquet"
BASELINE = PROJECT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
OUTPUT = PROJECT / "research_280/results/opdi_decisive_test.json"
AIRPORTS = ("LSZH", "LEBL", "EDDM")


def norm(series):
    """Callsign key: letters plus the numeric tail without leading zeros (OPDI AAL92 vs challenge AAL093)."""
    s = series.astype(str).str.upper().str.strip()
    head = s.str.extract(r"^([A-Z]+)")[0].fillna("")
    tail = s.str.extract(r"([0-9].*)$")[0].fillna("").str.lstrip("0")
    return head + tail


def main():
    ex = pd.read_parquet(EVENTS)
    ex = ex.loc[ex.osm_airport.isin(AIRPORTS)].copy()
    ex["t_off"] = pd.to_datetime(ex.event_time, utc=True)
    ex["key"] = norm(ex.osn_flight_id)
    ex = ex.loc[ex.key.ne("")]
    print(f"OPDI exit-parking events at {AIRPORTS}: {len(ex):,}")

    cols = ["MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "PHASE_mvt",
            "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    dep = pq.ParquetFile(str(PROJECT / "data/training_2025-01-01_2025-02-01.parquet")).read(columns=cols).to_pandas()
    dep = dep.loc[dep.PHASE_mvt.eq("DEP") & dep.ADEP_mvt.isin(AIRPORTS)].copy()
    dep["mvt"] = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True)
    lo, hi = ex.t_off.min(), ex.t_off.max()
    dep = dep.loc[dep.mvt.between(lo, hi + pd.Timedelta(hours=2))]
    dep["key"] = norm(dep.CALLSIGN_flt.where(dep.CALLSIGN_flt.notna(), dep.FLIGHT_mvt))
    print(f"challenge departures in the same window: {len(dep):,}")

    joined = dep.merge(ex[["osm_airport", "key", "t_off", "osm_ref"]],
                       left_on=["ADEP_mvt", "key"], right_on=["osm_airport", "key"], how="inner")
    gap = (joined.mvt - joined.t_off).dt.total_seconds()
    joined = joined.loc[gap.between(0, 3600)].copy()
    joined["gap_s"] = (joined.mvt - joined.t_off).dt.total_seconds()
    joined = joined.sort_values("gap_s").drop_duplicates("MVT_ID_mvt")
    print(f"matched departures: {len(joined):,} ({len(joined)/len(dep):.1%} of the window's departures)")

    base = pd.read_parquet(BASELINE, columns=["MVT_ID_mvt", "y", "baseline"])
    m = joined.merge(base, on="MVT_ID_mvt", how="inner")
    print(f"of which also carry a held-out model prediction: {len(m):,}\n")
    if len(m) < 50:
        raise SystemExit("too few matched rows to conclude")

    y = m.y.to_numpy(float)
    e_model = m.baseline.to_numpy(float) - y
    opdi_taxi = m.gap_s.to_numpy(float)
    e_opdi = opdi_taxi - y
    rho = float(np.corrcoef(e_model, e_opdi)[0, 1])
    # least-squares combination, fitted on these rows: an upper bound, it sees its own answer
    X = np.column_stack([np.ones(len(y)), m.baseline.to_numpy(float), opdi_taxi])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e_comb = X @ beta - y
    out = {"airports": list(AIRPORTS), "matched_rows": int(len(m)),
           "rmse_model": float(np.sqrt(np.mean(e_model ** 2))),
           "rmse_opdi_raw": float(np.sqrt(np.mean(e_opdi ** 2))),
           "opdi_median_offset_s": float(np.median(e_opdi)),
           "rmse_opdi_after_removing_its_median_bias": float(np.sqrt(np.mean((e_opdi - np.median(e_opdi)) ** 2))),
           "correlation_of_the_two_errors": rho,
           "rmse_least_squares_combination_in_sample": float(np.sqrt(np.mean(e_comb ** 2))),
           "in_sample_gain_s": float(np.sqrt(np.mean(e_model ** 2)) - np.sqrt(np.mean(e_comb ** 2)))}
    for k, v in out.items():
        print(f"  {k:52s} {v if isinstance(v,(list,int)) else round(v,3)}")
    print("\nThe correlation is the decisive number: both errors are measured against the same written block")
    print("time, and whatever they share cannot be removed by combining them.")
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()

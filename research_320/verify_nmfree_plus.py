"""Independent pre-upload check of the NM-free candidate (research_320/build_nmfree_plus.py). No upload.

Fresh process, own checks against three files: the candidate, Codex's NM-free reference and the scored v60.
  1. hashes; ids, order and schema equal the template; submission.pruefe clean; all finite
  2. against the reference, only rows of the three layers change, and every one of them sat at the pre-ADS-B
     comparator in the reference; their new values recomputed here from the frozen models
  3. NM-free: no row that the reference restored still carries its v60 (NM) value, except where both sit at
     the 300 s floor
  4. every row that differs from v60 is either a restored NM row or a layer row on days 6, 13 or 20
Run from the repository root:  python3 -u -B research_320/verify_nmfree_plus.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import stand_experiment as S  # noqa: E402

MS, R = S.MS, S.R
import submission as SUB  # noqa: E402

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
CAND = HERE / ".cache/research_320/nmfree_plus/candidate_nmfree_plus.parquet"
REF = HERE / "research_330_nmfree/private/external_nm_free_reference.parquet"
V60 = HERE / ".cache/research_320/volume_candidate/candidate.parquet"
OUT = HERE / "research_320/results/nmfree_plus_verification.json"
SEEDS = (177, 277, 377)
EH_SCHEMA = ["comparator", "ersatz_s", "schedule", "gate", "hour_sin", "hour_cos", "first_age_s", "last_age_s", "span_s",
             "endpoint_distance_m", "weighted_speed_kt", "weighted_still_share", "observed_interval_share", "is_july"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / "research_320/results/nmfree_plus.json").read_text())
    c = {"candidate_sha_equals_build": sha(CAND) == build["candidate_sha256"],
         "reference_sha": sha(REF) == build["base"]["sha256"],
         "v60_sha": sha(V60) == json.loads((HERE / "results/upbeat-goblin_v60.json").read_text())["file_sha256"]}
    cand, ref, v60 = pd.read_parquet(CAND), pd.read_parquet(REF), pd.read_parquet(V60)
    tmpl = pd.read_parquet(HERE / "data/submitting.parquet")
    c["ids_order_equal_template"] = bool(cand[ID].equals(tmpl[ID]) and ref[ID].equals(tmpl[ID]) and v60[ID].equals(tmpl[ID]))
    c["schema_equal_v60"] = bool(list(cand.columns) == list(v60.columns) and (cand.dtypes == v60.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(cand, tmpl, None)]
    c["all_finite"] = bool(np.isfinite(cand[T]).all())
    vc, vr, v6 = cand[T].to_numpy(float), ref[T].to_numpy(float), v60[T].to_numpy(float)
    restored = vr != v6
    layer = vc != vr

    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet").set_index(ID)
    ids = cand.loc[layer, ID]
    sub = ctx.loc[ids].reset_index()
    sub["cell"] = sub.airport + "/" + sub.date.str[5:7]
    c["layer_rows"] = int(layer.sum())
    c["layer_rows_on_days_6_13_20_only"] = bool(sub.date.str[8:10].isin(["06", "13", "20"]).all())
    c["layer_rows_were_at_comparator_in_reference"] = bool(np.array_equal(ref.set_index(ID).loc[ids, T].to_numpy(float), sub.comparator.to_numpy(float)))

    # recompute every layer value from the frozen models
    new = pd.Series(np.nan, index=ids.to_numpy())
    part = sub[sub.category.eq("partial_ground") & ~sub.cell.eq("EHAM/01")].reset_index(drop=True)
    corr = np.mean([R.joblib.load(HERE / f".cache/research_320/volume/models/arm4_real_{s}.pkl").predict(MS.inputs(part, "real")) for s in SEEDS], axis=0)
    new.loc[part[ID].to_numpy()] = np.maximum(part.comparator.to_numpy(float) + corr, 300.0)
    stand = sub[sub.category.eq("observed_at_assigned_stand")].reset_index(drop=True)
    days = pd.concat([pd.read_parquet(p) for p in sorted((HERE / ".cache/research_320/nmfree_plus").glob("day_*.parquet"))]).set_index(ID)
    stand["detector_taxi_s"] = days.loc[stand[ID], "detector_taxi_s"].to_numpy(float)
    corr_s = np.mean([R.joblib.load(S.CACHE / "models" / f"real_{s}.pkl").predict(stand[S.COLS]) for s in S.SEEDS], axis=0)
    new.loc[stand[ID].to_numpy()] = np.maximum(stand.comparator.to_numpy(float) + 0.5 * corr_s, 300.0)
    eh = sub[sub.cell.eq("EHAM/01")].reset_index(drop=True)
    feats = pd.concat([pd.read_parquet(HERE / f".cache/codex_eham/eham_recovery/features_2026-01-{d}.parquet") for d in ("06", "13", "20")])
    eh = eh.merge(feats[[ID] + EH_SCHEMA[6:13]], on=ID, how="left", validate="one_to_one")
    corr_e = np.mean([R.joblib.load(HERE / f".cache/codex_eham/eham_recovery/real_{s}.pkl").predict(eh[EH_SCHEMA]) for s in SEEDS], axis=0)
    new.loc[eh[ID].to_numpy()] = np.maximum(eh.comparator.to_numpy(float) + corr_e, 300.0)
    c["layer_rows_recomputed"] = int(new.notna().sum())
    c["layer_values_equal_recomputed"] = bool(np.array_equal(new.loc[ids].to_numpy().astype(cand[T].dtype), vc[layer].astype(cand[T].dtype)))
    c["rows_outside_layers_equal_reference"] = bool(np.array_equal(vc[~layer], vr[~layer]))

    c["restored_nm_rows"] = int(restored.sum())
    c["restored_rows_still_at_v60_value"] = int((restored & (vc == v6)).sum())
    # Allowed only at the 300 s floor, which the old NM value and the new layer value can both hit on their own
    c["restored_rows_at_v60_value_not_at_floor"] = int((restored & (vc == v6) & (vc != 300.0)).sum())
    diff60 = vc != v6
    c["rows_differing_from_v60"] = int(diff60.sum())
    c["every_v60_difference_is_restored_or_layer"] = bool(not (diff60 & ~restored & ~layer).any())
    ok = (all(v for k, v in c.items() if isinstance(v, bool)) and not c["pruefe_errors"]
          and c["restored_rows_at_v60_value_not_at_floor"] == 0 and c["layer_rows_recomputed"] == c["layer_rows"])
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()

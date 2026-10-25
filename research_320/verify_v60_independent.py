"""Independent check of the scored v60 (Codex's build and upload) against the tested arm 4. No fit, no upload.

Own code path, separate from Codex's builder and reviewer: rows and inputs through the frozen v56 inputs() of the
read-only checkout, models from the hash-recorded research_320 cache. The uploaded file must equal, byte for byte in
value, the scored v59 with the 66,292 deployed partial-track rows set to max(comparator + mean arm 4 correction, 300),
and v59 on those rows must equal the deployed correction's formula. Also records the realised transfer.
Run from the repository root:  python3 -u -B research_320/verify_v60_independent.py
"""
import hashlib
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
ID, T, SEEDS = "MVT_ID_mvt", "TAXITIME_SEC_mvt", (177, 277, 377)
V60 = HERE / ".cache/research_320/volume_candidate/candidate.parquet"
V59 = HERE / ".cache/research_320/eham_candidate/candidate_v59_eham.parquet"
MODELS = HERE / ".cache/research_320/volume/models"
OUT = HERE / "research_320/results/v60_independent_check.json"
N = 344841


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rec60 = json.loads((HERE / "results/upbeat-goblin_v60.json").read_text())
    rec59 = json.loads((HERE / "results/upbeat-goblin_v59.json").read_text())
    volume = json.loads((HERE / "research_320/results/volume_capacity.json").read_text())
    c = {"v60_file_is_scored_file": sha(V60) == rec60["file_sha256"], "v59_file_is_scored_file": sha(V59) == rec59["file_sha256"],
         "models_as_tested": all(sha(MODELS / f"{a}_real_{s}.pkl") == volume["model_hashes"][f"{a}_real_{s}.pkl"]
                                 for a in ("arm1", "arm4") for s in SEEDS)}
    upd = pd.read_parquet(HERE / ".cache/codex_v56/partial_release_v55/updates.parquet")
    rows = upd[[ID, "residual"]].merge(pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet"),
                                       on=ID, how="left", validate="one_to_one")
    x = MS.inputs(rows, "real")
    mean = lambda arm: np.mean([R.joblib.load(MODELS / f"{arm}_real_{s}.pkl").predict(x) for s in SEEDS], axis=0)
    old, new = mean("arm1"), mean("arm4")
    comp = rows.comparator.to_numpy(float)
    v59, v60 = pd.read_parquet(V59), pd.read_parquet(V60)
    c["same_ids_same_order"] = bool(v60[ID].equals(v59[ID]))
    c["same_schema"] = bool(list(v60.columns) == list(v59.columns) and (v60.dtypes == v59.dtypes).all())
    at59, at60 = v59.set_index(ID).loc[rows[ID], T], v60.set_index(ID).loc[rows[ID], T]
    c["old_correction_equals_codex_2026_file"] = bool(np.array_equal(old, rows.residual.to_numpy(float)))
    c["v59_rows_equal_deployed_formula"] = bool(np.array_equal(np.maximum(comp + old, 300.).astype(at59.dtype), at59.to_numpy()))
    c["v60_rows_equal_arm4_formula"] = bool(np.array_equal(np.maximum(comp + new, 300.).astype(at60.dtype), at60.to_numpy()))
    outside = ~v60[ID].isin(rows[ID]).to_numpy()
    c["every_other_row_identical_to_v59"] = bool(v60.loc[outside].reset_index(drop=True).equals(v59.loc[outside].reset_index(drop=True)))
    c["rows_changed"] = int((v60[T].to_numpy() != v59[T].to_numpy()).sum())
    c["rows_changed_outside_overlay"] = int((v60.loc[outside, T].to_numpy() != v59.loc[outside, T].to_numpy()).sum())
    realised = rec59["score"] ** 2 - rec60["score"] ** 2
    plan = volume["planning"]
    report = {"checks": c, "verdict": "v60 is the tested arm 4 on exact v59" if all(v for v in c.values() if isinstance(v, bool))
              and c["rows_changed_outside_overlay"] == 0 else "MISMATCH",
              "transfer": {"realised_mse_per_scored_row": realised, "predicted_full_arm4": plan["arm4"]["mse_per_scored_row"],
                           "ratio_to_full_arm4": realised / plan["arm4"]["mse_per_scored_row"],
                           "realised_per_overlay_row": realised * N / len(rows),
                           "predicted_full_arm3_for_reference": plan["arm3"]["mse_per_scored_row"]}}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

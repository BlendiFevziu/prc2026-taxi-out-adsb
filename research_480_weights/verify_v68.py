"""Independent pre-upload check of v68 (research_480_weights/build_v68.py), in a fresh process. No upload.

1. hashes; ids, order and schema equal v67 and the template; submission.pruefe clean; all finite
2. every changed row is an EHAM January partial row of Codex's 2026 feature table, and everything else equals v67
3. on those rows, v67 equals max(base + add-on, 300) and v68 equals max(base + (add-on + C1) / 2, 300), both corrections
   recomputed here by loading the frozen seed models directly (the build went through layers.correction), with the base
   from the pre-NM base file
Run from the repository root:  python3 -u -B research_480_weights/verify_v68.py
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V68 = HERE / ".cache/research_480/v68/candidate_v68.parquet"
V67 = HERE / ".cache/research_450/v67/candidate_v67.parquet"
OUT = HERE / "research_480_weights/results/v68_verification.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / "research_480_weights/results/v68_build.json").read_text())
    rec67 = json.loads((HERE / "results/upbeat-goblin_v67.json").read_text())
    c = {"v68_sha_equals_build": sha(V68) == build["candidate_sha256"], "v67_is_scored_file": sha(V67) == rec67["file_sha256"]}
    v68, v67 = pd.read_parquet(V68), pd.read_parquet(V67)
    tmpl = pd.read_parquet(HERE / "data/submitting.parquet")
    c["ids_order_equal"] = bool(v68[ID].equals(v67[ID]) and v68[ID].equals(tmpl[ID]))
    c["schema_equal_v67"] = bool(list(v68.columns) == list(v67.columns) and (v68.dtypes == v67.dtypes).all())
    c["pruefe_errors"] = [str(e) for e in SUB.pruefe(v68, tmpl, None)]
    c["all_finite"] = bool(np.isfinite(v68[T]).all())
    f26 = pd.read_parquet(HERE / ".cache/codex_v56/full_coverage_2026/features.parquet", columns=[ID, "airport", "date", "category"])
    eham01 = set(f26.loc[f26.airport.eq("EHAM") & f26.date.str[5:7].eq("01") & f26.category.eq("partial_ground"), ID])
    diff = v68[T].to_numpy() != v67[T].to_numpy()
    c["rows_changed"] = int(diff.sum())
    c["every_change_is_eham_january_partial"] = bool(v68.loc[diff, ID].isin(eham01).all())
    feats = pd.concat([pd.read_parquet(p, columns=[ID] + list(L.EHAM_TRACK))
                       for p in sorted(glob.glob(str(HERE / ".cache/codex_eham/eham_recovery/features_2026-01-*.parquet")))], ignore_index=True)
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    frame = ctx[ctx[ID].isin(eham01)].merge(feats.drop_duplicates(ID), on=ID, how="inner", validate="one_to_one")
    frame["comparator"] = H.current_base_2026().loc[frame[ID]].to_numpy(float)
    x_addon = frame[list(L.CONTEXT) + list(L.EHAM_TRACK) + ["is_july"]]
    addon = np.mean([joblib.load(HERE / L.LAYERS["eham_january"].models.format(seed=s)).predict(x_addon) for s in L.SEEDS], axis=0)
    c1 = np.mean([joblib.load(HERE / L.LAYERS["partial"].models.format(seed=s)).predict(MS.inputs(frame, "real")) for s in L.SEEDS], axis=0)
    b = frame.comparator.to_numpy(float)
    at67 = v67.set_index(ID).loc[frame[ID], T].to_numpy()
    owned = np.maximum(b + addon, 300.0).astype(at67.dtype) == at67
    c["eham_january_rows_carrying_the_addon_in_v67"] = int(owned.sum())
    c["equals_build_rows"] = int(owned.sum()) == build["checks"]["owned_rows"]
    f = frame[owned].reset_index(drop=True)
    expect = np.maximum(b[owned] + (addon[owned] + c1[owned]) / 2, 300.0).astype(v68[T].dtype)
    c["values_equal_average"] = bool(np.array_equal(v68.set_index(ID).loc[f[ID], T].to_numpy(), expect))
    c["outside_rows_identical_to_v67"] = bool(np.array_equal(v68.loc[~v68[ID].isin(f[ID]), T].to_numpy(), v67.loc[~v67[ID].isin(f[ID]), T].to_numpy()))
    ok = all(v for v in c.values() if isinstance(v, bool)) and not c["pruefe_errors"]
    c["verdict"] = "verified" if ok else "MISMATCH"
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()

"""Rebuild the v46 submission file from its scored inputs and compare it with the upload.

v46 is the scored v45 submission with its 5,290 fallback predictions replaced by the
neighbour-taxi anchored predictions of research_240/anchor_preview_2026_production.py.
This is the build step that ran inline on 13 September 2026, unchanged apart from the
arguments and the final comparison with results/upbeat-goblin_v46_manifest.json.

Inputs, local to the release Mac and never in Git:
  .cache/clock_queue_release_20260912/candidate.parquet       scored v45 submission
  .cache/clock_queue_release_20260912/data/ranking.parquet    hat_aobt per ranking row
  .cache/research_240/anchor_preview_2026_production.parquet  anchored fallback predictions

Run from the repository root:
  python3 research_240/build_v46_candidate.py --output .cache/research_240/v46_rebuild/candidate.parquet

Exit code 0 when the rebuilt file has the uploaded SHA-256. Never uploads anything and
never overwrites a different existing file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output if args.output.is_absolute() else project / args.output
    sys.path.insert(0, str(project))
    os.chdir(project)
    import main_model as H  # noqa: F401  torch before tree libraries on macOS
    import submission as S
    import numpy as np
    import pandas as pd

    v45_path = Path(".cache/clock_queue_release_20260912/candidate.parquet")
    ranking_path = Path(".cache/clock_queue_release_20260912/data/ranking.parquet")
    anchored_path = Path(".cache/research_240/anchor_preview_2026_production.parquet")
    manifest45 = json.loads(Path("results/upbeat-goblin_v45_manifest.json").read_text())
    manifest46 = json.loads(Path("results/upbeat-goblin_v46_manifest.json").read_text())
    assert sha(v45_path) == manifest45["submission_sha256"], "local v45 file is not the scored v45 submission"
    assert sha(anchored_path) == manifest46["anchored_fallback_predictions_sha256"], "anchored predictions changed"

    template = pd.read_parquet("data/submitting.parquet")
    v45 = pd.read_parquet(v45_path)
    assert not S.pruefe(v45, template, None), S.pruefe(v45, template, None)
    rank = pd.read_parquet(ranking_path, columns=["MVT_ID_mvt", "hat_aobt"])
    preview = pd.read_parquet(anchored_path)
    gate = json.loads(Path("research_240/results/anchor_preview_2026_production.json").read_text())["replication_gate"]
    assert gate["passed"] and preview.MVT_ID_mvt.is_unique
    fallback_ids = pd.Index(rank.loc[rank.hat_aobt.eq(0), "MVT_ID_mvt"])
    main_ids = pd.Index(rank.loc[rank.hat_aobt.eq(1), "MVT_ID_mvt"])
    assert set(preview.MVT_ID_mvt) == set(fallback_ids) and len(fallback_ids) == 5290
    v45i = v45.set_index("MVT_ID_mvt", verify_integrity=True)
    np.testing.assert_array_equal(preview.set_index("MVT_ID_mvt").v45_official.loc[fallback_ids].to_numpy(),
                                  v45i.TAXITIME_SEC_mvt.loc[fallback_ids].to_numpy())

    out = v45.copy()
    anchored = preview.set_index("MVT_ID_mvt").anchored
    values = out.TAXITIME_SEC_mvt.to_numpy().copy()
    pos = pd.Index(out.MVT_ID_mvt).get_indexer(anchored.index)
    assert (pos >= 0).all()
    values[pos] = anchored.to_numpy()
    out["TAXITIME_SEC_mvt"] = pd.Series(values).astype(template.TAXITIME_SEC_mvt.dtype).to_numpy()

    errors = S.pruefe(out, template, None)
    assert not errors, errors
    assert np.isfinite(out.TAXITIME_SEC_mvt).all()
    outi = out.set_index("MVT_ID_mvt")
    assert np.array_equal(outi.TAXITIME_SEC_mvt.loc[main_ids].to_numpy(), v45i.TAXITIME_SEC_mvt.loc[main_ids].to_numpy())
    assert np.array_equal(outi.TAXITIME_SEC_mvt.loc[fallback_ids].to_numpy(),
                          anchored.loc[fallback_ids].astype(template.TAXITIME_SEC_mvt.dtype).to_numpy())
    changed = out.TAXITIME_SEC_mvt.to_numpy() != v45.TAXITIME_SEC_mvt.to_numpy()
    assert set(out.MVT_ID_mvt[changed]) <= set(fallback_ids)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(output), out, check_exact=True)
    else:
        temporary = output.with_name(output.name + ".partial")
        out.to_parquet(temporary, index=False)
        temporary.replace(output)
    rebuilt = sha(output)
    report = {"output": str(output), "sha256": rebuilt, "uploaded_sha256": manifest46["submission_sha256"],
              "sha256_matches_upload": rebuilt == manifest46["submission_sha256"], "rows": len(out),
              "rows_changed": int(changed.sum()),
              "mean_change_on_changed_rows_s": float((out.TAXITIME_SEC_mvt.to_numpy()[changed]
                                                     - v45.TAXITIME_SEC_mvt.to_numpy()[changed]).mean())}
    uploaded = Path(manifest46["submission_path"])
    if uploaded.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(uploaded), out, check_exact=True)
        report["frame_equals_local_uploaded_file"] = True
    print(json.dumps(report, indent=2))
    return 0 if report["sha256_matches_upload"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Replay the explicit candidate schema against one complete cached source split."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--core-valid", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
sys.path.insert(0, str(args.project.resolve()))
import main_model as H
import pandas as pd
from pandas.testing import assert_frame_equal
sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_features as B
import features_time_units as U
import features_correct_queue as Q

source = json.loads(args.core_valid.with_suffix(".json").read_text())
core = pd.read_parquet(args.core_valid)
raw = pd.concat([pd.read_parquet(args.project/"data"/f, columns=B.RAW_COLUMNS)
                 for f in source["source_files"]], ignore_index=True)
legacy = core.drop(columns=[c for c in B.NEW_COLUMNS if c in core])
replayed = B.attach(raw, legacy)
assert_frame_equal(replayed[legacy.columns], legacy, check_exact=True)
assert_frame_equal(replayed[U.MERKMALE], core[U.MERKMALE], check_exact=True)
screen_x = core[H.MERKMALE_HAUPT].copy()
for col in U.CORRECTED_COLUMNS:
    screen_x[col] = core["tu_"+col].to_numpy()
queue = Q.baue(raw, core[Q.DEP_COLUMNS])
for col in Q.MERKMALE:
    screen_x[col] = queue[col].to_numpy()
explicit_x = replayed[B.main_features(H.MERKMALE_HAUPT)].copy()
explicit_x.columns = screen_x.columns
assert_frame_equal(explicit_x, screen_x, check_exact=True)
fallback_columns = [c for c in H.MERKMALE_RUECK if c in legacy]
assert_frame_equal(replayed[fallback_columns], legacy[fallback_columns], check_exact=True)
report = {"source_months": source["source_months"], "departure_rows": len(core),
          "raw_rows": len(raw), "main_features": len(explicit_x.columns),
          "identical_to_screen_feature_matrix": True, "legacy_values_and_dtypes_unchanged": True,
          "corrected_values_match_original_cache_exactly": True,
          "fallback_available_columns_unchanged": fallback_columns,
          "note": "No model fit or submission. Stand target encoding is not part of the raw core and remains a separate training-only step.",
          "schema": B.schema(H.MERKMALE_HAUPT, H.MERKMALE_RUECK)}
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

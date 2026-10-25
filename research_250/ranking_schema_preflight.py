"""Check label-free candidate construction on ranking inputs; never predict/upload."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
args.project, args.output = args.project.resolve(), args.output.resolve()
if args.output.exists():
    raise FileExistsError(args.output)
os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent/"matplotlib"))
sys.path.insert(0, str(args.project))
os.chdir(args.project)
import main_model as H
import features_geometry as G
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_features as B

print("Building the isolated candidate schema on ranking inputs; no weights loaded.", flush=True)
ranking = B.baue([args.project/"data/ranking.parquet"], G.alle())
template = pd.read_parquet(args.project/"data/submitting.parquet", columns=["MVT_ID_mvt"])
assert ranking.MVT_ID_mvt.is_unique and template.MVT_ID_mvt.is_unique
assert set(ranking.MVT_ID_mvt) == set(template.MVT_ID_mvt)
assert ranking.y.isna().all(), "Ranking targets must remain unobserved"
main_columns = B.main_features(H.MERKMALE_HAUPT)
assert set(main_columns) <= set(ranking.columns)
assert set(H.MERKMALE_RUECK)-{"q_stand", "q_stand_n"} <= set(ranking.columns)
assert not np.isinf(ranking[B.NEW_COLUMNS].to_numpy(float)).any()
feature_summary = {}
for col in B.NEW_COLUMNS:
    s = ranking[col]
    feature_summary[col] = {"missing_fraction": float(s.isna().mean()),
                            "p01": float(s.quantile(.01)), "median": float(s.median()),
                            "p99": float(s.quantile(.99))}
report = {"departure_rows": len(ranking), "main_rows": int(ranking.hat_aobt.eq(1).sum()),
          "fallback_rows": int(ranking.hat_aobt.eq(0).sum()),
          "matches_template_ids": True, "ranking_targets_unobserved": True,
          "all_91_main_columns_available": True, "new_columns_contain_no_infinity": True,
          "feature_summary": feature_summary, "schema": B.schema(H.MERKMALE_HAUPT, H.MERKMALE_RUECK),
          "note": "Input/schema preflight only. No weights trained or loaded, no predictions or submission generated, no upload."}
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)

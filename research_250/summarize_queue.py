"""Require all six source pairs before summarizing the locked queue candidate."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .summarize250 import summarize
else:
    from summarize250 import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", required=True, type=Path)
    parser.add_argument("--confirmation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frames, reports = [], []
    for first in range(1, 7):
        suffix = f"{first}_{first+6}"
        folder = args.screen_dir if first == 1 else args.confirmation_dir
        report = json.loads((folder/f"queue_result_{suffix}.json").read_text())
        assert report["source_months"] == [first, first+6]
        frame = pd.read_parquet(folder/f"queue_predictions_{suffix}.parquet")
        assert len(frame) == report["rows"]
        assert frame[["y", "v44_main", "units_replaced", "units_plus_queue"]].notna().all().all()
        for name in report["scores"]:
            np.testing.assert_allclose(np.mean((frame[name]-frame.y)**2), report["scores"][name]["mse"], rtol=1e-12)
        frame["fold"] = suffix
        frames.append(frame)
        reports.append(report)
    for name in ("features_correct_queue.py", "features_time_units.py", "main_model.py"):
        assert len({r["code_sha256"][name] for r in reports}) == 1, f"Recipe changed: {name}"
    assert len({r["code_sha256"]["queue_screen.py"] for r in reports[1:]}) == 1
    pooled = pd.concat(frames, ignore_index=True)
    base_error = (pooled.v44_main-pooled.y)**2
    candidate_error = (pooled.units_plus_queue-pooled.y)**2
    worst = base_error.nlargest(int(np.ceil(len(pooled)*.01))).index
    tail = {}
    for label, mask in (("baseline_worst_one_percent", pooled.index.isin(worst)),
                        ("other_ninety_nine_percent", ~pooled.index.isin(worst))):
        tail[label] = {"rows": int(mask.sum()), "baseline_mse": float(base_error[mask].mean()),
                       "candidate_mse": float(candidate_error[mask].mean()),
                       "mse_gain": float(base_error[mask].mean()-candidate_error[mask].mean())}
    result = {"official_best_rmse": 272.334, "official_score_unchanged": True,
        "validation": "single-LightGBM main component only; no full ensemble, new model bundle or submission",
        "selection_note": "January/July selected the queue family; five other pairs use the same configuration. Early stopping observes each validation fold, as in the baseline. Bootstrap excludes selection and future-year uncertainty.",
        "all_six_vs_v44_component": summarize(frames, "v44_main", "units_plus_queue"),
        "all_six_incremental_over_corrected_units": summarize(frames, "units_replaced", "units_plus_queue"),
        "five_confirmation_pairs_incremental": summarize(frames[1:], "units_replaced", "units_plus_queue"),
        "baseline_defined_tail_diagnostic": tail,
        "tail_note": "Diagnostic only, not a filtered training or scoring set; all rows remain in primary RMSE.",
        "fold_reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "fold_reports"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

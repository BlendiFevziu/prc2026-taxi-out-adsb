"""One fresh-process stage of the locally authorized validation/retraining run."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=["parity", "tune", "baseline", "candidate", "gate", "final", "export"])
    args = parser.parse_args()
    project, run = args.project.resolve(), args.run.resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(run/"matplotlib"))
    sys.path.insert(0, str(project))
    os.chdir(project)
    from research_250 import release_common as R
    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import submission as S

    manifest = R.check_manifest(run, project)
    stage, jobs = args.stage, manifest["jobs"]
    holdout = manifest["outer_source_months"]
    learning = sorted(set(range(1, 13))-set(holdout))
    R.progress(run, stage, "Starting stage")

    if stage == "parity":
        frame = R.dataset(run, project, ranking=True)
        old_path = Path(manifest["legacy_model"])
        assert R.sha(old_path) == manifest["legacy_model_sha256"]
        assert R.sha(manifest["legacy_submission"]) == manifest["legacy_submission_sha256"]
        old = joblib.load(old_path)
        prediction = R.predict(old, frame)
        scored = pd.read_parquet(manifest["legacy_submission"]).set_index("MVT_ID_mvt", verify_integrity=True)
        assert set(scored.index) == set(frame.MVT_ID_mvt)
        reference = scored.loc[frame.MVT_ID_mvt, "TAXITIME_SEC_mvt"].to_numpy()
        difference = prediction-reference
        maximum = float(np.abs(difference).max())
        rms = float(np.sqrt(np.mean(difference**2)))
        report = {"rows": len(frame), "maximum_absolute_difference_seconds": maximum,
                  "rms_difference_seconds": rms, "max_tolerance_seconds": .05,
                  "rms_tolerance_seconds": .005, "passed": maximum <= .05 and rms <= .005,
                  "note": "Numerical replay tolerance allows GPU floating-point variation; no score estimated."}
        R.json_write(run/"parity.json", report)
        assert report["passed"], report
    elif stage == "tune":
        inner = manifest["inner_source_months"]
        assert set(inner) <= set(learning)
        train = R.dataset(run, project, sorted(set(learning)-set(inner)))
        valid = R.dataset(run, project, inner)
        R.split_check(train, valid)
        train = train.loc[train.hat_aobt.eq(1)].copy()
        valid = valid.loc[valid.hat_aobt.eq(1)].copy()
        features = R.B.main_features(R.H.MERKMALE_HAUPT)
        model = R.tree(3000, 177, jobs)
        R.progress(run, stage, "Selecting candidate training length on inner source months only")
        model.fit(train[features], train.y-train.ersatz_s,
            eval_set=[(valid[features], valid.y-valid.ersatz_s)], eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(300)])
        report = {"inner_learning_source_months": sorted(set(learning)-set(inner)),
                  "inner_validation_source_months": inner, "outer_source_months_not_read": holdout,
                  "best_iteration": int(model.best_iteration_),
                  "candidate_lgb_rounds": int(math.ceil(model.best_iteration_*1.05)),
                  "selection": "one training-only inner split; fixed 5% round uplift; same rounds in outer validation and final fit",
                  "max_rounds": 3000, "hit_round_limit": model.best_iteration_ == 3000}
        R.json_write(run/"tuning.json", report)
    elif stage in ("baseline", "candidate", "final"):
        if stage != "baseline":
            tuning = json.loads((run/"tuning.json").read_text())
        if stage == "final":
            assert json.loads((run/"gate.json").read_text())["passed"]
        months = list(range(1,13)) if stage == "final" else learning
        frame = R.dataset(run, project, months)
        recipe = {"candidate": stage != "baseline", "jobs": jobs,
                  "lgb_rounds": 890 if stage == "baseline" else tuning["candidate_lgb_rounds"],
                  "source_months": months, "outer_source_months": [] if stage == "final" else holdout,
                  "lgb_seeds": list(R.H.SAATEN), "catboost_seeds": list(R.H.CB_SAATEN),
                  "catboost_rounds": R.H.CB_RUNDEN, "neural_seeds": list(R.NM.SAATEN),
                  "neural_epochs": R.NM.EPOCHEN, "blends_unchanged": True}
        fallback_source = (manifest["legacy_model"] if stage == "final" else
                           run/"baseline/model.pkl" if stage == "candidate" else None)
        model = R.fit_ensemble(frame, run, stage, recipe, fallback_source)
        if stage != "final":
            valid = R.dataset(run, project, holdout)
            R.split_check(frame, valid)
            prediction = R.predict(model, valid)
            out = valid[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "hat_aobt", "y"]].copy()
            out["prediction"] = prediction
            out.to_parquet(run/stage/"holdout_predictions.parquet", index=False)
            R.json_write(run/stage/"scores.json", {"source_months": holdout,
                "validation": "complete 5-LightGBM/3-CatBoost/5-neural/10-airport ensemble plus reporting gate and unchanged fallback",
                "recipe": recipe, **R.score(valid, prediction)})
    elif stage == "gate":
        base = pd.read_parquet(run/"baseline/holdout_predictions.parquet").set_index("MVT_ID_mvt", verify_integrity=True)
        candidate = pd.read_parquet(run/"candidate/holdout_predictions.parquet").set_index("MVT_ID_mvt", verify_integrity=True)
        assert set(base.index) == set(candidate.index)
        candidate = candidate.loc[base.index]
        np.testing.assert_array_equal(base.y, candidate.y)
        np.testing.assert_array_equal(base.hat_aobt, candidate.hat_aobt)
        before, after = R.score(base, base.prediction.to_numpy()), R.score(candidate, candidate.prediction.to_numpy())
        main_gain = before["main_rmse"]-after["main_rmse"]
        total_gain = before["rmse"]-after["rmse"]
        fallback_same = np.array_equal(base.loc[base.hat_aobt.eq(0), "prediction"],
                                       candidate.loc[candidate.hat_aobt.eq(0), "prediction"])
        e0, e1 = (base.prediction-base.y)**2, (candidate.prediction-candidate.y)**2
        airports = [{"airport": str(airport), "rows": len(group),
                     "baseline_rmse": float(np.sqrt(e0.loc[group.index].mean())),
                     "candidate_rmse": float(np.sqrt(e1.loc[group.index].mean()))}
                    for airport, group in base.groupby("ADEP_mvt", observed=True)]
        top = e0.nlargest(int(np.ceil(.01*len(base)))).index
        report = {"passed": main_gain >= manifest["main_gain_bar_seconds"] and total_gain > 0 and fallback_same,
            "main_gain_bar_seconds": manifest["main_gain_bar_seconds"], "main_rmse_gain": main_gain,
            "overall_rmse_gain": total_gain, "baseline": before, "candidate": after,
            "fallback_predictions_identical": fallback_same, "airports": airports,
            "baseline_worst_one_percent_mse_gain": float((e0.loc[top]-e1.loc[top]).mean()),
            "other_ninety_nine_percent_mse_gain": float((e0.drop(top)-e1.drop(top)).mean()),
            "validation_note": "One complete January/July paired ensemble holdout, plus prior six-pair single-LightGBM evidence. Not six-pair full-ensemble validation. Not an official score."}
        R.json_write(run/"gate.json", report)
        if not report["passed"]:
            raise RuntimeError("Candidate failed the frozen full-ensemble gate; no final training or upload")
    elif stage == "export":
        assert json.loads((run/"gate.json").read_text())["passed"]
        model_path = run/"final/model.pkl"
        model = joblib.load(model_path)
        assert model["feature_schema"] == R.B.SCHEMA_VERSION
        assert model["recipe"]["source_months"] == list(range(1,13))
        assert model["merkmale_haupt"] == R.B.main_features(R.H.MERKMALE_HAUPT)
        frame = R.dataset(run, project, ranking=True)
        prediction = R.predict(model, frame)
        template = pd.read_parquet(project/"data/submitting.parquet")
        assert template.MVT_ID_mvt.is_unique and set(template.MVT_ID_mvt) == set(frame.MVT_ID_mvt)
        out = template[["MVT_ID_mvt"]].copy()
        out["TAXITIME_SEC_mvt"] = pd.Series(prediction, index=frame.MVT_ID_mvt).reindex(out.MVT_ID_mvt).to_numpy()
        out["TAXITIME_SEC_mvt"] = out.TAXITIME_SEC_mvt.astype(template.TAXITIME_SEC_mvt.dtype)
        assert not S.pruefe(out, template, prediction)
        assert np.isfinite(out.TAXITIME_SEC_mvt).all()
        old = pd.read_parquet(manifest["legacy_submission"]).set_index("MVT_ID_mvt")
        fallback_ids = frame.loc[frame.hat_aobt.eq(0), "MVT_ID_mvt"]
        np.testing.assert_array_equal(out.set_index("MVT_ID_mvt").loc[fallback_ids, "TAXITIME_SEC_mvt"],
                                      old.loc[fallback_ids, "TAXITIME_SEC_mvt"])
        path = run/"candidate.parquet"
        if path.exists():
            # Resume a crash after export without overwriting a possibly different file.
            pd.testing.assert_frame_equal(pd.read_parquet(path), out, check_exact=True)
        else:
            temporary = path.with_name(path.name+".partial")
            out.to_parquet(temporary, index=False)
            temporary.replace(path)
        R.json_write(run/"ready_for_upload.json", {"candidate_path": str(path),
            "candidate_sha256": R.sha(path), "model_path": str(model_path), "model_sha256": R.sha(model_path),
            "rows": len(out), "self_checks_passed": True, "fallback_matches_scored_v44_exactly": True,
            "authorized_by_user": "Yes to full validation/retraining then one upload once checks pass, 2026-09-12 Europe/Berlin",
            "uploaded": False, "feature_schema": R.B.SCHEMA_VERSION,
            "next_step": "Choose an unused version after a fresh bucket listing, upload exactly once, retrieve official result; do not tune against feedback."})
    R.json_write(run/(stage+"_complete.json"), {"stage": stage, "finished_at": R.utc()})
    R.progress(run, stage, "Stage complete")


if __name__ == "__main__":
    main()

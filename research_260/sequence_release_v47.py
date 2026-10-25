"""v47 release candidate: origin-grouped neighbour sequence nets as the neural quarter of the frozen v45 main ensemble.

Written 14 September 2026 by Claude Code before any fit of this script, after the owner asked in chat to submit and to
decide the course of action. Nothing in this script uploads.

Evidence it rests on (not re-derived here):
  the team, research_260/results/seq_six_pair.json and seq_permutation_control_1_7.json: the origin-grouped SEQ net beats the
  matched flat net by 3.814 s pooled over six source-month pairs, 6/6 pairs, positive without the 100 most favourable rows;
  shuffled neighbours collapse the gain to 0.022 s.
  Codex, handoff of 14 September: in the frozen January/July ensemble with only the neural quarter replaced by the two-seed
  (13, 21) origin SEQ mean, overall RMSE 306.9022 -> 305.9076 (+0.9946 s), both months and all ten airports improve. The
  corrected local grouping scored 306.0782 and is not used. Codex's cache is outside this session's readable folders.
Reused unchanged: research_260/sequence_features.py (origin mode), sequence_model.py (the team's recipe),
sequence_ensemble_check.py (candidate-stage replay and substitution), research_250/release_common.py (frozen predict).

Stages (subcommands). Each writes into --output and resumes rather than overwriting finished products.
  validate-prepare   learning months 2-6 and 8-12, held-out 1 and 7: main rows, research encoding and origin tensors as in
                     research_260/airport_ablation.py prepare
  validate-fit       origin seed 13 on the learning months.
                     GATE V1: floored component RMSE on January/July within 0.5 s of the team's 224.5488 (Codex's gate), and
                     the saved package replays within 1e-3 s
  validate-baseline  sequence_ensemble_check.baseline: the candidate-stage ensemble replays its saved holdout within 1e-3 s
  validate-ensemble  the neural quarter replaced by the seed-13 SEQ residual. GATE V2: overall RMSE gain positive. Months,
                     airports and the top-100 check are reported; one seed is a sanity check of Codex's two-seed result
  release-prepare    all twelve 2025 months (the final ensemble's training rows) and the 2026 ranking main rows
  release-fit        origin seeds 13, 21, 29, 37, 45 (the frozen ensemble's neural seeds) on all twelve months, one after
                     another; each package and its raw ranking residuals are saved before the next seed starts
  build              GATE R1: release_common.predict with final/model.pkl reproduces the scored v45 file on every ranking
                     row (max 0.05 s, RMS 0.005 s, the release parity tolerance). The same call with only
                     neural_model.sage_voraus returning the five-seed mean raw SEQ residual gives the new main predictions.
                     v47 = the scored v46 file with its main rows replaced.
                     GATE R2: submission.pruefe passes, all values finite, fallback rows bit-identical to v46, only main
                     rows change. Writes candidate.parquet, build_report.json and ready_for_upload.json.
Not six-pair full-ensemble validation and not a leaderboard forecast.

Run from the repository root, one stage at a time (the runner in the output folder chains them):
  python3 -u -B research_260/sequence_release_v47.py <stage>
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import neural_model as NM  # noqa: F401  torch before tree libraries on macOS
import torch
import main_model as H
import joblib
import numpy as np
import pandas as pd
from research_250 import candidate_features as B
from research_260 import sequence_features as F, sequence_model as S
from research_260.airport_ablation import sha, write_json, log, source_paths

PROJECT = Path(__file__).resolve().parents[1]
FEATURES = B.main_features(H.MERKMALE_HAUPT)
MODE = "origin"
LEARN, HOLDOUT, ALL = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], [1, 7], list(range(1, 13))
RELEASE_SEEDS = [13, 21, 29, 37, 45]
KEVIN_ORIGIN_SEED13 = 224.54881227985408
RANKING_MAIN_ROWS, RANKING_FALLBACK_ROWS = 339551, 5290
AUTHORIZATION = ("User in chat, 14 September 2026: 'lets submit and see if we get any better rmse. codex worked in the folder "
                 "New Project in documents - skenderbeu selector - it finished the caches : check them out and decide for "
                 "yourself what the best course of action is.'")


def require_mps():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing a silent CPU fallback")


def dump(obj, path):
    partial = path.with_name(path.name + ".partial")
    joblib.dump(obj, partial)
    partial.replace(path)


def code_hashes():
    from research_250 import release_common as R
    from research_260 import airport_ablation as A, sequence_ensemble_check as E
    return {Path(m.__file__).name: sha(Path(m.__file__)) for m in (sys.modules[__name__], F, S, A, E, R, H, NM)}


def load_main(path, labelled):
    columns = list(dict.fromkeys(FEATURES + ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "hat_aobt", "y", "ersatz_s"]))
    frame = pd.read_parquet(path, columns=columns)
    keep = frame.hat_aobt.eq(1).to_numpy()
    if labelled:
        keep &= frame.y.notna().to_numpy()
    frame = frame[keep].reset_index(drop=True)
    if not frame.MVT_ID_mvt.is_unique or not np.isfinite(frame.ersatz_s.to_numpy(float)).all():
        raise ValueError(f"unexpected main rows in {path}")
    return frame


def encode(frame, prep=None):
    """Research encoding as airport_ablation.prepare: categories are the training rows' strings, unseen values missing."""
    for col in H.KAT:
        categories = pd.unique(frame[col].astype(str)) if prep is None else list(prep["karten"][col])
        frame[col] = pd.Categorical(frame[col].astype(str), categories=categories)
    return S.prepare(frame[FEATURES], FEATURES, H.KAT, prep)


def build_arrays(out, prefix, table, labelled, raw_paths, prep):
    frame = load_main(table, labelled)
    xk, xn, prep = encode(frame, prep)
    np.save(out / f"{prefix}_xk.npy", xk)
    np.save(out / f"{prefix}_xn.npy", xn)
    if labelled:
        np.save(out / f"{prefix}_residual.npy", (frame.y.to_numpy(float) - frame.ersatz_s.to_numpy(float)).astype(np.float32))
    rows = frame[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "y", "ersatz_s"]].copy()
    rows["ADEP_mvt"] = rows.ADEP_mvt.astype(str)
    rows.to_parquet(out / f"{prefix}_rows.parquet", index=False)
    ids = frame.MVT_ID_mvt.to_numpy()
    del frame, xk, xn
    gc.collect()
    log(f"neighbours {prefix} {MODE} for {len(ids)} departures")
    raw = pd.concat([pd.read_parquet(p, columns=F.RAW) for p in raw_paths], ignore_index=True)
    xs, ms = F.neighbours(raw, ids, MODE)
    del raw
    gc.collect()
    np.save(out / f"{prefix}_{MODE}_xs.npy", xs)
    np.save(out / f"{prefix}_{MODE}_ms.npy", ms)
    info = {"table": str(table), "table_sha256": sha(table), "raw_sha256": {Path(p).name: sha(p) for p in raw_paths},
            "rows": int(len(ids)), "mean_valid_neighbours": float(ms.sum(1).mean()), "no_neighbours": int((ms.sum(1) == 0).sum()),
            "tensors_finite": bool(np.isfinite(xs).all())}
    del xs, ms
    gc.collect()
    return prep, info


def validate_prepare(args):
    out = args.output / "validate"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        log("validation arrays already prepared")
        return
    data = args.release / "data"
    prep, train = build_arrays(out, "train", data / "months_2_3_4_5_6_8_9_10_11_12.parquet", True, source_paths(PROJECT, LEARN), None)
    dump(prep, out / "preprocessing.pkl")
    _, valid = build_arrays(out, "valid", data / "months_1_7.parquet", True, source_paths(PROJECT, HOLDOUT), prep)
    write_json(out / "manifest.json", {"mode": MODE, "schema": F.SCHEMA, "train": train, "valid": valid, "code_sha256": code_hashes()})
    log("PREPARED validation " + json.dumps({"train_rows": train["rows"], "valid_rows": valid["rows"],
                                             "valid_mean_neighbours": valid["mean_valid_neighbours"]}))


def validate_fit(args):
    out = args.output / "validate"
    name = f"validate_{MODE}_13"
    if (out / f"{name}.json").exists():
        log(f"{name} already complete")
        return
    require_mps()
    prep = joblib.load(out / "preprocessing.pkl")
    xk, xn, y = (np.load(out / f"train_{part}.npy") for part in ("xk", "xn", "residual"))
    xs, ms = np.load(out / f"train_{MODE}_xs.npy"), np.load(out / f"train_{MODE}_ms.npy")
    log(f"FIT {name} on MPS, {len(y)} rows")
    package = S.fit_net(True, 13, xk, xn, xs, ms, y, prep["kdim"], log=lambda message: log(f"{name} {message}"))
    package.update(preprocessing=prep, mode=MODE, feature_schema=F.SCHEMA)
    dump(package, out / f"{name}.pkl")
    del xk, xn, xs, ms, y
    gc.collect()
    torch.mps.empty_cache()
    vk, vn = np.load(out / "valid_xk.npy"), np.load(out / "valid_xn.npy")
    vs, vm = np.load(out / f"valid_{MODE}_xs.npy"), np.load(out / f"valid_{MODE}_ms.npy")
    residual = S.predict(package, vk, vn, vs, vm)
    n = min(4096, len(vk))
    replay = float(np.abs(S.predict(joblib.load(out / f"{name}.pkl"), vk[:n], vn[:n], vs[:n], vm[:n]) - residual[:n]).max())
    rows = pd.read_parquet(out / "valid_rows.parquet")
    rows["residual"] = residual
    rows.to_parquet(out / f"{name}_predictions.parquet", index=False)
    y_valid = rows.y.to_numpy(float)
    error = np.maximum(rows.ersatz_s.to_numpy(float) + residual, 300.) - y_valid
    month = pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    rmse = float(np.sqrt(np.mean(error ** 2)))
    gate = {"reference_kevin_origin_seed13": KEVIN_ORIGIN_SEED13, "difference_s": rmse - KEVIN_ORIGIN_SEED13,
            "saved_replay_max_abs_s": replay, "passed": bool(abs(rmse - KEVIN_ORIGIN_SEED13) <= .5 and replay <= 1e-3)}
    result = {"mode": MODE, "seed": 13, "rows": int(len(rows)), "floored_component_rmse": rmse,
              "by_month_rmse": {str(m): float(np.sqrt(np.mean(error[month == m] ** 2))) for m in np.unique(month)},
              "seconds": package["seconds"], "model_sha256": sha(out / f"{name}.pkl"), "gate_v1": gate}
    write_json(out / f"{name}.json", result)
    log("RESULT " + json.dumps(result))
    if not gate["passed"]:
        raise SystemExit("GATE V1 failed: component RMSE does not reproduce the team's origin seed 13")


def validate_baseline(args):
    from research_260 import sequence_ensemble_check as E
    out = args.output / "validate"
    out.mkdir(parents=True, exist_ok=True)
    E.baseline(SimpleNamespace(release=args.release, fallback=args.fallback, output=out))


def validate_ensemble(args):
    from research_260 import sequence_ensemble_check as E
    out = args.output / "validate"
    target = out / "validate_ensemble.json"
    if target.exists():
        log("validate-ensemble already complete")
        return
    info = json.loads((out / "ensemble_baseline.json").read_text())
    if sha(out / "ensemble_baseline.parquet") != info["baseline_sha256"]:
        raise ValueError("ensemble baseline changed")
    rows = pd.read_parquet(out / "ensemble_baseline.parquet")
    prediction = pd.read_parquet(out / f"validate_{MODE}_13_predictions.parquet").set_index("MVT_ID_mvt")
    if set(prediction.index) != set(rows.MVT_ID_mvt) or not prediction.index.is_unique:
        raise ValueError("prediction IDs do not match the ensemble baseline")
    np.testing.assert_allclose(prediction.loc[rows.MVT_ID_mvt, "y"], rows.y, atol=0, rtol=0)
    residual = prediction.loc[rows.MVT_ID_mvt, "residual"].to_numpy()
    candidate = E.postprocess(rows.ersatz_s.to_numpy(), rows.non_neural_residual.to_numpy(), residual, rows.gate.to_numpy(),
                              rows.schedule.to_numpy())
    metrics = E.metrics(rows, rows.baseline.to_numpy(), candidate, info)
    report = {"note": "January/July candidate-stage ensemble, neural quarter replaced by one origin SEQ seed; not a leaderboard score",
              "baseline": {k: info[k] for k in ("main_rmse", "overall_rmse", "main_rows", "fallback_rows", "replay_max_abs_s")},
              "seed13_substitution": metrics,
              "codex_two_seed_reported": {"baseline_overall_rmse": 306.9022, "overall_rmse": 305.9076, "gain_s": 0.9946},
              "gate_v2": {"overall_rmse_gain_positive": bool(metrics["overall_rmse_gain"] > 0)}}
    report["gate_v2"]["passed"] = report["gate_v2"]["overall_rmse_gain_positive"]
    write_json(target, report)
    log("RESULT " + json.dumps({k: metrics[k] for k in ("main_rmse", "main_rmse_gain", "overall_rmse", "overall_rmse_gain",
                                                         "mse_gain_without_top100", "by_month_mse_gain", "airports_positive")}))
    if not report["gate_v2"]["passed"]:
        raise SystemExit("GATE V2 failed: one origin seed does not improve the frozen January/July ensemble")


def release_prepare(args):
    out = args.output / "release"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        log("release arrays already prepared")
        return
    final = json.loads((args.release / "final/model.json").read_text())
    if final["recipe"]["source_months"] != ALL:
        raise ValueError("final model is not the all-year fit")
    data = args.release / "data"
    prep, train = build_arrays(out, "train", data / "months_1_2_3_4_5_6_7_8_9_10_11_12.parquet", True, source_paths(PROJECT, ALL), None)
    dump(prep, out / "preprocessing.pkl")
    _, ranking = build_arrays(out, "ranking", data / "ranking.parquet", False, [PROJECT / "data/ranking.parquet"], prep)
    if ranking["rows"] != RANKING_MAIN_ROWS:
        raise ValueError(f"expected {RANKING_MAIN_ROWS} ranking main rows, found {ranking['rows']}")
    write_json(out / "manifest.json", {"mode": MODE, "schema": F.SCHEMA, "train": train, "ranking": ranking,
                                       "final_model_training_rows_all_lanes": final["training_rows"], "code_sha256": code_hashes()})
    log("PREPARED release " + json.dumps({"train_rows": train["rows"], "ranking_rows": ranking["rows"],
                                          "ranking_mean_neighbours": ranking["mean_valid_neighbours"]}))


def release_fit(args):
    out = args.output / "release"
    require_mps()
    prep = joblib.load(out / "preprocessing.pkl")
    train = ranking = None
    for seed in RELEASE_SEEDS:
        name = f"release_{MODE}_{seed}"
        if (out / f"{name}.json").exists():
            log(f"{name} already complete")
            continue
        if train is None:
            train = [np.load(out / f"train_{part}.npy") for part in ("xk", "xn")] + [np.load(out / "train_residual.npy"),
                     np.load(out / f"train_{MODE}_xs.npy"), np.load(out / f"train_{MODE}_ms.npy")]
            ranking = [np.load(out / "ranking_xk.npy"), np.load(out / "ranking_xn.npy"),
                       np.load(out / f"ranking_{MODE}_xs.npy"), np.load(out / f"ranking_{MODE}_ms.npy")]
        xk, xn, y, xs, ms = train
        log(f"FIT {name} on MPS, {len(y)} rows")
        package = S.fit_net(True, seed, xk, xn, xs, ms, y, prep["kdim"], log=lambda message, name=name: log(f"{name} {message}"))
        package.update(preprocessing=prep, mode=MODE, feature_schema=F.SCHEMA)
        dump(package, out / f"{name}.pkl")
        torch.mps.empty_cache()
        residual = S.predict(package, *ranking)
        np.save(out / f"{name}_ranking_residual.npy", residual.astype(np.float64))
        n = 4096
        replay = float(np.abs(S.predict(joblib.load(out / f"{name}.pkl"), *(a[:n] for a in ranking)) - residual[:n]).max())
        if replay > 1e-3:
            raise ValueError(f"{name}: saved-package replay differs by {replay} s")
        result = {"mode": MODE, "seed": seed, "training_rows": int(len(y)), "seconds": package["seconds"],
                  "saved_replay_max_abs_s": replay, "model_sha256": sha(out / f"{name}.pkl"),
                  "ranking_residual_mean_s": float(residual.mean()), "ranking_residual_sd_s": float(residual.std())}
        write_json(out / f"{name}.json", result)
        log("RESULT " + json.dumps(result))
        del package
        gc.collect()
    residuals = np.array([np.load(out / f"release_{MODE}_{seed}_ranking_residual.npy") for seed in RELEASE_SEEDS])
    correlations = np.corrcoef(residuals)
    write_json(out / "release_fit_summary.json", {"seeds": RELEASE_SEEDS,
               "pairwise_ranking_residual_correlation_min": float(correlations[np.triu_indices(len(RELEASE_SEEDS), 1)].min()),
               "pairwise_ranking_residual_correlation_mean": float(correlations[np.triu_indices(len(RELEASE_SEEDS), 1)].mean())})
    log("RESULT release fits complete")


def build(args):
    import submission as SUB
    from research_250 import release_common as R
    out = args.output / "release"
    target = out / "candidate.parquet"
    m46 = json.loads((PROJECT / "results/upbeat-goblin_v46_manifest.json").read_text())
    v45_path, v46_path, model_path = PROJECT / m46["base_submission_path"], PROJECT / m46["submission_path"], PROJECT / m46["main_model_path"]
    hashes = {"v45_submission": sha(v45_path) == m46["base_submission_sha256"], "v46_submission": sha(v46_path) == m46["submission_sha256"],
              "final_model": sha(model_path) == m46["main_model_sha256"]}
    if not all(hashes.values()):
        raise ValueError(f"input hashes differ from the v46 manifest: {hashes}")
    model = joblib.load(model_path)
    if (model["feature_schema"] != B.SCHEMA_VERSION or model["recipe"]["source_months"] != ALL
            or model["merkmale_haupt"] != FEATURES or model.get("anteil_netz") != .25):
        raise ValueError("final model does not match the frozen v45 release recipe")
    frame = pd.read_parquet(args.release / "data/ranking.parquet")
    hat = frame.hat_aobt.eq(1).to_numpy()
    ids = frame.MVT_ID_mvt.to_numpy()
    if int(hat.sum()) != RANKING_MAIN_ROWS or int((~hat).sum()) != RANKING_FALLBACK_ROWS or not frame.MVT_ID_mvt.is_unique:
        raise ValueError("unexpected ranking rows")
    v45 = pd.read_parquet(v45_path).set_index("MVT_ID_mvt", verify_integrity=True)
    reference = v45.loc[ids, "TAXITIME_SEC_mvt"].to_numpy(float)
    log("GATE R1 replay of the frozen final ensemble on the ranking rows")
    replay = R.predict(model, frame)
    difference = replay - reference
    gate_r1 = {"max_abs_s": float(np.abs(difference).max()), "rms_s": float(np.sqrt(np.mean(difference ** 2))),
               "main_max_abs_s": float(np.abs(difference[hat]).max()), "max_tolerance_s": .05, "rms_tolerance_s": .005}
    gate_r1["passed"] = bool(gate_r1["max_abs_s"] <= .05 and gate_r1["rms_s"] <= .005)
    log("GATE R1 " + json.dumps(gate_r1))
    if not gate_r1["passed"]:
        raise SystemExit("GATE R1 failed: the frozen final ensemble does not reproduce v45")
    rows = pd.read_parquet(out / "ranking_rows.parquet")
    residuals = np.array([np.load(out / f"release_{MODE}_{seed}_ranking_residual.npy") for seed in RELEASE_SEEDS])
    sequence = pd.Series(residuals.mean(axis=0), index=rows.MVT_ID_mvt.to_numpy())
    main_ids, fallback_ids = ids[hat], ids[~hat]
    if set(sequence.index) != set(main_ids):
        raise ValueError("sequence residual IDs differ from the ranking main rows")
    original, captured = R.NM.sage_voraus, {}

    def substitute(net, x):
        wanted = frame.loc[x.index, "MVT_ID_mvt"].to_numpy()
        captured["flat"], captured["ids"] = original(net, x), wanted
        return sequence.loc[wanted].to_numpy()

    R.NM.sage_voraus = substitute
    try:
        log("substituting the neural quarter")
        new = R.predict(model, frame)
    finally:
        R.NM.sage_voraus = original
    if len(captured.get("ids", [])) != RANKING_MAIN_ROWS:
        raise ValueError("neural component was not substituted exactly once on all main rows")
    new_by_id = pd.Series(new, index=ids)
    v46 = pd.read_parquet(v46_path)
    template = pd.read_parquet(PROJECT / "data/submitting.parquet")
    candidate = v46.copy()
    values = candidate.TAXITIME_SEC_mvt.to_numpy().copy()
    position = pd.Index(candidate.MVT_ID_mvt).get_indexer(main_ids)
    if (position < 0).any() or len(np.unique(position)) != len(main_ids):
        raise ValueError("main IDs missing from the v46 file")
    values[position] = new_by_id.loc[main_ids].to_numpy()
    candidate["TAXITIME_SEC_mvt"] = pd.Series(values).astype(template.TAXITIME_SEC_mvt.dtype).to_numpy()
    errors = SUB.pruefe(candidate, template, None)
    v46i, candidatei = v46.set_index("MVT_ID_mvt"), candidate.set_index("MVT_ID_mvt")
    changed = candidate.TAXITIME_SEC_mvt.to_numpy() != v46.TAXITIME_SEC_mvt.to_numpy()
    gate_r2 = {"pruefe_errors": [str(e) for e in errors], "all_finite": bool(np.isfinite(candidate.TAXITIME_SEC_mvt.to_numpy(float)).all()),
               "fallback_identical_to_v46": bool(np.array_equal(candidatei.loc[fallback_ids, "TAXITIME_SEC_mvt"].to_numpy(),
                                                                v46i.loc[fallback_ids, "TAXITIME_SEC_mvt"].to_numpy())),
               "only_main_rows_change": bool(set(candidate.MVT_ID_mvt[changed]) <= set(main_ids)),
               "v46_main_rows_equal_v45": bool(np.array_equal(v46i.loc[main_ids, "TAXITIME_SEC_mvt"].to_numpy(float), reference[hat])),
               "rows": int(len(candidate)), "rows_changed": int(changed.sum())}
    gate_r2["passed"] = bool(not errors and gate_r2["all_finite"] and gate_r2["fallback_identical_to_v46"]
                             and gate_r2["only_main_rows_change"] and gate_r2["v46_main_rows_equal_v45"])
    log("GATE R2 " + json.dumps(gate_r2))
    if not gate_r2["passed"]:
        raise SystemExit("GATE R2 failed: candidate file checks")
    delta = new[hat] - reference[hat]
    airports = frame.loc[hat, "ADEP_mvt"].astype(str).to_numpy()
    flat = captured["flat"]
    seq_main = sequence.loc[captured["ids"]].to_numpy()
    report = {"note": "label-free change report on the 2026 ranking rows; prediction changes, not error changes",
              "input_hashes_match_v46_manifest": hashes, "gate_r1": gate_r1, "gate_r2": gate_r2,
              "delta_vs_v46_main_s": {"mean": float(delta.mean()), "mean_abs": float(np.abs(delta).mean()),
                                      "p01": float(np.quantile(delta, .01)), "p50": float(np.median(delta)), "p99": float(np.quantile(delta, .99)),
                                      "share_abs_over_60s": float(np.mean(np.abs(delta) > 60)), "share_abs_over_300s": float(np.mean(np.abs(delta) > 300))},
              "by_airport": {a: {"rows": int((airports == a).sum()), "mean_delta_s": float(delta[airports == a].mean()),
                                 "mean_abs_delta_s": float(np.abs(delta[airports == a]).mean())} for a in sorted(set(airports))},
              "sequence_vs_frozen_flat_residual": {"correlation": float(np.corrcoef(seq_main, flat)[0, 1]),
                                                   "mean_sequence_s": float(seq_main.mean()), "mean_flat_s": float(flat.mean()),
                                                   "quarter_of_mean_abs_difference_s": float(.25 * np.abs(seq_main - flat).mean())},
              "seeds": RELEASE_SEEDS, "release_models_sha256": {s: sha(out / f"release_{MODE}_{s}.pkl") for s in RELEASE_SEEDS}}
    if target.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(target), candidate, check_exact=True)
    else:
        partial = target.with_name(target.name + ".partial")
        candidate.to_parquet(partial, index=False)
        partial.replace(target)
    write_json(out / "build_report.json", report)
    validate = args.output / "validate"
    ready = {"version": "v47", "candidate_path": str(target), "candidate_sha256": sha(target), "rows": int(len(candidate)),
             "change": ("Main lane only: the neural quarter of the frozen v45 all-year ensemble is replaced by the mean raw residual of five "
                        "origin-grouped neighbour sequence nets (research_260, the team's recipe, seeds 13/21/29/37/45) trained on all twelve "
                        "2025 months; trees, CatBoost, airport models, reporting gate, floor and blend weights unchanged. Fallback rows are "
                        "the scored v46 predictions."),
             "base_submission_path": str(v46_path), "base_submission_sha256": m46["submission_sha256"],
             "validation": {"gate_v1": json.loads((validate / f"validate_{MODE}_13.json").read_text())["gate_v1"],
                            "gate_v2": json.loads((validate / "validate_ensemble.json").read_text())},
             "build_report": str(out / "build_report.json"), "authorized_by_user": AUTHORIZATION,
             "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(out / "ready_for_upload.json", ready)
    log("SUMMARY " + json.dumps({"candidate_sha256": ready["candidate_sha256"], "rows_changed": gate_r2["rows_changed"],
                                 "delta_vs_v46_main_s": report["delta_vs_v46_main_s"],
                                 "sequence_vs_frozen_flat_residual": report["sequence_vs_frozen_flat_residual"]}))


STAGES = {"validate-prepare": validate_prepare, "validate-fit": validate_fit, "validate-baseline": validate_baseline,
          "validate-ensemble": validate_ensemble, "release-prepare": release_prepare, "release-fit": release_fit, "build": build}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--release", type=Path, default=PROJECT / ".cache/clock_queue_release_20260912")
    parser.add_argument("--fallback", type=Path, default=PROJECT / ".cache/research_240/fallback_anchor_production/predictions_1_7.parquet")
    parser.add_argument("--output", type=Path, default=PROJECT / ".cache/sequence_release_v47")
    args = parser.parse_args()
    args.release, args.fallback, args.output = args.release.resolve(), args.fallback.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()

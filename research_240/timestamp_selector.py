"""Pre-registered 13 September 2026: direct-MSE multi-clock fallback selector.

New relative to rejected R9: a regularized softmax combines the scored v46 baseline,
its normal/high branches, schedule-copy, neighbour-taxi, previous-stand-arrival and
calendar-day alternatives, rather than a two-branch gate on the old normal model.
No promise of improvement. This is a fallback-only screen, not a main-lane redesign.
AOBT/EOBT are absent in this lane; do not invent them or use departure BLOCK/TAXITIME.

Before first fit, fixed protocol: six outer source-month pairs (1,7)..(6,12).
For each outer training table, five inner month-pair fits supply honest baseline
and branch predictions to the selector. No outer validation labels enter any fit,
stand rate, scaler or category dictionary. Existing label-free contextual features
retain their source-split construction. Preserve production categorical universes.
Fit one model at a time. Baseline must reproduce saved prod_anchor within 1e-6 s.

Two pre-specified linear softmax gates: ridge 0.1 and 1.0, offsets -8 relative to
baseline, direct untrimmed squared error divided by 1200**2, max 300 L-BFGS steps.
Use finite nonnegative alternatives <=172800 s (baseline always retained); no target
clipping. Fixed 20% and 100% application strengths, no post-hoc blend tuning.
Pass only if pooled MSE gain >0, >=4/6 pairs positive, January and July both positive,
gain survives deleting the 100 most favourable rows, and beats all five within-pair,
airport context permutations (seeds 101..505). Required features must exist in the
2026 ranking cache; candidate availability may differ by <=5 percentage points.
Any passing gate still needs independent release checks; no upload is authorized.
If all fail, stop this selector family and retain v46. Oracle numbers are hindsight
headroom diagnostics, never attainable predictions or leaderboard forecasts.

Inputs: cached source tables from fallback_rescreen_production.py; saved baseline
predictions from fallback_anchor_production.py; ranking features from the v46 anchor
preview. Run from this checkout, passing --cache-project for a separate Mac cache.
All row-level artifacts and fitted coefficients are saved ONLY to ignored .cache/.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

PAIRS = [(m, m+6) for m in range(1, 7)]
RIDGES = (0.1, 1.0)
STRENGTHS = (0.2, 1.0)
SEEDS = (101, 202, 303, 404, 505)
SCALE = 1200.0
CANDIDATES = ("baseline", "normal", "high", "schedule", "neighbour", "arrival",
              "neighbour_plus_day", "schedule_minus_day", "schedule_plus_day")
NUMERIC = ("versp_min", "stunde", "minute_tag", "wtag", "tag_versch", "sched_sek",
           "um_seit_ank", "um_ver_ank", "um_versp_ank", "nt_anchor", "nt_n60", "nt_n180",
           "id_abw", "id_abw2", "ank_med_30min", "ank_med_90min", "lage_50", "lage_200",
           "strecke_m")


def candidate_matrix(frame, parts):
    """Explicit ranking-time allowlist; labels and departure block times unread."""
    delay = frame.versp_min.to_numpy(float)*60
    nt = frame.nt_anchor.to_numpy(float)
    c = np.column_stack([parts.baseline, parts.normal, parts.high, delay, nt,
                         frame.um_seit_ank, nt+86400, delay-86400, delay+86400])
    mask = np.isfinite(c) & (c >= 0) & (c <= 172800)
    if not np.isfinite(c[:, 0]).all() or (c[:, 0] < 0).any():
        raise ValueError("invalid production baseline")
    mask[:, 0] = True
    return np.where(mask, c, 0.0), mask


def context(frame, parts):
    x = frame[list(NUMERIC)].astype(float).copy()
    for name in ("p", "normal", "high", "baseline", "q_stand", "q_stand_n"):
        x[name] = parts[name].to_numpy(float)
    # Signed log compresses seconds and sparse count extremes without dropping rows.
    x = np.sign(x)*np.log1p(np.abs(x))
    _, mask = candidate_matrix(frame, parts)
    for i, name in enumerate(CANDIDATES[1:], 1):
        x["available_"+name] = mask[:, i].astype(float)
    return x.to_numpy(float)


class Design:
    """Train-only imputation/scaling and airport dictionary, no raw identifiers."""
    def fit(self, frame, parts):
        raw = context(frame, parts)
        self.mean = np.nanmean(raw, axis=0)
        self.mean = np.nan_to_num(self.mean)
        filled = np.where(np.isfinite(raw), raw, self.mean)
        self.std = np.maximum(filled.std(axis=0), 0.1)
        self.airports = sorted(frame.ADEP_mvt.astype(str).unique())
        return self

    def transform(self, frame, parts):
        raw = context(frame, parts)
        numeric = np.clip((np.where(np.isfinite(raw), raw, self.mean)-self.mean)/self.std, -8, 8)
        cats = np.column_stack([(frame.ADEP_mvt.astype(str) == a).to_numpy(float) for a in self.airports])
        # Airport-specific slopes for delay, date crossing and the OOF extreme probability.
        indices = [0, 4, len(NUMERIC)]
        interactions = (cats[:, :, None]*numeric[:, None, indices]).reshape(len(frame), -1)
        return np.column_stack([np.ones(len(frame)), numeric, ~np.isfinite(raw), cats, interactions])


def weights(theta, x, mask):
    logits = np.column_stack([np.zeros(len(x)), -8+x@theta])
    logits[~mask] = -np.inf
    return softmax(logits, axis=1)


def loss_gradient(flat, x, c, mask, y, ridge):
    theta = flat.reshape(x.shape[1], c.shape[1]-1)
    w = weights(theta, x, mask)
    prediction = (w*c).sum(axis=1)
    residual = prediction-y
    loss = np.mean(residual**2)/SCALE**2 + ridge*np.sum(theta**2)
    dz = (2/SCALE**2/len(y))*residual[:, None]*w[:, 1:]*(c[:, 1:]-prediction[:, None])
    grad = x.T@dz + 2*ridge*theta
    return float(loss), grad.ravel()


def fit_gate(x, c, mask, y, ridge):
    start = np.zeros(x.shape[1]*(c.shape[1]-1))
    result = minimize(loss_gradient, start, args=(x, c, mask, y, ridge), jac=True,
                      method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-6})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError("nonfinite selector fit")
    return result.x.reshape(x.shape[1], c.shape[1]-1), {
        "success": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "objective": float(result.fun)}


def metrics(frame, prediction):
    y, base = frame.y.to_numpy(float), frame.baseline.to_numpy(float)
    gain = (base-y)**2-(prediction-y)**2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[-min(100, len(gain)):]] = False
    folds = frame.fold.to_numpy()
    return {"rmse": float(np.sqrt(np.mean((prediction-y)**2))), "mse_gain": float(gain.mean()),
            "without_top100_mse_gain": float(gain[keep].mean()) if keep.any() else None,
            "positive_pairs": int(sum(gain[folds == f].mean() > 0 for f in np.unique(folds))),
            "pair_mse_gains": {f: float(gain[folds == f].mean()) for f in np.unique(folds)},
            "january_mse_gain": float(gain[frame.monat.eq(1)].mean()),
            "july_mse_gain": float(gain[frame.monat.eq(7)].mean())}


def outer_inner_masks(months):
    for pair in PAIRS:
        mask = np.isin(months, pair)
        if mask.any():
            if mask.all():
                raise ValueError("inner training set empty")
            yield pair, mask


def source_month_index(paths):
    """Source filename, not possibly shifted movement/schedule month, owns the split."""
    frames = []
    for path in paths:
        match = re.fullmatch(r"training_2025-(\d{2})-01_.*\.parquet", path.name)
        if not match:
            raise ValueError(f"unexpected source filename {path.name}")
        ids = pd.read_parquet(path, columns=["MVT_ID_mvt"])
        ids["source_month"] = int(match.group(1))
        frames.append(ids)
    index = pd.concat(frames, ignore_index=True).set_index("MVT_ID_mvt").source_month
    if not index.index.is_unique or set(index.unique()) != set(range(1,13)):
        raise ValueError("source IDs duplicate or source months incomplete")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(".cache/research_240/timestamp_selector"))
    parser.add_argument("--report", type=Path, default=Path("research_240/results/timestamp_selector.json"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    # Torch must initialize before LightGBM on this Mac. Only sequential tree fits.
    import main_model as H
    import features_stand_rate as SR
    import lightgbm as lgb
    from research_240 import features_neighbour_taxi as NT
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=args.jobs)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache_project.resolve()/".cache/research_240"
    sidecars = list((cache/"rescreen_production").glob("rescreen_*.json"))
    index = {tuple(sorted(json.loads(p.read_text())["source_months"])): p.with_suffix(".parquet") for p in sidecars}
    required = set(PAIRS) | {tuple(m for m in range(1, 13) if m not in pair) for pair in PAIRS}
    if not required.issubset(index):
        raise FileNotFoundError("Run fallback_rescreen_production.py to create six-pair cached tables first")
    rank_paths = sorted((cache/"anchor_preview").glob("fallback_2026_ranking_*.parquet"))
    if len(rank_paths) != 1:
        raise ValueError("Expected exactly one ranking feature table")
    ranking = pd.read_parquet(rank_paths[0])
    if not set(NUMERIC).issubset(ranking.columns):
        raise ValueError("ranking feature contract missing")
    inputs = list(index[k] for k in sorted(required)) + rank_paths
    inputs += sorted((cache/"fallback_anchor_production").glob("predictions_*.parquet"))
    source_files = sorted(Path("data").glob("training_2025-*.parquet"))
    source_index = source_month_index(source_files)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"start_commit": "d48604963f2c6191c79f533318c2c0f3c8c904db",
                "script_sha256": sha(Path(__file__)),
                "input_sha256": {str(p.relative_to(args.cache_project.resolve())): sha(p) for p in inputs},
                "production_code_sha256": {p: sha(Path(p)) for p in ("main_model.py", "features_stand_rate.py", "research_240/features_neighbour_taxi.py")},
                "source_data_sha256": {str(p):sha(p) for p in source_files},
                "variants": {"ridges": RIDGES, "strengths": STRENGTHS}, "started_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    manifest_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    # No silent checkpoint reuse with a changed script or input.
    if (output/"manifest.json").exists():
        raise FileExistsError("Choose an unused output directory; no implicit resume")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    base_features = list(H.MERKMALE_RUECK)

    def reg(rounds, leaves, minimum):
        return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05,
                                num_leaves=leaves, min_child_samples=minimum, verbose=-1,
                                n_jobs=args.jobs, random_state=0)

    def parts(train, valid):
        if not set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt)):
            raise ValueError("overlapping training and prediction IDs")
        tr = train.copy()
        tr["z"] = (tr.y > H.SCHWELLE_EXTREM).astype(int)
        va = SR.wende_an(SR.passe_an(tr), valid)
        tr = SR.ausser_faltung(tr)
        normal_rows = tr.z.eq(0).to_numpy()
        y = tr.y.to_numpy(float)
        classifier = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05,
                                       num_leaves=31, min_child_samples=20, verbose=-1,
                                       n_jobs=args.jobs, random_state=0)
        p = classifier.fit(tr[base_features], tr.z).predict_proba(va[base_features])[:, 1]
        high = .7*va.versp_min.to_numpy(float)*60+.3*reg(300,15,10).fit(tr.loc[~normal_rows,base_features], y[~normal_rows]).predict(va[base_features])
        normal = reg(400,31,20).fit(tr.loc[normal_rows,base_features], y[normal_rows]).predict(va[base_features])
        nttr, ntva = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
        useful, present = normal_rows & np.isfinite(nttr), np.isfinite(ntva)
        features = base_features+list(NT.MERKMALE)
        model = reg(400,31,20).fit(tr.loc[useful,features], y[useful]-nttr[useful])
        normal[present] = ntva[present]+model.predict(va.loc[present,features])
        baseline = np.maximum((1-p**2.5)*normal+p**2.5*high, 0)
        return pd.DataFrame({"baseline": baseline, "p": p, "normal": normal, "high": high,
                             "q_stand": va.q_stand.to_numpy(), "q_stand_n": va.q_stand_n.to_numpy()}, index=valid.index)

    frames, diagnostics, all_coverage, fits = [], [], [], []
    started = time.time()
    for pair in PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        tr = pd.read_parquet(index[tuple(m for m in range(1, 13) if m not in pair)]).reset_index(drop=True)
        va = pd.read_parquet(index[pair]).reset_index(drop=True)
        train_months = tr.MVT_ID_mvt.map(source_index)
        valid_months = va.MVT_ID_mvt.map(source_index)
        if train_months.isna().any() or valid_months.isna().any() or set(train_months) & set(pair) or set(valid_months) != set(pair):
            raise ValueError("source month mismatch")
        valid_parts = parts(tr, va)
        reference = pd.read_parquet(cache/"fallback_anchor_production"/f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[va.MVT_ID_mvt]
        parity = float(np.max(np.abs(valid_parts.baseline.to_numpy()-reference.prod_anchor.to_numpy())))
        if parity > 1e-6 or not np.array_equal(va.y.to_numpy(), reference.y.to_numpy()):
            raise ValueError(f"production parity failed {key}: {parity}")
        print(f"PARITY {key}: {parity:.3g} s; building nested OOF", flush=True)
        oof = pd.DataFrame(np.nan, columns=valid_parts.columns, index=tr.index)
        for inner_pair, hold in outer_inner_masks(train_months.to_numpy()):
            oof.loc[hold] = parts(tr.loc[~hold], tr.loc[hold])
            print(f"  INNER {key} exclude {inner_pair}: done", flush=True)
        if not np.isfinite(oof.to_numpy()).all():
            raise ValueError("incomplete inner OOF predictions")
        oof.assign(MVT_ID_mvt=tr.MVT_ID_mvt).to_parquet(output/f"inner_parts_{key}.parquet", index=False)
        valid_parts.assign(MVT_ID_mvt=va.MVT_ID_mvt).to_parquet(output/f"outer_parts_{key}.parquet", index=False)
        design = Design().fit(tr, oof)
        x, xv = design.transform(tr, oof), design.transform(va, valid_parts)
        c, mask = candidate_matrix(tr, oof)
        cv, mv = candidate_matrix(va, valid_parts)
        frame = va[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "monat", "y"]].copy()
        frame["source_month"] = valid_months.to_numpy()
        frame["fold"], frame["baseline"] = key, valid_parts.baseline.to_numpy()
        errors = np.where(mv, (cv-va.y.to_numpy()[:,None])**2, np.inf)
        # Convex hull oracle is the best a soft combination could ever attain.
        low = np.where(mv, cv, np.inf).min(axis=1)
        high = np.where(mv, cv, -np.inf).max(axis=1)
        diagnostics.append({"pair": key, "rows": len(va), "parity_max_s": parity,
                            "baseline_sse": float(((frame.baseline-frame.y)**2).sum()),
                            "best_single_oracle_sse": float(errors.min(axis=1).sum()),
                            "convex_oracle_sse": float(((np.clip(va.y,low,high)-va.y)**2).sum())})
        all_coverage.append(mv)
        for ridge in RIDGES:
            theta, fit = fit_gate(x,c,mask,tr.y.to_numpy(float),ridge)
            fit.update(pair=key, ridge=ridge)
            fits.append(fit)
            model_info = {"theta": theta.tolist(), "mean": design.mean.tolist(), "std": design.std.tolist(),
                          "airports": design.airports, "candidates": CANDIDATES, "manifest_id": manifest_id,
                          "fit": fit}
            (output/f"gate_{key}_ridge{ridge}.json").write_text(json.dumps(model_info))
            pred = (weights(theta,xv,mv)*cv).sum(axis=1)
            for strength in STRENGTHS:
                name = f"ridge{ridge}_blend{strength}"
                frame[name] = frame.baseline+strength*(pred-frame.baseline)
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    order = np.arange(len(va))
                    airport = va.ADEP_mvt.astype(str).to_numpy()
                    for a in np.unique(airport):
                        rows = np.flatnonzero(airport == a)
                        order[rows] = rng.permutation(rows)
                    permuted = (weights(theta,xv[order],mv)*cv).sum(axis=1)
                    frame[f"{name}_perm{seed}"] = frame.baseline+strength*(permuted-frame.baseline)
            print(f"  GATE {key} ridge={ridge} iterations={fit['iterations']} success={fit['success']}", flush=True)
        frame.to_parquet(output/f"predictions_{key}.parquet", index=False)
        frames.append(frame)
        print(f"PAIR_DONE {key} elapsed={time.time()-started:.1f}s", flush=True)
    pooled = pd.concat(frames, ignore_index=True)
    # Candidate availability needs no labels or trained ranking predictions.
    dummy = pd.DataFrame({"baseline": np.ones(len(ranking))*1200, "normal": np.ones(len(ranking))*1200,
                          "high": np.ones(len(ranking))*6000})
    _, rm = candidate_matrix(ranking,dummy)
    coverage = np.concatenate(all_coverage).mean(axis=0)
    coverage_report = {name: {"validation": float(coverage[i]), "ranking": float(rm[:,i].mean())}
                       for i,name in enumerate(CANDIDATES) if i >= 3}
    coverage_pass = all(abs(v["validation"]-v["ranking"]) <= .05 for v in coverage_report.values())
    report = {"manifest": manifest, "rows": len(pooled), "seconds": time.time()-started,
              "baseline_rmse": float(np.sqrt(np.mean((pooled.baseline-pooled.y)**2))),
              "diagnostics": diagnostics, "fits": fits, "candidate_coverage": coverage_report,
              "coverage_pass": coverage_pass, "variants": {}}
    for ridge in RIDGES:
        for strength in STRENGTHS:
            name = f"ridge{ridge}_blend{strength}"
            m = metrics(pooled, pooled[name].to_numpy())
            controls = {str(s): metrics(pooled, pooled[f"{name}_perm{s}"].to_numpy())["mse_gain"] for s in SEEDS}
            m["permutation_mse_gains"] = controls
            m["optimizer_converged"] = all(f["success"] for f in fits if f["ridge"] == ridge)
            m["passes_screen"] = bool(m["mse_gain"] > 0 and m["positive_pairs"] >= 4
                and m["without_top100_mse_gain"] > 0 and m["january_mse_gain"] > 0
                and m["july_mse_gain"] > 0 and m["mse_gain"] > max(controls.values())
                and coverage_pass and m["optimizer_converged"])
            report["variants"][name] = m
    report["decision"] = "ADVANCE_TO_INDEPENDENT_CHECKS" if any(m["passes_screen"] for m in report["variants"].values()) else "REJECT_KEEP_V46"
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    pooled.to_parquet(output/"pooled_predictions.parquet",index=False)
    print(json.dumps({k:report[k] for k in ("rows","seconds","baseline_rmse","variants","decision")},indent=2),flush=True)


if __name__ == "__main__":
    main()

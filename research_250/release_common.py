"""Checkpointed full-ensemble candidate; legacy production modules stay unchanged."""
import gc
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import main_model as H  # Loads torch before tree libraries.
import neural_model as NM
import features_stand_rate as SR
import joblib
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd

from research_250 import candidate_features as B

FALLBACK_KEYS = ["ohne_aobt", "ohne_aobt_klass", "ohne_aobt_hoehe", "anteil_hm",
                 "hoch", "standquote", "merkmale_rueck", "ziel_ohne_aobt"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def progress(run, stage, detail):
    json_write(Path(run)/"progress.json", {"stage": stage, "detail": detail, "updated_at": utc()})
    print(f"[{utc()}] {stage}: {detail}", flush=True)


def check_manifest(run, project):
    manifest = json.loads((Path(run)/"run.json").read_text())
    from research_250.release_pipeline import versions
    if versions() != manifest["package_versions"]:
        raise RuntimeError("Library versions changed; do not mix checkpoints")
    for relative, digest in manifest["code_sha256"].items():
        if sha(Path(project)/relative) != digest:
            raise RuntimeError(f"Code changed since run initialization: {relative}; do not mix checkpoints")
    for relative, source in manifest["input_files"].items():
        path = Path(project)/relative
        stat = path.stat()
        if stat.st_size != source["size"] or stat.st_mtime_ns != source["mtime_ns"]:
            if sha(path) != source["sha256"]:
                raise RuntimeError(f"Source input changed: {relative}")
    return manifest


def dataset(run, project, months=None, ranking=False):
    run, project = Path(run), Path(project)
    manifest = check_manifest(run, project)
    name = "ranking" if ranking else "months_"+"_".join(map(str, sorted(months)))
    path = run/"data"/(name+".parquet")
    if path.exists():
        return pd.read_parquet(path)
    progress(run, "features", f"Building independent source reference {name}")
    if ranking:
        files = [project/"data/ranking.parquet"]
    else:
        files = [project/manifest["training_files"][str(month)] for month in sorted(months)]
    frame = B.baue(files, H.MG.alle())
    keep = list(dict.fromkeys(H.MERKMALE_HAUPT+H.MERKMALE_RUECK+B.NEW_COLUMNS+
                             ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "y"]))
    frame = frame[[c for c in keep if c in frame]].copy()
    if not frame.MVT_ID_mvt.is_unique:
        raise ValueError("Duplicate departure IDs")
    if ranking:
        assert frame.y.isna().all()
    else:
        assert frame.y.notna().all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return frame


def split_check(train, valid):
    assert train.MVT_ID_mvt.is_unique and valid.MVT_ID_mvt.is_unique
    assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))


def checkpoint(run, stage, name, recipe, build):
    directory = Path(run)/stage/"checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(name+".pkl")
    expected = {"run_sha256": sha(Path(run)/"run.json"), "recipe": recipe, "name": name}
    if path.exists():
        saved = joblib.load(path)
        if saved["identity"] != expected:
            raise ValueError(f"Checkpoint identity mismatch: {path}")
        progress(run, stage, f"Reusing verified checkpoint {name}")
        return saved["model"]
    progress(run, stage, f"Training {name}")
    model = build()
    temporary = path.with_name(path.name+".partial")
    joblib.dump({"identity": expected, "model": model}, temporary)
    temporary.replace(path)
    progress(run, stage, f"Completed {name}")
    return model


def tree(rounds, seed, jobs):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.03,
        num_leaves=255, min_child_samples=50, subsample=.8, subsample_freq=1,
        colsample_bytree=.8, verbose=-1, n_jobs=jobs, random_state=seed)


def fallback_fit(frame, jobs):
    lo = frame.loc[frame.hat_aobt.eq(0)].copy()
    lo["z"] = (lo.y > H.SCHWELLE_EXTREM).astype(int)
    quote = SR.passe_an(lo)
    lo = SR.ausser_faltung(lo)
    features = H.MERKMALE_RUECK
    classifier = lgb.LGBMClassifier(objective="binary", n_estimators=600,
        learning_rate=.05, num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=jobs)
    classifier.fit(lo[features], lo.z)
    normal = lgb.LGBMRegressor(objective="l2", n_estimators=400, learning_rate=.05,
        num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=jobs)
    normal.fit(lo.loc[lo.z.eq(0), features], lo.loc[lo.z.eq(0), "y"])
    high = lgb.LGBMRegressor(objective="l2", n_estimators=300, learning_rate=.05,
        num_leaves=15, min_child_samples=10, verbose=-1, n_jobs=jobs)
    high.fit(lo.loc[lo.z.eq(1), features], lo.loc[lo.z.eq(1), "y"])
    return {"ohne_aobt": normal, "ohne_aobt_klass": classifier, "ohne_aobt_hoehe": high,
            "anteil_hm": H.ANTEIL_HM, "hoch": H.HOCH, "standquote": quote,
            "merkmale_rueck": list(features), "ziel_ohne_aobt": "mischung"}


def fallback_from(path):
    model = joblib.load(path)
    selected = {key: model[key] for key in FALLBACK_KEYS}
    assert selected["merkmale_rueck"] == H.MERKMALE_RUECK
    del model
    gc.collect()
    return selected


def fit_ensemble(frame, run, stage, recipe, fallback_source=None):
    main = frame.loc[frame.hat_aobt.eq(1)].copy()
    main["rest"] = main.y-main.ersatz_s
    features = B.main_features(H.MERKMALE_HAUPT) if recipe["candidate"] else list(H.MERKMALE_HAUPT)
    assert np.isfinite(main.rest).all()
    rounds, jobs = recipe["lgb_rounds"], recipe["jobs"]
    saved = lambda name, build: checkpoint(run, stage, name, recipe, build)
    # Every worker is a fresh process. All NN training happens before any tree fit.
    net = None
    for seed in NM.SAATEN:
        item = saved(f"neural_{seed}", lambda seed=seed: NM.lerne(main, features, H.KAT, saaten=(seed,)))
        if net is None:
            net = dict(item)
            net["zustaende"] = list(item["zustaende"])
        else:
            net["zustaende"].extend(item["zustaende"])
    main["erf"] = ((main.y-main.versp_min*60).abs() <= 10).astype(int)

    def detector():
        model = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=.05,
            num_leaves=127, min_child_samples=40, verbose=-1, n_jobs=jobs, random_state=0)
        return model.fit(main[features], main.erf)

    fabricated = saved("reporting_classifier", detector)
    global_trees = []
    for seed in H.SAATEN:
        global_trees.append(saved(f"lgb_{seed}", lambda seed=seed:
                                  tree(rounds, seed, jobs).fit(main[features], main.rest)))
    pool = Pool(H.kat_str(main[features]), main.rest, cat_features=H.KAT)
    cats = []
    for seed in H.CB_SAATEN:
        def fit_cat(seed=seed):
            model = CatBoostRegressor(loss_function="RMSE", iterations=H.CB_RUNDEN,
                learning_rate=.06, depth=8, verbose=500, thread_count=jobs,
                random_seed=seed, allow_writing_files=False)
            return model.fit(pool)
        cats.append(saved(f"catboost_{seed}", fit_cat))
    del pool
    airports = {}
    for airport in sorted(main.ADEP_mvt.astype(str).unique()):
        mask = main.ADEP_mvt.astype(str).eq(airport).to_numpy()
        airports[airport] = saved("airport_"+airport, lambda mask=mask:
                                 tree(rounds, 0, jobs).fit(main.loc[mask, features], main.loc[mask, "rest"]))
    back = saved("fallback", lambda: fallback_from(fallback_source) if fallback_source else fallback_fit(frame, jobs))
    model = {"mit_aobt": global_trees, "mit_aobt_cb": cats, "anteil_cb": H.ANTEIL_CB,
        "mit_aobt_netz": net, "anteil_netz": H.ANTEIL_NETZ, "erf_klass": fabricated,
        "erf_hoch": H.ERF_HOCH, "mit_aobt_flh": airports, "anteil_flh": H.ANTEIL_FLH,
        "merkmale_haupt": features, "ziel_mit_aobt": "rest",
        "feature_schema": B.SCHEMA_VERSION if recipe["candidate"] else "v44_legacy_v1",
        "recipe": recipe, "training_rows": len(frame), "run_sha256": sha(Path(run)/"run.json"), **back}
    assert len(model["mit_aobt"]) == 5 and len(model["mit_aobt_cb"]) == 3
    assert len(model["mit_aobt_netz"]["zustaende"]) == 5 and len(airports) == 10
    path = Path(run)/stage/"model.pkl"
    temporary = path.with_name("model.pkl.partial")
    joblib.dump(model, temporary)
    temporary.replace(path)
    json_write(path.with_suffix(".json"), {"model_sha256": sha(path), "recipe": recipe,
               "feature_schema": model["feature_schema"], "training_rows": len(frame)})
    return model


def predict(model, frame):
    """Same fixed blend/gate/floors as submission.py, without writing or uploading."""
    if model["ziel_mit_aobt"] != "rest" or model["ziel_ohne_aobt"] != "mischung":
        raise ValueError("Unsupported target definition")
    frame = SR.wende_an(model["standquote"], frame)
    hat = frame.hat_aobt.eq(1).to_numpy()
    output = np.empty(len(frame), float)
    x = frame.loc[hat, model["merkmale_haupt"]]
    trees = model["mit_aobt"]
    trees = trees if isinstance(trees, list) else [trees]
    residual = np.mean([tree.predict(x) for tree in trees], axis=0)
    airport_weight = model.get("anteil_flh", 0.)
    if airport_weight:
        local = residual.copy()
        codes = frame.loc[hat, "ADEP_mvt"].astype(str).to_numpy()
        for airport, tree in model["mit_aobt_flh"].items():
            mask = codes == airport
            if mask.any():
                local[mask] = tree.predict(x.loc[mask])
        residual = (1-airport_weight)*residual+airport_weight*local
    cat_weight = model.get("anteil_cb", 0.)
    if cat_weight:
        cats = model["mit_aobt_cb"]
        cats = cats if isinstance(cats, (list, tuple)) else [cats]
        cat_x = H.kat_str(x)
        cat_prediction = np.mean([tree.predict(cat_x) for tree in cats], axis=0)
        residual = (1-cat_weight)*residual+cat_weight*cat_prediction
    net_weight = model.get("anteil_netz", 0.)
    if net_weight and model.get("mit_aobt_netz") is not None:
        residual = (1-net_weight)*residual+net_weight*NM.sage_voraus(model["mit_aobt_netz"], x)
    output[hat] = frame.loc[hat, "ersatz_s"].to_numpy()+residual
    if model.get("erf_klass") is not None:
        p = model["erf_klass"].predict_proba(x)[:, 1]**model.get("erf_hoch", 2.5)
        output[hat] = (1-p)*output[hat]+p*frame.loc[hat, "versp_min"].to_numpy()*60
    if (~hat).any():
        x = frame.loc[~hat, model["merkmale_rueck"]]
        p = model["ohne_aobt_klass"].predict_proba(x)[:, 1]**model.get("hoch", 1.)
        weight = model.get("anteil_hm", 0.)
        high = (1-weight)*frame.loc[~hat, "versp_min"].to_numpy()*60
        if weight:
            high += weight*model["ohne_aobt_hoehe"].predict(x)
        output[~hat] = (1-p)*model["ohne_aobt"].predict(x)+p*high
    output = np.clip(output, 0, None)
    output[hat] = np.maximum(output[hat], 300)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite predictions")
    return output


def score(frame, prediction):
    error = (prediction-frame.y.to_numpy())**2
    hat = frame.hat_aobt.eq(1).to_numpy()
    return {"rows": len(frame), "rmse": float(np.sqrt(error.mean())), "mse": float(error.mean()),
            "main_rmse": float(np.sqrt(error[hat].mean())),
            "fallback_rmse": float(np.sqrt(error[~hat].mean())),
            "top5_sse_share": float(np.sort(error)[-5:].sum()/error.sum())}

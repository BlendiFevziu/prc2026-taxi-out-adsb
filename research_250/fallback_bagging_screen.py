"""Fixed bootstrap ensemble of the existing fallback recipe, without new features.

Six final predictions receive equal weight: the original fit and five bootstrap
replicates. An independent replace_traffic mode tests substituting the corrected
count for the legacy count, without bagging or additional predictors.
Stand encodings are fixed, training-only, and out-of-month for training
rows. This is learner bagging conditional on that encoding, not a bootstrap of the
whole feature pipeline. Bootstrap sampling is stratified by movement-time month and
extreme/non-extreme class. No held-out target selects weights or parameters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--train", type=Path, required=True)
parser.add_argument("--valid", type=Path, required=True)
parser.add_argument("--baseline-predictions", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--jobs", type=int, default=8)
parser.add_argument("--mode", choices=["bootstrap","replace_traffic"], default="bootstrap")
args = parser.parse_args()
for key in ("project","train","valid","baseline_predictions","output"):
    setattr(args,key,getattr(args,key).resolve())
os.environ.setdefault("MPLCONFIGDIR",str(args.output/"matplotlib"))
sys.path.insert(0,str(args.project))
os.chdir(args.project)
import main_model as H
import features_stand_rate as SQ
import lightgbm as lgb
import numpy as np
import pandas as pd

train, valid = pd.read_parquet(args.train), pd.read_parquet(args.valid)
assert set(train.MVT_ID_mvt).isdisjoint(set(valid.MVT_ID_mvt))
train["z"] = train.y.gt(H.SCHWELLE_EXTREM).astype(int)
mapping = SQ.passe_an(train)
train,valid = SQ.ausser_faltung(train),SQ.wende_an(mapping,valid)
for col in H.KAT:
    if col in train:
        cats = pd.unique(train[col].astype(str))
        train[col] = pd.Categorical(train[col].astype(str),categories=cats)
        valid[col] = pd.Categorical(valid[col].astype(str),categories=cats)
features = H.MERKMALE_RUECK


def fit(data):
    x,xv = data[features],valid[features]
    shared = dict(learning_rate=.05,n_jobs=args.jobs,random_state=0,verbose=-1)
    classifier = lgb.LGBMClassifier(objective="binary",n_estimators=600,
                     num_leaves=31,min_child_samples=20,**shared).fit(x,data.z)
    normal = lgb.LGBMRegressor(objective="l2",n_estimators=400,
                     num_leaves=31,min_child_samples=20,**shared).fit(x[data.z.eq(0)],data.loc[data.z.eq(0),"y"])
    height = lgb.LGBMRegressor(objective="l2",n_estimators=300,
                     num_leaves=15,min_child_samples=10,**shared).fit(x[data.z.eq(1)],data.loc[data.z.eq(1),"y"])
    q = classifier.predict_proba(xv)[:,1]**H.HOCH
    high = (1-H.ANTEIL_HM)*valid.versp_min.to_numpy()*60+H.ANTEIL_HM*height.predict(xv)
    return np.maximum((1-q)*normal.predict(xv)+q*high,0)


started = time.time()
out = valid[["MVT_ID_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","y"]].copy()
out["v44_fallback"] = fit(train)
prior = pd.read_parquet(args.baseline_predictions).set_index("MVT_ID_mvt")
assert set(prior.index) == set(out.MVT_ID_mvt)
np.testing.assert_allclose(out.v44_fallback.to_numpy(),prior.v44_fallback.reindex(out.MVT_ID_mvt).to_numpy(),rtol=1e-9,atol=1e-7)
groups = [np.asarray(idx) for idx in train.groupby(["monat","z"],observed=True).indices.values()]
seeds = [11,23,37,53,71] if args.mode == "bootstrap" else []
members = ["v44_fallback"]
for seed in seeds:
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
    rng.shuffle(indices)
    name = f"bootstrap_{seed}"
    print("FIT",name,flush=True)
    out[name] = fit(train.iloc[indices].reset_index(drop=True))
    members.append(name)
if args.mode == "bootstrap":
    candidate = "bootstrap_ensemble"
    out[candidate] = out[members].mean(axis=1)
else:
    candidate = "traffic_replaced"
    train["auslast"] = train.tu_auslast.to_numpy()
    valid["auslast"] = valid.tu_auslast.to_numpy()
    out[candidate] = fit(train)
report = {"validation":"fixed fallback-only temporal screen; not a leaderboard result",
          "mode":args.mode,"features":features,"bootstrap_seeds":seeds,
          "equal_weight_members":members if args.mode == "bootstrap" else [],
          "rows":len(valid),"seconds":time.time()-started,"scores":{}}
for name in ["v44_fallback",candidate]:
    e = (out[name]-out.y)**2
    report["scores"][name] = {"rmse":float(np.sqrt(e.mean())),"mse":float(e.mean()),
                               "top5_sse_share":float(e.nlargest(5).sum()/e.sum())}
report["mse_gain"] = report["scores"]["v44_fallback"]["mse"]-report["scores"][candidate]["mse"]
report["prespecified_fallback_mse_gain_bar"] = 100000. if args.mode == "bootstrap" else None
report["clears_screen_bar"] = report["mse_gain"] >= 100000. if args.mode == "bootstrap" else None
report["code_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__),Path(H.__file__),Path(SQ.__file__))}
args.output.mkdir(parents=True,exist_ok=True)
out.to_parquet(args.output/f"{args.mode}_predictions.parquet",index=False)
(args.output/f"{args.mode}_result.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)

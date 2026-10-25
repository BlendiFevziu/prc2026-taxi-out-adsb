"""Pre-registered 13 September 2026: main-lane conditional timestamp selection.

Stage 1 uses the frozen v45/v46 FULL main ensemble and five saved target-view
experts, all trained on source months 2-6 and 8-12. Fit the selector on source
January, evaluate July, then reverse. Thus neither the baseline nor the selector
has fitted any label from its evaluation month. January/July have informed earlier
research, so this is a screening test, not fresh six-pair release validation.

New versus main_target_views.py: per-flight, direct-MSE weighting of 14 candidates,
not a constant blend of a single extra tree. Candidates: full baseline, five
learned target views, four raw off-block clocks, their median, schedule-copy, and
AOBT +/- one day. No departure BLOCK, TAXITIME, raw absolute ID or target in inputs.
Candidates outside [0,172800] are unavailable; baseline always available; output
floor 300 unchanged. No label clipping or row removal based on error magnitude.

Fixed before any fit: linear softmax ridge .01, max 200 L-BFGS iterations; a 64-unit
ReLU neural softmax, seed 271828, AdamW lr .002, weight_decay .01, batch 2048,
60 epochs, gradient norm cap 10, CPU, four threads. Both optimize untrimmed MSE
in units of 300**2. Neural output layer starts at zero. Priors: full baseline 0,
five learned views -3, raw clocks/median -7, schedule -9, day shifts -16.
Application strengths .2 and 1 are fixed, not selected on evaluation scores.
Controls: unchanged baseline, untrained prior, and five within-(source month,
airport) permutations of selector context, leaving candidate values in place.

Advance beyond this cheap screen only if a variant gains >=0.5 s pooled OVERALL
local RMSE with the v46 fallback held unchanged; gains in January AND July;
survives deleting its best 100 rows; improves >=7/10 airports; beats both the fixed
prior and all five permutation controls; and raw-candidate availability differs
by <=5 percentage points on 2026 ranking. A pass is NOT a release authorization:
it requires fresh/nested multi-period validation before any candidate or upload.
If none passes, stop this registered screen without post-hoc tuning.

Inputs: release/data/months_1_7.parquet and candidate/holdout_predictions.parquet,
main_target_views.py's predictions_outer.parquet + result_outer.json,
fallback_anchor_production.py's predictions_1_7.parquet. Other cached months' IDs
verify disjoint base training. Source filenames own the month split. No full
ensemble fitting, external data, leaderboard calls, or submission generation.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

VIEWS = ("rest", "direct", "eobt", "sched", "median4")
CLOCKS = ("ersatz_s", "ersatz_eobt_s", "ersatz_lobt_s", "ersatz_iobt_s", "anker_median")
CANDIDATES = ("baseline",)+tuple("view_"+v for v in VIEWS)+CLOCKS+("schedule", "aobt_plus_day", "aobt_minus_day")
PRIORS = np.array([0]+[-3]*5+[-7]*5+[-9,-16,-16], dtype=float)
NUMERIC = ("stunde", "wtag", "versp_min", "tag_versch", "sched_sek", "strecke_m",
           "anker_spanne", "anker_streu", "anker_anz", "anker_abw", "kand_sched",
           "um_seit_ank", "um_ver_ank", "um_versp_ank", "ank_med_30min", "ank_med_90min",
           "fid_abw", "fid_abw2", "id_abw", "id_abw2", "tu_auslast", "qt_bahn_vor_10",
           "qt_bahn_nach_10", "qt_bahn_luecke_vor", "dt_aobt_3_eobt_1", "dt_aobt_3_lobt",
           "dt_aobt_3_iobt", "dt_aobt_3_arvt_3")
CATEGORICAL = ("ADEP_mvt", "RUNWAY_mvt")
STRENGTHS = (.2, 1.)
SEEDS = (101,202,303,404,505)
FLOOR, SCALE = 300., 300.


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def candidates(frame, base, views):
    """Explicit allowlist of ranking-time fields; labels are never read."""
    c = np.column_stack([base]+[views[v].to_numpy(float) for v in VIEWS]
                        +[frame[v].to_numpy(float) for v in CLOCKS]
                        +[frame.versp_min.to_numpy(float)*60,frame.ersatz_s.to_numpy(float)+86400,
                          frame.ersatz_s.to_numpy(float)-86400])
    if c.shape[1] != len(PRIORS) or not np.isfinite(c[:,0]).all() or (c[:,0] < FLOOR).any():
        raise ValueError("invalid baseline or candidate schema")
    valid = np.isfinite(c) & (c >= 0) & (c <= 172800)
    valid[:,0] = True
    return np.where(valid,c,0.), valid


def context(frame, c, mask):
    raw = frame[list(NUMERIC)].to_numpy(float)
    delta = np.where(mask[:,1:],c[:,1:]-c[:,0,None],np.nan)
    raw = np.column_stack([raw,c[:,0],delta])
    return np.sign(raw)*np.log1p(np.abs(raw))


class Design:
    def fit(self, frame, c, mask):
        raw = context(frame,c,mask)
        valid = np.isfinite(raw)
        self.mean = np.divide(np.where(valid,raw,0).sum(axis=0),valid.sum(axis=0),
                              out=np.zeros(raw.shape[1]),where=valid.sum(axis=0)>0)
        filled = np.where(valid,raw,self.mean)
        self.std = np.maximum(filled.std(axis=0),.1)
        self.categories = {col: sorted(frame[col].astype(str).value_counts().loc[lambda s:s>=100].index)
                           for col in CATEGORICAL}
        return self

    def transform(self, frame, c, mask):
        raw = context(frame,c,mask)
        numeric = np.clip((np.where(np.isfinite(raw),raw,self.mean)-self.mean)/self.std,-8,8)
        blocks = [np.ones((len(frame),1)),numeric,(~np.isfinite(raw)).astype(float)]
        for col in CATEGORICAL:
            cat = frame[col].astype(str).to_numpy()
            blocks.extend([(cat==v).astype(float)[:,None] for v in self.categories[col]])
        return np.column_stack(blocks)

    def state(self):
        return {"mean":self.mean.tolist(),"std":self.std.tolist(),"categories":self.categories}


def soft_weights(logits,mask):
    z = np.column_stack([np.zeros(len(logits)),logits])+PRIORS
    return softmax(np.where(mask,z,-np.inf),axis=1)


def predict_linear(theta,x,c,mask):
    w = soft_weights(x@theta,mask)
    return np.maximum((w*c).sum(axis=1),FLOOR)


def linear_loss(flat,x,c,mask,y,ridge=.01):
    theta = flat.reshape(x.shape[1],len(PRIORS)-1)
    w = soft_weights(x@theta,mask)
    raw = (w*c).sum(axis=1)
    residual = np.maximum(raw,FLOOR)-y
    loss = np.mean(residual**2)/SCALE**2+ridge*np.sum(theta**2)
    dz = (2/len(y)/SCALE**2)*residual[:,None]*(raw>FLOOR)[:,None]*w[:,1:]*(c[:,1:]-raw[:,None])
    grad = x.T@dz+2*ridge*theta
    return float(loss),grad.ravel()


def fit_linear(x,c,mask,y):
    started = time.time()
    iterations = [0]
    def progress(flat):
        iterations[0] += 1
        if iterations[0] % 20 == 0:
            print(f"LINEAR iteration={iterations[0]} seconds={time.time()-started:.1f}",flush=True)
    fit = minimize(linear_loss,np.zeros(x.shape[1]*(len(PRIORS)-1)),args=(x,c,mask,y),jac=True,
                   callback=progress,method="L-BFGS-B",options={"maxiter":200,"ftol":1e-9,"gtol":1e-6})
    if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        raise ValueError("nonfinite linear fit")
    return fit.x.reshape(x.shape[1],-1),{"success":bool(fit.success),"iterations":int(fit.nit),
        "message":str(fit.message),"training_objective":float(fit.fun),"seconds":time.time()-started}


def fit_neural(x,c,mask,y,epochs=60):
    import torch  # no tree imports in this script
    torch.manual_seed(271828)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,len(PRIORS)-1))
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    xx,cc,yy = [torch.as_tensor(v,dtype=torch.float32) for v in (x,c,y)]
    mm = torch.as_tensor(mask,dtype=torch.bool)
    priors = torch.tensor(PRIORS,dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    curves,started = [],time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(y))
        sse = 0.
        for idx in order.split(2048):
            logits = torch.cat([torch.zeros(len(idx),1),model(xx[idx])],dim=1)+priors
            w = torch.softmax(logits.masked_fill(~mm[idx],-torch.inf),dim=1)
            pred = torch.clamp((w*cc[idx]).sum(dim=1),min=FLOOR)
            loss = ((pred-yy[idx])/SCALE).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite neural loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
            sse += float(loss.detach())*len(idx)
        if (epoch+1)%10==0 or epoch==0:
            curves.append({"epoch":epoch+1,"training_scaled_mse":sse/len(y)})
            print(f"NEURAL epoch={epoch+1}/{epochs} training_loss={sse/len(y):.6f} seconds={time.time()-started:.1f}",flush=True)
    return model.eval(),{"success":True,"epochs":epochs,"training_curve":curves,"seconds":time.time()-started}


def predict_neural(model,x,c,mask):
    import torch
    result = np.empty(len(x),float)
    with torch.no_grad():
        for start in range(0,len(x),8192):
            stop = min(start+8192,len(x))
            logits = model(torch.as_tensor(x[start:stop],dtype=torch.float32)).numpy().astype(float)
            result[start:stop] = np.maximum((soft_weights(logits,mask[start:stop])*c[start:stop]).sum(axis=1),FLOOR)
    return result


def source_months(frame,paths):
    chunks = []
    for month,path in paths.items():
        ids = pd.read_parquet(path,columns=["MVT_ID_mvt"])
        ids["source_month"] = int(month)
        chunks.append(ids)
    mapping = pd.concat(chunks).set_index("MVT_ID_mvt",verify_integrity=True).source_month
    result = frame.MVT_ID_mvt.map(mapping)
    if result.isna().any():
        raise ValueError("prediction row has no source file")
    return result.to_numpy(int)


def summarize(frame,prediction,main_rows,total_rows,base_overall_mse):
    y,base = frame.y.to_numpy(float),frame.baseline.to_numpy(float)
    gain = (base-y)**2-(prediction-y)**2
    keep = np.ones(len(gain),bool)
    keep[np.argsort(gain)[-min(100,len(gain)):]] = False
    airport = frame.ADEP_mvt.astype(str).to_numpy()
    month = frame.source_month.to_numpy(int)
    airport_gains = {a:float(gain[airport==a].mean()) for a in np.unique(airport)}
    overall = np.sqrt(base_overall_mse-gain.mean()*main_rows/total_rows)
    return {"main_rmse":float(np.sqrt(np.mean((prediction-y)**2))),"main_mse_gain":float(gain.mean()),
        "overall_rmse":float(overall),"overall_rmse_gain":float(np.sqrt(base_overall_mse)-overall),
        "mse_gain_without_top100":float(gain[keep].mean()) if keep.any() else None,
        "month_mse_gains":{str(m):float(gain[month==m].mean()) for m in np.unique(month)},
        "airport_mse_gains":airport_gains,"positive_airports":int(sum(v>0 for v in airport_gains.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-project",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path(".cache/research_240/main_timestamp_selector"))
    parser.add_argument("--report",type=Path,default=Path("research_240/results/main_timestamp_selector.json"))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=4)
    if (args.output/"manifest.json").exists():
        raise FileExistsError("use a new output directory; no unverified checkpoint reuse")
    args.output.mkdir(parents=True,exist_ok=True)
    project = args.cache_project.resolve()
    release = project/".cache/clock_queue_release_20260912"
    research = project/".cache/research_240"
    data_file = release/"data/months_1_7.parquet"
    base_file = release/"candidate/holdout_predictions.parquet"
    views_file = research/"main_views/predictions_outer.parquet"
    fallback_file = research/"fallback_anchor_production/predictions_1_7.parquet"
    view_meta_file = research/"main_views/result_outer.json"
    base_meta_file = release/"candidate/model.json"
    raw_paths = {m:next((project/"data").glob(f"training_2025-{m:02d}-01_*.parquet")) for m in (1,7)}
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"started_utc":pd.Timestamp.now(tz="UTC").isoformat(),
        "start_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "code_sha256":sha(Path(__file__)),"input_sha256":{str(p.relative_to(project)):sha(p) for p in
            [data_file,base_file,views_file,fallback_file,view_meta_file,base_meta_file]+list(raw_paths.values())},
        "validation":"base trained on 10 other source months; selector Jan->July and July->Jan",
        "models":["linear_ridge0.01","neural64"],"strengths":STRENGTHS,"candidates":CANDIDATES,
        "priors":PRIORS.tolist(),"numeric_features":NUMERIC,"categorical_features":CATEGORICAL}
    write_json(args.output/"manifest.json",manifest)
    view_meta = json.loads(view_meta_file.read_text())
    base_meta = json.loads(base_meta_file.read_text())
    if view_meta["train"]!="months_2_3_4_5_6_8_9_10_11_12" or view_meta["valid"]!="months_1_7":
        raise ValueError("expert source-split contract changed")
    expected_months = [2,3,4,5,6,8,9,10,11,12]
    if base_meta["recipe"]["source_months"]!=expected_months or base_meta["recipe"]["outer_source_months"]!=[1,7]:
        raise ValueError("full ensemble was not trained on the required disjoint source months")
    cols = list(dict.fromkeys(list(NUMERIC)+list(CLOCKS)+list(CATEGORICAL)+["hat_aobt","MVT_ID_mvt","MVT_TIME_UTC_mvt","y"]))
    all_data = pd.read_parquet(data_file,columns=cols)
    main_frame = all_data[all_data.hat_aobt.eq(1)].reset_index(drop=True)
    source = source_months(main_frame,raw_paths)
    base_all = pd.read_parquet(base_file).set_index("MVT_ID_mvt",verify_integrity=True)
    base_main = base_all.loc[main_frame.MVT_ID_mvt]
    views = pd.read_parquet(views_file).set_index("MVT_ID_mvt",verify_integrity=True).loc[main_frame.MVT_ID_mvt]
    np.testing.assert_array_equal(base_main.y.to_numpy(),main_frame.y.to_numpy())
    np.testing.assert_array_equal(views.y.to_numpy(),main_frame.y.to_numpy())
    base = base_main.prediction.to_numpy(float)
    if abs(np.sqrt(np.mean((base-main_frame.y.to_numpy())**2))-216.773463)>1e-5:
        raise ValueError("full-ensemble baseline parity failed")
    base_training_ids = pd.read_parquet(release/"data/months_2_3_4_5_6_8_9_10_11_12.parquet",columns=["MVT_ID_mvt"])
    if not set(main_frame.MVT_ID_mvt).isdisjoint(set(base_training_ids.MVT_ID_mvt)):
        raise ValueError("base training includes selector evaluation rows")
    del base_training_ids
    fallback = pd.read_parquet(fallback_file).set_index("MVT_ID_mvt",verify_integrity=True)
    if set(fallback.index)!=set(base_all[base_all.hat_aobt.eq(0)].index):
        raise ValueError("v46 fallback row mismatch")
    np.testing.assert_array_equal(base_all.loc[fallback.index,"y"],fallback.y)
    overall_base = base_all.prediction.copy()
    overall_base.loc[fallback.index] = fallback.prod_anchor
    base_mse = float(np.mean((overall_base-base_all.y)**2))
    c,mask = candidates(main_frame,base,views)
    out = main_frame[["MVT_ID_mvt","MVT_TIME_UTC_mvt","ADEP_mvt","y"]].copy()
    out["source_month"],out["baseline"] = source,base
    prior = np.maximum((soft_weights(np.zeros((len(c),len(PRIORS)-1)),mask)*c).sum(axis=1),FLOOR)
    for strength in STRENGTHS:
        out[f"prior_blend{strength}"] = base+strength*(prior-base)
    errors = np.where(mask,(c-main_frame.y.to_numpy()[:,None])**2,np.inf)
    report = {"manifest":manifest,"main_rows":len(out),"total_rows":len(base_all),
        "baseline_main_rmse":float(np.sqrt(np.mean((base-out.y)**2))),"baseline_overall_rmse":float(np.sqrt(base_mse)),
        "best_single_oracle_main_rmse":float(np.sqrt(errors.min(axis=1).mean())),"fits":[],"variants":{}}
    write_json(args.output/"preflight.json",report)
    print("PREFLIGHT",json.dumps({k:report[k] for k in ("main_rows","total_rows","baseline_main_rmse","baseline_overall_rmse","best_single_oracle_main_rmse")}),flush=True)
    started = time.time()
    for train_month,test_month in ((1,7),(7,1)):
        train,test = source==train_month,source==test_month
        if (train&test).any() or not (train|test).all():
            raise ValueError("invalid source split")
        design = Design().fit(main_frame.loc[train],c[train],mask[train])
        x = design.transform(main_frame.loc[train],c[train],mask[train])
        xv = design.transform(main_frame.loc[test],c[test],mask[test])
        y = main_frame.y.to_numpy(float)[train]
        np.savez_compressed(args.output/f"design_{train_month}.npz",mean=design.mean,std=design.std)
        write_json(args.output/f"design_{train_month}.json",design.state())
        print(f"SPLIT train={train_month} n={train.sum()} test={test_month} n={test.sum()} features={x.shape[1]}",flush=True)
        for family in ("linear","neural"):
            if family=="linear":
                model,fit = fit_linear(x,c[train],mask[train],y)
                np.save(args.output/f"linear_{train_month}.npy",model)
                predictor = predict_linear
            else:
                model,fit = fit_neural(x,c[train],mask[train],y)
                torch.save(model.state_dict(),args.output/f"neural_{train_month}.pt")
                predictor = predict_neural
            fit.update(family=family,train_month=train_month,test_month=test_month,train_rows=int(train.sum()),test_rows=int(test.sum()))
            report["fits"].append(fit)
            write_json(args.output/f"fit_{family}_{train_month}.json",fit)
            pred = predictor(model,xv,c[test],mask[test])
            if not np.isfinite(pred).all() or (pred<FLOOR).any():
                raise ValueError("invalid selector output")
            for strength in STRENGTHS:
                out.loc[test,f"{family}_blend{strength}"] = base[test]+strength*(pred-base[test])
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                airport = main_frame.loc[test,"ADEP_mvt"].astype(str).to_numpy()
                order = np.arange(test.sum())
                for a in np.unique(airport):
                    rows = np.flatnonzero(airport==a)
                    order[rows] = rng.permutation(rows)
                perm = predictor(model,xv[order],c[test],mask[test])
                for strength in STRENGTHS:
                    out.loc[test,f"{family}_blend{strength}_perm{seed}"] = base[test]+strength*(perm-base[test])
            out.loc[test].to_parquet(args.output/f"predictions_{family}_{test_month}.parquet",index=False)
            print(f"FIT_DONE {family} train={train_month} test={test_month} seconds={fit['seconds']:.1f}",flush=True)
            del model
            gc.collect()
        del x,xv
        gc.collect()
    out.to_parquet(args.output/"pooled_predictions.parquet",index=False)
    rank = pd.read_parquet(release/"data/ranking.parquet",columns=cols[:-1])
    rank = rank[rank.hat_aobt.eq(1)]
    dummy_views = pd.DataFrame({v:np.full(len(rank),1200.) for v in VIEWS})
    _,rank_mask = candidates(rank,np.full(len(rank),1200.),dummy_views)
    coverage = {name:{"validation":float(mask[:,i].mean()),"ranking":float(rank_mask[:,i].mean())}
                for i,name in enumerate(CANDIDATES) if i>=6}
    report["candidate_coverage"] = coverage
    coverage_pass = all(abs(v["validation"]-v["ranking"])<=.05 for v in coverage.values())
    report["coverage_pass"] = coverage_pass
    for family in ("linear","neural"):
        for strength in STRENGTHS:
            name = f"{family}_blend{strength}"
            stats = summarize(out,out[name].to_numpy(),len(out),len(base_all),base_mse)
            prior_stats = summarize(out,out[f"prior_blend{strength}"].to_numpy(),len(out),len(base_all),base_mse)
            stats["prior_main_mse_gain"] = prior_stats["main_mse_gain"]
            controls = {str(s):summarize(out,out[f"{name}_perm{s}"].to_numpy(),len(out),len(base_all),base_mse)["main_mse_gain"] for s in SEEDS}
            stats["permutation_main_mse_gains"] = controls
            stats["all_fits_complete"] = all(f["success"] for f in report["fits"] if f["family"]==family)
            stats["passes_stage1"] = bool(stats["overall_rmse_gain"]>=.5 and min(stats["month_mse_gains"].values())>0
                and stats["mse_gain_without_top100"]>0 and stats["positive_airports"]>=7
                and stats["main_mse_gain"]>prior_stats["main_mse_gain"] and stats["main_mse_gain"]>max(controls.values())
                and coverage_pass and stats["all_fits_complete"])
            report["variants"][name] = stats
    report["seconds"] = time.time()-started
    report["decision"] = "ADVANCE_TO_MULTI_PERIOD_VALIDATION" if any(v["passes_stage1"] for v in report["variants"].values()) else "STOP_STAGE1_KEEP_V46"
    report["artifact_sha256"] = {p.name:sha(p) for p in args.output.iterdir() if p.is_file()}
    write_json(args.report,report)
    print(json.dumps({k:report[k] for k in ("baseline_main_rmse","baseline_overall_rmse","variants","seconds","decision")},indent=2),flush=True)


if __name__ == "__main__":
    main()

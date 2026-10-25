"""Frozen full-ensemble replay and sequence substitution, no tree fits or uploads.

Follows airport_ablation.py's pre-registered Stage B. Reconstruct the unfloored
main blend from the existing candidate-stage checkpoints (not all-year models).
Verify it against the saved holdout, then replace ONLY the raw neural residual
at weight 0.25. Keep trees, schedule-copy gate, floor and v46 fallback fixed.
Never adjust a floored prediction by a raw component difference. Report matched
FLAT substitutions as controls and never tune blend weights on this holdout.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main_model as H
import neural_model as NM
import joblib
import numpy as np
import pandas as pd
from research_250 import candidate_features as B
from research_260.airport_ablation import sha,write_json,log


def postprocess(anchor, tree, neural, gate, schedule):
    return np.maximum((1-gate)*(anchor+.75*tree+.25*neural)+gate*schedule,300.)


def baseline(args):
    out=args.output
    if (out/'ensemble_baseline.json').exists():
        log('baseline already frozen')
        return
    features=B.main_features(H.MERKMALE_HAUPT)
    frame=pd.read_parquet(args.release/'data/months_1_7.parquet')
    main=frame.loc[frame.hat_aobt.eq(1)].copy()
    x=main[features]
    assert (H.ANTEIL_NETZ,H.ANTEIL_CB,H.ANTEIL_FLH,H.ERF_HOCH)==(.25,.6,.2,2.5)
    checkpoints=args.release/'candidate/checkpoints'
    hashes={}
    def load(name):
        path=checkpoints/(name+'.pkl')
        hashes[path.name]=sha(path)
        saved=joblib.load(path)
        return saved['model']
    predictions=[]
    for seed in H.SAATEN:
        log(f'replay global tree {seed}')
        model=load(f'lgb_{seed}'); predictions.append(model.predict(x)); del model; gc.collect()
    global_mean=np.mean(predictions,axis=0); del predictions
    local=global_mean.copy()
    airports=main.ADEP_mvt.astype(str).to_numpy()
    for airport in sorted(set(airports)):
        model=load('airport_'+airport); mask=airports==airport
        local[mask]=model.predict(x.loc[mask]); del model; gc.collect()
    tree=.8*global_mean+.2*local
    cat_x=H.kat_str(x)
    cats=[]
    for seed in H.CB_SAATEN:
        log(f'replay CatBoost {seed}')
        model=load(f'catboost_{seed}'); cats.append(model.predict(cat_x)); del model; gc.collect()
    tree=.4*tree+.6*np.mean(cats,axis=0); del cats,cat_x
    nets=[]
    for seed in NM.SAATEN:
        log(f'replay saved flat net {seed}')
        model=load(f'neural_{seed}'); nets.append(NM.sage_voraus(model,x)); del model; gc.collect()
    neural=np.mean(nets,axis=0)
    detector=load('reporting_classifier')
    gate=detector.predict_proba(x)[:,1]**2.5
    del detector,x; gc.collect()
    anchor=main.ersatz_s.to_numpy(float)
    schedule=main.versp_min.to_numpy(float)*60
    replay=postprocess(anchor,tree,neural,gate,schedule)
    saved=pd.read_parquet(args.release/'candidate/holdout_predictions.parquet').set_index('MVT_ID_mvt')
    ids=main.MVT_ID_mvt.to_numpy()
    reference=saved.loc[ids,'prediction'].to_numpy(float)
    np.testing.assert_allclose(saved.loc[ids,'y'],main.y,atol=0,rtol=0)
    maxdiff=float(np.abs(replay-reference).max())
    if maxdiff>1e-3: raise ValueError(f'full ensemble replay failed: {maxdiff}')
    fallback=pd.read_parquet(args.fallback).set_index('MVT_ID_mvt')
    fallback_ids=frame.loc[frame.hat_aobt.eq(0),'MVT_ID_mvt'].to_numpy()
    if set(fallback.index)!=set(fallback_ids): raise ValueError('fallback ID mismatch')
    np.testing.assert_allclose(fallback.loc[fallback_ids,'y'],saved.loc[fallback_ids,'y'],atol=0,rtol=0)
    fallback_sse=float(np.sum((fallback.loc[fallback_ids,'prod_anchor'].to_numpy(float)-saved.loc[fallback_ids,'y'].to_numpy(float))**2))
    rows=main[['MVT_ID_mvt','MVT_TIME_UTC_mvt','ADEP_mvt','y','ersatz_s']].copy()
    rows['non_neural_residual']=tree; rows['old_neural_residual']=neural
    rows['gate']=gate; rows['schedule']=schedule; rows['baseline']=reference
    for seed,prediction in zip(NM.SAATEN,nets): rows[f'old_neural_{seed}']=prediction
    rows.to_parquet(out/'ensemble_baseline.parquet',index=False)
    sse=float(np.sum((reference-main.y.to_numpy(float))**2))
    report={'main_rows':len(main),'fallback_rows':len(fallback_ids),'fallback_sse':fallback_sse,
            'rows':len(frame),'main_rmse':float(np.sqrt(sse/len(main))),
            'overall_rmse':float(np.sqrt((sse+fallback_sse)/len(frame))),
            'replay_max_abs_s':maxdiff,'checkpoints_sha256':hashes,'fallback_predictions_sha256':sha(args.fallback),
            'baseline_sha256':sha(out/'ensemble_baseline.parquet')}
    write_json(out/'ensemble_baseline.json',report)
    log('BASELINE '+json.dumps({k:v for k,v in report.items() if 'sha256' not in k}))


def metrics(rows, base, candidate, baseline_info):
    y=rows.y.to_numpy(float)
    e0,e1=(base-y)**2,(candidate-y)**2
    gain=e0-e1
    months=pd.to_datetime(rows.MVT_TIME_UTC_mvt,utc=True).dt.month.to_numpy()
    airports=rows.ADEP_mvt.astype(str).to_numpy()
    total,back=baseline_info['rows'],baseline_info['fallback_sse']
    trim=np.sort(gain)[:-100]
    out={'main_rmse':float(np.sqrt(e1.mean())), 'main_rmse_gain':float(np.sqrt(e0.mean())-np.sqrt(e1.mean())),
         'main_mse_gain':float(gain.mean()), 'mse_gain_without_top100':float(trim.mean()),
         'overall_rmse':float(np.sqrt((e1.sum()+back)/total)),
         'overall_rmse_gain':float(np.sqrt((e0.sum()+back)/total)-np.sqrt((e1.sum()+back)/total)),
         'by_month_mse_gain':{str(m):float(gain[months==m].mean()) for m in np.unique(months)},
         'by_airport_mse_gain':{a:float(gain[airports==a].mean()) for a in sorted(set(airports))}}
    out['airports_positive']=sum(v>0 for v in out['by_airport_mse_gain'].values())
    out['advance_full_five_seed']=bool(out['main_rmse_gain']>=.3 and out['overall_rmse_gain']>0
        and out['by_month_mse_gain'].get('1',0)>0 and out['by_month_mse_gain'].get('7',0)>0
        and out['mse_gain_without_top100']>0 and out['airports_positive']>=7)
    return out


def compare(args):
    out=args.output
    info=json.loads((out/'ensemble_baseline.json').read_text())
    if sha(out/'ensemble_baseline.parquet')!=info['baseline_sha256']: raise ValueError('baseline changed')
    rows=pd.read_parquet(out/'ensemble_baseline.parquet')
    seeds=[int(s) for s in args.seeds.split(',')]
    preds={}
    components={}
    report={'seeds':seeds,'note':'January/July main-component substitution, not a leaderboard score or release gate',
            'baseline':{k:info[k] for k in ['main_rmse','overall_rmse','main_rows','fallback_rows']},'modes':{},'pairwise':{}}
    for mode in ['flat','origin','local']:
        residuals=[]
        for seed in seeds:
            path=out/f'{mode}_{seed}_predictions.parquet'
            if not path.exists(): break
            pred=pd.read_parquet(path).set_index('MVT_ID_mvt')
            if set(pred.index)!=set(rows.MVT_ID_mvt) or not pred.index.is_unique: raise ValueError('prediction ID mismatch')
            np.testing.assert_allclose(pred.loc[rows.MVT_ID_mvt,'y'],rows.y,atol=0,rtol=0)
            residuals.append(pred.loc[rows.MVT_ID_mvt,'residual'].to_numpy())
        if len(residuals)!=len(seeds): continue
        residual=np.mean(residuals,axis=0)
        prediction=postprocess(rows.ersatz_s.to_numpy(),rows.non_neural_residual.to_numpy(),residual,rows.gate.to_numpy(),rows.schedule.to_numpy())
        preds[mode]=prediction
        components[mode]=np.maximum(rows.ersatz_s.to_numpy()+residual,300.)
        report['modes'][mode]=metrics(rows,rows.baseline.to_numpy(),prediction,info)
        report['modes'][mode]['component_rmse']=float(np.sqrt(np.mean((components[mode]-rows.y.to_numpy())**2)))
        legacy=np.mean([np.maximum(rows.ersatz_s.to_numpy()+r,300.) for r in residuals],axis=0)
        report['modes'][mode]['historical_floor_before_average_component_rmse']=float(np.sqrt(np.mean((legacy-rows.y.to_numpy())**2)))
    for mode in ['origin','local']:
        if mode not in preds or 'flat' not in preds: continue
        report['pairwise'][mode+'_vs_flat']=metrics(rows,preds['flat'],preds[mode],info)
        component=metrics(rows,components['flat'],components[mode],info)
        component={k:v for k,v in component.items() if not k.startswith('overall') and k!='advance_full_five_seed'}
        component['stage_a_pass']=bool(component['main_rmse_gain']>=.3 and component['mse_gain_without_top100']>0
                                      and component['by_month_mse_gain'].get('1',0)>0 and component['by_month_mse_gain'].get('7',0)>0)
        report['pairwise'][mode+'_component_vs_flat']=component
        report['modes'][mode]['advance_full_five_seed'] &= bool(len(seeds)>=2 and component['stage_a_pass']
                                                              and report['pairwise'][mode+'_vs_flat']['main_mse_gain']>0)
    if 'origin' in preds and 'local' in preds:
        report['pairwise']['local_vs_origin']=metrics(rows,preds['origin'],preds['local'],info)
        comp=metrics(rows,components['origin'],components['local'],info)
        report['prefer_local_over_origin']=bool(comp['main_mse_gain']>0 and comp['mse_gain_without_top100']>0
                                               and comp['by_month_mse_gain'].get('1',0)>0 and comp['by_month_mse_gain'].get('7',0)>0)
    suffix='_'.join(map(str,seeds))
    write_json(out/f'ensemble_comparison_{suffix}.json',report)
    rows[['MVT_ID_mvt']].assign(**preds).to_parquet(out/f'ensemble_comparison_{suffix}.parquet',index=False)
    log('COMPARISON '+json.dumps(report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['baseline','compare'])
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--fallback',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',default='13,21')
    args=parser.parse_args()
    {'baseline':baseline,'compare':compare}[args.command](args)


if __name__=='__main__': main()

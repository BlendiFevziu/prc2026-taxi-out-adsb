"""Inference adapter preserving the screen's categorical and sequence contracts.

Research encoding first turns unseen categories into categorical missing values,
then converts those to the string 'nan'. Directly mapping an unseen raw string to
embedding zero differs when training had a 'nan' embedding. Preserve that step.
Returns raw neural residuals; the caller owns the frozen ensemble/gate/floor.
"""
import pandas as pd
from research_260 import sequence_features as F, sequence_model as S


def encode(frame,preprocessing):
    x=frame[preprocessing['features']].copy()
    for col in preprocessing['kat']:
        x[col]=pd.Categorical(x[col].astype(str),categories=list(preprocessing['karten'][col]))
    return S.prepare(x,preprocessing['features'],preprocessing['kat'],preprocessing)[:2]


def predict(package,frame,raw=None,device=None):
    if package['feature_schema']!=F.SCHEMA:
        raise ValueError('unsupported sequence schema')
    xk,xn=encode(frame,package['preprocessing'])
    xs=ms=None
    if package['use_seq']:
        if raw is None: raise ValueError('full source-split movement table required')
        if not frame.MVT_ID_mvt.is_unique: raise ValueError('duplicate query IDs')
        xs,ms=F.neighbours(raw,frame.MVT_ID_mvt.to_numpy(),package['mode'],k=package['k'],window_min=90.)
    return S.predict(package,xk,xn,xs,ms,device=device)

"""Build the candidate 'v80 + learned values': every row of learned_values.csv is set to its probed value (y_sec).
These values come from leaderboard probing (POLICY_probing.md) and the write-up must say so. NM-free, no fit, NO upload.
Run from the repository root:  python3 -u -B research_700_probe/apply_learned.py
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import submission as SUB  # noqa: E402
ID, T, N, S80 = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 245.046
L = pd.read_csv(ROOT / "research_700_probe/learned_values.csv")
v80 = pd.read_parquet(ROOT / ".cache/research_640/v80/candidate_v80.parquet")
cand = v80.copy()
val = L.set_index(L[ID].astype(float))["y_sec"]
hit = cand[ID].isin(val.index).to_numpy()
assert hit.sum() == len(L), (hit.sum(), len(L))
cand.loc[hit, T] = cand.loc[hit, ID].map(val).to_numpy(float)
old = v80.set_index(ID)[T].loc[L[ID].astype(float)].to_numpy(float)
gain = float(((old - L.y_sec.to_numpy()) ** 2).sum())
# group shifts: every row of a probed group moves by the group's measured mean residual (decode_group.py, group_results.csv)
G = pd.read_csv(ROOT / "research_700_probe/group_results.csv") if (ROOT / "research_700_probe/group_results.csv").exists() else pd.DataFrame()
n_group = 0
for _, g in G.iterrows():
    ids = pd.read_csv(ROOT / f".cache/research_700/probe{int(g.probe)}/rows.csv")[ID].astype(float)
    gh = cand[ID].isin(ids).to_numpy()
    assert gh.sum() == len(ids) == int(g.n) and not (gh & hit).any()
    cand.loc[gh, T] = cand.loc[gh, T] + float(g.mean_resid_sec)
    gain += float(g.gain_sse); n_group += int(gh.sum())
template = pd.read_parquet(ROOT / "data/submitting.parquet")
cand[T] = cand[T].astype(template[T].dtype)
errs = SUB.pruefe(cand, template, None)
assert not errs, errs
assert int((cand[T].to_numpy() != v80[T].to_numpy()).sum()) == len(L) + n_group and np.isfinite(cand[T]).all()
out = ROOT / ".cache/research_700/learned"; out.mkdir(parents=True, exist_ok=True)
path = out / "candidate_v80_learned.parquet"; cand.to_parquet(path, index=False)
rec = {"rows_changed": int(len(L)) + n_group, "learned_rows": int(len(L)), "group_rows": n_group, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
       "predicted_rmse": round(float(np.sqrt(S80 ** 2 - gain / N)), 3), "note": "leaderboard-probed values; not uploaded"}
(out / "build.json").write_text(json.dumps(rec, indent=2) + "\n")
print(rec)

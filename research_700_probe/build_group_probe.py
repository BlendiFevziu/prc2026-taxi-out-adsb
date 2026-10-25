"""Build group probes: v80 with every row of a group moved by +C seconds (policy: POLICY_probing.md). One upload then gives the exact
sum of residuals over the group: dSSE = n*C^2 + 2*C*sum(p - y)  ->  mean shift that is optimal for the group = -mean(p - y).
Groups come from the cached inventory (rome = LIRF rows with delay > 1 h, other = other fallback rows), minus the rows already
learned in learned_values.csv. Rome is split into four equal parts by the classifier's shift24 probability.
Ledger columns reused: row_id = 'group:<name>', old = n rows, step = C, new = mean new prediction. No upload.
Run from the repository root:  python3 -u -B research_700_probe/build_group_probe.py 11 12 13 14 15
"""
import csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import submission as SUB  # noqa: E402
ID, T, C = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 1000.0
LEDGER = ROOT / "research_700_probe/ledger.csv"
FIELDS = ["probe", "key", "row_id", "old", "new", "step", "sha256", "intent_at", "official_score", "dSSE_vs_v80"]
NAMES = {11: "rome_q1", 12: "rome_q2", 13: "rome_q3", 14: "rome_q4", 15: "other_fallback"}
KEYS = {11: "upbeat-goblin_v93.parquet", 12: "upbeat-goblin_v94.parquet", 13: "upbeat-goblin_v95.parquet",
        14: "upbeat-goblin_v96.parquet", 15: "upbeat-goblin_v97.parquet"}
inv = pd.read_parquet(ROOT / ".cache/research_700/inventory.parquet")
learned = set(pd.read_csv(ROOT / "research_700_probe/learned_values.csv")[ID].astype(float))
inv = inv[~inv[ID].astype(float).isin(learned)].reset_index(drop=True)
rome = inv[inv.group.eq("rome")].sort_values("q_shift24").reset_index(drop=True)
parts = np.array_split(np.arange(len(rome)), 4)
SETS = {f"rome_q{i + 1}": rome.loc[parts[i], ID].astype(float).to_numpy() for i in range(4)}
SETS["other_fallback"] = inv[inv.group.eq("other_fallback")][ID].astype(float).to_numpy()
rows = list(csv.DictReader(LEDGER.open()))
scored = {r["probe"] for r in rows if r["official_score"]}
v80 = pd.read_parquet(ROOT / ".cache/research_640/v80/candidate_v80.parquet")
template = pd.read_parquet(ROOT / "data/submitting.parquet")
for arg in sys.argv[1:]:
    no = int(arg); name = NAMES[no]
    if str(no) in scored:
        raise SystemExit(f"probe {no} already has a score")
    ids = SETS[name]
    hit = v80[ID].isin(ids).to_numpy()
    assert hit.sum() == len(ids), (name, hit.sum(), len(ids))
    cand = v80.copy(); cand.loc[hit, T] = cand.loc[hit, T] + C
    cand[T] = cand[T].astype(template[T].dtype)
    errs = SUB.pruefe(cand, template, None); assert not errs, errs
    assert int((cand[T].to_numpy() != v80[T].to_numpy()).sum()) == len(ids) and np.isfinite(cand[T]).all()
    out = ROOT / f".cache/research_700/probe{no}"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"candidate_probe{no}.parquet"; cand.to_parquet(path, index=False)
    pd.Series(ids, name=ID).to_csv(out / "rows.csv", index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / f"ready_for_upload_probe{no}.json").write_text(json.dumps(
        {"version": f"probe{no}", "key": KEYS[no], "candidate_path": str(path), "candidate_sha256": digest,
         "recipe": f"v80 with the {len(ids)} rows of group {name} moved +{C:.0f} s", "authorised": "Owner, 2 October 2026: 'do anything u think that might bring us an rmse of 215' (group probes)"}, indent=2) + "\n")
    rows = [r for r in rows if r["probe"] != str(no)]
    rows.append(dict(probe=no, key=KEYS[no], row_id=f"group:{name}", old=len(ids), new=float(cand.loc[hit, T].mean()), step=C, sha256=digest,
                     intent_at=datetime.now(timezone.utc).isoformat(), official_score="", dSSE_vs_v80=""))
    print(f"probe {no}: group {name} n={len(ids)} sha {digest[:16]}", flush=True)
with LEDGER.open("w", newline="") as f:
    w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)

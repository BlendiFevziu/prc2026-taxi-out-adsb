"""Build v69 = v68 (built and verified, not uploaded) with the rule of research_500_midnight/PREREG_midnight.md: Rome's
fallback rows with a delay above 8 h and no local-midnight crossing get their delay. NM-free, no fit. No upload.
Run from the repository root:  python3 -u -B research_500_midnight/build_v69.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule as R  # noqa: E402

sys.path.insert(0, str(R.ROOT))
import submission as SUB  # noqa: E402

N, V68_CENTRAL = 344841, 255.89          # v68's fixed central prediction (not yet scored)
P_COPY, NORMAL_TAXI_S = 13 / 14, 1000.0  # Laplace on 12 of 12; a typical Rome taxi for the risk case
V68 = R.ROOT / ".cache/research_480/v68/candidate_v68.parquet"
OUT = R.ROOT / ".cache/research_500/v69"
REPORT = R.ROOT / "research_500_midnight/results/v69_build.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score_at(mse_gain):
    return round(float(np.sqrt(V68_CENTRAL ** 2 - mse_gain)), 2)


def main():
    build68 = json.loads((R.ROOT / "research_480_weights/results/v68_build.json").read_text())
    checks = {"v68_is_built_file": sha(V68) == build68["candidate_sha256"]}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    v68 = pd.read_parquet(V68)
    rk = R.annotate(pd.read_parquet(R.ROOT / "data/ranking.parquet", columns=R.COLS + [R.T]))
    rk = rk[rk.PHASE_mvt.eq("DEP")].reset_index(drop=True)
    checks["ranking_departures_unlabelled"] = bool(rk[R.T].isna().all())
    sel = rk[R.selected(rk)]
    new = pd.Series(sel.delay.to_numpy(float), index=sel[R.ID].to_numpy())
    checks["rows_selected"] = int(len(new))
    cand = v68.copy()
    hit = cand[R.ID].isin(new.index).to_numpy()
    old = cand.loc[hit, R.T].to_numpy(float)
    cand.loc[hit, R.T] = cand.loc[hit, R.ID].map(new).to_numpy(float)
    template = pd.read_parquet(R.ROOT / "data/submitting.parquet")
    cand[R.T] = cand[R.T].astype(template[R.T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v68.loc[~hit].reset_index(drop=True))
    newv = cand.loc[hit, R.T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v68=bool(untouched),
                  all_finite=bool(np.isfinite(cand[R.T]).all()), rows_changed=int((cand[R.T].to_numpy() != v68[R.T].to_numpy()).sum()))
    gain_copy = (old - newv) ** 2                                              # squared error removed if the row is a copy
    loss_normal = (newv - NORMAL_TAXI_S) ** 2 - (old - NORMAL_TAXI_S) ** 2     # added if it is a normal taxi
    worst = int(np.argmax(loss_normal))
    rows = [{"id": str(i), "v68": float(o), "v69": float(n), "gain_if_copy": float(g), "change_if_normal": float(-l)}
            for i, o, n, g, l in zip(cand.loc[hit, R.ID], old, newv, gain_copy, loss_normal)]
    central = float(np.sum(P_COPY * gain_copy - (1 - P_COPY) * loss_normal)) / N
    best = float(gain_copy.sum()) / N
    worst_mse = float(gain_copy.sum() - gain_copy[worst] - loss_normal[worst]) / N
    report = {"checks": checks, "base": {"version": "v68", "sha256": build68["candidate_sha256"], "central": V68_CENTRAL},
              "rows": rows,
              "prediction_fixed_before_upload": {
                  "assumptions": f"p(copy) {P_COPY:.3f} per row (Laplace on 12 of 12); a normal row taxis {NORMAL_TAXI_S:.0f} s; "
                                 "best: all copies; worst: the row with the largest risk is normal, the others copies",
                  "central": score_at(central * N / N), "best": score_at(best), "worst": score_at(worst_mse),
                  "gain_s_central_vs_v68": round(V68_CENTRAL - score_at(central), 2)}}
    ok = not errors and untouched and checks["all_finite"] and checks["ranking_departures_unlabelled"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v69.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

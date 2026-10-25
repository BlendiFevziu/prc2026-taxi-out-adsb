"""Build v70 = v69 plus arm A on the 2026 slice of research_510_rome_classifier/PREREG_rome_classifier.md, addendum A:
Rome fallback departures with delay above 1 h and at most 8 h (not a v69 rule row) get (v69 value + expected value) / 2
from the label-mechanism classifier fitted on all 2025 cell rows. NM-free. No upload.
Run from the repository root:  python3 -u -B research_510_rome_classifier/build_v70.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rome_classifier as C  # noqa: E402

R, ROOT = C.R, C.ROOT
import submission as SUB  # noqa: E402

V69 = ROOT / ".cache/research_500/v69/candidate_v69.parquet"
OUT = ROOT / ".cache/research_510/v70"
REPORT = ROOT / "research_510_rome_classifier/results/v70_build.json"
EXTRA = ["FLIGHT_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def training():
    d = pd.concat([pd.read_parquet(p, columns=R.COLS + [R.T] + EXTRA) for p in sorted((ROOT / "data").glob("training_2025-*.parquet"))],
                  ignore_index=True)
    d = R.annotate(d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF") & d.AOBT_3_flt.isna()]).reset_index(drop=True)
    d = d[C.cell(d)].reset_index(drop=True)
    d["kind"] = R.label_kind(d[R.T], d.delay)
    return d


def slice_2026():
    k = pd.read_parquet(ROOT / "data/ranking.parquet", columns=R.COLS + [R.T] + EXTRA)
    k = R.annotate(k[k.PHASE_mvt.eq("DEP") & k.ADEP_mvt.eq("LIRF") & k.AOBT_3_flt.isna()]).reset_index(drop=True)
    return k[C.cell(k) & (k.delay <= 8 * 3600).to_numpy()].reset_index(drop=True)


def expected_2026():
    d, k = training(), slice_2026()
    p = C.fit_predict(C.features(d), d.kind, C.features(k))
    t = float(d.loc[d.kind.eq("normal"), R.T].median())
    return k, C.expected(p, k.delay.to_numpy(float), t), p, len(d)


def main():
    build69 = json.loads((ROOT / "research_500_midnight/results/v69_build.json").read_text())
    checks = {"v69_is_built_file": sha(V69) == build69["candidate_sha256"]}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    k, exp, probs, n_train = expected_2026()
    checks.update(training_rows=n_train, training_rows_as_registered=n_train == 1363, slice_rows=int(len(k)),
                  slice_rows_as_registered=len(k) == 326, ranking_unlabelled=bool(k[R.T].isna().all()))
    v69 = pd.read_parquet(V69)
    at69 = v69.set_index(R.ID)[R.T]
    old = at69.loc[k[R.ID]].to_numpy(float)
    new = pd.Series((old + exp) / 2, index=k[R.ID].to_numpy())
    cand = v69.copy()
    hit = cand[R.ID].isin(new.index).to_numpy()
    cand.loc[hit, R.T] = cand.loc[hit, R.ID].map(new).to_numpy(float)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    cand[R.T] = cand[R.T].astype(template[R.T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v69.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, R.T].to_numpy(float) - v69.loc[hit, R.T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v69=bool(untouched),
                  all_finite=bool(np.isfinite(cand[R.T]).all()), rows_changed=int((cand[R.T].to_numpy() != v69[R.T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2))), "max_abs": float(np.abs(change).max())},
                  p_copy={"mean": float(probs[:, 0].mean()), "share_above_0_5": float((probs[:, 0] > .5).mean())})
    report = {"checks": checks, "base": {"version": "v69", "sha256": build69["candidate_sha256"], "central": 255.32},
              "prediction_fixed_before_upload": {"source": "PREREG_rome_classifier.md addendum A (committed 142a1ee before this build)",
                                                 "central": 253.57, "best": 249.48, "worst": 256.32}}
    ok = not errors and untouched and checks["all_finite"] and checks["ranking_unlabelled"] and checks["training_rows_as_registered"] \
        and checks["slice_rows_as_registered"]
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v70.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

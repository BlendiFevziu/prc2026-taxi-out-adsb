"""Build the C6 candidate: a thresholded copy increment on the fallback lane. Nothing is uploaded.

Recipe. The shipped fallback prediction is

    yhat_A0 = (1 - w_z) * normal  +  w_z * (0.7*versp + 0.3*height)        w_z = p_z ** HOCH

C6 lets a SECOND classifier, trained on the copy mechanism |y - versp| <= 10 s, add weight where the shipped gate
is silent, and gives that increment a bare versp height rather than the ANTEIL_HM-blended one:

    yhat_C6 = (1 - w_z - delta) * normal + w_z * hoehe + delta * versp
            = yhat_A0 + delta * (versp - normal)
    delta   = max(p_kopie_thresholded ** HOCH - w_z, 0)

CORRECTED 23.09.2026 after the first build failed its own gate. `models.pkl` does NOT reproduce the shipped
fallback rows, because v46 replaced the lane with the ANCHORED model of
research_240/anchor_preview_2026_production.py. That recipe keeps the z-classifier and the height model on the
35 features of H.MERKMALE_RUECK and replaces only the NORMAL branch by `nt_anchor + offset`, the offset fitted on
41 features. So `q` is unchanged and C6 stays additive, with `normal_anchor` in place of `normal`. The shipped
prediction is preserved exactly wherever delta is zero.

Evidence for the design, all in STATUS.md and experiments.csv of 23.09.2026: the increment fires on 1.31 % of
2025 fallback rows at 75.6 % +/- 4.8 % precision, holding in 6 of 6 folds, and never above versp 18,425 s, so the
24-hour band that can wreck a submission is untouched. The pooled +1.191 s comes from a harness that fails its
own calibration probe and is NOT the expected gain; discount it to roughly 0.4 s.

Gates, all hard:
  G1  base parquet hash matches the v52 manifest
  G2  model.pkl hash matches the v46 manifest
  G3  the shipped fallback prediction is reproduced from model.pkl on the scored rows
  G4  submission.pruefe passes, main rows are byte-identical to v52, only fallback rows move, all finite

Run from the repository root:
  python3 -u -B research_280/build_c6_candidate.py
"""

import glob, hashlib, json, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import neural_model as NM  # noqa: F401
import lightgbm as lgb
import submission as SUB
import main_model as H
import features_stand_rate as MSQ
from research_240 import features_neighbour_taxi as NT

BASE = Path(".cache/research_280/release/candidate_v52_combined.parquet")
ANCH = Path(".cache/research_240/anchor_preview_2026_production.parquet")
L25 = ".cache/research_240/anchor_preview/fallback_2025_all_*.parquet"
R26 = ".cache/research_240/anchor_preview/fallback_2026_ranking_*.parquet"
OUT = Path(".cache/research_280/release")
FEAT = list(H.MERKMALE_RUECK)
WITH_NT = FEAT + NT.MERKMALE
SCHWELLE, TOL, G3_MAX = 0.5, 10, 1.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def reg(r, l, m):
    return lgb.LGBMRegressor(objective="l2", n_estimators=r, learning_rate=.05, num_leaves=l,
                             min_child_samples=m, verbose=-1, n_jobs=-1, random_state=0)


def klass():
    return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                              min_child_samples=20, verbose=-1, n_jobs=-1, random_state=0)


def main():
    m52 = json.loads(Path("results/upbeat-goblin_v52.json").read_text())
    m46 = json.loads(Path("results/upbeat-goblin_v46_manifest.json").read_text())
    g1 = sha(BASE) == m52["candidate_sha256"]
    g2 = sha(ANCH) == m46["anchored_fallback_predictions_sha256"]
    log(f"GATE G1 base hash matches v52: {g1}")
    log(f"GATE G2 anchored-predictions hash matches v46: {g2}")
    if not (g1 and g2):
        raise SystemExit("input hashes differ from the manifests")

    lo = pd.read_parquet(sorted(glob.glob(L25))[0]).copy()
    rank = pd.read_parquet(sorted(glob.glob(R26))[0]).copy()
    lo["versp_s"] = lo.versp_min.to_numpy() * 60
    rank["versp_s"] = rank.versp_min.to_numpy() * 60
    lo["z"] = (lo.y > H.SCHWELLE_EXTREM).astype(int)
    lo["kopie"] = (lo.y - lo.versp_s).abs().le(TOL).astype(int)
    quote = MSQ.passe_an(lo)
    lo = MSQ.ausser_faltung(lo)
    frame = MSQ.wende_an(quote, rank)
    y = lo.y.to_numpy(float); z0 = lo.z.eq(0).to_numpy()
    log(f"{len(lo):,} learning rows, {int(lo.kopie.sum()):,} copies; {len(frame):,} scored fallback rows")

    q = klass().fit(lo[FEAT], lo.z).predict_proba(frame[FEAT])[:, 1] ** H.HOCH
    high = ((1 - H.ANTEIL_HM) * frame.versp_s.to_numpy()
            + H.ANTEIL_HM * reg(300, 15, 10).fit(lo.loc[~z0, FEAT], y[~z0]).predict(frame[FEAT]))
    normal = reg(400, 31, 20).fit(lo.loc[z0, FEAT], y[z0]).predict(frame[FEAT])
    a_l, a_p = lo.nt_anchor.to_numpy(float), frame.nt_anchor.to_numpy(float)
    rows = z0 & np.isfinite(a_l)
    off = reg(400, 31, 20).fit(lo.loc[rows, WITH_NT], y[rows] - a_l[rows])
    has = np.isfinite(a_p)
    normal_a = normal.copy()
    normal_a[has] = a_p[has] + off.predict(frame.loc[has, WITH_NT])
    anchored = np.clip((1 - q) * normal_a + q * high, 0, None)

    stored = pd.read_parquet(ANCH).set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt.to_numpy(), "anchored"].to_numpy(float)
    diff = np.abs(anchored - stored)
    g3 = {"max_abs_s": float(diff.max()), "median_abs_s": float(np.median(diff)),
          "share_over_1s": float((diff > 1).mean()), "passed": bool(diff.max() <= G3_MAX)}
    log("GATE G3 reproduction of the shipped anchored fallback " + json.dumps(g3))
    if not g3["passed"]:
        raise SystemExit("GATE G3 failed; the anchored recipe was not reproduced, stopping")

    pk = klass().fit(lo[FEAT], lo.kopie).predict_proba(frame[FEAT])[:, 1]
    pk = np.where(pk >= SCHWELLE, pk, 0.0)
    delta = np.maximum(pk ** H.HOCH - q, 0.0)
    fires = delta > 0
    log(f"increment fires on {int(fires.sum()):,} of {len(frame):,} rows ({fires.mean():.2%})"
        + (f", max versp where it fires {frame.versp_s.to_numpy()[fires].max():,.0f} s" if fires.any() else ""))

    base = pd.read_parquet(BASE)
    pos = pd.Index(base.MVT_ID_mvt).get_indexer(frame.MVT_ID_mvt.to_numpy())
    if (pos < 0).any():
        raise SystemExit("fallback IDs missing from the base submission")
    shipped = base.TAXITIME_SEC_mvt.to_numpy(float)[pos]
    rome = np.abs(shipped - stored) > 1e-6
    overlap = int((rome & fires).sum())
    log(f"rows where v52 already differs from the plain anchored model (Rome gate): {int(rome.sum())}; "
        f"overlap with the increment: {overlap}")
    # The identity yhat_C6 = yhat_A0 + delta*(versp - normal_anchor) assumes the base IS the anchored
    # blend. On the Rome-gate rows it is a hard substitution instead, so adding to it would not be C6
    # at all and overshoots versp. Those rows keep the shipped value.
    delta = np.where(rome, 0.0, delta)
    fires = delta > 0
    neu = np.clip(shipped + delta * (frame.versp_s.to_numpy() - normal_a), 0, None)

    template = pd.read_parquet("data/submitting.parquet")
    cand = base.copy()
    v = cand.TAXITIME_SEC_mvt.to_numpy().copy(); v[pos] = neu
    cand["TAXITIME_SEC_mvt"] = pd.Series(v).astype(template.TAXITIME_SEC_mvt.dtype).to_numpy()
    errors = SUB.pruefe(cand, template, None)
    changed = cand.TAXITIME_SEC_mvt.to_numpy() != base.TAXITIME_SEC_mvt.to_numpy()
    rk = pd.read_parquet(".cache/clock_queue_release_20260912/data/ranking.parquet")
    main_ids = set(rk[rk.hat_aobt == 1].MVT_ID_mvt.to_numpy())
    g4 = {"pruefe_errors": [str(e) for e in errors],
          "all_finite": bool(np.isfinite(cand.TAXITIME_SEC_mvt.to_numpy(float)).all()),
          "rows": int(len(cand)), "rows_changed": int(changed.sum()),
          "only_fallback_changed": bool(not (set(cand.MVT_ID_mvt[changed]) & main_ids)),
          "min": float(cand.TAXITIME_SEC_mvt.min()), "max": float(cand.TAXITIME_SEC_mvt.max())}
    g4["passed"] = bool(not errors and g4["all_finite"] and g4["only_fallback_changed"])
    log("GATE G4 " + json.dumps(g4))
    if not g4["passed"]:
        raise SystemExit("GATE G4 failed")

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "candidate_v53_c6.parquet"
    cand.to_parquet(target, index=False)
    d = neu - shipped
    report = {
        "recipe": "v52 plus a thresholded copy increment on the ANCHORED fallback lane; main lane untouched",
        "formula": "yhat = yhat_v52 + max(p_kopie_thresholded**HOCH - p_z**HOCH, 0) * (versp - normal_anchor)",
        "schwelle": SCHWELLE, "hoch": H.HOCH, "anteil_hm": H.ANTEIL_HM, "kopie_tol_s": TOL,
        "base": str(BASE), "base_sha256": m52["candidate_sha256"], "base_score": m52["score"],
        "gate_g1_base_hash": g1, "gate_g2_anchor_hash": g2, "gate_g3_reproduction": g3, "gate_g4": g4,
        "rome_gate_rows": int(rome.sum()), "rome_overlap_with_increment_suppressed": overlap,
        "fires": {"rows": int(fires.sum()), "share": float(fires.mean()),
                  "max_versp_s": float(frame.versp_s.to_numpy()[fires].max()) if fires.any() else None,
                  "mean_delta": float(delta[fires].mean()) if fires.any() else None},
        "change_s": {"mean_abs": float(np.abs(d).mean()), "max": float(d.max()), "min": float(d.min()),
                     "p99_abs": float(np.quantile(np.abs(d), .99))},
        "validation_2025_anchored": {"tool": "research_280/kopie_anchored_sixpair.py",
                                     "fires_share": 0.0136, "precision": 0.765,
                                     "pooled_s": 1.219, "pairs": "6/6",
                                     "note": "the harness fails its own calibration probe; treat the sign as "
                                             "the claim and expect far less, roughly 0.4 s"},
        "candidate_sha256": sha(target),
        "created_at": datetime.now(timezone.utc).isoformat(), "uploaded": False}
    (OUT / "build_report_v53_c6.json").write_text(json.dumps(report, indent=2) + "\n")
    log("written " + str(target))
    print(json.dumps({k: report[k] for k in ("fires", "change_s", "gate_g3_reproduction",
                                             "rome_overlap_with_increment_suppressed")}, indent=1))


if __name__ == "__main__":
    main()

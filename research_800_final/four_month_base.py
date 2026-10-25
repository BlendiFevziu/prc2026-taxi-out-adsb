"""Final phase (14 October 2026): the frozen NM-free base applied to a ranking file of any months.

The final dataset (final_ranking.parquet) holds January, February, June and July 2026. Design is unchanged; only the
input file differs. This module reproduces, for an arbitrary ranking file:

  main lane     the frozen v45 all-year ensemble with its neural quarter replaced by the five origin-grouped
                neighbour sequence nets (v47), at the challenge-only weights sequence 0.35 / airport models 0.30
                (v50, v51 settings, without the NM add-on);
  fallback lane the v46 neighbour-taxi anchor (production encoding refit on all twelve 2025 months);
  Rome rules    v69 (delay above 8 h, no local-midnight crossing -> the delay) and v70 (1 h to 8 h delay: mean of the
                anchored prediction and the label-mechanism classifier's expected value).

No label of any 2026 month is read, no third-party NM data is used, nothing is uploaded.

Replication gates (run on the OLD ranking.parquet, where scored artefacts exist):
  G1  saved sequence-net residuals are reproduced by the inference path used here
  G2  with the v45 weights (0.25 / 0.20) the main lane reproduces the scored v47 candidate (max 0.05 s)
  G3  the anchored fallback reproduces .cache/research_240/anchor_preview_2026_production.parquet (max 0.05 s)
  G4  copy gate + v69 + v70 on the anchored fallback reproduce the v70 candidate's fallback rows (all 5,290)

Run from the repository root:
  python3 -u -B research_800_final/four_month_base.py --gates          # gates on the old ranking file
  python3 -u -B research_800_final/four_month_base.py --final          # final_ranking.parquet
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_500_midnight"))
sys.path.insert(0, str(ROOT / "research_510_rome_classifier"))
os.chdir(ROOT)

from research_250 import release_common as RC   # imports main_model first: torch before tree libraries
import main_model as H
import neural_model as NM  # noqa: F401
import features_geometry as MG
import features_stand_rate as MSQ
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import rule as RULE
import rome_classifier as ROMEC
from research_240 import features_neighbour_taxi as NT
from research_250 import candidate_features as B
from research_260 import sequence_features as F, sequence_inference as SI, sequence_model as S

OUT = ROOT / ".cache/research_800"
SEQ = ROOT / ".cache/sequence_release_v47/release"
MODEL = ROOT / ".cache/clock_queue_release_20260912/final/model.pkl"
SEEDS = [13, 21, 29, 37, 45]
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
T0 = time.time()


def log(message):
    print(f"[{time.time() - T0:7.0f}s] {message}", flush=True)


def features(path):
    """Production feature table of a ranking file (same builder and columns as release_common.dataset)."""
    frame = B.baue([Path(path)], H.MG.alle())
    keep = list(dict.fromkeys(H.MERKMALE_HAUPT + H.MERKMALE_RUECK + B.NEW_COLUMNS + [ID, "MVT_TIME_UTC_mvt", "y"]))
    frame = frame[[c for c in keep if c in frame]].copy()
    assert frame[ID].is_unique and frame.y.isna().all()
    return frame


def sequence_residuals(frame, raw_path):
    """Mean raw residual of the five sequence nets for the main-lane rows of `frame` (Series indexed by MVT id)."""
    hat = frame.hat_aobt.eq(1).to_numpy()
    main = frame[hat].reset_index(drop=True)
    packages = [joblib.load(SEQ / f"release_origin_{seed}.pkl") for seed in SEEDS]
    raw = pd.read_parquet(raw_path, columns=F.RAW)
    xk, xn = SI.encode(main, packages[0]["preprocessing"])
    log(f"neighbours for {len(main)} main rows")
    xs, ms = F.neighbours(raw, main[ID].to_numpy(), packages[0]["mode"], k=packages[0]["k"], window_min=90.)
    per_seed = {}
    for seed, package in zip(SEEDS, packages):
        per_seed[seed] = S.predict(package, xk, xn, xs, ms)
        log(f"seed {seed}: mean {per_seed[seed].mean():.2f}, sd {per_seed[seed].std():.2f}")
    mean = np.mean([per_seed[s] for s in SEEDS], axis=0)
    return pd.Series(mean, index=main[ID].to_numpy()), per_seed


def main_lane(frame, sequence, net_weight=None, airport_weight=None):
    """release_common.predict with the neural component replaced by the sequence residual, as the v47 build did."""
    model = joblib.load(MODEL)
    if net_weight is not None:
        model["anteil_netz"] = net_weight
    if airport_weight is not None:
        model["anteil_flh"] = airport_weight
    original = RC.NM.sage_voraus

    def substitute(net, x):
        return sequence.loc[frame.loc[x.index, ID].to_numpy()].to_numpy()

    RC.NM.sage_voraus = substitute
    try:
        values = RC.predict(model, frame)
    finally:
        RC.NM.sage_voraus = original
    return pd.Series(values, index=frame[ID].to_numpy())


def fallback_models():
    """The v46 production fallback: refit on the cached twelve-month table (as anchor_preview_2026_production.py)."""
    tables = ROOT / ".cache/research_240/anchor_preview"
    lo = pd.read_parquet(sorted(tables.glob("fallback_2025_all_*.parquet"))[0])
    base = list(H.MERKMALE_RUECK)
    with_nt = base + NT.MERKMALE

    def reg(rounds, leaves, minimum):
        return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05, num_leaves=leaves,
                                 min_child_samples=minimum, verbose=-1, n_jobs=4, random_state=0)

    lo["z"] = (lo.y > H.SCHWELLE_EXTREM).astype(int)
    quote = MSQ.passe_an(lo)
    lo = MSQ.ausser_faltung(lo)
    z0 = lo.z.eq(0).to_numpy()
    y = lo.y.to_numpy(float)
    classifier = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                                    min_child_samples=20, verbose=-1, n_jobs=4, random_state=0).fit(lo[base], lo.z)
    normal_model = reg(400, 31, 20).fit(lo.loc[z0, base], y[z0])
    height_model = reg(300, 15, 10).fit(lo.loc[~z0, base], y[~z0])
    anchor = lo.nt_anchor.to_numpy(float)
    rows = z0 & np.isfinite(anchor)
    offset = reg(400, 31, 20).fit(lo.loc[rows, with_nt], y[rows] - anchor[rows])
    return dict(quote=quote, classifier=classifier, normal=normal_model, height=height_model, offset=offset,
                base=base, with_nt=with_nt)


def fallback_table(path):
    """Fallback rows of a ranking file with the neighbour-taxi columns (as anchor_preview_2026.fallback_table)."""
    base = list(H.MERKMALE_RUECK)
    keep = list(dict.fromkeys([f for f in base if f not in MSQ.MERKMALE]
                              + [ID, "STAND_mvt", "MVT_TIME_UTC_mvt", "y", "monat", "hat_aobt"]))
    full = H.baue([str(path)], MG.alle())
    mask = full.hat_aobt.eq(0).to_numpy()
    built = NT.baue(full, query=mask)
    small = full.loc[mask, keep].copy()
    for col in NT.MERKMALE:
        small[col] = built[col].to_numpy()[mask]
    return small.reset_index(drop=True)


def fallback_lane(models, table):
    frame = MSQ.wende_an(models["quote"], table)
    base = models["base"]
    q = models["classifier"].predict_proba(frame[base])[:, 1] ** H.HOCH
    high = (1 - H.ANTEIL_HM) * frame.versp_min.to_numpy(float) * 60 + H.ANTEIL_HM * models["height"].predict(frame[base])
    normal = models["normal"].predict(frame[base])
    plain = np.clip((1 - q) * normal + q * high, 0, None)
    anchor = frame.nt_anchor.to_numpy(float)
    has = np.isfinite(anchor)
    normal_anchor = normal.copy()
    normal_anchor[has] = anchor[has] + models["offset"].predict(frame.loc[has, models["with_nt"]])
    anchored = np.clip((1 - q) * normal_anchor + q * high, 0, None)
    ids = frame[ID].to_numpy()
    return pd.Series(plain, index=ids), pd.Series(anchored, index=ids)


def _pick_threshold(probability, label, target):
    order = np.argsort(probability)[::-1]
    hits = np.cumsum(label[order])
    precision = hits / np.arange(1, len(order) + 1)
    ok = np.flatnonzero((precision >= target) & (hits >= 10))
    return 1.01 if len(ok) == 0 else float(probability[order][ok[-1]])


def rome_copy_gate(path):
    """The v49 surgical copy gate (research_280/build_rome_copy_candidate.py): fallback rows whose copy probability
    reaches the 0.99-precision threshold chosen inside the 2025 months get the schedule delay. Returns a Series
    indexed by MVT id (flagged rows only) and the threshold."""
    base = list(H.MERKMALE_RUECK)
    tables = ROOT / ".cache/research_240/rescreen_production"
    index = {tuple(sorted(json.loads(s.read_text())["source_months"])): s.with_suffix(".parquet")
             for s in tables.glob("rescreen_*.json")}
    ten = tuple(m for m in range(1, 13) if m not in (1, 7))
    train = pd.concat([pd.read_parquet(index[ten]), pd.read_parquet(index[(1, 7)])], ignore_index=True)
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    train["zk"] = (np.abs(train.y.to_numpy(float) - 60 * train.versp_min.to_numpy(float)) <= 60).astype(int)
    mapping = MSQ.passe_an(train)
    train = MSQ.ausser_faltung(train)
    categories = {}
    for column in base:
        if not pd.api.types.is_numeric_dtype(train[column]):
            values = train[column].astype(str)
            categories[column] = pd.unique(values)
            train[column] = pd.Categorical(values, categories=categories[column])
    label = train.zk.to_numpy()

    def classifier():
        return lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                                  min_child_samples=20, verbose=-1, n_jobs=4, random_state=0)

    inner = np.full(len(train), np.nan)
    for month in sorted(train.monat.unique()):
        hold = train.monat.eq(month).to_numpy()
        inner[hold] = classifier().fit(train.loc[~hold, base], label[~hold]).predict_proba(train.loc[hold, base])[:, 1]
    threshold = _pick_threshold(inner, label, .99)
    model = classifier().fit(train[base], label)
    rk = MSQ.wende_an(mapping, H.baue([str(path)], MG.alle()))
    fallback = rk.loc[rk.hat_aobt.eq(0)].copy()
    for column, cats in categories.items():
        fallback[column] = pd.Categorical(fallback[column].astype(str), categories=cats)
    flagged = fallback.loc[model.predict_proba(fallback[base])[:, 1] >= threshold]
    substitute = np.maximum(np.nan_to_num(60 * flagged.versp_min.to_numpy(float)), 0.0)
    return pd.Series(substitute, index=flagged[ID].to_numpy()), threshold


def rome_rules(path, fallback):
    """v69 and v70 on the fallback predictions of a ranking file. `fallback` is a Series indexed by MVT id."""
    extra = ["FLIGHT_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt"]
    k = pd.read_parquet(path, columns=RULE.COLS + extra)
    k = RULE.annotate(k[k.PHASE_mvt.eq("DEP") & k.ADEP_mvt.eq("LIRF") & k.AOBT_3_flt.isna()]).reset_index(drop=True)
    out = fallback.copy()
    rule_rows = RULE.selected(k)                                   # v69
    out.loc[k.loc[rule_rows, ID].to_numpy()] = k.loc[rule_rows, "delay"].to_numpy(float)
    # v70: training cell rows of 2025, classifier fitted on all of them
    d = pd.concat([pd.read_parquet(p, columns=RULE.COLS + [T] + extra) for p in sorted((ROOT / "data").glob("training_2025-*.parquet"))],
                  ignore_index=True)
    d = RULE.annotate(d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF") & d.AOBT_3_flt.isna()]).reset_index(drop=True)
    d = d[ROMEC.cell(d)].reset_index(drop=True)
    d["kind"] = RULE.label_kind(d[T], d.delay)
    cell = k[ROMEC.cell(k) & (k.delay <= 8 * 3600).to_numpy()].reset_index(drop=True)
    probs = ROMEC.fit_predict(ROMEC.features(d), d.kind, ROMEC.features(cell))
    t = float(d.loc[d.kind.eq("normal"), T].median())
    expected = ROMEC.expected(probs, cell.delay.to_numpy(float), t)
    old = out.loc[cell[ID].to_numpy()].to_numpy(float)
    out.loc[cell[ID].to_numpy()] = (old + expected) / 2
    info = {"v69_rows": int(rule_rows.sum()), "v70_rows": int(len(cell)), "training_rows": int(len(d))}
    return out, info


def compare(name, a, b, tolerance):
    both = a.index.intersection(b.index)
    d = (a.loc[both].to_numpy(float) - b.loc[both].to_numpy(float))
    res = {"gate": name, "rows": int(len(both)), "max_abs_s": float(np.abs(d).max()), "rms_s": float(np.sqrt(np.mean(d ** 2))),
           "tolerance_s": tolerance, "passed": bool(np.abs(d).max() <= tolerance)}
    log(json.dumps(res))
    return res


def gates():
    out, report = OUT, []
    out.mkdir(parents=True, exist_ok=True)
    release = ROOT / ".cache/clock_queue_release_20260912"
    frame = pd.read_parquet(release / "data/ranking.parquet")            # the cached features the v47 build used
    sequence, per_seed = sequence_residuals(frame, ROOT / "data/ranking.parquet")
    saved = pd.read_parquet(SEQ / "ranking_rows.parquet")[ID].to_numpy()
    ref13 = pd.Series(np.load(SEQ / "release_origin_13_ranking_residual.npy"), index=saved)
    mine13 = pd.Series(per_seed[13], index=sequence.index)
    report.append(compare("G1 seed 13 residual vs saved", mine13, ref13, 1e-3))
    v47 = pd.read_parquet(SEQ / "candidate.parquet").set_index(ID)[T]
    hat = frame.hat_aobt.eq(1).to_numpy()
    pred_v45w = main_lane(frame, sequence)
    main_ids = frame.loc[hat, ID].to_numpy()
    report.append(compare("G2 main lane at v45 weights vs scored v47 candidate", pred_v45w.loc[main_ids], v47.loc[main_ids], .05))
    models = fallback_models()
    table = pd.read_parquet(sorted((ROOT / ".cache/research_240/anchor_preview").glob("fallback_2026_ranking_*.parquet"))[0])
    plain, anchored = fallback_lane(models, table)
    ref = pd.read_parquet(ROOT / ".cache/research_240/anchor_preview_2026_production.parquet").set_index(ID)
    report.append(compare("G3 anchored fallback vs saved preview", anchored, ref.anchored, .05))
    report.append(compare("G3b plain fallback vs scored v45", plain, ref.v45_official, .05))
    gate_values, threshold = rome_copy_gate(ROOT / "data/ranking.parquet")
    gated = anchored.copy()
    gated.loc[gate_values.index] = gate_values.to_numpy()
    v68 = pd.read_parquet(ROOT / ".cache/research_480/v68/candidate_v68.parquet").set_index(ID)[T]
    report.append(compare(f"G4a copy gate ({len(gate_values)} rows, threshold {threshold:.4f}) vs v68 fallback rows", gated, v68, .05))
    rome, info = rome_rules(ROOT / "data/ranking.parquet", gated)
    v70 = pd.read_parquet(ROOT / ".cache/research_510/v70/candidate_v70.parquet").set_index(ID)[T]
    report.append(compare("G4b all fallback rows after gate + v69 + v70 vs v70 candidate", rome, v70, .05))
    (out / "gates.json").write_text(json.dumps({"gates": report, "rome": info}, indent=2) + "\n")
    return all(r["passed"] for r in report)


def final(path=ROOT / "data/final_ranking.parquet"):
    OUT.mkdir(parents=True, exist_ok=True)
    frame = features(path)
    log(f"features {frame.shape}")
    sequence, _ = sequence_residuals(frame, path)
    main = main_lane(frame, sequence, net_weight=.35, airport_weight=.30)
    models = fallback_models()
    table = fallback_table(path)
    plain, anchored = fallback_lane(models, table)
    gate_values, threshold = rome_copy_gate(path)
    gated = anchored.copy()
    gated.loc[gate_values.index] = gate_values.to_numpy()
    log(f"copy gate flags {len(gate_values)} rows (threshold {threshold:.4f})")
    rome, info = rome_rules(path, gated)
    info["copy_gate_rows"] = int(len(gate_values))
    hat = frame.hat_aobt.eq(1).to_numpy()
    ids = frame[ID].to_numpy()
    result = main.copy()
    fb_ids = frame.loc[~hat, ID].to_numpy()
    result.loc[fb_ids] = rome.loc[fb_ids].to_numpy()
    out = pd.DataFrame({ID: ids, T: result.loc[ids].to_numpy(float), "lane": np.where(hat, "main", "fallback")})
    out.to_parquet(OUT / "base_final_predictions.parquet", index=False)
    log(f"wrote {len(out)} rows; fallback rows {int((~hat).sum())}; rome {info}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.gates:
        ok = gates()
        log("ALL GATES PASSED" if ok else "GATE FAILED")
        sys.exit(0 if ok else 1)
    if args.final:
        final()

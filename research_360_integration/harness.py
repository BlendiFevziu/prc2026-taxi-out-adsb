"""Integration harness for a new main-lane base under the v62 pipeline. Research only: no upload, no NM input.

Two halves:
  2026  `ownership_v62()` assigns every scored row to exactly one owner (base, partial, stand, eham_january,
        direct_detector, fallback). `assemble_2026(base)` rebuilds the full submission from ANY 2026 main-lane base:
        base-owned rows take the base, the three residual layers are re-applied on top of it, detector and fallback
        rows are frozen from v62. With the current pre-NM base and the frozen models it must reproduce v62 exactly.
  2025  `policy_2025(base, tag)` gives the complete main-lane policy on held-out January/July rows for any base:
        layers applied on the substituted base, untouched rows at the base, detector rows set aside (base-free and
        identical in every arm), fallback frozen and scored separately. `compare(...)` pairs two policies with
        Codex's metric definitions (research_350_base/metrics.py).
Run the real-data reproduction checks from the repository root:  python3 -u -B research_360_integration/reproduce.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers as L  # noqa: E402

ID, T = L.ID, L.T
C = HERE / ".cache"
V62 = C / "research_320/v62/candidate_v62.parquet"
PRE_NM_BASE = C / "sequence_release_v49/release/candidate_main_w0.35_flh0.30.parquet"
PROVENANCE = HERE / "research_330_nmfree/private/reference_provenance.parquet"
REFERENCE = HERE / "research_330_nmfree/private/external_nm_free_reference.parquet"
CONTEXT_2026 = C / "codex_v56/partial_release/context.parquet"
VALID = C / "sequence_release_v47/validate"
FALLBACK_2025 = C / "research_280/rome_copy_hard_gate_rows.parquet"
NM_DAYS = ("06", "13", "20")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ 2026: ownership and assembly

def ownership_v62():
    """One owner per scored row, from Codex's provenance of the NM-free reference plus the NM-free layer rows."""
    prov = pd.read_parquet(PROVENANCE)[[ID, "airport", "date", "component"]]
    ref = pd.read_parquet(REFERENCE).set_index(ID)[T]
    ctx = pd.read_parquet(CONTEXT_2026)
    ctx["cell"] = ctx.airport + "/" + ctx.date.str[5:7]
    nm = ctx[ctx.date.str[8:10].isin(NM_DAYS)]
    at_comp = ref.loc[nm[ID]].to_numpy(float) == nm.comparator.to_numpy(float)
    # the NM-free build's own rule (research_320/build_nmfree_plus.py) for each layer on days 6, 13 and 20
    added = {"partial": nm[at_comp & nm.category.eq("partial_ground") & nm.cell.isin(L.LAYERS["partial"].cells)][ID],
             "stand": nm[at_comp & nm.category.eq("observed_at_assigned_stand") & nm.cell.isin(L.LAYERS["stand"].cells)][ID],
             "eham_january": nm[at_comp & nm.category.eq("partial_ground") & nm.cell.eq("EHAM/01")][ID]}
    feats = pd.concat([pd.read_parquet(C / f"codex_eham/eham_recovery/features_2026-01-{d}.parquet", columns=[ID]) for d in NM_DAYS])
    added["eham_january"] = added["eham_january"][added["eham_january"].isin(feats[ID])]   # inner join in the build
    names = {"challenge_main": "base", "partial_arm4": "partial", "stand_model": "stand", "eham_recovery": "eham_january",
             "direct_detector": "direct_detector", "fallback": "fallback"}
    own = prov.assign(owner=prov.component.map(names))
    for layer, ids in added.items():
        hit = own[ID].isin(ids)
        if (own.loc[hit, "owner"].ne("base") & own.loc[hit, "owner"].ne(layer)).any():
            raise ValueError(f"{layer}: an added row already belongs to another owner")
        own.loc[hit, "owner"] = layer
    if own.owner.isna().any() or not own[ID].is_unique:
        raise ValueError("ownership is incomplete or ambiguous")
    return own[[ID, "airport", "date", "owner"]]


def layer_frame_2026(layer, ids):
    """Inputs of one layer for its 2026 rows (context table plus the layer's own track columns)."""
    ctx = pd.read_parquet(CONTEXT_2026).set_index(ID)
    rows = ctx.loc[pd.Index(ids)].reset_index()
    if layer == "stand":
        days = pd.concat([pd.read_parquet(p) for p in sorted((C / "research_320/stand_candidate").glob("day_*.parquet"))]
                         + [pd.read_parquet(p) for p in sorted((C / "research_320/nmfree_plus").glob("day_*.parquet"))])
        det = days.drop_duplicates(ID).set_index(ID).detector_taxi_s
        rows["detector_taxi_s"] = det.loc[rows[ID]].to_numpy(float)
    if layer == "eham_january":
        feats = pd.concat([pd.read_parquet(p) for p in sorted((C / "codex_eham/eham_recovery").glob("features_2026-01-*.parquet"))])
        rows = rows.merge(feats[[ID] + list(L.EHAM_TRACK)], on=ID, how="left", validate="one_to_one")
    return rows


def assemble_2026(base, own=None, models=None, frozen=None):
    """Full 2026 submission from a main-lane base (Series indexed by ID, covering every main row).

    base-owned rows -> base; partial/stand/eham_january -> max(base + w * correction, 300) with the layer's models
    (frozen v62 models unless `models` maps a layer to refitted ones); direct_detector and fallback -> `frozen`
    (v62 by default). Returns the frame in template order.
    """
    own = ownership_v62() if own is None else own
    frozen = pd.read_parquet(V62) if frozen is None else frozen
    out = frozen[[ID, T]].copy().set_index(ID)
    base_rows = own.loc[own.owner.eq("base"), ID]
    out.loc[base_rows, T] = base.loc[base_rows].to_numpy(float)
    for layer in ("partial", "stand", "eham_january"):
        ids = own.loc[own.owner.eq(layer), ID]
        frame = L.substitute_base(layer_frame_2026(layer, ids), base)
        corr = L.correction(layer, frame, (models or {}).get(layer))
        out.loc[frame[ID], T] = L.apply(frame, corr, L.LAYERS[layer].weight)
    res = out.reset_index()
    res[T] = res[T].astype(frozen[T].dtype)
    return res


def current_base_2026():
    """The pre-NM .35/.30 ensemble for every scored row (Codex's source for restoring the NM rows)."""
    f = pd.read_parquet(PRE_NM_BASE)
    return f.set_index(ID)[T].astype(float)


# ------------------------------------------------------------------ 2025: complete policy and comparison

def components_2025():
    """Saved held-out January/July main components (the rows Codex's study also scores)."""
    base = pd.read_parquet(VALID / "ensemble_baseline.parquet")
    parts = pd.read_parquet(VALID / "components_main.parquet")
    seq = np.mean([pd.read_parquet(VALID / f"validate_origin_{s}_predictions.parquet").set_index(ID).loc[base[ID], "residual"].to_numpy(float)
                   for s in (13, 21, 29, 37, 45)], axis=0)
    rows = base[[ID, "MVT_TIME_UTC_mvt", "ADEP_mvt", "y", "ersatz_s", "gate", "schedule"]].merge(
        parts[[ID, "trees", "local", "cat"]], on=ID, validate="one_to_one")
    rows["sequence"] = seq
    rows["date"] = pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    rows["airport"] = rows.ADEP_mvt.astype(str)
    rows["source_month"] = pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True).dt.month   # calendar month; see PLAN.md
    return rows


def policy_2025(base, rows, layer_frames, models=None, set_aside=()):
    """Complete main-lane policy for held-out January/July rows under any base.

    base: Series ID -> base prediction covering `rows`. rows: the main-lane rows to score (all ten airports, e.g.
    components_2025() filtered to the evaluation dates). layer_frames: dict layer -> frame of the rows that layer
    owns on those dates, with its track columns (from the research_320 extraction caches; the frozen owner lists
    are identical in every arm). Every other row takes the base. set_aside: IDs of direct-detector rows, which are
    base-free and identical in every arm; they are removed and counted rather than scored.
    Returns (scored rows, prediction array aligned with them).
    """
    keep = rows[~rows[ID].isin(set(set_aside))].reset_index(drop=True)
    pred = pd.Series(base.loc[keep[ID]].to_numpy(float), index=keep[ID].to_numpy())
    for layer, frame in layer_frames.items():
        if not frame[ID].isin(pred.index).all():
            raise ValueError(f"{layer}: owned rows outside the scored rows")
        f = L.substitute_base(frame, base)
        pred.loc[f[ID].to_numpy()] = L.apply(f, L.correction(layer, f, (models or {}).get(layer)), L.LAYERS[layer].weight)
    return keep, pred.loc[keep[ID]].to_numpy(float)


def fallback_2025(dates=None):
    """Frozen held-out fallback (anchored fallback plus the Rome gate, fold 1_7): base-free, identical in every arm."""
    f = pd.read_parquet(FALLBACK_2025)
    f = f[f.fold.eq("1_7")].copy()
    f["date"] = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    f["prediction"] = f.candidate.astype(float)
    return (f if dates is None else f[f.date.isin(set(dates))]).reset_index(drop=True)


def trimmed(gain, n=100):
    return float(np.sort(gain)[:-n].mean()) if len(gain) > n else None


def bootstrap(rows, gain, draws=5000, seed=350):
    """Codex's definition: airport-day blocks resampled within source month, error sums and counts together."""
    table = rows[["source_month", "airport", "date"]].copy()
    table["gain"], table["n"] = gain, 1
    groups = table.groupby(["source_month", "airport", "date"], observed=True)[["gain", "n"]].sum()
    rng = np.random.default_rng(seed)
    numer, denom = np.zeros(draws), np.zeros(draws)
    for _, g in groups.groupby(level=0):
        idx = rng.integers(0, len(g), (draws, len(g)))
        numer += g.gain.to_numpy()[idx].sum(axis=1)
        denom += g.n.to_numpy()[idx].sum(axis=1)
    b = numer / denom
    return {"blocks": int(len(groups)), "ci95": np.quantile(b, [.025, .975]).tolist(), "ci98_75": np.quantile(b, [.00625, .99375]).tolist()}


def compare(rows, before, after, fallback=None):
    """Paired complete-policy comparison on identical rows; fallback scored separately and must be identical."""
    y = rows.y.to_numpy(float)
    e0, e1 = (np.asarray(before, float) - y) ** 2, (np.asarray(after, float) - y) ** 2
    gain = e0 - e1
    out = {"rows": int(len(y)), "rmse_before": float(np.sqrt(e0.mean())), "rmse_after": float(np.sqrt(e1.mean())),
           "mse_gain": float(gain.mean()), "best100_removed_mse_gain": trimmed(gain), "airport_day": bootstrap(rows, gain),
           "by_month": {int(m): {"rows": int((rows.source_month == m).sum()), "mse_gain": float(gain[(rows.source_month == m).to_numpy()].mean()),
                                  "best100_removed_mse_gain": trimmed(gain[(rows.source_month == m).to_numpy()])}
                        for m in sorted(rows.source_month.unique())},
           "by_airport": {a: float(gain[(rows.airport == a).to_numpy()].mean()) for a in sorted(rows.airport.unique())}}
    if fallback is not None:
        fb = (fallback.prediction.to_numpy(float) - fallback.y.to_numpy(float)) ** 2
        n = len(y) + len(fb)
        out["fallback"] = {"rows": int(len(fb)), "rmse": float(np.sqrt(fb.mean())), "identical_in_both_arms": True}
        out["combined_rmse_before"] = float(np.sqrt((e0.sum() + fb.sum()) / n))
        out["combined_rmse_after"] = float(np.sqrt((e1.sum() + fb.sum()) / n))
    return out

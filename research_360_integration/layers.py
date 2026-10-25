"""The base-dependent ADS-B layers of v62, as data: what each reads from the base, what it predicts, how it is applied.

A layer is a residual model on the base ("comparator"): it is fitted on y - base, reads the base itself as an input
column, and is applied as max(base + weight * mean_over_seeds(correction), 300). A new base therefore changes three
things for every layer here: an input column, the training target and the application point. Eligibility (which
rows a layer owns) comes from the ADS-B extraction and the frozen drift gates and does not depend on the base.

Layers that do NOT depend on the base and stay frozen: the direct stand detectors at Munich and Amsterdam (their
value is MVT minus a detected off-block, no base term) and the fallback lane (its own models, no main-lane base).
NM-free throughout: no third-party NM input anywhere.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
FLOOR = 300.0
SEEDS = (177, 277, 377)

# the 23 partial-track summaries (research_310/partial_track_features.FEATURES at 9884150) are imported lazily from
# the frozen checkout; the context columns are the base-dependent part
CONTEXT = ("comparator", "ersatz_s", "schedule", "gate", "hour_sin", "hour_cos")
AIRPORT_FLAGS = tuple(f"airport_{a}" for a in ("EDDF", "EGLL", "EHAM", "LEBL", "LSZH"))
EHAM_TRACK = ("first_age_s", "last_age_s", "span_s", "endpoint_distance_m", "weighted_speed_kt",
              "weighted_still_share", "observed_interval_share")
BASE_DEPENDENT_COLUMNS = ("comparator",)      # "gate" too, if a new base changes the reporting classifier


@dataclass(frozen=True)
class Layer:
    name: str
    category: str                  # ADS-B category of the rows it owns
    cells: tuple                   # airport/month cells it is deployed in (frozen drift gates)
    columns: str                   # "partial35", "stand36" or "eham14"
    weight: float
    trees: int
    leaves: int
    training: str                  # training rows, as a tag of the research_320 extraction caches
    models: str                    # folder and file pattern of the frozen seed models
    order: int                     # application order; owners are disjoint, so order only documents lineage
    evidence: str
    extra: dict = field(default_factory=dict)


LAYERS = {
    "partial": Layer("partial", "partial_ground",
                     ("EDDF/01", "EGLL/01", "EHAM/07", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07"),
                     "partial35", 1.0, 800, 31, "E: v56's 8 days + 6 extra (24,093 rows)",
                     ".cache/research_320/volume/models/c1_real_{seed}.pkl", 1,
                     "research_320/PREREG_capacity2.md (C1), v62"),
    "stand": Layer("stand", "observed_at_assigned_stand",
                   ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07"),
                   "stand36", 0.5, 200, 7, "S: v56's 8 days (2,619 rows)",
                   ".cache/research_320/stand/models/real_{seed}.pkl", 2,
                   "research_320/PREREG_stand.md, v57"),
    "eham_january": Layer("eham_january", "partial_ground", ("EHAM/01",), "eham14", 1.0, 200, 7,
                          "Codex's TRAIN dates, EHAM rows only",
                          ".cache/codex_eham/eham_recovery/real_{seed}.pkl", 3,
                          "research_310/EHAM_RECOVERY_RESULT.md (8d98953), research_320/PREREG_eham_weight.md, v59"),
}


def compose_base(rows, trees=None, cat=None, sequence=None, gate=None, airport_share=.3, neural_share=.35):
    """The pre-ADS-B main comparator from its saved components, in the exact arithmetic of
    research_310/supporting_evidence.comparator (9884150), which produced the `comparator` column every ADS-B layer
    was trained and evaluated on.

    Codex's research_350_base/metrics.compose computes the same blend with the weights multiplied out
    (.65 * (.4 * (.7 * trees + ...)) + .35 * sequence); that order differs from this one by up to ~2e-13 s on some
    rows. Harmless for scores, but exact-equality checks must use this function. Pass new arrays for any refitted
    family; every other component is taken from `rows`.
    """
    f = lambda v, col: rows[col].to_numpy(float) if v is None else np.asarray(v, float)
    trees, cat, sequence, gate = f(trees, "trees"), f(cat, "cat"), f(sequence, "sequence"), f(gate, "gate")
    tree = .4 * ((1 - airport_share) * trees + airport_share * rows.local.to_numpy(float)) + .6 * cat
    return np.maximum((1 - gate) * (rows.ersatz_s.to_numpy(float) + (1 - neural_share) * tree + neural_share * sequence)
                      + gate * rows.schedule.to_numpy(float), FLOOR)


def compose_codex(rows, trees=None, cat=None):
    """research_350_base/metrics.compose, transcribed (codex/v61-base-capacity d9cbef5), for tolerance checks only."""
    trees = rows.trees.to_numpy(float) if trees is None else np.asarray(trees, float)
    cat = rows.cat.to_numpy(float) if cat is None else np.asarray(cat, float)
    r = .65 * (.4 * (.7 * trees + .3 * rows.local.to_numpy(float)) + .6 * cat) + .35 * rows.sequence.to_numpy(float)
    return np.maximum((1 - rows.gate.to_numpy(float)) * (rows.ersatz_s.to_numpy(float) + r)
                      + rows.gate.to_numpy(float) * rows.schedule.to_numpy(float), FLOOR)


def substitute_base(frame, base, column="comparator"):
    """Replace the base column of a layer frame by a new base, matched by ID; every row must be covered."""
    base = pd.Series(np.asarray(base.values if isinstance(base, pd.Series) else base[column], float),
                     index=base.index if isinstance(base, pd.Series) else base[ID])
    if not base.index.is_unique:
        raise ValueError("duplicate IDs in the new base")
    missing = ~frame[ID].isin(base.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} rows have no new base prediction")
    out = frame.copy()
    out[column] = base.loc[frame[ID]].to_numpy(float)
    return out


def residual_target(frame, column="comparator"):
    """What every layer is fitted on: the error the base leaves."""
    return (frame.y - frame[column]).to_numpy(float)


def apply(frame, correction, weight, column="comparator"):
    """Application rule shared by all three layers."""
    return np.maximum(frame[column].to_numpy(float) + weight * np.asarray(correction, float), FLOOR)


def eham_inputs(frame):
    cols = list(CONTEXT) + list(EHAM_TRACK) + ["is_july"]
    return frame[cols]


def inputs(layer, frame):
    """The layer's input matrix in its frozen column order."""
    if LAYERS[layer].columns == "eham14":
        return eham_inputs(frame)
    import sys
    sys.path.insert(0, str(HERE))
    from research_320 import stand_experiment as S      # loads the frozen v56 modules
    if LAYERS[layer].columns == "partial35":
        return S.MS.inputs(frame, "real")
    return S.inputs(frame, "real")


def load_models(layer):
    import joblib
    return [joblib.load(HERE / LAYERS[layer].models.format(seed=s)) for s in SEEDS]


def correction(layer, frame, models=None):
    x = inputs(layer, frame)
    return np.mean([m.predict(x) for m in (models or load_models(layer))], axis=0)


def fit(layer, train, seeds=SEEDS, column="comparator", jobs=4):
    """Refit one layer with its frozen recipe on residuals of whatever base `train[column]` holds.

    Not run during this preparation (no training while Codex's study occupies the machine). The recipe is the
    deployed one: LightGBM regression, learning rate 0.03, min leaf 80, L2 100, row 0.8, column 0.9, max_bin 63.
    """
    import lightgbm as lgb
    spec = LAYERS[layer]
    x, y = inputs(layer, train), residual_target(train, column)
    return [lgb.LGBMRegressor(objective="regression", n_estimators=spec.trees, learning_rate=.03, num_leaves=spec.leaves,
                              min_child_samples=80, reg_lambda=100., subsample=.8, subsample_freq=1, colsample_bytree=.9,
                              max_bin=63, n_jobs=jobs, random_state=s, verbosity=-1).fit(x, y) for s in seeds]

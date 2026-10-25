"""Explicit research schema for a future time-corrected, queue-aware ensemble.

This module does not modify main_model globals, train weights, or select a
submission version. Legacy columns remain byte-for-byte available to the fallback
and scored v44. A retrained main model must use main_features(), not the old names.
"""
import pandas as pd

if __package__:
    from . import features_time_units as U
    from . import features_correct_queue as Q
else:
    import features_time_units as U
    import features_correct_queue as Q

SCHEMA_VERSION = "post_v44_time_queue_v1"
RAW_COLUMNS = list(dict.fromkeys(U.RAW_COLUMNS+Q.RAW_COLUMNS))
NEW_COLUMNS = U.MERKMALE+Q.MERKMALE


def main_features(legacy_main):
    """Preserve main-column order, replace six definitions, then append the queue."""
    if len(legacy_main) != len(set(legacy_main)):
        raise ValueError("Duplicate legacy main features")
    if not set(U.CORRECTED_COLUMNS) <= set(legacy_main):
        raise ValueError("Expected the six audited v44 columns")
    if set(legacy_main) & set(NEW_COLUMNS):
        raise ValueError("Pass the unmodified legacy feature list")
    return ["tu_"+c if c in U.CORRECTED_COLUMNS else c for c in legacy_main]+Q.MERKMALE


def attach(raw, dep):
    """Add candidate columns without changing any legacy values or their dtypes.

    raw and dep must be the same source-file split; dep includes every departure
    in that split, before filtering labels or main/fallback rows. As in the tested
    recipe, references include the query departure itself where a count says so.
    """
    if set(dep.columns) & set(NEW_COLUMNS):
        raise ValueError("Candidate columns already present; do not mix cached schemas")
    corrected = U.corrected_values(raw[U.RAW_COLUMNS], dep[U.DEP_COLUMNS])
    queue = Q.baue(raw[Q.RAW_COLUMNS], dep[Q.DEP_COLUMNS])
    out = dep.copy()
    for col in U.CORRECTED_COLUMNS:
        out["tu_"+col] = corrected[col].to_numpy()
    for col in Q.MERKMALE:
        out[col] = queue[col].to_numpy()
    return out


def baue(paths, geometry):
    """Build one complete source split, retaining the legacy fallback semantics."""
    import main_model as H
    paths = list(paths)
    dep = H.baue(paths, geometry)
    raw = pd.concat([pd.read_parquet(p, columns=RAW_COLUMNS) for p in paths], ignore_index=True)
    return attach(raw, dep)


def schema(legacy_main, legacy_fallback):
    return {"feature_schema": SCHEMA_VERSION,
            "merkmale_haupt": main_features(legacy_main),
            "merkmale_rueck": list(legacy_fallback),
            "legacy_columns_unchanged": True,
            "production_ready": False,
            "validation_required": "full ensemble with unchanged submission postprocessing"}

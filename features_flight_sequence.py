"""Local Network-Manager flight-ID sequence features.

``FLIGHT_ID_mvt`` is an independent sequence from the movement record.  Its absolute
value is a calendar proxy and must not be used.  Its deviation from neighbouring
departures at the same airport in movement-time order is stable between 2025 and the
2026 ranking data and adds predictive information for the AOBT/block reporting residual.

The builder intentionally sees only the table passed to it.  Thus training features are
built from the training movement table and ranking features from ranking.parquet, exactly
as they will be at inference time.
"""
import numpy as np
import pandas as pd


MERKMALE = [
    "fid_abw", "fid_abw2", "fid_abw3",
    "fid_luecke_vor", "fid_luecke_nach", "fid_zeit_vor", "fid_zeit_nach",
    "fid_rate_vor", "fid_rate_nach", "fid_span201",
]


def baue(dep):
    """Add centred, airport-local FLIGHT_ID deviations in MVT-time order.

    No targets or future scoring values are used.  Centring keeps the useful local event
    ordering while removing the non-transferable global calendar trend in the raw ID.
    """
    d = dep.copy()
    d["_i"] = np.arange(len(d))
    d["_fid"] = pd.to_numeric(d["FLIGHT_ID_mvt"], errors="coerce")
    d["_zeit"] = pd.to_datetime(d["MVT_TIME_UTC_mvt"]).astype("int64") / 1e9
    d = d.sort_values(["ADEP_mvt", "MVT_TIME_UTC_mvt"], kind="stable")
    g = d.groupby("ADEP_mvt", observed=True)
    for name, window, minimum in (("fid_abw", 41, 5),
                                  ("fid_abw2", 201, 20),
                                  ("fid_abw3", 1001, 100)):
        local_median = g["_fid"].transform(
            lambda values: values.rolling(window, center=True, min_periods=minimum).median()
        )
        d[name] = d._fid - local_median
    # The local spacing says how the Network-Manager ID sequence diverges from the
    # nearby airport movement sequence.  Both directions are permitted: every row of
    # the supplied ranking movement table is available when a submission is generated.
    d["fid_luecke_vor"] = g["_fid"].diff()
    d["fid_luecke_nach"] = g["_fid"].shift(-1) - d["_fid"]
    d["fid_zeit_vor"] = g["_zeit"].diff()
    d["fid_zeit_nach"] = g["_zeit"].shift(-1) - d["_zeit"]
    d["fid_rate_vor"] = d.fid_luecke_vor / d.fid_zeit_vor.clip(lower=1)
    d["fid_rate_nach"] = d.fid_luecke_nach / d.fid_zeit_nach.clip(lower=1)
    d["fid_span201"] = g["_fid"].transform(
        lambda values: values.rolling(201, center=True, min_periods=20).max()
        - values.rolling(201, center=True, min_periods=20).min()
    )
    # Bound only the most extreme raw spacing values.  The thresholds use the movement
    # table available to the builder, not a target, so the exact same definition works
    # independently for training and the ranking table.
    for name in MERKMALE[3:]:
        lower, upper = d[name].quantile([0.001, 0.999])
        d[name] = d[name].clip(lower, upper)
    return d.sort_values("_i", kind="stable").drop(columns=["_i", "_fid", "_zeit"])

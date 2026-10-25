"""Prespecified, target-free reporting context for the next PRC experiment.

Call ``baue`` on the complete *departure* input table before selecting main/fallback
rows. Build training and validation/ranking references independently. Only the five
columns in REQUIRED are read; no off-block time, target, or arrival label is used.

The 12 features describe missing-network-record clusters, movement timestamp batches,
within-day ordering, and neighboring flight-ID disorder. Unlike an individual FID
residual, these remain informative candidates when a departure has no own flight ID.
No fitting, absolute ID feature, clipping learned from labels, or cross-table state.
"""

import numpy as np
import pandas as pd


REQUIRED = (
    "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "FLIGHT_ID_mvt", "MVT_ID_mvt",
)
RAW_COLUMNS = list(REQUIRED)
MERKMALE = [
    "rc_missing_share_15m", "rc_missing_share_90m",
    "rc_missing_run_size", "rc_nearest_observed_fid_min",
    "rc_same_second_batch", "rc_mvt_second", "rc_round_minute_share_15m",
    "rc_sched_rank_displacement", "rc_mvtid_rank_displacement",
    "rc_fid_reversal_share_15m", "rc_neighbor_fid_absdev41",
    "rc_neighbor_fid_absdev201",
]


def _neighbor_mean(values, times_ns, radius_minutes):
    """Mean in +/- radius_minutes, excluding own value, >=2 observed neighbors."""
    radius_ns = int(radius_minutes * 60 * 1_000_000_000)
    left = np.searchsorted(times_ns, times_ns - radius_ns, side="left")
    right = np.searchsorted(times_ns, times_ns + radius_ns, side="right")
    valid = np.isfinite(values)
    total = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0))]
    number = np.r_[0, np.cumsum(valid)]
    numerator = total[right] - total[left] - np.where(valid, values, 0.0)
    denominator = number[right] - number[left] - valid
    result = np.full(len(values), np.nan)
    np.divide(numerator, denominator, out=result, where=denominator >= 2)
    return result


def _nearest_observed(times_ns, observed):
    """Minutes to nearest other departure carrying a flight ID."""
    n = len(times_ns)
    positions = np.arange(n)
    previous = np.maximum.accumulate(np.where(observed, positions, -1))
    previous = np.r_[-1, previous[:-1]]
    following = np.minimum.accumulate(np.where(observed, positions, n)[::-1])[::-1]
    following = np.r_[following[1:], n]
    before = np.full(n, np.inf)
    after = np.full(n, np.inf)
    valid = previous >= 0
    before[valid] = (times_ns[valid] - times_ns[previous[valid]]) / 60e9
    valid = following < n
    after[valid] = (times_ns[following[valid]] - times_ns[valid]) / 60e9
    result = np.minimum(before, after)
    # An empty night or omitted month is not a measurable local reporting outage.
    result[result > 360] = np.nan
    return result


def baue(dep):
    """Return a copy with 12 float32 features, preserving input index/order/columns.

    Clock windows are centered because all supplied ranking departures are observable.
    Their radius is 15/90 minutes. Event-order windows are confined to an airport's
    UTC calendar day. Missing-run counts break at gaps >6 hours. Missing movement
    times produce missing features; they are never interpreted as an ancient date.
    """
    absent = [name for name in REQUIRED if name not in dep]
    if absent:
        raise ValueError(f"Missing reporting-context input columns: {absent}")
    result = dep.copy()
    features = np.full((len(dep), len(MERKMALE)), np.nan, dtype=np.float32)
    if not len(dep):
        for k, name in enumerate(MERKMALE):
            result[name] = features[:, k]
        return result

    # Work with positional row IDs: duplicate or nonconsecutive input indices are safe.
    raw = dep.loc[:, REQUIRED].reset_index(drop=True).copy()
    # Parquet inputs can use microseconds; integer conversion retains their storage
    # unit. Normalize explicitly before comparing against nanosecond-sized windows.
    raw["_time"] = pd.to_datetime(raw.MVT_TIME_UTC_mvt, errors="coerce", utc=True).dt.as_unit("ns")
    raw["_sched"] = pd.to_datetime(raw.SCHED_TIME_UTC_mvt, errors="coerce", utc=True).dt.as_unit("ns")
    raw["_fid"] = pd.to_numeric(raw.FLIGHT_ID_mvt, errors="coerce")
    raw["_mid"] = pd.to_numeric(raw.MVT_ID_mvt, errors="coerce")
    # MVT ID breaks equal-clock ties deterministically; its raw value is not emitted.
    raw = raw[raw._time.notna()].sort_values(["_time", "_mid"], kind="stable")
    for _, airport in raw.groupby("ADEP_mvt", observed=True, dropna=False, sort=False):
        positions = airport.index.to_numpy()
        times = airport._time.astype("int64").to_numpy()
        fid = airport._fid.to_numpy(dtype=float, na_value=np.nan)
        missing = ~np.isfinite(fid)
        local = np.full((len(airport), len(MERKMALE)), np.nan)
        local[:, 0] = _neighbor_mean(missing.astype(float), times, 15)
        local[:, 1] = _neighbor_mean(missing.astype(float), times, 90)

        breaks = np.r_[True, (missing[1:] != missing[:-1]) | (np.diff(times) > 360 * 60e9)]
        run = np.cumsum(breaks) - 1
        sizes = np.bincount(run)
        local[:, 2] = np.where(missing, sizes[run], 0)
        local[:, 3] = _nearest_observed(times, ~missing)
        _, batch, counts = np.unique(times // 1_000_000_000, return_inverse=True, return_counts=True)
        local[:, 4] = counts[batch] - 1
        local[:, 5] = (times % 60_000_000_000) / 1e9
        local[:, 6] = _neighbor_mean((local[:, 5] == 0).astype(float), times, 15)

        day = airport._time.dt.floor("D").to_numpy()
        daily = pd.DataFrame({"day": day, "time": times, "sched": airport._sched.to_numpy(),
                              "mid": airport._mid.to_numpy(), "fid": fid})
        days = daily.groupby("day", sort=False, observed=True)
        n_day = days["time"].transform("size").to_numpy()
        time_rank = days["time"].rank(method="average").to_numpy()
        local[:, 7] = (time_rank - days["sched"].rank(method="average").to_numpy()) / n_day
        local[:, 8] = (time_rank - days["mid"].rank(method="average").to_numpy()) / n_day

        # Reversal is defined only on adjacent observed FIDs within 30 minutes;
        # otherwise a missing sequence or month boundary must remain unknown.
        reversal = np.full(len(airport), np.nan)
        observed_positions = np.flatnonzero(~missing)
        if len(observed_positions) > 1:
            a, b = observed_positions[:-1], observed_positions[1:]
            close = (times[b] - times[a]) <= 30 * 60e9
            reversal[b[close]] = (fid[b[close]] < fid[a[close]]).astype(float)
        local[:, 9] = _neighbor_mean(reversal, times, 15)

        median = days["fid"].transform(
            lambda s: s.rolling(201, center=True, min_periods=5).median()
        )
        daily["absdev"] = np.abs(daily.fid - median)
        for column, window in ((10, 41), (11, 201)):
            local[:, column] = daily.groupby("day", sort=False, observed=True).absdev.transform(
                lambda s: s.rolling(window, center=True, min_periods=5).median()
            ).to_numpy()
        features[positions, :] = local.astype(np.float32)

    for k, name in enumerate(MERKMALE):
        result[name] = features[:, k]
    return result


def _self_check():
    """Small contract checks; no competition data, fitting, or writes."""
    n = 48
    clock = pd.date_range("2025-01-04 10:00:00", periods=n, freq="30s")
    fid = np.arange(n, dtype=float) + 1000
    fid[16:20] = np.nan
    d = pd.DataFrame({"ADEP_mvt": ["LIRF"] * 40 + ["EHAM"] * 8,
                      "MVT_TIME_UTC_mvt": clock,
                      "SCHED_TIME_UTC_mvt": clock - pd.Timedelta(minutes=20),
                      "MVT_ID_mvt": np.arange(n) + 5000, "FLIGHT_ID_mvt": fid,
                      "y": np.arange(n), "BLOCK_TIME_UTC_mvt": clock})
    d.loc[2, "MVT_TIME_UTC_mvt"] = d.loc[1, "MVT_TIME_UTC_mvt"]
    d.loc[47, "MVT_TIME_UTC_mvt"] = pd.NaT
    d = d.sample(frac=1, random_state=7)
    d.index = pd.Index(np.arange(n) // 2, name="duplicate_index")
    original = d.copy(deep=True)
    built = baue(d)
    pd.testing.assert_frame_equal(d, original)
    pd.testing.assert_frame_equal(built[d.columns], d)
    assert built.index.equals(d.index) and len(built) == n
    assert len(MERKMALE) == len(set(MERKMALE)) == 12
    assert built.loc[d.FLIGHT_ID_mvt.isna(), "rc_missing_run_size"].eq(4).all()
    assert built.loc[d.MVT_TIME_UTC_mvt.isna(), MERKMALE].isna().all().all()
    assert built.rc_same_second_batch.max() == 1
    # Removing labels altogether must be identical, as must a constant ID shift.
    labels_removed = baue(d.loc[:, REQUIRED])
    pd.testing.assert_frame_equal(built[MERKMALE], labels_removed[MERKMALE])
    shifted = d.copy()
    shifted["FLIGHT_ID_mvt"] += 100_000_000
    shifted["MVT_ID_mvt"] += 900_000_000
    shifted["y"] = -123456
    shifted["BLOCK_TIME_UTC_mvt"] = pd.NaT
    pd.testing.assert_frame_equal(built[MERKMALE], baue(shifted)[MERKMALE])
    # Equal timestamps encoded by Parquet as microseconds must give equal features.
    micros = d.copy()
    for name in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"):
        micros[name] = pd.to_datetime(micros[name], utc=True).dt.as_unit("us")
    pd.testing.assert_frame_equal(built[MERKMALE], baue(micros)[MERKMALE])
    # A separate airport must never enter another airport's context.
    isolated = d[d.ADEP_mvt == "LIRF"]
    pd.testing.assert_frame_equal(built.loc[d.ADEP_mvt == "LIRF", MERKMALE], baue(isolated)[MERKMALE])
    assert baue(d.iloc[:0])[MERKMALE].shape == (0, 12)
    print("Reporting-context contract checks passed: order/index, labels, ID shift, ns/us equivalence, airport isolation, missing times, empty input.")


if __name__ == "__main__":
    _self_check()

"""Target-free time anomalies in movement-record order, for isolated experiments.

Invert the existing ID-in-time representation: sort records by movement ID and
measure their movement/scheduled time against neighbouring records. Absolute IDs
and absolute dates are never exposed as predictors. Reference tables must be built
separately for learning and held-out movements, including all departure rows.
"""
import numpy as np
import pandas as pd

RAW_COLUMNS = ["ADEP_mvt", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
MERKMALE = [f"rt_{kind}_dev{window}_min" for kind in ("mvt", "sched")
            for window in (41, 201)]


def baue(dep):
    d = dep[RAW_COLUMNS].copy()
    d["_position"] = np.arange(len(d))
    d["_id"] = pd.to_numeric(d.MVT_ID_mvt, errors="coerce")
    for kind, col in (("mvt", "MVT_TIME_UTC_mvt"), ("sched", "SCHED_TIME_UTC_mvt")):
        stamps = pd.to_datetime(d[col], utc=True, errors="coerce").dt.as_unit("ns")
        d["_" + kind] = stamps.astype("int64") / 60e9
        d.loc[stamps.isna(), "_" + kind] = np.nan
    # Stable tie-breaks make input permutation irrelevant when IDs are duplicated.
    d = d.sort_values(["ADEP_mvt", "_id", "_mvt", "_sched"], kind="stable")
    d = d.reset_index(drop=True)
    g = d.groupby("ADEP_mvt", observed=True, sort=False)
    for kind in ("mvt", "sched"):
        for window, minimum in ((41, 5), (201, 20)):
            local = g["_" + kind].transform(
                lambda s: s.rolling(window, center=True, min_periods=minimum).median())
            col = f"rt_{kind}_dev{window}_min"
            d[col] = (d["_" + kind] - local).astype("float32")
            d.loc[d._id.isna(), col] = np.nan
    result = d.sort_values("_position", kind="stable")[MERKMALE]
    result.index = dep.index
    return result


def _self_check():
    n = 251
    t = pd.date_range("2025-01-02", periods=n, freq="min", tz="UTC")
    dep = pd.DataFrame({"ADEP_mvt": "TEST", "MVT_ID_mvt": np.arange(n),
                        "MVT_TIME_UTC_mvt": t, "SCHED_TIME_UTC_mvt": t})
    dep.loc[125, "MVT_TIME_UTC_mvt"] += pd.Timedelta(hours=24)
    dep.index = np.arange(n)[::-1]
    expected = baue(dep)
    assert expected.index.equals(dep.index)
    assert expected.iloc[125].rt_mvt_dev41_min == 1439
    us = dep.copy()
    for col in RAW_COLUMNS[2:]:
        us[col] = us[col].dt.as_unit("us")
    pd.testing.assert_frame_equal(expected, baue(us))
    shifted = dep.copy()
    shifted.MVT_ID_mvt += 1000000
    for col in RAW_COLUMNS[2:]:
        shifted[col] += pd.Timedelta(days=370)
    pd.testing.assert_frame_equal(expected, baue(shifted))
    pd.testing.assert_frame_equal(expected.sort_index(),
                                  baue(dep.sample(frac=1, random_state=13)).sort_index())
    poisoned = dep.assign(TAXITIME_SEC_mvt=np.arange(n)*1e7,
                          BLOCK_TIME_UTC_mvt=t, y=-12345.)
    pd.testing.assert_frame_equal(expected, baue(poisoned))
    second = dep.copy()
    second.ADEP_mvt = "OTHER"
    second.MVT_TIME_UTC_mvt += pd.Timedelta(days=800)
    pd.testing.assert_frame_equal(expected, baue(pd.concat([dep, second])).iloc[:n])
    assert baue(dep.iloc[:0]).empty
    print("Record-time checks passed: units, ID/date shifts, ordering, labels, airports.")


if __name__ == "__main__":
    _self_check()

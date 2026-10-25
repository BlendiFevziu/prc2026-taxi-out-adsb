"""Clock-unit-safe ablation of the previously inactive runway-queue features."""
import pandas as pd
import features_queue as Q
if __package__:
    from . import features_time_units as U
else:
    import features_time_units as U

RAW_COLUMNS = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]
DEP_COLUMNS = ["ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]
MERKMALE = ["qt_"+name for name in Q.MERKMALE]


def baue(raw, dep):
    r = U.normalize_datetime_columns(raw[RAW_COLUMNS])
    d = U.normalize_datetime_columns(dep[DEP_COLUMNS])
    for frame in (r, d):
        for col in ("ADEP_mvt", "ADES_mvt", "RUNWAY_mvt"):
            if col in frame:
                frame[col] = frame[col].astype(object).fillna("(missing)").astype(str)
    result = Q.baue(r, d)[Q.MERKMALE].rename(columns=dict(zip(Q.MERKMALE, MERKMALE)))
    assert result.index.equals(dep.index)
    return result


def _self_check():
    t = pd.Timestamp("2025-01-01", tz="UTC")
    dep = pd.DataFrame({"ADEP_mvt": ["TEST"]*3, "RUNWAY_mvt": ["01"]*3,
                        "MVT_TIME_UTC_mvt": [t+pd.Timedelta(minutes=m) for m in [0,5,15]]})
    raw = dep.assign(PHASE_mvt="DEP", ADES_mvt="OTHER")[RAW_COLUMNS]
    built = baue(raw, dep)
    assert built.qt_bahn_vor_10.tolist() == [1,2,2]
    assert built.qt_bahn_nach_10.tolist() == [2,2,1]
    assert built.qt_bahn_luecke_vor.tolist() == [-1,300,600]
    assert built.qt_bahn_luecke_nach.tolist() == [300,600,-1]
    us_r, us_d = raw.copy(), dep.copy()
    for frame in (us_r, us_d):
        frame["MVT_TIME_UTC_mvt"] = frame.MVT_TIME_UTC_mvt.dt.as_unit("us")
    pd.testing.assert_frame_equal(built, baue(us_r, us_d))
    pd.testing.assert_frame_equal(built, baue(raw.assign(TAXITIME_SEC_mvt=-1234), dep.assign(y=-999)))
    print("Correct-clock queue checks passed: exact windows/gaps, units and label independence.")


if __name__ == "__main__":
    _self_check()

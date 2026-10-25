"""The rule of research_500_midnight/PREREG_midnight.md: shared definitions for the evidence, the build and its tests."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
TZ = "Europe/Rome"
MIN_DELAY_S = 28800.0
COLS = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]


def local_day(ts, tz=TZ):
    return pd.to_datetime(ts, utc=True).dt.tz_convert(tz).dt.normalize().dt.tz_localize(None)


def annotate(frame):
    """delay (MVT - SCHED, s) and crossing (days between the local date of MVT - 15 min and of SCHED)."""
    f = frame.copy()
    mvt, sched = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True), pd.to_datetime(f.SCHED_TIME_UTC_mvt, utc=True)
    f["delay"] = (mvt - sched).dt.total_seconds()
    f["crossing"] = (local_day(mvt - pd.Timedelta(minutes=15)) - local_day(sched)).dt.days
    return f


def selected(frame):
    """Rows the rule sets to their delay (frame must be annotated)."""
    return (frame.PHASE_mvt.eq("DEP") & frame.ADEP_mvt.eq("LIRF") & frame.AOBT_3_flt.isna()
            & frame.delay.gt(MIN_DELAY_S) & frame.crossing.eq(0)).to_numpy()


def label_kind(y, delay):
    """copy / shift24 / normal, as in the pre-registration (2025 only)."""
    y, delay = np.asarray(y, float), np.asarray(delay, float)
    return np.select([(y > 20 * 3600) & (y < 28 * 3600), np.abs(y - delay) <= 60], ["shift24", "copy"], "normal")

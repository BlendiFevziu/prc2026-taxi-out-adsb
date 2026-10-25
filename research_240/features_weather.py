"""Weather and operating-direction features for departures, label-free.

Weather comes from public METAR reports downloaded by research_240/fetch_metar.py (Iowa
Environmental Mesonet archive). Each departure gets the latest report at or before its runway
movement time (MVT_TIME_UTC_mvt) at its departure airport, if that report is at most three
hours old; otherwise its weather features are missing. Temperatures are in degrees C, wind in
knots, visibility in kilometres. Present-weather strings are parsed group by group into
two-letter tokens, so "SHSGRASN" reads as showers, snow grains, rain and snow.

Operating direction comes from the arrival rows of the same source table: the most common
two-digit arrival runway designator at the airport in the clock hour of the movement, when at
least five arrivals landed in that hour. Landings face the wind, so this is the direction in
use; departures can use dedicated runways (EDDF 18) and are not used for it. The headwind
component uses the designator times ten degrees as the runway heading.
"""
from pathlib import Path

import numpy as np
import pandas as pd

WEATHER = ["wx_temp_c", "wx_dewpoint_spread_c", "wx_wind_kt", "wx_gust_excess_kt",
           "wx_visibility_km", "wx_frozen", "wx_freezing", "wx_icing_risk"]
DIRECTION = ["od_dir", "od_purity", "od_minutes_since_change", "od_headwind_kt"]
FROZEN = {"SN", "SG", "PL", "GS", "GR", "IC"}
MAX_AGE = pd.Timedelta(hours=3)
MIN_ARRIVALS = 5


def utc_ns(values):
    """Timestamps as a UTC nanosecond Series with a fresh RangeIndex."""
    stamps = pd.to_datetime(pd.Series(values).reset_index(drop=True), utc=True, errors="coerce")
    return stamps.dt.as_unit("ns")


def phenomena(codes):
    """Two-letter present-weather tokens of a METAR weather string, descriptors included."""
    tokens = set()
    if not isinstance(codes, str):
        return tokens
    for group in codes.split():
        group = group.lstrip("+-")
        if group.startswith("VC"):
            group = group[2:]
        if group in ("M", "NSW"):
            continue
        for i in range(0, len(group) - len(group) % 2, 2):
            tokens.add(group[i:i + 2])
    return tokens


def load_metar(folder):
    frames = [pd.read_csv(path, na_values=["M"], keep_default_na=False, dtype={"wxcodes": str})
              for path in sorted(Path(folder).glob("metar_*.csv"))]
    if not frames:
        raise FileNotFoundError(f"no metar_*.csv in {folder}")
    metar = pd.concat(frames, ignore_index=True)
    metar["valid"] = utc_ns(metar.valid)
    return metar


def report_features(metar):
    """Weather features per report, keyed by airport and report time, sorted by time."""
    tmpf = pd.to_numeric(metar.tmpf, errors="coerce").to_numpy(float)
    dwpf = pd.to_numeric(metar.dwpf, errors="coerce").to_numpy(float)
    sknt = pd.to_numeric(metar.sknt, errors="coerce").to_numpy(float)
    gust = pd.to_numeric(metar.gust, errors="coerce").to_numpy(float)
    tokens = [phenomena(codes) for codes in metar.wxcodes]
    out = pd.DataFrame({"airport": metar.station.astype(str).to_numpy()})
    out["valid"] = utc_ns(metar.valid)
    out["wx_temp_c"] = (tmpf - 32) * 5 / 9
    out["wx_dewpoint_spread_c"] = (tmpf - dwpf) * 5 / 9
    out["wx_wind_kt"] = sknt
    out["wx_gust_excess_kt"] = np.where(np.isnan(sknt), np.nan, np.maximum(np.where(np.isnan(gust), sknt, gust) - sknt, 0))
    out["wx_visibility_km"] = pd.to_numeric(metar.vsby, errors="coerce").to_numpy(float) * 1.609344
    out["wx_frozen"] = [float(bool(t & FROZEN)) for t in tokens]
    out["wx_freezing"] = [float("FZ" in t) for t in tokens]
    known = ~(np.isnan(out.wx_temp_c.to_numpy()) | np.isnan(out.wx_dewpoint_spread_c.to_numpy()))
    icing = (out.wx_temp_c.to_numpy() <= 3) & (out.wx_dewpoint_spread_c.to_numpy() <= 3)
    out["wx_icing_risk"] = np.where(known, icing.astype(float), np.nan)
    out["wind_dir_deg"] = pd.to_numeric(metar.drct, errors="coerce").to_numpy(float)
    return out.dropna(subset=["valid"]).sort_values(["valid", "airport"], kind="mergesort").reset_index(drop=True)


def weather_for(airports, times, reports, max_age=MAX_AGE):
    """Latest report at or before each time at the same airport, at most max_age old."""
    left = pd.DataFrame({"airport": pd.Series(airports).astype(str).to_numpy(), "_row": np.arange(len(airports))})
    left["time"] = utc_ns(times)
    columns = WEATHER + ["wind_dir_deg"]
    out = pd.DataFrame(np.nan, index=np.arange(len(left)), columns=columns + ["wx_age_min"])
    ok = left.time.notna().to_numpy()
    if not ok.any():
        return out
    right = reports[["airport", "valid"] + columns].sort_values("valid", kind="mergesort")
    joined = pd.merge_asof(left[ok].sort_values("time", kind="mergesort"), right, left_on="time", right_on="valid",
                           by="airport", direction="backward", tolerance=max_age)
    rows = joined._row.to_numpy()
    for col in columns:
        out.loc[rows, col] = joined[col].to_numpy(float)
    out.loc[rows, "wx_age_min"] = (joined.time - joined.valid).dt.total_seconds().to_numpy() / 60
    return out


def operating_direction(arrivals):
    """Per airport and clock hour: modal arrival designator, its purity and whether it changed."""
    frame = pd.DataFrame({"airport": arrivals.ADES_mvt.astype(str).to_numpy()})
    frame["hour"] = utc_ns(arrivals.MVT_TIME_UTC_mvt).dt.floor("h")
    frame["dir"] = pd.to_numeric(arrivals.RUNWAY_mvt.astype(str).str.extract(r"^(\d{2})", expand=False),
                                 errors="coerce").to_numpy()
    frame = frame.dropna(subset=["hour", "dir"])
    counts = frame.groupby(["airport", "hour", "dir"]).size().rename("n").reset_index()
    counts["total"] = counts.groupby(["airport", "hour"]).n.transform("sum")
    counts["purity"] = counts.n / counts.total
    top = (counts.sort_values(["airport", "hour", "n", "dir"], ascending=[True, True, False, True], kind="mergesort")
           .drop_duplicates(["airport", "hour"]))
    top = top[top.total >= MIN_ARRIVALS].sort_values(["airport", "hour"], kind="mergesort").reset_index(drop=True)
    previous = top.groupby("airport").dir.shift()
    top["changed"] = previous.notna() & (top.dir != previous)
    return top[["airport", "hour", "dir", "purity", "total", "changed"]]


def direction_for(airports, times, directions, wind_dir_deg=None, wind_kt=None):
    """Operating direction at each time: designator, purity, minutes since the last change, headwind."""
    left = pd.DataFrame({"airport": pd.Series(airports).astype(str).to_numpy(), "_row": np.arange(len(airports))})
    left["time"] = utc_ns(times)
    left["hour"] = left.time.dt.floor("h")
    merged = left.merge(directions[["airport", "hour", "dir", "purity"]], on=["airport", "hour"], how="left")
    merged = merged.sort_values("_row", kind="mergesort")
    out = pd.DataFrame({"od_dir": merged.dir.to_numpy(float), "od_purity": merged.purity.to_numpy(float)})
    since = np.full(len(left), np.nan)
    changes = (directions.loc[directions.changed, ["airport", "hour"]].rename(columns={"hour": "change"})
               .sort_values("change", kind="mergesort"))
    ok = left.time.notna().to_numpy()
    if len(changes) and ok.any():
        joined = pd.merge_asof(left[ok].sort_values("time", kind="mergesort"), changes, left_on="time",
                               right_on="change", by="airport", direction="backward")
        since[joined._row.to_numpy()] = (joined.time - joined.change).dt.total_seconds().to_numpy() / 60
    out["od_minutes_since_change"] = since
    if wind_dir_deg is not None:
        heading = out.od_dir.to_numpy(float) * 10
        out["od_headwind_kt"] = np.asarray(wind_kt, float) * np.cos(np.radians(np.asarray(wind_dir_deg, float) - heading))
    return out

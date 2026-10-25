"""Label-free runway-configuration and destination-bearing columns for the main lane (research_370).

Pre-registered in research_370_konfig/PREREG_konfig.md before any fit. Reads only PHASE_mvt, ADEP_mvt, ADES_mvt,
RUNWAY_mvt and MVT_TIME_UTC_mvt, never a label; all five are 100 % filled in the ranking file, so every column is
available for the scored rows. Computed per calendar month, the unit of both the training files and the ranking file
(January and July 2026 complete), so a month's first hours lack history in training and in 2026 alike.

  kf_konfig            arrival runway set of the departure's clock hour at its airport: the runways carrying at least
                       20 % of that hour's arrivals, sorted and joined with '+'; '(wenig)' below five arrivals.
                       Categorical.
  kf_seit_umkehr_min   minutes since the airport's arrival flow last reversed, capped at 720
  kf_bis_umkehr_min    minutes until it next reverses, capped at 720
  kf_peil_sin/_cos     initial great-circle bearing ADEP -> ADES from OurAirports coordinates (public domain)

Flow. An arrival lies on side A if its runway heading is less than 90 degrees from the airport's principal arrival
heading (PRINCIPAL), else on side B. PRINCIPAL is the most used arrival runway over the twelve 2025 training files,
counted label-free on 29 September 2026 and fixed here, because a per-table mode could flip the axis at LFPG, where
26L and 08R carry 32 and 29 % of arrivals. An hour with at least five arrivals has flow A if at least half of them lie
on side A. A reversal is such an hour whose flow differs from the previous such hour at the same airport; its time is
the first arrival of that hour on the new side. Arrivals rather than departures, because EDDF 18 is a departures-only
runway (betriebsrichtung.py). By flow, every hour is 99 to 100 % one-sided at all ten airports in 2025, with 0.29
(EGLL) to 2.17 (LSZH) reversals per day; runway sets change far more often (EHAM 10.8 per day), mostly between one
and two runways of the same flow, which is why the reversal columns use the flow and the category uses the set.
"""
import hashlib

import numpy as np
import pandas as pd

MERKMALE = ["kf_konfig", "kf_seit_umkehr_min", "kf_bis_umkehr_min", "kf_peil_sin", "kf_peil_cos"]
KATEGORISCH = ["kf_konfig"]
RAW_COLUMNS = ["PHASE_mvt", "ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]
DEP_COLUMNS = ["ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]
PRINCIPAL = {"EDDF": 250, "EDDM": 260, "EGLL": 90, "EHAM": 180, "LEBL": 240,     # degrees, from 25L, 26R, 09L, 18R, 24R,
             "LEMD": 320, "LFPG": 260, "LIRF": 160, "LSZH": 140, "LTFM": 350}    # 32R, 26L, 16L, 14, 35R
MIN_ARR, SET_SHARE, CAP_MIN, WENIG = 5, 0.2, 720.0, "(wenig)"
COORDS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def seconds(values):
    """UTC seconds as float, NaN where missing; independent of the datetime unit (ns or us)."""
    t = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    return ((t - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(float)


def month_key(values):
    t = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    return (t.dt.year * 100 + t.dt.month).to_numpy()


def runway(values):
    """Normalised designator ('7' -> '07', '25l' -> '25L') and heading in degrees; missing for 'H', 'HEL', 'NA'."""
    s = pd.Series(values).astype(str).str.strip().str.upper().reset_index(drop=True)
    m = s.str.extract(r"^(\d{1,2})([LCR]?)$")
    num = pd.to_numeric(m[0], errors="coerce")
    ok = num.between(1, 36)
    name = pd.Series(np.where(ok, num.fillna(0).astype(int).map("{:02d}".format) + m[1].fillna(""), None))
    return name, (num * 10 % 360).where(ok)


def hours(raw):
    """One row per airport and clock hour with arrivals: n, runway set, flow (1 = side A, 0 = side B, NaN if unknown)
    and the first arrival time on each side."""
    ph = raw.PHASE_mvt.astype(str).str.upper().str.startswith("ARR").to_numpy()
    a = raw[ph]
    name, hdg = runway(a.RUNWAY_mvt.to_numpy())
    f = a.ADES_mvt.astype(str).to_numpy()
    t = seconds(a.MVT_TIME_UTC_mvt.to_numpy())
    arr = pd.DataFrame({"f": f, "t": t, "bahn": name.to_numpy(), "hdg": hdg.to_numpy(float)})
    arr = arr[arr.bahn.notna() & np.isfinite(arr.t)].copy()
    arr["slot"] = arr.t // 3600 * 3600
    princ = arr.f.map(PRINCIPAL).astype(float)
    diff = ((arr.hdg - princ + 180) % 360 - 180).abs()
    arr["side_a"] = np.where(princ.notna(), (diff < 90).astype(float), np.nan)
    keys = ["f", "slot"]
    h = arr.groupby(keys).size().rename("n").to_frame()
    c = arr.groupby(keys + ["bahn"]).size()
    share = c / c.groupby(level=[0, 1]).transform("sum")
    main = share[share >= SET_SHARE].reset_index().groupby(keys).bahn.agg(lambda s: "+".join(sorted(s)))
    every = arr.groupby(keys).bahn.agg(lambda s: "+".join(sorted(set(s))))
    h["konfig"] = main.reindex(h.index).fillna(every.reindex(h.index))
    h.loc[h.n < MIN_ARR, "konfig"] = WENIG
    h["share_a"] = arr[arr.side_a.notna()].groupby(keys).side_a.mean().reindex(h.index)
    h["flow"] = np.where((h.n >= MIN_ARR) & h.share_a.notna(), (h.share_a >= 0.5).astype(float), np.nan)
    h["first_a"] = arr[arr.side_a.eq(1)].groupby(keys).t.min().reindex(h.index)
    h["first_b"] = arr[arr.side_a.eq(0)].groupby(keys).t.min().reindex(h.index)
    return h


def reversals(h):
    """Sorted reversal times per airport."""
    v = h[h.flow.notna()].reset_index().sort_values(["f", "slot"])
    prev = v.groupby("f").flow.shift()
    rev = v[prev.notna() & v.flow.ne(prev)]
    times = pd.Series(np.where(rev.flow.eq(1), rev.first_a, rev.first_b), index=rev.f.to_numpy())
    return {f: np.sort(x.to_numpy(float)) for f, x in times.groupby(level=0)}


def _configuration(dep, raw):
    out = pd.DataFrame(index=dep.index, columns=MERKMALE[:3], dtype=object)
    out[MERKMALE[1:3]] = np.nan
    out["kf_konfig"] = WENIG
    t_dep = seconds(dep.MVT_TIME_UTC_mvt.to_numpy())
    m_dep, m_raw = month_key(dep.MVT_TIME_UTC_mvt.to_numpy()), month_key(raw.MVT_TIME_UTC_mvt.to_numpy())
    f_dep = dep.ADEP_mvt.astype(str).to_numpy()
    for month in pd.unique(m_dep[~pd.isna(m_dep)]):
        h = hours(raw[m_raw == month])
        rev = reversals(h)
        konfig = h.konfig.to_dict()
        for f in pd.unique(f_dep[m_dep == month]):
            i = np.flatnonzero((m_dep == month) & (f_dep == f) & np.isfinite(t_dep))
            t = t_dep[i]
            out.iloc[i, 0] = [konfig.get((f, s), WENIG) for s in t // 3600 * 3600]
            if f not in PRINCIPAL:
                continue
            r = rev.get(f, np.array([], float))
            p = np.searchsorted(r, t, "right")                       # reversals at or before the departure
            seit = np.where(p > 0, (t - r[np.maximum(p - 1, 0)]) / 60 if len(r) else CAP_MIN, CAP_MIN)
            bis = np.where(p < len(r), (r[np.minimum(p, len(r) - 1)] - t) / 60 if len(r) else CAP_MIN, CAP_MIN)
            out.iloc[i, 1] = np.minimum(seit, CAP_MIN)
            out.iloc[i, 2] = np.minimum(bis, CAP_MIN)
    out[MERKMALE[1:3]] = out[MERKMALE[1:3]].astype(float)
    out["kf_konfig"] = out.kf_konfig.astype(str)
    return out


def load_coords(path):
    """ICAO code -> (lat, lon) from OurAirports airports.csv; closed airports dropped; within a column the first row
    wins, and icao_code takes priority over gps_code over ident."""
    c = pd.read_csv(path, usecols=["ident", "type", "latitude_deg", "longitude_deg", "icao_code", "gps_code"],
                    dtype=str, keep_default_na=False)
    c = c[c.type.ne("closed")].copy()
    c["lat"] = pd.to_numeric(c.latitude_deg, errors="coerce")
    c["lon"] = pd.to_numeric(c.longitude_deg, errors="coerce")
    c = c[c.lat.notna() & c.lon.notna()]
    out = {}
    for col in ("ident", "gps_code", "icao_code"):                   # later columns overwrite earlier ones
        s = c[c[col].str.strip().str.len() > 0].drop_duplicates(col, keep="first")
        out.update({code.strip().upper(): (la, lo) for code, la, lo in zip(s[col], s.lat, s.lon)})
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bearing(adep, ades, coords):
    """sin and cos of the initial great-circle bearing; NaN where a code is unknown or both codes are equal."""
    adep = pd.Series(adep).astype(str).str.strip().str.upper().to_numpy()
    ades = pd.Series(ades).astype(str).str.strip().str.upper().to_numpy()
    look = lambda codes: np.array([coords.get(c, (np.nan, np.nan)) for c in codes], float).reshape(-1, 2)
    p1, p2 = np.radians(look(adep)), np.radians(look(ades))
    dl = p2[:, 1] - p1[:, 1]
    y = np.sin(dl) * np.cos(p2[:, 0])
    x = np.cos(p1[:, 0]) * np.sin(p2[:, 0]) - np.sin(p1[:, 0]) * np.cos(p2[:, 0]) * np.cos(dl)
    theta = np.arctan2(y, x)
    bad = ~np.isfinite(theta) | (adep == ades)
    return np.where(bad, np.nan, np.sin(theta)), np.where(bad, np.nan, np.cos(theta))


def attach(dep, raw, coords):
    """The five kf_ columns for the departures in dep, same index. raw: movements of the same calendar months (only
    arrivals and only RAW_COLUMNS are read). coords: load_coords() output."""
    out = _configuration(dep[DEP_COLUMNS], raw[RAW_COLUMNS])
    out["kf_peil_sin"], out["kf_peil_cos"] = bearing(dep.ADEP_mvt.to_numpy(), dep.ADES_mvt.to_numpy(), coords)
    assert out.index.equals(dep.index) and list(out.columns) == MERKMALE
    return out

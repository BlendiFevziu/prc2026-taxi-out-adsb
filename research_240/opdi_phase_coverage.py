"""Coverage of OPDI flight events for competition departures, and the phase durations they imply.

Research check, no model. OPDI v0.0.2 flight events (EUROCONTROL PRC with the OpenSky Network, derived from
open ADS-B; code MIT, data license not stated) for 5-15 January 2025, matched to the competition's 2025
departures at the ten airports by ICAO callsign (NM callsign, else movement flight), departure airport and
take-off time within 5 minutes.

Per matched departure: T = OPDI take-off; R = last OPDI runway entry at that airport within 15 minutes before
T; P = last OPDI parking-position exit within 3 hours before R; de-icing when a de-icing pad entry lies between
P (or R - 1 h) and R. Phases where observed: P to R (pushback, taxi and queue) and R to T (line-up and take-off
roll); alignment with competition timestamps: T - MVT_TIME, P - AOBT_3 and P - BLOCK_TIME.

Usage: python3 opdi_phase_coverage.py <opdi events parquet> <competition training parquet> <output json>
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OPDI, TRAIN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
TYPES = ["take-off", "entry-runway", "exit-parking_position", "entry-deicing_pad"]
START, END = pd.Timestamp("2025-01-05"), pd.Timestamp("2025-01-15")


def parse(text):
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        return json.loads(text)
    except ValueError:
        return {}


ev = pq.read_table(OPDI, columns=["flight_id", "type", "event_time", "info"], filters=[("type", "in", TYPES)]).to_pandas()
info = ev["info"].map(parse)
ev["airport"] = info.map(lambda d: d.get("osm_airport"))
ev["ref"] = info.map(lambda d: d.get("osm_ref"))
ev["cs"] = info.map(lambda d: (d.get("osn_flight_id") or "").strip().upper())
ev["event_time"] = pd.to_datetime(ev.event_time)
callsign = ev[ev.cs.ne("")].groupby("flight_id").cs.first()
counts = {t: int(n) for t, n in ev.type.value_counts().items()}

takeoff = ev[ev.type.eq("take-off")][["flight_id", "event_time"]].rename(columns={"event_time": "T"})
runway = (ev[ev.type.eq("entry-runway") & ev.airport.isin(AIRPORTS)][["flight_id", "event_time", "airport", "ref"]]
          .rename(columns={"event_time": "R", "ref": "runway_ref"}))
m = takeoff.merge(runway, on="flight_id")
m = m[(m.R <= m["T"]) & (m["T"] - m.R <= pd.Timedelta(minutes=15))]
dep = m.sort_values("R").groupby(["flight_id", "T"], as_index=False).last()
parking = ev[ev.type.eq("exit-parking_position")][["flight_id", "event_time", "ref"]].rename(columns={"event_time": "P", "ref": "stand_ref"})
cand = dep.merge(parking, on="flight_id")
cand = cand[(cand.P <= cand.R) & (cand.R - cand.P <= pd.Timedelta(hours=3))]
dep = dep.join(cand.sort_values("P").groupby(["flight_id", "T"]).agg(P=("P", "last"), stand_ref=("stand_ref", "last")), on=["flight_id", "T"])
deice = ev[ev.type.eq("entry-deicing_pad")][["flight_id", "event_time"]].rename(columns={"event_time": "D"})
cd = dep.merge(deice, on="flight_id")
cd = cd[(cd.D >= cd.P.fillna(cd.R - pd.Timedelta(hours=1))) & (cd.D <= cd.R)]
dep["deicing"] = pd.MultiIndex.from_frame(dep[["flight_id", "T"]]).isin(pd.MultiIndex.from_frame(cd[["flight_id", "T"]]))
dep["cs"] = dep.flight_id.map(callsign).fillna("")

cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"]
comp = pd.read_parquet(TRAIN, columns=cols)
comp = comp[comp.PHASE_mvt.astype(str).str.upper().str.startswith("DEP").to_numpy()].copy()
for c in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"):
    comp[c] = pd.to_datetime(comp[c], utc=True, errors="coerce").dt.tz_convert(None)
comp = comp[(comp.MVT_TIME_UTC_mvt >= START) & (comp.MVT_TIME_UTC_mvt < END)].copy()
comp["airport"] = comp.ADEP_mvt.astype(str)
nm = comp.CALLSIGN_flt.astype(object)
comp["cs"] = nm.where(nm.notna(), comp.FLIGHT_mvt.astype(object)).astype(str).str.strip().str.upper()

pairs = comp.merge(dep, on=["airport", "cs"], suffixes=("", "_opdi"))
pairs["dt_takeoff_s"] = (pairs["T"] - pairs.MVT_TIME_UTC_mvt).dt.total_seconds()
pairs = pairs[pairs.dt_takeoff_s.abs() <= 300]
pairs = pairs.loc[pairs.dt_takeoff_s.abs().groupby(pairs.MVT_ID_mvt).idxmin()].copy()
pairs["runway_to_takeoff_s"] = (pairs["T"] - pairs.R).dt.total_seconds()
pairs["stand_exit_to_runway_s"] = (pairs.R - pairs.P).dt.total_seconds()
pairs["stand_exit_minus_aobt3_s"] = (pairs.P - pairs.AOBT_3_flt).dt.total_seconds()
pairs["stand_exit_minus_block_s"] = (pairs.P - pairs.BLOCK_TIME_UTC_mvt).dt.total_seconds()
pairs["airport_taxi_s"] = (pairs.MVT_TIME_UTC_mvt - pairs.BLOCK_TIME_UTC_mvt).dt.total_seconds()


def med(series):
    series = series.dropna()
    return float(series.median()) if len(series) else None


report = {"note": __doc__.split("\n\n")[0], "window": [str(START.date()), str(END.date())], "opdi_event_counts_selected_types": counts,
          "airports": {}}
for airport in AIRPORTS + ["ALL"]:
    c = comp if airport == "ALL" else comp[comp.airport == airport]
    p = pairs if airport == "ALL" else pairs[pairs.airport == airport]
    report["airports"][airport] = {
        "competition_departures": int(len(c)), "with_opdi_takeoff_and_runway_entry": int(len(p)),
        "coverage_takeoff_runway": float(len(p) / max(len(c), 1)),
        "coverage_stand_exit": float(p.P.notna().sum() / max(len(c), 1)),
        "deicing_share_of_matched": float(p.deicing.mean()) if len(p) else None,
        "median_takeoff_minus_mvt_s": med(p.dt_takeoff_s), "median_runway_entry_to_takeoff_s": med(p.runway_to_takeoff_s),
        "median_stand_exit_to_runway_entry_s": med(p.stand_exit_to_runway_s), "median_stand_exit_minus_aobt3_s": med(p.stand_exit_minus_aobt3_s),
        "median_stand_exit_minus_block_s": med(p.stand_exit_minus_block_s), "median_airport_taxi_s": med(p.airport_taxi_s)}
OUT.write_text(json.dumps(report, indent=2))
print(f"OPDI selected events: {counts}")
print(f"{'airport':8s}{'deps':>7}{'T+R cov':>8}{'stand cov':>10}{'T-MVT':>7}{'R->T':>6}{'P->R':>6}{'P-AOBT':>8}{'P-BLOCK':>8}{'deice':>7}")
for a, s in report["airports"].items():
    f = lambda v: "-" if v is None else f"{v:.0f}"
    print(f"{a:8s}{s['competition_departures']:>7,}{s['coverage_takeoff_runway']:>8.1%}{s['coverage_stand_exit']:>10.1%}{f(s['median_takeoff_minus_mvt_s']):>7}"
          f"{f(s['median_runway_entry_to_takeoff_s']):>6}{f(s['median_stand_exit_to_runway_entry_s']):>6}{f(s['median_stand_exit_minus_aobt3_s']):>8}"
          f"{f(s['median_stand_exit_minus_block_s']):>8}{('-' if s['deicing_share_of_matched'] is None else format(s['deicing_share_of_matched'], '.1%')):>7}")

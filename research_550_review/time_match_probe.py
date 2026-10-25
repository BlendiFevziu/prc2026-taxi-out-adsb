"""Label-free precision of time-only matching for rows the extraction leaves unmatched (review item, exploratory).

For every main-lane departure the callsign matcher DID match (plus_callsign_flt, as in the v56 extraction), pretend it is
unmatched: remove its own visit from nothing, remove every visit claimed by OTHER matched departures, and take the
unique remaining visit whose last point lies within W seconds of the recorded take-off. Precision = share where that
visit is the true one. Reach = how many truly unmatched departures get a unique candidate the same way.
Run from the repository root:  python3 -u -B research_550_review/time_match_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EH = ROOT.parent / "skenderbeu-eham-readonly"
sys.path.insert(0, str(EH))
import main_model  # noqa: F401,E402
from research_300 import arm_test as AT  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402

DAYS = ("2025-07-01", "2025-07-19", "2025-01-10", "2025-01-22")
AIRPORTS = ("EDDF", "EGLL", "EHAM", "LEBL", "LSZH")
WINDOWS = (60, 120, 300)
OUT = ROOT / "research_550_review/results/time_match_probe.json"


def main():
    mapping = AT.OB.callsign_mapping()
    raw_cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "AOBT_3_flt", "FLIGHT_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "STAND_mvt"]
    report = {}
    for ap in AIRPORTS:
        stats = {w: {"prec_n": 0, "prec_ok": 0, "unmatched": 0, "reach": 0} for w in WINDOWS}
        for date in DAYS:
            raw = pd.read_parquet(Path(MD.month_file(date)), columns=raw_cols)
            dep = raw[raw.PHASE_mvt.eq("DEP") & raw.AOBT_3_flt.notna() & raw.ADEP_mvt.eq(ap)].copy()
            dep["date"] = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
            dep = dep[dep.date.eq(date)].copy()
            dep["airport"] = ap
            dep["callsign"] = dep.FLIGHT_mvt.astype(str).str.strip().str.upper()
            dep["mvt_epoch"] = (pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True) - MD.EPOCH0).dt.total_seconds()
            tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet", filters=[("airport", "=", ap)]).drop_duplicates(["hex", "ts"])
            visits = AT.OB.visits(tracks)
            if not len(visits):
                continue
            visits["key"] = visits.hex + "|" + visits.t_first.astype(str)
            m = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "plus_callsign_flt", mapping)), on=["MVT_ID_mvt", "airport"], how="inner",
                          suffixes=("", "_v"))
            m["key"] = m.hex + "|" + m.t_first.astype(str)
            m = m[m.groupby("key").MVT_ID_mvt.transform("size").eq(1)]
            claimed = dict(zip(m.MVT_ID_mvt, m.key))
            un = dep[~dep.MVT_ID_mvt.isin(claimed)]
            vt, vk = visits.t_last.to_numpy(float), visits.key.to_numpy()
            for w in WINDOWS:
                s = stats[w]
                for mid, t in zip(dep.MVT_ID_mvt, dep.mvt_epoch):
                    others = set(claimed.values()) - {claimed.get(mid)}
                    cand = [k for k, tl in zip(vk, vt) if abs(tl - t) <= w and k not in others]
                    if mid in claimed:
                        if len(cand) == 1:
                            s["prec_n"] += 1
                            s["prec_ok"] += int(cand[0] == claimed[mid])
                    else:
                        s["unmatched"] += 1
                        s["reach"] += int(len(cand) == 1)
        report[ap] = {str(w): {"precision": v["prec_ok"] / v["prec_n"] if v["prec_n"] else None, "precision_n": v["prec_n"],
                               "unmatched": v["unmatched"], "unique_candidate": v["reach"]} for w, v in stats.items()}
        print(ap, json.dumps(report[ap]), flush=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

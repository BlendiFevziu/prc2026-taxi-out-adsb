"""POST-HOC, after the geometry verdict: are EHAM's very large errors wrong-aircraft matches?

On the geometry dates (verdict already fixed), for every EHAM and EDDF departure the X-Plane-first/OSM arm covered,
compare the ADS-B callsign it was matched to with its filed callsign CALLSIGN_flt, and split the errors against the
recorded block by that. Description only; it motivates the matching test's design, which runs on other dates.
Run from the repository root:  python3 -u -B research_300/posthoc_eham_tail.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "adsb-lol")
import offblock_from_tracks as OB  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402
from research_300 import arm_test as AT  # noqa: E402

OUT = ROOT / "research_300/results/posthoc_eham_tail.json"


def main():
    spec, mapping, report = AT.TESTS["geometry"], OB.callsign_mapping(), {}
    filed = pd.concat(pd.read_parquet(p, columns=["MVT_ID_mvt", "CALLSIGN_flt"])
                      for p in sorted((ROOT / "data").glob("training_2025-0[17]-01_*.parquet")))
    rows = pd.read_parquet(ROOT / ".cache/research_300/arm_test_geometry.parquet")
    for ap, arm in (("EHAM", "xplane_first|flight_mvt"), ("EDDF", "osm|flight_mvt")):
        got = []
        for date in spec["dates"]:
            tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
            t = tracks[tracks.airport.astype(str).eq(ap)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
            dep = MD.departures(date, ap).merge(filed, on="MVT_ID_mvt", how="left")
            m = AT.match_candidates(dep, OB.visits(t), AT.options_for(dep, "flight_mvt", mapping))
            got.append(dep.merge(m[["MVT_ID_mvt", "callsign_adsb"]], on="MVT_ID_mvt"))
        g = pd.concat(got).merge(rows[["MVT_ID_mvt", arm]], on="MVT_ID_mvt")
        g = g[g[arm].notna()].copy()
        g["err"] = g[arm] - (g.mvt_epoch - g.block_epoch)      # the column is the detector's taxi time; minus the true one
        g["same_as_filed"] = g.callsign_adsb.eq(g.CALLSIGN_flt.astype(str).str.strip().str.upper())
        def part(k):
            e = g.loc[k, "err"].to_numpy(float)
            return {"n": int(len(e)), "mae_s": float(np.abs(e).mean()) if len(e) else None,
                    "rmse_s": float(np.sqrt((e ** 2).mean())) if len(e) else None,
                    "beyond_600_s": int((np.abs(e) > 600).sum())}
        big = g[g.err.abs() > 600]
        report[ap] = {"arm": arm, "covered": int(len(g)),
                      "matched_callsign_equals_filed": part(g.same_as_filed), "differs_from_filed": part(~g.same_as_filed),
                      "errors_beyond_600_s": int(len(big)), "of_which_callsign_differs_from_filed": int((~big.same_as_filed).sum()),
                      "share_of_squared_error_from_differs": float((g.err[~g.same_as_filed] ** 2).sum() / (g.err ** 2).sum())}
        print(ap, json.dumps(report[ap]), flush=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Does wind explain the operating direction, and do direction and weather go with taxi time?

Label-free part: the operating direction per airport and hour from the 2025 arrival runways
(features_weather.operating_direction) against the METAR wind at half past that hour. A landing
direction faces the wind when the headwind component along it is positive. Among hours with at
least 5 kt of wind, the share facing the wind is compared with the same share after shuffling
the wind reports among that airport's hours (no wind effect), five seeds.

Label-based part (2025 training labels, no 2026 label): median taxi-out time, the physical
proxy MVT_TIME - AOBT_3 and the reporting gap AOBT_3 - BLOCK_TIME, for the two most used
directions per airport and for weather states. Descriptive only; nothing is fitted or selected.

Run from the repository root after research_240/fetch_metar.py:
  python3 research_240/wind_vs_direction.py
"""
import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_240 import features_weather as W

COLUMNS = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt",
           "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"]
SEEDS = (1, 2, 3, 4, 5)


def seconds(a, b):
    return (W.utc_ns(a) - W.utc_ns(b)).dt.total_seconds().to_numpy()


def median(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None


def facing_share(airports, wind_dir, wind_kt, direction, minimum=5.0):
    headwind = wind_kt * np.cos(np.radians(wind_dir - direction * 10))
    valid = np.isfinite(headwind) & (wind_kt >= minimum)
    out = {}
    for airport in sorted(set(airports)) + ["ALL"]:
        m = valid if airport == "ALL" else valid & (airports == airport)
        out[airport] = {"hours": int(m.sum()),
                        "facing_wind_share": float((headwind[m] > 0).mean()) if m.any() else None,
                        "tailwind_over_5kt_share": float((headwind[m] < -5).mean()) if m.any() else None}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", type=Path, default=Path(".cache/research_240/weather"))
    parser.add_argument("--output", type=Path, default=Path("research_240/results/wind_vs_direction.json"))
    args = parser.parse_args()
    paths = sorted(glob.glob("data/training_2025-*.parquet"))
    assert len(paths) == 12, paths
    raw = pd.concat([pd.read_parquet(p, columns=COLUMNS) for p in paths], ignore_index=True)
    phase = raw.PHASE_mvt.astype(str).str.upper().str[:3].to_numpy()
    arrivals = raw[phase == "ARR"].reset_index(drop=True)
    departures = raw[phase == "DEP"].reset_index(drop=True)
    del raw
    reports = W.report_features(W.load_metar(args.weather))
    directions = W.operating_direction(arrivals)
    wx = W.weather_for(directions.airport, directions.hour + pd.Timedelta(minutes=30), reports)
    airports = directions.airport.to_numpy()
    wind_dir, wind_kt = wx.wind_dir_deg.to_numpy(float), wx.wx_wind_kt.to_numpy(float)
    heading = directions.dir.to_numpy(float)

    report = {"note": "2025 training data; direction from arrival runways per airport-hour (at least five arrivals); "
                      "METAR wind at half past the hour; descriptive, nothing fitted",
              "hours_with_direction": int(len(directions)),
              "share_of_direction_hours_with_wind_report": float(np.isfinite(wind_kt).mean()),
              "facing_wind_5kt": facing_share(airports, wind_dir, wind_kt, heading),
              "facing_wind_10kt": facing_share(airports, wind_dir, wind_kt, heading, 10.0)}
    shuffled = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        d, k = wind_dir.copy(), wind_kt.copy()
        for airport in np.unique(airports):
            idx = np.flatnonzero(airports == airport)
            perm = rng.permutation(idx)
            d[idx], k[idx] = wind_dir[perm], wind_kt[perm]
        shuffled.append(facing_share(airports, d, k, heading))
    report["facing_wind_5kt_shuffled_mean"] = {a: float(np.mean([s[a]["facing_wind_share"] for s in shuffled]))
                                               for a in report["facing_wind_5kt"]}
    report["top_directions"] = {a: {str(int(k)): round(float(v), 3) for k, v in g.dir.value_counts(normalize=True).head(3).items()}
                                for a, g in directions.groupby("airport")}

    dep_air = departures.ADEP_mvt.astype(str).to_numpy()
    od = W.direction_for(dep_air, departures.MVT_TIME_UTC_mvt, directions)
    wxd = W.weather_for(dep_air, departures.MVT_TIME_UTC_mvt, reports)
    y = pd.to_numeric(departures.TAXITIME_SEC_mvt, errors="coerce").to_numpy(float)
    phys = seconds(departures.MVT_TIME_UTC_mvt, departures.AOBT_3_flt)
    gap = seconds(departures.AOBT_3_flt, departures.BLOCK_TIME_UTC_mvt)
    od_dir = od.od_dir.to_numpy(float)
    icing = wxd.wx_icing_risk.to_numpy(float)
    frozen = np.maximum(np.nan_to_num(wxd.wx_frozen.to_numpy(float)), np.nan_to_num(wxd.wx_freezing.to_numpy(float)))
    report["departures_with_weather_share"] = float(np.isfinite(wxd.wx_temp_c.to_numpy(float)).mean())
    report["departures_with_direction_share"] = float(np.isfinite(od_dir).mean())

    def cell(mask):
        return {"departures": int(mask.sum()), "median_taxi_s": median(y[mask]),
                "median_mvt_minus_aobt3_s": median(phys[mask]), "median_aobt3_minus_block_s": median(gap[mask])}

    by_direction, by_weather = {}, {}
    for airport in sorted(set(dep_air)):
        m = (dep_air == airport) & np.isfinite(y)
        counts = pd.Series(od_dir[m]).value_counts()
        if len(counts) >= 2:
            a, b = counts.index[0], counts.index[1]
            entry = {"A": {"direction": int(a), **cell(m & (od_dir == a))}, "B": {"direction": int(b), **cell(m & (od_dir == b))}}
            entry["taxi_median_A_minus_B_s"] = entry["A"]["median_taxi_s"] - entry["B"]["median_taxi_s"]
            entry["physical_median_A_minus_B_s"] = entry["A"]["median_mvt_minus_aobt3_s"] - entry["B"]["median_mvt_minus_aobt3_s"]
            by_direction[airport] = entry
        by_weather[airport] = {"icing_risk": cell(m & (icing == 1)), "frozen_or_freezing_precipitation": cell(m & (frozen == 1)),
                               "neither": cell(m & (icing == 0) & (frozen == 0))}
    report["taxi_by_direction"] = by_direction
    report["taxi_by_weather"] = by_weather
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"direction hours {report['hours_with_direction']:,}, with wind report {report['share_of_direction_hours_with_wind_report']:.1%}")
    print(f"{'airport':8s}{'hours>=5kt':>11}{'facing wind':>13}{'shuffled':>10}{'tailwind>5kt':>14}  top directions")
    for a, v in report["facing_wind_5kt"].items():
        print(f"{a:8s}{v['hours']:>11,}{v['facing_wind_share']:>13.1%}{report['facing_wind_5kt_shuffled_mean'][a]:>10.1%}"
              f"{v['tailwind_over_5kt_share']:>14.1%}  {report['top_directions'].get(a, '')}")
    print(f"\n{'airport':8s}{'dir A':>6}{'dir B':>6}{'taxi A-B':>10}{'phys A-B':>10}{'n A':>9}{'n B':>9}")
    for a, e in by_direction.items():
        print(f"{a:8s}{e['A']['direction']:>6}{e['B']['direction']:>6}{e['taxi_median_A_minus_B_s']:>9.0f}s{e['physical_median_A_minus_B_s']:>9.0f}s"
              f"{e['A']['departures']:>9,}{e['B']['departures']:>9,}")
    print(f"\n{'airport':8s}{'icing: n, taxi, phys':>28}{'frozen/freezing: n, taxi, phys':>36}{'neither: taxi, phys':>24}")
    for a, e in by_weather.items():
        i, f, n = e["icing_risk"], e["frozen_or_freezing_precipitation"], e["neither"]
        fmt = lambda c: "-" if c["median_taxi_s"] is None else f"{c['median_taxi_s']:.0f}/{c['median_mvt_minus_aobt3_s'] or float('nan'):.0f}"
        print(f"{a:8s}{i['departures']:>12,} {fmt(i):>15}{f['departures']:>18,} {fmt(f):>17}{fmt(n):>24}")


if __name__ == "__main__":
    main()

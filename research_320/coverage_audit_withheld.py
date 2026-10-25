"""LABEL-FREE coverage audit: does any already-used 2025 date carry the 2026 July observation regime at EGLL or EDDF?

v56 withholds Heathrow and Frankfurt July because their 2026 tracks differ from its four July training dates
(Codex, research_310/WITHHELD_AUDIT_RESULT.md on codex/withheld-track-recovery): Heathrow 2026 is dense and starts at
the stand (median 81 ground points, first point 44 m from it) where July 2025 had almost no ground points; Frankfurt
2026 is sparse and starts late (9 points, 1,125 m) where July 2025 had 31 points at 433 m. Broader 2025 training can
only help if the 2026 regime exists somewhere in 2025. This counts it, per date, from the feature rows already
extracted with Codex's frozen code for 22 used dates (research_320 caches: v56's training, development-test and
replication dates). No label is read, no model is fitted; the eight reserve dates are not touched.

Also: how many 2026 July scored main-lane rows at each airport have a stand and runway that appear among 2025 partial
rows, on v56's four July training dates against all used 2025 dates.
Run from the repository root:  python3 -u -B research_320/coverage_audit_withheld.py
"""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / ".cache/research_320/features"
OUT = HERE / "research_320/results/coverage_audit_withheld.json"
AIRPORTS = ("EGLL", "EDDF")
V56_JULY_TRAIN = ("2025-07-02", "2025-07-04", "2025-07-18", "2025-07-23")


def codex_2026():
    raw = subprocess.run(["git", "show", "codex/withheld-track-recovery:research_310/results/withheld_audit.json"],
                         cwd=HERE, capture_output=True, text=True, check=True).stdout
    cells = json.loads(raw)["cells"]
    out = {}
    for ap in AIRPORTS:
        c = cells[f"{ap}/07"]
        main = sum(c["current_categories"].values())
        f = c["features"]
        out[ap] = {"main_rows": main,
                   "category_share": {k: round(v / main, 3) for k, v in c["current_categories"].items()},
                   "partial_median": {k: f[k]["current"]["p10_p50_p90_p99"][1]
                                      for k in ("ground_points", "stand_first_m", "ground_span_s", "first_ground_age_s")}}
    return out


def main():
    frames = [pd.read_parquet(CACHE / f"{tag}.parquet") for tag in ("train", "dev_test", "screen")]
    d = pd.concat(frames, ignore_index=True).drop_duplicates("MVT_ID_mvt")
    d = d[d.airport.isin(AIRPORTS)].copy()
    meta = pd.concat(pd.read_parquet(HERE / f"data/training_2025-{m}-01_2025-{n}-01.parquet",
                                     columns=["MVT_ID_mvt", "STAND_mvt", "RUNWAY_mvt"]) for m, n in (("01", "02"), ("07", "08")))
    d = d.merge(meta, on="MVT_ID_mvt", how="left", validate="one_to_one")
    d["month"] = d.date.str[5:7]
    report = {"dates_2025": sorted(d.date.unique()), "codex_2026_july": codex_2026(), "per_date": {}, "composition": {}}
    for (ap, date), g in d.groupby(["airport", "date"]):
        p = g[g.category.eq("partial_ground")]
        report["per_date"][f"{ap} {date}"] = {
            "main_rows": int(len(g)),
            "no_pre_takeoff_share": round(float(g.category.eq("no_pre_takeoff_points").mean()), 3),
            "at_stand_share": round(float(g.category.eq("observed_at_assigned_stand").mean()), 3),
            "partial_share": round(float(len(p) / len(g)), 3),
            "partial_median": {k: (float(p[k].median()) if len(p) else None)
                               for k in ("ground_points", "stand_first_m", "ground_span_s", "first_ground_age_s")}}
    rank = pd.read_parquet(HERE / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "ADEP_mvt", "AOBT_3_flt",
                                                                    "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
    rank = rank[rank.PHASE_mvt.eq("DEP") & rank.TAXITIME_SEC_mvt.isna() & rank.AOBT_3_flt.notna()]
    rank = rank[pd.to_datetime(rank.MVT_TIME_UTC_mvt, utc=True).dt.month.eq(7)]
    for ap in AIRPORTS:
        target = rank[rank.ADEP_mvt.astype(str).eq(ap)]
        partial = d[d.airport.eq(ap) & d.category.eq("partial_ground")]
        rec = {"scored_july_main_rows": int(len(target))}
        for name, src in (("v56_july_train", partial[partial.date.isin(V56_JULY_TRAIN)]),
                          ("all_used_july", partial[partial.month.eq("07")]),
                          ("all_used_jan_and_july", partial)):
            rec[name] = {col: round(float(target[col].astype(str).isin(set(src[col].astype(str))).mean()), 3)
                         for col in ("STAND_mvt", "RUNWAY_mvt")}
            rec[name]["partial_rows"] = int(len(src))
        report["composition"][ap] = rec
    rules = {"EGLL": (lambda x: (x.ground_points >= 50) & (x.stand_first_m <= 100), ">=50 points, first point <=100 m"),
             "EDDF": (lambda x: (x.ground_points <= 12) & (x.stand_first_m >= 900), "<=12 points, first point >=900 m")}
    report["row_level_2026_like"] = {}
    for ap, (rule, text) in rules.items():
        for m in ("01", "07"):
            g = d[d.airport.eq(ap) & d.month.eq(m) & d.category.eq("partial_ground")]
            report["row_level_2026_like"][f"{ap}-{m}"] = {"definition": text, "partial_rows": int(len(g)),
                                                          "rows_2026_like": int(rule(g).sum()), "share": round(float(rule(g).mean()), 3)}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("2026 July (Codex):", json.dumps(report["codex_2026_july"]))
    for key, r in report["per_date"].items():
        m = r["partial_median"]
        print(f"{key}: main {r['main_rows']:4d} | no-ground {r['no_pre_takeoff_share']:.2f} at-stand {r['at_stand_share']:.2f} "
              f"partial {r['partial_share']:.2f} | partial median points {m['ground_points']}, first-point stand m "
              f"{None if m['stand_first_m'] is None else round(m['stand_first_m'])}, span s "
              f"{None if m['ground_span_s'] is None else round(m['ground_span_s'])}")
    print("stand/runway coverage of 2026 July scored rows:", json.dumps(report["composition"]))


if __name__ == "__main__":
    main()

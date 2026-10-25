"""Label-free drift audit of every deployed layer cell (v77): the v56 drift rule (partial_coverage_2026.drift) between
the cell's 2025 rows (all locally extracted days) and its 2026 rows, for partial rows (C1, C1+) and stand rows (K2+).
No label is read. Run from the repository root:  python3 -u -B research_600_drift_audit/audit.py
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research_360_integration", ROOT / "research_400_cells", ROOT / "research_590_stand_new"):
    sys.path.insert(0, str(p))
import stand_new as T  # noqa: E402
import extract as E  # noqa: E402
import cells_confirm as CC  # noqa: E402

S, V = T.S, T.V
TAGS = (("train", tuple(S.TRAIN)), ("volume_extra", V.E_EXTRA), ("development", tuple(S.STAGES["development"])),
        ("confirm", tuple(S.STAGES["confirm"])), ("fresh_july", ("2025-07-01", "2025-07-05", "2025-07-19", "2025-07-20", "2025-07-24", "2025-07-28", "2025-07-30")),
        ("eddf_july_confirm", ("2025-07-06", "2025-07-08", "2025-07-10")),
        ("fresh_january", ("2025-01-03", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-15", "2025-01-17", "2025-01-22", "2025-01-27", "2025-01-29", "2025-01-31")))
OUT = ROOT / "research_600_drift_audit/results/audit.json"


def main():
    ref = pd.concat([S.extract(d, t) for t, d in TAGS], ignore_index=True).drop_duplicates("MVT_ID_mvt")
    ref["cell"] = ref.airport + "/" + ref.date.str[5:7]
    ctx = pd.read_parquet(ROOT / ".cache/codex_v56/partial_release/context.parquet")
    ctx["cell"] = ctx.airport + "/" + ctx.date.str[5:7]
    layers = {"C1+ (partial)": ("partial_ground", ("EDDF/01", "EDDF/07", "EGLL/01", "EHAM/07", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07")),
              "EHAM Jan average (partial)": ("partial_ground", ("EHAM/01",)),
              "K2+ (stand)": ("observed_at_assigned_stand", ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07", "LSZH/01", "LSZH/07", "EHAM/01", "EHAM/07"))}
    report = {}
    for name, (cat, cells) in layers.items():
        for c in cells:
            r = ref[ref.cell.eq(c) & ref.category.eq(cat)]
            p = ctx[ctx.cell.eq(c) & ctx.category.eq(cat)]
            st = V.P.drift(r, p)
            report[f"{name} {c}"] = {"rows_2025": int(len(r)), "rows_2026": int(len(p)), "flags": [k for k, v in st.items() if v["diagnostic_flag"]]}
    # the old C1 at Munich (both months, reference Munich January 2025) and Madrid July
    for ap, days, cells in (("EDDM", CC.JAN, ("EDDM/01", "EDDM/07")), ("LEMD", CC.JUL, ("LEMD/07",))):
        r = E.load(ap, days)
        r = r[r.category.eq("partial_ground")]
        r26 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / f".cache/research_590/days/{ap}_2026-*.parquet")))], ignore_index=True)
        for c in cells:
            p = r26[r26.date.str[5:7].eq(c[-2:]) & r26.category.eq("partial_ground")]
            st = V.P.drift(r, p)
            report[f"C1 (partial) {c}"] = {"rows_2025": int(len(r)), "rows_2026": int(len(p)), "flags": [k for k, v in st.items() if v["diagnostic_flag"]]}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for k, v in report.items():
        print(f"{k:34s} 2025 {v['rows_2025']:6d}  2026 {v['rows_2026']:6d}  flags {v['flags']}")


if __name__ == "__main__":
    main()

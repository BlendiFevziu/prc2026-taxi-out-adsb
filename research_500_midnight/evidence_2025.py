"""The 2025 tables behind research_500_midnight/PREREG_midnight.md, recomputed from the training files. No fit.
Run from the repository root:  python3 -u -B research_500_midnight/evidence_2025.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule as R  # noqa: E402

OUT = R.ROOT / "research_500_midnight/results/evidence_2025.json"


def main():
    d = pd.concat([pd.read_parquet(p, columns=R.COLS + [R.T, "BLOCK_TIME_UTC_mvt"])
                   for p in sorted((R.ROOT / "data").glob("training_2025-*.parquet"))], ignore_index=True)
    d = R.annotate(d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.eq("LIRF")])
    d["kind"] = R.label_kind(d[R.T], d.delay)
    d["lane"] = d.AOBT_3_flt.isna().map({True: "fallback", False: "main"})
    out = {"shift24_without_crossing": {lane: int(((d.lane == lane) & (d.crossing == 0) & (d.kind == "shift24")).sum())
                                        for lane in ("fallback", "main")},
           "noncopy_without_crossing": {lane: int(((d.lane == lane) & (d.crossing == 0) & (d.kind != "copy")).sum())
                                        for lane in ("fallback", "main")}}
    fb = d[(d.lane == "fallback") & (d.crossing == 0)]
    for lo in (4, 6, 8):
        z = fb[fb.delay > lo * 3600]
        out[f"fallback_no_crossing_delay_above_{lo}h"] = {k: int(v) for k, v in z.kind.value_counts().items()}
    out["rule_rows_2025"] = int(R.selected(d).sum())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

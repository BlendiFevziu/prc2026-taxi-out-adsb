"""EXPLORATORY, Codex's own screen and confirmation dates (spent): is his EHAM January add-on under-weighted at 0.5?

His 21 frozen models (hash-checked copies of his cache, .cache/codex_eham/eham_recovery/) on his cached prediction
tables for January 4, 28 (screen) and 14, 21, 26 (confirmation): eligible rows = partial_ground and not protected by
the guarded Amsterdam detector, exactly his definition. Gain against the comparator at weights s; no fitting.
Run from the repository root:  python3 -u -B research_320/eham_weight_headroom.py
"""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
C = HERE / ".cache/codex_eham/eham_recovery"
SCHEMA = ["comparator", "ersatz_s", "schedule", "gate", "hour_sin", "hour_cos", "first_age_s", "last_age_s", "span_s",
          "endpoint_distance_m", "weighted_speed_kt", "weighted_still_share", "observed_interval_share", "is_july"]
GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def main():
    sys.path.insert(0, str(HERE))
    import main_model  # noqa: F401  (torch before LightGBM on macOS)
    out = {}
    for stage in ("screen", "confirmation"):
        p = pd.read_parquet(C / f"{stage}_predictions.parquet")
        hit = (p.category.eq("partial_ground") & ~p.protected.astype(bool)).to_numpy()
        e = p[hit].reset_index(drop=True)
        corr = np.mean([joblib.load(C / f"real_{s}.pkl").predict(e[SCHEMA]) for s in (177, 277, 377)], axis=0)
        y, base = e.y.to_numpy(float), e.comparator.to_numpy(float)
        assert np.allclose(np.maximum(base + .5 * corr, 300.), e.real.to_numpy(float)), "replay of his 0.5 predictions failed"
        curve = {str(s): float(np.mean((base - y) ** 2 - (np.maximum(base + s * corr, 300.) - y) ** 2)) for s in GRID}
        out[stage] = {"eligible_rows": int(hit.sum()), "curve": curve, "best_s": max(curve, key=curve.get)}
        print(stage, out[stage]["eligible_rows"], "rows | " + " ".join(f"{s}:{v:.0f}" for s, v in curve.items()), "| best", out[stage]["best_s"])
    (HERE / "research_320/results/eham_weight_headroom.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

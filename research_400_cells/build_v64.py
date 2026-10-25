"""Build v64 = the exact scored v63 with the frozen C1 model switched on in the two cells that passed
research_400_cells/PREREG_cells.md: EDDM January and LEMD July. NM-free. No fit, no upload.

The 2026 context of these rows (comparator, ersatz_s, schedule, gate, hour phase) is rebuilt with Codex's recipe
(research_310/build_partial_candidate.context at 9884150: the pre-NM base, the release's ranking table, its reporting
classifier to the power 2.5), and the rebuild must first reproduce 3,000 rows of his own context table bit for bit.
Track summaries come from the 2026 coverage audit's cache (the same frozen extract_day, OSM stands, as the 2025 test).
The rows must be base-owned in v63 and sit exactly at the comparator; the new value is max(comparator + C1, 300).
Run from the repository root:  python3 -u -B research_400_cells/build_v64.py
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import extract_episodes as X  # noqa: E402

MS = X.v56_modules()
R = MS.R
import submission as SUB  # noqa: E402
import harness as H  # noqa: E402
import layers as L  # noqa: E402

ID, T, N, SEEDS = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, (177, 277, 377)
CELLS = ("EDDM/01", "LEMD/07")
FLAGS = ("EDDF", "EGLL", "EHAM", "LEBL", "LSZH")
RUN = HERE / ".cache/clock_queue_release_20260912"
V63 = HERE / ".cache/research_360/v63/candidate_v63.parquet"
AUDIT = HERE / ".cache/research_320/airport_coverage"
OUT = HERE / ".cache/research_400/v64"
REPORT = HERE / "research_400_cells/results/v64_build.json"
CONTEXT = ("comparator", "ersatz_s", "schedule", "gate", "hour_sin", "hour_cos")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def context_2026(features):
    """Codex's context recipe for the given 2026 feature rows (airport, date, track summaries)."""
    rank = pd.read_parquet(RUN / "data/ranking.parquet").set_index(ID, verify_integrity=True)
    classifier = joblib.load(RUN / "final/checkpoints/reporting_classifier.pkl")["model"]
    cols = list(classifier.feature_name_)
    ids = features[ID].to_numpy()
    out = features.copy()
    out["comparator"] = H.current_base_2026().loc[ids].to_numpy(float)
    out["ersatz_s"] = rank.loc[ids, "ersatz_s"].to_numpy(float)
    out["schedule"] = rank.loc[ids, "versp_min"].to_numpy(float) * 60
    out["gate"] = classifier.predict_proba(rank.loc[ids, cols])[:, 1] ** 2.5
    raw = pd.read_parquet(HERE / "data/ranking.parquet", columns=[ID, "MVT_TIME_UTC_mvt"]).set_index(ID)
    epoch = (pd.to_datetime(raw.loc[ids, "MVT_TIME_UTC_mvt"], utc=True) - R.MD.EPOCH0).dt.total_seconds().to_numpy()
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * (epoch % 86400) / 86400), np.cos(2 * np.pi * (epoch % 86400) / 86400)
    for ap in FLAGS:
        out[f"airport_{ap}"] = out.airport.eq(ap).astype(float)
    out["is_july"] = out.date.str[5:7].eq("07").astype(float)
    if not np.isfinite(out[list(CONTEXT)].to_numpy()).all():
        raise SystemExit("nonfinite 2026 context")
    return out


def prediction(cells, used):
    """Fixed before upload: each cell's registered planning gain; central at v63's realised ratio to its registered
    value (0.72), worst at 0.4, best at full transfer."""
    registered = {"EDDM/01": 6802, "LEMD/07": 8692}
    total = sum(cells[c]["planning"]["mse_per_scored_row"] * used[c[:4]] / registered[c] for c in CELLS)
    ref = 258.0008
    at = lambda k: round(ref - (ref - np.sqrt(ref ** 2 - k * total)), 2)
    return {"assumptions": "sum of the two cells' registered planning values (group-level reweighting), scaled to the base-owned rows; central at v63's realised "
                           "0.72 of its registered value, worst 0.4, best 1.0", "mse_per_scored_row_full": total,
            "central": at(0.72), "worst": at(0.4), "best": at(1.0)}


def main():
    rec63 = json.loads((HERE / "results/upbeat-goblin_v63.json").read_text())
    cap = json.loads((HERE / "research_320/results/capacity2.json").read_text())
    conf = json.loads((HERE / "research_400_cells/results/cells_confirm.json").read_text())
    checks = {"v63_is_scored_file": sha(V63) == rec63["file_sha256"],
              "cells_passed": all(conf["cells"][c]["verdict"] == "pass" for c in CELLS),
              "c1_models_as_tested": all(sha(HERE / f".cache/research_320/volume/models/c1_real_{s}.pkl") == cap["model_hashes"][f"c1_real_{s}.pkl"] for s in SEEDS)}
    if not all(checks.values()):
        raise SystemExit(f"a precondition failed: {checks}")
    # 1. the recipe must reproduce Codex's own context rows exactly
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet")
    probe = ctx.sample(3000, random_state=64).reset_index(drop=True)
    rebuilt = context_2026(probe[[c for c in probe.columns if c not in CONTEXT and not c.startswith("airport_") and c != "is_july"]])
    checks["context_reproduced"] = {c: bool(np.array_equal(rebuilt[c].to_numpy(float), probe[c].to_numpy(float))) for c in CONTEXT}
    if not all(checks["context_reproduced"].values()):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2) + "\n")
        raise SystemExit(f"context recipe does not reproduce Codex's rows: {checks['context_reproduced']}")
    # 2. the rows of the two cells
    feats = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(AUDIT / "day_2026-*.parquet")))], ignore_index=True)
    feats["cell"] = feats.airport + "/" + feats.date.str[5:7]
    rows = feats[feats.cell.isin(CELLS) & feats.category.eq("partial_ground")].drop(columns="cell").reset_index(drop=True)
    own = H.ownership_v62().set_index(ID).owner            # v63 changed only EDDF July, so v62 ownership holds
    owner = own.loc[rows[ID]].to_numpy()
    checks["partial_rows_by_cell_and_owner"] = pd.crosstab(rows.airport.to_numpy(), owner).to_dict()
    rows = rows[owner == "base"].reset_index(drop=True)    # the registration: base-owned rows only; detector rows stay
    checks["rows_by_cell"] = rows.airport.value_counts().to_dict()
    checks["all_base_owned"] = bool(own.loc[rows[ID]].eq("base").all())
    frame = context_2026(rows)
    v63 = pd.read_parquet(V63)
    at63 = v63.set_index(ID).loc[frame[ID], T].to_numpy()
    checks["v63_rows_at_comparator"] = bool(np.array_equal(frame.comparator.to_numpy(float).astype(at63.dtype), at63))
    corr = L.correction("partial", frame)
    rev = L.correction("partial", frame.iloc[::-1].reset_index(drop=True))[::-1]
    checks["c1_reverse_replay"] = bool(np.array_equal(corr, rev))
    if not (checks["all_base_owned"] and checks["v63_rows_at_comparator"] and checks["c1_reverse_replay"]):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"a replay gate failed: {checks}")
    new = pd.Series(L.apply(frame, corr, 1.0), index=frame[ID].to_numpy())
    cand = v63.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v63.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v63.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v63=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int((cand[T].to_numpy() != v63[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    ok = not errors and untouched and checks["all_finite"]
    report = {"checks": checks, "base": {"version": "v63", "score": rec63["score"], "sha256": rec63["file_sha256"]},
              "prediction_fixed_before_upload": prediction(conf["cells"], checks["rows_by_cell"])}
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v64.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

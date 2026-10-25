"""Build v55 on top of v52 (the base of v54, with v52's NM add-on): the two validated open-data detectors. No upload.

Recipe. v52 everywhere, except scored MAIN-lane departures where a validated detector configuration estimates:
  EDDM  X-Plane-first stands, FLIGHT_mvt spellings, detect_onset_v2 plus v54's arrival-and-stay guard
        (research_300/PREREG_geometry.md, addendum B: passed against v54's OSM configuration). This replaces v54's
        Munich rows: where only the OSM configuration estimated, the row returns to v52, exactly the validated policy.
  EHAM  the same, plus the tow and false-onset guard (research_300/PREREG_towguard.md, addendum A: passed against the
        held-out v45 ensemble on eight fresh dates)
There the prediction becomes the detector's taxi time, MVT_TIME minus the onset midpoint, unfloored, as validated.
Fallback rows and every other airport are never touched.

Owner's decision, 26 September 2026: "yes build the combined candidate with the v54 add on and upload it". The
candidate keeps v52's NM add-on on its covered rows, as v54 does.

Gates, all required:
  G0  this pipeline reproduces v54: with v54's Munich configuration (OSM stands) it rebuilds all of v54's replaced
      values exactly, and nothing else differs from the uploaded v54 file
  G1  the base is v52 and the v54 file is the uploaded one (SHA-256 against their records)
  G2  the validated code is unchanged: stand_detector.py since ffccea1, geometry since 8ed8741, tow_guard.py since
      fa2e2ab (committed and in the working tree), and the four arm_test functions used here equal their 8ed8741 text
  G3  both validations passed (arm_test_geometry.json EDDM, tow_guard_test.json comparison 2)
  G4  submission.pruefe clean; every row outside the replaced rows identical to v52; all finite; against v54 only
      EDDM and EHAM scored main-lane rows differ
Row-level output goes to .cache/combined_candidate/ (never committed); the manifest to research_300/results/.
Run from the repository root:  python3 -u -B research_300/build_combined_candidate.py
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "adsb-lol")
import offblock_from_tracks as OB  # noqa: E402
import submission as SUB  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import scored_coverage_2026 as SC  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402
from research_300 import arm_test as AT  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402
from research_300 import tow_guard as TG  # noqa: E402

BASE = ROOT / ".cache/research_280/release/candidate_v52_combined.parquet"
V54 = ROOT / ".cache/munich_candidate/candidate_v52_munich_detector.parquet"
OUT = ROOT / ".cache/combined_candidate"
MANIFEST = ROOT / "research_300/results/combined_candidate_manifest.json"
KEY = "upbeat-goblin_v55.parquet"
CONFIG = {"EDDM": ("xplane_first", False), "EHAM": ("xplane_first", True)}     # (geometry, tow guard)
V54_CONFIG = ("osm", False)
OFFICIAL_V54, N_SCORED = 264.2328, 344841
# per replaced row against the held-out v45 ensemble, from the two tests; v52 is stronger than v45 on any row by about
# 705 (sequence nets) + 715 (NM add-on on its covered days), the deduction used for v54 (PREREG_stand_detector.md)
EHAM_GAIN_VS_V45, V52_OVER_V45 = 17044.857407687694, 705.0 + 715.0
TRANSFER = {"low_0.74": 0.74, "central_1.0": 1.0, "high_1.52_v54_realised": 1.52}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_unchanged(since, *paths):
    committed = subprocess.run(["git", "diff", "--quiet", since, "HEAD", "--", *paths], cwd=ROOT).returncode == 0
    working = subprocess.run(["git", "diff", "--quiet", "--", *paths], cwd=ROOT).returncode == 0
    return committed and working


def functions_as_validated(commit="8ed8741", names=("guarded", "lookup", "match_candidates", "options_for")):
    old = subprocess.run(["git", "show", f"{commit}:research_300/arm_test.py"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    nodes = {n.name: ast.get_source_segment(old, n) for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
    return all(nodes.get(name, "").strip() == inspect.getsource(getattr(AT, name)).strip() for name in names)


def scored_days():
    frame = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
    frame = frame[frame.PHASE_mvt.eq("DEP") & frame.TAXITIME_SEC_mvt.isna()]
    return sorted(pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").unique())


def estimates(tracks, date, airport, geometry, tow_guard, mapping, maps):
    """Detector taxi times for the scored main-lane departures of one airport-day: MVT_ID_mvt -> seconds."""
    t = tracks[tracks.airport.astype(str).eq(airport)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
    dep = SC.scored_departures(date)
    dep = dep[dep.airport.eq(airport)].reset_index(drop=True)
    out, withdrawn = {}, 0
    visits = OB.visits(t) if len(t) else pd.DataFrame()
    if not len(visits):
        return out, withdrawn
    j = dep.merge(AT.match_candidates(dep, visits, AT.options_for(dep, "flight_mvt", mapping)),
                  on=["MVT_ID_mvt", "airport"], how="inner", suffixes=("", "_v"))
    j = j[j.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
    for row in j[j.lane.eq("main")].itertuples(index=False):
        vertices, _ = AT.lookup(geometry, *maps, row.STAND_mvt)
        if vertices is None:
            continue
        ts, lat, lon, gs, ground = MD.points_of(t, row)
        o = AT.guarded(ts, lat, lon, gs, vertices, row.mvt_epoch)
        if tow_guard and o["status"] == "ok":
            o = TG.check(ts, lat, lon, gs, ground, vertices, o, row.mvt_epoch)
            withdrawn += o["status"] != "ok"
        if o["status"] == "ok":
            out[row.MVT_ID_mvt] = row.mvt_epoch - (o["t_before"] + o["t_after"]) / 2
    return out, withdrawn


def apply(base, taxi, dtype):
    cand = base.copy()
    hit = cand.MVT_ID_mvt.isin(taxi.keys()).to_numpy()
    cand.loc[hit, "TAXITIME_SEC_mvt"] = cand.loc[hit, "MVT_ID_mvt"].map(taxi).to_numpy(float)
    cand["TAXITIME_SEC_mvt"] = cand.TAXITIME_SEC_mvt.astype(dtype)
    return cand, hit


def main():
    gates = {}
    v52_record = json.loads((ROOT / "results/upbeat-goblin_v52.json").read_text())
    v54_ready = json.loads((ROOT / ".cache/munich_candidate/ready_for_upload_v54.json").read_text())
    gates["G1_base_is_v52"] = sha(BASE) == v52_record["candidate_sha256"]
    gates["G1_v54_is_uploaded_file"] = sha(V54) == v54_ready["candidate_sha256"]
    gates["G2_detector_unchanged"] = git_unchanged("ffccea1", "research_290/stand_detector.py")
    gates["G2_geometry_unchanged"] = git_unchanged("8ed8741", "research_300/geometry_compare.py", "research_300/geometry")
    gates["G2_tow_guard_unchanged"] = git_unchanged("fa2e2ab", "research_300/tow_guard.py")
    gates["G2_arm_test_functions_as_validated"] = functions_as_validated()
    geo = json.loads((ROOT / "research_300/results/arm_test_geometry.json").read_text())["airports"]["EDDM"]
    tow = json.loads((ROOT / "research_300/results/tow_guard_test.json").read_text())["comparison_2_decision"]
    gates["G3_munich_geometry_passed"] = geo["verdict"] == "pass"
    gates["G3_amsterdam_passed"] = tow["verdict"] == "pass"
    print("gates", gates, flush=True)
    if not all(gates.values()):
        raise SystemExit("a G1 to G3 gate failed")

    days = scored_days()
    missing = [d for d in days if not (MD.ADSB / f"tracks/{d}.parquet").exists()]
    if missing:
        raise SystemExit(f"tracks missing for {missing}")
    mapping = OB.callsign_mapping()
    maps = {ap: (SD.stand_table(ap), GC.xplane_table(ap)[0]) for ap in CONFIG}
    taxi, reproduced, per_day = {}, {}, {}
    for date in days:
        tracks = pd.read_parquet(MD.ADSB / f"tracks/{date}.parquet")
        rec = {}
        for ap, (geometry, tow_guard) in CONFIG.items():
            est, withdrawn = estimates(tracks, date, ap, geometry, tow_guard, mapping, maps[ap])
            taxi.update(est)
            rec[ap] = {"replaced": len(est), "withdrawn_by_tow_guard": withdrawn}
        est, _ = estimates(tracks, date, "EDDM", *V54_CONFIG, mapping, maps["EDDM"])
        reproduced.update(est)
        rec["EDDM_v54_config"] = len(est)
        per_day[date] = rec
        print(date, rec, flush=True)

    base, v54 = pd.read_parquet(BASE), pd.read_parquet(V54)
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    dtype = template.TAXITIME_SEC_mvt.dtype
    rebuilt, hit54 = apply(base, reproduced, dtype)
    gates["G0_reproduces_v54"] = {"rows": int(hit54.sum()), "expected_rows": int(v54_ready["rows_changed_vs_v52"]),
                                  "identical_to_uploaded_v54": bool(rebuilt.equals(v54))}
    print("G0", gates["G0_reproduces_v54"], flush=True)
    if not gates["G0_reproduces_v54"]["identical_to_uploaded_v54"]:
        MANIFEST.write_text(json.dumps({"gates": gates, "per_day": per_day}, indent=2, default=str) + "\n")
        raise SystemExit("G0 failed: this pipeline does not reproduce v54; nothing built")

    cand, hit = apply(base, taxi, dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = bool(cand.loc[~hit, "TAXITIME_SEC_mvt"].equals(base.loc[~hit, "TAXITIME_SEC_mvt"]))
    ranking = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "AOBT_3_flt"])
    info = cand[["MVT_ID_mvt"]].merge(ranking, on="MVT_ID_mvt", how="left")
    differs54 = (cand.TAXITIME_SEC_mvt != v54.TAXITIME_SEC_mvt).to_numpy()
    allowed = (info.ADEP_mvt.astype(str).isin(CONFIG) & info.AOBT_3_flt.notna()).to_numpy()
    gates["G4"] = {"pruefe_errors": [str(e) for e in errors], "untouched_rows_identical_to_v52": untouched,
                   "all_finite": bool(np.isfinite(cand.TAXITIME_SEC_mvt).all()),
                   "differences_from_v54_only_at_EDDM_EHAM_main": bool(not (differs54 & ~allowed).any())}
    print("G4", gates["G4"], flush=True)
    if errors or not untouched or not gates["G4"]["all_finite"] or not gates["G4"]["differences_from_v54_only_at_EDDM_EHAM_main"]:
        MANIFEST.write_text(json.dumps({"gates": gates, "per_day": per_day}, indent=2, default=str) + "\n")
        raise SystemExit("GATE G4 failed; nothing built")

    airport = info.ADEP_mvt.astype(str).to_numpy()
    eham = int((hit & (airport == "EHAM")).sum())
    munich_changed = int((differs54 & (airport == "EDDM")).sum())
    munich = {"xplane_rows": int((hit & (airport == "EDDM")).sum()), "v54_rows": int(hit54.sum()),
              "rows_differing_from_v54": munich_changed,
              "both_estimate": int((hit & hit54 & (airport == "EDDM")).sum()),
              "only_xplane": int((hit & ~hit54 & (airport == "EDDM")).sum()),
              "only_v54_osm_back_to_v52": int((~hit & hit54).sum())}
    mse_gain = (eham * (EHAM_GAIN_VS_V45 - V52_OVER_V45) + munich_changed * geo["gain_per_changed_row"]) / N_SCORED
    prediction = {k: round(float(np.sqrt(OFFICIAL_V54 ** 2 - mse_gain * f)), 2) for k, f in TRANSFER.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "candidate_v55_combined.parquet"
    cand.to_parquet(target, index=False)
    digest = sha(target)
    shift = cand.TAXITIME_SEC_mvt.to_numpy(float) - v54.TAXITIME_SEC_mvt.to_numpy(float)
    report = {"gates": gates, "scored_days": len(days), "rows_differing_from_v52": int(hit.sum()),
              "rows_differing_from_v54": int(differs54.sum()), "eham_rows_replaced": eham, "munich": munich,
              "eham_withdrawn_by_tow_guard": int(sum(v["EHAM"]["withdrawn_by_tow_guard"] for v in per_day.values())),
              "shift_vs_v54_s": {"median": float(np.median(shift[differs54])),
                                 "p10": float(np.percentile(shift[differs54], 10)),
                                 "p90": float(np.percentile(shift[differs54], 90))},
              "prediction_inputs": {"eham_gain_per_row_vs_v45": EHAM_GAIN_VS_V45, "v52_over_v45_per_row": V52_OVER_V45,
                                    "munich_gain_per_changed_row_vs_v54_config": geo["gain_per_changed_row"],
                                    "munich_gain_per_changed_row_without_top5pct": geo["gain_per_changed_row_without_top5pct"],
                                    "mse_gain_at_full_transfer": mse_gain},
              "prediction_fixed_before_upload": prediction, "candidate_path": str(target), "candidate_sha256": digest,
              "per_day": per_day}
    MANIFEST.write_text(json.dumps(report, indent=2, default=str) + "\n")
    ready = {"key": KEY, "candidate_path": str(target), "candidate_sha256": digest,
             "rows_changed_vs_v52": int(hit.sum()), "rows_changed_vs_v54": int(differs54.sum()),
             "recipe": ("v52 (with its NM add-on, as in v54), with scored main-lane rows at EDDM replaced by the ADS-B "
                        "stand detector with X-Plane-first stands (replacing v54's OSM-stand rows) and at EHAM by the "
                        "same detector plus the tow and false-onset guard; every other row byte-identical to v52. "
                        "Validated: research_300/PREREG_geometry.md addendum B (EDDM) and PREREG_towguard.md addendum A "
                        "(EHAM, against the held-out v45 ensemble on eight fresh dates)."),
             "prediction_fixed_before_upload": dict(prediction, caveat=(
                 "Munich's X-Plane gain is validated in January only (July 2025 had almost no Munich ground coverage) "
                 "and is concentrated (4,578 per changed row without the top 5 %); Amsterdam is validated in January "
                 "and July. v54 realised 1.52 times its validated gain.")),
             "authorised": "owner, 26 September 2026 ('yes build the combined candidate with the v54 add on and upload it')",
             "written_at": datetime.now(timezone.utc).isoformat()}
    (OUT / "ready_for_upload_v55.json").write_text(json.dumps(ready, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_day"}, indent=1, default=str))


if __name__ == "__main__":
    main()

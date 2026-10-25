"""Transfer test of the FROZEN v2 stand detector to other airports, against production-strength predictions.

Rules, airports, dates and decision: research_290/PREREG_transfer_2025.md, committed and pushed before any
validation date was downloaded. The detector is stand_detector.detect_onset_v2, unchanged since ffccea1.

Two parts, each judged on its own:
  december   the nine December 2025 days already downloaded for Munich, at the other airports, whose labels were
             never read. Comparator: research_290/december_model.py (one LightGBM on source months 1 to 11),
             weaker than production, so gains are overstated.
  july2025   four new July 2025 days. Comparator: the full v45 ensemble's held-out main-lane predictions
             (.cache/sequence_release_v47/validate/ensemble_baseline.parquet, column `baseline`: 216.77 s on
             339,046 rows, replayed exactly from checkpoints trained without January and July). Its SHA-256 is
             checked against ensemble_baseline.json before use. Production today is somewhat stronger (sequence
             nets, NM add-on), so a gain here is still slightly overstated, far less than in the december part.

Per airport, the Munich rules (PREREG_stand_detector.md section 4), with V2 judged on two fixed blocks of dates
instead of single dates, because most airports cover far fewer rows per day than Munich:
  V1  MAE against the recorded block <= 60 s and >= 75 % within 60 s, on all estimates
  V2  detector RMSE below the comparator's on identical main-lane rows in EACH of the part's two date blocks;
      a block with fewer than MIN_ROWS rows makes the airport "insufficient" (no verdict)
  V3  squared-error gain > 0 after removing the most favourable 5 % of rows
  V4  2,000-draw flight bootstrap 95 % interval of the gain above zero

--dry prints detector statuses and coverage per airport and date. It computes no error against any label.
Row-level output goes to .cache/transfer_2025/ (never committed); aggregates to research_290/results/.
Run from the repository root:  python3 -u -B research_290/transfer_2025.py {december,july2025} [--dry]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADSB = Path("adsb-lol")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADSB))
import offblock_from_tracks as OB  # noqa: E402
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402

AIRPORTS = ("EHAM", "EGLL", "LEMD", "EDDF", "LSZH", "LEBL", "LFPG")
PARTS = {
    "december": {"blocks": (("2025-12-03", "2025-12-05", "2025-12-08", "2025-12-10", "2025-12-11"),
                            ("2025-12-13", "2025-12-14", "2025-12-17", "2025-12-20")),
                 "comparator": "december", "output": "transfer_december.json"},
    "july2025": {"blocks": (("2025-07-02", "2025-07-09"), ("2025-07-16", "2025-07-23")),
                 "comparator": "ensemble", "output": "transfer_july2025.json"},
}
MIN_ROWS = 20
VALIDATE = ROOT / ".cache/sequence_release_v47/validate"
CACHE = ROOT / ".cache/transfer_2025"
OFFICIAL, N_SCORED, N_MAIN = 265.2839, 344841, 339551


def comparator(kind):
    if kind == "december":
        frame = pd.read_parquet(ROOT / ".cache/munich_stand_detector/december_model.parquet")
        return frame.rename(columns={"pred": "baseline"})[["MVT_ID_mvt", "y", "ersatz_s", "baseline"]]
    path, info = VALIDATE / "ensemble_baseline.parquet", json.loads((VALIDATE / "ensemble_baseline.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["baseline_sha256"]:
        raise SystemExit("ensemble_baseline.parquet does not match its recorded SHA-256")
    return pd.read_parquet(path, columns=["MVT_ID_mvt", "y", "ersatz_s", "baseline"])


def detections(date, airports):
    tracks = pd.read_parquet(ADSB / f"tracks/{date}.parquet")
    mapping, rows, departures = OB.callsign_mapping(), [], {}
    for airport in airports:
        t = tracks[tracks.airport.astype(str).eq(airport)].drop_duplicates(["hex", "ts"]).reset_index(drop=True)
        dep = MD.departures(date, airport)
        departures[airport] = int(len(dep))
        if not len(t) or not len(dep):
            continue
        joined = dep.merge(OB.match(dep, OB.visits(t), mapping), on=["MVT_ID_mvt", "airport"], how="inner",
                           suffixes=("", "_v"))
        joined = joined[joined.groupby(["hex", "t_first"]).MVT_ID_mvt.transform("size").eq(1)]
        table = SD.stand_table(airport)
        for row in joined.itertuples(index=False):
            vertices, stand_status = SD.lookup(table, row.STAND_mvt)
            rec = {"date": date, "airport": airport, "MVT_ID_mvt": row.MVT_ID_mvt, "lane": row.lane,
                   "stand_status": stand_status, "block_epoch": row.block_epoch, "mvt_epoch": row.mvt_epoch}
            if vertices is None:
                rec["status"] = stand_status
            else:
                ts, lat, lon, gs, _ = MD.points_of(t, row)
                out = SD.detect_onset_v2(ts, lat, lon, gs, vertices, row.mvt_epoch, SD.R_PARK_M)
                rec["status"] = out["status"]
                if out["status"] == "ok":
                    rec["estimate"] = (out["t_before"] + out["t_after"]) / 2
            rows.append(rec)
    return departures, pd.DataFrame(rows)


def judge(ok, model, blocks):
    e = (ok.estimate - ok.block_epoch).to_numpy(float)
    accuracy = {"n": int(len(e)), "median_s": float(np.median(e)), "mae_s": float(np.abs(e).mean()),
                "rmse_s": float(np.sqrt((e ** 2).mean())), "within_60": int((np.abs(e) <= 60).sum()),
                "within_60_share": float((np.abs(e) <= 60).mean()), "worst": [float(e.min()), float(e.max())]}
    main = ok[ok.lane.eq("main")].merge(model, on="MVT_ID_mvt", how="inner")
    if len(main) != int(ok.lane.eq("main").sum()):
        raise SystemExit("row alignment failed: covered main-lane rows missing from the comparator")
    y, mod = main.y.to_numpy(float), main.baseline.to_numpy(float)
    det = main.mvt_epoch.to_numpy() - main.estimate.to_numpy()
    label_check = float(np.abs((main.mvt_epoch - main.block_epoch).to_numpy() - y).max()) if len(main) else 0.0
    rmse = lambda p, t: float(np.sqrt(np.mean((p - t) ** 2)))
    per_block, judged = [], []
    for block in blocks:
        m = main.date.isin(block).to_numpy()
        entry = {"dates": list(block), "rows": int(m.sum())}
        if m.sum():
            entry.update({"detector_rmse": rmse(det[m], y[m]), "comparator_rmse": rmse(mod[m], y[m])})
        per_block.append(entry)
        judged.append(m.sum() >= MIN_ROWS and entry["detector_rmse"] < entry["comparator_rmse"])
    gain = (mod - y) ** 2 - (det - y) ** 2
    report = {"accuracy": accuracy, "main_rows": int(len(main)), "label_check_max_diff_s": label_check,
              "per_block": per_block}
    if any(b["rows"] < MIN_ROWS for b in per_block):
        report["verdict"] = "insufficient"
        return report, gain
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:int(np.ceil(0.05 * len(gain)))]] = False
    boot = gain[np.random.default_rng(0).integers(0, len(gain), size=(2000, len(gain)))].mean(axis=1)
    report.update({"detector_rmse": rmse(det, y), "comparator_rmse": rmse(mod, y),
                   "ersatz_rmse": rmse(main.ersatz_s.to_numpy(float), y), "mse_gain_per_row": float(gain.mean()),
                   "mse_gain_without_top5pct": float(gain[keep].mean()),
                   "bootstrap_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]})
    checks = {"V1": accuracy["mae_s"] <= 60 and accuracy["within_60_share"] >= .75, "V2": all(judged),
              "V3": report["mse_gain_without_top5pct"] > 0, "V4": report["bootstrap_95"][0] > 0}
    report["checks"], report["verdict"] = checks, "pass" if all(checks.values()) else "fail"
    return report, gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=sorted(PARTS))
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--dates", nargs="*", help="dry runs only: other dates to inspect")
    parser.add_argument("--airports", nargs="*", default=list(AIRPORTS))
    args = parser.parse_args()
    part = PARTS[args.part]
    blocks = part["blocks"]
    if args.dates and not args.dry:
        raise SystemExit("--dates is for dry runs; judged runs use the pre-registered dates")
    args.dates = args.dates or [d for block in blocks for d in block]
    frames, departures = [], {}
    for date in args.dates:
        dep, frame = detections(date, args.airports)
        departures[date] = dep
        frames.append(frame)
        counts = frame.groupby("airport").status.apply(lambda s: int(s.eq("ok").sum())).to_dict() if len(frame) else {}
        print(date, "departures", dep, "estimates", counts, flush=True)
    rows = pd.concat(frames, ignore_index=True)
    coverage = {a: {"departures": int(sum(departures[d].get(a, 0) for d in args.dates)),
                    "matched_unique": int(rows.airport.eq(a).sum()),
                    "status": rows.loc[rows.airport.eq(a), "status"].value_counts().to_dict()} for a in args.airports}
    if args.dry:
        print(json.dumps({"dry_run": True, "coverage": coverage}, indent=1))
        return
    model = comparator(part["comparator"])
    scored = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt"])
    scored = scored[scored.PHASE_mvt.eq("DEP") & scored.TAXITIME_SEC_mvt.isna()]
    share = scored.ADEP_mvt.astype(str).value_counts(normalize=True).to_dict()
    report, projected = {"preregistration": "research_290/PREREG_transfer_2025.md", "part": args.part,
                         "blocks": [list(b) for b in blocks], "comparator": part["comparator"], "airports": {}}, 0.0
    for airport in args.airports:
        ok = rows[rows.airport.eq(airport) & rows.status.eq("ok")]
        if not len(ok):
            report["airports"][airport] = {"coverage": coverage[airport], "verdict": "insufficient"}
            continue
        result, gain = judge(ok, model, blocks)
        per_departure = result["main_rows"] / max(coverage[airport]["departures"], 1)
        result.update({"coverage": coverage[airport], "main_estimates_per_departure": per_departure,
                       "scored_share": share.get(airport, 0.0)})
        if result["verdict"] == "pass":
            projected += (N_MAIN / N_SCORED) * share.get(airport, 0.0) * per_departure * float(gain.mean())
        report["airports"][airport] = result
    report["projection"] = {
        "passing_airports": [a for a, r in report["airports"].items() if r["verdict"] == "pass"],
        "official_scale_gain_s_at_validation_coverage": float(OFFICIAL - np.sqrt(max(OFFICIAL ** 2 - projected, 0))),
        "why_an_upper_bound": "comparator weaker than today's production; coverage of the validation days, not "
                              "the scored 2026 days; no transfer discount"}
    CACHE.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(CACHE / f"rows_{args.part}.parquet", index=False)
    (ROOT / "research_290/results" / part["output"]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

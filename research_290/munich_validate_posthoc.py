"""POST-HOC diagnosis of the Munich validation. It explains the failure and changes no rule and no verdict.

The pre-registered decision in research_290/results/munich_validate.json is final: V1 passed, V2, V3 and V4 failed,
so the frozen detector does not justify further work. Everything below was computed AFTER that decision, on the
same four dates, and is therefore description, not evidence. Any successor rule it suggests needs its own
pre-registration and dates nobody has opened.

It re-runs the unchanged detector (stand_detector.detect_onset, frozen constants) on the rows it got wrong, to see
what happened physically, and splits the error by how the stand was mapped.

Aggregates go to research_290/results/munich_validate_posthoc.json; per-flight descriptions to
.cache/munich_stand_detector/posthoc_rows.json, never committed.
Run from the repository root after munich_validate.py:  python3 -u -B research_290/munich_validate_posthoc.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_290 import munich_dev as MD  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402

AIRPORT = "EDDM"
CACHE = ROOT / ".cache/munich_stand_detector"
OUTPUT = ROOT / "research_290/results/munich_validate_posthoc.json"
DETAILS = CACHE / "posthoc_rows.json"   # per-flight descriptions stay out of git
MISS_S = 120.0
GATE_S = 900.0


def summary(e):
    e = np.asarray(e, float)
    return {"n": int(len(e)), "median_s": float(np.median(e)), "mae_s": float(np.abs(e).mean()),
            "rmse_s": float(np.sqrt((e ** 2).mean())), "within_60": int((np.abs(e) <= 60).sum()),
            "beyond_300": int((np.abs(e) > 300).sum())}


def longest_still_stretch(ts, gs, start, end):
    """Longest time between consecutive observations below REST_KT, after the chosen transition and before take-off."""
    m = (ts >= start) & (ts <= end)
    t, still = ts[m], gs[m] < SD.REST_KT
    best, first = 0.0, None
    for ti, si in zip(t, still):
        if si:
            first = ti if first is None else first
            best = max(best, ti - first)
        else:
            first = None
    return float(best)


longest_still = longest_still_stretch


def misses_explained(misses):
    table, out = SD.stand_table(AIRPORT), []
    for date, group in misses.groupby("date"):
        tracks, _, joined = MD.matched_set(date, AIRPORT)
        joined = joined[joined.unique]
        for miss in group.itertuples(index=False):
            row = joined[joined.MVT_ID_mvt.eq(miss.MVT_ID_mvt)].iloc[0]
            vertices, _ = SD.lookup(table, row.STAND_mvt)
            ts, lat, lon, gs, _ = MD.points_of(tracks, row)
            det = SD.detect_onset(ts, lat, lon, gs, vertices, row.mvt_epoch, SD.R_PARK_M)
            dist = SD._metres(lat, lon, vertices)
            block = row.block_epoch
            # the recorded block time inside the track: nearest point, its distance to the stand and speed
            k = int(np.argmin(np.abs(ts - block)))
            out.append({"date": date, "stand": str(row.STAND_mvt), "stand_status": miss.stand_status,
                        "error_s": float(miss.err), "onset_count": int(det.get("onsets", 0)),
                        "onset_spread_s": float(det.get("onset_spread_s", np.nan)),
                        "bracket_s": float(det.get("gap_s", np.nan)),
                        "parked_distance_m": float(det.get("parked_distance_m", np.nan)),
                        "run_points": int(det.get("run_points", 0)),
                        "nearest_point_to_block_s": float(ts[k] - block),
                        "distance_at_block_m": float(dist[k]), "speed_at_block_kt": float(gs[k]),
                        "min_distance_before_onset_m": float(dist[ts < det["t_after"]].min())
                        if (ts < det["t_after"]).any() else None,
                        # a transition from the arrival roll-in: the aircraft was still moving fast shortly before it
                        "max_speed_in_600s_before_onset_kt": float(np.nanmax(gs[(ts < det["t_after"])
                                                                               & (ts >= det["t_before"] - 600)])),
                        "longest_still_stretch_after_onset_s": longest_still(ts, gs, det["t_after"], row.mvt_epoch),
                        "largest_track_gap_within_600s_of_block_s": float(np.diff(ts[np.abs(ts - block) <= 600]).max())
                        if (np.abs(ts - block) <= 600).sum() > 1 else None})
    return out


FAST_KT = 30.0          # landing roll or taxi-in speed: a transition this soon after it belongs to the arrival
TURNAROUND_S = 600.0    # a parked stretch shorter than this, after motion at the stand, is a push-back stop


def signature(ts, lat, lon, gs, vertices, t_takeoff):
    """The run of stand_detector.detect_onset replicated line for line, plus three descriptive facts about the
    transition it picks. Only for description: the frozen detector itself is not changed or re-scored."""
    ts, lat, lon, gs = (np.asarray(a, float) for a in (ts, lat, lon, gs))
    keep = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lon) & (ts <= t_takeoff + SD.TAKEOFF_MARGIN_S)
    order = np.argsort(ts[keep], kind="stable")
    ts, lat, lon, gs = (a[keep][order] for a in (ts, lat, lon, gs))
    d = SD._metres(lat, lon, vertices)
    at = d <= SD.R_STAND_M
    end = int(np.flatnonzero(at)[-1])
    start = end
    while start > 0 and at[start - 1]:
        start -= 1
    stop = min(end + 1, len(ts) - 1)
    still = np.isfinite(gs) & (gs < SD.REST_KT)
    moving = (np.isfinite(gs) & (gs >= SD.REST_KT)) | ~at
    onsets = [j for j in range(start + 1, stop + 1) if still[j - 1] and d[j - 1] <= SD.R_PARK_M and moving[j]]
    j = onsets[0]
    first_still = j - 1
    while first_still > start and still[first_still - 1]:
        first_still -= 1
    recent = (ts >= ts[j - 1] - 600) & (ts < ts[j])
    return {"arrival": bool(np.nanmax(gs[recent]) > FAST_KT) if recent.any() else False,
            "motion_before_parked": bool(moving[start:first_still].any()),
            "parked_stretch_s": float(ts[j - 1] - ts[first_still]),
            "several_onsets": len(onsets) > 1}


def classify(sig):
    if sig["arrival"]:
        return "arrival"
    if sig["motion_before_parked"] and sig["parked_stretch_s"] < TURNAROUND_S:
        return "pushback_unobserved"
    if sig["several_onsets"]:
        return "several_onsets"
    return "clean"


def signatures_all(frame):
    """Class of every estimate, by error band. frame: date, MVT_ID_mvt, err."""
    table, out = SD.stand_table(AIRPORT), []
    for date, group in frame.groupby("date"):
        tracks, _, joined = MD.matched_set(date, AIRPORT)
        joined = joined[joined.unique].set_index("MVT_ID_mvt")
        for est in group.itertuples(index=False):
            row = joined.loc[est.MVT_ID_mvt]
            vertices, _ = SD.lookup(table, row.STAND_mvt)
            ts, lat, lon, gs, _ = MD.points_of(tracks, row)
            out.append({"date": date, "err": est.err, "class": classify(signature(ts, lat, lon, gs, vertices,
                                                                                  row.mvt_epoch))})
    out = pd.DataFrame(out)
    out["band"] = pd.cut(out.err.abs(), [0, 60, 120, 600, np.inf], labels=["<=60", "60-120", "120-600", ">600"])
    return out


def main():
    rows = pd.read_parquet(CACHE / "validation_rows.parquet")
    ok = rows[rows.status.eq("ok")].copy()
    ok["err"] = ok.estimate - ok.block_epoch
    model = pd.read_parquet(CACHE / "december_model.parquet")
    x = ok[ok.lane.eq("main")].merge(model, on="MVT_ID_mvt", how="inner")
    x["det"] = x.mvt_epoch - x.estimate
    x["gain"] = (x.pred - x.y) ** 2 - (x.det - x.y) ** 2
    order = x.sort_values("gain").reset_index(drop=True)
    worst = order.iloc[0]
    sse = float(((x.det - x.y) ** 2).sum())
    rest = order.iloc[1:]
    report = {
        "note": "POST-HOC, computed after the pre-registered verdict; description only, changes no decision",
        "error_bins_s": {k: int(v) for k, v in pd.cut(ok.err.abs(), [0, 60, 120, 300, 600, 1200, np.inf])
                         .value_counts().sort_index().rename(lambda i: f"{i.left:g}-{i.right:g}").items()},
        "by_stand_mapping": {s: summary(g.err) for s, g in ok.groupby("stand_status")},
        "sign_of_misses": {"late": int((ok.err > MISS_S).sum()), "early": int((ok.err < -MISS_S).sum())},
        "worst_row": {"date": worst.date, "stand_status": worst.stand_status, "error_s": float(worst.err),
                      "share_of_detector_sse": float((worst.det - worst.y) ** 2 / sse)},
        "without_worst_row": {"rows": int(len(rest)),
                              "detector_rmse": float(np.sqrt(((rest.det - rest.y) ** 2).mean())),
                              "comparator_rmse": float(np.sqrt(((rest.pred - rest.y) ** 2).mean())),
                              "mse_gain_per_row": float(rest.gain.mean())},
    }
    details = {"worst_row": {"detector_taxi_s": float(worst.det), "comparator_taxi_s": float(worst.pred),
                             "true_taxi_s": float(worst.y)},
               "misses_explained": misses_explained(ok[ok.err.abs() > MISS_S])}
    DETAILS.write_text(json.dumps(details, indent=2, default=float) + "\n")
    report["misses_beyond_120s_by_stand_block"] = pd.Series(
        [m["stand"][:2] + "x" for m in details["misses_explained"]]).value_counts().to_dict()
    # A model-disagreement gate was considered as the successor rule and rejected: where the two disagree most,
    # the detector is usually the one that is right (long taxi-outs the model underestimates).
    far = x[(x.det - x.pred).abs() > GATE_S]
    right = far[(far.det - far.y).abs() < (far.pred - far.y).abs()]
    report["disagreement_gate_descriptive"] = {
        "gate_s": GATE_S, "rows": int(len(far)), "detector_closer": int(len(right)),
        "squared_error_given_up_where_detector_right": float(right.gain.sum()),
        "squared_error_saved_where_model_right": float(-far.drop(right.index).gain.sum())}
    classes = signatures_all(ok[["date", "MVT_ID_mvt", "err"]])
    dev = pd.read_parquet(MD.ADSB / f"tracks/_dev_rows_{MD.DATE}.parquet")
    dev = dev[dev.unique & dev.on30_status.eq("ok")].copy()
    dev["date"] = MD.DATE
    dev["err"] = (dev.on30_t_before + dev.on30_t_after) / 2 - dev.block_epoch
    dev_classes = signatures_all(dev[["date", "MVT_ID_mvt", "err"]])
    report["signature_thresholds"] = {"fast_kt": FAST_KT, "turnaround_s": TURNAROUND_S}
    for name, c in (("validation_dates", classes), ("development_day", dev_classes)):
        report[f"classes_by_error_band_{name}"] = {
            k: {str(b): int(v) for b, v in g.band.value_counts().sort_index().items()} for k, g in c.groupby("class")}
        report[f"clean_only_{name}"] = summary(c.loc[c["class"].eq("clean"), "err"])
    OUTPUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()

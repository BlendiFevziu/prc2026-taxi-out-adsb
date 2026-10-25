"""DIAGNOSTIC, no model: do the owner's leads 2, 3, 4 and 6 show structure that production leaves in its residuals?

Residuals of the full v45 ensemble's held-out predictions on January and July 2025 main-lane rows
(.cache/sequence_release_v47/validate/ensemble_baseline.parquet, SHA-256 checked; the ADS-B corrections are not in these
predictions, which is stated with every result). Every signal is computed from challenge fields known at ranking time
(times, runway, stand, aircraft type, other flights' movement times) and open OSM geometry; no label enters a signal.

Honest potential per signal, two transfers: bins within each airport; residual bin means learned on January rows are
applied to July rows and vice versa (across season), and, closer to the task (2025 to 2026 in the same month), bins
within airport and month learned on odd days are applied to even days and vice versa (within month). Means are shrunk by
n / (n + 50); the held-out MSE reduction is reported, pooled and per month, with the share from the best 100 rows.
  lead 3  AOBT_3 minus EOBT_1, AOBT_3 minus SCHED, AOBT_3 rounded to the minute, exact 20/25 min gaps
  lead 4  departures at the airport in the previous 10 minutes, same runway in the previous 10 minutes, gap to the
          previous departure on the same runway, heavy previous departure (wake proxy)
  lead 2  straight-line distance from the stand to the threshold of the departure runway (OSM)
  lead 6  where the largest residuals concentrate (share of squared error in the top 1 %, observables there)
Run from the repository root:  python3 -u -B research_320/leads_diagnostic.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_290 import stand_detector as SD  # noqa: E402

VALID = HERE / ".cache/sequence_release_v47/validate"
OUT = HERE / "research_320/results/leads_diagnostic.json"
HEAVY = ("A33", "A34", "A35", "A38", "B74", "B76", "B77", "B78", "B74", "A30", "IL9", "MD1")
ID = "MVT_ID_mvt"


def load():
    info = json.loads((VALID / "ensemble_baseline.json").read_text())
    if hashlib.sha256((VALID / "ensemble_baseline.parquet").read_bytes()).hexdigest() != info["baseline_sha256"]:
        raise SystemExit("comparator hash mismatch")
    base = pd.read_parquet(VALID / "ensemble_baseline.parquet", columns=[ID, "y", "baseline"])
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt",
            "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]
    raw = pd.concat(pd.read_parquet(HERE / f"data/training_2025-{m}-01_2025-{n}-01.parquet", columns=cols)
                    for m, n in (("01", "02"), ("07", "08")))
    dep = raw[raw.PHASE_mvt.eq("DEP")].copy()
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt"):
        dep[c] = pd.to_datetime(dep[c], utc=True)
    dep["airport"] = dep.ADEP_mvt.astype(str)
    dep["t"] = (dep.MVT_TIME_UTC_mvt - pd.Timestamp("2025-01-01", tz="UTC")).dt.total_seconds()
    # queue signals over ALL departures (movement times are known at ranking time), then restrict to scored main rows
    dep = dep.sort_values(["airport", "t"]).reset_index(drop=True)
    dep["dep_prev10"] = np.nan
    dep["rwy_prev10"] = np.nan
    dep["rwy_gap_s"] = np.nan
    dep["prev_heavy"] = np.nan
    for ap, g in dep.groupby("airport"):
        t = g.t.to_numpy()
        dep.loc[g.index, "dep_prev10"] = np.searchsorted(t, t, side="left") - np.searchsorted(t, t - 600, side="left")
        for rwy, h in g.groupby(g.RUNWAY_mvt.astype(str)):
            th = h.t.to_numpy()
            dep.loc[h.index, "rwy_prev10"] = np.searchsorted(th, th, side="left") - np.searchsorted(th, th - 600, side="left")
            dep.loc[h.index, "rwy_gap_s"] = np.concatenate([[np.nan], np.diff(th)])
            heavy = h.AIRCRAFT_TYPE_mvt.astype(str).str[:3].isin(HEAVY).to_numpy().astype(float)
            dep.loc[h.index, "prev_heavy"] = np.concatenate([[np.nan], heavy[:-1]])
    d = base.merge(dep, on=ID, how="inner", validate="one_to_one")
    d["resid"] = d.y - d.baseline
    d["month"] = d.MVT_TIME_UTC_mvt.dt.month.map({1: "01", 7: "07"})
    d["aobt_minus_eobt_min"] = (d.AOBT_3_flt - d.EOBT_1_flt).dt.total_seconds() / 60
    d["aobt_minus_sched_min"] = (d.AOBT_3_flt - d.SCHED_TIME_UTC_mvt).dt.total_seconds() / 60
    d["aobt_rounded"] = (d.AOBT_3_flt.dt.second.eq(0)).astype(float)
    gap = d.aobt_minus_eobt_min.round(6)
    d["gap_default"] = np.select([gap.abs().eq(0), gap.abs().eq(20), gap.abs().eq(25)], [0.0, 20.0, 25.0], np.nan)
    d["stand_rwy_m"] = stand_runway_distance(d)
    return d


def runway_ends(icao):
    data = json.loads((HERE / f"geometry/{icao.lower()}_osm.json").read_text())
    out = {}
    for e in data.get("elements", []):
        tags = e.get("tags", {})
        if tags.get("aeroway") != "runway" or not tags.get("ref") or not e.get("geometry"):
            continue
        a, b = e["geometry"][0], e["geometry"][-1]
        brg = np.degrees(np.arctan2((b["lon"] - a["lon"]) * np.cos(np.radians(a["lat"])), b["lat"] - a["lat"])) % 360
        for des in tags["ref"].split("/"):
            des = des.strip()
            digits = "".join(ch for ch in des if ch.isdigit())
            if not digits:
                continue
            h = int(digits) * 10 % 360
            diff = min(abs(brg - h), 360 - abs(brg - h))
            start = a if diff < 90 else b             # take-off roll starts at the end the heading points away from
            out[des.upper()] = (start["lat"], start["lon"])
    return out


def stand_runway_distance(d):
    dist = np.full(len(d), np.nan)
    for ap in d.airport.unique():
        try:
            ends, table = runway_ends(ap), SD.stand_table(ap)
        except Exception:
            continue
        idx = np.flatnonzero(d.airport.eq(ap).to_numpy())
        for i in idx:
            rwy = str(d.RUNWAY_mvt.iat[i]).upper().strip()
            if rwy not in ends:
                continue
            v, _ = SD.lookup(table, d.STAND_mvt.iat[i])
            if v is None:
                continue
            c = v.mean(axis=0)
            dist[i] = float(SD._metres(np.array([c[0]]), np.array([c[1]]), np.array([ends[rwy]]))[0])
    return dist


def cross_fitted(d, col, bins=10, k=50.0):
    """Out-of-month MSE reduction from shrunk airport-bin means of the residual."""
    x = d[col].to_numpy(float)
    ok = np.isfinite(x)
    key = np.full(len(d), "", dtype=object)
    for ap in d.airport.unique():
        m = ok & d.airport.eq(ap).to_numpy()
        if m.sum() < 200:
            continue
        uniq = np.unique(x[m])
        if len(uniq) <= bins:
            lab = x[m].astype(str)
        else:
            edges = np.unique(np.quantile(x[m], np.linspace(0, 1, bins + 1)))
            lab = np.searchsorted(edges[1:-1], x[m], side="right").astype(str)
        key[m] = ap + "|" + lab
    r = d.resid.to_numpy(float)
    gain = np.zeros(len(d))
    for learn, apply in (("01", "07"), ("07", "01")):
        L, P = d.month.eq(learn).to_numpy() & (key != ""), d.month.eq(apply).to_numpy() & (key != "")
        stats = pd.DataFrame({"k": key[L], "r": r[L]}).groupby("k").r.agg(["sum", "count"])
        corr = (stats["sum"] / (stats["count"] + k)).to_dict()
        b = np.array([corr.get(kk, 0.0) for kk in key[P]])
        gain[P] = r[P] ** 2 - (r[P] - b) ** 2
    top = np.sort(gain)[-100:].sum()
    total = gain.sum()
    return {"rows_with_signal": int((key != "").sum()), "gain_mse_per_main_row": float(gain.mean()),
            "january": float(gain[d.month.eq("01").to_numpy()].mean()), "july": float(gain[d.month.eq("07").to_numpy()].mean()),
            "share_from_best_100": float(top / total) if total > 0 else None,
            "official_s_upper": float(gain.mean() * 339551 / 344841 / (2 * 260.93))}


def cross_day(d, col, bins=10, k=50.0):
    """Within-month transfer: airport-month-bin means learned on odd days, applied to even days, and vice versa."""
    x = d[col].to_numpy(float)
    ok = np.isfinite(x)
    key = np.full(len(d), "", dtype=object)
    for (ap, mo) in d.groupby(["airport", "month"]).groups:
        m = ok & d.airport.eq(ap).to_numpy() & d.month.eq(mo).to_numpy()
        if m.sum() < 200:
            continue
        uniq = np.unique(x[m])
        lab = x[m].astype(str) if len(uniq) <= bins else np.searchsorted(
            np.unique(np.quantile(x[m], np.linspace(0, 1, bins + 1)))[1:-1], x[m], side="right").astype(str)
        key[m] = ap + "|" + mo + "|" + lab
    odd = (d.MVT_TIME_UTC_mvt.dt.day % 2).eq(1).to_numpy()
    r = d.resid.to_numpy(float)
    gain = np.zeros(len(d))
    for L, P in ((odd, ~odd), (~odd, odd)):
        L, P = L & (key != ""), P & (key != "")
        stats = pd.DataFrame({"k": key[L], "r": r[L]}).groupby("k").r.agg(["sum", "count"])
        corr = (stats["sum"] / (stats["count"] + k)).to_dict()
        b = np.array([corr.get(kk, 0.0) for kk in key[P]])
        gain[P] = r[P] ** 2 - (r[P] - b) ** 2
    total = gain.sum()
    return {"gain_mse_per_main_row": float(gain.mean()), "january": float(gain[d.month.eq("01").to_numpy()].mean()),
            "july": float(gain[d.month.eq("07").to_numpy()].mean()),
            "share_from_best_100": float(np.sort(gain)[-100:].sum() / total) if total > 0 else None,
            "official_s_upper": float(gain.mean() * 339551 / 344841 / (2 * 260.93))}


def main():
    d = load()
    report = {"rows": int(len(d)), "comparator": "v45 full ensemble held-out, January and July 2025 (ADS-B corrections not included)",
              "leads": {}}
    for lead, cols in (("3_source_regime", ("aobt_minus_eobt_min", "aobt_minus_sched_min", "aobt_rounded", "gap_default")),
                       ("4_runway_queue", ("dep_prev10", "rwy_prev10", "rwy_gap_s", "prev_heavy")),
                       ("2_stand_runway_distance", ("stand_rwy_m",))):
        report["leads"][lead] = {c: dict(cross_fitted(d, c), within_month=cross_day(d, c)) for c in cols}
    se = d.resid.to_numpy(float) ** 2
    top = se >= np.quantile(se, 0.99)
    report["leads"]["6_main_tail"] = {
        "top_1pct_share_of_sse": float(se[top].sum() / se.sum()),
        "top_1pct_by_airport": d[top].airport.value_counts(normalize=True).round(3).to_dict(),
        "all_rows_by_airport": d.airport.value_counts(normalize=True).round(3).to_dict(),
        "top_1pct_mean_resid_s": float(d.resid[top].mean()),
        "top_1pct_median_abs_aobt_minus_sched_min": float(d.aobt_minus_sched_min[top].abs().median()),
        "all_median_abs_aobt_minus_sched_min": float(d.aobt_minus_sched_min.abs().median())}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['rows']} main-lane rows; residual RMSE {np.sqrt(se.mean()):.2f} s")
    for lead, sigs in report["leads"].items():
        if lead == "6_main_tail":
            print(lead, json.dumps(sigs))
            continue
        for c, r in sigs.items():
            print(f"{lead:26s} {c:22s} rows {r['rows_with_signal']:6d}  gain/row {r['gain_mse_per_main_row']:7.1f}  "
                  f"Jan {r['january']:7.1f} Jul {r['july']:7.1f}  best-100 share {r['share_from_best_100'] if r['share_from_best_100'] is None else round(r['share_from_best_100'], 2)}  "
                  f"~{r['official_s_upper']:.3f} s")
            w = r["within_month"]
            print(f"{'':26s} {'  within month':22s} {'':11s}  gain/row {w['gain_mse_per_main_row']:7.1f}  Jan {w['january']:7.1f} Jul {w['july']:7.1f}  "
                  f"best-100 share {w['share_from_best_100'] if w['share_from_best_100'] is None else round(w['share_from_best_100'], 2)}  ~{w['official_s_upper']:.3f} s")


if __name__ == "__main__":
    main()

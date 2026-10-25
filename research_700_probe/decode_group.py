"""Decode group probes from ledger.csv: from the official score, the sum and mean of (y - p) over the group and the gain of the
optimal uniform shift. dSSE = N*(S^2 - S80^2) = n*C^2 + 2*C*sum(p - y)  ->  sum(y - p) = (n*C^2 - dSSE)/(2*C).
Score rounding gives +-1.7e4 in dSSE, i.e. +-1.7e4/(2*C) in the sum. Gain of shifting the group by its mean residual r: n*r^2.
Writes research_700_probe/group_results.csv. Run from the repository root:  python3 -B research_700_probe/decode_group.py
"""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
N, S80, H = 344841, 245.046, 5e-5
out = []
for r in csv.DictReader((ROOT / "research_700_probe/ledger.csv").open()):
    if not r["row_id"].startswith("group:") or not r["official_score"]:
        continue
    n, C, S = int(float(r["old"])), float(r["step"]), float(r["official_score"])
    dsse = N * (S ** 2 - S80 ** 2)
    tot = (n * C ** 2 - dsse) / (2 * C)
    tol = 1.7e4 / (2 * C)
    mean = tot / n
    out.append(dict(probe=r["probe"], group=r["row_id"][6:], n=n, score=S, sum_resid=round(tot), sum_tol=round(tol), mean_resid_sec=round(mean, 1), gain_sse=round(n * mean ** 2)))
    print(f"probe {r['probe']} {r['row_id'][6:]:15s} n={n:5d} mean(y-p) = {mean:9.1f} s (+-{tol / n:.2f})  optimal uniform shift gain {n * mean ** 2:.3e} SSE")
if out:
    with (ROOT / "research_700_probe/group_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)
    print(f"total gain if all shifts applied: {sum(o['gain_sse'] for o in out):.3e} SSE")

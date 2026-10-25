"""Decode single-row probes from ledger.csv: from the official score, the implied label y of the changed row.
dSSE = N*(S^2 - S80^2) = (new-y)^2 - (old-y)^2 = (new-old)*(new+old-2y)  ->  y = (new+old-dSSE/(new-old))/2.
The score is rounded to 4 decimals, so the half-step bounds give the uncertainty. Reference: v80 = 245.046 (also rounded).
Writes y and its bounds to research_700_probe/decoded.csv. Run from the repository root:  python3 -B research_700_probe/decode_probe.py
"""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
N, S80, H = 344841, 245.046, 5e-5
rows = [r for r in csv.DictReader((ROOT / "research_700_probe/ledger.csv").open()) if r["official_score"] and ":" not in r["row_id"]]
out = []
for r in rows:
    o, n, S = float(r["old"]), float(r["new"]), float(r["official_score"])
    y = lambda s, s0: (n + o - N * (s ** 2 - s0 ** 2) / (n - o)) / 2
    ys = [y(S + a, S80 + b) for a in (-H, H) for b in (-H, H)]
    mid = y(S, S80)
    out.append(dict(probe=r["probe"], row_id=r["row_id"], old=o, new=n, score=S, y=round(mid, 1), y_lo=round(min(ys), 1), y_hi=round(max(ys), 1)))
    print(f"probe {r['probe']} row {r['row_id']}: y = {mid:,.1f} s  [{min(ys):,.1f} .. {max(ys):,.1f}]  ({mid/3600:.3f} h)  old {o:,.1f}")
if out:
    with (ROOT / "research_700_probe/decoded.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)

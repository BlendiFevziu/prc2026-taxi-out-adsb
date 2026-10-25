"""Decode the 12 Rome rows changed by v82 from the official scores of v80 and v82 (probe 0 of the ledger).
Contamination: v81's 9,683 Amsterdam rows, expected -4.5e6 SSE, allowed within [-2e7, +1e7]."""
import itertools, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
N, S80, S82 = 344841, 245.046, 247.3075
rows = json.loads((ROOT / "research_680_rome_overnight/results/v82_build.json").read_text())["rows"]
target = N * (S82 ** 2 - S80 ** 2)
lo_c, hi_c = -2e7, 1e7                     # v81 contribution band
TAXI = (300.0, 3000.0)
def contrib(o, n, y):                       # change from o to n, label y
    return (n - y) ** 2 - (o - y) ** 2
opts = []
for r in rows:
    d, o, n = float(r["delay"]), float(r["old"]), float(r["new_written"])
    cls = {"copy": (d, d), "normal": TAXI, "shift24": (86400 + TAXI[0], 86400 + TAXI[1])}
    rng = {}
    for k, (a, b) in cls.items():
        ys = np.linspace(a, b, 50)
        c = contrib(o, n, ys)
        rng[k] = (c.min(), c.max())
    opts.append((r["MVT_ID_mvt"], r["part"], d, rng))
print(f"target dSSE {target:.4e}")
sols = []
for combo in itertools.product(("copy", "normal", "shift24"), repeat=len(opts)):
    lo = sum(o[3][k][0] for o, k in zip(opts, combo)) + lo_c
    hi = sum(o[3][k][1] for o, k in zip(opts, combo)) + hi_c
    if lo - 2e4 <= target <= hi + 2e4:
        sols.append(combo)
print(len(sols), "consistent combinations of", 3 ** len(opts))
from collections import Counter
for i, (mid, part, d, rng) in enumerate(opts):
    c = Counter(s[i] for s in sols)
    print(f"{mid} part {part} delay {d/3600:5.1f} h  " + "  ".join(f"{k}:{c.get(k,0)}" for k in ("copy","normal","shift24")))

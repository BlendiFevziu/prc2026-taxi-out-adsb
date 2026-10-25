"""Can one upload decode k rows at once? For rows 6..(5+k) of the inventory, search step sizes so that all 3^k label-kind
combinations (copy, normal, shift24) give dSSE intervals that do not overlap. Interval per row and kind: copy y = delay +-5 s,
normal y in [300, 3000] s, shift24 y in 86400 + [300, 3000] s. Score resolution: +-1.7e4 SSE (4-decimal rounding of two scores).
Run from the repository root:  python3 -B research_700_probe/design_multi.py [k]
"""
import itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TOL = 1.7e4
inv = pd.read_parquet(ROOT / ".cache/research_700/inventory.parquet").iloc[5:5 + K]
d, p = inv.delay.to_numpy(float), inv.p.to_numpy(float)
rng = {"copy": lambda d: (d - 5, d + 5), "normal": lambda d: (300.0, 3000.0), "shift24": lambda d: (86700.0, 89400.0)}

def interval(o, n, kind, dl):
    a, b = rng[kind](dl)
    ys = np.array([a, b]); c = (n - ys) ** 2 - (o - ys) ** 2          # monotone in y, so the ends bound it
    return c.min(), c.max()

def ambiguity(steps):
    iv = {}
    for combo in itertools.product(rng, repeat=K):
        lo = hi = 0.0
        for i, kind in enumerate(combo):
            a, b = interval(p[i], p[i] + steps[i], kind, d[i]); lo += a; hi += b
        iv[combo] = (lo - TOL, hi + TOL)
    items = sorted(iv.values()); worst = bad = 0
    for i in range(len(items)):                                         # sorted sweep: count overlapping pairs
        j = i + 1
        while j < len(items) and items[j][0] <= items[i][1]:
            bad += 1; j += 1
    return bad / (len(items) * (len(items) - 1) / 2), iv

best = None
gen = np.random.default_rng(0)
for _ in range(400):
    steps = gen.choice([-1, 1], K) * gen.uniform(2e3, 6e4, K)
    a, _ = ambiguity(steps)
    if best is None or a < best[0]:
        best = (a, steps)
    if a == 0:
        break
print(f"k={K} rows {inv.MVT_ID_mvt.astype(str).tolist()}  overlapping combo pairs: {best[0]:.4%}  steps {np.round(best[1]).tolist()}")
print("separable in one upload" if best[0] == 0 else "NOT fully separable with these tolerances")

"""Out-of-sample single-LightGBM comparator for February and June 2025 main-lane departures.

Purpose: to check the frozen ADS-B stand detectors on months they were not validated on, against a production-recipe
tree model that has not seen the month it predicts (the project's six-pair convention: pairs (2,8) and (6,12)).
Recipe is release_common.tree (1,106 rounds, 255 leaves, learning rate 0.03) on the 91-column main schema, target
y - ersatz_s, the 300 s floor of the production predictor. One seed, no CatBoost, no networks, so it is WEAKER than
the production ensemble; every comparison against it must say so.

Run from the repository root:  python3 -u -B research_800_final/comparator_feb_jun_2025.py
Output: .cache/research_800/comparator_feb_jun_2025.parquet
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_250 import release_common as RC   # torch before the tree libraries
import main_model as H
import numpy as np
import pandas as pd
from research_250 import candidate_features as B

OUT = ROOT / ".cache/research_800/comparator_feb_jun_2025.parquet"
TABLE = ROOT / ".cache/clock_queue_release_20260912/data/months_1_2_3_4_5_6_7_8_9_10_11_12.parquet"
PAIRS = {2: (2, 8), 6: (6, 12)}
T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)

frame = pd.read_parquet(TABLE)
frame = frame[frame.hat_aobt.eq(1)].reset_index(drop=True)
frame["month"] = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.month
frame["rest"] = frame.y - frame.ersatz_s
features = B.main_features(H.MERKMALE_HAUPT)
log(f"main rows {len(frame)}; per month {frame.month.value_counts().sort_index().to_dict()}")
parts = []
for target, pair in PAIRS.items():
    train = frame[~frame.month.isin(pair)]
    test = frame[frame.month.eq(target)]
    log(f"fit without months {pair}: {len(train)} rows -> predict month {target}: {len(test)} rows")
    model = RC.tree(1106, 0, 8).fit(train[features], train.rest)
    pred = np.maximum(test.ersatz_s.to_numpy() + model.predict(test[features]), 300.0)
    parts.append(pd.DataFrame({"MVT_ID_mvt": test.MVT_ID_mvt.to_numpy(), "comparator": pred, "y": test.y.to_numpy(float),
                               "ersatz_s": test.ersatz_s.to_numpy(float), "ADEP_mvt": test.ADEP_mvt.astype(str).to_numpy(),
                               "MVT_TIME_UTC_mvt": test.MVT_TIME_UTC_mvt.to_numpy(), "month": target}))
    rmse = float(np.sqrt(np.mean((parts[-1].comparator - parts[-1].y) ** 2)))
    log(f"month {target}: comparator RMSE on main rows {rmse:.2f}")
out = pd.concat(parts, ignore_index=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT, index=False)
log(f"wrote {OUT} ({len(out)} rows)")

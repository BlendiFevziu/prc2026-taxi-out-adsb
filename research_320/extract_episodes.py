"""v56's partial-track feature rows plus ordered stop/move episodes, for any January/July 2025 dates. Label-free inputs.

Runs Codex's own research_310/multiairport_screen.extract() from the frozen, read-only checkout of commit 9884150
(../skenderbeu-v56-readonly): same departures, comparator join, matching (FLIGHT_mvt plus CALLSIGN_flt, unique
visits), X-Plane-first stand map at EHAM, category rules and 23 trajectory summaries. The only change is that its
summarise() also returns research_320/episodes.episodes() computed on the SAME points, stand and take-off, so every
row carries both. The date list is set on the module; nothing in the checkout is edited.

Row caches go to .cache/research_320/features/ (never committed). Used by the audit and the experiment.
"""
import hashlib
from pathlib import Path
import subprocess
import sys

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
V56 = HERE.parent / "skenderbeu-v56-readonly"
COMMIT = "9884150"
FEATURES_CODE = "53b8cee98a02ca7b3e34dc0b509072fce2bf8ed31ac8ddeeba7a335ce9637248"   # partial_track_features.py
SCREEN_CODE = "66e003dbe740af70a6daf4fae168ac6eb6dc68da5e1cb0498a5a5521d85d216c"     # multiairport_screen.py
CACHE = HERE / ".cache/research_320/features"


def v56_modules():
    head = subprocess.run(["git", "-C", str(V56), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    for name, want in (("research_310/partial_track_features.py", FEATURES_CODE),
                       ("research_310/multiairport_screen.py", SCREEN_CODE)):
        if hashlib.sha256((V56 / name).read_bytes()).hexdigest() != want or not head.startswith(COMMIT):
            raise SystemExit(f"the v56 checkout is not the frozen {COMMIT}")
    if str(V56) not in sys.path:
        sys.path.insert(0, str(V56))
    from research_310 import multiairport_screen as MS
    return MS


def extract(dates, tag):
    """All v56 feature rows of those dates at the five airports, plus the episode columns; cached per tag."""
    target = CACHE / f"{tag}.parquet"
    if target.exists():
        return pd.read_parquet(target)
    from research_320 import episodes as E
    MS = v56_modules()
    CACHE.mkdir(parents=True, exist_ok=True)
    saved = (MS.TRAIN, MS.TEST, MS.CACHE, MS.summarise)
    original = MS.summarise
    MS.TRAIN, MS.TEST, MS.CACHE = tuple(dates), (), CACHE / f"days_{tag}"
    MS.CACHE.mkdir(parents=True, exist_ok=True)
    MS.summarise = lambda points, stand, takeoff: {**original(points, stand, takeoff),
                                                   **E.episodes(points, stand, takeoff)}
    try:
        frame, _, _ = MS.extract()
    finally:
        MS.TRAIN, MS.TEST, MS.CACHE, MS.summarise = saved
    frame.to_parquet(target, index=False)
    return frame

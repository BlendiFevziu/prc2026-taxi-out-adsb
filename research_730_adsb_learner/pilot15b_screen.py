"""Pilot 15b (PREREG_corrector_capacity_small.md): pilot 15 with a smaller grid; ties go to the default.
  python3 -u -B research_730_adsb_learner/pilot15b_screen.py test | build
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_730_adsb_learner import pilot15_screen as C

C.OUTPUT = C.ROOT / "research_730_adsb_learner/results/pilot15b_screen.json"
C.GRID = [C.DEFAULT] + [(r, l) for r in (300, 600) for l in (15, 31, 63) if (r, l) != C.DEFAULT]   # default first: the tie rule
C.PREREG = "research_730_adsb_learner/PREREG_corrector_capacity_small.md"

if __name__ == "__main__":
    {"test": C.test, "build": C.build}[sys.argv[1]]()

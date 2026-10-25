"""Were the sequence nets' training constants ever chosen? Probe the epoch count on one seed.

`neural_model.py` carries EPOCHEN 12, STAPEL 8192, LERNRATE 2e-3, BREITEN [1024, 512, 256] and DROP 0.15 as bare
constants with no tuning recorded anywhere in the repository. Three of today's four gains came from exactly that:
a constant nobody had checked. These sit on the component that supplies 35 % of the main-lane residual, and the
blend grid wanted to give it MORE weight (0.45 was the pooled minimum), so a better net is worth more than usual.

One caveat shapes the test. `fit_net` drives OneCycleLR with `total_steps` computed from the epoch count, so the
schedule anneals to zero at the last epoch: the nets are NOT left mid-descent at 12, and "train longer" means
"run a longer schedule", not "resume". Each epoch count is therefore a separate, complete run.

This probes ONE seed against its recorded result to decide whether a full refit is worth about two hours. Seed 29
scored a floored component RMSE of 224.3060 at 12 epochs, in 889 s.

Decision rule, fixed before the run: a refit of all five seeds is worth proposing only if 20 epochs improves the
component RMSE by at least 0.5 s. Below that the ensemble effect, at a 0.35 blend weight, cannot repay the
compute or the release risk.

  python3 -u -B research_280/sequence_epochs_probe.py --epochs 20 --seed 29
"""
import argparse
import json
import time
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import neural_model as NM  # noqa: F401  torch before tree libraries on macOS
import torch
import joblib
import numpy as np
import pandas as pd
from research_260 import sequence_features as F, sequence_model as S

VALIDATE = PROJECT / ".cache/sequence_release_v47/validate"
OUTPUT = PROJECT / "research_280/results/sequence_epochs_probe.json"
MODE = "origin"
RECORDED = {29: 224.30602527423758, 13: None, 21: None, 37: 225.47897957591172, 45: 224.39414070941186}
BAR = 0.5


def log(*p):
    print(time.strftime("%H:%M:%S"), *p, flush=True)


def main(epochs, seed):
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable; refusing a silent CPU fallback")
    prep = joblib.load(VALIDATE / "preprocessing.pkl")
    xk, xn, y = (np.load(VALIDATE / f"train_{p}.npy") for p in ("xk", "xn", "residual"))
    xs, ms = np.load(VALIDATE / f"train_{MODE}_xs.npy"), np.load(VALIDATE / f"train_{MODE}_ms.npy")
    log(f"seed {seed}, {epochs} epochs, {len(y):,} rows (recorded at 12 epochs: {RECORDED[seed]:.4f})")
    started = time.time()
    package = S.fit_net(True, seed, xk, xn, xs, ms, y, prep["kdim"],
                        log=lambda m: log(f"  {m}"), epochs=epochs)
    del xk, xn, xs, ms, y
    torch.mps.empty_cache()
    vk, vn = np.load(VALIDATE / "valid_xk.npy"), np.load(VALIDATE / "valid_xn.npy")
    vs, vm = np.load(VALIDATE / f"valid_{MODE}_xs.npy"), np.load(VALIDATE / f"valid_{MODE}_ms.npy")
    package.update(preprocessing=prep, mode=MODE, feature_schema=F.SCHEMA)
    residual = S.predict(package, vk, vn, vs, vm)
    rows = pd.read_parquet(VALIDATE / "valid_rows.parquet")
    error = np.maximum(rows.ersatz_s.to_numpy(float) + residual, 300.) - rows.y.to_numpy(float)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    gain = RECORDED[seed] - rmse
    res = {"seed": seed, "epochs": epochs, "component_rmse": rmse, "recorded_at_12_epochs": RECORDED[seed],
           "gain_s": gain, "bar": BAR, "worth_a_full_refit": bool(gain >= BAR),
           "seconds": time.time() - started}
    np.save(VALIDATE / f"probe_{MODE}_{seed}_e{epochs}_residual.npy", residual)
    print()
    print(json.dumps(res, indent=1))
    print(f"\n{'PASSES' if gain >= BAR else 'FAILS'} the 0.5 s bar: "
          f"{RECORDED[seed]:.4f} -> {rmse:.4f} ({gain:+.4f} s)")
    prev = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else []
    OUTPUT.write_text(json.dumps(prev + [res], indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--seed", type=int, default=29)
    args = a.parse_args()
    main(args.epochs, args.seed)

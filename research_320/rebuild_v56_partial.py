"""Rebuild v56's partial-track residual screen from Codex's frozen code and verify it against his record. No upload.

Why: Codex's private artefacts (the 21 saved models, all_features.parquet, predictions.parquet) sit in his worktree
under ~/Documents, which macOS does not let this app read. Every INPUT they were built from is readable here and
hash-identical to his record (25 of 25 source hashes). So his exact research_310/multiairport_screen.main() is run
from a detached, read-only checkout of commit 9884150, with only its two output paths redirected to this repository's
cache, and the result is compared with research_310/results/multiairport_screen.json of that commit: all 21 model
hashes, and every arm's and seed's metrics.

The checkout: ../skenderbeu-v56-readonly (git worktree add --detach, 9884150). Nothing there is edited; outputs go
to .cache/research_320/v56_rebuild/ here. Runs under the shared heavy-work lock (main() takes it itself).
Run from this repository's root:  python3 -u -B research_320/rebuild_v56_partial.py
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
V56 = HERE.parent / "skenderbeu-v56-readonly"
COMMIT = "9884150"
OUT = HERE / ".cache/research_320/v56_rebuild"
RESULT = HERE / "research_320/results/v56_rebuild.json"
CODE = {"research_310/multiairport_screen.py": "66e003dbe740af70a6daf4fae168ac6eb6dc68da5e1cb0498a5a5521d85d216c",
        "research_310/partial_track_features.py": "53b8cee98a02ca7b3e34dc0b509072fce2bf8ed31ac8ddeeba7a335ce9637248",
        "research_310/partial_track_audit.py": "5e558c87abee45a6e8aa5d852a37fe2cffc70c84d39262bbe52a4016c1c4c737",
        "research_310/supporting_evidence.py": "4b62a0578f35cfaec18851f4b8bef74e9834a669e9febe05e1be6b0f3789e03d"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


def main():
    head = subprocess.run(["git", "-C", str(V56), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(V56), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
    if not head.startswith(COMMIT) or dirty.strip():
        raise SystemExit(f"checkout is not a clean {COMMIT}: {head} {dirty[:200]}")
    code_ok = {f: sha(V56 / f) == h for f, h in CODE.items()}
    if not all(code_ok.values()):
        raise SystemExit(f"code hashes differ: {code_ok}")
    recorded = json.loads((V56 / "research_310/results/multiairport_screen.json").read_text())
    sys.path.insert(0, str(V56))
    from research_310 import multiairport_screen as MS
    OUT.mkdir(parents=True, exist_ok=True)
    MS.CACHE, MS.REPORT = OUT, OUT / "multiairport_screen_rebuilt.json"
    if not MS.REPORT.exists():
        MS.main()
    rebuilt = json.loads(MS.REPORT.read_text())
    models = {name: rebuilt["model_hashes"].get(name) == h for name, h in recorded["model_hashes"].items()}
    checks = {"commit": head, "code_hashes_equal": all(code_ok.values()),
              "source_hash_values_equal": sorted(rebuilt["source_hashes"].values()) == sorted(recorded["source_hashes"].values()),
              "model_hashes_equal": f"{sum(models.values())} of {len(models)}",
              "model_hashes_differing": sorted(n for n, ok in models.items() if not ok),
              "real_models_identical": all(models[f"real_{s}.pkl"] for s in (177, 277, 377)),
              "verdict_equal": rebuilt["verdict"] == recorded["verdict"],
              "rows_equal": (rebuilt["train_rows"], rebuilt["test_partial_rows"]) == (recorded["train_rows"], recorded["test_partial_rows"]),
              "arm_metrics_equal": {a: close(rebuilt["arms"][a], recorded["arms"][a]) for a in recorded["arms"]},
              "seed_metrics_equal": all(close(rebuilt["seeds"][s], recorded["seeds"][s]) for s in recorded["seeds"])}
    checks["faithful"] = (checks["code_hashes_equal"] and checks["real_models_identical"] and checks["source_hash_values_equal"] and checks["verdict_equal"]
                          and checks["rows_equal"] and all(checks["arm_metrics_equal"].values()) and checks["seed_metrics_equal"])
    checks["real_arm"] = {k: rebuilt["arms"]["real"][k] for k in ("rows", "changed", "base_rmse", "rmse", "gain_mse",
                                                                   "gain_without_best100", "by_month")}
    RESULT.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()

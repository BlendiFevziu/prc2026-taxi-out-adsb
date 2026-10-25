"""Sequential, restartable full-ensemble release. Never uploads by itself.

Run with --detach to return immediately; stage logs and state remain in --run.
Re-running the identical command resumes completed component checkpoints.
"""
import argparse
from datetime import datetime, timezone, date
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

STAGES = ("parity", "tune", "baseline", "candidate", "gate", "final", "export")
MODEL_SHA = "81f05970fdef9e8dbe65a9c47bf97a9408fb51f5580bb69b6d8b68fa8ede48c3"
SUBMISSION_SHA = "9b8cb29eb9597c6f7a06d74cbaeff59c812582af1a8ae0d525e404b283ca8773"
PACKAGES = ("numpy", "pandas", "pyarrow", "torch", "lightgbm", "catboost", "joblib", "scikit-learn")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_name(path.name+".partial")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def month_of_file(path):
    match = re.fullmatch(r"training_(2025-\d{2}-01)_(\d{4}-\d{2}-01)\.parquet", Path(path).name)
    if not match:
        raise ValueError(f"Unexpected training filename: {path}")
    start, end = map(date.fromisoformat, match.groups())
    expected = date(2026, 1, 1) if start.month == 12 else date(2025, start.month+1, 1)
    if end != expected:
        raise ValueError(f"Invalid source interval: {path}")
    return start.month


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def verify(manifest, project):
    if manifest["package_versions"] != versions():
        raise RuntimeError("Library versions changed; do not mix checkpoints")
    for relative, digest in manifest["code_sha256"].items():
        if sha(project/relative) != digest:
            raise RuntimeError(f"Run code changed: {relative}")
    for relative, info in manifest["input_files"].items():
        source = project/relative
        stat = source.stat()
        if stat.st_size != info["size"] or stat.st_mtime_ns != info["mtime_ns"]:
            if sha(source) != info["sha256"]:
                raise RuntimeError(f"Run input changed: {relative}")
    for key in ("legacy_model", "legacy_submission"):
        if sha(manifest[key]) != manifest[key+"_sha256"]:
            raise RuntimeError(f"Scored artifact changed: {key}")


def initialize(args):
    run, project = args.run, args.project
    path = run/"run.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest["project"] != str(project) or manifest["jobs"] != args.jobs or
                manifest["legacy_model"] != str(args.legacy_model) or
                manifest["legacy_submission"] != str(args.legacy_submission)):
            raise ValueError("Existing run arguments differ; preserve its recipe")
        verify(manifest, project)
        return manifest
    files = sorted((project/"data").glob("training_*.parquet"))
    months = {month_of_file(p): str(p.relative_to(project)) for p in files}
    assert len(files) == 12 and sorted(months) == list(range(1,13))
    files += [project/"data/ranking.parquet", project/"data/submitting.parquet"]
    geometry = sorted((project/"geometry").glob("*_osm.json"))
    assert len(geometry) >= 10
    files += geometry
    inputs = {str(p.relative_to(project)): {"size": p.stat().st_size,
              "mtime_ns": p.stat().st_mtime_ns, "sha256": sha(p)} for p in files}
    code = [project/"main_model.py", project/"neural_model.py", project/"submission.py"]
    code += sorted(project.glob("features_*.py"))
    code += [project/"research_250"/name for name in (
        "__init__.py", "candidate_features.py", "features_time_units.py", "features_correct_queue.py",
        "release_common.py", "release_worker.py", "release_pipeline.py")]
    manifest = {"created_at": utc(), "project": str(project), "jobs": args.jobs,
        "authorization": "User approved full validation/retraining and one upload after checks on 2026-09-12 Europe/Berlin; no leaderboard tuning or GitHub push.",
        "training_files": {str(k): v for k,v in months.items()}, "input_files": inputs,
        "code_sha256": {str(p.relative_to(project)): sha(p) for p in code},
        "legacy_model": str(args.legacy_model), "legacy_model_sha256": MODEL_SHA,
        "legacy_submission": str(args.legacy_submission), "legacy_submission_sha256": SUBMISSION_SHA,
        "outer_source_months": [1,7], "inner_source_months": [3,9],
        "main_gain_bar_seconds": .3, "package_versions": versions(),
        "python": sys.version, "machine": platform.machine(),
        "policy": "One complete Jan/Jul paired ensemble check, supported by six-pair component evidence. Inner-only candidate LGB length; unchanged blends/gates/floors; final fallback copied from scored v44. Stop if gate fails. No automatic upload in this program."}
    verify(manifest, project)
    write_json(path, manifest)
    return manifest


def run_pipeline(args):
    # Children inherit this descriptor: an orphaned training worker still owns the lock.
    with (args.run/"pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This run already has a controller or worker; no duplicate launched")
        stage = "initializing"
        worker = None
        state = lambda status, **extra: write_json(args.run/"state.json", {
            "status": status, "stage": stage, "controller_pid": os.getpid(),
            "worker_pid": worker.pid if worker else None, "updated_at": utc(), **extra})
        try:
            state("running")
            manifest = initialize(args)
            for stage in STAGES:
                verify(manifest, args.project)
                if (args.run/(stage+"_complete.json")).exists():
                    print(f"[{utc()}] Already complete: {stage}", flush=True)
                    continue
                if stage == "final" and not json.loads((args.run/"gate.json").read_text())["passed"]:
                    raise RuntimeError("Validation failed: final fit prohibited")
                print(f"[{utc()}] Launching {stage} in a fresh process", flush=True)
                worker = subprocess.Popen([sys.executable, "-u", "-B",
                    str(args.project/"research_250/release_worker.py"),
                    "--project", str(args.project), "--run", str(args.run), "--stage", stage],
                    cwd=args.project, stdin=subprocess.DEVNULL, pass_fds=(lock.fileno(),))
                state("running")
                code = worker.wait()
                worker = None
                if code:
                    raise RuntimeError(f"Stage {stage} exited {code}; inspect run.log before resuming")
                if not (args.run/(stage+"_complete.json")).exists():
                    raise RuntimeError(f"Stage {stage} returned without a completion marker")
            state("ready_for_upload")
            print(f"[{utc()}] Validated candidate ready; nothing uploaded by this program", flush=True)
        except BaseException as exc:
            # Do not write 'failed' while a live child is silently still training.
            if worker is not None and worker.poll() is None:
                worker.terminate()
                worker.wait()
            status = "rejected" if stage == "gate" and (args.run/"gate.json").exists() else "failed"
            state(status, error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--legacy-model", type=Path, required=True)
    parser.add_argument("--legacy-submission", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    for name in ("project", "run", "legacy_model", "legacy_submission"):
        setattr(args, name, getattr(args, name).resolve())
    if not 1 <= args.jobs <= 16:
        parser.error("Use 1–16 CPU threads and sequential fits")
    args.run.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve())]+[
            item for item in sys.argv[1:] if item != "--detach"]
        if Path("/usr/bin/caffeinate").exists():
            command = ["/usr/bin/caffeinate", "-i"]+command
        with (args.run/"run.log").open("a") as log:
            process = subprocess.Popen(command, cwd=args.project, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"detached_process_pid": process.pid, "run": str(args.run),
                          "log": str(args.run/"run.log"), "note": "Check state.json to confirm startup"}))
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()

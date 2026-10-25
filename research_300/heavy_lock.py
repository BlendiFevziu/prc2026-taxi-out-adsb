"""Shared lock for heavy jobs on this Mac, so Claude Code and Codex never run two model fits or large extractions at
once. Agreed path: /private/tmp/skenderbeu-heavy-work.lock, taken with fcntl.flock(LOCK_EX).

Measured on this Mac (25 September 2026): flock and lockf locks block each other in all four combinations, so this
lock also excludes a process that uses fcntl.lockf on the same path. The lock is advisory and held only while the
`with` block runs; it is released when the process exits, even after a crash.

Holder note, Codex's convention (seen 25 September): while holding the lock, the holder writes
{"pid": ..., "job": ..., "utc": ...} into the lock file itself. This helper does the same, clears the note when it
releases, and a waiting job prints the current note.

    from research_300.heavy_lock import heavy_work
    with heavy_work("weather screen, seed 177"):
        fit(...)
"""
import contextlib
import fcntl
import json
import os
import time
from datetime import datetime, timezone

LOCK = "/private/tmp/skenderbeu-heavy-work.lock"


def holder(path=LOCK):
    try:
        with open(path) as f:
            note = json.loads(f.read() or "null")
        return note if isinstance(note, dict) else None
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def heavy_work(label, path=LOCK, wait=True, poll_s=30.0, log=print):
    """Hold the shared lock for the duration of the block. wait=False raises BlockingIOError when it is taken."""
    f = open(path, "a+")
    try:
        started = time.time()
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    raise
                log(f"heavy-work lock held by {holder(path)}; waiting ({time.time() - started:.0f} s so far)")
                time.sleep(poll_s)
        f.seek(0)
        f.truncate()
        f.write(json.dumps({"pid": os.getpid(), "job": label, "utc": datetime.now(timezone.utc).isoformat()}))
        f.flush()
        yield
    finally:
        with contextlib.suppress(OSError):
            note = holder(path)
            if note and note.get("pid") == os.getpid():
                f.seek(0)
                f.truncate()
                f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()

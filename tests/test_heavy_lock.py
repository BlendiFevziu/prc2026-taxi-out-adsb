from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_300 import heavy_lock as HL  # noqa: E402

PROBE = r'''
import sys
sys.path.insert(0, sys.argv[2])
from research_300 import heavy_lock as HL
try:
    with HL.heavy_work("probe", path=sys.argv[1], wait=False):
        print("acquired")
except BlockingIOError:
    print("blocked")
'''


class HeavyLockTest(unittest.TestCase):
    def setUp(self):
        self.path = str(Path(tempfile.mkdtemp()) / "lock")
        self.root = str(Path(__file__).resolve().parents[1])

    def probe(self):
        return subprocess.run([sys.executable, "-c", PROBE, self.path, self.root], capture_output=True,
                              text=True).stdout.strip()

    def test_another_process_is_blocked_while_held(self):
        with HL.heavy_work("test", path=self.path, wait=False):
            self.assertEqual(self.probe(), "blocked")

    def test_released_after_the_block(self):
        with HL.heavy_work("test", path=self.path, wait=False):
            pass
        self.assertEqual(self.probe(), "acquired")

    def test_released_after_an_exception(self):
        with self.assertRaises(ValueError):
            with HL.heavy_work("test", path=self.path, wait=False):
                raise ValueError("boom")
        self.assertEqual(self.probe(), "acquired")

    def test_holder_note_is_written_into_the_lock_file_and_cleared(self):
        with HL.heavy_work("fit X", path=self.path, wait=False):
            note = HL.holder(self.path)
            self.assertEqual((note["job"], set(note)), ("fit X", {"pid", "job", "utc"}))
        self.assertIsNone(HL.holder(self.path))

    def test_reads_a_note_written_the_codex_way(self):
        with open(self.path, "w") as f:
            f.write('{"pid": 9019, "job": "NM six-hour archive-boundary control", "utc": "2026-09-25T12:57:02"}')
        self.assertEqual(HL.holder(self.path)["pid"], 9019)

    def test_lockf_holder_blocks_too(self):
        import fcntl
        f = open(self.path, "a+")
        fcntl.lockf(f.fileno(), fcntl.LOCK_EX)
        try:
            self.assertEqual(self.probe(), "blocked")
        finally:
            f.close()


if __name__ == "__main__":
    unittest.main()

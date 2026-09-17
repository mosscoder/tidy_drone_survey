"""0.3.3: the TIDYSURVEY_HEARTBEAT hook (tidysurvey/heartbeat.py)."""
import os
import tempfile
import unittest

from tidysurvey import heartbeat as HB


class Heartbeat(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop(HB.HEARTBEAT_ENV, None)
        HB._last = -float("inf")

    def tearDown(self):
        os.environ.pop(HB.HEARTBEAT_ENV, None)
        if self._env is not None:
            os.environ[HB.HEARTBEAT_ENV] = self._env
        HB._last = -float("inf")

    def test_unset_is_a_noop(self):
        with tempfile.TemporaryDirectory() as d:
            HB.beat()
            self.assertEqual(os.listdir(d), [])

    def test_first_beat_creates_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "hb"); os.environ[HB.HEARTBEAT_ENV] = p
            self.assertFalse(os.path.exists(p))
            HB.beat()
            self.assertTrue(os.path.exists(p))

    def test_throttled_to_one_per_second_then_updates_mtime(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "hb"); os.environ[HB.HEARTBEAT_ENV] = p
            HB.beat(); os.utime(p, (1_000_000, 1_000_000)); m0 = os.stat(p).st_mtime
            HB.beat()                                       # within a second: no touch
            self.assertEqual(os.stat(p).st_mtime, m0)
            HB._last -= 2.0                                 # force the throttle open
            HB.beat()
            self.assertGreater(os.stat(p).st_mtime, m0)

    def test_unwritable_path_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ[HB.HEARTBEAT_ENV] = os.path.join(d, "no", "such", "dir", "hb")
            HB.beat()
            HB._last -= 2.0; HB.beat()


if __name__ == "__main__":
    unittest.main()

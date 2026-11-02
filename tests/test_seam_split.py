"""0.5.0: the seams measured elsewhere. The stitch exports each seam box's two grey images to a work folder, workers
measure them, the stitch renders with their flows. On the same device the result must be the inline stitch byte for
byte; the folder is cleared afterwards; a box is claimed by one worker only, and a dead worker's claim is taken over."""
import json
import os
import tempfile
import threading
import time
import unittest

import numpy as np
import rasterio

import tidysurvey.merge as M
from tidysurvey import seamstore as S
from tests.test_seam_merge_e2e import _inputs, RES

QUIET = lambda m: None


class Split(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["EFOLKI_DEVICE"] = "cpu"                 # one device on both sides: byte-identical is the bar
        cls.tmp = tempfile.mkdtemp(prefix="seam_split_")
        cls.inputs = _inputs(cls.tmp)
        cls.rep_inline = M.seam_merge(cls.inputs, f"{cls.tmp}/inline.tif", res=RES, block=1024,
                                      ownership_out=f"{cls.tmp}/own_inline.tif", log=QUIET)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("EFOLKI_DEVICE", None)

    def _read(self, p):
        with rasterio.open(p) as d:
            return d.read()

    def test_a_local_worker_reproduces_the_inline_stitch_byte_for_byte(self):
        work = f"{self.tmp}/work_local"; lines = []
        rep = M.seam_merge(self.inputs, f"{self.tmp}/split.tif", res=RES, block=1024, ownership_out=f"{self.tmp}/own_split.tif",
                           seams=work, seams_local_workers=1, log=lines.append)
        np.testing.assert_array_equal(self._read(f"{self.tmp}/split.tif"), self._read(f"{self.tmp}/inline.tif"))
        np.testing.assert_array_equal(self._read(f"{self.tmp}/own_split.tif"), self._read(f"{self.tmp}/own_inline.tif"))
        self.assertEqual(rep["seams"], self.rep_inline["seams"]); self.assertEqual(rep["blocks"], self.rep_inline["blocks"])
        n = sum(self.rep_inline["seams_measured_on"].values())
        self.assertGreater(n, 0); self.assertEqual(rep["seams_measured_on"], {"cpu": n})          # every box from the folder
        self.assertTrue(any(l.startswith(f"    seams: exporting boxes to {work}") for l in lines))
        self.assertEqual([p for p in os.listdir(work) if not p.startswith(".")], [])              # the scratch is cleared

    def test_an_outside_worker_does_the_measuring(self):
        """No local worker: a worker in another thread (standing in for a GPU elsewhere) measures every box."""
        work = f"{self.tmp}/work_outside"; res = {}
        t = threading.Thread(target=lambda: res.update(S.measure(work, device="cpu", poll_s=0.2, log=QUIET)))
        t.start()
        M.seam_merge(self.inputs, f"{self.tmp}/outside.tif", res=RES, block=1024, seams=work, seams_local_workers=0, log=QUIET)
        t.join(60)
        np.testing.assert_array_equal(self._read(f"{self.tmp}/outside.tif"), self._read(f"{self.tmp}/inline.tif"))
        self.assertEqual(res["boxes"], sum(self.rep_inline["seams_measured_on"].values()))


class Claims(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="seam_claims_")
        self.st = S.SeamStore(self.root)

    def test_one_claim_wins(self):
        self.assertTrue(self.st.claim("k", "w1")); self.assertFalse(self.st.claim("k", "w2"))
        self.st.release("k"); self.assertTrue(self.st.claim("k", "w2"))

    def test_a_stale_claim_is_taken_over_once(self):
        self.assertTrue(self.st.claim("k", "dead"))
        old = json.loads((S.Path(self.root) / "claims" / "k").read_text()); old["t"] -= 3600
        (S.Path(self.root) / "claims" / "k").write_text(json.dumps(old))
        wins = [self.st.claim("k", f"w{i}", lease_s=600) for i in range(3)]
        self.assertEqual(wins.count(True), 1)

    def test_box_and_flow_round_trip(self):
        ga = np.random.default_rng(1).random((40, 50), np.float32); gb = ga[::-1].copy()
        self.st.put_box("b", ga, gb, dict(box=[0, 40, 0, 50], keep=[5, 35, 10, 40]))
        got = self.st.get_box("b"); np.testing.assert_array_equal(got[0], ga); self.assertEqual(got[2]["keep"], [5, 35, 10, 40])
        self.st.put_flow("b", ga[5:35, 10:40], gb[5:35, 10:40], dict(box=[0, 40, 0, 50], device="cpu"))
        u, v, meta = self.st.get_flow("b"); self.assertEqual(u.shape, (30, 30)); self.assertEqual(meta["device"], "cpu")
        self.assertEqual(self.st.flow_keys(), {"b"}); self.assertEqual(self.st.box_keys(), {"b"})
        self.st.clear(); self.assertEqual(self.st.flow_keys() | self.st.box_keys(), set())

    def test_a_worker_leaves_when_the_export_is_complete_and_measured(self):
        self.st.finish_export([], {}); t = time.time()
        res = S.measure(self.root, device="cpu", poll_s=0.1, log=QUIET)
        self.assertEqual(res["boxes"], 0); self.assertLess(time.time() - t, 5)


if __name__ == "__main__":
    unittest.main()

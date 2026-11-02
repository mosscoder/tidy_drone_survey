"""A map's edge moves as one piece with the ground beside it (0.4.2).

eFOLKI's shift is a measurement only where its largest window, 32 px, lies on ground in both maps: the intersection
of the two data masks with a negative buffer of that radius. Beyond the edge the solve matched fill, not ground, and
placed the edge about 4 px wrong (the audit's edge_known_warp.py, 2026-11-01). Every other pixel takes the nearest
measured shift.

Here eFOLKI is faked. It returns the known shift where the reference carries real ground and random junk of up to
200 px where the reference is painted flat, which is everywhere outside the mission's data and a 30 px strip inside
its jagged edge. With the junk replaced by the nearest measured shift, the registered map must be byte for byte the
one the known shift alone gives."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

import tidysurvey.efolki as E
import tidysurvey.registration as R
from test_register_efolki import RES, _write, texture
from test_register_sharpness import small_blocks

U, V, FLAT = 1.7, -0.9, 3.0


def register(mission, anchor, out, junk):
    rng = np.random.default_rng(0)
    def fake(reference, moving, dev=None, **k):
        u = np.full(reference.shape, U, np.float32); v = np.full(reference.shape, V, np.float32)
        if junk:
            flat = np.abs(reference - FLAT) < 0.01
            u[flat] = rng.uniform(-200, 200, int(flat.sum())); v[flat] = rng.uniform(-200, 200, int(flat.sum()))
        return u, v
    with mock.patch.object(E, "flow", fake):
        small_blocks(lambda: R.register_survey_dense(mission, anchor, out, resolution_m=RES, log=lambda m: None))
    with rasterio.open(out) as o:
        return o.read(), o.transform


class EdgeMovesWithTheGround(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = Path(tempfile.mkdtemp(prefix="reg_edge_"))
        h, w = 1000, 1200
        tr = from_origin(500000.0, 5170000.0, RES, RES)
        g = texture(h, w, seed=41)
        yy, xx = np.mgrid[0:h, 0:w]
        rng = np.random.default_rng(7)
        corner = xx + (h - 1 - yy) < 700 + 5 * np.sin(yy / 7.0) + rng.integers(-2, 3, size=(h, 1))   # a jagged corner
        alpha = ~corner
        m = np.stack([g * 200 + 20, g * 180 + 30, (1 - g) * 150 + 40]).astype(np.uint8)
        m[:, ~alpha] = 0
        mission = _write(tmp / "m.tif", np.concatenate([m, (alpha * 255).astype(np.uint8)[None]]), tr)
        a = np.stack([g * 200 + 20] * 3).astype(np.uint8)
        strip = distance_transform_edt(alpha) <= 30                   # outside the mission, and 30 px inside its edge
        a[:, strip] = int(FLAT)                                      # the reference is flat there: the fake returns junk
        anchor = _write(tmp / "a.tif", np.concatenate([a, np.full((1, h, w), 255, np.uint8)]), tr)
        cls.clean, T1 = register(mission, anchor, str(tmp / "clean.tif"), junk=False)
        cls.junk, T2 = register(mission, anchor, str(tmp / "junk.tif"), junk=True)
        assert T1 == T2
        cls.alpha = alpha

    def test_junk_at_the_edge_changes_nothing(self):
        d = int((self.junk != self.clean).sum())
        self.assertEqual(d, 0, f"{d} values differ from the known-shift map")

    def test_the_edge_is_there(self):
        self.assertGreater(int((self.clean[3] == 255).sum()), 500000)
        self.assertGreater(int((self.clean[3] == 0).sum()), 100000)


if __name__ == "__main__":
    unittest.main()

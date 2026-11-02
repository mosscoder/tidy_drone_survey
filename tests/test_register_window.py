"""A shift eFOLKI cannot have measured must not reach the registered map or size the warp's read (0.4.2).

On 2026-11-01 the register job died on SIGSEGV five times out of five on the 2025 spring mission 00 visible map.
Where the mission has no data, eFOLKI's shift means nothing, and there it reached ±30000 px. The warp sized its read
of the native pixels from every pixel of the block, so one read passed cv2.remap's 32767 px limit, and a few such
pixels landed on the mission and pulled in ground from up to 22 m away. Here eFOLKI is faked to a known shift that
turns wild over a patch of ground where the mission has no data: far off the map, or just past eFOLKI's reach and
onto the mission. Either way the native reads are no larger than with the known shift alone, the patch comes back
transparent and empty, and every other pixel is the same as with the known shift alone, byte for byte."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy.ndimage import binary_dilation

import tidysurvey.efolki as E
import tidysurvey.registration as R
from test_register_efolki import CRS, RES, _write, texture
from test_register_sharpness import small_blocks

U, V = 1.7, -0.9                                             # the known shift, reference pixels
WILD = {"far": (30000.0, -30000.0),                          # off the map, as on spring mission 00
        "onto the mission": (-1.5 * E.REACH_PX, -1.5 * E.REACH_PX)}   # just past the reach, landing on mission data


def run(mission, anchor, out, wild=None):
    """Register with eFOLKI replaced by the known shift, `wild` where the reference is the flat patch; return the
    largest native read the cubic warp was given."""
    def fake(reference, moving, dev=None, **k):
        u = np.full(reference.shape, U, np.float32); v = np.full(reference.shape, V, np.float32)
        if wild:
            flat = reference < 10                           # the patch, grey 2, where the mission has no data
            u[flat], v[flat] = wild
        return u, v
    reads, remap = [], cv2.remap
    def recorder(src, m1, m2, interp, *a, **kw):
        if interp == cv2.INTER_CUBIC:
            reads.append(max(src.shape[:2]))
        return remap(src, m1, m2, interp, *a, **kw)
    with mock.patch.object(E, "flow", fake), mock.patch.object(cv2, "remap", recorder):
        small_blocks(lambda: R.register_survey_dense(mission, anchor, out, resolution_m=RES, log=lambda m: None))
    with rasterio.open(out) as o:
        return max(reads), o.read(), o.transform, o.height, o.width


class ShiftPastTheReach(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = Path(tempfile.mkdtemp(prefix="reg_window_"))
        atr = from_origin(500000.0, 5170000.0, RES, RES)
        g = texture(1000, 1200, seed=21)
        a = np.stack([g * 200 + 20, g * 180 + 30, (1 - g) * 150 + 40, np.full_like(g, 255)]).astype(np.uint8)
        a[:3, 600:, 700:] = np.array([1, 2, 3], np.uint8)[:, None, None]      # a flat patch, grey 2
        cls.anchor = _write(tmp / "a.tif", a, atr)
        mres = 0.04                                                            # finer than the grid, off its lattice
        mt = from_origin(500000.0 + 0.0137, 5170000.0 - 0.0291, mres, mres)
        h, w = 1150, 1400
        t = texture(h, w, seed=22)
        m = np.stack([t * 200 + 20, t * 180 + 30, (1 - t) * 150 + 40, np.full_like(t, 255)]).astype(np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        X = mt.c + (xx + 0.5) * mres; Y = mt.f - (yy + 0.5) * mres
        m[:, (Y < atr.f - 600 * RES) & (X > atr.c + 700 * RES)] = 0            # no data under the patch
        cls.mission = _write(tmp / "m.tif", m, mt)
        cls.read_clean, cls.clean, T, oh, ow = run(cls.mission, cls.anchor, str(tmp / "clean.tif"))
        cls.wild = {k: run(cls.mission, cls.anchor, str(tmp / f"wild_{i}.tif"), w)[:2] for i, (k, w) in enumerate(WILD.items())}
        with rasterio.open(cls.anchor) as s, WarpedVRT(s, crs=CRS, transform=T, width=ow, height=oh,
                                                       resampling=Resampling.nearest) as v:
            cls.patch = v.read(2) == 2
        cls.near = binary_dilation(cls.patch, iterations=12)    # where the field blends the wild shift into its neighbours

    def test_reads_no_larger_than_the_known_shift(self):
        for k, (read, _) in self.wild.items():
            with self.subTest(k):
                self.assertLessEqual(read, self.read_clean, f"a native read of {read} px; with the known shift alone {self.read_clean}")

    def test_the_patch_transparent_and_empty(self):
        p = self.patch & ~binary_dilation(~self.patch, iterations=12)
        self.assertGreater(int(p.sum()), 1000)
        for k, (_, img) in self.wild.items():
            with self.subTest(k):
                self.assertTrue(np.all(img[:, p] == 0), f"{int((img[3][p] > 0).sum())} patch pixels kept")

    def test_every_other_pixel_unchanged(self):
        far = ~self.near
        self.assertGreater(int((self.clean[3][far] == 255).sum()), 100000)
        for k, (_, img) in self.wild.items():
            with self.subTest(k):
                diff = int((img[:, far] != self.clean[:, far]).sum())
                self.assertEqual(diff, 0, f"{diff} values changed away from the patch")


if __name__ == "__main__":
    unittest.main()

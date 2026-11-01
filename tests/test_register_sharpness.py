"""The registered map is one cubic resample of the mission's own pixels, never two.

On 2026-07-17 the registered maps were found softened by stacked resamples (01_merge/
registered_sharpness_audit: a second bilinear resample lost 26 % of fine detail). The fix, kept in 0.4.0, is
one cubic warp from the native pixels onto a grid snapped to the global lattice. These tests hold it there.

1. A map registered to itself on its own lattice comes back unchanged, byte for byte: no hidden resample.
2. With eFOLKI replaced by a known constant shift, so only the warp is tested, a mission finer than the grid
   and off its lattice comes back as one cubic sample of its own pixels at the shifted position, rounded and
   clipped. It keeps more fine detail than the two bilinear resamples July removed."""
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

import tidysurvey.efolki as E
import tidysurvey.registration as R
from test_register_efolki import CRS, RES, _write, texture


def fine_detail(x):
    """Power in the upper half of the frequencies the grid can show."""
    F = np.abs(np.fft.fft2(x - x.mean())) ** 2
    r = np.hypot(np.fft.fftfreq(x.shape[0])[:, None], np.fft.fftfreq(x.shape[1])[None, :])
    return float(F[(r > 0.25) & (r < 0.5)].sum())


def small_blocks(fn):
    core, pad = R._CORE, R._PAD
    R._CORE, R._PAD = 400, 160
    try:
        return fn()
    finally:
        R._CORE, R._PAD = core, pad


class SelfRegistration(unittest.TestCase):
    def test_unchanged_byte_for_byte(self):
        tmp = Path(tempfile.mkdtemp(prefix="reg_sharp_self_"))
        g = texture(900, 1100, seed=11)
        img = np.stack([g * 200 + 20, g * 180 + 30, (1 - g) * 150 + 40, np.full_like(g, 255)]).astype(np.uint8)
        tr = from_origin(500000.0, 5170000.0, RES, RES)
        anchor, mission = _write(tmp / "a.tif", img, tr), _write(tmp / "m.tif", img, tr)
        out = str(tmp / "o.tif")
        s = small_blocks(lambda: R.register_survey_dense(mission, anchor, out, log=lambda m: None))
        self.assertGreater(s["blocks"]["measured"], 4)
        with rasterio.open(out) as o:
            self.assertEqual((o.transform.c, o.transform.f, o.res[0]), (tr.c, tr.f, RES))
            got = o.read([1, 2, 3])[:, :900, :1100]
        self.assertTrue(np.array_equal(got, img[:3]), f"{int((got != img[:3]).sum())} pixels changed")


class OneCubicResample(unittest.TestCase):
    U, V = 1.7, -0.9                                   # the faked shift, in reference pixels

    @classmethod
    def setUpClass(cls):
        tmp = Path(tempfile.mkdtemp(prefix="reg_sharp_cubic_"))
        g = texture(1000, 1200, seed=12)
        cls.anchor = _write(tmp / "a.tif", np.stack([g * 200 + 20] * 3 + [np.full_like(g, 255)]).astype(np.uint8),
                            from_origin(500000.0, 5170000.0, RES, RES))
        mres = 0.04                                    # finer than the 5 cm grid, origin off its lattice
        cls.mt = from_origin(500000.0 + 0.0137, 5170000.0 - 0.0291, mres, mres)
        h, w = 1150, 1400
        yy, xx = np.mgrid[0:h, 0:w]
        checker = ((xx // 3 + yy // 3) % 2) * 200.0       # hard edges, so the cubic overshoots and must be clipped
        t = texture(h, w, seed=13) * 55
        cls.native = np.stack([checker + t, checker * 0.6 + t + 20, 255 - checker - t]).clip(0, 255).astype(np.uint8)
        mission = _write(tmp / "m.tif", np.concatenate([cls.native, np.full((1, h, w), 255, np.uint8)]), cls.mt)
        cls.out = str(tmp / "o.tif")
        fake = lambda reference, moving, dev=None, **k: (np.full(reference.shape, cls.U, np.float32),
                                                          np.full(reference.shape, cls.V, np.float32))
        with mock.patch.object(E, "flow", fake):
            small_blocks(lambda: R.register_survey_dense(mission, cls.anchor, cls.out, resolution_m=RES,
                                                         log=lambda m: None))
        with rasterio.open(cls.out) as o:
            cls.got = o.read([1, 2, 3]).astype(np.float32); cls.alpha = o.read(4); cls.T = o.transform
        # the same sample, computed here: each output pixel centre moved by the shift, then into native pixels
        oh, ow = cls.alpha.shape
        r, c = np.mgrid[0:oh, 0:ow].astype(np.float64)
        X = cls.T.c + (c + 0.5 + cls.U) * RES; Y = cls.T.f - (r + 0.5 + cls.V) * RES
        nx = ((X - cls.mt.c) / cls.mt.a - 0.5).astype(np.float32); ny = ((Y - cls.mt.f) / cls.mt.e - 0.5).astype(np.float32)
        cls.cubic = np.stack([cv2.remap(cls.native[b].astype(np.float32), nx, ny, cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0) for b in range(3)])
        cls.inner = np.zeros_like(cls.alpha, bool); cls.inner[12:-12, 12:-12] = True
        cls.inner &= cls.alpha == 255
        # July's two resamples: bilinear onto the grid, then a bilinear shift
        with rasterio.open(mission) as m, WarpedVRT(m, crs=CRS, transform=cls.T, width=ow, height=oh,
                                                   resampling=Resampling.bilinear) as v:
            grid = v.read([1, 2, 3]).astype(np.float32)
        gy, gx = np.mgrid[0:oh, 0:ow].astype(np.float32)
        cls.two = np.stack([cv2.remap(grid[b], gx + cls.U, gy + cls.V, cv2.INTER_LINEAR) for b in range(3)])

    def test_on_the_lattice(self):
        self.assertAlmostEqual(self.T.a, RES)
        self.assertAlmostEqual(self.T.c / RES, round(self.T.c / RES), places=6)
        self.assertAlmostEqual(self.T.f / RES, round(self.T.f / RES), places=6)

    def test_one_cubic_sample_of_the_native_pixels(self):
        want = np.rint(self.cubic.clip(0, 255))
        d = np.abs(self.got - want)[:, self.inner]
        self.assertEqual(float(np.median(d)), 0.0)
        self.assertLessEqual(float(np.percentile(d, 99.9)), 2.0)

    def test_rounded_not_truncated(self):
        bias = float(np.mean((self.got - self.cubic.clip(0, 255))[:, self.inner]))
        self.assertLess(abs(bias), 0.05, f"mean error {bias:.3f} DN")

    def test_overshoot_clipped(self):
        over = (self.cubic > 255.5)[:, self.inner]
        self.assertGreater(int(over.sum()), 100, "the pattern did not make the cubic overshoot")
        self.assertTrue(np.all(self.got[:, self.inner][over] == 255))

    def test_sharper_than_two_resamples(self):
        sl = np.s_[20:-20, 20:-20]
        ratio = fine_detail(self.got[1][sl]) / fine_detail(self.two[1][sl])
        self.assertGreater(ratio, 1.2, f"fine detail {ratio:.2f} x the two-resample rendering")


if __name__ == "__main__":
    unittest.main()

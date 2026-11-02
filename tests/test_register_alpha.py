"""The registered map has data exactly where the map it copies from has data (0.4.2).

DroneDeploy marks its map's data in an alpha band, pixel by pixel. Until 0.4.2 registration decided the output's
data on squares of 8 x 8 output pixels (25.6 cm on a visible map, 44 cm on a multispectral one): a square more than
half data was all data, so its empty pixels came out as black data, and a square less than half data was dropped
with its real pixels. Now each output pixel has data only where the native pixel it samples has data.

With eFOLKI faked to no shift and the mission on the output grid, every output pixel is one native pixel, so the
registered map must give back the mission's own mask and pixels: a jagged edge, a hole, a sliver two pixels wide,
and real ground that is pure black, which stays data."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import from_origin

import tidysurvey.efolki as E
import tidysurvey.registration as R
from test_register_efolki import RES, _write, texture
from test_register_sharpness import small_blocks


class DataExactlyWhereTheMissionHasData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = Path(tempfile.mkdtemp(prefix="reg_alpha_"))
        h, w = 900, 1100
        tr = from_origin(500000.0, 5170000.0, RES, RES)                 # on the output lattice
        g = texture(h, w, seed=31)
        rgb = np.stack([g * 200 + 20, g * 180 + 30, (1 - g) * 150 + 40]).astype(np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        rng = np.random.default_rng(5)
        edge = 700 + 6 * np.sin(yy / 9.0) + rng.integers(-3, 4, size=(h, 1))   # a jagged edge, ragged per row
        alpha = xx < edge
        alpha &= ~((yy - 300) ** 2 + (xx - 300) ** 2 < 60 ** 2)               # a hole
        alpha |= (xx >= 899) & (xx <= 900) & (yy > 100) & (yy < 800)            # a sliver two pixels wide, apart
        alpha[600:640, 200:260] = True; rgb[:, 600:640, 200:260] = 0           # real ground, pure black
        rgb[:, ~alpha] = 0                                                     # DroneDeploy's empty pixels: all zeros
        cls.alpha, cls.rgb = alpha, rgb
        mission = _write(tmp / "m.tif", np.concatenate([rgb, (alpha * 255).astype(np.uint8)[None]]), tr)
        anchor = _write(tmp / "a.tif", np.stack([g * 200 + 20] * 3 + [np.full_like(g, 255)]).astype(np.uint8), tr)
        cls.out = str(tmp / "o.tif")
        zero = lambda reference, moving, dev=None, **k: (np.zeros(reference.shape, np.float32),) * 2
        with mock.patch.object(E, "flow", zero):
            small_blocks(lambda: R.register_survey_dense(mission, anchor, cls.out, resolution_m=RES, log=lambda m: None))
        with rasterio.open(cls.out) as o:                                     # the output covers the data's extent
            c0, r0 = round((o.transform.c - tr.c) / RES), round((tr.f - o.transform.f) / RES)
            assert o.res[0] == RES and c0 >= 0 and r0 >= 0
            cls.got = np.zeros((4, h, w), np.uint8)
            img = o.read()[:, :h - r0, :w - c0]
            cls.got[:, r0:r0 + img.shape[1], c0:c0 + img.shape[2]] = img

    def test_mask_reproduced_pixel_for_pixel(self):
        got = self.got[3] == 255
        self.assertEqual(int((got != self.alpha).sum()), 0,
                         f"{int((self.alpha & ~got).sum())} data pixels dropped, {int((got & ~self.alpha).sum())} empty pixels made data")

    def test_the_sliver_kept(self):
        self.assertTrue(np.all(self.got[3][101:800, 899:901] == 255))

    def test_pixels_copied_exactly(self):
        self.assertTrue(np.array_equal(self.got[:3][:, self.alpha], self.rgb[:, self.alpha]))

    def test_black_ground_stays_data(self):
        self.assertTrue(np.all(self.got[3][600:640, 200:260] == 255))
        self.assertTrue(np.all(self.got[:3, 600:640, 200:260] == 0))


if __name__ == "__main__":
    unittest.main()

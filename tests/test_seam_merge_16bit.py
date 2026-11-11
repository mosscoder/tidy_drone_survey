"""The seam merge keeps its inputs' number type.

Until this the mosaic was always written 8-bit, which is what DroneDeploy's maps are. The ODM fork's multispectral
maps are 16-bit (four bands and an alpha band named "alpha", values up to about 45,000), and an 8-bit mosaic wrapped
them. Two 16-bit flights of one textured ground, the second with its content moved two pixels east: the mosaic must
be 16-bit, every pixel outside the seam band must be its owner's pixel exactly, the blend inside the band must stay
within the inputs' range, and the seam must still be measured at its known offset. 8-bit inputs are covered by
test_seam_merge_e2e.py, which this change must leave passing byte for byte."""
import tempfile
import unittest

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

import tidysurvey.merge as M
import tidysurvey.validate as V

W, H, RES = 1800, 1200, 0.055
CRS = "EPSG:32612"
BANDS = ["Red", "Green", "NIR", "RedEdge"]


def _inputs(tmp, dtype="uint16", top=45000.0):
    rng = np.random.default_rng(1)
    g = sum(gaussian_filter(rng.normal(size=(H, W)), s) / gaussian_filter(rng.normal(size=(H, W)), s).std() for s in (1.5, 4.0, 12.0))
    g = ((g - g.min()) / (g.max() - g.min())).astype(np.float32)
    ground = np.stack([g * 0.5 + 0.1, g * 0.4 + 0.15, (1 - g) * 0.8 + 0.2, (1 - g) * 0.6 + 0.2]) * top
    cols = np.arange(W)[None, :].repeat(H, 0)
    masks = {"06": cols < 1100, "07": cols > 700}
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = []
    for name, mask in masks.items():
        dx = 2.0 if name == "07" else 0.0
        img = np.stack([cv2.remap(b, xx - dx, yy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) for b in ground])
        img = (np.clip(img, 1, top) * mask).astype(dtype)
        p = f"{tmp}/{name}_{dtype}.tif"
        with rasterio.open(p, "w", driver="GTiff", width=W, height=H, count=5, dtype=dtype, crs=CRS,
                           transform=from_origin(0, H * RES, RES, RES), tiled=True, compress="deflate") as d:   # on the merge's lattice: no redraw
            d.write(img, [1, 2, 3, 4]); d.write(np.where(mask, 255, 0).astype(dtype), 5)
            for i, b in enumerate(BANDS + ["alpha"], 1):               # the alpha is named, not tagged, as the fork writes it
                d.set_band_description(i, b)
        out.append((name, p))
    return out


class SixteenBit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="seam_16bit_")
        cls.inputs = _inputs(cls.tmp)
        cls.out = f"{cls.tmp}/mosaic.tif"
        cls.logs = []
        cls.rep = M.seam_merge(cls.inputs, cls.out, res=RES, block=1024, band_names=BANDS, ownership_out=f"{cls.tmp}/own.tif", log=cls.logs.append)

    def test_the_mosaic_is_16_bit_and_not_wrapped(self):
        with rasterio.open(self.out) as d, rasterio.open(self.inputs[0][1]) as a:
            self.assertEqual((d.count, set(d.dtypes)), (5, {"uint16"}))
            mosaic, src = d.read(), a.read()
        self.assertTrue(any("uint16 inputs" in m for m in self.logs))
        self.assertLessEqual(set(np.unique(mosaic[4]).tolist()), {0, 255})                  # the alpha band stays 0/255
        self.assertGreater((mosaic[4] == 255).mean(), 0.99)
        dr, dc = mosaic.shape[1] - src.shape[1], mosaic.shape[2] - src.shape[2]             # the mosaic's grid may start a pixel out
        self.assertTrue(any(np.array_equal(mosaic[:4, r:r + 1000, c:c + 600], src[:4, :1000, :600])      # far from the seam: the
                            for r in range(dr + 1) for c in range(dc + 1)))                 # owner's pixels exactly
        self.assertGreater(int(mosaic[:4].max()), 40000)                                    # not clipped at 255, not wrapped at 32,767
        self.assertLessEqual(int(mosaic[:4].max()), 45000)
        band = mosaic[:4, 100:1000, 860:940].astype(np.float64)                             # across the seam, around column 900
        ref = src[:4, 100:1000, 860:940].astype(np.float64)
        self.assertLess(abs(band.mean() - ref.mean()) / ref.mean(), 0.02)                   # a blend of two near-equal images, not noise

    def test_interiors_unchanged_and_the_seam_measured(self):
        V.assert_interiors_unchanged(self.inputs, self.out, f"{self.tmp}/own.tif", band_width_m=1.0, block_px=1024, log=lambda m: None).assert_ok()
        self.assertEqual(self.rep["n_seams"], 1)
        self.assertAlmostEqual(self.rep["seams"][0]["med_offset_cm"], 2.0 * RES * 100, delta=1.5)

    def test_inputs_of_two_number_types_are_refused(self):
        eight = _inputs(self.tmp, dtype="uint8", top=250.0)
        with self.assertRaises(ValueError) as e:
            M.seam_merge([self.inputs[0], eight[1]], f"{self.tmp}/mixed.tif", res=RES, band_names=BANDS, log=lambda m: None)
        self.assertIn("all be 8-bit or all be 16-bit", str(e.exception))


if __name__ == "__main__":
    unittest.main()

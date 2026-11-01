"""0.4.1: the season stitch reads each source once, cubic, and the seam band warps cubic.

Two flights of one ground meet in a north-south seam. m0 is on the merge's lattice. m1 is the same ground at
the same pixel size with its corner off the lattice by 0.37 and 0.61 px, the way an unregistered summer map
arrives. Away from the band, m0 comes through byte for byte and m1 as one cubic read of its own pixels, the
read validate.assert_interiors_unchanged repeats, with more fine detail than the `average` read it had until
0.4.1. The band warp is exact where the taper is zero and cubic where it is not."""
import tempfile
import unittest

import cv2
import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

import tidysurvey.fields as F
import tidysurvey.merge as M
import tidysurvey.validate as V
from test_register_efolki import texture
from test_register_sharpness import fine_detail

W, H, RES = 2000, 1400, 0.05
CRS = "EPSG:32611"
OFF = (0.37, 0.61)                                      # m1's corner east and south of the lattice, in pixels


def _write(path, rgb, mask, tr):
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=4, dtype="uint8", crs=CRS,
                       transform=tr, tiled=True, compress="zstd") as d:
        d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        d.write((np.clip(rgb, 1, 255) * mask).astype(np.uint8), [1, 2, 3])
        d.write(np.where(mask, 255, 0).astype(np.uint8), 4)
    return path


class OffLatticeInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.mkdtemp(prefix="stitch_sharp_")
        g = texture(H + 8, W + 8, seed=21).astype(np.float32)
        ground = np.stack([g * 200 + 20, g * 170 + 40, (1 - g) * 150 + 50])
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cols = np.arange(W)[None, :].repeat(H, 0)
        cls.m0 = ground[:, :H, :W]
        m1 = np.stack([cv2.remap(b, xx + OFF[0], yy + OFF[1], cv2.INTER_CUBIC) for b in ground])
        cls.inputs = [("m0", _write(f"{tmp}/m0.tif", cls.m0, cols < 1200, from_origin(0, H * RES, RES, RES))),
                      ("m1", _write(f"{tmp}/m1.tif", m1, cols > 800,
                                    from_origin(OFF[0] * RES, (H - OFF[1]) * RES, RES, RES)))]
        cls.out, cls.own = f"{tmp}/out.tif", f"{tmp}/own.tif"
        M.seam_merge(cls.inputs, cls.out, res=RES, block=1024, ownership_out=cls.own, log=lambda m: None)
        with rasterio.open(cls.out) as d:
            cls.T, cls.mosaic = d.transform, d.read([1, 2, 3]).astype(np.float32)
        oh, ow = cls.mosaic.shape[1:]
        cls.read = {}
        for rs in (Resampling.cubic, Resampling.average):
            with rasterio.open(cls.inputs[1][1]) as s, WarpedVRT(s, crs=CRS, transform=cls.T, width=ow, height=oh,
                                                                 resampling=rs) as v:
                cls.read[rs.name] = v.read([1, 2, 3]).astype(np.float32)
        cls.east = np.s_[:, 60:H - 60, 1500:W - 60]     # m1's own ground, clear of the band

    def test_lattice_input_copied_exactly(self):
        self.assertEqual((self.T.c, self.T.f), (0.0, H * RES))
        west = np.s_[:, 60:H - 60, 60:600]            # the mosaic is a row taller: m1 reaches 0.61 px below
        self.assertTrue(np.array_equal(self.mosaic[west], np.clip(self.m0, 1, 255).astype(np.uint8)[west]))

    def test_off_lattice_input_is_one_cubic_read(self):
        d = np.abs(self.mosaic[self.east] - self.read["cubic"][self.east])
        self.assertEqual(float(np.median(d)), 0.0)
        self.assertLessEqual(float(d.max()), 1.0)
        V.assert_interiors_unchanged(self.inputs, self.out, self.own, band_width_m=1.0, block_px=1024,
                                     log=lambda m: None).assert_ok()

    def test_sharper_than_the_average_read(self):
        ratio = fine_detail(self.mosaic[1][self.east[1:]]) / fine_detail(self.read["average"][1][self.east[1:]])
        self.assertGreater(ratio, 1.15, f"fine detail {ratio:.2f} x the average read")   # 1.26 here, 1.74 on a real map


class BandWarp(unittest.TestCase):
    def test_exact_outside_cubic_inside(self):
        g = (texture(200, 240, seed=22) * 230 + 10).astype(np.float32)
        spec = np.stack([g, 255 - g])
        taper = np.zeros((200, 240), np.float32); taper[:, 120:] = 1.0
        fx, fy = np.full_like(taper, 0.6), np.full_like(taper, -0.3)
        reg, vreg = F.apply_field_confined(spec, np.ones((200, 240), bool), fx, fy, taper)
        self.assertTrue(np.array_equal(reg[:, :, :118], spec[:, :, :118]))
        yy, xx = np.mgrid[0:200, 0:240].astype(np.float32)
        want = np.stack([cv2.remap(b, xx + 0.6, yy - 0.3, cv2.INTER_CUBIC, borderValue=0) for b in spec])
        self.assertTrue(np.allclose(reg[:, 4:-4, 124:-4], want[:, 4:-4, 124:-4], atol=1e-3))


if __name__ == "__main__":
    unittest.main()

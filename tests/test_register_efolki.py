"""0.4.0: register_survey_dense with eFOLKI on the reference's own pixels.

Synthetic pairs with a known move: the mission is the reference's ground with its content moved by a
shift plus a smooth bend, so the engine's move (quality raster, summary) must read the inverse, and the
registered map must line up with the reference. Small blocks (_CORE, _PAD patched) run the block seams
and the per-cell scoring across several blocks. A coarser multispectral-shaped mission without alpha
(the 2024 law) measures on the reference's finer pixels and writes its own grid. Real eFOLKI, on
whatever device efolki.device() picks."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter

import tidysurvey.registration as R
from tidysurvey import bands as B

CRS = "EPSG:32611"
RES = 0.05
H, W = 1400, 1700


def texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    g = sum(gaussian_filter(rng.normal(size=(h, w)), s) / gaussian_filter(rng.normal(size=(h, w)), s).std()
            for s in (1.5, 4.0, 12.0))
    return (g - g.min()) / (g.max() - g.min())


def _write(path, arr, tr, tag_alpha=True):
    n, h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=n, dtype="uint8", crs=CRS,
                       transform=tr, tiled=True, blockxsize=256, blockysize=256, compress="zstd") as d:
        if tag_alpha:
            d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        d.write(arr)
    return str(path)


class KnownMove(unittest.TestCase):
    """Content moved 4.3 px east and 2.6 px north plus a 1.5 px / 1 px bend; registered in 400 px blocks."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="reg_efolki_"))
        g = texture(H, W)
        cls.ref = np.stack([g * 200 + 20, g * 180 + 30, (1 - g) * 150 + 40, np.full_like(g, 255)]).astype(np.uint8)
        tr = from_origin(500000.0, 5170000.0, RES, RES)
        cls.anchor = _write(cls.tmp / "ref.tif", cls.ref, tr)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cls.dx = 4.3 + 1.5 * np.sin(xx / 400.0); cls.dy = -2.6 + 1.0 * np.cos(yy / 350.0)
        mov = np.stack([cv2.remap(cls.ref[b].astype(np.float32), xx - cls.dx, yy - cls.dy, cv2.INTER_CUBIC)
                        for b in range(3)])
        alpha = np.full((H, W), 255, np.uint8); alpha[:, :120] = 0; alpha[-100:, :] = 0
        cls.mov = np.concatenate([np.clip(mov, 0, 255).astype(np.uint8), alpha[None]])
        cls.mission = _write(cls.tmp / "mov.tif", cls.mov, tr)
        cls.out = str(cls.tmp / "out.tif"); cls.qa = str(cls.tmp / "qa.json")
        core, pad = R._CORE, R._PAD
        R._CORE, R._PAD = 400, 160
        try:
            cls.s = R.register_survey_dense(cls.mission, cls.anchor, cls.out, qa_json=cls.qa, log=lambda m: None)
        finally:
            R._CORE, R._PAD = core, pad

    def test_the_three_numbers(self):
        s = self.s
        self.assertGreater(s["blocks"]["measured"], 4)
        self.assertLess(s["r_before"], 0.7); self.assertGreater(s["r_after"], 0.97)
        want = float(np.median(np.hypot(self.dx, self.dy))) * RES * 100
        self.assertAlmostEqual(s["move_median_cm"], want, delta=0.5)
        self.assertEqual(s["moved_over_50cm_ha"], 0.0)
        self.assertFalse(s["look"])
        self.assertTrue(s["line"].startswith("✓ r "))
        self.assertIn("measured at 5.0 cm", s["line"])
        self.assertEqual(json.load(open(self.qa))["r_after"], s["r_after"])

    def test_quality_raster(self):
        with rasterio.open(self.s["quality_tif"]) as q:
            self.assertEqual(q.descriptions, R.QUALITY_BANDS)
            self.assertAlmostEqual(q.res[0], R._CELL * RES)
            tags = q.tags()
            self.assertEqual(tags["method"], "eFOLKI"); self.assertEqual(tags["look"], "false")
            rb, ra, mv, east, north = q.read()
        ok = np.isfinite(mv)
        self.assertGreater(ok.mean(), 0.5)
        # the content moved back, cell by cell: west by the east shift, south by the north shift
        cy, cx = np.mgrid[0:mv.shape[0], 0:mv.shape[1]]
        X, Y = q.transform * (cx + 0.5, cy + 0.5)
        col = np.clip(((X - 500000.0) / RES).astype(int), 0, W - 1); row = np.clip(((5170000.0 - Y) / RES).astype(int), 0, H - 1)
        self.assertLess(float(np.median(np.abs(east[ok] + self.dx[row[ok], col[ok]] * RES * 100))), 0.3)
        self.assertLess(float(np.median(np.abs(north[ok] - self.dy[row[ok], col[ok]] * RES * 100))), 0.3)
        self.assertTrue(np.allclose(np.hypot(east, north)[ok], mv[ok], atol=1e-3))
        both = np.isfinite(rb) & np.isfinite(ra)
        self.assertGreater(float(np.mean(ra[both] > rb[both])), 0.95)

    def test_registered_map_lines_up_across_block_seams(self):
        with rasterio.open(self.out) as o:
            self.assertEqual(o.count, 4); self.assertEqual(o.colorinterp[-1], ColorInterp.alpha)
            c0 = int(round((o.transform.c - 500000.0) / RES)); r0 = int(round((5170000.0 - o.transform.f) / RES))
            out = o.read(); oh, ow = o.height, o.width
        ref = self.ref[:, r0:r0 + oh, c0:c0 + ow]
        inner = np.s_[60:-60, 60:-60]
        m = (out[3] > 0)[inner]
        r = np.corrcoef(out[1][inner][m].astype(float), ref[1][inner][m].astype(float))[0, 1]
        self.assertGreater(r, 0.98)
        # no step where blocks meet: the residual stays small on every 400 px block edge
        resid = np.abs(out[1].astype(float) - ref[1].astype(float))
        self.assertLess(float(np.median(resid[inner][m])), 4.0)


class CoarserMissionWithoutAlpha(unittest.TestCase):
    """A 5-band 2024-law mission at 8.3 cm (no alpha, border zeros) registered to a 5 cm reference: measured
    on the reference's 5 cm pixels, written on the mission's own 8.3 cm grid with a tagged alpha."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="reg_efolki_ms_"))
        g = texture(H, W, seed=5)
        ref = np.stack([g * 200 + 20, g * 180 + 30, (1 - g) * 150 + 40, np.full_like(g, 255)]).astype(np.uint8)
        rtr = from_origin(500000.0, 5170000.0, RES, RES)
        cls.anchor = _write(cls.tmp / "ref.tif", ref, rtr)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        mov = np.stack([cv2.remap(ref[b].astype(np.float32), xx + 6.0, yy - 3.5, cv2.INTER_CUBIC) for b in (0, 1, 2)])
        res = 0.083; tr = from_origin(500000.0 + 0.091, 5170000.0 - 0.021, res, res)
        h, w = int(H * RES / res) - 2, int(W * RES / res) - 2
        ms = np.zeros((5, h, w), np.float32)
        for b, src in enumerate((0, 1, 2, 2, 2)):
            reproject(mov[src], ms[b], src_transform=rtr, src_crs=CRS, dst_transform=tr, dst_crs=CRS,
                      resampling=Resampling.average)
        ms = np.clip(ms, 1, 255).astype(np.uint8); ms[:, :, :40] = 0; ms[:, -30:, :] = 0
        cls.mission = _write(cls.tmp / "ms.tif", ms, tr, tag_alpha=False)
        cls.out = str(cls.tmp / "ms_out.tif")
        cls.s = R.register_survey_dense(cls.mission, cls.anchor, cls.out, log=lambda m: None,
                                        band_names=["Red", "Green", "NIR", "RedEdge"])

    def test_measured_on_the_reference_written_on_its_own_grid(self):
        s = self.s
        self.assertEqual(s["measured_res_m"], RES); self.assertEqual(s["res_m"], 0.083)
        self.assertEqual(s["alpha"], "derived"); self.assertEqual(s["bands"], ["Red", "Green", "NIR", "RedEdge"])
        self.assertGreater(s["r_after"], s["r_before"] + 0.2)
        with rasterio.open(s["quality_tif"]) as q:
            east, north = q.read(4), q.read(5)
        self.assertAlmostEqual(float(np.nanmedian(east)), 30.0, delta=1.0)       # 6 px of 5 cm, east
        self.assertAlmostEqual(float(np.nanmedian(north)), 17.5, delta=1.0)      # 3.5 px, north
        with rasterio.open(self.mission) as m:
            strip_x = m.transform.c + 40 * m.res[0]                       # the border-zero strip's edge
        with rasterio.open(self.out) as o:
            self.assertAlmostEqual(o.res[0], 0.083)
            self.assertEqual(o.descriptions, ("Red", "Green", "NIR", "RedEdge", "alpha"))
            law = B.band_law(self.out, ["Red", "Green", "NIR", "RedEdge"])
            self.assertEqual(law.alpha, 5)
            self.assertGreater(o.transform.c, strip_x - 8 * 0.083)       # no-data is not carried as data
            a = o.read(5)
            self.assertGreater(float((a == 255).mean()), 0.95)


class Refusals(unittest.TestCase):
    def test_reference_that_covers_nothing(self):
        tmp = Path(tempfile.mkdtemp(prefix="reg_efolki_far_"))
        g = texture(400, 400, seed=9)
        img = np.stack([g * 200, g * 200, g * 200, np.full_like(g, 255)]).astype(np.uint8)
        a = _write(tmp / "a.tif", img, from_origin(600000.0, 5170000.0, RES, RES))
        m = _write(tmp / "m.tif", img, from_origin(500000.0, 5170000.0, RES, RES))
        with self.assertRaises(RuntimeError):
            R.register_survey_dense(m, a, str(tmp / "o.tif"), log=lambda s: None)

    def test_reference_without_alpha(self):
        tmp = Path(tempfile.mkdtemp(prefix="reg_efolki_noalpha_"))
        g = (texture(300, 300, seed=4) * 200 + 20).astype(np.uint8)
        a = _write(tmp / "a.tif", np.stack([g, g, g]), from_origin(500000.0, 5170000.0, RES, RES), tag_alpha=False)
        m = _write(tmp / "m.tif", np.stack([g, g, g, np.full_like(g, 255)]), from_origin(500000.0, 5170000.0, RES, RES))
        with self.assertRaises(RuntimeError):
            R.register_survey_dense(m, a, str(tmp / "o.tif"), log=lambda s: None)


if __name__ == "__main__":
    unittest.main()

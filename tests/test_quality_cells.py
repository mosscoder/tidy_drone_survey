"""The per-cell diagnostics raster of register_survey_dense (CELL_BANDS), the scorer on
that lattice (validate.registration_r_cells like=), and ownership recycling in seam_merge
(owner_in / geometry_only / the 2-band ownership raster). Synthetic rasters, fake matcher."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import ColorInterp

os.environ["TIDYSURVEY_REG_FAKE"] = "1"          # the chunk workers use the synthetic matcher
os.environ.setdefault("TIDYSURVEY_REG_CHUNK", "1000000")

import tidysurvey.registration as R
import tidysurvey.validate as V
import tidysurvey.fields as F
import tidysurvey.merge as M

CRS = "EPSG:32611"


def _rgba(path, W, H, res, seed, mask=None, smooth=True):
    rng = np.random.default_rng(seed)
    if smooth:                                    # textured but correlated imagery
        import cv2
        coarse = rng.integers(40, 210, (3, H // 16 + 1, W // 16 + 1)).astype(np.float32)
        rgb = np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC) for c in coarse])
        rgb = np.clip(rgb + rng.normal(0, 6, rgb.shape), 0, 255).astype(np.uint8)
    else:
        rgb = rng.integers(40, 210, (3, H, W)).astype(np.uint8)
    if mask is None:
        mask = np.ones((H, W), bool)
    rgb = rgb * mask
    prof = dict(driver="GTiff", width=W, height=H, count=4, dtype="uint8", crs=CRS,
                transform=from_origin(0, H * res, res, res), tiled=True, compress="zstd")
    with rasterio.open(path, "w", **prof) as d:
        d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        for b in range(3):
            d.write(rgb[b], b + 1)
        d.write(np.where(mask, 255, 0).astype(np.uint8), 4)
    return path


class CellsRaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cells_"))
        cls.W, cls.H, cls.res = 1600, 1200, 0.1
        cls.mission = str(_rgba(cls.tmp / "m.tif", cls.W, cls.H, cls.res, 1))
        cls.anchor = str(_rgba(cls.tmp / "a.tif", cls.W, cls.H, cls.res, 1))   # same imagery
        cls.out = str(cls.tmp / "reg.tif"); cls.qa = str(cls.tmp / "qa.json")
        cls.summary = R.register_survey_dense(cls.mission, cls.anchor, cls.out,
                                              resolution_m=cls.res, qa_json=cls.qa,
                                              log=lambda m: None)

    def test_one_raster_with_described_bands_and_tags(self):
        s = self.summary
        self.assertEqual(s["cells_tif"], str(Path(self.out).with_suffix(".cells.tif")))
        self.assertEqual(s["cell_bands"], list(R.CELL_BANDS))
        self.assertFalse(Path(self.out).with_suffix(".coverage.tif").exists())
        self.assertFalse(Path(self.out).with_suffix(".disp.tif").exists())
        with rasterio.open(s["cells_tif"]) as c:
            self.assertEqual(c.count, len(R.CELL_BANDS))
            self.assertEqual(tuple(c.descriptions), R.CELL_BANDS)
            self.assertTrue(np.isnan(c.nodata))
            self.assertAlmostEqual(abs(c.transform.a), R._FS * self.res, places=6)
            t = c.tags()
            self.assertEqual(int(t["cell_px"]), R._FS)
            self.assertEqual(int(t["cells"]), s["cells"])
            arr = c.read()
        d, dx, dy, n, fx, fy, cov = arr
        kept = np.isfinite(d)
        self.assertEqual(int(kept.sum()), s["cells"])
        self.assertTrue(np.allclose(d[kept], np.hypot(dx[kept], dy[kept])))
        # coverage: kept cells are matched (1); every kept cell has matches counted
        self.assertTrue((cov[kept] == 1).all())
        self.assertTrue((n[kept] >= R._MIN_CELL).all())
        self.assertEqual(int((cov == 0).sum()), s["failed_cells"])
        # the field is defined over the matched bbox, finite wherever a cell was kept
        self.assertTrue(np.isfinite(fx[kept]).all() and np.isfinite(fy[kept]).all())
        # a whole-raster mission: the field at kept cells is close to the raw pooled shift
        self.assertLess(float(np.median(np.abs(fx[kept] - dx[kept]))), 3.0)   # cm

    def test_scorer_lands_on_the_engine_lattice(self):
        s = self.summary
        out = str(self.tmp / "r.tif")
        r = V.registration_r_cells(self.out, self.anchor, out, like=s["cells_tif"],
                                   min_n=50, log=lambda m: None)
        with rasterio.open(out) as a, rasterio.open(s["cells_tif"]) as c:
            self.assertEqual((a.width, a.height), (c.width, c.height))
            self.assertEqual(a.transform, c.transform)
            self.assertEqual(a.crs, c.crs)
            rr = a.read(1)
        self.assertGreater(r["cells"], 0)
        self.assertGreater(r["median_r"], 0.9)          # same imagery, sub-pixel fake shift
        self.assertEqual(rr.shape, (c.height, c.width))

    def test_scorer_refuses_a_lattice_off_the_pixel_grid(self):
        bad = str(self.tmp / "bad_like.tif")
        with rasterio.open(bad, "w", driver="GTiff", width=4, height=3, count=1, dtype="float32",
                           crs=CRS, transform=from_origin(0, 100, 0.55, 0.55)) as d:
            d.write(np.zeros((3, 4), np.float32), 1)
        with self.assertRaises(ValueError):
            V.registration_r_cells(self.out, self.anchor, str(self.tmp / "x.tif"), like=bad,
                                   log=lambda m: None)


class OwnershipRecycling(unittest.TestCase):
    """Three overlapping inputs (a triple junction). Run 1 writes the ownership; run 2
    recycles it and must reproduce run 1 exactly; geometry_only writes only the
    ownership; a recycled owner that is not valid somewhere falls back."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="own_"))
        W, H, res = 2000, 1500, 0.1
        cls.res = res
        cols = np.arange(W)[None, :].repeat(H, 0); rows = np.arange(H)[:, None].repeat(W, 1)
        masks = {"m0": cols < 1200, "m1": (cols > 800) & (rows < 800), "m2": (cols > 800) & (rows > 700)}
        cls.inputs = [(n, str(_rgba(cls.tmp / f"{n}.tif", W, H, res, i, mask=m, smooth=False)))
                      for i, (n, m) in enumerate(masks.items())]
        cls._orig = (F.build_matcher, F.match_tile, F.release_matcher_cache)

        def fake_build(device=None):
            import torch
            return object(), torch.device("cpu")

        def fake_match(matcher, dev, ga, gb, reject_px, min_tile):
            h, w = ga.shape[-2:]
            ys, xs = np.mgrid[30:h - 30:40, 30:w - 30:40]
            kp = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
            rng = np.random.default_rng(int(kp.sum()) % 9999)
            disp = np.column_stack([np.full(len(kp), 0.6), np.full(len(kp), -0.4)]) + rng.normal(0, 0.05, (len(kp), 2))
            return kp, disp
        F.build_matcher, F.match_tile, F.release_matcher_cache = fake_build, fake_match, (lambda d: None)

    @classmethod
    def tearDownClass(cls):
        F.build_matcher, F.match_tile, F.release_matcher_cache = cls._orig

    def _merge(self, tag, **kw):
        out = str(self.tmp / f"{tag}.tif")
        rep = M.seam_merge(self.inputs, out, res=self.res, ownership_out=str(self.tmp / f"{tag}.own.tif"),
                           report_json=str(self.tmp / f"{tag}.json"), log=lambda m: None, **kw)
        return out, rep

    def test_recycled_ownership_reproduces_the_stitch(self):
        out1, rep1 = self._merge("first")
        with rasterio.open(rep1["ownership"]) as o:
            self.assertEqual(o.count, 2)
            self.assertEqual(o.tags()["names"], "m0,m1,m2")
            cat, own = o.read(1), o.read(2)
        self.assertEqual(int(cat.max()), 4)                       # N + 1 marks the seam band
        self.assertEqual(int(own.max()), 3)                       # band 2: the owner everywhere
        self.assertTrue(((own == cat) | (cat == 4)).all())
        out2, rep2 = self._merge("second", owner_in=rep1["ownership"])
        self.assertEqual(rep2["recycled"][1], 0)                  # nothing fell back: same alphas
        self.assertGreater(rep2["recycled"][0], 0)
        with rasterio.open(out1) as a, rasterio.open(out2) as b:
            self.assertTrue(np.array_equal(a.read(), b.read()))
        with rasterio.open(rep2["ownership"]) as o:
            self.assertTrue(np.array_equal(o.read(2), own))
            self.assertEqual(o.tags()["recycled_from"], "first.own.tif")

    def test_geometry_only_writes_the_ownership_and_nothing_else(self):
        out, rep = self._merge("geom", geometry_only=True)
        self.assertEqual(rep["kind"], "ownership")
        self.assertFalse(Path(out).exists())
        self.assertTrue(Path(rep["ownership"]).exists())
        self.assertFalse(Path(str(out).rsplit(".", 1)[0] + "_coarse").exists())
        with rasterio.open(rep["ownership"]) as o:
            self.assertEqual(o.count, 2)

    def test_invalid_recycled_owner_falls_back(self):
        # an ownership that hands the whole grid to m1, which is invalid in the lower left
        out, rep = self._merge("base", geometry_only=True)
        with rasterio.open(rep["ownership"]) as o:
            prof = o.profile; tags = o.tags()
        prof.update(count=2)
        forged = str(self.tmp / "forged.own.tif")
        with rasterio.open(forged, "w", **prof) as d:
            d.update_tags(**tags)
            d.write(np.full((prof["height"], prof["width"]), 2, np.uint8), 1)
            d.write(np.full((prof["height"], prof["width"]), 2, np.uint8), 2)
        out2, rep2 = self._merge("fallback", owner_in=forged)
        kept, fell = rep2["recycled"]
        self.assertGreater(fell, 0)                               # m1 is not valid everywhere
        self.assertGreater(kept, 0)                               # but where it is, it owns
        with rasterio.open(out2) as d:
            arr = d.read()
        self.assertGreater(float((arr[-1] > 0).mean()), 0.9)      # the union footprint rendered

    def test_unknown_names_are_refused(self):
        out, rep = self._merge("names", geometry_only=True)
        with rasterio.open(rep["ownership"]) as o:
            prof = o.profile
        bad = str(self.tmp / "bad.own.tif")
        with rasterio.open(bad, "w", **prof) as d:
            d.update_tags(names="m0,m1,other")
            d.write(np.ones((2, prof["height"], prof["width"]), np.uint8))
        with self.assertRaises(ValueError):
            self._merge("refused", owner_in=bad)


if __name__ == "__main__":
    unittest.main()


class PlanWithoutCalibration(unittest.TestCase):
    def test_reference_none_drops_scene_and_calibrate(self):
        import tidysurvey.config as C
        cfg = C.Config(survey="s", crs="EPSG:6514", run_dir=".", georeferencing="gcp")
        cfg.visible.orthos = [C.NamedInput("a", "a.tif")]; cfg.ms.missions = [C.NamedInput("a", "m.tif")]
        self.assertEqual(cfg.plan(), ["scene", "stitch/visible", "align/ms", "stitch/ms", "calibrate", "tiles", "report"])
        cfg.calibrate.reference = "none"
        self.assertEqual(cfg.plan(), ["stitch/visible", "align/ms", "stitch/ms", "tiles", "report"])

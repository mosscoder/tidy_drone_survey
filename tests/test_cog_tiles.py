"""0.3.4: the delivery COG and the web map, written for real on a small survey.

finalize_cog raised NameError in 0.3.3 (a module-level `gdal` the lazy import had removed); build_pmtiles failed
its read-back on a survey smaller than a pixel at the lowest zooms (the writer stamps the zooms it holds).  Both
were first run on a cloud VM by the season stitch.  Skipped where the GDAL bindings or pmtiles are not installed."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

HAVE = importlib.util.find_spec("osgeo") is not None and importlib.util.find_spec("pmtiles") is not None


def _rgba(path, w=2400, h=1800, res=0.032):
    rng = np.random.default_rng(0)
    rgb = rng.integers(40, 210, (3, h, w), dtype=np.uint8)
    prof = dict(driver="GTiff", width=w, height=h, count=4, dtype="uint8", crs="EPSG:6514",
                transform=from_origin(499840.0, 300000.8, res, res), tiled=True, blockxsize=512, blockysize=512, compress="zstd")
    with rasterio.open(path, "w", **prof) as d:
        d.write(rgb, [1, 2, 3]); d.write(np.full((h, w), 255, np.uint8), 4)
        d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
    return path


@unittest.skipUnless(HAVE, "needs the GDAL bindings and pmtiles")
class CogAndTiles(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="cogtiles_"))

    def test_finalize_writes_a_cog_with_webp_overviews(self):
        from osgeo import gdal
        from tidysurvey import cog
        out = cog.finalize_cog(_rgba(self.d / "master.tif"), self.d / "final.tif", lossless=False, log=lambda m: None)
        ds = gdal.Open(out); md = ds.GetMetadata("IMAGE_STRUCTURE")
        self.assertEqual(md.get("LAYOUT"), "COG"); self.assertEqual(md.get("COMPRESSION"), "ZSTD")
        self.assertGreater(ds.GetRasterBand(1).GetOverviewCount(), 0)
        cog.finalize_cog(self.d / "master.tif", self.d / "lossless.tif", lossless=True, log=lambda m: None)

    def test_a_small_survey_gets_its_archive_and_viewer(self):
        from tidysurvey import cog, tiles
        src = cog.finalize_cog(_rgba(self.d / "master.tif"), self.d / "base.tif", lossless=False, log=lambda m: None)
        lines = []
        rep = tiles.build_pmtiles(src, self.d / "base.pmtiles", min_zoom=8, log=lines.append)
        self.assertGreater(rep["min_zoom"], 8)                         # 77 m wide: nothing to draw at z8
        self.assertTrue(any("held no pixels" in x for x in lines))
        html = Path(rep["map_html"]); self.assertTrue(html.exists()); self.assertIn('"base.pmtiles"', html.read_text())
        from pmtiles.reader import MmapSource, Reader
        with open(self.d / "base.pmtiles", "rb") as f:
            h = Reader(MmapSource(f)).header()
        self.assertEqual((h["min_zoom"], h["max_zoom"]), (rep["min_zoom"], rep["max_zoom"]))


if __name__ == "__main__":
    unittest.main()

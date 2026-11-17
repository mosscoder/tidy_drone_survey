"""The fade at a seam follows the seam line, whatever footprints the line came from (2026-11-16).

Until this, the fade's weights came from each map's own data edges, while the line itself could come from another
map's footprints (owner_in: the multispectral stitch reuses the visible map's seams). Where the two disagreed by more
than the band, one map kept full weight across the whole band, and the change from one map to the other was a hard
edge at the band's rim. The 2024 summer multispectral map was stitched that way: the change was 11 cm wide where a
1 m fade was meant. The existing test of owner_in reused a line made from the SAME footprints, where the two agree,
so it could not see it.

Two maps of one scene, the second twice as bright as the first, so the share of the first map in the stitched map can
be read off at every pixel: share = (stitched - second) / (first - second).
"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

import tidysurvey.merge as M

CRS = "EPSG:32611"
W, H, RES = 1400, 400, 0.1
BAND_M = 4.0
B = BAND_M / RES                                   # the fade's width in pixels


def _scene():
    rng = np.random.default_rng(3)
    coarse = rng.uniform(0.6, 1.4, (H // 16 + 1, W // 16 + 1)).astype(np.float32)
    return cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC)


def _write(path, level, mask):
    """An RGB + alpha map: the scene at `level`, valid where `mask`."""
    t = _scene()
    rgb = np.clip(np.stack([t * level, t * level * 0.9, t * level * 0.8]), 1, 255).astype(np.uint8) * mask
    prof = dict(driver="GTiff", width=W, height=H, count=4, dtype="uint8", crs=CRS,
                transform=from_origin(0, H * RES, RES, RES), tiled=True, compress="zstd")
    with rasterio.open(path, "w", **prof) as d:
        d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        d.write(rgb[0], 1); d.write(rgb[1], 2); d.write(rgb[2], 3)
        d.write(np.where(mask, 255, 0).astype(np.uint8), 4)
    return str(path)


def _share_of_first(out, a_path, b_path, rows=slice(100, 300)):
    """Column by column, the first map's share in the stitched map (median over the rows)."""
    with rasterio.open(out) as o, rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        O, A, Bm = (d.read(1).astype(np.float32) for d in (o, a, b))
    t = _scene()
    full_a = np.clip(t * 80, 1, 255); full_b = np.clip(t * 160, 1, 255)       # what each map holds where it has data
    return np.median(((O - full_b) / (full_a - full_b))[rows], axis=0)


def _width_and_middle(share):
    """Where the share passes 0.9 and 0.1 going from the first map's side to the second's: (width, middle) in pixels."""
    hi = np.nonzero(share > 0.9)[0].max()
    lo = np.nonzero(share < 0.1)[0].min()
    return lo - hi, (lo + hi) / 2.0


class FadeFollowsTheSeamLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fade_"))
        cols = np.arange(W)[None, :].repeat(H, 0)
        # the maps whose footprints MAKE the seam line: they overlap from column 600 to 1000, so the line is at 800
        cls.line_a = _write(cls.tmp / "line_a.tif", 80, cols < 1000)
        cls.line_b = _write(cls.tmp / "line_b.tif", 160, cols >= 600)
        # the maps that are stitched: they overlap from column 300 to 900, so their own middle is at 600
        cls.a = _write(cls.tmp / "a.tif", 80, cols < 900)
        cls.b = _write(cls.tmp / "b.tif", 160, cols >= 300)

    def _merge(self, tag, inputs, **kw):
        out = str(self.tmp / f"{tag}.tif")
        rep = M.seam_merge(inputs, out, res=RES, band_width_m=BAND_M, ownership_out=str(self.tmp / f"{tag}.own.tif"),
                           report_json=str(self.tmp / f"{tag}.json"), log=lambda m: None, **kw)
        return out, rep

    def test_its_own_line(self):
        """Two maps and the line their own footprints give: half and half on the line, the band wide."""
        out, rep = self._merge("own", [("a", self.a), ("b", self.b)])
        share = _share_of_first(out, self.a, self.b)
        width, middle = _width_and_middle(share)
        self.assertAlmostEqual(middle, 600, delta=3)
        self.assertAlmostEqual(width, 0.8 * B, delta=0.1 * B)                     # a straight ramp: 0.8 of the band between 10% and 90%
        self.assertAlmostEqual(float(share[600]), 0.5, delta=0.1)
        self.assertGreater(rep["seams"][0]["mix_on_line"], 0.4)

    def test_a_line_from_other_footprints(self):
        """The line reused from maps with other footprints (the multispectral stitch on the visible map's seams): the
        fade is on THAT line, 200 pixels from the stitched maps' own middle."""
        _, own = self._merge("line", [("a", self.line_a), ("b", self.line_b)], geometry_only=True)
        out, rep = self._merge("reused", [("a", self.a), ("b", self.b)], owner_in=own["ownership"])
        self.assertEqual(rep["recycled"][1], 0)                                   # the line was kept everywhere
        share = _share_of_first(out, self.a, self.b)
        width, middle = _width_and_middle(share)
        self.assertAlmostEqual(middle, 800, delta=3)
        self.assertAlmostEqual(width, 0.8 * B, delta=0.1 * B)
        self.assertAlmostEqual(float(share[800]), 0.5, delta=0.1)
        self.assertGreater(float(share[800 - int(B / 2) - 4]), 0.95)              # past the fade, one map alone
        self.assertLess(float(share[800 + int(B / 2) + 4]), 0.05)
        self.assertTrue((np.diff(share[740:860]) < 0.05).all())                   # one way only: no step back
        self.assertGreater(rep["seams"][0]["mix_on_line"], 0.4)                   # and the report says the line is mixed

    def test_a_hole_in_one_map_inside_the_fade(self):
        """Where the second map has no data inside the fade, on the first map's side of the line, the first map
        fills it and the second map's share fades in from the hole's rim, with no hard step there. (A hole that cuts
        the seam line itself is filled the way a hole anywhere in the map is, and is not part of this.)"""
        cols = np.arange(W)[None, :].repeat(H, 0); rows = np.arange(H)[:, None].repeat(W, 1)
        hole = (cols >= 776) & (cols < 792) & (rows >= 152) & (rows < 248)        # on 8-pixel cells, 8 to 24 px from the line
        b_holed = _write(self.tmp / "b_holed.tif", 160, (cols >= 300) & ~hole)
        _, own = self._merge("line2", [("a", self.line_a), ("b", self.line_b)], geometry_only=True)
        out, rep = self._merge("holed", [("a", self.a), ("b", b_holed)], owner_in=own["ownership"])
        with rasterio.open(out) as o:
            self.assertTrue((o.read(4)[:, 300:900] > 0).all())                    # no gap where either map has data
        share = _share_of_first(out, self.a, b_holed, rows=slice(190, 210))       # a strip through the hole
        self.assertGreater(float(share[784]), 0.95)                               # in the hole: the first map alone
        self.assertAlmostEqual(float(share[800]), 0.5, delta=0.12)                # the line is still half and half
        self.assertLess(float(np.abs(np.diff(share[700:899])).max()), 0.2)        # no step at the hole's rim


if __name__ == "__main__":
    unittest.main()

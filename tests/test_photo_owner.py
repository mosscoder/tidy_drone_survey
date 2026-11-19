"""Ownership by photo positions (merge.photo_owner; mpg-aerial-pipeline issue 39).

Two flights side by side. Flight A's photos lie on the west, flight B's on the east, but A's map reaches far east,
over B's photos, the way a map's margin runs past where its photos were taken. The distance rule (deepest inside
each map's own data) gives that margin to A; the photo rule gives it to B, whose photos are overhead. Where the
chosen flight has no data (a hole in B), the distance rule's owner stays. Also: the positions CSV is read into the
merge's coordinate system, and a whole stitch runs with positions given."""
import os
import shutil
import tempfile
import unittest

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, gaussian_filter

import tidysurvey.merge as M

CRES = 0.5                                    # the coarse grid's cell, m
HC, WC = 120, 240                             # 60 m by 120 m
TR = Affine.translation(0, HC * CRES) * Affine.scale(CRES, -CRES)


def _xcells(x0, x1):
    return slice(int(x0 / CRES), int(x1 / CRES))


def _setup(hole=False):
    valid = np.zeros((2, HC, WC), bool)
    valid[0][:, _xcells(0, 118)] = True       # A's map: 0 to 118 m (its margin runs far east, past B's map)
    valid[1][:, _xcells(55, 104)] = True      # B's map: 55 to 104 m
    if hole:
        valid[1][:, _xcells(84, 90)] = False  # a hole in B's map
    ds = np.stack([distance_transform_edt(v) for v in valid])
    owner = np.where(valid.any(0), ds.argmax(0), -1).astype(np.int16)
    xs = np.arange(2.0, 41.0, 6.0); ys = np.arange(5.0, 60.0, 8.0)
    a = np.array([(x, y) for x in xs for y in ys])                  # A's photos: 2 to 38 m east
    b = np.array([(x + 60.0, y) for x in xs for y in ys])           # B's photos: 62 to 98 m east
    return valid, owner, {"A": a, "B": b}


def _col(x):
    return int(x / CRES)


class PhotoOwner(unittest.TestCase):
    def test_margin_goes_to_the_flight_overhead(self):
        valid, owner, photos = _setup()
        row = HC // 2
        self.assertEqual(owner[row, _col(100)], 0)                  # the distance rule: A's margin wins at 100 m
        new, (kept, fell_back) = M.photo_owner(["A", "B"], photos, valid, owner.copy(), TR, WC, HC, log=lambda *a: None)
        self.assertEqual(new[row, _col(100)], 1)                    # by photos: B, whose photos are overhead
        self.assertEqual(new[row, _col(20)], 0)                     # A's own ground stays A's
        self.assertEqual(new[row, _col(112)], 0)                    # only A's map reaches here: A
        self.assertLess(fell_back, 0.02 * kept)                     # only along B's data edge, where the 2 m grid straddles it
        both = valid[0, row] & valid[1, row]                        # the overlap: 55 to 104 m
        self.assertTrue((owner[row][both] == 0).any())              # the distance rule gives part of it to A's margin
        self.assertTrue((new[row][both] == 1).all())                # by photos all of it is B's: the midline between
                                                                    # the two flights' photos (~50 m) is west of B's map

    def test_no_data_falls_back_to_the_distance_rule(self):
        valid, owner, photos = _setup(hole=True)
        new, (kept, fell_back) = M.photo_owner(["A", "B"], photos, valid, owner.copy(), TR, WC, HC, log=lambda *a: None)
        row = HC // 2
        self.assertEqual(new[row, _col(87)], 0)                     # B has a hole here: A, the only map with data
        self.assertEqual(new[row, _col(95)], 1)
        self.assertGreater(fell_back, 0)

    def test_inputs_without_photos_do_not_compete(self):
        valid, owner, photos = _setup()
        new, _ = M.photo_owner(["A", "B"], {"B": photos["B"]}, valid, owner.copy(), TR, WC, HC, log=lambda *a: None)
        self.assertEqual(new[HC // 2, _col(100)], 1)
        self.assertEqual(new[HC // 2, _col(20)], 0)                 # only A has data here: kept by the fallback

    def test_positions_csv(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "photos.csv")
            with open(p, "w") as fh:
                fh.write("name,lon,lat\n00,-114.0192,46.7160\n00,-114.0191,46.7161\n01,-114.0100,46.7200\n")
            got = M.read_photo_positions(p, "EPSG:6514")
            self.assertEqual(sorted(got), ["00", "01"])
            self.assertEqual(got["00"].shape, (2, 2))
            x, y = got["00"][0]
            self.assertTrue(254000 < x < 256000 and 283000 < y < 285000)   # NAD83(2011) / Montana, m
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


W, H, RES = 2400, 1200, 0.05
CRS = "EPSG:32611"


def _maps(tmp):
    """Two flights of one textured ground. A covers x 0 to 100 m of 120, B 40 to 120: the distance rule's seam
    lies near 70 m. A's photos are at 0 to 40 m and B's at 60 to 120, so by photos the seam moves west, to ~50 m."""
    rng = np.random.default_rng(1)
    g = sum(gaussian_filter(rng.normal(size=(H, W)), s) for s in (1.5, 4.0, 12.0))
    g = ((g - g.min()) / (g.max() - g.min()))
    rgb = np.stack([g * 200 + 20, g * 170 + 40, (1 - g) * 150 + 50])
    cols = np.arange(W)[None, :].repeat(H, 0)
    out = []
    for name, mask in (("A", cols < 2000), ("B", cols >= 800)):
        p = f"{tmp}/{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", width=W, height=H, count=4, dtype="uint8", crs=CRS,
                           transform=from_origin(0, H * RES, RES, RES), tiled=True, compress="zstd") as d:
            d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
            d.write((np.clip(rgb, 1, 255) * mask).astype(np.uint8), [1, 2, 3])
            d.write(np.where(mask, 255, 0).astype(np.uint8), 4)
        out.append((name, p))
    return out


class StitchWithPhotos(unittest.TestCase):
    def test_seam_moves_to_between_the_photos(self):
        tmp = tempfile.mkdtemp()
        try:
            inputs = _maps(tmp)
            ys = np.arange(3.0, 60.0, 8.0)
            photos = {"A": np.array([(x, y) for x in np.arange(0.0, 41.0, 8.0) for y in ys]),
                      "B": np.array([(x, y) for x in np.arange(60.0, 121.0, 8.0) for y in ys])}
            seams = {}
            for tag, ph in (("edges", None), ("photos", photos)):
                own = f"{tmp}/own_{tag}.tif"
                rep = M.seam_merge(inputs, f"{tmp}/out_{tag}.tif", band_width_m=1.0, ownership_out=own,
                                   geometry_only=True, photos=ph, log=lambda *a: None)
                with rasterio.open(own) as s:
                    o = s.read(2)
                row = o[o.shape[0] // 2]
                cs = [c for c in range(1, len(row)) if row[c] != row[c - 1] and row[c] and row[c - 1]]
                seams[tag] = cs[0] * abs(s.transform.a)
                if tag == "photos":
                    self.assertIsNotNone(rep.get("photo_owner"))
            self.assertTrue(65 <= seams["edges"] <= 75, seams)          # the distance rule: mid-overlap of the maps
            self.assertTrue(45 <= seams["photos"] <= 55, seams)         # by photos: between the two flights' photos
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

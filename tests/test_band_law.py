"""Band law (tidysurvey/bands.py): DroneDeploy's 2024 5-band (no alpha) and 2025+ 6-band
(tagged alpha) multispectral exports, RGBA visible orthos, and the registration engine's
use of them — a 6-band mission must register to 4 named spectral bands + a tagged alpha,
never to 7 bands with the source alpha resampled as data (the July 2026 run's shape)."""
import os
import json
import unittest
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import ColorInterp

os.environ.setdefault("TIDYSURVEY_REG_FAKE", "1")     # registration test: synthetic matcher
os.environ.setdefault("TIDYSURVEY_REG_CHUNK", "4")

from tidysurvey import bands as B                      # noqa: E402

NAMES = ["Red", "Green", "NIR", "RedEdge"]
CRS = "EPSG:32611"


def _write(path, arr, tag_alpha=False, descriptions=None):
    n, h, w = arr.shape
    prof = dict(driver="GTiff", width=w, height=h, count=n, dtype="uint8", crs=CRS,
                transform=from_origin(0, h * 0.1, 0.1, 0.1), tiled=True, compress="zstd")
    with rasterio.open(path, "w", **prof) as d:
        # tags go on BEFORE the pixels: on a tiled/compressed GTiff a tag set after
        # writing is silently dropped (bands.tag_output documents the same rule)
        if tag_alpha:
            ci = [ColorInterp.gray] + [ColorInterp.undefined] * (n - 2) + [ColorInterp.alpha]
            if n == 4:
                ci = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
            d.colorinterp = ci
        if descriptions:
            for i, s in enumerate(descriptions, 1):
                d.set_band_description(i, s)
        d.write(arr)


def _ms(h=300, w=400, seed=0):
    """R, G, NIR, RE random; band 5 = NIR again; alpha 255 except: a zero CORNER (all bands
    zero, alpha 0), an interior ISLAND of zero data with alpha 255 (valid zero data), and an
    interior PATCH of non-zero data with alpha 0 (alpha says invalid; only alpha can know)."""
    rng = np.random.default_rng(seed)
    spec = rng.integers(20, 200, (4, h, w)).astype(np.uint8)
    alpha = np.full((h, w), 255, np.uint8)
    spec[:, :60, :60] = 0; alpha[:60, :60] = 0                  # corner: nodata
    spec[:, 120:140, 150:170] = 0                               # island: valid zeros
    alpha[200:220, 300:320] = 0                                 # patch: alpha-only invalid
    six = np.concatenate([spec, spec[2:3], alpha[None]], 0)     # 6-band: + NIR dup + alpha
    five = six[:5]                                              # 2024 shape: no alpha
    return six, five


class TestBandLaw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bandlaw_")
        six, five = _ms()
        cls.six, cls.five = six, five
        cls.p6 = f"{cls.tmp}/ms6.tif";   _write(cls.p6, six, tag_alpha=True)
        cls.p5 = f"{cls.tmp}/ms5.tif";   _write(cls.p5, five)
        rgba = np.concatenate([six[:3], six[5:6]], 0)
        cls.prgba_t = f"{cls.tmp}/rgba_t.tif"; _write(cls.prgba_t, rgba, tag_alpha=True)
        cls.prgba_u = f"{cls.tmp}/rgba_u.tif"; _write(cls.prgba_u, rgba)
        cls.prgb = f"{cls.tmp}/rgb.tif";  _write(cls.prgb, six[:3])
        cls.pnamed = f"{cls.tmp}/named.tif"
        _write(cls.pnamed, np.concatenate([six[:4], six[5:6]], 0), descriptions=NAMES + ["alpha"])

    def test_six_band_tagged(self):
        law = B.band_law(self.p6, NAMES)
        self.assertEqual(law.spectral, (1, 2, 3, 4))         # the NIR duplicate is dropped
        self.assertEqual(law.alpha, 6)
        self.assertEqual(law.names, tuple(NAMES))
        self.assertEqual(law.nspec, 4)

    def test_six_band_unnamed(self):
        law = B.band_law(self.p6)
        self.assertEqual(law.spectral, (1, 2, 3, 4, 5))
        self.assertEqual(law.alpha, 6)
        self.assertEqual(law.names[0], "band_1")

    def test_five_band_legacy(self):
        law = B.band_law(self.p5, NAMES)
        self.assertEqual(law.spectral, (1, 2, 3, 4))
        self.assertIsNone(law.alpha)                         # 2024: no alpha anywhere

    def test_rgba_and_rgb(self):
        self.assertEqual((B.band_law(self.prgba_t).spectral, B.band_law(self.prgba_t).alpha), ((1, 2, 3), 4))
        self.assertEqual((B.band_law(self.prgba_u).spectral, B.band_law(self.prgba_u).alpha), ((1, 2, 3), 4))
        self.assertEqual((B.band_law(self.prgb).spectral, B.band_law(self.prgb).alpha), ((1, 2, 3), None))

    def test_named_untagged_alpha(self):
        law = B.band_law(self.pnamed, NAMES)
        self.assertEqual(law.alpha, 5)                       # description 'alpha' is honoured

    def test_too_many_names(self):
        with self.assertRaises(ValueError):
            B.band_law(self.prgb, NAMES)

    def test_validity_tagged_is_exact(self):
        law = B.band_law(self.p6, NAMES)
        with rasterio.open(self.p6) as s:
            v = B.read_valid(s, law)
        self.assertTrue((v == (self.six[5] > 127)).all())
        self.assertFalse(v[10, 10]); self.assertTrue(v[130, 160]); self.assertFalse(v[210, 310])

    def test_validity_derived_from_zeros(self):
        law = B.band_law(self.p5, NAMES)
        with rasterio.open(self.p5) as s:
            v = B.read_valid(s, law)
        self.assertFalse(v[10, 10])                          # border-connected zero: nodata
        self.assertTrue(v[130, 160])                         # interior zero island: data
        self.assertTrue(v[210, 310])                         # non-zero data: data (no alpha to say otherwise)
        with rasterio.open(self.p5) as s:                    # windowed: plain any-non-zero
            from rasterio.windows import Window
            vw = B.read_valid(s, law, window=Window(100, 100, 100, 100))
        self.assertFalse(vw[30, 60]); self.assertTrue(vw[0, 0])

    def test_tag_output_roundtrip(self):
        p = f"{self.tmp}/tagged.tif"
        arr = np.concatenate([self.six[:4], self.six[5:6]], 0)
        prof = dict(driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=5,
                    dtype="uint8", crs=CRS, transform=from_origin(0, 30, 0.1, 0.1))
        with rasterio.open(p, "w", **prof) as d:
            B.tag_output(d, NAMES)
            d.write(arr)
        with rasterio.open(p) as s:
            self.assertEqual(s.colorinterp[-1], ColorInterp.alpha)
            self.assertEqual(s.descriptions, tuple(NAMES) + ("alpha",))
        law = B.band_law(p, NAMES)
        self.assertEqual((law.spectral, law.alpha), ((1, 2, 3, 4), 5))


class TestRegistrationBandLaw(unittest.TestCase):
    """A 6-band DroneDeploy-shaped mission registered (fake matcher) to an RGBA anchor."""
    @classmethod
    def setUpClass(cls):
        import tidysurvey.registration as R
        cls.tmp = tempfile.mkdtemp(prefix="bandlaw_reg_")
        H, W = 1500, 2000
        rng = np.random.default_rng(3)
        # smooth, band-independent fields (a sub-pixel shift must not decorrelate them)
        import cv2
        spec = np.stack([cv2.resize(rng.random((30, 40)).astype(np.float32), (W, H),
                                    interpolation=cv2.INTER_CUBIC) for _ in range(4)])
        spec = (40 + 170 * np.clip(spec, 0, 1)).astype(np.uint8)
        alpha = np.full((H, W), 255, np.uint8)
        spec[:, :300, :300] = 0; alpha[:300, :300] = 0             # nodata corner
        alpha[700:760, 900:960] = 0                                 # alpha-only hole
        six = np.concatenate([spec, spec[2:3], alpha[None]], 0)
        cls.mission = f"{cls.tmp}/mission6.tif"; _write(cls.mission, six, tag_alpha=True)
        rgba = np.concatenate([rng.integers(40, 210, (3, H, W)).astype(np.uint8),
                               np.full((1, H, W), 255, np.uint8)], 0)
        cls.anchor = f"{cls.tmp}/anchor.tif"; _write(cls.anchor, rgba, tag_alpha=True)
        cls.out = f"{cls.tmp}/registered.tif"; cls.qa = f"{cls.tmp}/qa.json"
        cls.summary = R.register_survey_dense(cls.mission, cls.anchor, cls.out, resolution_m=0.1,
                                              qa_json=cls.qa, band_names=NAMES, log=lambda m: None)
        cls.six = six

    def test_output_shape_and_tags(self):
        with rasterio.open(self.out) as s:
            self.assertEqual(s.count, 5)                             # 4 spectral + alpha, not 7
            self.assertEqual(s.colorinterp[-1], ColorInterp.alpha)
            self.assertEqual(s.descriptions, tuple(NAMES) + ("alpha",))
            law = B.band_law(self.out, NAMES)
            self.assertEqual((law.spectral, law.alpha), ((1, 2, 3, 4), 5))
            a = s.read(5)
            self.assertEqual(int(a[10, 10]), 0)                      # corner stays nodata
            self.assertEqual(int(a[730, 930]), 0)                    # alpha-only hole honoured
            self.assertEqual(int(a[1000, 1500]), 255)
        self.assertEqual(self.summary["bands"], NAMES)
        self.assertEqual(self.summary["alpha"], "tagged")
        self.assertEqual(json.load(open(self.qa))["alpha"], "tagged")

    def test_spectral_content_is_the_first_four(self):
        # the fake matcher shifts by (0.6, -0.4) px: bands correlate strongly with their source,
        # and band 4 (RedEdge) must NOT be the NIR duplicate
        with rasterio.open(self.out) as s:
            out = s.read([1, 2, 3, 4]).astype(np.float32)
        src = self.six[:4].astype(np.float32)
        sl = (slice(400, 1400), slice(400, 1900))
        for b in range(4):
            r = np.corrcoef(out[b][sl].ravel(), src[b][sl].ravel())[0, 1]
            self.assertGreater(r, 0.95, f"band {b + 1} r={r:.2f}")
        r34 = np.corrcoef(out[3][sl].ravel(), src[2][sl].ravel())[0, 1]
        self.assertLess(r34, 0.5, "band 4 must be RedEdge, not the NIR duplicate")


if __name__ == "__main__":
    unittest.main()

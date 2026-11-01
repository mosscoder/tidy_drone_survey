"""0.4.0: the seam merge end to end, with eFOLKI measuring the seams inside the render.

Three flights of one textured ground meet at a triple junction, each with its content moved by a known
amount (flight 1 two pixels east of flight 0, flight 2 one and a half pixels north), so every seam's
offset is known. Checks: the three seams are found and measured at their known offsets, the split honours
them, nothing outside the 1 m band changes (validate.assert_interiors_unchanged), and both resume paths,
the coarse-pass reuse and the render block-resume, reproduce the clean run byte for byte."""
import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

import tidysurvey.merge as M
import tidysurvey.validate as V

W, H, RES = 2400, 1800, 0.05
CRS = "EPSG:32611"
SHIFT = {"m0": (0.0, 0.0), "m1": (2.0, 0.0), "m2": (0.0, -1.5)}     # content moved (east px, row px)


def _inputs(tmp):
    rng = np.random.default_rng(0)
    g = sum(gaussian_filter(rng.normal(size=(H, W)), s) / gaussian_filter(rng.normal(size=(H, W)), s).std()
            for s in (1.5, 4.0, 12.0))
    g = ((g - g.min()) / (g.max() - g.min())).astype(np.float32)
    ground = np.stack([g * 200 + 20, g * 170 + 40, (1 - g) * 150 + 50])
    cols = np.arange(W)[None, :].repeat(H, 0); rows = np.arange(H)[:, None].repeat(W, 1)
    masks = {"m0": cols < 1400, "m1": (cols > 1000) & (rows < 1000), "m2": (cols > 1000) & (rows > 800)}
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = []
    for name, mask in masks.items():
        dx, dy = SHIFT[name]
        rgb = np.stack([cv2.remap(b, xx - dx, yy - dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) for b in ground])
        rgb = (np.clip(rgb, 1, 255) * mask).astype(np.uint8)
        p = f"{tmp}/{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", width=W, height=H, count=4, dtype="uint8", crs=CRS,
                           transform=from_origin(0, H * RES, RES, RES), tiled=True, compress="zstd") as d:
            d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
            d.write(rgb, [1, 2, 3]); d.write(np.where(mask, 255, 0).astype(np.uint8), 4)
        out.append((name, p))
    return out


def _clear(out):
    base = out.rsplit(".", 1)[0]
    shutil.rmtree(base + "_coarse", ignore_errors=True)
    for f in (out, base + "_render_ckpt.json"):
        if os.path.exists(f):
            os.remove(f)


class SeamMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="seam_e2e_")
        cls.inputs = _inputs(cls.tmp)
        cls.out = f"{cls.tmp}/clean.tif"
        cls.rep = M.seam_merge(cls.inputs, cls.out, res=RES, block=1024, ownership_out=f"{cls.tmp}/own.tif",
                               report_json=f"{cls.tmp}/stitch.json", log=lambda m: None)
        with rasterio.open(cls.out) as d:
            cls.clean = d.read()

    def test_three_seams_at_their_known_offsets(self):
        rep = self.rep
        self.assertEqual(rep["method"], "eFOLKI"); self.assertEqual(rep["n_seams"], 3)
        want = {"m0-m1": 2.0, "m0-m2": 1.5, "m1-m2": float(np.hypot(2.0, 1.5))}     # pixels
        for s in rep["seams"]:
            self.assertGreater(s["seam_m"], 10)
            self.assertAlmostEqual(s["med_offset_cm"], want[s["pair"]] * RES * 100, delta=1.0, msg=s["pair"])
            self.assertEqual(s["over50_m"], 0.0)
            self.assertLess(s["residual_cm"], 2.0)                 # the split honours the measurements
        V.seam_tripwire(rep, max_cm=20).assert_ok()
        self.assertEqual(json.load(open(f"{self.tmp}/stitch.json"))["n_seams"], 3)

    def test_interiors_unchanged(self):
        r = V.assert_interiors_unchanged(self.inputs, self.out, f"{self.tmp}/own.tif", band_width_m=1.0,
                                         block_px=1024, log=lambda m: None)
        r.assert_ok()

    def test_coarse_reuse_reproduces_the_clean_run(self):
        out = f"{self.tmp}/coarse.tif"; _clear(out)
        orig = M._combinations

        def boom(*a, **k):
            raise RuntimeError("injected: crash right after the coarse pass")
        M._combinations = boom
        try:
            with self.assertRaises(RuntimeError):
                M.seam_merge(self.inputs, out, res=RES, block=1024, log=lambda m: None)
        finally:
            M._combinations = orig
        self.assertTrue(os.path.exists(out.rsplit(".", 1)[0] + "_coarse/_coarse_done.json"))
        logs = []
        M.seam_merge(self.inputs, out, res=RES, block=1024, log=logs.append)
        self.assertTrue(any("reusing complete geometry" in m for m in logs))
        with rasterio.open(out) as d:
            self.assertTrue(np.array_equal(d.read(), self.clean))
        self.assertFalse(os.path.exists(out.rsplit(".", 1)[0] + "_coarse"))

    def test_render_resume_reproduces_the_clean_run(self):
        out = f"{self.tmp}/resume.tif"; _clear(out)
        orig = M._write_json_atomic
        n = {"k": 0}

        def crashing(path, obj):
            if "render_ckpt" in str(path):
                n["k"] += 1
                if n["k"] == 2:                                   # chunk 0 checkpointed, die on chunk 1
                    raise RuntimeError("injected: crash mid-render")
            return orig(path, obj)
        M._write_json_atomic = crashing
        try:
            with self.assertRaises(RuntimeError):
                M.seam_merge(self.inputs, out, res=RES, block=1024, render_chunk=1, log=lambda m: None)
        finally:
            M._write_json_atomic = orig
        ck = out.rsplit(".", 1)[0] + "_render_ckpt.json"
        with open(ck) as f:
            self.assertEqual(json.load(f)["done"], [0])
        logs = []
        rep = M.seam_merge(self.inputs, out, res=RES, block=1024, render_chunk=1, log=logs.append)
        self.assertTrue(any("composite: resuming" in m for m in logs))
        self.assertEqual(rep["resumed_blocks"], 1)
        with rasterio.open(out) as d:
            self.assertTrue(np.array_equal(d.read(), self.clean))
        self.assertFalse(os.path.exists(ck))

    def test_anchored_gauge_keeps_the_first_input(self):
        out = f"{self.tmp}/anchored.tif"; _clear(out)
        M.seam_merge(self.inputs, out, res=RES, block=1024, gauge="anchored", log=lambda m: None)
        with rasterio.open(out) as d, rasterio.open(self.inputs[0][1]) as s0:
            a, src = d.read(), s0.read()
        # on m0's side of the m0-m1 seam, inside the band, m0 is not moved: the blend there is m0 and the
        # moved m1, so the band differs from the clean (free) run but m0's pixels away from it do not
        self.assertFalse(np.array_equal(a, self.clean))
        self.assertTrue(np.array_equal(a[:3, 100:700, 100:900], src[:3, 100:700, 100:900]))


if __name__ == "__main__":
    unittest.main()

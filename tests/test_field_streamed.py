"""0.3.2: registration memory independent of map size.

StreamedField serves fill_field's values by column band without the (out_h, out_w) array;
resize_nn_offsets reproduces cv2.resize INTER_NEAREST offsets so a block of the native-resolution
validity mask is gathered from the GEOM_DS mask. Both must be bit-identical to the arrays they
replace (the registered map and cells.tif are byte-identical to 0.3.1 — see the 0.3.2 notes).

The slow case registers a large synthetic pair in a subprocess and bounds its peak RSS from the
block/halo/worker arithmetic; enable with TIDYSURVEY_SLOW_TESTS=1.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

import tidysurvey.fields as F


class StreamedFieldExact(unittest.TestCase):
    SHAPES = [(151, 201), (17, 33), (793, 51), (320, 1093), (96, 96), (5, 70), (1200, 1601)]

    def _pts(self, rng, out_w, out_h, n=60):
        pts = np.column_stack([rng.uniform(-3, out_w + 3, n), rng.uniform(-3, out_h + 3, n)])
        return pts, rng.normal(0, 2.5, n), rng.normal(0, 2.5, n)

    def test_columns_reproduce_fill_field_bit_for_bit(self):
        rng = np.random.default_rng(1)
        for out_h, out_w in self.SHAPES:
            pts, vx, _ = self._pts(rng, out_w, out_h)
            full = F.fill_field(pts, vx, out_w, out_h)
            sf = F.StreamedField(pts, vx, out_w, out_h)
            self.assertEqual(sf.hp.shape, (-(-out_h // 8), out_w))          # 1/8 of the field
            got = np.empty_like(full)
            for c0 in range(0, out_w, 97):
                got[:, c0:c0 + 97] = sf.columns(c0, c0 + 97)
            self.assertTrue(np.array_equal(full, got), (out_h, out_w))
            self.assertEqual(sf.absmax(band=64), float(np.abs(full).max()))

    def test_sample_reproduces_remap_with_replicate_border(self):
        rng = np.random.default_rng(2)
        for out_h, out_w in self.SHAPES[:5]:
            pts, vx, _ = self._pts(rng, out_w, out_h)
            full = F.fill_field(pts, vx, out_w, out_h)
            sf = F.StreamedField(pts, vx, out_w, out_h)
            mx = rng.uniform(-2, out_w + 1, (40, 61)).astype(np.float32)
            my = rng.uniform(-2, out_h + 1, (40, 61)).astype(np.float32)
            ref = cv2.remap(full, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            self.assertTrue(np.array_equal(ref, sf.sample(mx, my, band=64)), (out_h, out_w))

    def test_empty_samples_give_zero_field(self):
        sf = F.StreamedField(np.zeros((0, 2)), np.zeros(0), 40, 30)
        self.assertTrue((sf.columns(0, 40) == 0).all()); self.assertEqual(sf.absmax(), 0.0)
        self.assertTrue((sf.sample(np.ones((2, 3), np.float32), np.ones((2, 3), np.float32)) == 0).all())


class NearestOffsets(unittest.TestCase):
    def test_offsets_reproduce_cv2_resize_nearest(self):
        rng = np.random.default_rng(3)
        for sh, sw, dh, dw in [(201, 151, 1601, 1207), (200, 150, 1600, 1200), (1, 3, 7, 23),
                               (67, 201, 534, 1602), (293, 407, 2341, 3251)]:
            v = rng.integers(0, 2, (sh, sw)).astype(np.uint8)
            ref = cv2.resize(v, (dw, dh), interpolation=cv2.INTER_NEAREST)
            got = v[np.ix_(F.resize_nn_offsets(sh, dh), F.resize_nn_offsets(sw, dw))]
            self.assertTrue(np.array_equal(ref, got), (sh, sw, dh, dw))


_RUNNER = r"""
import os, sys, json, resource
os.environ["TIDYSURVEY_REG_FAKE"] = "1"; os.environ["TIDYSURVEY_REG_CHUNK"] = "1000000"
import tidysurvey.registration as R
mission, out, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = R.register_survey_dense(mission, mission, out, workers=workers, log=lambda m: None)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak = peak / 2**20 if sys.platform == "darwin" else peak / 2**10          # bytes on macOS, KiB on Linux
print(json.dumps(dict(peak_mib=peak, maxF_cm=s["maxF_cm"], cells=s["cells"], out=s["out"])))
"""


@unittest.skipUnless(os.environ.get("TIDYSURVEY_SLOW_TESTS"), "set TIDYSURVEY_SLOW_TESTS=1 (several minutes, ~3 GB disk)")
class RegistrationMemoryIndependentOfMapSize(unittest.TestCase):
    """Synthetic missions of 10k x 10k and 20k x 20k (4 bands: 100 and 400 Mpx) register with the fake
    matcher, each in its own subprocess, and (1) the 20k peak stays inside a bound derived from what the
    engine holds rather than from the map, (2) the 20k peak exceeds the 10k peak by no more than the
    map-scaled arrays plus the allocator's run-to-run spread. Terms of the bound, in MiB:

        imports                       256   numpy, scipy, cv2, rasterio, GDAL (measured 223)
        GDAL block cache              256   GDAL_CACHEMAX=256 is set for the runs, so the bound is machine-independent
        coarse-geometry transients    320   the two GEOM_DS WarpedVRT passes (warper working memory, read_valid);
                                            measured 270-320 above the cache at both 10k and 20k: not map-scaled
        per worker           per_worker()   every array a worker holds, allocated once at the largest haloed block
                                            S = BLOCK + 2*halo: gx, gy, mapx, mapy, rem (4 B each), a8 (1 B),
                                            outb (nspec+1 B), the native window uint8 + float32 (5*nspec B over
                                            (S+2*PAD+2)^2), the two field upsample buffers (4 B over (S+16)^2), the
                                            contiguous write buffer (nspec+1 B over BLOCK^2), the per-block validity
                                            gather and its bool mask (2 B over S^2); plus GDAL's own staging of a
                                            written window and its compression queue (2 x (nspec+1) B over BLOCK^2)
        map-scaled arrays              64   GEOM_DS validity x2 (H/8 * W/8 B), StreamedField x2 (H/64 * W/8 floats),
                                            the cell rasters (H/64 * W/64 * 7 floats), the pooled matches: ~25 at 20k
        interpreter + allocator       256   the same 20k run measured 1504 and 1610 MiB on two occasions: macOS's
                                            allocator retains freed pages unevenly across worker threads
        ------------------------------------------------------------------
        bound = 832 + workers * per_worker + 64 + 256           (1666 for two workers)

    0.3.1 held the native-resolution validity (400 MB at 20k) and the FIELD_DS field (2 x 25 MB) on top,
    and its warp re-allocated ~700 MB per worker per block: it measured 2421 MiB on the same 20k fixture.
    """
    WORKERS, NSPEC = 2, 3

    @staticmethod
    def per_worker_mib(nspec, halo=17, pad=3):
        import tidysurvey.registration as R
        S = R._BLOCK + 2 * halo; SN = S + 2 * pad + 2; SB = S + 2 * R._FIELD_DS; B = R._BLOCK
        held = (S * S * (5 * 4 + 1 + (nspec + 1)) + SN * SN * (nspec + 4 * nspec) + 2 * SB * SB * 4
                + B * B * (nspec + 1) + 2 * S * S)
        gdal_write = 2 * B * B * (nspec + 1)
        return (held + gdal_write) / 2**20

    @property
    def BOUND_MIB(self):
        return 256 + 256 + 320 + self.WORKERS * self.per_worker_mib(self.NSPEC) + 64 + 256

    def _fixture(self, tmp, N):
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.enums import ColorInterp
        mission = str(tmp / f"m{N}.tif")
        prof = dict(driver="GTiff", width=N, height=N, count=4, dtype="uint8", crs="EPSG:32611",
                    transform=from_origin(0, N * 0.05, 0.05, 0.05), tiled=True, blockxsize=512,
                    blockysize=512, compress="zstd", BIGTIFF="YES")
        rng = np.random.default_rng(5)
        with rasterio.open(mission, "w", **prof) as d:              # written in strips: the fixture never sits in RAM
            d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
            for r0 in range(0, N, 2048):
                h = min(2048, N - r0)
                coarse = rng.integers(40, 210, (3, h // 16 + 1, N // 16 + 1)).astype(np.float32)
                strip = np.stack([cv2.resize(c, (N, h), interpolation=cv2.INTER_CUBIC) for c in coarse])
                strip = np.clip(strip + rng.normal(0, 6, strip.shape), 0, 255).astype(np.uint8)
                for b in range(3):
                    d.write(strip[b], b + 1, window=rasterio.windows.Window(0, r0, N, h))
                d.write(np.full((h, N), 255, np.uint8), 4, window=rasterio.windows.Window(0, r0, N, h))
        return mission

    def _register(self, mission, out):
        env = dict(os.environ, GDAL_CACHEMAX="256", PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        r = subprocess.run([sys.executable, "-c", _RUNNER, mission, out, str(self.WORKERS)],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_peak_rss_within_bound_and_flat_across_map_size(self):
        import rasterio
        tmp = Path(tempfile.mkdtemp(prefix="regmem_"))
        res = {}
        for N in (10000, 20000):
            res[N] = self._register(self._fixture(tmp, N), str(tmp / f"reg{N}.tif"))
            print(f"\n  {N//1000}k x {N//1000}k, {self.WORKERS} workers: peak RSS {res[N]['peak_mib']:.0f} MiB "
                  f"(bound {self.BOUND_MIB:.0f}), maxF {res[N]['maxF_cm']} cm, cells {res[N]['cells']}")
            with rasterio.open(res[N]["out"]) as o:
                self.assertEqual((o.width, o.height, o.count), (N, N, 4))
        self.assertLess(res[20000]["peak_mib"], self.BOUND_MIB)
        self.assertLess(res[20000]["peak_mib"] - res[10000]["peak_mib"], 64 + 256)   # map-scaled arrays + allocator spread
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()


if __name__ == "__main__":
    unittest.main()

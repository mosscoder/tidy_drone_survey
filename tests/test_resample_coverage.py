"""Refactor checks: single-cubic output warp (geometry-neutral + sharper, no
overshoot) and the durable per-cell coverage + displacement diagnostics.

The register warp now reads the mission at NATIVE res and folds the native->grid
affine + the correction field into ONE cubic remap (was: bilinear VRT read THEN a
bilinear correction remap). The grid origin is lattice-snapped so the stitch can
crop interiors instead of re-averaging them.
"""
import os
os.environ.setdefault("TIDYSURVEY_REG_FAKE", "1")
os.environ.setdefault("TIDYSURVEY_REG_CHUNK", "4")

import math
import tempfile
import shutil

import numpy as np
import rasterio
import cv2
from rasterio.transform import from_origin, Affine
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling, ColorInterp

import tidysurvey.registration as R


# ---- 1) the composed native->grid cubic is geometry-correct + sharper + clipped
def test_composition_geometry_sharpness_overshoot():
    NW, NH, nres = 800, 600, 0.05
    M = from_origin(1000.37, 2000.61, nres, nres)          # arbitrary (non-lattice) origin
    yy, xx = np.mgrid[0:NH, 0:NW]
    pat = ((((xx // 6 + yy // 6) % 2) * 220 + 20) + (xx * 0.1).astype(int) % 40).astype(np.uint8)
    tmp = tempfile.mktemp(suffix=".tif")
    with rasterio.open(tmp, "w", driver="GTiff", width=NW, height=NH, count=1, dtype="uint8",
                       crs="EPSG:6514", transform=M) as d:
        d.write(pat, 1)

    res = 0.066                                            # downsample, snapped origin
    mbL, mbT, mbR, mbB = M.c, M.f, M.c + NW * nres, M.f - NH * nres
    ox = math.floor(mbL / res) * res; oy = math.ceil(mbT / res) * res
    W = int(np.ceil((mbR - ox) / res)); H = int(np.ceil((oy - mbB) / res))
    TR = Affine.translation(ox, oy) * Affine.scale(res, -res)
    sx = TR.a / M.a; sy = TR.e / M.e
    ax = 0.5 * sx + (TR.c - M.c) / M.a - 0.5
    ay = 0.5 * sy + (TR.f - M.f) / M.e - 0.5
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = (sx * gx + ax).astype(np.float32); mapy = (sy * gy + ay).astype(np.float32)
    with rasterio.open(tmp) as m:
        native = m.read(1).astype(np.float32)
    cub = cv2.remap(native, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mine = np.clip(cub, 0, 255)
    bil = np.clip(cv2.remap(native, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, 255)
    with rasterio.open(tmp) as m, WarpedVRT(m, crs="EPSG:6514", transform=TR, width=W, height=H,
                                            resampling=Resampling.cubic) as v:
        ref = v.read(1).astype(np.float32)
    os.remove(tmp)

    v2 = (mine > 1) & (ref > 1)
    a = mine[v2] - mine[v2].mean(); b = ref[v2] - ref[v2].mean()
    ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    sh = lambda g: float(np.mean(np.hypot(*np.gradient(g))[v2]))
    assert ncc > 0.98, f"composition misaligned vs GDAL cubic: NCC={ncc}"
    assert sh(mine) > sh(bil), "cubic not sharper than bilinear"
    assert cub.max() > 255.5 or cub.min() < -0.5, "test invalid: pattern didn't force overshoot"
    assert mine.min() >= 0 and mine.max() <= 255, "overshoot not clipped"
    print(f"1 PASS  geometry NCC={ncc:.4f}  cubic/bilinear sharpness {sh(mine)/sh(bil):.2f}x  clipped OK")


# ---- 2) register writes valid, georeferenced coverage + displacement rasters
def _rgba(path, seed, W, H, tr):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(40, 210, (3, H, W)).astype(np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=4, dtype="uint8",
                       crs="EPSG:32611", transform=tr, tiled=True, compress="zstd") as d:
        for b in range(3):
            d.write(rgb[b], b + 1)
        d.write(np.full((H, W), 255, np.uint8), 4)
        d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]


def test_coverage_and_disp_rasters():
    TMP = tempfile.mkdtemp(prefix="cov_")
    W, H, res = 2000, 1500, 0.1
    tr = from_origin(0, H * res, res, res)
    _rgba(f"{TMP}/m.tif", 1, W, H, tr); _rgba(f"{TMP}/a.tif", 2, W, H, tr)
    s = R.register_survey_dense(f"{TMP}/m.tif", f"{TMP}/a.tif", f"{TMP}/reg.tif",
                                resolution_m=res, qa_json=f"{TMP}/qa.json", log=lambda x: None)
    for k in ("coverage", "disp", "failed_cells"):
        assert k in s, f"summary missing {k}"
    with rasterio.open(s["coverage"]) as d:
        c = d.read(1)
        assert d.res == (64 * res, 64 * res), f"coverage not on the _FS cell lattice: {d.res}"
        assert d.nodata == 255 and str(d.crs) == "EPSG:32611"
        assert (c == 1).sum() > 0 and (c == 0).sum() == s["failed_cells"]
    with rasterio.open(s["disp"]) as d:
        assert d.count == 3 and d.descriptions[0] == "|d|_cm"
        assert abs(float(np.nanmedian(d.read(1))) - s["d_med_cm"]) < 2
    shutil.rmtree(TMP)
    print(f"2 PASS  coverage matched/failed={ (c==1).sum() }/{s['failed_cells']}  disp |d|~{s['d_med_cm']}cm")


if __name__ == "__main__":
    test_composition_geometry_sharpness_overshoot()
    test_coverage_and_disp_rasters()
    print("\nALL RESAMPLE+COVERAGE CHECKS PASSED")

"""One way to write files: the standard Cloud-Optimized GeoTIFF recipes.

Stages stream their output as tiled ZSTD GeoTIFFs (fast to write in
parallel); `finalize_cog` recompresses that master into ONE compact COG with
overviews built during the translate — lossy WEBP overviews for the visible
viewing mosaic, lossless ZSTD throughout for multispectral and calibrated
products (the current multispectral recipe, kept)."""
from __future__ import annotations

import os
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()


def finalize_cog(src, dst, lossless=True, level=9, blocksize=512,
                 overview_quality=85, workers=None, log=print):
    """Recompress a streamed master GeoTIFF into the delivery COG.

    lossless=True  : ZSTD base + ZSTD (average) overviews — multispectral,
                     calibrated reflectance, any data product.
    lossless=False : ZSTD base + WEBP overviews — the visible viewing mosaic
                     (zoom-outs are for looking at; the base stays lossless).
    """
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    opts = [
        "COMPRESS=ZSTD", f"LEVEL={level}", "PREDICTOR=2", f"BLOCKSIZE={blocksize}",
        "OVERVIEW_RESAMPLING=AVERAGE", "BIGTIFF=YES", f"NUM_THREADS={workers}",
    ]
    if lossless:
        opts.append("OVERVIEW_COMPRESS=ZSTD")
    else:
        opts += ["OVERVIEW_COMPRESS=WEBP", f"OVERVIEW_QUALITY={overview_quality}"]
    gdal.SetConfigOption("GDAL_CACHEMAX", "8192")
    log(f"[cog] {src} -> {dst} ({'lossless' if lossless else 'webp overviews'})")
    ds = gdal.Translate(str(dst), str(src), format="COG", creationOptions=opts)
    ds = None
    return str(dst)

"""Visible mosaic -> PMTiles: the whole web map in one static file.

PMTiles (protomaps.com/pmtiles) is a single-file archive of WebMercator
XYZ tiles readable over HTTP range requests — a slippy map served from any
static host or bucket, no tile server. Tiles are rendered straight from the
delivered COG (whose overviews make the low zooms cheap), encoded as WEBP,
and written in clustered (Hilbert) order.

The archive is a VIEWING artifact derived from the deliverable, like the
COG's own overviews — analysis stays on the COG.
"""
from __future__ import annotations

import io
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

WEB_MERC = "EPSG:3857"
_E = 20037508.342789244            # mercator half-world; world = [-_E, _E]


def _res_z(z: int, tile_px: int) -> float:
    """Mercator metres per pixel of zoom z with tile_px tiles."""
    return 2 * _E / ((1 << z) * tile_px)


def auto_max_zoom(src_res_m: float, center_lat_deg: float, tile_px: int) -> int:
    """Smallest zoom whose GROUND resolution matches the source GSD.
    Mercator metres shrink by cos(lat) on the ground, so the zoom that
    matches a 3.2 cm GSD in Montana is coarser than at the equator."""
    ground_world = 2 * _E * math.cos(math.radians(center_lat_deg))
    return max(0, math.ceil(math.log2(ground_world / (src_res_m * tile_px))))


def build_pmtiles(src, dst, tile_px=512, quality=85, min_zoom=8, max_zoom=None,
                  workers=None, log=print):
    """Render `src` (any GDAL-openable raster, RGBA uint8) into `dst` .pmtiles.

    tile_px  : 512 (the @2x raster convention — a quarter the tiles of 256).
    quality  : WEBP quality for tile encode (the COG base stays lossless;
               like its overviews, tiles are for looking at).
    min_zoom : lowest zoom rendered (8 ~ the survey is a dot; lower is noise).
    max_zoom : None = derived from the source GSD at the survey's latitude.

    Zoomed-out tiles render from the COG's own overview pyramid (for the
    visible product those are WEBP — lossy feeding lossy, fine for a viewing
    artifact; the top zooms render from the lossless base).

    Fully-transparent tiles are skipped — the archive is sparse over the
    survey's irregular footprint. Identical tiles are stored once (writer
    dedup). Returns a dict for the stage report.
    """
    import faulthandler
    from PIL import Image
    from pmtiles.tile import Compression, TileType, zxy_to_tileid
    from pmtiles.writer import write as pm_write

    faulthandler.enable()
    t0 = time.perf_counter()
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    src, dst = str(src), str(dst)

    with rasterio.open(src) as s:
        merc = transform_bounds(s.crs, WEB_MERC, *s.bounds, densify_pts=21)
        lonlat = transform_bounds(s.crs, "EPSG:4326", *s.bounds, densify_pts=21)
        src_res = abs(s.res[0])
        n_bands = s.count
        n_px = s.width * s.height
        ov_factors = s.overviews(1)              # the COG's pyramid rungs
    center_lat = (lonlat[1] + lonlat[3]) / 2
    if not ov_factors and n_px > 4e8:
        raise ValueError(
            f"{src} has NO overviews ({n_px / 1e9:.1f} Gpx): every zoomed-out tile "
            "would resample the full base (measured >20 min and >10 GB per tile). "
            "Finalize the source as a COG first (cog.finalize_cog) — the pipeline's "
            "products already are.")
    if max_zoom is None:
        max_zoom = auto_max_zoom(src_res, center_lat, tile_px)
    if min_zoom > max_zoom:
        min_zoom = max_zoom

    # every (z, x, y) whose tile intersects the survey, in Hilbert order
    jobs = []
    for z in range(min_zoom, max_zoom + 1):
        size = 2 * _E / (1 << z)
        x0 = max(0, int((merc[0] + _E) // size))
        x1 = min((1 << z) - 1, int((merc[2] + _E) // size))
        y0 = max(0, int((_E - merc[3]) // size))
        y1 = min((1 << z) - 1, int((_E - merc[1]) // size))
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                jobs.append((zxy_to_tileid(z, x, y), z, x, y))
    jobs.sort()
    log(f"[pmtiles] {src} -> {dst}")
    log(f"    z{min_zoom}..z{max_zoom} ({tile_px}px tiles, webp q{quality}) — "
        f"{len(jobs)} candidate tiles over the survey bbox")

    tls = threading.local()

    def _overview_level(z):
        """The pyramid rung that feeds this zoom. A plain WarpedVRT on the
        base ignores overviews, so one z8 tile average-resamples the ENTIRE
        survey (measured: >20 min and >10 GB for a single tile — several in
        parallel killed the first build). Opening the right overview makes
        every tile a ~1-2x resample of a small window. None = the base."""
        ground = _res_z(z, tile_px) * math.cos(math.radians(center_lat))
        need = ground / src_res
        level = None
        for i, f in enumerate(ov_factors):
            if f <= need:
                level = i
        return level

    def vrt_for(z):
        if not hasattr(tls, "vrts"):
            tls.vrts = {}
        if z not in tls.vrts:
            lvl = _overview_level(z)
            ds = (rasterio.open(src) if lvl is None
                  else rasterio.open(src, OVERVIEW_LEVEL=lvl))
            size = 2 * _E / (1 << z)
            x0 = max(0, int((merc[0] + _E) // size))
            x1 = min((1 << z) - 1, int((merc[2] + _E) // size))
            y0 = max(0, int((_E - merc[3]) // size))
            y1 = min((1 << z) - 1, int((_E - merc[1]) // size))
            from affine import Affine
            tr = (Affine.translation(-_E + x0 * size, _E - y0 * size)
                  * Affine.scale(_res_z(z, tile_px), -_res_z(z, tile_px)))
            tls.vrts[z] = ((x0, y0), WarpedVRT(
                ds, crs=WEB_MERC, transform=tr,
                width=(x1 - x0 + 1) * tile_px, height=(y1 - y0 + 1) * tile_px,
                # bilinear where tiles outresolve the source (the top zoom),
                # average for the mild residual decimation off the overview
                resampling=(Resampling.bilinear if _res_z(z, tile_px) < src_res
                            else Resampling.average)))
        return tls.vrts[z]

    def render(job):
        tileid, z, x, y = job
        (x0, y0), v = vrt_for(z)
        arr = v.read(window=Window((x - x0) * tile_px, (y - y0) * tile_px,
                                   tile_px, tile_px))
        if n_bands < 4:
            alpha = np.full((tile_px, tile_px), 255, np.uint8)
            arr = np.concatenate([arr[:3], alpha[None]], 0)
        if not arr[3].any():
            return tileid, None                          # fully transparent: skip
        img = Image.fromarray(np.moveaxis(arr[:4], 0, 2), "RGBA")
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=quality)
        return tileid, buf.getvalue()

    written = skipped = 0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with pm_write(dst) as w:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            CHUNK = 512                                  # bounded RAM: ~CHUNK tiles in flight
            t_last = t0
            for i in range(0, len(jobs), CHUNK):
                for tileid, data in ex.map(render, jobs[i:i + CHUNK]):
                    if data is None:
                        skipped += 1
                    else:
                        w.write_tile(tileid, data)
                        written += 1
                now = time.perf_counter()
                done = min(i + CHUNK, len(jobs))
                if now - t_last >= 60 or done == len(jobs):
                    t_last = now
                    rate = done / max(now - t0, 1e-9)
                    eta_m = (len(jobs) - done) / rate / 60
                    log(f"    [{done}/{len(jobs)}] {100 * done / len(jobs):4.1f}%  "
                        f"written={written} empty={skipped}  ETA ~{eta_m:.0f} min")
        header = {
            "tile_type": TileType.WEBP,
            "tile_compression": Compression.NONE,        # webp is already compressed
            "min_lon_e7": int(lonlat[0] * 1e7), "min_lat_e7": int(lonlat[1] * 1e7),
            "max_lon_e7": int(lonlat[2] * 1e7), "max_lat_e7": int(lonlat[3] * 1e7),
            "center_zoom": min(min_zoom + 4, max_zoom),
            "center_lon_e7": int((lonlat[0] + lonlat[2]) / 2 * 1e7),
            "center_lat_e7": int(center_lat * 1e7),
        }
        metadata = {"name": Path(dst).stem, "format": "webp",
                    "minzoom": str(min_zoom), "maxzoom": str(max_zoom),
                    "bounds": ",".join(f"{v:.7f}" for v in lonlat)}
        w.finalize(header, metadata)

    # read the archive back like a client would — silent corruption is the
    # failure mode a build log cannot see
    from pmtiles.reader import MmapSource, Reader
    with open(dst, "rb") as f:
        hdr = Reader(MmapSource(f)).header()
    if hdr["tile_type"] != TileType.WEBP or hdr["min_zoom"] != min_zoom \
            or hdr["max_zoom"] != max_zoom:
        raise RuntimeError(f"pmtiles read-back mismatch: {hdr}")
    log(f"    read-back OK: webp, z{hdr['min_zoom']}..z{hdr['max_zoom']}, "
        f"{hdr['addressed_tiles_count']} addressed tiles")

    out_mb = os.path.getsize(dst) / 1e6
    rep = dict(kind="tiles", out=dst, tiles_written=written, tiles_empty=skipped,
               min_zoom=min_zoom, max_zoom=max_zoom, tile_px=tile_px,
               quality=quality, size_mb=round(out_mb, 1),
               seconds=round(time.perf_counter() - t0, 1))
    rep["map_html"] = write_leaflet_html(
        Path(dst).with_name(Path(dst).stem + "_map.html"), Path(dst).name,
        Path(dst).stem, lonlat, min_zoom, max_zoom, log=log)
    log(f"[pmtiles] done ({rep['seconds']:.0f}s) — {written} tiles, "
        f"{out_mb:.0f} MB -> {dst}")
    return rep


_VIEWER_HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  html,body,#map{height:100%;margin:0}
  /* shown only when the archive fails to load or render */
  #err{display:none;position:absolute;left:50%;bottom:18px;transform:translateX(-50%);
    z-index:1000;max-width:min(680px,92vw);padding:10px 14px;border-radius:8px;
    background:#fff;border:1px solid #c33;color:#922;
    font:13px/1.5 system-ui,sans-serif;box-shadow:0 8px 24px -12px #0008}
  #err code{background:#f3f3f3;padding:0 4px;border-radius:3px}
</style>

<div id="map"></div>
<div id="err"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/pmtiles@3.2.1/dist/pmtiles.js"></script>
<script>
// Generated by `tidysurvey tiles`. Default archive: the sibling .pmtiles;
// view any other with ?src=<url>. Needs a byte-range HTTP server:
//   tidysurvey serve --config <survey>.toml     (file:// cannot work)
const SRC = new URLSearchParams(location.search).get("src") || "__PMTILES__";
const fail = m => { const e = document.getElementById("err");
                    e.style.display = "block"; e.innerHTML = m; };

const map = L.map("map", { maxZoom: 26 });
map.fitBounds(L.latLngBounds([__SOUTH__, __WEST__], [__NORTH__, __EAST__]),
              { padding: [16, 16] });
L.control.scale({ imperial: false }).addTo(map);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  { maxZoom: 26, maxNativeZoom: 19, attribution: "&copy; OpenStreetMap" }).addTo(map);

const p = new pmtiles.PMTiles(SRC);
p.getHeader().then(h => {
  if (h.tileType === 1) {
    fail("This archive holds <code>MVT vector</code> tiles; this viewer renders raster PMTiles.");
    return;
  }
  const bounds = L.latLngBounds([h.minLat, h.minLon], [h.maxLat, h.maxLon]);
  const overlay = pmtiles.leafletRasterLayer(p, { attribution: "__TITLE__",
    maxNativeZoom: h.maxZoom, maxZoom: 26, bounds: bounds }).addTo(map);
  map.fitBounds(bounds, { padding: [16, 16] });
  // diagnose "header loaded but nothing renders" instead of a silent blank map
  let ok = 0, err = 0;
  overlay.on("tileload", () => { ok++; });
  overlay.on("tileerror", () => { err++; });
  setTimeout(() => {
    if (ok > 0) return;
    fail("Archive header loaded (z" + h.minZoom + "&ndash;z" + h.maxZoom + ") but <b>no tiles rendered</b>"
      + (err ? " (" + err + " tile errors)" : "")
      + (h.tileType === 4 ? " &mdash; tiles are WEBP; a browser without WEBP decode shows blank." : "")
      + " Check the console / Network tab.");
  }, 4000);
}).catch(e => {
  fail("Could not load <code>" + SRC + "</code>.<br>Most common cause: opened via "
    + "<code>file://</code> or a server without HTTP Range support "
    + "(<code>python -m http.server</code> does NOT have it). Run "
    + "<code>tidysurvey serve --config &lt;survey&gt;.toml</code> and use the printed URL."
    + "<br><br><span style='color:#777'>" + (e && e.message ? e.message : e) + "</span>");
});
</script>
"""


def write_leaflet_html(out_html, pmtiles_name, title, lonlat, min_zoom, max_zoom,
                       log=print):
    """The archive's double-clickable face — an inspector, not just a map:
    header-driven metadata HUD, opacity slider, basemap/overlay toggles, live
    coordinate strip, ?src= override to point at any other archive, and loud
    diagnoses for the two classic failure modes (no byte ranges; WEBP decode).
    The .pmtiles is referenced by RELATIVE name so the pair works from any
    byte-range-capable static host, and locally via `tidysurvey serve`.
    (Patterns adopted from the user's cog2pmtiles inspector, 2026-06.)"""
    west, south, east, north = lonlat
    html = (_VIEWER_HTML
            .replace("__TITLE__", str(title))
            .replace("__PMTILES__", str(pmtiles_name))
            .replace("__WEST__", f"{west:.7f}")
            .replace("__SOUTH__", f"{south:.7f}")
            .replace("__EAST__", f"{east:.7f}")
            .replace("__NORTH__", f"{north:.7f}"))
    Path(out_html).write_text(html)
    log(f"[pmtiles] viewer -> {out_html} (view via `tidysurvey serve`)")
    return str(out_html)

"""Alignment: move each mission onto the reference by the shift eFOLKI measures at every pixel.

A mission (a visible or a multispectral map) registers to a reference map, in practice the same year's
summer visible map. eFOLKI (efolki.py) measures the shift on the reference map's own pixels, with the
mission averaged onto them, in blocks of 4096 pixels read with 1024 pixels of context on every side. Each
block of the output is then warped once, cubic, from the mission's own pixels onto its grid. There is no
camera model, no global transform and no feature matching. The method and its evidence are in the audit,
02_registration/METHOD.md and matcher_ablation/report.md section "Native resolution".

Every registration reports three numbers, measured on the same pixels: agreement with the reference (Pearson
r of the grey images) before and after, the median move, and the area that moved more than 50 cm. A map is
flagged for a look when more than 0.5 percent of it moved that far or agreement did not rise. The per-cell
version is the quality raster beside the output (<out>.quality.tif, QUALITY_BANDS).

The first-generation per-chip machinery is at tag `v0.1.0-legacy`; the LoFTR engine (matched tiles,
corroborated 64 px cells, one smooth field) is at `v0.3.4`. The public entry point
`register_survey_by_chips` keeps its exact signature and runs this engine.
"""
from __future__ import annotations

import json as _json
import math as _math
import os
import threading as _threading
import time as _time
from pathlib import Path as _Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor as _TPE

import numpy as np
import cv2
import rasterio
from rasterio.crs import CRS as _CRS
from rasterio.vrt import WarpedVRT as _WarpedVRT
import scipy.ndimage as _ndi
from rasterio.windows import Window as _Window
from rasterio.enums import Resampling as _Resampling
from affine import Affine as _Affine
from . import bands as _B
from . import heartbeat as _HB

_GEOM_DS = 8                 # the coarse validity mask, in output pixels
_CORE, _PAD = 4096, 512      # a measured block and the context read around it, in reference pixels. The settings need
                             # about 350 px (reach 256, pyramid, rank, the shift); 1024 until 0.4.2 cost 31 % more eFOLKI
                             # work for the same result to within floating-point noise (spring 00, 2026-11-01)
_CELL = 64                   # the quality raster's cell, in reference pixels (2.1 m on a 3.3 cm map)
_MIN_SHARE = 0.01            # a block is measured when the mission and the reference each cover this share of it
_MIN_CELL = 0.25             # a cell is scored when both maps cover this share of it
_BLOCKS_AT_ONCE = 2          # blocks in flight by default; TIDYSURVEY_BLOCKS_AT_ONCE or blocks_at_once= sets it per machine
BIG_MOVE_CM = 50.0           # the size of a shadow slide: the area that moved more than this is reported
LOOK_PCT = 0.5               # a map is flagged for a look when more than this percent of it moved over BIG_MOVE_CM
QUALITY_BANDS = ("r_before", "r_after", "move_cm", "move_east_cm", "move_north_cm")


def _prewarp_to_grid(src, out_crs, TR, W, H, bands, resampling, out_path, workers):
    """Stream `src` (given bands) onto the (TR, W, H) grid as a LOCAL GTiff so
    the chunk workers read locally instead of re-fetching a network COG per tile
    (audit stage 0 — the biggest read-side win). Threaded readers overlap the
    network latency. Temporary; removed with the .regwork dir."""
    prof = dict(driver="GTiff", height=H, width=W, count=len(bands), dtype="uint8",
                crs=out_crs, transform=TR, tiled=True, blockxsize=512,
                blockysize=512, compress="zstd", ZSTD_LEVEL=1, BIGTIFF="IF_SAFER",
                num_threads=str(workers))
    dst = rasterio.open(out_path, "w", **prof)
    with rasterio.open(src) as s0:                 # carry the source's tags (alpha!) onto the stage
        if s0.count == len(bands) and s0.colorinterp:
            try:
                dst.colorinterp = [s0.colorinterp[b - 1] for b in bands]
            except Exception:
                pass
    lock = _threading.Lock(); tls = _threading.local()
    blocks = [(r0, c0) for r0 in range(0, H, 4096) for c0 in range(0, W, 4096)]

    def _one(bc):
        r0, c0 = bc
        if not hasattr(tls, "v"):
            tls.v = _WarpedVRT(rasterio.open(src), crs=out_crs, transform=TR,
                               width=W, height=H, resampling=resampling)
        bh, bw = min(4096, H - r0), min(4096, W - c0)
        win = _Window(c0, r0, bw, bh)
        with lock:
            dst.write(tls.v.read(bands, window=win), window=win)

    with _TPE(max_workers=max(2, min(6, workers))) as ex:
        list(ex.map(_one, blocks))
    dst.close()


def prewarp_union_anchor(anchor, out_crs, missions, res, out_path, log=print):
    """Stage the borrowed anchor ONCE onto the UNION extent of all missions as a
    LOCAL COG, so it is not re-fetched from the NAS per tile per mission (21x
    redundant). The union grid is SNAPPED to the anchor's own pixel lattice, so
    when the anchor's CRS matches this stage is a pixel-exact CROP (no resample),
    and the registration then measures on the anchor's own pixels. Shared across
    the align pass; cli._align removes it afterwards."""
    from rasterio.warp import transform_bounds as _tbounds
    x0 = y0 = float("inf"); x1 = y1 = float("-inf")
    for m in missions:
        with rasterio.open(m) as s:
            b = _tbounds(s.crs, out_crs, *s.bounds)
        x0, y0, x1, y1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
    with rasterio.open(anchor) as a:
        nb, at = a.count, a.transform
        same_crs = str(a.crs) == str(out_crs)
    if same_crs:                         # snap to the anchor lattice -> exact crop
        ares, ax, ay = abs(at.a), at.c, at.f
        x0 = ax + _math.floor((x0 - ax) / ares) * ares
        y1 = ay - _math.floor((ay - y1) / ares) * ares
        gres, rs, how = ares, _Resampling.nearest, "exact crop"
    else:                                # cross-CRS reproject -> one resample here
        gres, rs, how = res, _Resampling.average, "reproject"
    W = int(_math.ceil((x1 - x0) / gres)); H = int(_math.ceil((y1 - y0) / gres))
    TR = _Affine.translation(x0, y1) * _Affine.scale(gres, -gres)
    workers = max(1, (os.cpu_count() or 4) - 1)
    log(f"    staging anchor ONCE onto the union {W}x{H} grid "
        f"({how}; {len(missions)} missions) ...")
    _ts = _time.time()
    _prewarp_to_grid(anchor, out_crs, TR, W, H, list(range(1, nb + 1)), rs, out_path, workers)
    log(f"    anchor union stage done ({_time.time() - _ts:.0f}s) -> local")
    return out_path


def stage_local(src, dst, log=print):
    """Byte-copy a remote mission COG (https / gs:// / /vsicurl) to a LOCAL file
    so its per-tile reads are local, not over the network — the dominant chunk
    cost once the anchor is local. The WarpedVRT read still applies the single
    native->grid warp; this only relocates the bytes. Temporary; removed after
    the mission registers. Retries transient network failures."""
    import urllib.request as _ur, urllib.error as _ue
    import shutil as _sh
    s = str(src)
    if s.startswith("gs://"):
        s = "https://storage.googleapis.com/" + s[5:]
    elif s.startswith("/vsicurl/"):
        s = s[len("/vsicurl/"):]
    _Path(dst).parent.mkdir(parents=True, exist_ok=True)
    t0, last = _time.time(), None
    for _ in range(3):
        try:
            with _ur.urlopen(s, timeout=120) as r, open(dst, "wb") as f:
                _sh.copyfileobj(r, f, 16 * 1024 * 1024)
            last = None
            break
        except (_ue.URLError, OSError, TimeoutError) as e:
            last = e
            _Path(dst).unlink(missing_ok=True)
            _time.sleep(3)
    if last is not None:
        raise RuntimeError(f"stage_local: failed to fetch {src}: {last}")
    log(f"    staged mission local ({_time.time() - t0:.0f}s · "
        f"{_Path(dst).stat().st_size / 1e9:.1f} GB)")
    return dst




# --------------------------------------------------------------------------- #
# the public entry point — same signature as always, current internals
# --------------------------------------------------------------------------- #
def register_survey_by_chips(
    unreg_survey_path: str,
    reg_reference_path: str,
    output_registered_survey_path: str,
    # legacy chip parameters — accepted for compatibility, ignored by the
    # engine (device_for_loftr names the eFOLKI device; "cpu" or None = automatic)
    target_total_chip_width_px: int = 640,
    target_total_chip_height_px: int = 480,
    adaptive_buffer_fractions: List[float] = [0.1, 0.2, 0.3],
    device_for_loftr: str = 'cpu',
    max_loader_workers: int = 4,
    loftr_batch_size: int = 8,
    processing_chunk_size: int = 32,
    loftr_model_name: str = 'outdoor',
    target_size_hw_for_loftr_preprocessing: Tuple[int, int] = (480, 640),
    min_loftr_matches_for_fundamental_matrix: int = 7,
    loftr_reproj_threshold_px_levels_in_resized_space: List[float] = [0.5, 1.0, 2.0],
    ransac_confidence_levels_for_fm: List[float] = [0.999, 0.95],
    min_gcps_for_warp: int = 10,
    gdal_polynomial_order: int = 3,
    gdal_resampling_algorithm: str = "average",
    gdal_src_nodata: Optional[float] = 0,
    gdal_dst_nodata: Optional[float] = 0,
    output_stats_raster_path: Optional[str] = None,
    debug_output_dir_for_warped_chips: Optional[str] = None,
    engine: str = "dense",
    band_names=None,
):
    """Register a survey raster to a reference. Same signature as the
    first-generation function; the internals are the current engine
    (register_survey_dense). engine="chips" is retired — see module docstring."""
    if engine != "dense":
        raise NotImplementedError(
            "the per-chip engine was retired by the refactor (ill-posed camera "
            "model, per-chip edge disagreements, dropped-chip holes — audit "
            "§3.2). It is preserved at tag `v0.1.0-legacy` in git history.")
    return register_survey_dense(
        unreg_survey_path, reg_reference_path, output_registered_survey_path,
        device=None if device_for_loftr in (None, "cpu") else device_for_loftr,
        band_names=band_names,
    )




# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
def _measure_grid(anchor, out_crs, bounds, pad_px):
    """The reference's own pixel lattice over `bounds` (in out_crs) grown by pad_px pixels on every side:
    (transform, width, height, pixel size, same_crs). In the anchor's CRS the lattice is the anchor's own, so
    a read is a pixel-exact crop. In another CRS it is the anchor's pixel size there, on a global lattice of
    that size, and the anchor is resampled once onto it."""
    from rasterio.warp import transform_bounds
    with rasterio.open(anchor) as a:
        same = a.crs == _CRS.from_user_input(out_crs)
        if same:
            mres, ox, oy = abs(a.transform.a), a.transform.c, a.transform.f
        else:
            b = transform_bounds(a.crs, out_crs, *a.bounds, densify_pts=21)
            mres, ox, oy = (b[2] - b[0]) / a.width, 0.0, 0.0
    left, bottom, right, top = bounds
    pad = pad_px * mres
    x0 = ox + _math.floor((left - pad - ox) / mres) * mres
    y1 = oy - _math.floor((oy - top - pad) / mres) * mres
    W = int(_math.ceil((right + pad - x0) / mres))
    H = int(_math.ceil((y1 - bottom + pad) / mres))
    return _Affine.translation(x0, y1) * _Affine.scale(mres, -mres), W, H, mres, same


def _pearson(s):
    """Pearson r from stacked sums (n, sx, sy, sxx, syy, sxy) along axis 0; NaN where undefined."""
    n, sx, sy, sxx, syy, sxy = s
    with np.errstate(invalid="ignore", divide="ignore"):
        return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))


def register_survey_dense(
    unreg_survey_path: str,
    reg_reference_path: str,
    output_registered_survey_path: str,
    resolution_m: float = None,          # the output grid; None = the mission's own pixel size
    device=None,                         # the eFOLKI device (efolki.device); None = CUDA, then MPS, then the CPU
    workers: int = None,                 # GDAL threads compressing the output; None = all CPUs but one
    blocks_at_once: int = None,          # blocks in flight; eFOLKI runs one at a time, the reads and warps in parallel
    qa_json: str = None,
    log=print,
    band_names=None,       # the config's spectral band names (first N non-alpha bands); None = all
):
    """Register one mission to the reference with eFOLKI, warp it once, and score it.

    Measure: the reference's own pixel lattice over the mission (efolki.flow on the grey images: the mean of
    the reference's first three bands, the mean of the mission's first three spectral bands averaged onto
    that lattice, each with its no-data filled from the nearest valid pixel). Blocks of _CORE reference pixels
    are measured with _PAD pixels of context on every side. A block is measured when the mission and the
    reference each cover _MIN_SHARE of it; elsewhere the shift is zero.

    Warp: each output block takes the shift at its pixel centres (bilinear between reference pixels) and
    reads the mission at its native resolution through ONE cubic remap (the native->grid scaling and the
    shift together). The alpha comes from the coarse validity mask at the shifted position, nearest.

    Score: on the reference pixels of each block, Pearson r of the grey images before (the mission as
    averaged onto them) and after (the same image moved by the shift, bilinear), where both have data, and
    the move: the mean shift over a cell, as the distance the mission's content moved, east and north.
    Written per _CELL x _CELL cell to <out>.quality.tif (QUALITY_BANDS, float32, NaN = no data) and pooled over
    the map for the three numbers. Output carries the mission's spectral bands + a tagged alpha band.
    """
    from . import efolki as _E
    t0 = _time.time()
    phases, _tp = {}, [_time.time()]

    def _phase(name):
        now = _time.time(); phases[name] = round(now - _tp[0], 1); _tp[0] = now
    mission, anchor = str(unreg_survey_path), str(reg_reference_path)
    dev = device if hasattr(device, "type") else _E.device(device)

    with rasterio.open(mission) as s:
        out_crs = s.crs
        res = resolution_m or round(abs(s.res[0]), 3)
        mb = s.bounds
        mt = s.transform                            # native transform (same CRS as out_crs)
        mW, mH = s.width, s.height
    # band law (bands.py): spectral bands = the config names (first N non-alpha), validity =
    # the tagged alpha, else derived from zeros. Never guess from the band count.
    law = _B.band_law(mission, band_names)
    if law.alpha is None:                           # DroneDeploy's exports have one; its no-data mask decides data
        raise RuntimeError(f"register_survey_dense: {mission} has no alpha band")
    nspec, spec_bands = law.nspec, list(law.spectral)
    alaw = _B.band_law(anchor)
    if alaw.alpha is None:
        raise RuntimeError(f"register_survey_dense: reference {anchor} has no alpha/validity band")
    a_bands = list(alaw.spectral[:3]) + [alaw.alpha]
    # snap the output origin to a global `res` lattice so every same-GSD mission
    # (and the stitch union grid, likewise snapped) shares one pixel lattice — the
    # merge then crops interiors pixel-exactly instead of re-resampling them.
    ox = _math.floor(mb.left / res) * res
    oy = _math.ceil(mb.top / res) * res
    W = int(np.ceil((mb.right - ox) / res))
    H = int(np.ceil((oy - mb.bottom) / res))
    TR = _Affine.translation(ox, oy) * _Affine.scale(res, -res)
    # grid-pixel -> native-pixel affine (axis-aligned, same CRS): n = s*g + a. Lets
    # the output warp read the mission at NATIVE res and fold the native->grid warp
    # AND the shift into ONE cubic remap (no intermediate grid resample).
    sx = TR.a / mt.a; sy = TR.e / mt.e
    ax = 0.5 * sx + (TR.c - mt.c) / mt.a - 0.5
    ay = 0.5 * sy + (TR.f - mt.f) / mt.e - 0.5
    log(f"[register] {os.path.basename(mission)} -> grid {W}x{H} @ {res} m")
    log(f"    band law: {law.describe()}")

    # ---- where the mission has data, on a coarse grid: its bbox and which blocks are empty ---- #
    Wc, Hc = -(-W // _GEOM_DS), -(-H // _GEOM_DS)
    tr_c = TR * _Affine.scale(_GEOM_DS)
    # DroneDeploy's no-data mask, the alpha band, decides: an output pixel has data only where the native pixel it
    # samples has data (the warp below). On the coarse grid only a bound is kept, every cell holding any native
    # data pixel, read at full resolution, so skipping empty ground can never drop one.
    f = _GEOM_DS
    anyb = np.zeros((-(-mH // f), -(-mW // f)), bool)              # any data in each native f x f block
    with rasterio.open(mission) as s:
        for q0 in range(0, mH, 256 * f):
            hq = min(256 * f, mH - q0)
            al = np.pad(s.read(law.alpha, window=_Window(0, q0, mW, hq)) > 127, ((0, -hq % f), (0, -mW % f)))
            anyb[q0 // f:q0 // f + al.shape[0] // f] = al.reshape(al.shape[0] // f, f, -1, f).any((1, 3))
            del al
    qr, qc = np.nonzero(anyb); del anyb
    bx = mt.c + (qc + 0.5) * f * mt.a; by = mt.f + (qr + 0.5) * f * mt.e      # block centres
    ci = np.floor((bx - TR.c) / (res * f)).astype(np.int64); ri = np.floor((TR.f - by) / (res * f)).astype(np.int64)
    okb = (ri >= 0) & (ri < Hc) & (ci >= 0) & (ci < Wc)
    vbound = np.zeros((Hc, Wc), bool); vbound[ri[okb], ci[okb]] = True
    vbound = _ndi.binary_dilation(vbound, iterations=max(1, int(np.ceil(abs(mt.a) / res))))   # a block spans <= 2 cells a side
    del qr, qc, bx, by, ci, ri, okb
    ys, xs = np.where(vbound)
    if len(ys) == 0:
        raise RuntimeError("register_survey_dense: mission has no valid data")
    r0b = int(ys.min()) * _GEOM_DS; r1b = min(H, (int(ys.max()) + 1) * _GEOM_DS)
    c0b = int(xs.min()) * _GEOM_DS; c1b = min(W, (int(xs.max()) + 1) * _GEOM_DS)
    bh_all, bw_all = r1b - r0b, c1b - c0b

    # ---- the measuring grid: the reference's own pixels over the bbox ------------------ #
    left, top = TR * (c0b, r0b); right, bottom = TR * (c1b, r1b)
    MTR, MW, MH, mres, same = _measure_grid(anchor, out_crs, (left, bottom, right, top), _PAD + 8)
    k = mres / res                                # reference pixels -> output pixels
    ex, ey = MTR.c - TR.c, TR.f - MTR.f           # the measuring origin east and south of the output origin
    jstart = lambda c: int(_math.ceil((c * res - ex) / mres - 0.5))   # first reference column centred at or past output column edge c
    istart = lambda r: int(_math.ceil((r * res - ey) / mres - 0.5))
    OB = max(512, int(round(_CORE * mres / res / 512)) * 512)          # output block whose core is about _CORE reference pixels
    mis_rs = _Resampling.average if abs(mt.a) <= mres * 1.001 else _Resampling.bilinear   # averaged down, interpolated up
    log(f"    measuring on the reference's own pixels: {MW}x{MH} @ {mres * 100:.2f} cm"
        f"{'' if same else ' (reprojected once)'}; eFOLKI on {dev.type}, radii "
        f"{'-'.join(str(r) for r in _E.SETTINGS['radius'])}, {_E.SETTINGS['levels']} levels; "
        f"output blocks {OB} px")
    HC, WC = -(-MH // _CELL), -(-MW // _CELL)
    acc = np.zeros((15, HC, WC), np.float64)      # before (6 sums), after (6 sums), move (n, su, sv)
    alock, wlock = _threading.Lock(), _threading.Lock()
    tls = _threading.local()
    counts = dict(measured=0, unmeasured=0, empty=0)
    secs = dict(measure=0.0, warp=0.0)
    _phase("geometry")

    tr_b = TR * _Affine.translation(c0b, r0b)
    prof = dict(driver="GTiff", height=bh_all, width=bw_all, count=nspec + 1, dtype="uint8",
                crs=out_crs, transform=tr_b, tiled=True, blockxsize=512, blockysize=512,
                compress="zstd", predictor=2, ZSTD_LEVEL=3, BIGTIFF="IF_SAFER",
                num_threads=str(workers or max(1, (os.cpu_count() or 4) - 1)))
    _Path(output_registered_survey_path).parent.mkdir(parents=True, exist_ok=True)
    dst = rasterio.open(output_registered_survey_path, "w", **prof)
    _B.tag_output(dst, law.names)                  # names + tagged alpha: no guessing downstream

    def _readers():
        if not hasattr(tls, "a"):
            tls.a = _WarpedVRT(rasterio.open(anchor), crs=out_crs, transform=MTR, width=MW, height=MH,
                               resampling=_Resampling.average)
            tls.m = _WarpedVRT(rasterio.open(mission), crs=out_crs, transform=MTR, width=MW, height=MH,
                               resampling=mis_rs)
            tls.ma = _WarpedVRT(rasterio.open(mission), crs=out_crs, transform=MTR, width=MW, height=MH,
                                resampling=_Resampling.nearest)
            tls.n = rasterio.open(mission)                       # native read for the warp
        return tls

    def _mission_valid(rd, win, wi0, wj0, h, w):
        """The mission's validity on the measuring window: its alpha, read nearest."""
        return rd.ma.read(law.alpha, window=win) > 127

    def one(blk):
        r0, c0 = blk
        bh, bw = min(OB, r1b - r0), min(OB, c1b - c0)
        out_win = _Window(c0 - c0b, r0 - r0b, bw, bh)
        if not vbound[r0 // _GEOM_DS:-(-(r0 + bh) // _GEOM_DS), c0 // _GEOM_DS:-(-(c0 + bw) // _GEOM_DS)].any():
            with wlock:
                dst.write(np.zeros((nspec + 1, bh, bw), np.uint8), window=out_win)
                counts["empty"] += 1
            _HB.beat()
            return
        rd = _readers()
        t = _time.time()
        # the block's core on the measuring grid (the reference pixels centred in this output block) + context
        j0, j1, i0, i1 = jstart(c0), jstart(c0 + bw), istart(r0), istart(r0 + bh)
        wj0, wj1 = max(0, j0 - _PAD), min(MW, j1 + _PAD)
        wi0, wi1 = max(0, i0 - _PAD), min(MH, i1 + _PAD)
        h, w = wi1 - wi0, wj1 - wj0
        win = _Window(wj0, wi0, w, h)
        A = rd.a.read(a_bands, window=win)
        gr = A[:-1].astype(np.float32).mean(0); vr = A[-1] > 127
        del A
        gm = rd.m.read(spec_bands[:3], window=win).astype(np.float32).mean(0)
        vm = _mission_valid(rd, win, wi0, wj0, h, w)
        cs = np.s_[i0 - wi0:i1 - wi0, j0 - wj0:j1 - wj0]
        measured = vm[cs].mean() >= _MIN_SHARE and vr[cs].mean() >= _MIN_SHARE
        if measured:
            Gu, Gv = _E.flow(_E.fill_nearest(gr, vr), _E.fill_nearest(gm, vm), dev=dev)
            # a shift is measured only where eFOLKI's largest window lies on ground in both maps: the intersection
            # of the two data masks with a negative buffer of the largest radius (32 px, 1 m on a 3.2 cm map).
            # Every other pixel takes the nearest measured shift, so a map's edge moves as one piece with the
            # ground beside it. Beyond the edge the solve matched fill, not ground, and placed the edge about
            # 4 px wrong (02_registration/matcher_ablation/poc/edge_known_warp.py, 2026-11-01). 0.4.2.
            trusted = _ndi.distance_transform_edt(vr & vm) > max(_E.SETTINGS["radius"])
            if trusted.any():
                if not trusted.all():
                    near = _ndi.distance_transform_edt(~trusted, return_distances=False, return_indices=True)
                    Gu = Gu[near[0], near[1]]; Gv = Gv[near[0], near[1]]
                    del near
            else:                                                    # the maps share less than one window: no measurement
                Gu = np.zeros((h, w), np.float32); Gv = np.zeros((h, w), np.float32)
            del trusted
        else:
            Gu = np.zeros((h, w), np.float32); Gv = np.zeros((h, w), np.float32)
        t_meas = _time.time() - t; t = _time.time()

        # ---- score the core: before, after, and the move, summed per cell -------------- #
        ch, cw = i1 - i0, j1 - j0
        if measured and ch > 0 and cw > 0:
            yy, xx = np.mgrid[i0 - wi0:i1 - wi0, j0 - wj0:j1 - wj0].astype(np.float32)
            gu, gv = Gu[cs], Gv[cs]
            after = cv2.remap(gm, xx + gu, yy + gv, cv2.INTER_LINEAR)
            v_after = cv2.remap(vm.astype(np.uint8), xx + gu, yy + gv, cv2.INTER_NEAREST) > 0
            del xx, yy
            ci0, cj0 = i0 // _CELL, j0 // _CELL
            nhc, nwc = (i1 - 1) // _CELL - ci0 + 1, (j1 - 1) // _CELL - cj0 + 1
            cid = ((np.arange(i0, i1) // _CELL - ci0)[:, None] * nwc + (np.arange(j0, j1) // _CELL - cj0)[None, :])
            sums = np.zeros((15, nhc * nwc), np.float64)
            g_r = gr[cs]
            for base, x, ok in ((0, gm[cs], vm[cs] & vr[cs]), (6, after, v_after & vr[cs])):
                ids = cid[ok]; xv = x[ok].astype(np.float64); yv = g_r[ok].astype(np.float64)
                for q, wts in enumerate((None, xv, yv, xv * xv, yv * yv, xv * yv)):
                    sums[base + q] = np.bincount(ids, weights=wts, minlength=nhc * nwc)
            ok = vm[cs] & vr[cs]; ids = cid[ok]
            sums[12] = np.bincount(ids, minlength=nhc * nwc)
            sums[13] = np.bincount(ids, weights=gu[ok].astype(np.float64), minlength=nhc * nwc)
            sums[14] = np.bincount(ids, weights=gv[ok].astype(np.float64), minlength=nhc * nwc)
            with alock:
                acc[:, ci0:ci0 + nhc, cj0:cj0 + nwc] += sums.reshape(15, nhc, nwc)
            del after, v_after, cid, sums, g_r, gu, gv

        # ---- warp the output block once: native pixels -> grid, shifted ----------------- #
        rr = np.arange(r0, r0 + bh, dtype=np.float64); cc = np.arange(c0, c0 + bw, dtype=np.float64)
        my = (((rr + 0.5) * res - ey) / mres - 0.5 - wi0).astype(np.float32)
        mx = (((cc + 0.5) * res - ex) / mres - 0.5 - wj0).astype(np.float32)
        MX, MY = np.meshgrid(mx, my)
        fu = cv2.remap(Gu, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fv = cv2.remap(Gv, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        del MX, MY, Gu, Gv, gr, gm, vr, vm
        gxs = cc.astype(np.float32)[None, :] + fu * np.float32(k)   # the source position, output pixels
        gys = rr.astype(np.float32)[:, None] + fv * np.float32(k)
        measured_px = fu * fu + fv * fv <= np.float32(_E.REACH_PX) ** 2   # no shift past eFOLKI's reach was measured
        del fu, fv
        nx = gxs * np.float32(sx) + np.float32(ax)                  # ... and in native pixels
        ny = gys * np.float32(sy) + np.float32(ay)
        # the pixels the output keeps: a shift within eFOLKI's reach, to a position on the mission. The rest are
        # transparent. Where the mission has no data eFOLKI's shift means nothing (it reached ±30000 px on 2025 spring
        # mission 00, and a few such pixels landed on the mission and pulled in ground from up to 22 m away), so they
        # neither size the native read nor get sampled. Sized by them, the read passed cv2.remap's 32767 px limit and
        # the job died on SIGSEGV (0.4.2).
        cy = np.floor((gys + 0.5) / _GEOM_DS).astype(np.int64); cx = np.floor((gxs + 0.5) / _GEOM_DS).astype(np.int64)
        keep = measured_px & (cy >= 0) & (cy < Hc) & (cx >= 0) & (cx < Wc)
        keep &= vbound[np.clip(cy, 0, Hc - 1), np.clip(cx, 0, Wc - 1)]                # the bound; with alpha, below decides
        del cy, cx, measured_px
        out = np.zeros((nspec + 1, bh, bw), np.uint8)
        P = 3                                                        # the cubic kernel's reach, padded
        if keep.any():
            kx, ky = nx[keep], ny[keep]
            nc0 = max(0, int(_math.floor(float(kx.min()))) - P); nc1 = min(mW, int(_math.ceil(float(kx.max()))) + P + 1)
            nr0 = max(0, int(_math.floor(float(ky.min()))) - P); nr1 = min(mH, int(_math.ceil(float(ky.max()))) + P + 1)
            del kx, ky
        else:
            nc0 = nc1 = nr0 = nr1 = 0
        if nc1 > nc0 and nr1 > nr0:
            nwin = _Window(nc0, nr0, nc1 - nc0, nr1 - nr0)
            nat = rd.n.read(spec_bands, window=nwin).astype(np.float32)
            nx -= np.float32(nc0); ny -= np.float32(nr0)
            na = rd.n.read(law.alpha, window=nwin)                   # data where the sampled native pixel has data
            keep &= cv2.remap(na, nx, ny, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
            del na
            nx[~keep] = np.float32(-2 * P); ny[~keep] = np.float32(-2 * P)   # off the source: the border value, 0
            for b in range(nspec):
                rem = cv2.remap(nat[b], nx, ny, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                np.clip(rem, 0, 255, out=rem)                        # cubic can overshoot [0,255]
                out[b] = np.rint(rem)                                # rounded: a bare cast truncates, half a DN dark
            del nat, rem
        else:
            keep[:] = False                                          # nothing of the mission to sample
        out[nspec][keep] = 255
        del gxs, gys, nx, ny, keep
        with wlock:
            dst.write(out, window=out_win)
            counts["measured" if measured else "unmeasured"] += 1
            secs["measure"] += t_meas; secs["warp"] += _time.time() - t
        _HB.beat()                                                   # one heartbeat per block

    blocks = [(r0, c0) for r0 in range(r0b, r1b, OB) for c0 in range(c0b, c1b, OB)]
    at_once = max(1, int(blocks_at_once or os.environ.get("TIDYSURVEY_BLOCKS_AT_ONCE") or _BLOCKS_AT_ONCE))
    log(f"    {len(blocks)} blocks, {at_once} at once")
    t_last = [_time.time()]
    with _TPE(max_workers=at_once) as ex_:
        for n, _ in enumerate(ex_.map(one, blocks), 1):
            if _time.time() - t_last[0] >= 60 or n == len(blocks):
                t_last[0] = _time.time()
                log(f"      [{n}/{len(blocks)}] blocks done ({_time.time() - t0:.0f}s)")
    dst.close()
    _phase("register")

    # ---- the quality raster and the three numbers ------------------------------------ #
    min_n = _MIN_CELL * _CELL * _CELL
    rb_c = np.where(acc[0] >= min_n, _pearson(acc[0:6]), np.nan)
    ra_c = np.where(acc[6] >= min_n, _pearson(acc[6:12]), np.nan)
    mapped = acc[12] >= min_n
    with np.errstate(invalid="ignore", divide="ignore"):
        east = np.where(mapped, -acc[13] / acc[12] * mres * 100, np.nan)
        north = np.where(mapped, acc[14] / acc[12] * mres * 100, np.nan)
    move = np.hypot(east, north)
    if not mapped.any():
        raise RuntimeError("register_survey_dense: the reference covers none of the mission")
    r_before = float(_pearson(acc[0:6].sum(axis=(1, 2))))
    r_after = float(_pearson(acc[6:12].sum(axis=(1, 2))))
    cell_ha = (_CELL * mres) ** 2 / 1e4
    mv = move[mapped]
    big = mv > BIG_MOVE_CM
    moved_ha, moved_pct = float(big.sum() * cell_ha), float(100 * big.mean())
    move_med = float(np.median(mv))
    look = moved_pct > LOOK_PCT or not r_after > r_before
    line = (f"{'⚠ look' if look else '✓'} r {r_before:.2f} → {r_after:.2f} · median move {move_med:.0f} cm · "
            f">50 cm {moved_ha:.2f} ha ({moved_pct:.2f} %) · measured at {mres * 100:.1f} cm, "
            f"the reference's pixels; registered map at {res * 100:.1f} cm")
    quality = str(_Path(output_registered_survey_path).with_suffix(".quality.tif"))
    with rasterio.open(quality, "w", driver="GTiff", width=WC, height=HC, count=len(QUALITY_BANDS),
                       dtype="float32", crs=out_crs, transform=MTR * _Affine.scale(_CELL), nodata=float("nan"),
                       tiled=True, compress="zstd") as q:
        for i, nm in enumerate(QUALITY_BANDS, 1):          # tags BEFORE pixels (bands.py)
            q.set_band_description(i, nm)
        q.update_tags(method="eFOLKI", mission=os.path.basename(mission), reference=os.path.basename(anchor),
                      measured_res_m=round(mres, 5), cell_px=_CELL, cell_m=round(_CELL * mres, 3),
                      r_before=round(r_before, 4), r_after=round(r_after, 4), move_median_cm=round(move_med, 1),
                      moved_over_50cm_ha=round(moved_ha, 3), moved_over_50cm_pct=round(moved_pct, 3),
                      look=str(look).lower(),
                      bands="r_before, r_after: Pearson r of the grey images in the cell before and after; "
                            "move_cm, move_east_cm, move_north_cm: how far the mission's content moved")
        for i, a in enumerate((rb_c, ra_c, move, east, north), 1):
            q.write(a.astype(np.float32), i)
    _phase("quality")
    log(f"    {line}")
    log("    phases: " + ", ".join(f"{k_} {v}s" for k_, v in phases.items())
        + f" (eFOLKI and reads {secs['measure']:.0f}s, scoring and warp {secs['warp']:.0f}s, summed over blocks)")

    summary = dict(kind="align", method="eFOLKI", settings=dict(_E.SETTINGS, block_px=_CORE, context_px=_PAD),
                   mission=os.path.basename(mission), reference=os.path.basename(anchor),
                   bands=list(law.names), alpha="tagged",
                   measured_res_m=round(mres, 5), res_m=res, cell_m=round(_CELL * mres, 3),
                   r_before=round(r_before, 4), r_after=round(r_after, 4), move_median_cm=round(move_med, 1),
                   moved_over_50cm_ha=round(moved_ha, 3), moved_over_50cm_pct=round(moved_pct, 3),
                   mapped_ha=round(float(mapped.sum() * cell_ha), 2), look=bool(look), line=line,
                   out=str(output_registered_survey_path), quality_tif=quality, quality_bands=list(QUALITY_BANDS),
                   blocks=dict(counts), blocks_at_once=at_once, device=dev.type, phases=phases,
                   block_seconds={k_: round(v, 1) for k_, v in secs.items()},
                   seconds=round(_time.time() - t0, 1))
    if qa_json:
        _Path(qa_json).parent.mkdir(parents=True, exist_ok=True)
        _Path(qa_json).write_text(_json.dumps(summary, indent=2))
    log(f"    registered -> {output_registered_survey_path} ({summary['seconds']:.0f}s)")
    return summary

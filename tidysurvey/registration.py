"""Alignment: move each mission toward the ground reference with ONE smooth,
corroborated correction field (the audited dense engine).

The first-generation per-chip machinery (Chip loaders, per-chip LoFTR + GDAL
polynomial warps, chip merging) was retired by the refactor — the audit found
the camera-geometry model ill-posed for flat maps, independent per-chip warps
disagreeing at their edges, and dropped chips leaving holes (production_
refactor.md §3.2). It lives on branch `main` / git history. The public entry
point `register_survey_by_chips` keeps its exact signature and now runs the
dense internals (agreement with the anchor 0.685 -> 0.863 on the 2024 survey,
98.6% of cells improved).

Ported from the audit repo: 02_registration/poc/m2_full.py.
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
from rasterio.vrt import WarpedVRT as _WarpedVRT
from rasterio.windows import Window as _Window
from rasterio.enums import Resampling as _Resampling
from affine import Affine as _Affine
from scipy.ndimage import label as _ndlabel

from . import fields as _F

_TILE_H, _TILE_W = 480, 640
_FS = 64                     # pooling cell (px)
_MIN_NB = 8                  # corroboration: neighbours required in the 5x5 ring
_REJECT_PX, _MIN_TILE, _MIN_CELL = 4.0, 8, 2
_GEOM_DS, _FIELD_DS, _BLOCK = 8, 8, 2048


# --------------------------------------------------------------------------- #
# the public entry point — same signature as always, dense internals
# --------------------------------------------------------------------------- #
def register_survey_by_chips(
    unreg_survey_path: str,
    reg_reference_path: str,
    output_registered_survey_path: str,
    # legacy chip parameters — accepted for compatibility, ignored by the
    # dense engine (device_for_loftr and max_loader_workers are honoured)
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
):
    """Register a survey raster to a reference. Same signature as the
    first-generation function; the INTERNALS are the audited dense engine
    (one smooth corroborated field per mission, warped once toward the
    reference). engine="chips" is retired — see module docstring."""
    if engine != "dense":
        raise NotImplementedError(
            "the per-chip engine was retired by the refactor (ill-posed camera "
            "model, per-chip edge disagreements, dropped-chip holes — audit "
            "§3.2). It is preserved on branch `main` / git history.")
    return register_survey_dense(
        unreg_survey_path, reg_reference_path, output_registered_survey_path,
        device=None if device_for_loftr in (None, "cpu") else device_for_loftr,
        workers=max_loader_workers,
    )


# --------------------------------------------------------------------------- #
# the dense engine
# --------------------------------------------------------------------------- #
def register_survey_dense(
    unreg_survey_path: str,
    reg_reference_path: str,
    output_registered_survey_path: str,
    resolution_m: float = None,          # None = the mission's native GSD
    device=None,
    workers: int = None,
    qa_json: str = None,
    log=print,
):
    """Align one mission to the reference with ONE smooth correction field.

    Dense LoFTR pass over the mission footprint against the reference
    (anchored gauge — the reference is the truth), per-tile robust rejection,
    per-cell pooled medians, 5x5-ring corroboration gate + boundary-cell
    filter, then a single smooth field warps the whole mission once. No
    camera-geometry model, no per-chip polynomials, no dropped-chip holes.
    Output carries the mission's bands + a real alpha band.

    Note: torch's MPS allocator grows across long match passes; the cache is
    released every 100 tiles and fully at the end of the pass
    (fields.release_matcher_cache). The audited runner's stronger remedy —
    LoFTR in chunked subprocesses (m2_full.py telemetry, which also dodged an
    MPS per-process slowdown) — remains the fallback if a mission's footprint
    still climbs or throughput decays: chunk externally.
    """
    t0 = _time.time()
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    mission, anchor = str(unreg_survey_path), str(reg_reference_path)

    with rasterio.open(mission) as s:
        out_crs = str(s.crs)
        res = resolution_m or round(abs(s.res[0]), 3)
        n_bands = s.count
        mb = s.bounds
    nspec = max(1, n_bands - 1) if n_bands in (4, 5) else n_bands   # last band = alpha if RGBA/5-band
    W = int(np.ceil((mb.right - mb.left) / res))
    H = int(np.ceil((mb.top - mb.bottom) / res))
    TR = _Affine.translation(mb.left, mb.top) * _Affine.scale(res, -res)
    log(f"[register_dense] {os.path.basename(mission)} -> grid {W}x{H} @ {res} m")

    # ---- coarse validity (mission) + reference coverage -------------------- #
    Wc, Hc = -(-W // _GEOM_DS), -(-H // _GEOM_DS)
    tr_c = TR * _Affine.scale(_GEOM_DS)
    with rasterio.open(mission) as s:
        with _WarpedVRT(s, crs=out_crs, transform=tr_c, width=Wc, height=Hc,
                        resampling=_Resampling.average) as v:
            arr = v.read()
    data = (arr != 0).any(0)
    del arr
    vmask = data
    zw = ~data
    if zw.any():                                # fill enclosed holes (interior nodata)
        lab, _ = _ndlabel(zw, structure=np.ones((3, 3), int))
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        vmask = data | ((lab > 0) & ~np.isin(lab, border[border > 0]))
    with rasterio.open(anchor) as a:
        a_alpha = a.count if a.count in (4, 5) else a.count
        with _WarpedVRT(a, crs=out_crs, transform=tr_c, width=Wc, height=Hc,
                        resampling=_Resampling.average) as v:
            a_ok_c = v.read(a_alpha) > 127

    ys, xs = np.where(vmask)
    if len(ys) == 0:
        raise RuntimeError("register_survey_dense: mission has no valid data")
    r0b = int(ys.min()) * _GEOM_DS; r1b = min(H, (int(ys.max()) + 1) * _GEOM_DS)
    c0b = int(xs.min()) * _GEOM_DS; c1b = min(W, (int(xs.max()) + 1) * _GEOM_DS)
    tiles = []
    for r0 in range(r0b, r1b - _TILE_H + 1, _TILE_H):
        for c0 in range(c0b, c1b - _TILE_W + 1, _TILE_W):
            cr, cc = r0 // _GEOM_DS, c0 // _GEOM_DS
            ch, cw = _TILE_H // _GEOM_DS, _TILE_W // _GEOM_DS
            if (vmask[cr:cr + ch, cc:cc + cw].mean() > 0.05
                    and a_ok_c[cr:cr + ch, cc:cc + cw].mean() > 0.05):
                tiles.append((r0, c0))
    log(f"    {len(tiles)} data tiles (bbox {r1b - r0b}x{c1b - c0b} px)")

    # ---- dense match pass --------------------------------------------------- #
    matcher, dev = _F.build_matcher(device)
    vrt_m = _WarpedVRT(rasterio.open(mission), crs=out_crs, transform=TR,
                       width=W, height=H, resampling=_Resampling.bilinear)
    vrt_a = _WarpedVRT(rasterio.open(anchor), crs=out_crs, transform=TR,
                       width=W, height=H, resampling=_Resampling.average)
    buf_an, buf_d, n_used = [], [], 0
    for n_run, (r0, c0) in enumerate(tiles, 1):
        win = _Window(c0, r0, _TILE_W, _TILE_H)
        spec = vrt_m.read(list(range(1, min(3, nspec) + 1)), window=win)
        if not (spec != 0).any():
            continue
        aal = vrt_a.read(a_alpha, window=win)
        if not (aal > 127).any():
            continue
        argb = vrt_a.read([1, 2, 3], window=win)
        m = _F.match_tile(matcher, dev,
                          _F.gray_stretch(spec.astype(np.float32)),
                          _F.gray_stretch(argb.astype(np.float32)),
                          _REJECT_PX, _MIN_TILE)
        if m is None:
            continue
        k0, d = m
        buf_an.append(k0 + [c0, r0]); buf_d.append(d); n_used += 1
        if n_run % 100 == 0:
            _F.release_matcher_cache(dev)   # cap MPS allocator growth
        if n_run % 250 == 0:
            log(f"    [{n_run}/{len(tiles)}] tiles, {n_used} with matches")
    vrt_m.close(); vrt_a.close()
    del matcher
    _F.release_matcher_cache(dev)   # match pass done; solve + warp run on CPU
    if not buf_an:
        raise RuntimeError("register_survey_dense: no matches")
    an = np.concatenate(buf_an); d = np.concatenate(buf_d)

    # ---- cells -> corroboration -> boundary filter -------------------------- #
    pts, vx, vy = _F.pool_to_cells(an, d, _FS, _MIN_CELL)
    keep = _F.corroborate_cells(pts, vx, vy, _FS, _MIN_NB, _REJECT_PX)
    cell_f = _FS // _GEOM_DS                       # boundary: cell + ring valid in BOTH
    Hcc, Wcc = Hc // cell_f, Wc // cell_f
    vm = vmask[:Hcc * cell_f, :Wcc * cell_f].reshape(Hcc, cell_f, Wcc, cell_f).min(axis=(1, 3))
    am = a_ok_c[:Hcc * cell_f, :Wcc * cell_f].reshape(Hcc, cell_f, Wcc, cell_f).min(axis=(1, 3))
    ok = cv2.erode((vm & am).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ccy = np.clip((pts[:, 1] // _FS).astype(int), 0, Hcc - 1)
    ccx = np.clip((pts[:, 0] // _FS).astype(int), 0, Wcc - 1)
    keep &= ok[ccy, ccx]
    n_rej = int((~keep).sum())
    pts, vx, vy = pts[keep], vx[keep], vy[keep]
    if len(pts) == 0:
        raise RuntimeError("register_survey_dense: all cells rejected by the gates")
    d_med = float(np.median(np.hypot(vx, vy)) * res * 100)
    log(f"    {len(pts)} cells after gates ({n_rej} rejected)  |d| med={d_med:.1f} cm")

    # ---- one smooth field (bbox-local, FIELD_DS lattice) --------------------- #
    bh, bw = r1b - r0b, c1b - c0b
    Hg, Wg = bh // _FIELD_DS + 1, bw // _FIELD_DS + 1
    pl = (pts - [c0b, r0b]) / _FIELD_DS
    Fx = _F.fill_field(pl, vx, Wg, Hg)
    Fy = _F.fill_field(pl, vy, Wg, Hg)
    maxF = float(max(np.abs(Fx).max(), np.abs(Fy).max()))
    halo = int(_math.ceil(maxF)) + 16

    # ---- warp once (block-streamed, haloed) ---------------------------------- #
    tr_b = TR * _Affine.translation(c0b, r0b)
    prof = dict(driver="GTiff", height=bh, width=bw, count=nspec + 1, dtype="uint8",
                crs=out_crs, transform=tr_b, tiled=True, blockxsize=512, blockysize=512,
                compress="zstd", predictor=2, ZSTD_LEVEL=3, BIGTIFF="IF_SAFER",
                num_threads=str(workers))
    vc = vmask[r0b // _GEOM_DS:-(-r1b // _GEOM_DS), c0b // _GEOM_DS:-(-c1b // _GEOM_DS)]
    vfull = cv2.resize(vc.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST)
    _Path(output_registered_survey_path).parent.mkdir(parents=True, exist_ok=True)
    dst = rasterio.open(output_registered_survey_path, "w", **prof)
    wlock = _threading.Lock()
    tls = _threading.local()

    def upf(arr, hr0, hc0, hh, hw):
        return _F.upsample_block(arr, _FIELD_DS, hr0 - r0b, hc0 - c0b, hh, hw)

    def wone(bc):
        r0, c0 = bc
        if not hasattr(tls, "v"):
            tls.v = _WarpedVRT(rasterio.open(mission), crs=out_crs, transform=TR,
                               width=W, height=H, resampling=_Resampling.bilinear)
        bh2, bw2 = min(_BLOCK, r1b - r0), min(_BLOCK, c1b - c0)
        hr0, hc0 = max(r0b, r0 - halo), max(c0b, c0 - halo)
        hr1, hc1 = min(r1b, r0 + bh2 + halo), min(c1b, c0 + bw2 + halo)
        hh, hw = hr1 - hr0, hc1 - hc0
        iy, ix = r0 - hr0, c0 - hc0
        spec = tls.v.read(list(range(1, nspec + 1)),
                          window=_Window(hc0, hr0, hw, hh)).astype(np.float32)
        val = vfull[hr0 - r0b:hr1 - r0b, hc0 - c0b:hc1 - c0b]
        fxs = upf(Fx, hr0, hc0, hh, hw); fys = upf(Fy, hr0, hc0, hh, hw)
        gx, gy = np.meshgrid(np.arange(hw, dtype=np.float32),
                             np.arange(hh, dtype=np.float32))
        mapx, mapy = gx - fxs, gy - fys                     # inverse map: pull from source
        outb = np.zeros((nspec + 1, hh, hw), np.uint8)
        for b in range(nspec):
            outb[b] = cv2.remap(spec[b], mapx, mapy, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.uint8)
        outb[nspec] = (cv2.remap(val, mapx, mapy, cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
                       ).astype(np.uint8) * 255
        with wlock:
            dst.write(outb[:, iy:iy + bh2, ix:ix + bw2],
                      window=_Window(c0 - c0b, r0 - r0b, bw2, bh2))

    wblocks = [(r0, c0) for r0 in range(r0b, r1b, _BLOCK) for c0 in range(c0b, c1b, _BLOCK)]
    with _TPE(max_workers=workers) as ex:
        list(ex.map(wone, wblocks))
    dst.close()

    summary = dict(kind="align", mission=os.path.basename(mission),
                   reference=os.path.basename(anchor), tiles=len(tiles),
                   matches=int(len(an)), cells=int(len(pts)), rejected=n_rej,
                   d_med_cm=round(d_med, 1),
                   d_p99_cm=round(float(np.percentile(np.hypot(vx, vy), 99)) * res * 100, 1),
                   maxF_cm=round(maxF * res * 100, 1),
                   out=str(output_registered_survey_path),
                   seconds=round(_time.time() - t0, 1))
    if qa_json:
        _Path(qa_json).parent.mkdir(parents=True, exist_ok=True)
        _Path(qa_json).write_text(_json.dumps(summary, indent=2))
    log(f"    registered -> {output_registered_survey_path} ({summary['seconds']:.0f}s)")
    return summary

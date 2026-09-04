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
import subprocess as _subprocess
import shutil as _shutil
import sys as _sys
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
from . import fields as _F
from . import bands as _B

_TILE_H, _TILE_W = 480, 640
_FS = 64                     # pooling cell (px)
_MIN_NB = 8                  # corroboration: neighbours required in the 5x5 ring
_REJECT_PX, _MIN_TILE, _MIN_CELL = 4.0, 8, 2
_GEOM_DS, _FIELD_DS, _BLOCK = 8, 8, 2048

# --------------------------------------------------------------------------- #
# dense match pass — run as fresh subprocess CHUNKS (audit 02_registration §7).
# LoFTR on MPS decays with varied-shape calls in a long-lived process
# (496 -> 1829 ms/tile; in-loop empty_cache accelerates it). A fresh process
# per chunk resets the allocator -> flat throughput. ONLY these workers touch
# the GPU; register_survey_dense (the parent) stays on the CPU.
_CHUNK_TILES = int(os.environ.get("TIDYSURVEY_REG_CHUNK", "250"))


class _FakeMatcher:                                     # test hook only
    """TIDYSURVEY_REG_FAKE=1: a fixed grid of correspondences with a constant
    sub-pixel shift, to exercise chunk orchestration without loading LoFTR."""
    def __call__(self, d):
        import torch
        h, w = d["image0"].shape[-2:]
        ys, xs = np.mgrid[30:h - 30:40, 30:w - 30:40]
        kp = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
        return {"keypoints0": torch.from_numpy(kp),
                "keypoints1": torch.from_numpy(kp + np.float32([0.6, -0.4]))}


def _worker_matcher():
    if os.environ.get("TIDYSURVEY_REG_FAKE"):
        import torch
        return _FakeMatcher(), torch.device("cpu")
    return _F.build_matcher()


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
    when the anchor's CRS matches this stage is a pixel-exact CROP (no resample):
    the anchor is then resampled EXACTLY ONCE — at the per-mission VRT read onto
    the mission grid — the same single warp the seam walk applies to every
    source. Shared across the align pass; cli._align removes it afterwards."""
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


def _run_reg_chunk(work_dir, chunk_idx):
    """Match ONE chunk of the tile list in this (fresh) process, write
    part_<idx>.npz, exit. Invoked via `python -m tidysurvey._regchunk`. The
    process exit is the point — it frees the MPS allocator. No in-loop
    empty_cache; preallocated, reused device buffers."""
    import torch
    work = _Path(work_dir); ci = int(chunk_idx)
    st = _json.loads((work / "state.json").read_text())
    tiles = np.load(work / "tiles.npy")
    sub = tiles[ci * _CHUNK_TILES:(ci + 1) * _CHUNK_TILES]
    TR = _Affine(*st["tr"]); W, H, nspec = st["W"], st["H"], st["nspec"]
    th, tw, a_alpha = st["tile_h"], st["tile_w"], st["a_alpha"]
    reject, mintile = st["reject_px"], st["min_tile"]
    matcher, dev = _worker_matcher()
    ta = torch.empty((1, 1, th, tw), device=dev)        # preallocated, reused
    tb = torch.empty((1, 1, th, tw), device=dev)
    # mission read from its source (GCS) per tile; anchor read from the shared
    # union pre-warp (local) so it is NOT re-fetched from the NAS per mission.
    vm = _WarpedVRT(rasterio.open(st["mission"]), crs=st["out_crs"], transform=TR,
                    width=W, height=H, resampling=_Resampling.bilinear)
    va = _WarpedVRT(rasterio.open(st["anchor"]), crs=st["out_crs"], transform=TR,
                    width=W, height=H, resampling=_Resampling.average)
    ans, dss, n_used, rdm, rda, inf = [], [], 0, 0.0, 0.0, 0.0
    for (r0, c0) in sub:
        r0, c0 = int(r0), int(c0)
        win = _Window(c0, r0, tw, th)
        t = _time.time()
        spec = vm.read(st["spec_bands"][:3], window=win)     # first (<=3) spectral bands
        rdm += _time.time() - t
        if not (spec != 0).any():
            continue
        t = _time.time()
        aal = va.read(a_alpha, window=win)
        ok = (aal > 127).any()
        if ok:
            argb = va.read([1, 2, 3], window=win)
        rda += _time.time() - t
        if not ok:
            continue
        t = _time.time()
        m = _F.match_tile_pre(matcher, dev, ta, tb,
                              _F.gray_stretch(spec.astype(np.float32)),
                              _F.gray_stretch(argb.astype(np.float32)),
                              reject, mintile)
        inf += _time.time() - t
        if m is None:
            continue
        k0, dd = m
        ans.append(k0 + [c0, r0]); dss.append(dd); n_used += 1
    vm.close(); va.close()
    print(f"    chunk {ci}: {n_used}/{len(sub)} · mission-read {rdm:.0f}s · "
          f"anchor-read {rda:.0f}s · infer {inf:.0f}s", flush=True)
    part = work / f"part_{ci}.npz"
    if ans:
        np.savez(part, an=np.concatenate(ans), d=np.concatenate(dss),
                 n_used=np.int64(n_used))
    else:
        np.savez(part, an=np.zeros((0, 2)), d=np.zeros((0, 2)), n_used=np.int64(0))


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
    band_names=None,
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
        workers=max_loader_workers, band_names=band_names,
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
    band_names=None,       # the config's spectral band names (first N non-alpha bands); None = all
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
        mb = s.bounds
        mt = s.transform                            # native transform (same CRS as out_crs)
        mW, mH = s.width, s.height
    # band law (bands.py): spectral bands = the config names (first N non-alpha), validity =
    # the tagged alpha, else derived from zeros. Never guess from the band count.
    law = _B.band_law(mission, band_names)
    nspec, spec_bands = law.nspec, list(law.spectral)
    alaw = _B.band_law(anchor)
    if alaw.alpha is None:
        raise RuntimeError(f"register_survey_dense: reference {anchor} has no alpha/validity band")
    a_alpha = alaw.alpha
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
    # AND the correction field into ONE cubic remap (no intermediate grid resample).
    sx = TR.a / mt.a; sy = TR.e / mt.e
    ax = 0.5 * sx + (TR.c - mt.c) / mt.a - 0.5
    ay = 0.5 * sy + (TR.f - mt.f) / mt.e - 0.5
    log(f"[register_dense] {os.path.basename(mission)} -> grid {W}x{H} @ {res} m")
    log(f"    band law: {law.describe()}")

    # ---- coarse validity (mission) + reference coverage -------------------- #
    Wc, Hc = -(-W // _GEOM_DS), -(-H // _GEOM_DS)
    tr_c = TR * _Affine.scale(_GEOM_DS)
    with rasterio.open(mission) as s:
        with _WarpedVRT(s, crs=out_crs, transform=tr_c, width=Wc, height=Hc,
                        resampling=_Resampling.average) as v:
            vmask = _B.read_valid(v, law)       # exact alpha, else zeros + border flood fill
    with rasterio.open(anchor) as a:
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

    # ---- dense match pass: fresh subprocess per CHUNK ----------------------- #
    # GPU work runs ONLY in short-lived chunk workers so the MPS per-process
    # inference decay (audit 02_registration §7: 496->1829 ms/tile; in-loop
    # empty_cache accelerates it) can't accumulate across a mission. This parent
    # never builds a matcher — geometry, field fit and warp are all CPU. Results
    # are identical to a single pass (same tiles, same deterministic matcher).
    work = _Path(str(output_registered_survey_path) + ".regwork")
    _shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    # NOTE: `anchor` is expected to be the shared UNION pre-warp (local COG) built
    # once by cli._align for borrowed-anchor seasons — the workers read it locally
    # instead of re-fetching the NAS anchor per tile per mission. The mission is
    # read from its own source (GCS) per tile, NOT downloaded.
    np.save(work / "tiles.npy", np.asarray(tiles, dtype=np.int64))
    (work / "state.json").write_text(_json.dumps(dict(
        mission=mission, anchor=anchor, out_crs=out_crs,
        tr=[TR.a, TR.b, TR.c, TR.d, TR.e, TR.f], W=int(W), H=int(H),
        nspec=int(nspec), spec_bands=spec_bands, a_alpha=int(a_alpha), reject_px=_REJECT_PX,
        min_tile=_MIN_TILE, tile_h=_TILE_H, tile_w=_TILE_W)))
    n_chunks = -(-len(tiles) // _CHUNK_TILES)
    log(f"    matching {len(tiles)} tiles in {n_chunks} x {_CHUNK_TILES}-tile "
        f"subprocess chunks (a fresh GPU process each — resets the MPS decay)")
    for ci in range(n_chunks):
        rc = _subprocess.run(
            [_sys.executable, "-m", "tidysurvey._regchunk", str(work), str(ci)])
        if rc.returncode != 0:
            _shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError(f"register_survey_dense: match chunk "
                               f"{ci + 1}/{n_chunks} failed (rc={rc.returncode})")
        log(f"    [chunk {ci + 1}/{n_chunks}] done")
    buf_an, buf_d, n_used = [], [], 0
    for ci in range(n_chunks):
        z = np.load(work / f"part_{ci}.npz")
        if len(z["an"]):
            buf_an.append(z["an"]); buf_d.append(z["d"]); n_used += int(z["n_used"])
    _shutil.rmtree(work, ignore_errors=True)
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
    _cpts, _ckeep, _cok = pts, keep, ok            # cell-coverage inputs (pre-subset)
    pts, vx, vy = pts[keep], vx[keep], vy[keep]
    if len(pts) == 0:
        raise RuntimeError("register_survey_dense: all cells rejected by the gates")
    d_med = float(np.median(np.hypot(vx, vy)) * res * 100)
    log(f"    {len(pts)} cells after gates ({n_rej} rejected)  |d| med={d_med:.1f} cm")
    # --- durable per-region diagnostics on the _FS-cell lattice --------------------
    #  coverage (1=matched / 0=expected-but-failed / 255=not-expected) is the
    #  failed-match map; disp (|d|_cm, dx, dy) are the RAW pooled displacements (not
    #  the smooth field) — the honest per-region error signal the deferred
    #  through-time analysis stacks across seasons. On the mission's snapped lattice
    #  so they are cross-season stackable.
    _celltr = _Affine.translation(TR.c, TR.f) * _Affine.scale(_FS * res, -_FS * res)
    cov = np.where(_cok, 0, 255).astype(np.uint8)
    cov[np.clip((_cpts[_ckeep, 1] // _FS).astype(int), 0, Hcc - 1),
        np.clip((_cpts[_ckeep, 0] // _FS).astype(int), 0, Wcc - 1)] = 1
    disp = np.full((3, Hcc, Wcc), np.nan, np.float32)
    _dy = np.clip((pts[:, 1] // _FS).astype(int), 0, Hcc - 1)
    _dx = np.clip((pts[:, 0] // _FS).astype(int), 0, Wcc - 1)
    disp[0, _dy, _dx] = np.hypot(vx, vy) * res * 100
    disp[1, _dy, _dx] = vx * res * 100; disp[2, _dy, _dx] = vy * res * 100
    cov_out = str(_Path(output_registered_survey_path).with_suffix(".coverage.tif"))
    disp_out = str(_Path(output_registered_survey_path).with_suffix(".disp.tif"))
    with rasterio.open(cov_out, "w", driver="GTiff", width=Wcc, height=Hcc, count=1,
                       dtype="uint8", crs=out_crs, transform=_celltr, nodata=255,
                       tiled=True, compress="zstd") as _dc:
        _dc.write(cov, 1)
        _dc.set_band_description(1, "match coverage: 1=matched 0=failed-but-expected 255=not-expected")
    with rasterio.open(disp_out, "w", driver="GTiff", width=Wcc, height=Hcc, count=3,
                       dtype="float32", crs=out_crs, transform=_celltr, nodata=float("nan"),
                       tiled=True, compress="zstd") as _dd:
        _dd.write(disp)
        for _i, _n in enumerate(("|d|_cm", "dx_cm", "dy_cm"), 1):
            _dd.set_band_description(_i, _n)
    failed_cells = int((cov == 0).sum())

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
    _B.tag_output(dst, law.names)                  # names + tagged alpha: no guessing downstream
    wlock = _threading.Lock()
    tls = _threading.local()

    def upf(arr, hr0, hc0, hh, hw):
        return _F.upsample_block(arr, _FIELD_DS, hr0 - r0b, hc0 - c0b, hh, hw)

    def wone(bc):
        r0, c0 = bc
        if not hasattr(tls, "m"):
            tls.m = rasterio.open(mission)                  # native read, per worker
        bh2, bw2 = min(_BLOCK, r1b - r0), min(_BLOCK, c1b - c0)
        hr0, hc0 = max(r0b, r0 - halo), max(c0b, c0 - halo)
        hr1, hc1 = min(r1b, r0 + bh2 + halo), min(c1b, c0 + bw2 + halo)
        hh, hw = hr1 - hr0, hc1 - hc0
        iy, ix = r0 - hr0, c0 - hc0
        # native read window covering this (haloed) grid block, padded for the cubic kernel
        PAD = 3
        nc0 = max(0, int(_math.floor(sx * hc0 + ax)) - PAD)
        nr0 = max(0, int(_math.floor(sy * hr0 + ay)) - PAD)
        nc1 = min(mW, int(_math.ceil(sx * hc1 + ax)) + PAD)
        nr1 = min(mH, int(_math.ceil(sy * hr1 + ay)) + PAD)
        fxs = upf(Fx, hr0, hc0, hh, hw); fys = upf(Fy, hr0, hc0, hh, hw)
        gx, gy = np.meshgrid(np.arange(hw, dtype=np.float32),
                             np.arange(hh, dtype=np.float32))
        outb = np.zeros((nspec + 1, hh, hw), np.uint8)
        if nc1 > nc0 and nr1 > nr0:
            native = tls.m.read(spec_bands,
                                window=_Window(nc0, nr0, nc1 - nc0, nr1 - nr0)).astype(np.float32)
            # ONE resample: corrected source GRID coord -> NATIVE pixel (local to the read)
            mapx = (sx * (hc0 + gx - fxs) + ax - nc0).astype(np.float32)
            mapy = (sy * (hr0 + gy - fys) + ay - nr0).astype(np.float32)
            for b in range(nspec):
                rem = cv2.remap(native[b], mapx, mapy, cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                outb[b] = np.clip(rem, 0, 255).astype(np.uint8)     # cubic can overshoot [0,255]
        # alpha stays grid-space + nearest (no blur), via the grid correction map
        val = vfull[hr0 - r0b:hr1 - r0b, hc0 - c0b:hc1 - c0b]
        outb[nspec] = (cv2.remap(val, (gx - fxs).astype(np.float32), (gy - fys).astype(np.float32),
                                 cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
                       ).astype(np.uint8) * 255
        with wlock:
            dst.write(outb[:, iy:iy + bh2, ix:ix + bw2],
                      window=_Window(c0 - c0b, r0 - r0b, bw2, bh2))

    wblocks = [(r0, c0) for r0 in range(r0b, r1b, _BLOCK) for c0 in range(c0b, c1b, _BLOCK)]
    with _TPE(max_workers=workers) as ex:
        list(ex.map(wone, wblocks))
    dst.close()

    summary = dict(kind="align", mission=os.path.basename(mission),
                   reference=os.path.basename(anchor),
                   bands=list(law.names), alpha="tagged" if law.alpha else "derived",
                   tiles=len(tiles),
                   matches=int(len(an)), cells=int(len(pts)), rejected=n_rej,
                   failed_cells=failed_cells,
                   d_med_cm=round(d_med, 1),
                   d_p99_cm=round(float(np.percentile(np.hypot(vx, vy), 99)) * res * 100, 1),
                   maxF_cm=round(maxF * res * 100, 1),
                   out=str(output_registered_survey_path),
                   coverage=cov_out, disp=disp_out,
                   seconds=round(_time.time() - t0, 1))
    if qa_json:
        _Path(qa_json).parent.mkdir(parents=True, exist_ok=True)
        _Path(qa_json).write_text(_json.dumps(summary, indent=2))
    log(f"    registered -> {output_registered_survey_path} ({summary['seconds']:.0f}s)")
    return summary

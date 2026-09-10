"""The quality checks as reusable code — the thing production most lacked.

Two kinds, deliberately distinct:

  HARD GATES (physical thresholds; a breach stops the run):
    - seam_tripwire            post-stitch seams must line up within max_cm
    - assert_interiors_unchanged  pixels outside the blend band are the owner
                                  source VERBATIM (sampled byte-identity)

  SHIPPED EVIDENCE (no pass/fail — the metric is scene-dependent, so the map
  is read per survey, like the calibration MAE bands):
    - registration_r_cells     per-cell texture-masked correlation of the
                               registered product's green band vs the anchor's
                               green band (7.55 m cells) -> the promoted
                               registration reliability raster

Ported from the audited scorer (audit repo: 02_registration/poc/
baseline_production_ms.py — the map that read production 0.685 -> 0.863).
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from . import bands as _B


class CheckResult(dict):
    """dict of findings + .assert_ok(); print-friendly."""

    def __init__(self, ok, kind, **info):
        super().__init__(ok=bool(ok), kind=kind, **info)

    def assert_ok(self):
        if not self["ok"]:
            raise AssertionError(f"{self['kind']} FAILED: "
                                 f"{json.dumps({k: v for k, v in self.items() if k != 'ok'})}")
        return self


# --------------------------------------------------------------------------- #
# registration reliability: the promoted per-cell r map (evidence, not a gate)
# --------------------------------------------------------------------------- #
def registration_r_cells(ms_path, anchor_path, out_tif, cell_px=128, res=None,
                         min_n=500, block=2048, green_band=2, workers=None,
                         like=None, log=print):
    """Per-cell texture-masked Pearson r of the registered product's GREEN band
    vs the anchor's GREEN band, on gradient-above-median co-valid pixels
    (>= min_n per cell). Wall-to-wall, streamed in constant memory. Writes a
    2-band raster (r, ms-missing fraction) and returns the summary.

    like : path of a raster whose grid defines the cells (typically the
           engine's <out>.cells.tif), so the score lands on the SAME lattice
           as the displacement diagnostics and stacks with them band for
           band. Its cell size must be a whole number of product pixels.
           Without it the lattice is the product/anchor overlap in cell_px
           cells.

    Non-circular by construction: the LoFTR matcher consumes a 3-band
    grayscale; this scorer correlates raw green only. Pearson r is
    affine-invariant, so exposure differences don't penalise — residual
    radiometry (flight-line striping) is calibration's problem, not
    registration's. IMPORTANT for maintainers: cells align by georeferenced
    bounds, never by array shape.
    """
    t0 = time.time()
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    mlaw = _B.band_law(ms_path)
    with rasterio.open(ms_path) as m:
        crs_out = m.crs
        res = res or round(abs(m.res[0]), 3)
    alaw = _B.band_law(anchor_path)
    a_alpha = alaw.alpha or alaw.count                   # tagged alpha; untagged legacy: last band
    if like:
        with rasterio.open(like) as lk:
            crs_out = lk.crs
            ncx, ncy = lk.width, lk.height
            L, T, cell_m = lk.transform.c, lk.transform.f, float(abs(lk.transform.a))
        cell_px = int(round(cell_m / res))
        if cell_px < 1 or abs(cell_px * res - cell_m) > 1e-6:
            raise ValueError(f"registration_r_cells: `like` cell {cell_m} m is not a whole "
                             f"number of {res} m pixels")
    else:
        cell_m = cell_px * res
        with rasterio.open(anchor_path) as a_s, rasterio.open(ms_path) as m_s, \
                WarpedVRT(a_s, crs=crs_out) as va, WarpedVRT(m_s, crs=crs_out) as vm:
            L = max(va.bounds.left, vm.bounds.left)
            R = min(va.bounds.right, vm.bounds.right)
            B = max(va.bounds.bottom, vm.bounds.bottom)
            T = min(va.bounds.top, vm.bounds.top)
        ncx = int((R - L) // cell_m)
        ncy = int((T - B) // cell_m)
        if ncx < 1 or ncy < 1:
            raise ValueError("registration_r_cells: no overlap between product and anchor")
        T = B + ncy * cell_m
    block -= block % cell_px                            # blocks hold whole cells
    if block < cell_px:
        block = cell_px
    W, H = ncx * cell_px, ncy * cell_px
    transform = from_origin(L, T, res, res)
    log(f"[reg_r_cells] grid {W}x{H} @ {res} m ({ncx}x{ncy} cells of {cell_m:.2f} m)")

    vrt_kw = dict(crs=crs_out, transform=transform, width=W, height=H)
    Z = lambda: np.zeros((ncy, ncx), np.float64)
    acc = {k: Z() for k in ("n", "sx", "sy", "sxx", "syy", "sxy", "miss", "anch")}
    lock = threading.Lock()
    tls = threading.local()

    def cellsum(arr, bh, bw):
        return arr.reshape(bh // cell_px, cell_px, bw // cell_px, cell_px).sum(axis=(1, 3))

    def get_v():
        if not hasattr(tls, "v"):
            a = rasterio.open(anchor_path)
            m = rasterio.open(ms_path)
            tls.v = (WarpedVRT(a, resampling=Resampling.average, **vrt_kw),
                     WarpedVRT(a, resampling=Resampling.nearest, **vrt_kw),
                     WarpedVRT(m, resampling=Resampling.bilinear, **vrt_kw),
                     WarpedVRT(m, resampling=Resampling.nearest, **vrt_kw))
        return tls.v

    def do_block(bc):
        r0, c0 = bc
        vg, va_, mg, mv = get_v()
        bh, bw = min(block, H - r0), min(block, W - c0)
        bh -= bh % cell_px; bw -= bw % cell_px
        if bh <= 0 or bw <= 0:
            return
        win = Window(c0, r0, bw, bh)
        al_a = va_.read(a_alpha, window=win) > 127
        if not al_a.any():
            return
        g_a = vg.read(green_band, window=win).astype(np.float32)
        g_m = mg.read(green_band, window=win).astype(np.float32)
        val_m = mv.read(mlaw.alpha, window=win) > 127 if mlaw.alpha else \
            (mv.read(list(mlaw.spectral[:4]), window=win) != 0).any(axis=0)
        co = al_a & val_m
        out = {"anch": cellsum(al_a.astype(np.float64), bh, bw),
               "miss": cellsum((al_a & ~val_m).astype(np.float64), bh, bw)}
        if co.any():
            gy, gx = np.gradient(g_a)
            mag = np.hypot(gx, gy)
            tex = co & (mag > np.median(mag[co]))
            x = np.where(tex, g_m, 0).astype(np.float64)
            y = np.where(tex, g_a, 0).astype(np.float64)
            out.update(n=cellsum(tex.astype(np.float64), bh, bw),
                       sx=cellsum(x, bh, bw), sy=cellsum(y, bh, bw),
                       sxx=cellsum(x * x, bh, bw), syy=cellsum(y * y, bh, bw),
                       sxy=cellsum(x * y, bh, bw))
        with lock:
            sl = np.s_[r0 // cell_px:r0 // cell_px + bh // cell_px,
                       c0 // cell_px:c0 // cell_px + bw // cell_px]
            for k, v in out.items():
                acc[k][sl] += v

    blocks = [(r0, c0) for r0 in range(0, H, block) for c0 in range(0, W, block)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do_block, blocks))

    n, sx, sy = acc["n"], acc["sx"], acc["sy"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = n * acc["sxy"] - sx * sy
        var = (n * acc["sxx"] - sx * sx) * (n * acc["syy"] - sy * sy)
        r = np.where((n >= min_n) & (var > 0), cov / np.sqrt(var), np.nan).astype(np.float32)
        miss = np.where(acc["anch"] > 0, acc["miss"] / acc["anch"], np.nan).astype(np.float32)

    Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", driver="GTiff", width=ncx, height=ncy, count=2,
                       dtype="float32", crs=crs_out,
                       transform=from_origin(L, T, cell_m, cell_m),
                       nodata=np.nan, tiled=True, compress="zstd") as d:
        d.write(r, 1)
        d.write(miss, 2)
        d.set_band_description(1, "texture-masked pearson r (product green vs anchor green)")
        d.set_band_description(2, "product missing fraction where anchor valid")

    ok = np.isfinite(r)
    summary = dict(kind="registration_r_cells", out=str(out_tif),
                   cells=int(ok.sum()), cell_m=round(cell_m, 3),
                   median_r=(round(float(np.median(r[ok])), 3) if ok.any() else None),
                   p10_r=(round(float(np.percentile(r[ok], 10)), 3) if ok.any() else None),
                   seconds=round(time.time() - t0, 1))
    log(f"[reg_r_cells] {summary['cells']} cells, median r {summary['median_r']} "
        f"({summary['seconds']:.0f}s) -> {out_tif}")
    return summary


# --------------------------------------------------------------------------- #
# hard gates
# --------------------------------------------------------------------------- #
def seam_tripwire(stitch_report, max_cm=20.0) -> CheckResult:
    """HARD GATE: every post-solve seam residual must be <= max_cm. Reads the
    report written by merge.seam_merge (path or dict). A breach should stop
    the run — the specified fallback is stitch.gauge = "anchored"."""
    rep = stitch_report
    if not isinstance(rep, dict):
        rep = json.loads(Path(stitch_report).read_text())
    bad = [s for s in rep.get("seams", [])
           if s.get("residual_cm") is not None and s["residual_cm"] > max_cm]
    return CheckResult(len(bad) == 0, "seam_tripwire", max_cm=max_cm,
                       n_seams=rep.get("n_seams"),
                       breaches=[{k: s[k] for k in ("pair", "residual_cm")} for s in bad])


def assert_interiors_unchanged(inputs, mosaic, ownership, band_width_m=1.0,
                               samples_per_input=8, window_px=256,
                               block_px=2048, halo_px=64,
                               seed=1234, log=print) -> CheckResult:
    """HARD GATE (sampled): away from the seam band, the mosaic must contain
    what the composite READ from the owner source, unaltered.

    Two subtleties the first full-scale run taught us (2026-07-06):
    * Windows must clear the band ENTIRELY (centre + half-diagonal), else they
      clip the fringe, where pixels are blended by design.
    * The reference is read through the WRITER'S exact window geometry
      (block-aligned halo read, then crop): GDAL's approximating reprojection
      is deterministic per request shape but NOT across shapes, so a
      different-shaped read can differ by ±1 DN at ~1e-4 of pixels. Byte
      identity is still demanded against the geometry-matched read; if only
      warp dust remains (|Δ| <= 1 DN at <= 0.1% of pixels, e.g. under a
      different PROJ context than the writing process), the window passes and
      is counted separately. Structured differences — anything > 1 DN or
      denser than 0.1% — fail: that is blend leakage, the thing this gate
      exists to catch."""
    from scipy.ndimage import distance_transform_edt

    rng = np.random.RandomState(seed)
    names = [n for n, _ in inputs]
    paths = {n: p for n, p in inputs}
    with rasterio.open(ownership) as o:
        cat = o.read(1)
        o_tr = o.transform
        seam_val = int(o.tags().get("seam_band_value", len(names) + 1))

    checked, bit_identical, dust, mismatched = 0, 0, [], []
    with rasterio.open(mosaic) as m:
        m_tr, m_crs = m.transform, m.crs
        nb = m.count
        # the WHOLE window must clear the band, not just its centre
        reach = int(np.ceil((window_px / 2) * abs(m_tr.a) / abs(o_tr.a) * 1.42)) + 2
        far = distance_transform_edt(cat != seam_val) >= reach
        for i, name in enumerate(names, start=1):
            ys, xs = np.where((cat == i) & far)
            if len(ys) == 0:
                continue
            pick = rng.choice(len(ys), size=min(samples_per_input, len(ys)), replace=False)
            with rasterio.open(paths[name]) as s, \
                    WarpedVRT(s, crs=m_crs, transform=m_tr, width=m.width,
                              height=m.height, resampling=Resampling.average) as v:
                for j in pick:
                    wx, wy = o_tr * (xs[j] + 0.5, ys[j] + 0.5)   # world centre of the cell
                    col, row = ~m_tr * (wx, wy)
                    r0 = int(max(0, min(m.height - window_px, row - window_px // 2)))
                    c0 = int(max(0, min(m.width - window_px, col - window_px // 2)))
                    # clamp the window inside ONE composite block, then read the
                    # reference through that block's halo request — the writer's
                    # exact geometry
                    br0 = (r0 // block_px) * block_px
                    bc0 = (c0 // block_px) * block_px
                    bh = min(block_px, m.height - br0)
                    bw = min(block_px, m.width - bc0)
                    if bh < window_px or bw < window_px:
                        continue                    # edge sliver block: skip
                    r0 = int(min(max(r0, br0), br0 + bh - window_px))
                    c0 = int(min(max(c0, bc0), bc0 + bw - window_px))
                    hr0, hc0 = max(0, br0 - halo_px), max(0, bc0 - halo_px)
                    hr1 = min(m.height, br0 + bh + halo_px)
                    hc1 = min(m.width, bc0 + bw + halo_px)
                    ref = v.read(window=Window(hc0, hr0, hc1 - hc0, hr1 - hr0))
                    b = ref[:nb - 1, r0 - hr0:r0 - hr0 + window_px,
                            c0 - hc0:c0 - hc0 + window_px]
                    win = Window(c0, r0, window_px, window_px)
                    a = m.read(list(range(1, nb)), window=win)          # spectral bands
                    valid = m.read(nb, window=win) > 127
                    if not valid.any():
                        continue
                    checked += 1
                    if np.array_equal(a[:, valid], b[:, valid]):
                        bit_identical += 1
                        continue
                    d = a[:, valid].astype(np.int16) - b[:, valid].astype(np.int16)
                    frac = float((d != 0).mean())
                    entry = dict(input=name, row=r0, col=c0,
                                 mismatch_frac=round(frac, 6),
                                 max_abs_delta=int(np.abs(d).max()))
                    if entry["max_abs_delta"] <= 1 and frac <= 1e-3:
                        dust.append(entry)                    # reprojection dust: passes
                    else:
                        mismatched.append(entry)              # structured change: fails
    ok = len(mismatched) == 0 and checked > 0
    log(f"[interiors] {checked} windows compared: {bit_identical} bit-identical, "
        f"{len(dust)} warp-dust (|Δ|<=1, <=0.1%), {len(mismatched)} FAILED")
    return CheckResult(ok, "interiors_unchanged", windows_checked=checked,
                       bit_identical=bit_identical, warp_dust=dust,
                       mismatched=mismatched)


# --------------------------------------------------------------------------- #
# calibration evidence summary (for the report; no threshold)
# --------------------------------------------------------------------------- #
def calibration_summary(qa_tif) -> dict:
    """Medians/p90 of the 7 MAE bands + n_px from the reliability raster."""
    out = {}
    with rasterio.open(qa_tif) as s:
        for i, name in enumerate(s.descriptions, start=1):
            b = s.read(i)
            v = b[np.isfinite(b)]
            if v.size:
                out[name] = dict(median=round(float(np.median(v)), 4),
                                 p90=round(float(np.percentile(v, 90)), 4))
    return out

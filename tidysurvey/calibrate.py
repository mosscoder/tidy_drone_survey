"""Per-tile cross-band linear calibration to a satellite reference.

Ported from the audited deployment (audit repo: 03_calibration/deploy/
block_ridge_bilinear_full_opt.py + phase3_ridge_perblock_qa.py) — the method
that shipped the 2024 reflectance product (pooled held-out band R2 0.864 vs
0.806 single-band; NDVI/NDRE/CIre MAE 0.024/0.018/0.068).

The idea: per 100 m tile, fit EACH satellite band from ALL FOUR drone bands —
a small linear formula, lightly ridge-stabilised (the drone bands are
collinear; ridge keeps the coefficient vectors stable enough to interpolate).
The 5-coefficient vectors are BILINEARLY interpolated from the 4 nearest tile
centres to every native pixel and applied to the native DN — exact for a
linear fit, so tile edges are seamless and the apply is pure arithmetic (no
per-pixel model). Output is clipped to the reference scene's range.

The companion reliability raster is the 7 MAE values: per tile, 5-fold
within-tile cross-validated mean absolute error — RAW per-band reflectance
x1e4 (read each against ITS OWN band's magnitude, never across bands) and
VI-MAE in index units (NDVI/NDRE/CIre) — plus n_px for context. No pass/fail
threshold: the map is shipped evidence, read per survey.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from . import bands as _B


# --------------------------------------------------------------------------- #
# training pairs: drone mosaic downsampled onto the reference grid
# --------------------------------------------------------------------------- #
def build_pairs(drone_path, reference_path, nspec=4):
    """Pair every valid reference pixel with the average-downsampled drone DN
    at the same spot. Returns dict(drone (n,4), s2 (n,4), x, y) in CRS metres."""
    with rasterio.open(reference_path) as ref:
        R = ref.read().astype(np.float32)               # (4, h, w) reflectance x1e4
        rt = ref.transform
        nod = ref.nodata
        h, w = ref.height, ref.width
        ref_crs = ref.crs
    ref_valid = np.ones((h, w), bool)
    if nod is not None:
        ref_valid &= ~np.any(R == nod, axis=0)
    ref_valid &= np.all(np.isfinite(R), axis=0) & np.any(R != 0, axis=0)

    law = _B.band_law(drone_path)
    alpha_idx = law.alpha or law.count                   # tagged alpha; untagged legacy mosaics: last band
    with rasterio.open(drone_path) as d:
        with WarpedVRT(d, crs=ref_crs, transform=rt, width=w, height=h,
                       resampling=Resampling.average) as v:
            D = v.read(list(law.spectral[:nspec])).astype(np.float32)
            A = v.read(alpha_idx)
    dr_valid = A > 200                                   # ≥~80% of native px valid in the 10 m cell

    m = ref_valid & dr_valid
    ys, xs = np.where(m)
    cx = rt.c + (xs + 0.5) * rt.a
    cy = rt.f + (ys + 0.5) * rt.e
    return dict(drone=D[:, m].T, s2=R[:, m].T, x=cx, y=cy)


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
def _ridge_fit(X, Y, alpha):
    """Multivariate ridge: centre -> per-feature penalty (alpha * Sxx diagonal)
    on the slopes, intercept unpenalised -> recover intercept."""
    xb = X.mean(0); yb = Y.mean(0)
    Xc = X - xb; Yc = Y - yb
    A = Xc.T @ Xc
    dA = np.diag(A).copy()
    Areg = A + np.diag(alpha * dA + 1e-6 * (dA.max() + 1.0))
    try:
        Bs = np.linalg.solve(Areg, Xc.T @ Yc)            # (4 slopes, 4 out-bands)
    except np.linalg.LinAlgError:
        Bs = np.zeros((X.shape[1], Y.shape[1]))
    return Bs, yb - xb @ Bs


def fit_block_ridge(drone, reference, block_m=100.0, alpha=1e-3, min_px=20,
                    band_names=("Red", "Green", "NIR", "RedEdge"), log=print):
    """Fit the per-tile multivariate ridge model.

    drone/reference: paths (pairs are built internally) or a prebuilt pairs
    dict from `build_pairs`. Returns the model dict consumed by
    `apply_bilinear` / `perblock_qa` / `save_model`.
    """
    t0 = time.time()
    pairs = drone if isinstance(drone, dict) else build_pairs(drone, reference)
    X, Y, x, y = pairs["drone"], pairs["s2"], pairs["x"], pairs["y"]
    lo = np.array([max(1.0, Y[:, i].min()) for i in range(4)], np.float32)
    hi = np.array([Y[:, i].max() for i in range(4)], np.float32)

    bxa = np.floor(x / block_m).astype(int)
    bya = np.floor(y / block_m).astype(int)
    bx0, by0 = int(bxa.min()), int(bya.min())
    nbx, nby = int(bxa.max()) - bx0 + 1, int(bya.max()) - by0 + 1
    idx = defaultdict(list)
    for p, (br, bc) in enumerate(zip(bya - by0, bxa - bx0)):
        idx[(br, bc)].append(p)

    CB = np.zeros((nby, nbx, 4, 5), np.float32)          # [slopes x4, intercept] per out-band
    PRES = np.zeros((nby, nbx), bool)
    for (gr, gc), ix in idx.items():
        if len(ix) < min_px:
            continue
        ix = np.array(ix)
        Bs, inter = _ridge_fit(X[ix].astype(np.float64), Y[ix].astype(np.float64), alpha)
        CB[gr, gc, :, :4] = Bs.T
        CB[gr, gc, :, 4] = inter
        PRES[gr, gc] = True
    log(f"[calibrate] fit {int(PRES.sum())} tiles @ {block_m:.0f} m "
        f"(multivariate ridge, all-linear) in {time.time() - t0:.1f}s")
    return dict(CB=CB, PRES=PRES, bx0=bx0, by0=by0, nbx=nbx, nby=nby,
                block_m=float(block_m), alpha=float(alpha), lo=lo, hi=hi,
                band_names=list(band_names), n_pairs=int(len(x)),
                pairs=pairs)                              # kept for perblock_qa


def save_model(model, path):
    """The tiny fitted model: tile weight vectors + offsets + clip range."""
    out = {k: model[k] for k in ("bx0", "by0", "nbx", "nby", "block_m", "alpha",
                                 "band_names", "n_pairs")}
    out["lo"] = model["lo"].tolist()
    out["hi"] = model["hi"].tolist()
    out["CB"] = model["CB"].tolist()
    out["PRES"] = model["PRES"].astype(int).tolist()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out))
    return path


# --------------------------------------------------------------------------- #
# the apply: bilinear coefficient interpolation at native resolution
# --------------------------------------------------------------------------- #
# A DatasetReader holds a C handle (self._hds) that cannot be pickled across the
# loky process boundary, so each worker opens — and caches — its own handle.
_WORKER_SRC: dict = {}


def _worker_src(path):
    ds = _WORKER_SRC.get(path)
    if ds is None:
        ds = rasterio.open(path)
        _WORKER_SRC[path] = ds
    return ds


def apply_bilinear(drone_path, model, out, tile=2048, workers=None, log=print):
    """Blend the tile coefficient vectors bilinearly to native resolution,
    apply to native DN, clip to the reference scene range, write a tiled ZSTD
    GeoTIFF (finalize to COG with cog.finalize_cog). Exact for a linear fit."""
    from joblib import Parallel, delayed

    t0 = time.time()
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    CB, PRES = model["CB"], model["PRES"]
    bx0, by0 = model["bx0"], model["by0"]
    nbx, nby = model["nbx"], model["nby"]
    bm = model["block_m"]
    lo, hi = model["lo"], model["hi"]
    names = tuple(f"{n}_refl_x1e4" for n in model["band_names"])

    src = rasterio.open(drone_path)
    prof = src.profile.copy()
    H, W, tr = src.height, src.width, src.transform
    law = _B.band_law(drone_path)
    alpha_idx = law.alpha or law.count                   # tagged alpha; untagged legacy mosaics: last band
    spec_idx = [b - 1 for b in law.spectral[:4]]
    prof.update(count=4, dtype="int16", nodata=0, compress="zstd", zstd_level=1,
                predictor=2, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
    for k in ("photometric", "alpha"):
        prof.pop(k, None)
    src.close()   # metadata only in the parent; workers open their own handles

    def process(r0, c0, h, w):
        arr = _worker_src(drone_path).read(window=Window(c0, r0, w, h))
        DN = arr[spec_idx].astype(np.float32)
        al = arr[alpha_idx - 1] > 0
        if not al.any():
            return r0, c0, np.zeros((4, h, w), np.int16)
        xs = tr.c + (c0 + np.arange(w) + 0.5) * tr.a
        ys = tr.f + (r0 + np.arange(h) + 0.5) * tr.e
        fcol = np.broadcast_to(xs / bm - 0.5, (h, w))
        frow = np.broadcast_to((ys / bm - 0.5)[:, None], (h, w))
        ic = np.floor(fcol).astype(int); fx = (fcol - ic).astype(np.float32)
        ir = np.floor(frow).astype(int); fy = (frow - ir).astype(np.float32)
        Cacc = np.zeros((h, w, 4, 5), np.float32)
        Wt = np.zeros((h, w), np.float32)
        for oi in (0, 1):
            wx = (1 - fx) if oi == 0 else fx
            for oj in (0, 1):
                w2 = wx * ((1 - fy) if oj == 0 else fy)
                gc = ic + oi - bx0; gr = ir + oj - by0
                inb = (gr >= 0) & (gr < nby) & (gc >= 0) & (gc < nbx)
                grc = np.clip(gr, 0, nby - 1); gcc = np.clip(gc, 0, nbx - 1)
                wgt = (w2 * (inb & PRES[grc, gcc])).astype(np.float32)
                Wt += wgt
                Cacc += wgt[..., None, None] * CB[grc, gcc]
        Wd = np.where(Wt > 0, Wt, 1)
        outb = np.zeros((4, h, w), np.int16)
        for b in range(4):
            v = (Cacc[..., b, 4] + sum(Cacc[..., b, j] * DN[j] for j in range(4))) / Wd
            v = np.clip(v, lo[b], hi[b])
            v[~al | (Wt <= 0)] = 0
            outb[b] = np.rint(v).astype(np.int16)
        return r0, c0, outb

    wins = [(r, c, min(tile, H - r), min(tile, W - c))
            for r in range(0, H, tile) for c in range(0, W, tile)]
    log(f"[calibrate] apply {W}x{H}: {len(wins)} tiles, {workers} workers "
        "(all-linear coefficient blend)")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tasks = (delayed(process)(r, c, h, w) for r, c, h, w in wins)
    with rasterio.open(out, "w", **prof) as dst:
        dst.descriptions = names
        done = 0
        for r0, c0, o in Parallel(n_jobs=workers, backend="loky",
                                  return_as="generator_unordered")(tasks):
            dst.write(o, window=Window(c0, r0, o.shape[2], o.shape[1]))
            done += 1
            if done % 200 == 0:
                log(f"    [{done}/{len(wins)}]")
    log(f"[calibrate] applied in {time.time() - t0:.0f}s -> {out}")
    return out


# --------------------------------------------------------------------------- #
# the companion reliability raster: the 7 MAE values (+ n_px)
# --------------------------------------------------------------------------- #
_QA_BANDS = ("MAE_Red_x1e4", "MAE_Green_x1e4", "MAE_NIR_x1e4", "MAE_RedEdge_x1e4",
             "MAE_NDVI", "MAE_NDRE", "MAE_CIre", "n_px")


def _vi(a):
    R, G, N, E = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    ndvi = (N - R) / np.where(N + R > 1, N + R, np.nan)
    ndre = (N - E) / np.where(N + E > 1, N + E, np.nan)
    cire = N / np.where(E > 1, E, np.nan) - 1
    return ndvi, ndre, cire


def perblock_qa(model, out, crs, log=print):
    """Write the reliability raster: 8 bands = 7 cross-validated MAE + n_px,
    one pixel per tile. Per-band MAE is RAW (reflectance x1e4, un-normalised —
    read each against its own band's magnitude); VI-MAE is in index units.
    5-fold within-tile CV, deterministic folds, closed-form ridge -> cheap."""
    from rasterio.transform import from_origin

    t0 = time.time()
    pairs = model["pairs"]
    X, Y, x, y = pairs["drone"], pairs["s2"], pairs["x"], pairs["y"]
    bm, alpha = model["block_m"], model["alpha"]
    bx0, by0, nbx, nby = model["bx0"], model["by0"], model["nbx"], model["nby"]

    T = from_origin(bx0 * bm, (by0 + nby) * bm, bm, bm)
    grid = np.full((8, nby, nbx), np.nan, np.float32)

    bxa = np.floor(x / bm).astype(int)
    bya = np.floor(y / bm).astype(int)
    idx = defaultdict(list)
    for p, (a, b) in enumerate(zip(bya, bxa)):
        idx[(a, b)].append(p)

    nfit = 0
    for (gr, gc), ix in idx.items():
        ix = np.array(ix)
        if len(ix) < 20:
            continue
        f = np.random.RandomState((gr * 131 + gc) % 99991).randint(0, 5, len(ix))
        oof = np.full((len(ix), 4), np.nan)
        for k in range(5):
            trn, ev = f != k, f == k
            if trn.sum() < 5 or ev.sum() == 0:
                continue
            Bs, ic = _ridge_fit(X[ix[trn]].astype(float), Y[ix[trn]].astype(float), alpha)
            oof[ev] = X[ix[ev]].astype(float) @ Bs + ic
        m = np.isfinite(oof).all(1)
        if m.sum() < 10:
            continue
        P, Tr = oof[m], Y[ix[m]].astype(float)
        rr, cc = rowcol(T, gc * bm + bm / 2, gr * bm + bm / 2)
        if not (0 <= rr < nby and 0 <= cc < nbx):
            continue
        for b in range(4):
            grid[b, rr, cc] = float(np.mean(np.abs(P[:, b] - Tr[:, b])))
        nP, nT = _vi(P), _vi(Tr)
        for j in range(3):
            mm = np.isfinite(nP[j]) & np.isfinite(nT[j])
            if mm.sum() >= 8:
                grid[4 + j, rr, cc] = float(np.mean(np.abs(nP[j][mm] - nT[j][mm])))
        grid[7, rr, cc] = len(ix)
        nfit += 1

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", dtype="float32", count=8, width=nbx, height=nby,
                crs=crs, transform=T, nodata=float("nan"), tiled=True,
                compress="zstd", predictor=2)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(grid)
        dst.descriptions = _QA_BANDS
    log(f"[calibrate] reliability raster: {nfit} tiles, 8 bands = 7 MAE + n_px "
        f"({time.time() - t0:.1f}s) -> {out}")

    summary = {"tiles": nfit}
    for b, nm in enumerate(_QA_BANDS[:7]):
        v = grid[b][np.isfinite(grid[b])]
        if v.size:
            summary[nm] = dict(median=round(float(np.median(v)), 4),
                               p90=round(float(np.percentile(v, 90)), 4))
    return summary


def pooled_oof_r2(model, log=print):
    """Aggregate model-performance number for the report: pooled within-tile
    5-fold OOF R2 vs the GLOBAL mean baseline, per band + VIs. (Per-tile R2 is
    variance-suppressed and deliberately NOT rasterised — MAE is the per-tile
    reliability channel.)"""
    pairs = model["pairs"]
    X, Y, x, y = pairs["drone"], pairs["s2"], pairs["x"], pairs["y"]
    bm, alpha = model["block_m"], model["alpha"]
    bxa = np.floor(x / bm).astype(int)
    bya = np.floor(y / bm).astype(int)
    idx = defaultdict(list)
    for p, (a, b) in enumerate(zip(bya, bxa)):
        idx[(a, b)].append(p)
    P = np.full_like(Y, np.nan, dtype=float)
    for (gr, gc), ix in idx.items():
        ix = np.array(ix)
        if len(ix) < 20:
            continue
        f = np.random.RandomState((gr * 131 + gc) % 99991).randint(0, 5, len(ix))
        for k in range(5):
            trn, ev = f != k, f == k
            if trn.sum() < 5 or ev.sum() == 0:
                continue
            Bs, ic = _ridge_fit(X[ix[trn]].astype(float), Y[ix[trn]].astype(float), alpha)
            P[ix[ev]] = X[ix[ev]].astype(float) @ Bs + ic
    m = np.isfinite(P).all(1)
    out = {}
    for b, nm in enumerate(model["band_names"]):
        yy, pp = Y[m, b].astype(float), P[m, b]
        out[nm] = round(1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2), 3)
    out["mean_band_R2"] = round(float(np.mean([out[n] for n in model["band_names"]])), 3)
    # VI R2 is variance-suppressed at block scale (a near-constant index has
    # almost no variance, so R2 = 1 - err/var collapses to ~0 or negative even
    # when absolute error is tiny). Nested under `vi_r2` so it never reads as a
    # band headline; VI reliability ships as MAE (perblock_qa / the raster).
    vP, vT = _vi(P[m]), _vi(Y[m].astype(float))
    vi = {}
    for j, nm in enumerate(("NDVI", "NDRE", "CIre")):
        mm = np.isfinite(vP[j]) & np.isfinite(vT[j])
        yy, pp = vT[j][mm], vP[j][mm]
        vi[nm] = round(1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2), 3)
    out["vi_r2"] = vi
    log(f"[calibrate] pooled OOF: {out}")
    return out

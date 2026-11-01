"""The seam composite, and the small helpers the seam merge shares.

The seam merge (merge.seam_merge) measures each pair of flights with eFOLKI
(efolki.py) where they meet, splits the offsets over the flights at every pixel
of the 1 m band, and composites here: outside the band every pixel is its single
owner's, verbatim; inside it the meeting owners are moved by their share of the
offset, confined by a taper, and cross-faded.

The LoFTR machinery that used to live here (matching tiles, pooled and
corroborated cells, smooth fields, the free-gauge node solve) is at tag v0.3.4.

Ported from the audited implementations (01_merge/poc/seamwalk + feather_blend).
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

cv2.setNumThreads(1)   # callers parallelise at the block level; no nested pools


# --------------------------------------------------------------------------- #
# grayscale + coarse-to-native
# --------------------------------------------------------------------------- #
def gray_stretch(spec: np.ndarray) -> np.ndarray:
    """[0,1] contrast-stretched grayscale (the seam measurement's input): equal-weight mean of the
    (first) bands + 2-98 percentile stretch. Handles low-contrast inputs and
    int16 sources (nodata sentinels like -32768 are zeroed before the mean —
    the robustness the first-gen chip loader gained in its final WIP fix)."""
    spec = np.where(spec <= -32000, 0, spec) if spec.dtype.kind in "if" else spec
    g = spec.mean(0) if spec.ndim == 3 else spec
    m = g > 0
    if m.sum() < 100:
        return np.zeros_like(g, np.float32)
    lo, hi = np.percentile(g[m], 2), np.percentile(g[m], 98)
    return np.clip((g - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def upsample_block(arr, fac, r0, c0, bh, bw, interp=cv2.INTER_LINEAR, out=None):
    """Upsample a factor-`fac` decimated global array to a native block,
    grid-aligned (used to bring coarse geometry/fields to native windows).
    `out`: an optional flat float32 buffer the resized sub-window is written
    into (the returned block is a view of it), so a worker can reuse one
    allocation across blocks; it must hold (cr1-cr0)*fac * (cc1-cc0)*fac."""
    cr0, cc0 = r0 // fac, c0 // fac
    cr1 = min(arr.shape[0], -(-(r0 + bh) // fac))
    cc1 = min(arr.shape[1], -(-(c0 + bw) // fac))
    sub = arr[cr0:cr1, cc0:cc1].astype(np.float32)
    dh, dw = (cr1 - cr0) * fac, (cc1 - cc0) * fac
    if out is not None:
        big = cv2.resize(sub, (dw, dh), dst=out[:dh * dw].reshape(dh, dw), interpolation=interp)
    else:
        big = cv2.resize(sub, (dw, dh), interpolation=interp)
    oy, ox = r0 - cr0 * fac, c0 - cc0 * fac
    return big[oy:oy + bh, ox:ox + bw]


# --------------------------------------------------------------------------- #
# seamline geometry + tapered composite (stitching apply)
# --------------------------------------------------------------------------- #
def apply_field_confined(spec, valid, fx, fy, taper):
    """Sample spec at p + taper*field, cubic (the caller rounds and clips). Where
    taper==0 the map is the identity grid and cubic is exact at whole pixels, so
    the single-source interior is returned unchanged (byte-identical outside the
    seam band). Bilinear until 0.4.1: a softer second resample inside the band."""
    nb, H, W = spec.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = (xs + taper * fx).astype(np.float32)
    mapy = (ys + taper * fy).astype(np.float32)
    reg = np.stack([cv2.remap(spec[b], mapx, mapy, cv2.INTER_CUBIC, borderValue=0)
                    for b in range(nb)])
    vreg = cv2.remap(valid.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST, borderValue=0) > 0
    return reg, vreg


def seamline_geometry(valids, res, band_m):
    """Field-independent seam geometry from the validity masks alone:
    owner = most-interior source (argmax EDT); faultline = owner change with
    >=2 sources on both sides (excludes survey-boundary edges); taper = 1 at
    the fault -> 0 at the band edge."""
    H, W = valids[0].shape
    B = max(band_m / res, 1.0)
    vstack = np.stack(valids)
    ds = np.stack([distance_transform_edt(v).astype(np.float32) for v in valids])
    any_valid = np.any(vstack, 0)
    cov2 = np.sum(vstack, 0) >= 2
    owner = np.where(any_valid, np.argmax(ds, 0).astype(np.int16), np.int16(-1))

    fault = np.zeros((H, W), bool)
    chg = (owner[:, :-1] != owner[:, 1:]) & cov2[:, :-1] & cov2[:, 1:]
    fault[:, :-1] |= chg; fault[:, 1:] |= chg
    chg = (owner[:-1, :] != owner[1:, :]) & cov2[:-1, :] & cov2[1:, :]
    fault[:-1, :] |= chg; fault[1:, :] |= chg

    D = (distance_transform_edt(~fault).astype(np.float32) if fault.any()
         else np.full((H, W), 1e9, np.float32))
    in_band = (D <= B) & cov2
    taper = (np.clip(1.0 - D / B, 0.0, 1.0) * in_band).astype(np.float32)
    return dict(B=B, ds=ds, any_valid=any_valid, owner=owner, D=D,
                in_band=in_band, taper=taper, edt_max=ds.max(0))


def seamline_composite(specs, valids, res, band_m, fields=None, geom=None):
    """N-way seam composite: outside the band every pixel is the single owner
    VERBATIM; within band_m of a fault the meeting owners are cross-faded and
    (if fields are given) each source's shift is applied, confined by the
    taper. Triple-junction safe. Returns (out float32, alpha uint8, diag)."""
    n = len(specs); ns = specs[0].shape[0]; H, W = valids[0].shape
    g = geom if geom is not None else seamline_geometry(valids, res, band_m)
    B, ds, any_valid = g["B"], g["ds"], g["any_valid"]
    owner, in_band, taper, edt_max = g["owner"], g["in_band"], g["taper"], g["edt_max"]

    shifted = [apply_field_confined(specs[i], valids[i], fields[i][0], fields[i][1], taper)
               if fields is not None else (specs[i], valids[i]) for i in range(n)]

    out = np.zeros((ns, H, W), np.float32)
    outside = any_valid & ~in_band
    for i in range(n):
        sel = outside & (owner == i)
        out[:, sel] = specs[i][:, sel]                       # verbatim: no shift, no blend
    acc = np.zeros((ns, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for i in range(n):
        si, vi = shifted[i]
        w = (np.clip(1.0 - (edt_max - ds[i]) / B, 0.0, 1.0) * vi * in_band).astype(np.float32)
        acc += w[None] * si; wsum += w
    nz = in_band & (wsum > 0)
    for c in range(ns):
        out[c][nz] = acc[c][nz] / wsum[nz]

    return np.clip(out, 0, 255), np.where(any_valid, 255, 0).astype(np.uint8), \
        dict(owner=owner, in_band=in_band, taper=taper, band_frac=float(in_band.mean()))

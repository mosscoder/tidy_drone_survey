"""The shared correction engine: one recipe, used by stitching and alignment.

    Find how two overlapping images disagree by matching many points between
    them (LoFTR on grayscale tiles). Turn those matches into a SMOOTH
    correction that varies gently across the map (per-tile outlier rejection
    -> per-cell pooled medians -> corroboration against neighbours -> linear
    field with nearest fill). Apply it only near the seam (tapered) or move
    a whole mission toward the reference. Judge the result by agreement with
    a trusted reference, never by the fit's own numbers.

What changes between the callers is only *what the correction carries* and
*what is treated as already correct*:

  - merge.seam_merge : per-seam-pair fields, FREE-GAUGE joint solve (no
    flight is the reference; corrections split evenly), applied in a tapered
    band so interiors stay byte-identical.
  - registration.register_survey_dense : one field per mission, ANCHORED
    (the base map is the truth; the mission moves fully).

Ported from the audited implementations (01_merge/poc/seamwalk + feather_blend,
02_registration/poc/m2_full) — the code paths that produced the audit's
validated numbers (seam r 0.40->0.62; registration median r 0.685->0.863).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import cv2
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt

cv2.setNumThreads(1)   # callers parallelise at the block level; no nested pools


# --------------------------------------------------------------------------- #
# grayscale + fields
# --------------------------------------------------------------------------- #
def gray_stretch(spec: np.ndarray) -> np.ndarray:
    """[0,1] contrast-stretched grayscale for LoFTR: equal-weight mean of the
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


def griddata_fill(pts, vals, EX, EY):
    """Linear interpolation with nearest-neighbour fill outside the hull."""
    if len(vals) >= 4:
        lin = griddata(pts, vals, (EX, EY), method="linear")
        near = griddata(pts, vals, (EX, EY), method="nearest")
        return np.where(np.isfinite(lin), lin, near)
    if len(vals) >= 1:
        return griddata(pts, vals, (EX, EY), method="nearest")
    return np.full(EX.shape, np.nan)


def fill_field(pts_px, vals, out_w, out_h, sample_ds=8) -> np.ndarray:
    """Sparse (x,y)->val samples -> dense (out_h,out_w) float32 field.
    Evaluated on a sample_ds-decimated lattice then bilinearly resized —
    the corrections are smooth, so sparse evaluation reconstructs them."""
    if len(vals) < 1:
        return np.zeros((out_h, out_w), np.float32)
    ex = np.arange(0, out_w, sample_ds)
    ey = np.arange(0, out_h, sample_ds)
    EX, EY = np.meshgrid(ex, ey)
    g = np.nan_to_num(griddata_fill(np.asarray(pts_px, float), np.asarray(vals), EX, EY))
    return cv2.resize(g.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def sample_field_nodes(pts_px, vals, out_w, out_h, node_r, node_c, sample_ds=8):
    """Value of fill_field(pts,vals,out_w,out_h) at the coarse nodes (node_r,
    node_c) WITHOUT materializing the full (out_h,out_w) field — the memory
    win for the joint solve, which only needs the field at corridor nodes.

    Builds the SAME sample_ds lattice as fill_field, then samples it with the
    same INTER_LINEAR kernel cv2.resize uses (via cv2.remap with the matching
    (p+0.5)*scale-0.5 coordinate convention). Validated bit-close to
    fill_field(...)[node] in the corridor regime — see the solve-equivalence
    test (max |Δ| < 1e-3 px where the solve operates; divergence only in deep
    extrapolation past the correspondence hull, which corridor_c excludes)."""
    node_r = np.asarray(node_r, np.float32); node_c = np.asarray(node_c, np.float32)
    n = len(node_r)
    if len(vals) < 1 or n == 0:
        return np.zeros(n, np.float32)
    ex = np.arange(0, out_w, sample_ds); ey = np.arange(0, out_h, sample_ds)
    EX, EY = np.meshgrid(ex, ey)
    g = np.nan_to_num(griddata_fill(np.asarray(pts_px, float),
                                    np.asarray(vals), EX, EY)).astype(np.float32)
    lat_h, lat_w = g.shape
    mapx = ((node_c + 0.5) * (lat_w / out_w) - 0.5).astype(np.float32)
    mapy = ((node_r + 0.5) * (lat_h / out_h) - 0.5).astype(np.float32)
    out = np.empty(n, np.float32)
    CH = 30000                                   # cv2.remap needs each map dim < SHRT_MAX
    for i in range(0, n, CH):
        out[i:i + CH] = cv2.remap(g, mapx[i:i + CH].reshape(1, -1),
                                  mapy[i:i + CH].reshape(1, -1), cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE).ravel()
    return out


def upsample_block(arr, fac, r0, c0, bh, bw, interp=cv2.INTER_LINEAR):
    """Upsample a factor-`fac` decimated global array to a native block,
    grid-aligned (used to bring coarse geometry/fields to native windows)."""
    cr0, cc0 = r0 // fac, c0 // fac
    cr1 = min(arr.shape[0], -(-(r0 + bh) // fac))
    cc1 = min(arr.shape[1], -(-(c0 + bw) // fac))
    sub = arr[cr0:cr1, cc0:cc1].astype(np.float32)
    big = cv2.resize(sub, ((cc1 - cc0) * fac, (cr1 - cr0) * fac), interpolation=interp)
    oy, ox = r0 - cr0 * fac, c0 - cc0 * fac
    return big[oy:oy + bh, ox:ox + bw]


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def loftr_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_matcher(device=None):
    import torch
    import kornia.feature as KF
    device = device or loftr_device()
    return KF.LoFTR(pretrained="outdoor").to(device).eval(), device


def release_matcher_cache(device):
    """Return the accelerator's cached allocator memory to the OS. torch's
    MPS (and CUDA) caching allocator grows across thousands of match_tile
    calls and never shrinks on its own — on Apple Silicon unified memory
    that growth counts against system RAM (observed ~100 GB process
    footprint over a long seam walk). Frees only cached, unused blocks, so
    it is always safe; the next allocations just pay to re-allocate. Call
    every ~100 tiles and at phase boundaries (del the matcher first at a
    boundary so its weights free too)."""
    import gc
    import torch
    gc.collect()
    dtype = getattr(device, "type", None)
    if dtype == "mps":
        torch.mps.empty_cache()
    elif dtype == "cuda":
        torch.cuda.empty_cache()


def match_tile(matcher, device, ga, gb, reject_px=4.0, min_tile=8):
    """LoFTR on one grayscale tile pair -> (anchors_px, deltas_px) after the
    per-tile robust gate: keep matches within reject_px of the tile's median
    displacement; drop the tile if fewer than min_tile survive."""
    import torch
    ta = torch.from_numpy(ga)[None, None].to(device)
    tb = torch.from_numpy(gb)[None, None].to(device)
    with torch.inference_mode():
        out = matcher({"image0": ta, "image1": tb})
    k0 = out["keypoints0"].cpu().numpy()
    k1 = out["keypoints1"].cpu().numpy()
    if len(k0) < 4:
        return None
    d = k1 - k0
    med = np.median(d, axis=0)
    keep = np.hypot(*(d - med).T) < reject_px
    if keep.sum() < min_tile:
        return None
    return k0[keep], d[keep]


def pool_to_cells(an, d, fs=64, min_cell=2):
    """Pool raw matches into fs-px cells -> per-cell (mean anchor, median
    displacement). Sparse-but-robust nodes for the smooth field."""
    cyi = (an[:, 1] // fs).astype(np.int64)
    cxi = (an[:, 0] // fs).astype(np.int64)
    acc = defaultdict(list)
    for j in range(len(an)):
        acc[(cyi[j], cxi[j])].append((an[j, 0], an[j, 1], d[j, 0], d[j, 1]))
    pts, vx, vy = [], [], []
    for lst in acc.values():
        a = np.array(lst)
        if len(a) < min_cell:
            continue
        pts.append((a[:, 0].mean(), a[:, 1].mean()))
        m = np.median(a[:, 2:4], axis=0)
        vx.append(m[0]); vy.append(m[1])
    if not pts:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    return np.array(pts, float), np.array(vx), np.array(vy)


def corroborate_cells(pts, vx, vy, fs=64, min_nb=8, reject_px=4.0):
    """Neighbourhood-corroboration gate: a cell survives only if >= min_nb of
    its 5x5 ring agree, and its own displacement sits within reject_px of the
    ring median. Kills isolated hallucinated matches."""
    if len(pts) == 0:
        return np.zeros(0, bool)
    cyi = (pts[:, 1] // fs).astype(np.int64)
    cxi = (pts[:, 0] // fs).astype(np.int64)
    idx = {(cyi[j], cxi[j]): j for j in range(len(pts))}
    keep = np.ones(len(pts), bool)
    for j in range(len(pts)):
        nx, ny = [], []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                k = idx.get((cyi[j] + dy, cxi[j] + dx))
                if k is not None:
                    nx.append(vx[k]); ny.append(vy[k])
        if len(nx) < min_nb:
            keep[j] = False
        elif math.hypot(vx[j] - np.median(nx), vy[j] - np.median(ny)) > reject_px:
            keep[j] = False
    return keep


# --------------------------------------------------------------------------- #
# the free-gauge joint solve (stitching)
# --------------------------------------------------------------------------- #
def solve_free_gauge(valid_c, efield_sp, corridor_c, spacing_c, n_sources,
                     w_gauge=0.05, anchored_index=None, sample_ds=8):
    """Joint solve over ALL pair edges on corridor nodes: for every node where
    sources i and j overlap, (x_j - x_i) should equal the measured seam field;
    a small gauge ridge (w_gauge * x = 0) pins the free translation. With
    anchored_index set, that source's unknowns are excluded (held at zero) —
    the 'pin everything to one reference flight' fallback.

    Memory-frugal: the pair fields are SPARSE (never densified), sampled at the
    solve nodes on the fly, and each solved per-source field is returned as its
    sample_ds LATTICE (~Wc/8 x Hc/8, tens of MB) rather than the full 2.7 GB
    coarse field. The composite upsamples the lattice per block. This removes
    the ~114 GB the old dense path built at the walk->solve boundary.

    valid_c   : list of n coarse validity masks (Hc,Wc)
    efield_sp : {(i,j): (pts, vx, vy)} SPARSE seam correspondences (coarse px)
    corridor_c: coarse mask of nodes to solve on (near-fault corridor)
    Returns   : (lat, sol, n_nodes, n_unknowns)
                lat[oi] = (fx_lat, fy_lat) float32 (Hc//sd, Wc//sd), or None
                sol[oi] = (P, cx, cy) sparse solved nodes for source oi, or None
                          (lets a caller evaluate solved values anywhere cheaply)
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr

    Hc, Wc = valid_c[0].shape
    rr = list(range(spacing_c // 2, Hc, spacing_c))
    cc = list(range(spacing_c // 2, Wc, spacing_c))
    nodes = [(r, c) for r in rr for c in cc if corridor_c[r, c]]
    nr = np.array([r for r, c in nodes]); nc = np.array([c for r, c in nodes])
    var = {}
    for oi in range(n_sources):
        if oi == anchored_index:
            continue
        V = valid_c[oi]
        for (r, c) in nodes:
            if V[r, c]:
                var[(oi, r, c)] = len(var)

    rows, cols, data, bx, by = [], [], [], [], []
    nrow = [0]

    def add(terms, rx, ry):
        for vi, co in terms:
            rows.append(nrow[0]); cols.append(vi); data.append(co)
        bx.append(rx); by.append(ry); nrow[0] += 1

    pair_terms = {}                              # (oi,oj) -> [(ti,tj,fxv,fyv)] for the residual
    for (oi, oj), (pts, vx, vy) in efield_sp.items():
        co = valid_c[oi] & valid_c[oj]
        # dense-equivalent field values at the nodes, without the full field
        fxn = sample_field_nodes(pts, vx, Wc, Hc, nr, nc, sample_ds)
        fyn = sample_field_nodes(pts, vy, Wc, Hc, nr, nc, sample_ds)
        terms = pair_terms.setdefault((oi, oj), [])
        for k, (r, c) in enumerate(nodes):
            if not co[r, c]:
                continue
            ti = var.get((oi, r, c)); tj = var.get((oj, r, c))
            fxv, fyv = float(fxn[k]), float(fyn[k])
            if oi == anchored_index and tj is not None:      # x_j = f
                add([(tj, 1.0)], fxv, fyv); terms.append((None, tj, fxv, fyv))
            elif oj == anchored_index and ti is not None:    # -x_i = f
                add([(ti, -1.0)], fxv, fyv); terms.append((ti, None, fxv, fyv))
            elif ti is not None and tj is not None:          # x_j - x_i = f
                add([(tj, 1.0), (ti, -1.0)], fxv, fyv); terms.append((ti, tj, fxv, fyv))
    for (oi, r, c), vi in var.items():
        add([(vi, w_gauge)], 0.0, 0.0)

    if nrow[0] == 0 or len(var) == 0:
        return [None] * n_sources, [None] * n_sources, {}, 0, 0
    A = sp.coo_matrix((data, (rows, cols)), shape=(nrow[0], len(var))).tocsr()
    cx = lsqr(A, np.array(bx), atol=1e-8, btol=1e-8)[0]
    cy = lsqr(A, np.array(by), atol=1e-8, btol=1e-8)[0]

    # post-solve residual per pair: median |(x_j - x_i) - f_meas| at its nodes
    resid = {}
    for pr, terms in pair_terms.items():
        rr = [np.hypot((0.0 if tj is None else cx[tj]) - (0.0 if ti is None else cx[ti]) - fxv,
                       (0.0 if tj is None else cy[tj]) - (0.0 if ti is None else cy[ti]) - fyv)
              for ti, tj, fxv, fyv in terms]
        resid[pr] = float(np.median(rr)) if rr else None

    ex = np.arange(0, Wc, sample_ds); ey = np.arange(0, Hc, sample_ds)
    EX, EY = np.meshgrid(ex, ey)
    lat = [None] * n_sources; sol = [None] * n_sources
    for oi in range(n_sources):
        P, vxo, vyo = [], [], []
        for (r, c) in nodes:
            vi = var.get((oi, r, c))
            if vi is not None:
                P.append((c, r)); vxo.append(cx[vi]); vyo.append(cy[vi])
        if not P:
            continue
        P = np.array(P, float)
        sol[oi] = (P, np.asarray(vxo), np.asarray(vyo))
        lat[oi] = (np.nan_to_num(griddata_fill(P, vxo, EX, EY)).astype(np.float32),
                   np.nan_to_num(griddata_fill(P, vyo, EX, EY)).astype(np.float32))
    return lat, sol, resid, len(nodes), len(var)


# --------------------------------------------------------------------------- #
# seamline geometry + tapered composite (stitching apply)
# --------------------------------------------------------------------------- #
def apply_field_confined(spec, valid, fx, fy, taper):
    """Sample spec at p + taper*field. Where taper==0 the map is the identity
    grid, so the single-source interior is returned unchanged (byte-identical
    outside the seam band)."""
    nb, H, W = spec.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = (xs + taper * fx).astype(np.float32)
    mapy = (ys + taper * fy).astype(np.float32)
    reg = np.stack([cv2.remap(spec[b], mapx, mapy, cv2.INTER_LINEAR, borderValue=0)
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

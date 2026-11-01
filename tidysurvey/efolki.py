"""eFOLKI: the dense optical flow both registration and the seam merge measure with.

For every pixel eFOLKI takes a window around it and solves for the one shift that best lines up the two
images there (Lucas-Kanade), on rank transforms of the images, so brightness and colour differences between
dates matter little, coarse to fine on an image pyramid. The result is a shift at every pixel.

    u, v = flow(reference, moving)       # moving(x + u, y + v) lines up with reference(x, y), in pixels

Sources, read in full:
  - Brigot, Colin-Koeniguer, Plyer and Janez 2016, IEEE J. Sel. Topics Appl. Earth Obs. Remote Sens. 9(7),
    2923-2939 (GeFolki, the eFOLKI family applied to radar, lidar and optical images of forest).
  - Haynes, Lucieer, Turner and Cimoli 2025, Drones 9, 132 (eFOLKI registering drone imaging spectroscopy
    to an RGB orthomosaic).
  - The authors' code, github.com/aplyer/gefolki, commit 3a77736 (python/: pyramid.BurtOF, folki.EFolkiIter,
    rank.rank_sup). GPL-3.0.

This module is a PyTorch port of that reference Python, line for line, so it runs on NVIDIA GPUs (CUDA),
Apple GPUs (MPS) or the CPU. It keeps the reference's behaviour, including the parts that shape the result:
min-max normalisation of both images; the Burt pyramid (5-tap kernel a = 0.4, zero padding, every second
pixel kept); the rank transform counting neighbours greater than the centre, zeros outside the image;
np.gradient (central differences, one-sided at the edges); window averages with zero padding, divided by
the full window size; the bilinear warp with edge replication; the flow passed to the next level by pixel
repetition, times 2; NaN and Inf flow set to 0. It differs in float32 (MPS has no float64) and the order of
floating-point sums. Checked against the reference in the audit (02_registration/matcher_ablation/poc/
check_port.py, cuda_check_synth.py): at most 0.05 px apart on three 1024 px crops, 0.003 px or less at the
99th percentile; at most 0.002 px on an NVIDIA L4. As a derivative of GPL-3.0 code it is GPL-3.0, and so is
tidysurvey (LICENSE).

SETTINGS are the adopted ones (audit 02_registration/METHOD.md): window radii 32, 24, 16, 8 (the authors'
eFOLKI example), 8 iterations per window and rank radius 4 (Haynes et al.), and 3 pyramid levels (ours: the
parameter is published, the value is not; the coarsest copy of a 3.3 cm map then has 26 cm pixels, which
keeps the ground's grain and stops the shadow slides seen at about 50 cm). Measure on the reference map's
own pixels; never on a coarser copy.
"""
from __future__ import annotations

import os
import threading

import numpy as np
from scipy.ndimage import distance_transform_edt

SETTINGS = dict(iteration=8, radius=(32, 24, 16, 8), rank=4, levels=3)
REACH_PX = max(SETTINGS["radius"]) * 2 ** SETTINGS["levels"]   # the coarsest window's half-width, in full-resolution pixels
TALON = 1.0e-8
_LOCK = threading.Lock()                                         # one flow on the device at a time; callers run in threads


def device(name: str | None = None):
    """The torch device: `name`, else EFOLKI_DEVICE, else CUDA, then Apple MPS, then the CPU."""
    import torch
    name = name or os.environ.get("EFOLKI_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fill_nearest(g: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """No-data filled from the nearest valid pixel, so the window sums never see a cliff at a map's edge
    (our step; the sources do not describe one)."""
    if valid.all() or not valid.any():
        return g
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return g[idx[0], idx[1]]


def _sep(I, k):
    """Separable correlation with a symmetric 1-D kernel, zero padding, same size (conv2SepMatlab)."""
    import torch.nn.functional as Fn
    r = (k.numel() - 1) // 2
    k = k.to(I.device, I.dtype)
    I = Fn.conv2d(I, k.view(1, 1, -1, 1), padding=(r, 0))
    return Fn.conv2d(I, k.view(1, 1, 1, -1), padding=(0, r))


def _box(I, rad):
    """Mean over a (2 rad + 1)^2 window with zeros outside the image."""
    import torch
    return _sep(I, torch.full((2 * rad + 1,), 1.0 / (2 * rad + 1)))


def _pyr_up(I):
    """BurtOF.pyrUp: smooth with the Burt kernel, keep every second row and column."""
    import torch
    burt = torch.tensor([0.25 - 0.4 / 2, 0.25, 0.4, 0.25, 0.25 - 0.4 / 2])
    return _sep(I, burt)[..., ::2, ::2]


def _pyr_down(I, shape):
    """BurtOF.pyrDown: repeat every pixel 2 x 2, crop (or zero-fill) to shape."""
    import torch
    J = I.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    out = torch.zeros((1, 1) + tuple(shape), device=I.device, dtype=I.dtype)
    r, c = min(shape[0], J.shape[-2]), min(shape[1], J.shape[-1])
    out[..., :r, :c] = J[..., :r, :c]
    return out


def _rank_sup(I, rad):
    """rank.rank_sup: per pixel, the number of neighbours in the window greater than it; zeros outside."""
    import torch
    import torch.nn.functional as Fn
    H, W = I.shape[-2:]
    P = Fn.pad(I, (rad, rad, rad, rad))
    R = torch.zeros_like(I)
    for i in range(-rad, rad + 1):
        for j in range(-rad, rad + 1):
            R += (P[..., rad - i:rad - i + H, rad - j:rad - j + W] > I).to(I.dtype)
    return R


def _gradient(I):
    """np.gradient: (d/drow, d/dcol); central inside, one-sided first order at the edges."""
    import torch
    gy = torch.empty_like(I); gx = torch.empty_like(I)
    gy[..., 1:-1, :] = (I[..., 2:, :] - I[..., :-2, :]) / 2
    gy[..., 0, :] = I[..., 1, :] - I[..., 0, :]
    gy[..., -1, :] = I[..., -1, :] - I[..., -2, :]
    gx[..., :, 1:-1] = (I[..., :, 2:] - I[..., :, :-2]) / 2
    gx[..., :, 0] = I[..., :, 1] - I[..., :, 0]
    gx[..., :, -1] = I[..., :, -1] - I[..., :, -2]
    return gy, gx


def _warp(I, u, v, base):
    """interp2: bilinear sample of I at (x + u, y + v), edge replicated outside."""
    import torch
    import torch.nn.functional as Fn
    H, W = I.shape[-2:]
    x = (base[0] + u).clamp(0, W - 1)                  # edge replication = clamping (MPS has no padding_mode="border")
    y = (base[1] + v).clamp(0, H - 1)
    gx = x * (2.0 / max(W - 1, 1)) - 1.0
    gy = y * (2.0 / max(H - 1, 1)) - 1.0
    grid = torch.stack([gx[0, 0], gy[0, 0]], dim=-1)[None]
    return Fn.grid_sample(I, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _efolki_iter(I0, I1, iteration, radius, rank, u, v):
    """folki.EFolkiIter on one pyramid level."""
    import torch
    if rank > 0:
        I0 = _rank_sup(I0, rank)
        I1 = _rank_sup(I1, rank)
    Iy, Ix = _gradient(I0)
    H, W = I0.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=I0.device, dtype=I0.dtype),
                            torch.arange(W, device=I0.device, dtype=I0.dtype), indexing="ij")
    base = (xx[None, None], yy[None, None])
    for rad in radius:
        Ixx = _box(Ix * Ix, rad) + TALON
        Iyy = _box(Iy * Iy, rad) + TALON
        Ixy = _box(Ix * Iy, rad)
        D = Ixx * Iyy - Ixy ** 2
        for _ in range(iteration):
            it = I0 - _warp(I1, u, v, base) + u * Ix + v * Iy
            Ixt = _box(Ix * it, rad)
            Iyt = _box(Iy * it, rad)
            un = (Iyy * Ixt - Ixy * Iyt) / D
            vn = (Ixx * Iyt - Ixy * Ixt) / D
            bad = ~torch.isfinite(un) | ~torch.isfinite(vn)
            u = torch.where(bad, torch.zeros_like(un), un)
            v = torch.where(bad, torch.zeros_like(vn), vn)
    return u, v


def flow(reference: np.ndarray, moving: np.ndarray, dev=None, iteration=None, radius=None, rank=None, levels=None):
    """pyramid.BurtOF(EFolkiIter): two 2-D images on one grid in, the float32 flow (u, v) on that grid out, in
    pixels: moving(x + u, y + v) lines up with reference(x, y). Settings default to SETTINGS."""
    import torch
    _tf32_off()
    s = dict(SETTINGS)
    s.update({k: v for k, v in dict(iteration=iteration, radius=radius, rank=rank, levels=levels).items() if v is not None})
    dev = dev if dev is not None and not isinstance(dev, str) else device(dev)
    with _LOCK, torch.inference_mode():
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None].to(dev)
        I0, I1 = t(reference), t(moving)
        I0 = (I0 - I0.min()) / (I0.max() - I0.min())
        I1 = (I1 - I1.min()) / (I1.max() - I1.min())
        P0, P1 = [I0], [I1]
        for _ in range(s["levels"], 0, -1):
            P0.append(_pyr_up(P0[-1])); P1.append(_pyr_up(P1[-1]))
        u = torch.zeros_like(P0[-1]); v = torch.zeros_like(P0[-1])
        for i in range(s["levels"], -1, -1):
            u, v = _efolki_iter(P0[i], P1[i], s["iteration"], list(s["radius"]), s["rank"], u, v)
            if i > 0:
                shape = P0[i - 1].shape[-2:]
                u = 2 * _pyr_down(u, shape); v = 2 * _pyr_down(v, shape)
        out = (u[0, 0].float().cpu().numpy(), v[0, 0].float().cpu().numpy())
        del P0, P1, u, v, I0, I1
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _tf32_off():
    """CUDA convolutions default to TF32 on recent NVIDIA GPUs (about 3 significant digits); keep full float32."""
    try:
        import torch
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    except ImportError:
        pass


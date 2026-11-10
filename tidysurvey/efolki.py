"""eFOLKI: the dense optical flow both registration and the seam merge measure with.

For every pixel eFOLKI takes a window around it and solves for the one shift that best lines up the two
images there (Lucas-Kanade), on rank transforms of the images, so brightness and colour differences between
dates matter little, coarse to fine on an image pyramid. The result is a shift at every pixel.

    u, v = flow(reference, moving)       # moving(x + u, y + v) lines up with reference(x, y), in pixels
    fields = flows(reference, [a, b])    # several moving images against one reference: [(u, v), (u, v)]

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
the full window size (by cumulative sums on a CPU, _box); the bilinear warp with edge replication; the flow passed to the next level by pixel
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

EFOLKI_VERSION names the numbers this module gives, apart from the package's version (efolki.py did not change
between tidysurvey 0.4.2 and 0.5.0, so the package's version does not say whether the flow did). Raise it when a
change can alter a flow: the settings, the pyramid, the rank transform, the window sums, the warp, the solve.
Leave it for changes that keep the numbers, like flows(), which gives flow()'s numbers faster. Callers that keep
results write it beside them (the ODM fork's registration records do).
"""
from __future__ import annotations

import os
import threading

import numpy as np
from scipy.ndimage import distance_transform_edt

EFOLKI_VERSION = 1                                               # the flow's numbers as of tidysurvey 0.4.0 to 0.5.0; see above
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
    """Mean over a (2 rad + 1)^2 window with zeros outside the image. On a GPU, the separable correlation of the
    reference code. On a CPU, the same sums by cumulative sums (summed in float64), which the season VM's cores do
    twice as fast as the 65-tap convolutions (24.5 s against 47.7 s per seam box, 2026-11-01); the two agree to
    3e-5 on window means of rank values, and GPUs run the convolution faster."""
    import torch
    if I.device.type != "cpu":
        return _sep(I, torch.full((2 * rad + 1,), 1.0 / (2 * rad + 1)))
    import torch.nn.functional as Fn
    k = 2 * rad + 1
    c = Fn.pad(torch.cumsum(Fn.pad(I, (rad, rad, 0, 0)).to(torch.float64), -1), (1, 0, 0, 0))
    J = c[..., k:] - c[..., :-k]
    c = Fn.pad(torch.cumsum(Fn.pad(J, (0, 0, rad, rad)), -2), (0, 0, 1, 0))
    return ((c[..., k:, :] - c[..., :-k, :]) / (k * k)).to(I.dtype)


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


def _warp_many(I, u, v, base):
    """_warp for several images at once: each sampled bilinearly at its own (x + u, y + v), edge replicated outside.
    (_warp takes one image: it builds the sampling grid from the first image's flow alone.)"""
    import torch
    import torch.nn.functional as Fn
    H, W = I.shape[-2:]
    x = (base[0] + u).clamp(0, W - 1)
    y = (base[1] + v).clamp(0, H - 1)
    gx = x * (2.0 / max(W - 1, 1)) - 1.0
    gy = y * (2.0 / max(H - 1, 1)) - 1.0
    grid = torch.stack([gx[:, 0], gy[:, 0]], dim=-1)
    return Fn.grid_sample(I, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _pyr_down_many(I, shape):
    """_pyr_down for several images at once: every pixel repeated 2 x 2, cropped (or zero-filled) to shape."""
    import torch
    J = I.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    out = torch.zeros(tuple(I.shape[:2]) + tuple(shape), device=I.device, dtype=I.dtype)
    r, c = min(shape[0], J.shape[-2]), min(shape[1], J.shape[-1])
    out[..., :r, :c] = J[..., :r, :c]
    return out


def flows(reference: np.ndarray, movings: list, dev=None, batch: int | None = None, iteration=None, radius=None, rank=None,
          levels=None) -> list:
    """The flows of several moving images against one reference, all on one grid: [(u, v), ...], each the numbers
    flow(reference, moving) gives (tests/test_efolki.py holds them equal on the CPU, bit for bit).

    flow() takes one pair and works both sides of it. When several images share a reference (a multispectral
    capture's bands against its green band, in the ODM fork's registration) the work is shared here:
    1. The reference's side is worked once: its pyramid, rank transforms, gradients and the window means of their
       products.
    2. The moving images go through the iterations together, `batch` at a time: by default all of them at once on a
       GPU, which does that faster than one after another, and one at a time on the CPU.
    It follows flow() and _efolki_iter line for line, with the same functions in the same order, except _warp and
    _pyr_down, which take one image and are written for several above. Written for the ODM fork's registration
    (aerial/odm_fork/ms_register.py, 2026-11-09), where it was one of three changes that together made a capture
    2.1 times faster on an M1 Max and 1.36 times on an NVIDIA L4 with the same output bit for bit, and moved here
    2026-11-10 so eFOLKI has one home."""
    import torch
    _tf32_off()
    s = dict(SETTINGS)
    s.update({k: v for k, v in dict(iteration=iteration, radius=radius, rank=rank, levels=levels).items() if v is not None})
    dev = dev if dev is not None and not isinstance(dev, str) else device(dev)
    n_levels, rk = s["levels"], s["rank"]
    ranked = lambda I: _rank_sup(I, rk) if rk > 0 else I                             # noqa: E731
    batch = int(batch or (len(movings) if dev.type != "cpu" else 1))

    def pyramid(images):
        I = torch.cat([torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None] for a in images]).to(dev)
        lo, hi = I.amin((-2, -1), keepdim=True), I.amax((-2, -1), keepdim=True)     # each image by its own range
        P = [(I - lo) / (hi - lo)]
        for _ in range(n_levels):
            P.append(_pyr_up(P[-1]))
        return P

    out = []
    with _LOCK, torch.inference_mode():
        P0 = pyramid([reference])
        side = []                                             # per pyramid level, all that depends on the reference alone
        for I0 in P0:
            R0 = ranked(I0)
            Iy, Ix = _gradient(R0)
            H, W = R0.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(H, device=R0.device, dtype=R0.dtype),
                                    torch.arange(W, device=R0.device, dtype=R0.dtype), indexing="ij")
            windows = []
            for rad in s["radius"]:
                Ixx = _box(Ix * Ix, rad) + TALON
                Iyy = _box(Iy * Iy, rad) + TALON
                Ixy = _box(Ix * Iy, rad)
                windows.append((rad, Ixx, Iyy, Ixy, Ixx * Iyy - Ixy ** 2))
            side.append((R0, Iy, Ix, (xx[None, None], yy[None, None]), windows))
        for k in range(0, len(movings), batch):
            P1 = pyramid(movings[k:k + batch])
            u = torch.zeros_like(P1[-1]); v = torch.zeros_like(P1[-1])
            for i in range(n_levels, -1, -1):
                R0, Iy, Ix, base, windows = side[i]
                R1 = ranked(P1[i])
                for rad, Ixx, Iyy, Ixy, D in windows:
                    for _ in range(s["iteration"]):
                        it = R0 - _warp_many(R1, u, v, base) + u * Ix + v * Iy
                        Ixt = _box(Ix * it, rad)
                        Iyt = _box(Iy * it, rad)
                        un = (Iyy * Ixt - Ixy * Iyt) / D
                        vn = (Ixx * Iyt - Ixy * Ixt) / D
                        bad = ~torch.isfinite(un) | ~torch.isfinite(vn)
                        u = torch.where(bad, torch.zeros_like(un), un)
                        v = torch.where(bad, torch.zeros_like(vn), vn)
                if i > 0:
                    shape = P0[i - 1].shape[-2:]
                    u = 2 * _pyr_down_many(u, shape); v = 2 * _pyr_down_many(v, shape)
            U, V = u[:, 0].float().cpu().numpy(), v[:, 0].float().cpu().numpy()
            out.extend((U[j], V[j]) for j in range(len(U)))
            del P1, R1, u, v
        del P0, side
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


"""Band law: which bands of a raster are spectral data and which one is validity (alpha).

What DroneDeploy exports actually carry (verified pixel-for-pixel on the bucket, 2026-09-04):

    visible         RGBA uint8 — alpha TAGGED (ColorInterp.alpha on band 4)
    multispectral   2024:  5 bands  R, G, NIR, RE, NIR again           — no alpha, nodata unset
                    2025+: 6 bands  R, G, NIR, RE, NIR again, alpha    — alpha TAGGED (0/255)

The colour tags on the multispectral files are wrong (band 3 is tagged 'blue' and is NIR), so
only the ALPHA tag is trusted. Spectral identity comes from the survey config's band names:
the first len(names) non-alpha bands, in order. Duplicates and untagged trailing bands are
dropped, never carried.

Validity, one rule everywhere (registration, merge, calibrate, validate):
  * a tagged alpha band IS the validity, read exactly (> 127 after any resampling);
  * with no alpha, validity is derived from the data — border-connected all-zero pixels are
    nodata, interior all-zero islands are data (audit 02_registration §3);
  * legacy exception: an UNTAGGED 4-band uint8 raster is RGBA by convention (pre-tag visible
    orthos and tidysurvey's own older outputs), last band = alpha.

Every tidysurvey writer tags its alpha band and names its bands (tag_output), so nothing
downstream has to guess twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from scipy.ndimage import label as _ndlabel


@dataclass(frozen=True)
class BandLaw:
    count: int                      # bands in the file
    spectral: Tuple[int, ...]       # 1-based band indexes carried as data, in order
    alpha: Optional[int]            # 1-based validity band; None = derive from zeros
    names: Tuple[str, ...]          # one per spectral band ("band_<i>" when unnamed)

    @property
    def nspec(self) -> int:
        return len(self.spectral)

    def describe(self) -> str:
        a = f"alpha=band {self.alpha}" if self.alpha else "alpha=none (validity from zeros)"
        return f"{self.count} bands; spectral={list(self.spectral)} {list(self.names)}; {a}"


def band_law(path, names: Optional[Sequence[str]] = None) -> BandLaw:
    """Resolve the band law of a raster. `names` = the config's spectral band names (the
    first N non-alpha bands are those, in order); None = every non-alpha band."""
    with rasterio.open(path) as s:
        count = s.count
        ci = list(s.colorinterp or [])
        dtype = s.dtypes[0]
        desc = [d or "" for d in (s.descriptions or [])]
    alpha = next((i + 1 for i, c in enumerate(ci) if c == ColorInterp.alpha), None)
    if alpha is None and desc and desc[-1].lower() == "alpha":   # named but untagged (our writers)
        alpha = count
    if alpha is None and count == 4 and dtype == "uint8" and not names:
        alpha = 4                                                 # legacy untagged RGBA
    non_alpha = tuple(b for b in range(1, count + 1) if b != alpha)
    if names:
        names = tuple(names)
        if len(names) > len(non_alpha):
            raise ValueError(f"{path}: config names {len(names)} bands {list(names)} but the "
                             f"file has only {len(non_alpha)} non-alpha bands")
        spectral = non_alpha[:len(names)]
    else:
        spectral = non_alpha
        names = tuple(f"band_{b}" for b in spectral)
    return BandLaw(count, spectral, alpha, names)


def valid_from_zeros(data: np.ndarray) -> np.ndarray:
    """`data` = bool (h, w) 'any spectral band non-zero'. Border-connected zero = nodata;
    interior all-zero islands = data (flood fill from the raster edge). FULL-extent grids
    only — a window's edge is not the raster's edge."""
    zw = ~data
    if not zw.any():
        return data.copy()
    lab, _ = _ndlabel(zw, structure=np.ones((3, 3), int))
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    border = border[border > 0]
    return data | ((lab > 0) & ~np.isin(lab, border))


def read_valid(ds, law: BandLaw, window=None) -> np.ndarray:
    """Validity mask from an open dataset / WarpedVRT under `law`. Exact with an alpha band;
    without one, derived from the spectral bands' zeros — flood-filled when the read is the
    full extent, plain any-non-zero for a window."""
    if law.alpha:
        return ds.read(law.alpha, window=window) > 127
    data = (ds.read(list(law.spectral), window=window) != 0).any(0)
    return valid_from_zeros(data) if window is None else data


def tag_output(dst, names: Sequence[str], alpha: bool = True) -> None:
    """Tag a dataset opened for writing: band descriptions from `names` (+ 'alpha'),
    ColorInterp RGB(A) for three spectral bands, gray/undefined(+alpha) otherwise.
    Call it right after rasterio.open(..., "w"), BEFORE any pixels are written: on a
    tiled/compressed GTiff a tag set after writing is silently dropped."""
    n = len(names)
    for i, d in enumerate(list(names) + (["alpha"] if alpha else []), 1):
        dst.set_band_description(i, d)
    ci = ([ColorInterp.red, ColorInterp.green, ColorInterp.blue] if n == 3
          else [ColorInterp.gray] + [ColorInterp.undefined] * (n - 1))
    if alpha:
        ci.append(ColorInterp.alpha)
    try:
        dst.colorinterp = ci
    except Exception:                 # a driver that refuses the tag still keeps the names
        pass

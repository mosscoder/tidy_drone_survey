import gc
import os
import shutil
import tempfile

import cv2
import numpy as np
from scipy import ndimage

import rasterio
from rasterio.enums import Resampling, ColorInterp  # noqa: F401 (ColorInterp used by seam_merge)
from rasterio import features as rio_features
from rasterio.windows import Window

import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union

from affine import Affine
import joblib

def _process_tile_for_hull(args):
    """
    Process a single tile and return unioned polygon.
    Module-level function for joblib compatibility.
    Each call opens its own file handle for thread safety.

    Args:
        args: Tuple of (raster_path, window_tuple, alpha_idx, decimation)
              - decimation: Factor to downsample the tile (1 = no downsampling)

    Returns:
        A single geometry (unary_union of tile polygons) or None if no data.
    """
    raster_path, window_tuple, alpha_idx, decimation = args
    window = Window(*window_tuple)
    tile_polys = []

    try:
        with rasterio.open(raster_path) as src:
            # Calculate downsampled dimensions
            out_height = max(1, int(window.height / decimation))
            out_width = max(1, int(window.width / decimation))

            if alpha_idx is not None:
                data = src.read(
                    alpha_idx,
                    window=window,
                    out_shape=(out_height, out_width),
                    resampling=Resampling.nearest
                )
                mask = data > 0
            else:
                data = src.read(
                    window=window,
                    out_shape=(src.count, out_height, out_width),
                    resampling=Resampling.nearest
                )
                mask = np.any(data != 0, axis=0)

            if not mask.any():
                return None

            # Exact scale factors for edge tiles (handles non-divisible sizes)
            scale_x = window.width / out_width
            scale_y = window.height / out_height
            base_transform = src.window_transform(window)
            scaled_transform = base_transform * Affine.scale(scale_x, scale_y)

            shapes_gen = rio_features.shapes(
                mask.astype('uint8'),
                mask=mask,
                transform=scaled_transform
            )

            for geom, val in shapes_gen:
                if val == 1:
                    tile_polys.append(shape(geom))

        # Return unioned geometry (reduces pickle overhead)
        if tile_polys:
            return unary_union(tile_polys)
        return None

    except Exception:
        return None


def define_hull_tiled(
    raster_path: str,
    output_geojson: str,
    tile_size: int = 2048,
    tol: float = 0.5,
    simplify_tolerance: float = 1.0,
    n_workers: int = 8,
    show_progress: bool = True,
) -> tuple:
    """
    Extract hull polygon from raster using tile-based streaming with decimation.

    Args:
        raster_path: Path or URL to input raster (supports /vsicurl/)
        output_geojson: Path to output hull GeoJSON
        tile_size: Tile size in pixels for streaming
        tol: Resolution tolerance for decimation (in CRS units). Determines how
             much to downsample tiles. Lower = finer detail, higher = faster.
        simplify_tolerance: Tolerance for final polygon simplification (in CRS units)
        n_workers: Number of parallel workers for tile processing
        show_progress: Whether to show tqdm progress bar

    Returns:
        Tuple of (raster_path, output_geojson) or (raster_path, None) on failure
    """
    from tqdm import tqdm

    print(f"Extracting hull (tiled) for: {raster_path}")

    try:
        # First pass: get metadata and build window list
        with rasterio.open(raster_path) as src:
            crs = src.crs
            width, height = src.width, src.height
            native_res = src.res[0]  # meters/pixel (assumes square pixels)

            # Calculate decimation factor: tol is the target resolution
            decimation = max(1, int(tol / native_res))
            print(f"  Native res: {native_res:.4f}m, target: {tol:.4f}m, decimation: {decimation}x")

            # Find alpha band
            alpha_idx = None
            if src.colorinterp:
                for i, interp in enumerate(src.colorinterp):
                    if interp == ColorInterp.alpha:
                        alpha_idx = i + 1  # 1-based
                        break

            if alpha_idx is None:
                print(f"  No alpha band found, will check all bands for zeros")
            else:
                print(f"  Using alpha band {alpha_idx}")

        # Build list of tile windows (as tuples for serialization)
        windows = []
        for row_off in range(0, height, tile_size):
            for col_off in range(0, width, tile_size):
                win_height = min(tile_size, height - row_off)
                win_width = min(tile_size, width - col_off)
                windows.append((col_off, row_off, win_width, win_height))

        print(f"  Processing {len(windows)} tiles with {n_workers} workers...")

        # Build args for parallel processing (includes decimation)
        tile_args = [(raster_path, w, alpha_idx, decimation) for w in windows]

        # Process tiles in parallel with progress bar
        results = joblib.Parallel(n_jobs=n_workers, backend='loky')(
            joblib.delayed(_process_tile_for_hull)(args)
            for args in tqdm(tile_args, desc="  Tiles", disable=not show_progress, leave=False)
        )

        # Collect results - now single geometries or None
        tile_geometries = [r for r in results if r is not None]
        n_tiles_with_data = len(tile_geometries)

        print(f"  Processed {len(windows)} tiles, {n_tiles_with_data} with data")

        if not tile_geometries:
            print(f"  Warning: No valid data found")
            return raster_path, None

        # Union all tile geometries, close gaps, and simplify
        print(f"  Unioning {n_tiles_with_data} tile geometries...")
        hull = unary_union(tile_geometries)
        hull = hull.buffer(1.0).buffer(-1.0)  # close tile boundary gaps
        if simplify_tolerance > 0:
            hull = hull.simplify(simplify_tolerance, preserve_topology=True)

        # Save to GeoJSON
        gdf = gpd.GeoDataFrame({'geometry': [hull]}, crs=crs)
        gdf.to_file(output_geojson, driver='GeoJSON')
        print(f"  Saved hull to {output_geojson}")

        return raster_path, output_geojson

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return raster_path, None


def find_seamlines(input_hulls, output_dir, tol, buffer_m: float, grid_res: float = None):
    """
    Partition the full union of input hulls into seam regions via rasterized
    distance transforms, catch mis‑assigned islands, then write & return.

    Args:
        input_hulls (list): List of paths to input hull GeoJSON files.
        output_dir (str): Directory to save the output seamlines GeoJSON files.
        tol (float): Tolerance for hull generation.
        buffer_m (float): Buffer distance in CRS units to apply to seamlines.
        grid_res (float, optional): Grid resolution for distance transform. If None,
            defaults to bounded value: max(1.0, min(tol * 2, 5.0)). Coarser grids
            are faster for large areas.

    Returns:
        list: List of paths to the created seamline GeoJSON files.
    """
    seamline_files = [None] * len(input_hulls)  # Initialize a list to hold seamline file paths

    # Bounded grid resolution: floor of 1.0, ceiling of 5.0
    if grid_res is None:
        grid_res = max(1.0, min(tol * 2, 5.0))

    print("Step 1: Reading and standardizing input hulls...")
    gdfs, names = [], []
    for path in input_hulls:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"  Reading {path!r} as hull '{name}'")
        names.append(name)
        g = gpd.read_file(path).dissolve(by=lambda _: name)
        g.index = [name]
        gdfs.append(g)
    crs = gdfs[0].crs
    for i, g in enumerate(gdfs[1:], 1):
        if g.crs != crs:
            print(f"  Reprojecting hull '{names[i]}' to match CRS")
            g.to_crs(crs, inplace=True)
    polys = [g.geometry.unary_union for g in gdfs]

    print("Step 2: Building union bounds and grid metadata...")
    minx, miny, maxx, maxy = unary_union(polys).bounds
    print(f"  Bounds: ({minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f})")
    nx = int(np.ceil((maxx - minx) / grid_res))
    ny = int(np.ceil((maxy - miny) / grid_res))
    print(f"  Grid: {nx}×{ny} cells (grid_res={grid_res}, tol={tol})")
    transform = Affine.translation(minx, miny) * Affine.scale(grid_res, grid_res)

    print("Step 3: Rasterizing each hull to mask...")
    masks = []
    for i, poly in enumerate(polys):
        print(f"  Rasterizing '{names[i]}'")
        masks.append(
            rio_features.rasterize(
                [(poly, 1)],
                out_shape=(ny, nx),
                transform=transform,
                fill=0,
                dtype="uint8",
            )
        )

    print("Step 4: Computing distance transforms...")
    dist_stack = np.stack([ndimage.distance_transform_edt(m) for m in masks])
    print(f"  Distance stack shape: {dist_stack.shape}")

    print("Step 5: Labeling cells by max distance hull...")
    labels = np.argmax(dist_stack, axis=0).astype("uint8") + 1
    labels[np.all(dist_stack == 0, axis=0)] = 0

    print("Step 6: Polygonizing contiguous label regions...")
    shards = list(rio_features.shapes(labels, mask=labels > 0, transform=transform))
    print(f"  Created {len(shards)} raw seam polygons")
    parts = gpd.GeoDataFrame([
        {"label": int(val), "geometry": shape(geom)}
        for geom, val in shards
    ], crs=crs)

    print("Step 7: Mapping labels to mission names...")
    parts["owner"] = parts["label"].apply(lambda x: names[x-1] if 1 <= x <= len(names) else None)
    parts = parts[parts["owner"].notna()]
    parts = parts.drop(columns=["label"])
    print(f"  {parts['owner'].nunique()} unique owners found")

    print("Step 8: Dissolving back to one polygon per hull...")
    seam_gdf = parts.dissolve(by="owner").reset_index().rename(columns={"owner": "name"})
    print(f"  Dissolved to {len(seam_gdf)} seam polygons")

    if buffer_m > 0:
        print(f"  Buffering seam polygons by {buffer_m} (CRS units)...")
        seam_gdf['geometry'] = seam_gdf.geometry.buffer(buffer_m)
        # Ensure geometry is valid after buffering and remove empty ones
        seam_gdf['geometry'] = seam_gdf.geometry.apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
        seam_gdf = seam_gdf[~seam_gdf.geometry.is_empty]
        if seam_gdf.empty:
            print("Warning: All seam polygons became empty after buffering. No seamlines will be saved.")
            return [None] * len(input_hulls) # Return list of Nones if all are empty

    # Save each seamline as a separate GeoJSON file
    for i, row in seam_gdf.iterrows():
        print(f"row: {row}")  # Debug: see what columns are present and their types
        hull_name = row['name'] if 'name' in row else row['owner']
        hull_name = str(hull_name)
        if hull_name not in names:
            print(f"Warning: hull_name {hull_name} not found in names list {names}")
            continue
        original_index = names.index(hull_name)
        seamline_file = os.path.join(output_dir, f"seamline_{original_index + 1}.geojson")
        single_geom_gdf = gpd.GeoDataFrame(geometry=[row.geometry], crs=seam_gdf.crs)
        single_geom_gdf.to_file(seamline_file, driver="GeoJSON")
        seamline_files[original_index] = seamline_file
        print(f"  Saved seamline to {seamline_file}")

    return seamline_files  # Return the list of seamline file paths

def generate_combined_boundaries(
    raster_paths: list,
    mission_names: list,
    output_geojson: str,
    tol: float = 0.5,
    simplify_tolerance: float = None,
    buffer_m: float = 0.0,
    n_workers: int = 4,
    tile_size: int = 2048,
    clip_to: str = None,
    output_epsg: int = 6514,
) -> str:
    """
    Generate non-overlapping boundaries for multiple rasters and save as combined GeoJSON.

    Uses distance transforms to partition overlapping regions, assigning each pixel
    to the raster whose boundary is furthest away (prioritizing better-georectified
    interior pixels).

    Args:
        raster_paths: List of paths/URLs to input rasters
        mission_names: List of names for each mission (same order as raster_paths)
        output_geojson: Path to output combined GeoJSON
        tol: Boundary precision tolerance for mask generation. Default 0.5.
        simplify_tolerance: Tolerance for geometry simplification. If None, uses tol.
        buffer_m: Buffer to expand final boundaries. Default 0.0 (no expansion).
        n_workers: Parallel workers for hull generation. Default 4.
        tile_size: Tile size in pixels for streaming reads. Default 2048.
        clip_to: Path to GeoJSON/shapefile to clip output boundaries. Default None.
        output_epsg: EPSG code for output CRS. Default 6514 (Montana State Plane).

    Returns:
        Path to output GeoJSON file.
    """
    if len(raster_paths) != len(mission_names):
        raise ValueError("raster_paths and mission_names must have the same length")

    with tempfile.TemporaryDirectory() as work_dir:
        print(f"Generating boundaries for {len(raster_paths)} missions...")

        # Step 1: Generate hulls for each raster sequentially (parallelism is within each mission's tiles)
        print("Step 1: Defining hulls...")

        hull_results = []
        for i, (raster_path, name) in enumerate(zip(raster_paths, mission_names)):
            print(f"\n[{i+1}/{len(raster_paths)}] Processing {name}...")
            output_hull = os.path.join(work_dir, f"hull_{i}_{name}.geojson")
            _, hull_file = define_hull_tiled(
                raster_path=raster_path,
                output_geojson=output_hull,
                tile_size=tile_size,
                tol=tol,
                simplify_tolerance=simplify_tolerance if simplify_tolerance else tol,
                n_workers=n_workers,
                show_progress=True
            )
            hull_results.append((name, hull_file))

        # Filter successful results and maintain order
        valid_results = [(name, hull) for name, hull in hull_results if hull is not None]
        if not valid_results:
            raise RuntimeError("No valid hulls could be generated.")

        names_ordered = [r[0] for r in valid_results]
        hulls_ordered = [r[1] for r in valid_results]
        print(f"\nSuccessfully generated {len(hulls_ordered)} hulls.")

        # Step 2: Find seamlines (non-overlapping regions)
        print("Step 2: Finding seamlines (non-overlapping regions)...")
        seamline_files = find_seamlines(
            input_hulls=hulls_ordered,
            output_dir=work_dir,
            tol=tol,
            buffer_m=buffer_m
        )

        # Step 3: Combine seamlines into single GeoJSON with mission names
        print("Step 3: Combining into single GeoJSON...")
        combined_features = []

        for i, seamline_file in enumerate(seamline_files):
            if seamline_file is None:
                print(f"  Warning: No seamline for {names_ordered[i]}, skipping")
                continue

            gdf = gpd.read_file(seamline_file)
            if gdf.empty:
                print(f"  Warning: Empty seamline for {names_ordered[i]}, skipping")
                continue

            # Get the geometry and assign mission name
            geom = gdf.geometry.unary_union
            combined_features.append({
                "name": names_ordered[i],
                "geometry": geom
            })
            print(f"  Added boundary for {names_ordered[i]}")

        if not combined_features:
            raise RuntimeError("No valid boundaries to combine.")

        # Create combined GeoDataFrame
        combined_gdf = gpd.GeoDataFrame(
            [{"name": f["name"]} for f in combined_features],
            geometry=[f["geometry"] for f in combined_features],
            crs=gpd.read_file(hulls_ordered[0]).crs
        )

        # Clip to boundary if provided
        if clip_to is not None:
            print(f"Step 4: Clipping to {clip_to}...")
            clip_gdf = gpd.read_file(clip_to)
            if clip_gdf.crs != combined_gdf.crs:
                clip_gdf = clip_gdf.to_crs(combined_gdf.crs)
            clip_geom = clip_gdf.unary_union
            combined_gdf = gpd.clip(combined_gdf, clip_geom)
            print(f"  Clipped to {len(combined_gdf)} features")

        # Reproject to output CRS
        print(f"Step {'5' if clip_to else '4'}: Reprojecting to EPSG:{output_epsg}...")
        combined_gdf = combined_gdf.to_crs(epsg=output_epsg)

        # Save to output
        os.makedirs(os.path.dirname(output_geojson) or ".", exist_ok=True)
        combined_gdf.to_file(output_geojson, driver="GeoJSON")
        print(f"Combined boundaries saved to: {output_geojson}")
        print(f"  Total features: {len(combined_gdf)}")

    return output_geojson

# =========================================================================== #
# seam_merge — the audited seam-walk blend (replaces the hard cut)
# =========================================================================== #
"""Everything below is the refactor's seam-walk engine, ported from the audited
implementation (audit repo: 01_merge/poc/seamwalk/seamwalk_full.py) that built
the 2024 4-batch visible base map (seam texture-r 0.40 -> 0.62, interiors
byte-identical, blend footprint 0.10%). Since 0.4.0 the seams are measured with
eFOLKI instead of LoFTR (audit 01_merge/report.md section 8.5: on the 3 km seam N
agreement in the band rose from 0.48 to 0.61 against LoFTR's 0.60, in 72 s
against 273 s). The legacy hull/seamline/mosaic functions above are retained;
`find_seamlines`-style ownership is computed internally here on a coarse grid."""

import json as _json
import threading as _threading
import time as _time
from collections import defaultdict as _dd
from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed
from itertools import combinations as _combinations
from pathlib import Path as _Path

from rasterio.vrt import WarpedVRT as _WarpedVRT
from rasterio.warp import transform_bounds as _transform_bounds
from scipy.ndimage import distance_transform_edt as _edt

from . import fields as _F
from . import bands as _B

_GEOM_DS = 8                         # coarse geometry decimation
_HALO = 64                           # block halo so band shifts never sample off-block
_EF_PAD = 512                        # eFOLKI's context around a faultline: twice its reach (efolki.REACH_PX)
_STAT_EVERY = 4                      # seam statistics sample every 4th faultline pixel and band pixel


def _write_json_atomic(path, obj):
    """Durable sentinel/checkpoint write: a SIGKILL can never leave a torn file
    (write to .tmp, then atomic os.replace). Used for the coarse-reuse sentinel
    and the composite render checkpoint."""
    path = _Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_json.dumps(obj))
    os.replace(tmp, path)


def _union_grid(paths, out_crs, res):
    bs = []
    for p in paths:
        with rasterio.open(p) as s:
            bs.append(_transform_bounds(s.crs, out_crs, *s.bounds, densify_pts=21))
    minx = min(b[0] for b in bs); miny = min(b[1] for b in bs)
    maxx = max(b[2] for b in bs); maxy = max(b[3] for b in bs)
    # snap the union origin to the SAME global `res` lattice the registered sources
    # use (registration.py's TR snap), so a lattice-aligned single-owner source reads
    # onto this grid as a pixel-exact CROP (no interior resample), not a sub-pixel
    # average — the last resample the raw->mosaic path was carrying.
    ox = float(np.floor(minx / res) * res)
    oy = float(np.ceil(maxy / res) * res)
    W = int(np.ceil((maxx - ox) / res)); H = int(np.ceil((oy - miny) / res))
    tr = Affine.translation(ox, oy) * Affine.scale(res, -res)
    return tr, W, H, dict(zip(range(len(paths)), bs))


def _recycle_owner(owner_in, names, valid_c, owner_c, out_crs, tr_c, Wc, Hc):
    """The coarse owner from an ownership raster: its `names` tag maps values to
    input names (every name must be an input here). Returns the owner grid with
    the recycled owner wherever it covers the cell AND is valid there, the
    distance-rule owner elsewhere, plus (kept, fell_back) counts."""
    with rasterio.open(owner_in) as s:
        tag_names = [n for n in (s.tags().get("names") or "").split(",") if n]
        if not tag_names:
            raise ValueError(f"seam_merge: {owner_in} has no `names` tag; not an ownership raster")
        unknown = [n for n in tag_names if n not in names]
        if unknown:
            raise ValueError(f"seam_merge: ownership raster names are not inputs here: {unknown}")
        band = 2 if s.count >= 2 else 1           # band 2 = owner everywhere; band 1 has the seam band
        with _WarpedVRT(s, crs=out_crs, transform=tr_c, width=Wc, height=Hc,
                        resampling=Resampling.nearest) as v:
            raw = v.read(band)
    rec = np.full((Hc, Wc), -1, np.int16)
    for k, nm in enumerate(tag_names, 1):
        rec[raw == k] = names.index(nm)
    del raw
    ok = np.zeros((Hc, Wc), bool)
    for i in range(len(names)):
        sel = rec == i
        if sel.any():
            ok[sel] = np.asarray(valid_c[i])[sel]
    covered = owner_c >= 0
    keep = ok & covered
    kept, fell_back = int(keep.sum()), int((covered & ~keep).sum())
    return np.where(keep, rec, owner_c).astype(np.int16), (kept, fell_back)


def seam_merge(inputs, out, band_width_m=1.0, gauge="free", res=None, out_crs=None,
               workers=None, block=2048, ownership_out=None, report_json=None,
               nspec=None, render_chunk=None, band_names=None, owner_in=None,
               geometry_only=False, log=print):
    """Merge N overlapping orthomosaics with the 1 m seam-walk blend.

    inputs   : list of (name, path) pairs — every input is named; names carry
               into the seam table and the stage report.
    out      : output GeoTIFF (tiled ZSTD; run cog.finalize_cog for delivery).
    gauge    : "free" (no input is the reference; corrections split evenly) or
               "anchored" (pin the FIRST input; others move fully — fallback).
    res      : output pixel size; None = median of the inputs' native GSDs.
    nspec    : spectral band count to carry (None = all spectral bands of the
               first input under the band law).
    band_names : the config's spectral band names — the first N non-alpha bands
               of every input are these, in order (bands.band_law). None = every
               non-alpha band. Validity = tagged alpha, else derived from zeros.

    owner_in : an ownership raster written by an earlier seam_merge (its
               `names` tag maps values to input names) whose partition is
               REUSED instead of the distance rule — the visible stitch's
               ownership recycled into the multispectral stitch, so both
               products share one set of seams. Where the recycled owner
               does not cover a cell, or is not valid there, the distance
               rule decides (the validity fallback); the report counts both.
    geometry_only : stop after the coarse pass and the ownership raster —
               the seams from the alphas alone, no matching, no render.
    ownership_out : a 2-band uint8 raster on the coarse grid: band 1 = owner
               index (1..N) with N+1 inside the seam band (what the interiors
               gate reads), band 2 = the owner index everywhere (what
               owner_in reads). 0 = no input covers the cell.

    The seams (eFOLKI, efolki.py): in every render block that has seam band,
    each pair of flights that meet there is measured on the merge grid in the
    box around their faultline, padded by twice eFOLKI's reach (_EF_PAD), where
    both have data. At each band pixel the pair offsets are split over the
    flights covering it (minimum-norm least squares: x_j - x_i = e_ij for every
    measured pair; with gauge="anchored" the first input stays put). Audit
    01_merge/report.md section 8.5 and 02_registration/METHOD.md.

    Geometry only — raw band values are never altered outside the seam band,
    and inside it only cross-faded between the meeting owners. Writes a stage
    report (per seam: its length, the median offset between the two flights on
    the faultline, the metres over 50 cm, the median misfit of the split) for
    the seam tripwire and the quality report.
    """
    t0 = _time.perf_counter()
    names = [n for n, _ in inputs]
    paths = [p for _, p in inputs]
    N = len(paths)
    if out_crs is None:
        with rasterio.open(paths[0]) as s:
            out_crs = str(s.crs)
    if res is None:
        rs = []
        for p in paths:
            with rasterio.open(p) as s:
                rs.append(abs(s.res[0]))
        res = round(float(np.median(rs)), 3)
    laws = [_B.band_law(p, band_names) for p in paths]
    if nspec is None:
        nspec = laws[0].nspec
    short = [n for n, l in zip(names, laws) if l.nspec < nspec]
    if short:
        raise ValueError(f"seam_merge: inputs with fewer than {nspec} spectral bands: {short}")
    aidx = {i: laws[i].alpha for i in range(N)}                    # None = validity from zeros
    spec_idx = {i: [b - 1 for b in laws[i].spectral[:nspec]] for i in range(N)}

    def _valid(i, arr):
        """validity of a windowed read of input i under its band law"""
        return arr[aidx[i] - 1] > 127 if aidx[i] else (arr[spec_idx[i]] != 0).any(0)
    log(f"    band law: {laws[0].describe()}" + (f" (first of {N} inputs)" if N > 1 else ""))
    workers = workers or max(1, (os.cpu_count() or 4) - 1)

    OUT_TR, OUT_W, OUT_H, bounds = _union_grid(paths, out_crs, res)
    log(f"[seam_merge] {N} inputs -> {OUT_W}x{OUT_H} @ {res} m {out_crs} "
        f"(gauge={gauge}, band={band_width_m} m)")

    # the grid manifest — shared by the coarse-reuse sentinel and the walk/render
    # checkpoints, so any plan change invalidates all three together.
    manifest = dict(W=OUT_W, H=OUT_H, res=round(float(res), 6), n=N, names=list(names),
                    owner_in=(str(owner_in) if owner_in else None))
    recycled = None                                   # (kept, fell back) when owner_in

    # ---- coarse geometry: validity, EDT owner, faultlines (STREAMED to disk) --- #
    # ds_c (per-source EDTs, ~57 GB at N=21) and valid_c (~14 GB) are only used
    # later -- ds_c/D_c per-block in the composite, valid_c at the solve -- so
    # they go to disk-backed memmaps and are NEVER held whole in RAM. The EDTs
    # are streamed one at a time with owner/coverage accumulated incrementally,
    # so even the coarse pass peaks at ~one EDT (not the 114 GB np.stack the old
    # path built). owner_c/cov2_c/D_c are byte-identical to the in-RAM version.
    # RESUMABLE: on a clean finish a `_coarse_done.json` sentinel (matching this
    # grid + the exact memmap byte sizes) is written LAST; a restart that finds
    # it reopens the memmaps instead of recomputing ~20 min of EDTs. A crash mid-
    # pass leaves no sentinel, so a partial dir is never trusted -> recomputed.
    Wc, Hc = -(-OUT_W // _GEOM_DS), -(-OUT_H // _GEOM_DS)
    tr_c = Affine.translation(OUT_TR.c, OUT_TR.f) * Affine.scale(res * _GEOM_DS, -res * _GEOM_DS)
    cres = res * _GEOM_DS
    mmdir = _Path(str(out).rsplit(".", 1)[0] + "_coarse")
    mmdir.mkdir(parents=True, exist_ok=True)
    coarse_done = mmdir / "_coarse_done.json"

    def _coarse_reusable():
        if not coarse_done.exists():
            return False
        try:
            if _json.loads(coarse_done.read_text()) != manifest:
                return False
        except Exception:
            return False
        want = {"ds.dat": N * Hc * Wc * 4, "valid.dat": N * Hc * Wc, "D.dat": Hc * Wc * 4}
        for fn, nbytes in want.items():
            p = mmdir / fn
            if not p.exists() or p.stat().st_size != nbytes:
                return False
        return (mmdir / "owner.npy").exists() and (mmdir / "cov2.npy").exists()

    if _coarse_reusable():
        log("    coarse pass: reusing complete geometry from disk "
            "(sentinel matches this grid — skipped the ~20 min EDT recompute)")
        ds_c = np.memmap(mmdir / "ds.dat", dtype=np.float32, mode="r", shape=(N, Hc, Wc))
        valid_c = np.memmap(mmdir / "valid.dat", dtype=np.bool_, mode="r", shape=(N, Hc, Wc))
        D_c = np.memmap(mmdir / "D.dat", dtype=np.float32, mode="r", shape=(Hc, Wc))
        owner_c = np.load(mmdir / "owner.npy")
        cov2_c = np.load(mmdir / "cov2.npy")
    else:
        log(f"    coarse pass: validity masks for {N} inputs @ {cres:.2f} m "
            f"({Wc}x{Hc} px each) ...")

        def read_alpha_c(i):
            with rasterio.open(paths[i]) as s:
                with _WarpedVRT(s, crs=out_crs, transform=tr_c, width=Wc, height=Hc,
                                resampling=Resampling.nearest) as v:
                    m = _B.read_valid(v, laws[i])
            log(f"      {names[i]}: {m.mean() * 100:.1f}% of the union grid")
            return m

        with _TPE(max_workers=min(N, 8)) as ex:
            valid_list = list(ex.map(read_alpha_c, range(N)))
        log(f"    coarse pass: streaming {N} per-source EDTs to disk "
            f"(one at a time; owner/coverage accumulated incrementally) ...")
        ds_c = np.memmap(mmdir / "ds.dat", dtype=np.float32, mode="w+", shape=(N, Hc, Wc))
        valid_c = np.memmap(mmdir / "valid.dat", dtype=np.bool_, mode="w+", shape=(N, Hc, Wc))
        owner_c = np.full((Hc, Wc), -1, np.int16)
        best = np.full((Hc, Wc), -1.0, np.float32)         # running max of ds (argmax)
        ncov = np.zeros((Hc, Wc), np.uint8)                # count of covering sources
        t_edt = _time.perf_counter()
        for i in range(N):
            v = valid_list[i]; valid_list[i] = None
            valid_c[i] = v
            d = _edt(v).astype(np.float32) * cres          # only ~one EDT in RAM at a time
            ds_c[i] = d
            better = d > best                              # strict > => argmax's first-max
            owner_c[better] = i; best[better] = d[better]
            ncov += v
            del v, d, better
            el = _time.perf_counter() - t_edt
            eta = el / (i + 1) * (N - i - 1)
            log(f"      EDT {i + 1}/{N}: {names[i]} streamed  "
                f"({el:.0f}s elapsed, ~{eta:.0f}s left)")
        ds_c.flush(); valid_c.flush()
        del best, valid_list
        cov2_c = ncov >= 2
        owner_c[ncov < 1] = np.int16(-1)                   # -1 where no source covers
        del ncov
        gc.collect()
        if owner_in:
            owner_c, recycled = _recycle_owner(owner_in, names, valid_c, owner_c,
                                               out_crs, tr_c, Wc, Hc)
            log(f"    ownership recycled from {os.path.basename(str(owner_in))}: "
                f"{recycled[0]} cells kept, {recycled[1]} fell back to the distance rule "
                f"(recycled owner absent or not valid there)")
        log("      all EDTs streamed; computing faultlines + coarse distance field "
            "(one final EDT) ...")
        fault_c = np.zeros((Hc, Wc), bool)
        chg = (owner_c[:, :-1] != owner_c[:, 1:]) & cov2_c[:, :-1] & cov2_c[:, 1:]
        fault_c[:, :-1] |= chg; fault_c[:, 1:] |= chg
        chg = (owner_c[:-1, :] != owner_c[1:, :]) & cov2_c[:-1, :] & cov2_c[1:, :]
        fault_c[:-1, :] |= chg; fault_c[1:, :] |= chg
        if not fault_c.any():
            raise RuntimeError("seam_merge: no flight<->flight faultline — nothing to merge")
        D_c = np.memmap(mmdir / "D.dat", dtype=np.float32, mode="w+", shape=(Hc, Wc))
        D_c[:] = _edt(~fault_c).astype(np.float32) * cres
        D_c.flush()
        log(f"    coarse pass done: {int(fault_c.sum())} faultline cells "
            f"(~{int(fault_c.sum()) * cres / 2000:.1f} km of seam; both sides marked); "
            f"ds_c/valid_c/D_c streamed to disk (held RAM-free through the walk)")
        del fault_c
        gc.collect()
        np.save(mmdir / "owner.npy", owner_c)              # persist the two RAM-only derived
        np.save(mmdir / "cov2.npy", cov2_c)                # arrays the walk/solve/composite need
        _write_json_atomic(coarse_done, manifest)          # sentinel LAST -> dir now trustworthy

    if ownership_out:
        own = np.zeros((Hc, Wc), np.uint8)
        for i in range(N):
            own[owner_c == i] = i + 1
        cat = own.copy()
        cat[(D_c <= band_width_m) & cov2_c] = N + 1
        prof = dict(driver="GTiff", height=Hc, width=Wc, count=2, dtype="uint8",
                    crs=out_crs, transform=tr_c, nodata=0, tiled=True,
                    compress="zstd", predictor=2, BIGTIFF="IF_SAFER")
        _Path(ownership_out).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(ownership_out, "w", **prof) as d:
            d.set_band_description(1, "owner index, seam band = names + 1")
            d.set_band_description(2, "owner index everywhere")
            d.update_tags(names=",".join(names), seam_band_value=str(N + 1),
                          band_width_m=str(band_width_m),
                          recycled_from=(os.path.basename(str(owner_in)) if owner_in else ""))
            d.write(cat, 1)
            d.write(own, 2)
        del own, cat
    if geometry_only:
        n_fault = int(((D_c <= 0) & cov2_c).sum())
        ds_c = valid_c = D_c = None
        gc.collect()
        shutil.rmtree(mmdir, ignore_errors=True)
        report = dict(kind="ownership", inputs=names, res_m=res, crs=out_crs,
                      grid=[OUT_W, OUT_H], faultline_cells=n_fault,
                      recycled=recycled, ownership=str(ownership_out) if ownership_out else None,
                      seconds=round(_time.perf_counter() - t0, 1))
        if report_json:
            _Path(report_json).parent.mkdir(parents=True, exist_ok=True)
            _Path(report_json).write_text(_json.dumps(report, indent=2))
        log(f"[seam_merge] ownership only ({report['seconds']:.0f}s) -> {ownership_out}")
        return report

    # ---- the seams: which flights meet (coarse ownership) ------------------- #
    # Measured later, block by block inside the render, where the band is.
    def _adjacency(ia, ib):
        A = owner_c == ia; B = owner_c == ib
        f = np.zeros((Hc, Wc), bool)
        adj = (A[:, :-1] & B[:, 1:]) | (B[:, :-1] & A[:, 1:]); f[:, :-1] |= adj; f[:, 1:] |= adj
        adj = (A[:-1, :] & B[1:, :]) | (B[:-1, :] & A[1:, :]); f[:-1, :] |= adj; f[1:, :] |= adj
        return f & cov2_c

    seam_pairs = [(ia, ib) for ia, ib in _combinations(range(N), 2) if _adjacency(ia, ib).any()]
    if not seam_pairs:
        raise RuntimeError("seam_merge: no flight<->flight faultline — nothing to merge")
    log(f"    {len(seam_pairs)} seams: " + ", ".join(f"{names[i]}-{names[j]}" for i, j in seam_pairs)
        + f"; measured with eFOLKI in the render, gauge {gauge}")

    # the composite needs only ds_c/D_c per block, and the ownership for the
    # faultlines -> free the validity masks and read the ownership memmapped
    valid_c = owner_c = cov2_c = None
    gc.collect()
    owner_mm = np.load(mmdir / "owner.npy", mmap_mode="r")
    cov2_mm = np.load(mmdir / "cov2.npy", mmap_mode="r")
    stats = {pr: dict(off=[], resid=[]) for pr in seam_pairs}
    slock = _threading.Lock()

    # ---- streamed N-way composite ------------------------------------------ #
    prof = dict(driver="GTiff", height=OUT_H, width=OUT_W, count=nspec + 1, dtype="uint8",
                crs=out_crs, transform=OUT_TR, tiled=True, blockxsize=512, blockysize=512,
                compress="zstd", predictor=2, ZSTD_LEVEL=3, BIGTIFF="YES",
                num_threads=str(workers), SPARSE_OK="TRUE")
    # SPARSE_OK: unwritten tiles stay absent instead of being zero-filled on the
    # first close(), so the block-resume below only ever FIRST-writes a tile
    # (never rewrites a compressed one) and uncovered blocks cost no disk.
    _Path(out).parent.mkdir(parents=True, exist_ok=True)
    dst = None                                    # opened in the render section (w fresh / r+ resume)
    wlock = _threading.Lock()
    tls = _threading.local()
    counts = _dd(int)

    def src_vrts():
        if not hasattr(tls, "v"):
            tls.v = {i: _WarpedVRT(rasterio.open(paths[i]), crs=out_crs, transform=OUT_TR,
                                   width=OUT_W, height=OUT_H, resampling=Resampling.average)
                     for i in range(N)}
        return tls.v

    def covers(i, bx0, by0, bx1, by1):
        l, b, r, t = bounds[i]
        return not (bx1 <= l or bx0 >= r or by1 <= b or by0 >= t)

    def _seam_fields(r0, c0, bh, bw, hr0, hc0, hh, hw, iy, ix, present, val, geom):
        """The shift of each present flight on the haloed block. Every pair of present
        flights that share a faultline within _EF_PAD of the block is measured with
        eFOLKI on the merge grid, in the box around that faultline padded by _EF_PAD,
        where both have data (no-data filled from the nearest pixel where both do);
        the pair offsets are then split at each band pixel (minimum-norm least
        squares over the flights covering it). Records the seam statistics."""
        from . import efolki as _E
        pr0, pc0 = max(0, r0 - _EF_PAD), max(0, c0 - _EF_PAD)
        pr1, pc1 = min(OUT_H, r0 + bh + _EF_PAD), min(OUT_W, c0 + bw + _EF_PAD)
        cr0, cc0 = pr0 // _GEOM_DS, pc0 // _GEOM_DS
        own_c = np.asarray(owner_mm[cr0:-(-pr1 // _GEOM_DS), cc0:-(-pc1 // _GEOM_DS)])
        c2_c = np.asarray(cov2_mm[cr0:-(-pr1 // _GEOM_DS), cc0:-(-pc1 // _GEOM_DS)])
        V = src_vrts()
        pos = {i: k for k, i in enumerate(present)}
        efield = {}
        for ia, ib in _combinations(present, 2):
            if (ia, ib) not in stats:
                continue                                  # these two never share a faultline
            A = own_c == ia; B = own_c == ib
            f = np.zeros(own_c.shape, bool)
            adj = (A[:, :-1] & B[:, 1:]) | (B[:, :-1] & A[:, 1:]); f[:, :-1] |= adj; f[:, 1:] |= adj
            adj = (A[:-1, :] & B[1:, :]) | (B[:-1, :] & A[1:, :]); f[:-1, :] |= adj; f[1:, :] |= adj
            f &= c2_c
            if not f.any():
                continue
            ys, xs = np.where(f)
            br0 = max(pr0, (cr0 + int(ys.min())) * _GEOM_DS - _EF_PAD); br1 = min(pr1, (cr0 + int(ys.max()) + 1) * _GEOM_DS + _EF_PAD)
            bc0 = max(pc0, (cc0 + int(xs.min())) * _GEOM_DS - _EF_PAD); bc1 = min(pc1, (cc0 + int(xs.max()) + 1) * _GEOM_DS + _EF_PAD)
            ir0, ir1 = max(br0, hr0), min(br1, hr0 + hh); ic0, ic1 = max(bc0, hc0), min(bc1, hc0 + hw)
            if ir1 <= ir0 or ic1 <= ic0:
                continue
            win = Window(bc0, br0, bc1 - bc0, br1 - br0)
            a = V[ia].read(window=win); b = V[ib].read(window=win)
            both = _valid(ia, a) & _valid(ib, b)
            if not both.any():
                continue
            ga = _E.fill_nearest(np.where(both, _F.gray_stretch(a[spec_idx[ia][:3]].astype(np.float32)), 0), both)
            gb = _E.fill_nearest(np.where(both, _F.gray_stretch(b[spec_idx[ib][:3]].astype(np.float32)), 0), both)
            del a, b
            u, v = _E.flow(ga, gb)                        # b(x + e) lines up with a(x)
            eu = np.full((hh, hw), np.nan, np.float32); ev = eu.copy()
            sb = np.s_[ir0 - br0:ir1 - br0, ic0 - bc0:ic1 - bc0]
            sh = np.s_[ir0 - hr0:ir1 - hr0, ic0 - hc0:ic1 - hc0]
            eu[sh] = np.where(both[sb], u[sb], np.nan); ev[sh] = np.where(both[sb], v[sb], np.nan)
            efield[(ia, ib)] = (eu, ev)
            del u, v, ga, gb, both
        n = len(present)
        fx = [np.zeros((hh, hw), np.float32) for _ in range(n)]
        fy = [np.zeros((hh, hw), np.float32) for _ in range(n)]
        pairs = list(efield)
        band = geom["in_band"]
        if pairs:
            rr, cc = np.where(band)
            have = np.stack([val[i][rr, cc] for i in present], 1)
            meas = np.stack([np.isfinite(efield[pr][0][rr, cc]) & have[:, pos[pr[0]]] & have[:, pos[pr[1]]]
                             for pr in pairs], 1)
            groups, inv = np.unique(np.c_[have, meas].astype(np.uint8), axis=0, return_inverse=True)
            inv = inv.ravel()
            pinned = pos.get(0) if gauge == "anchored" else None
            for g, key in enumerate(groups):
                rows = [pairs[q] for q in range(len(pairs)) if key[n + q]]
                if not rows:
                    continue
                sel = np.where(inv == g)[0]
                A = np.zeros((len(rows), n))
                for t, (i, j) in enumerate(rows):
                    A[t, pos[j]], A[t, pos[i]] = 1.0, -1.0
                if pinned is not None:
                    A[:, pinned] = 0.0                    # the first input stays put
                Ap = np.linalg.pinv(A)
                for comp, out_ in ((0, fx), (1, fy)):
                    x = Ap @ np.stack([efield[pr][comp][rr[sel], cc[sel]] for pr in rows])
                    for k in range(n):
                        out_[k][rr[sel], cc[sel]] = x[k]
        # seam statistics on the block's core: the offset on the faultline, the split's misfit in the band
        own = geom["owner"]
        core = np.zeros((hh, hw), bool); core[iy:iy + bh, ix:ix + bw] = True
        for (ia, ib) in [pr for pr in _combinations(present, 2) if pr in stats]:
            pa, pb = pos[ia], pos[ib]
            A = own == pa; B = own == pb
            line = np.zeros((hh, hw), bool)
            line[:, :-1] |= A[:, :-1] & B[:, 1:]; line[:, 1:] |= A[:, 1:] & B[:, :-1]
            line[:-1, :] |= A[:-1, :] & B[1:, :]; line[1:, :] |= A[1:, :] & B[:-1, :]
            line &= band & core
            n_line = int(line.sum())
            if not n_line:
                continue
            off = resid = np.zeros(0, np.float32)
            if (ia, ib) in efield:
                eu, ev = efield[(ia, ib)]
                off = np.hypot(eu[line], ev[line])
                off = off[np.isfinite(off)][::_STAT_EVERY] * res * 100
                m = band & core & np.isfinite(eu) & val[ia] & val[ib]
                resid = np.hypot(fx[pb][m] - fx[pa][m] - eu[m], fy[pb][m] - fy[pa][m] - ev[m])[::_STAT_EVERY] * res * 100
            with slock:
                st = stats[(ia, ib)]
                st["px"] = st.get("px", 0) + n_line
                st["off"].append(off.astype(np.float32)); st["resid"].append(resid.astype(np.float32))
        return list(zip(fx, fy))

    def process_block(r0, c0):
        bh = min(block, OUT_H - r0); bw = min(block, OUT_W - c0)
        hr0, hc0 = max(0, r0 - _HALO), max(0, c0 - _HALO)
        hr1, hc1 = min(OUT_H, r0 + bh + _HALO), min(OUT_W, c0 + bw + _HALO)
        hh, hw = hr1 - hr0, hc1 - hc0
        iy, ix = r0 - hr0, c0 - hc0
        bx0, by0 = OUT_TR * (hc0, hr1); bx1, by1 = OUT_TR * (hc1, hr0)
        present = [i for i in range(N) if covers(i, bx0, by0, bx1, by1)]
        if not present:
            return "skip"
        V = src_vrts()
        rg = {i: V[i].read(window=Window(hc0, hr0, hw, hh)) for i in present}
        val = {i: _valid(i, rg[i]) for i in present}
        present = [i for i in present if val[i].any()]
        if not present:
            return "skip"
        if len(present) == 1:
            i = present[0]
            outh = np.concatenate([rg[i][spec_idx[i]],
                                   np.where(val[i], 255, 0)[None].astype(np.uint8)], 0)
            kind = "copy"
        else:
            valids = [val[i] for i in present]
            specs = [rg[i][spec_idx[i]].astype(np.float32) for i in present]
            geom = _seam_geom_block(hr0, hc0, hh, hw, present, valids, ds_c, D_c,
                                    res, band_width_m, owner_c=owner_mm if owner_in else None)
            if geom["in_band"][iy:iy + bh, ix:ix + bw].any():
                flds = _seam_fields(r0, c0, bh, bw, hr0, hc0, hh, hw, iy, ix, present, val, geom)
                sf, al, _ = _F.seamline_composite(specs, valids, res, band_width_m,
                                                  fields=flds, geom=geom)
                kind = "seam"
            else:
                sf, al, _ = _F.seamline_composite(specs, valids, res, band_width_m,
                                                  fields=None, geom=geom)
                kind = "own"
            # rint, not truncate: on the band fringe the blend is the owner value
            # ± sub-LSB float dust; truncation would turn that into a 1-DN error
            outh = np.concatenate([np.clip(np.rint(sf), 0, 255).astype(np.uint8),
                                   al[None]], 0)
        with wlock:
            dst.write(outh[:, iy:iy + bh, ix:ix + bw], window=Window(c0, r0, bw, bh))
        return kind

    blocks = [(r0, c0) for r0 in range(0, OUT_H, block) for c0 in range(0, OUT_W, block)]
    # ---- render checkpoint: block-resume so a crash mid-composite never -------
    # restarts the ~183k x 237k render from block 0. Completed block indices are
    # recorded only AFTER a close() flush boundary, so a resume trusts only what
    # GDAL durably wrote; each block writes a disjoint, tile-aligned window
    # (block % blocksize == 0) and SPARSE_OK keeps the rest absent, so a resumed
    # write is always a FIRST-write -- no compressed tile is ever rewritten.
    rckpt = _Path(str(out).rsplit(".", 1)[0] + "_render_ckpt.json")
    rmanifest = dict(manifest, block=int(block), nspec=int(nspec))
    done = set()
    if os.path.exists(out) and rckpt.exists():
        try:
            rc = _json.loads(rckpt.read_text())
            if rc.get("manifest") == rmanifest:
                done = set(rc.get("done", []))
        except Exception:
            done = set()
    if done:
        dst = rasterio.open(out, "r+")
        log(f"    composite: resuming — {len(done)}/{len(blocks)} blocks already on "
            f"disk (crash-safe render checkpoint)")
    else:
        dst = rasterio.open(out, "w", **prof)
        _B.tag_output(dst, (list(band_names) if band_names else list(laws[0].names))[:nspec])
    todo = [bi for bi in range(len(blocks)) if bi not in done]
    log(f"    composite: {len(blocks)} blocks of {block}x{block} px, {workers} workers, "
        f"{len(todo)} to render (copy = one owner, byte-identical; own/seam = overlap)")
    chunk_n = render_chunk or max(64, min(512, len(blocks) // 40 or 64))   # flush/checkpoint granularity
    n_done = base = len(done)
    t_comp = _time.perf_counter()
    t_last = t_comp
    with _TPE(max_workers=workers) as ex:
        for ci in range(0, len(todo), chunk_n):
            chunk = todo[ci:ci + chunk_n]
            futs = {ex.submit(process_block, *blocks[bi]): bi for bi in chunk}
            for fut in _as_completed(futs):
                counts[fut.result()] += 1; n_done += 1
                now = _time.perf_counter()
                if n_done == len(blocks) or n_done % 100 == 0 or now - t_last >= 120:
                    t_last = now
                    gb = os.path.getsize(out) / 1e9 if os.path.exists(out) else 0.0
                    rate = (n_done - base) / max(now - t_comp, 1e-9)
                    eta_m = (len(blocks) - n_done) / rate / 60 if rate > 0 else 0.0
                    log(f"    [{n_done}/{len(blocks)}] {100 * n_done / len(blocks):4.1f}%  "
                        f"copy={counts['copy']} own={counts['own']} seam={counts['seam']} "
                        f"skip={counts['skip']}  {gb:.1f} GB on disk  ETA ~{eta_m:.0f} min")
            dst.close()                            # flush this chunk durably before marking done
            done.update(chunk)
            _write_json_atomic(rckpt, dict(manifest=rmanifest, done=sorted(done)))
            if len(done) < len(blocks):
                dst = rasterio.open(out, "r+")     # reopen; update mode won't zero-fill the sparse rest
    if dst is not None and not dst.closed:
        dst.close()   # vmain was closed after the walk; the composite uses tls VRTs
    ds_c = D_c = owner_mm = cov2_mm = None        # release the coarse memmaps
    gc.collect()
    shutil.rmtree(mmdir, ignore_errors=True)      # ~74 GB of coarse scratch, done
    rckpt.unlink(missing_ok=True)                 # render finished -> drop the checkpoint

    seams = []
    for (i, j) in seam_pairs:                     # per seam: length, offset on the faultline, misfit of the split
        st = stats[(i, j)]
        off = np.concatenate(st["off"]) if st["off"] else np.zeros(0)
        rs = np.concatenate(st["resid"]) if st["resid"] else np.zeros(0)
        seam_m = round(st.get("px", 0) * res, 1)
        rec = dict(pair=f"{names[i]}-{names[j]}", ij=[i, j], seam_m=seam_m,
                   med_offset_cm=round(float(np.median(off)), 1) if len(off) else None,
                   over50_m=round(float((off > 50).mean()) * seam_m, 1) if len(off) else None,
                   residual_cm=round(float(np.median(rs)), 1) if len(rs) else None)
        seams.append(rec)
        log(f"    seam {rec['pair']}: {seam_m:.0f} m, offset median {rec['med_offset_cm']} cm, "
            f"{rec['over50_m']} m over 50 cm, split misfit {rec['residual_cm']} cm")

    report = dict(kind="stitch", method="eFOLKI", inputs=names, n_seams=len(seams), seams=seams,
                  rendered_this_run=len(todo), resumed_blocks=base,
                  gauge=gauge, band_width_m=band_width_m, res_m=res, crs=out_crs,
                  grid=[OUT_W, OUT_H], blocks=dict(counts),
                  seconds=round(_time.perf_counter() - t0, 1), out=str(out),
                  ownership=str(ownership_out) if ownership_out else None,
                  recycled=recycled, owner_in=(str(owner_in) if owner_in else None))
    if report_json:
        _Path(report_json).parent.mkdir(parents=True, exist_ok=True)
        _Path(report_json).write_text(_json.dumps(report, indent=2))
    log(f"[seam_merge] done ({report['seconds']:.0f}s) -> {out}")
    return report


def _seam_geom_block(r0, c0, bh, bw, present, valids, ds_c, D_c, res, band_m,
                     owner_c=None):
    """Native-res seam geometry for one block, upsampled from the GLOBAL coarse
    EDTs (so ownership is globally correct even at block edges). With a coarse
    `owner_c` (a recycled ownership) the owner is read from it — nearest, then
    the distance rule only where the given owner is absent or not valid."""
    ds = np.stack([_F.upsample_block(ds_c[i], _GEOM_DS, r0, c0, bh, bw) / res
                   for i in present])
    D = _F.upsample_block(D_c, _GEOM_DS, r0, c0, bh, bw) / res
    vstack = np.stack(valids)
    cov2 = vstack.sum(0) >= 2
    any_valid = vstack.any(0)
    B = max(band_m / res, 1.0)
    owner = np.where(cov2, np.argmax(ds, 0).astype(np.int16), np.int16(-1))
    if owner_c is not None:
        og = _F.upsample_block(owner_c, _GEOM_DS, r0, c0, bh, bw,
                               interp=cv2.INTER_NEAREST).astype(np.int16)
        given = np.full(owner.shape, -1, np.int16)
        for p, i in enumerate(present):
            given[(og == i) & valids[p]] = p
        owner = np.where(cov2 & (given >= 0), given, owner)
    single = any_valid & ~cov2
    for p in range(len(present)):
        owner = np.where(single & valids[p], np.int16(p), owner)
    in_band = (D <= B) & cov2
    taper = (np.clip(1.0 - D / B, 0.0, 1.0) * in_band).astype(np.float32)
    return dict(B=B, ds=ds, any_valid=any_valid, owner=owner, D=D,
                in_band=in_band, taper=taper, edt_max=ds.max(0))

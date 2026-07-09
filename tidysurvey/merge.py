import os
import tempfile

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
byte-identical, blend footprint 0.10%). The legacy hull/seamline/mosaic
functions above are retained; `find_seamlines`-style ownership is computed
internally here on a coarse grid."""

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

_TILE_H, _TILE_W = 480, 640          # LoFTR tile
_FS = 120                            # seam pooling cell (px)
_SOLVE_S = 120                       # solve node spacing (px)
_W_GAUGE = 0.05                      # gauge ridge (free translation pin)
_REJECT_PX, _MIN_TILE, _MIN_CELL = 4.0, 8, 2
_SOLVE_CORR_M = 4.0                  # solve corridor half-width (m)
_GEOM_DS = 8                         # coarse geometry decimation
_HALO = 64                           # block halo so band shifts never sample off-block


def _alpha_index(path):
    with rasterio.open(path) as s:
        if s.colorinterp:
            for i, ci in enumerate(s.colorinterp):
                if ci == ColorInterp.alpha:
                    return i + 1
        return s.count      # convention: last band is validity when no alpha tag


def _union_grid(paths, out_crs, res):
    bs = []
    for p in paths:
        with rasterio.open(p) as s:
            bs.append(_transform_bounds(s.crs, out_crs, *s.bounds, densify_pts=21))
    minx = min(b[0] for b in bs); miny = min(b[1] for b in bs)
    maxx = max(b[2] for b in bs); maxy = max(b[3] for b in bs)
    W = int(np.ceil((maxx - minx) / res)); H = int(np.ceil((maxy - miny) / res))
    tr = Affine.translation(minx, maxy) * Affine.scale(res, -res)
    return tr, W, H, dict(zip(range(len(paths)), bs))


def seam_merge(inputs, out, band_width_m=1.0, gauge="free", res=None, out_crs=None,
               workers=None, block=2048, ownership_out=None, report_json=None,
               nspec=None, log=print):
    """Merge N overlapping orthomosaics with the 1 m seam-walk blend.

    inputs   : list of (name, path) pairs — every input is named; names carry
               into the seam table and the stage report.
    out      : output GeoTIFF (tiled ZSTD; run cog.finalize_cog for delivery).
    gauge    : "free" (no input is the reference; corrections split evenly) or
               "anchored" (pin the FIRST input; others move fully — fallback).
    res      : output pixel size; None = median of the inputs' native GSDs.
    nspec    : spectral band count to carry (None = all non-alpha bands of the
               first input). Alpha/validity = tagged alpha band, else the last.

    Geometry only — raw band values are never altered outside the seam band,
    and inside it only cross-faded between the meeting owners. Writes a stage
    report (per-seam matches, median shift, post-solve residual) for the seam
    tripwire and the quality report.
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
    aidx = {i: _alpha_index(p) for i, p in enumerate(paths)}
    if nspec is None:
        with rasterio.open(paths[0]) as s:
            nspec = min(aidx[0] - 1, s.count - 1) if s.count > 1 else 1
    workers = workers or max(1, (os.cpu_count() or 4) - 1)

    OUT_TR, OUT_W, OUT_H, bounds = _union_grid(paths, out_crs, res)
    log(f"[seam_merge] {N} inputs -> {OUT_W}x{OUT_H} @ {res} m {out_crs} "
        f"(gauge={gauge}, band={band_width_m} m)")

    # ---- coarse geometry: validity, EDT owner, faultlines ------------------ #
    Wc, Hc = -(-OUT_W // _GEOM_DS), -(-OUT_H // _GEOM_DS)
    tr_c = Affine.translation(OUT_TR.c, OUT_TR.f) * Affine.scale(res * _GEOM_DS, -res * _GEOM_DS)
    cres = res * _GEOM_DS

    log(f"    coarse pass: validity masks for {N} inputs @ {cres:.2f} m "
        f"({Wc}x{Hc} px each) ...")

    def read_alpha_c(i):
        with rasterio.open(paths[i]) as s:
            with _WarpedVRT(s, crs=out_crs, transform=tr_c, width=Wc, height=Hc,
                            resampling=Resampling.nearest) as v:
                m = v.read(aidx[i]) > 127
        log(f"      {names[i]}: {m.mean() * 100:.1f}% of the union grid")
        return m

    with _TPE(max_workers=min(N, 8)) as ex:
        valid_c = list(ex.map(read_alpha_c, range(N)))
    log("    coarse pass: EDT ownership + faultlines ...")
    ds_c = [_edt(v).astype(np.float32) * cres for v in valid_c]
    vstack = np.stack(valid_c)
    cov2_c = vstack.sum(0) >= 2
    owner_c = np.where(vstack.any(0), np.argmax(np.stack(ds_c), 0).astype(np.int16),
                       np.int16(-1))
    fault_c = np.zeros((Hc, Wc), bool)
    chg = (owner_c[:, :-1] != owner_c[:, 1:]) & cov2_c[:, :-1] & cov2_c[:, 1:]
    fault_c[:, :-1] |= chg; fault_c[:, 1:] |= chg
    chg = (owner_c[:-1, :] != owner_c[1:, :]) & cov2_c[:-1, :] & cov2_c[1:, :]
    fault_c[:-1, :] |= chg; fault_c[1:, :] |= chg
    D_c = (_edt(~fault_c).astype(np.float32) * cres if fault_c.any()
           else np.full((Hc, Wc), 1e9, np.float32))
    if not fault_c.any():
        raise RuntimeError("seam_merge: no flight<->flight faultline — nothing to merge")
    log(f"    coarse pass done: {int(fault_c.sum())} faultline cells "
        f"(~{int(fault_c.sum()) * cres / 2000:.1f} km of seam; both sides marked)")

    if ownership_out:
        cat = np.zeros((Hc, Wc), np.uint8)
        for i in range(N):
            cat[owner_c == i] = i + 1
        cat[(D_c <= band_width_m) & cov2_c] = N + 1
        prof = dict(driver="GTiff", height=Hc, width=Wc, count=1, dtype="uint8",
                    crs=out_crs, transform=tr_c, nodata=0, tiled=True,
                    compress="zstd", predictor=2, BIGTIFF="IF_SAFER")
        _Path(ownership_out).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(ownership_out, "w", **prof) as d:
            d.write(cat, 1)
            d.update_tags(names=",".join(names), seam_band_value=str(N + 1))

    # ---- per-pair seam walk (LoFTR), CHECKPOINTED -------------------------- #
    # each pair's sparse result is persisted the moment it finishes, so a crash
    # (or a jetsam kill at the solve) never re-walks a completed seam. efield is
    # kept SPARSE in RAM (a few MB) -- never densified -- for the low-memory solve.
    ckpt = _Path(str(out).rsplit(".", 1)[0] + "_seam_ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    manifest = dict(W=OUT_W, H=OUT_H, res=round(float(res), 6), n=N, names=list(names))
    mpath = ckpt / "_grid.json"
    if mpath.exists() and _json.loads(mpath.read_text()) != manifest:
        for f in ckpt.glob("*.npz"):
            f.unlink()
        log("    seam checkpoint: plan changed -> stale cache cleared")
    mpath.write_text(_json.dumps(manifest))

    def _adjacency(ia, ib):
        A = owner_c == ia; B = owner_c == ib
        f = np.zeros((Hc, Wc), bool)
        adj = (A[:, :-1] & B[:, 1:]) | (B[:, :-1] & A[:, 1:]); f[:, :-1] |= adj; f[:, 1:] |= adj
        adj = (A[:-1, :] & B[1:, :]) | (B[:-1, :] & A[1:, :]); f[:-1, :] |= adj; f[1:, :] |= adj
        return f

    matcher = dev = vmain = None
    efield_sp, seams = {}, []
    all_pairs = list(_combinations(range(N), 2))
    n_cached = sum((ckpt / f"seam_{ia}_{ib}.npz").exists() for ia, ib in all_pairs)
    if n_cached:
        log(f"    seam checkpoint: resuming — {n_cached}/{len(all_pairs)} pairs already on disk")
    for ia, ib in all_pairs:
        cf = ckpt / f"seam_{ia}_{ib}.npz"
        if cf.exists():
            z = np.load(cf, allow_pickle=True)
            if bool(z["ok"]):
                efield_sp[(ia, ib)] = (z["pts"], z["vx"], z["vy"])
                seams.append(dict(z["rec"].item()))
            continue
        f_ij = _adjacency(ia, ib)
        if not f_ij.any():
            np.savez(cf, ok=False)
            continue
        ys, xs = np.where(f_ij)
        seen, seeds = set(), []
        for y, x in zip(ys * _GEOM_DS + _GEOM_DS // 2, xs * _GEOM_DS + _GEOM_DS // 2):
            key = (y // _TILE_H, x // _TILE_W)
            if key not in seen:
                seen.add(key); seeds.append((int(y), int(x)))
        if matcher is None:                      # build the GPU matcher + read VRTs lazily
            matcher, dev = _F.build_matcher()
            vmain = {i: _WarpedVRT(rasterio.open(paths[i]), crs=out_crs, transform=OUT_TR,
                                   width=OUT_W, height=OUT_H, resampling=Resampling.bilinear)
                     for i in range(N)}
        log(f"    seam {names[ia]}-{names[ib]}: walking {len(seeds)} corridor tiles "
            f"(LoFTR on {dev.type}) ...")
        ans, dss, n_tiles = [], [], 0
        for n_seed, (y, x) in enumerate(seeds, 1):
            if n_seed % 100 == 0:
                _F.release_matcher_cache(dev)   # cap MPS allocator growth
            if n_seed % 25 == 0:
                log(f"      [{n_seed}/{len(seeds)}] tiles read+matched, "
                    f"{n_tiles} usable")
            r0 = min(max(y - _TILE_H // 2, 0), OUT_H - _TILE_H)
            c0 = min(max(x - _TILE_W // 2, 0), OUT_W - _TILE_W)
            a = vmain[ia].read(window=Window(c0, r0, _TILE_W, _TILE_H))
            b = vmain[ib].read(window=Window(c0, r0, _TILE_W, _TILE_H))
            va, vb = a[aidx[ia] - 1] > 127, b[aidx[ib] - 1] > 127
            if va.mean() < 0.4 or vb.mean() < 0.4:
                continue
            m = _F.match_tile(matcher, dev,
                              _F.gray_stretch(a[:min(3, nspec)].astype(np.float32)),
                              _F.gray_stretch(b[:min(3, nspec)].astype(np.float32)),
                              _REJECT_PX, _MIN_TILE)
            if m is None:
                continue
            an, d = m
            ans.append(an + [c0, r0]); dss.append(d); n_tiles += 1
        if not ans:
            np.savez(cf, ok=False)
            log(f"    seam {names[ia]}-{names[ib]}: no usable matches (skipped)")
            continue
        an = np.concatenate(ans); d = np.concatenate(dss)
        pts, vx, vy = _F.pool_to_cells(an, d, _FS, _MIN_CELL)
        if len(pts) == 0:
            np.savez(cf, ok=False)
            continue
        pts_c = np.stack([pts[:, 0] / _GEOM_DS, pts[:, 1] / _GEOM_DS], 1)   # coarse px
        vx = np.asarray(vx); vy = np.asarray(vy)
        med_cm = float(np.median(np.hypot(vx, vy)) * res * 100)
        rec = dict(pair=f"{names[ia]}-{names[ib]}", ij=[ia, ib], tiles=n_tiles,
                   matches=int(len(an)), nodes=int(len(pts)), med_shift_cm=round(med_cm, 1))
        np.savez(cf, ok=True, pts=pts_c, vx=vx, vy=vy, rec=rec)   # <- crash-safe checkpoint
        efield_sp[(ia, ib)] = (pts_c, vx, vy)
        seams.append(rec)
        log(f"    seam {names[ia]}-{names[ib]}: {n_tiles} tiles -> {len(an)} matches "
            f"-> {len(pts)} nodes (|e|={med_cm:.1f} cm)")
    if matcher is not None:
        del matcher
        _F.release_matcher_cache(dev)   # GPU work is done; composite runs for hours
    if vmain is not None:
        for v in vmain.values():
            v.close()
    if not efield_sp:
        raise RuntimeError("seam_merge: no seams matched")

    # ---- joint solve (SPARSE in, LATTICE out; memory-frugal) --------------- #
    corridor_c = (D_c <= _SOLVE_CORR_M) & cov2_c
    anchored = 0 if gauge == "anchored" else None
    lat_c, sol_c, resid, n_nodes, n_var = _F.solve_free_gauge(
        valid_c, efield_sp, corridor_c, max(1, round(_SOLVE_S / _GEOM_DS)),
        N, _W_GAUGE, anchored_index=anchored)
    log(f"    solve: {n_nodes} corridor nodes, {n_var} unknowns "
        f"({'anchored to ' + names[0] if anchored is not None else 'free gauge'})")
    for s in seams:                              # post-solve residual (px -> cm)
        rp = resid.get(tuple(s["ij"]))
        s["residual_cm"] = round(rp * res * 100, 1) if rp is not None else None

    # the composite needs only ds_c/D_c/lat_c per block -> free the ~16 GB of
    # validity masks + ownership before the (never-before-reached) long render
    valid_c = owner_c = cov2_c = None

    # ---- streamed N-way composite ------------------------------------------ #
    prof = dict(driver="GTiff", height=OUT_H, width=OUT_W, count=nspec + 1, dtype="uint8",
                crs=out_crs, transform=OUT_TR, tiled=True, blockxsize=512, blockysize=512,
                compress="zstd", predictor=2, ZSTD_LEVEL=3, BIGTIFF="YES",
                num_threads=str(workers))
    _Path(out).parent.mkdir(parents=True, exist_ok=True)
    dst = rasterio.open(out, "w", **prof)
    if nspec == 3:
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue,
                           ColorInterp.alpha]
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

    _FIELD_UP = _GEOM_DS * 8                      # solved lattice (8 coarse-px) -> native px
    def _lat_block(lat_i, r0, c0, bh, bw):
        """A block of a source's solved field, upsampled from its lattice (or
        zero where the source had no solved nodes). Bit-identical to upsampling
        the full coarse field the old path built (validated)."""
        if lat_i is None:
            return (np.zeros((bh, bw), np.float32), np.zeros((bh, bw), np.float32))
        return (_F.upsample_block(lat_i[0], _FIELD_UP, r0, c0, bh, bw),
                _F.upsample_block(lat_i[1], _FIELD_UP, r0, c0, bh, bw))

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
        val = {i: rg[i][aidx[i] - 1] > 127 for i in present}
        present = [i for i in present if val[i].any()]
        if not present:
            return "skip"
        if len(present) == 1:
            i = present[0]
            outh = np.concatenate([rg[i][:nspec],
                                   np.where(val[i], 255, 0)[None].astype(np.uint8)], 0)
            kind = "copy"
        else:
            valids = [val[i] for i in present]
            specs = [rg[i][:nspec].astype(np.float32) for i in present]
            geom = _seam_geom_block(hr0, hc0, hh, hw, present, valids, ds_c, D_c,
                                    res, band_width_m)
            if geom["in_band"][iy:iy + bh, ix:ix + bw].any():
                flds = [_lat_block(lat_c[i], hr0, hc0, hh, hw) for i in present]
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
    log(f"    composite: {len(blocks)} blocks of {block}x{block} px, {workers} workers "
        f"(copy = one owner, byte-identical; own/seam = overlap blocks)")
    t_comp = _time.perf_counter()
    t_last = t_comp
    with _TPE(max_workers=workers) as ex:
        futs = {ex.submit(process_block, r0, c0): (r0, c0) for r0, c0 in blocks}
        n_done = 0
        for fut in _as_completed(futs):
            counts[fut.result()] += 1; n_done += 1
            now = _time.perf_counter()
            if n_done == len(blocks) or n_done % 100 == 0 or now - t_last >= 120:
                t_last = now
                gb = os.path.getsize(out) / 1e9 if os.path.exists(out) else 0.0
                rate = n_done / max(now - t_comp, 1e-9)
                eta_m = (len(blocks) - n_done) / rate / 60
                log(f"    [{n_done}/{len(blocks)}] {100 * n_done / len(blocks):4.1f}%  "
                    f"copy={counts['copy']} own={counts['own']} seam={counts['seam']} "
                    f"skip={counts['skip']}  {gb:.1f} GB on disk  ETA ~{eta_m:.0f} min")
    dst.close()   # vmain was closed after the walk; the composite uses tls VRTs

    report = dict(kind="stitch", inputs=names, n_seams=len(seams), seams=seams,
                  gauge=gauge, band_width_m=band_width_m, res_m=res, crs=out_crs,
                  grid=[OUT_W, OUT_H], blocks=dict(counts),
                  seconds=round(_time.perf_counter() - t0, 1), out=str(out),
                  ownership=str(ownership_out) if ownership_out else None)
    if report_json:
        _Path(report_json).parent.mkdir(parents=True, exist_ok=True)
        _Path(report_json).write_text(_json.dumps(report, indent=2))
    log(f"[seam_merge] done ({report['seconds']:.0f}s) -> {out}")
    return report


def _seam_geom_block(r0, c0, bh, bw, present, valids, ds_c, D_c, res, band_m):
    """Native-res seam geometry for one block, upsampled from the GLOBAL coarse
    EDTs (so ownership is globally correct even at block edges)."""
    ds = np.stack([_F.upsample_block(ds_c[i], _GEOM_DS, r0, c0, bh, bw) / res
                   for i in present])
    D = _F.upsample_block(D_c, _GEOM_DS, r0, c0, bh, bw) / res
    vstack = np.stack(valids)
    cov2 = vstack.sum(0) >= 2
    any_valid = vstack.any(0)
    B = max(band_m / res, 1.0)
    owner = np.where(cov2, np.argmax(ds, 0).astype(np.int16), np.int16(-1))
    single = any_valid & ~cov2
    for p in range(len(present)):
        owner = np.where(single & valids[p], np.int16(p), owner)
    in_band = (D <= B) & cov2
    taper = (np.clip(1.0 - D / B, 0.0, 1.0) * in_band).astype(np.float32)
    return dict(B=B, ds=ds, any_valid=any_valid, owner=owner, D=D,
                in_band=in_band, taper=taper, edt_max=ds.max(0))

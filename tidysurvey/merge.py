import os
import warnings
import tempfile
import tempfile

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy import ndimage

import rasterio
from rasterio.enums import Resampling, ColorInterp
from rasterio import features as rio_features

from osgeo import gdal

import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union

from affine import Affine
import joblib
import shutil

# Define a progress callback function
def my_progress_callback(complete, message, user_data):
    """
    Prints the progress of the GDAL operation.
    
    Args:
        complete (float): Progress percentage (0.0 to 1.0).
        message (str): Message from GDAL.
        user_data: Custom data passed to the callback.
    """
    # The message can be empty, so handle that for cleaner output
    if message:
        print(f"gdal_translate Progress: {complete*100:.2f}% - {message}")
    else:
        print(f"gdal_translate Progress: {complete*100:.2f}%")
    return 1 # Must return 1 to continue processing

def define_hull(
    geotiff_path_in: str,
    hull_geojson_path: str,
    tol: float = 1.0
):
    if tol <= 0: raise ValueError("tol must be positive.")
    cutline_simplify_tolerance = tol

    print(f"Starting hull definition for: {geotiff_path_in}")
    print(f"  Output Hull GeoJSON: {hull_geojson_path}")
    print(f"  Target tolerance for hull: {tol}m")

    source_alpha_band_index_1based = None
    with rasterio.open(geotiff_path_in) as pre_src:
        print("Reading source GeoTIFF profile for initial dimensions, CRS, and alpha band detection...")
        profile_orig = pre_src.profile.copy()
        original_width, original_height = pre_src.width, pre_src.height
        original_transform, source_crs = pre_src.transform, pre_src.crs
        src_bounds = pre_src.bounds

        if pre_src.colorinterp:
            for i, interp in enumerate(pre_src.colorinterp):
                if interp == ColorInterp.alpha:
                    source_alpha_band_index_1based = i + 1
                    print(f"Source GeoTIFF has an alpha band (index {source_alpha_band_index_1based}). This band will be used for hull generation.")
                    break

        if source_crs and source_crs.is_geographic:
            warnings.warn("Geographic CRS: 'tol' in degrees. Projected CRS recommended.")

    print("Determining mask generation grid dimensions...")
    mask_gen_width = max(1, min(original_width, round(abs(src_bounds.right - src_bounds.left) / tol)))
    mask_gen_height = max(1, min(original_height, round(abs(src_bounds.top - src_bounds.bottom) / tol)))
    print(f"Calculated mask generation grid: {mask_gen_width}x{mask_gen_height}")

    temp_gdal_file_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
            temp_gdal_file_path = tmpfile.name
        print(f"Creating temporary local GeoTIFF for mask grid: {temp_gdal_file_path}")

        gdal_input_path = geotiff_path_in
        
        translate_options_dict = {
            "format": "GTiff",
            "width": int(mask_gen_width),
            "height": int(mask_gen_height),
            "callback": my_progress_callback,
            "callback_data": None # Optional: pass custom data to the callback
        }

        if source_alpha_band_index_1based is not None:
            translate_options_dict["bandList"] = [int(source_alpha_band_index_1based)]
            print(f"gdal.Translate will use source alpha band: {source_alpha_band_index_1based}")
        else:
            print("gdal.Translate will process all bands (no source alpha band specified for dedicated use).")

        gdal_translate_options = gdal.TranslateOptions(**translate_options_dict)

        print(f"Executing gdal.Translate from {gdal_input_path} to {temp_gdal_file_path}")
        # Replace subprocess call with gdal.Translate
        ds = gdal.Translate(temp_gdal_file_path, gdal_input_path, options=gdal_translate_options)
        
        if ds is None:
            raise RuntimeError(f"gdal.Translate failed for {gdal_input_path}. Output dataset is None.")
        
        # It's good practice to dereference the dataset object when done if not used further,
        # allowing GDAL to close it and flush writes.
        ds = None 

        with rasterio.open(temp_gdal_file_path) as src:
            profile = src.profile.copy()

            color_band_indices_1based, color_band_indices_0based = [], []
            alpha_idx_0based = -1

            if src.colorinterp:
                for i, interp in enumerate(src.colorinterp):
                    if interp == ColorInterp.alpha: alpha_idx_0based = i
                    else:
                        color_band_indices_1based.append(i + 1)
                        color_band_indices_0based.append(i)
            
            if not color_band_indices_1based:
                warnings.warn("No specific color band interpretation found in temp file or only alpha. Assuming all non-alpha bands are color/data.")
                for i in range(src.count):
                    if i != alpha_idx_0based:
                        color_band_indices_1based.append(i + 1)
                        color_band_indices_0based.append(i)

            if not color_band_indices_1based: raise ValueError("No color/data bands identified to process from temp file.")
            num_output_color_bands = len(color_band_indices_0based)
            print(f"Identified {num_output_color_bands} color/data band(s) to process from temp file.")
            if alpha_idx_0based != -1: print("Temp file contains an alpha band (original source might have had one).")

            transform_mask_grid = src.transform
            print(f"Using temp file resolution for mask/hull generation: {src.width}x{src.height}")
            
            if source_alpha_band_index_1based is not None:
                print("Reading the single (alpha) band from the temporary file for mask generation...")
                alpha_band_data = src.read(1)
                interior_mask = binary_fill_holes(alpha_band_data != 0)
                del alpha_band_data
            else:
                print("No source alpha band explicitly used. Identifying color/data bands from temporary file for mask generation...")
                
                color_band_indices_1based, color_band_indices_0based = [], []
                alpha_idx_0based_in_temp = -1

                if src.colorinterp:
                    for i, interp_val in enumerate(src.colorinterp):
                        if interp_val == ColorInterp.alpha: 
                            alpha_idx_0based_in_temp = i
                        else:
                            color_band_indices_1based.append(i + 1)
                            color_band_indices_0based.append(i)
                
                if not color_band_indices_1based:
                    warnings.warn("No specific color band interpretation found in temp file (post gdal_translate) or only alpha. Assuming all non-alpha bands are color/data.")
                    for i in range(src.count):
                        if i != alpha_idx_0based_in_temp:
                            color_band_indices_1based.append(i + 1)
                            color_band_indices_0based.append(i)

                if not color_band_indices_1based: 
                    if src.count > 0 :
                         raise ValueError("No non-alpha data bands identified in the temporary file to process for hull (multi-band path). Ensure the source has usable data bands if no explicit alpha band is present.")
                    else:
                         raise ValueError("Temporary file has 0 bands after gdal_translate. Cannot proceed.")

                print(f"Identified {len(color_band_indices_1based)} color/data band(s) from temp file to process.")
                if alpha_idx_0based_in_temp != -1: 
                    print("Note: The temporary file (post gdal_translate) also contains an alpha band, which is being ignored for mask generation in this multi-band path.")

                print("Reading color data for mask grid from local temp file (multi-band path)...")
                color_data_mask_grid = src.read(color_band_indices_1based, resampling=Resampling.nearest)
                
                if color_data_mask_grid.ndim == 2 and len(color_band_indices_1based) == 1: 
                    color_data_mask_grid = color_data_mask_grid[np.newaxis, :, :]
                
                print("Generating interior mask for vectorization (from identified data bands)...")
                interior_mask = binary_fill_holes(np.any(color_data_mask_grid != 0, axis=0))
                del color_data_mask_grid
            
            print("Interior mask generated.")

            print("Vectorizing interior mask to shapes...")
            shapes = list(rio_features.shapes(interior_mask.astype(np.uint8), mask=interior_mask, transform=transform_mask_grid))
            del interior_mask
            print(f"Vectorization resulted in {len(shapes)} shapes.")
            
            final_shapely_geom = None
            if shapes:
                print("Processing shapes to generate final hull geometry...")
                geoms = [shape(s_dict) for s_dict, val in shapes if s_dict and s_dict.get("type")]
                valid_geoms = [g.buffer(0) if not g.is_valid else g for g in geoms] 
                valid_geoms = [g for g in valid_geoms if g.is_valid and not g.is_empty]
                if valid_geoms:
                    print(f"Uniting {len(valid_geoms)} valid geometries...")
                    unioned_geom = unary_union(valid_geoms)
                    if cutline_simplify_tolerance is not None and cutline_simplify_tolerance > 0:
                        print(f"Simplifying geometry with tolerance {cutline_simplify_tolerance}...")
                        unioned_geom = unioned_geom.simplify(cutline_simplify_tolerance, preserve_topology=True)
                    if not unioned_geom.is_empty: final_shapely_geom = unioned_geom
                print("Hull geometry processing complete.")
            
            num_cutline_features = 0
            print(f"Saving hull to GeoJSON: {hull_geojson_path}")
            if final_shapely_geom:
                gdf = gpd.GeoDataFrame(
                    {'id': [1], 'description': ['Hull of valid data']},
                    geometry=[final_shapely_geom],
                    crs=source_crs
                )
                gdf.to_file(hull_geojson_path, driver='GeoJSON', encoding='utf-8')
                num_cutline_features = len(gdf)
            else:
                warnings.warn(f"No valid hull geometry found for {geotiff_path_in}. Empty GeoJSON will be created.")
                gdf = gpd.GeoDataFrame(geometry=[], crs=source_crs)
                gdf.to_file(hull_geojson_path, driver='GeoJSON', encoding='utf-8')
            print(f"Hull saved ({num_cutline_features} feature(s)).")

            print(f"Hull definition process for {geotiff_path_in} finished. GeoJSON saved to {hull_geojson_path}")

            return geotiff_path_in, hull_geojson_path

    except Exception as e:
        print(f"An error occurred during hull definition: {e}")
        return None, None  # Ensure to return None for both outputs

    finally:
        if temp_gdal_file_path and os.path.exists(temp_gdal_file_path):
            try:
                os.remove(temp_gdal_file_path)
                print(f"Temporary file {temp_gdal_file_path} removed.")
            except OSError as e:
                warnings.warn(f"Error removing temporary file {temp_gdal_file_path}: {e}")

def find_seamlines(input_hulls, output_dir, tol, buffer_m: float):
    """
    Partition the full union of input hulls into seam regions via rasterized
    distance transforms, catch mis‑assigned islands, then write & return.
    
    Args:
        input_hulls (list): List of paths to input hull GeoJSON files.
        output_dir (str): Directory to save the output seamlines GeoJSON files.
        tol (float): Tolerance for hull generation.
    
    Returns:
        list: List of paths to the created seamline GeoJSON files.
    """
    seamline_files = [None] * len(input_hulls)  # Initialize a list to hold seamline file paths
    
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
    nx = int(np.ceil((maxx - minx) / tol))
    ny = int(np.ceil((maxy - miny) / tol))
    print(f"  Grid: {nx}×{ny} cells (tol={tol})")
    transform = Affine.translation(minx, miny) * Affine.scale(tol, tol)

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

    print("Step 7: Correcting island assignments via spatial join...")
    hulls = gpd.GeoDataFrame({"name": names, "geometry": polys}, crs=crs)
    parts["rep_pt"] = parts.geometry.representative_point()
    parts = parts.set_geometry("rep_pt")
    parts = gpd.sjoin(parts, hulls, how="left", predicate="within")
    parts = parts.rename(columns={"name": "owner"}).set_geometry("geometry")
    parts = parts.drop(columns=["rep_pt", "label", "index_right"])
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

def clip(input_raster, seamline_geojson, output_raster, keep_bands=None):
    """
    Create a VRT from the input raster using the seamline GeoJSON as a cutline.
    
    Args:
        input_raster (str): Path to the input raster file.
        seamline_geojson (str): Path to the seamline GeoJSON file.
        output_raster (str): Path to save the output raster file.
        keep_bands (list): List of band indices to keep (1-based).
    """

    if keep_bands is None:
        warp_options = gdal.WarpOptions(cutlineDSName=seamline_geojson, cropToCutline=True, dstAlpha=True, callback=gdal.TermProgress_nocb)
    else:
        warp_options = gdal.WarpOptions(srcBands=keep_bands, cutlineDSName=seamline_geojson, cropToCutline=True, dstAlpha=True, callback=gdal.TermProgress_nocb)
    
    print(f"Clipping {input_raster} using seamlines from {seamline_geojson}...")
    ds = gdal.Warp(output_raster, input_raster, options=warp_options)
    ds.FlushCache()
    ds = None

    print(f"Clipped raster created: {output_raster}")
    
    return output_raster

def mosaic(raster_files, output_merged, n_jobs='ALL_CPUS'):
    """
    Merge multiple raster files into a single raster with Cloud-Optimized GeoTIFF (COG) options.

    Args:
        raster_files (list): List of paths to raster files to merge.
        output_merged (str): Path to save the merged raster (COG).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        gdal.SetConfigOption("GDAL_CACHEMAX", "4096")  
        output_vrt = os.path.join(tmpdir, "merged.vrt")
        vrt_options = gdal.BuildVRTOptions(srcNodata=0, resolution='highest')
        gdal.BuildVRT(output_vrt, raster_files, options=vrt_options)

        print(f"Merging rasters into {output_merged}...")
        translate_to_cog_opts = gdal.TranslateOptions(
            format='COG',
            creationOptions=[
                'COMPRESS=DEFLATE',
                'BLOCKSIZE=256',
                'OVERVIEWS=AUTO',
                'BIGTIFF=YES',
                f'NUM_THREADS={n_jobs}',
            ],
            callback=gdal.TermProgress_nocb,
        )
        ds = gdal.Translate(
            output_merged,
            output_vrt,
            options=translate_to_cog_opts
        )
        ds.FlushCache()
        ds = None
        print(f"Merged raster created: {output_merged}")

def meta_mosaic(orthos_in: list, output_path: str, keep_bands: list = None, intermediary_dir: str = None, tol: float = 1.0, n_workers: int = 4, buffer_m: float = 0.5):
    """
    Processes a list of input orthomosaics to produce a single, merged mosaic.

    The workflow includes:
    1. Defining hulls for each input orthomosaic.
    2. Finding seamlines between the hulls.
    3. Clipping each orthomosaic to its corresponding seamline region.
    4. Mosaicking the clipped rasters into a final output.

    Args:
        orthos_in (list): List of paths to input orthomosaic GeoTIFF files.
        output_path (str): Path to save the final merged mosaic GeoTIFF.
        keep_bands (list, optional): List of 1-based band indices to keep from the input rasters. 
                                     If None, all bands are kept. Defaults to None.
        intermediary_dir (str, optional): Directory to store intermediate files (hulls, seamlines, clipped rasters).
                                          If None, a temporary directory is created and deleted upon completion.
                                          Defaults to None.
        tol (float, optional): Tolerance used for hull generation and seamline finding. Defaults to 1.0.
        n_workers (int, optional): Number of parallel workers to use for processing. Defaults to 4.
        buffer_m (float, optional): Buffer distance in meters (or CRS units) to apply to seamlines. Defaults to 0.5.
    """
    
    manage_temp_dir = False
    if intermediary_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        work_dir = temp_dir_obj.name
        manage_temp_dir = True
        print(f"Using temporary directory for intermediate files: {work_dir}")
    else:
        work_dir = intermediary_dir
        os.makedirs(work_dir, exist_ok=True)
        print(f"Using provided directory for intermediate files: {work_dir}")

    try:
        # Step 1: Define Hulls
        print(f"Step 1: Defining hulls for {len(orthos_in)} orthomosaics...")
        
        def _extract_hull_batch(ortho_path, idx):
            output_hull_geojson = os.path.join(work_dir, f"hull_{idx}_{os.path.basename(ortho_path).replace('.tif', '.geojson')}")
            _, hull_file_out = define_hull(
                geotiff_path_in=ortho_path,
                hull_geojson_path=output_hull_geojson,
                tol=tol
            )
            return ortho_path, hull_file_out

        hull_results = joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(_extract_hull_batch)(ortho_path, i) for i, ortho_path in enumerate(orthos_in)
        )
        
        # Filter out None results from failed hull definitions
        valid_hull_results = [res for res in hull_results if res is not None and res[1] is not None]
        if not valid_hull_results:
            raise RuntimeError("No valid hulls could be generated. Aborting meta_mosaic.")

        processed_orthos_in = [r[0] for r in valid_hull_results]
        hulls_out = [r[1] for r in valid_hull_results]
        print(f"Successfully generated {len(hulls_out)} hulls.")

        # Step 2: Find Seamlines
        print("Step 2: Finding seamlines...")
        seamline_files = find_seamlines(
            input_hulls=hulls_out,
            output_dir=work_dir, # find_seamlines will create files like seamline_1.geojson, seamline_2.geojson etc.
            tol=tol,
            buffer_m=buffer_m
        )
        
        # Ensure seamline_files align with processed_orthos_in based on the order from find_seamlines
        # find_seamlines returns a list that should correspond to the order of hulls_out.
        # If find_seamlines modifies the order or can return fewer items, this needs careful handling.
        # Assuming find_seamlines preserves order and number corresponding to hulls_out.
        if len(seamline_files) != len(processed_orthos_in):
             # This might happen if some hulls lead to no seamlines or if find_seamlines filters.
             # For now, we'll proceed, but this could be a point of failure if indexing mismatches.
            print(f"Warning: Number of seamlines ({len(seamline_files)}) does not match number of processed orthos ({len(processed_orthos_in)}).")
            # Attempt to filter orthos that don't have a corresponding seamline (if None is placeholder)
            valid_seamlines_with_orthos = []
            temp_processed_orthos_in = []
            for i, seam_file in enumerate(seamline_files):
                if seam_file is not None and i < len(processed_orthos_in) : # check ortho index exists
                    valid_seamlines_with_orthos.append(seam_file)
                    temp_processed_orthos_in.append(processed_orthos_in[i])
                else:
                    print(f"Warning: No seamline generated or index issue for an ortho. Original index {i}")

            seamline_files = valid_seamlines_with_orthos
            processed_orthos_in = temp_processed_orthos_in
            
            if not seamline_files:
                raise RuntimeError("No valid seamlines available to proceed with clipping.")


        print(f"Generated {len(seamline_files)} seamline files.")
        print("Input orthomosaics for clipping:")
        for f_idx, f_path in enumerate(processed_orthos_in):
            print(f"  Ortho {f_idx}: {f_path} -> Seamline {f_idx}: {seamline_files[f_idx] if f_idx < len(seamline_files) else 'N/A'}")


        # Step 3: Clip Rasters
        print("Step 3: Clipping rasters...")
        
        def _clip_single_raster(ortho_path, seamline_geojson_path, clip_idx, bands_to_keep):
            output_clipped_raster = os.path.join(work_dir, f"clipped_{clip_idx}_{os.path.basename(ortho_path)}")
            # Ensure seamline_geojson_path is not None
            if seamline_geojson_path is None:
                print(f"Warning: Skipping clipping for {ortho_path} as its seamline is missing.")
                return None
            return clip(ortho_path, seamline_geojson_path, output_clipped_raster, keep_bands=bands_to_keep)

        clipped_rasters_paths = joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(_clip_single_raster)(
                processed_orthos_in[i],
                seamline_files[i], # Use the seamline file corresponding to the ortho
                i,
                keep_bands
            )
            for i in range(len(processed_orthos_in)) if i < len(seamline_files) # Ensure we don't go out of bounds for seamline_files
        )
        
        # Filter out None results from failed clippings
        final_clipped_rasters = [p for p in clipped_rasters_paths if p is not None]
        if not final_clipped_rasters:
            raise RuntimeError("No rasters were successfully clipped. Aborting mosaic.")
        print(f"Successfully clipped {len(final_clipped_rasters)} rasters.")

        # Step 4: Mosaic Rasters
        print("Step 4: Mosaicking clipped rasters...")
        # Ensure n_jobs for mosaic is at least 1
        mosaic_n_jobs = max(1, n_workers) if isinstance(n_workers, int) else 'ALL_CPUS'
        
        mosaic(final_clipped_rasters, output_path, n_jobs=mosaic_n_jobs)
        print(f"Successfully created final mosaic: {output_path}")

    finally:
        if manage_temp_dir:
            try:
                temp_dir_obj.cleanup()
                print(f"Temporary directory {work_dir} cleaned up.")
            except Exception as e:
                warnings.warn(f"Could not clean up temporary directory {work_dir}: {e}")
    
    return output_path
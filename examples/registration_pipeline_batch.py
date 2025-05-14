import os
import cv2
import geopandas as gpd
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from IPython.display import Image
from osgeo import gdal
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds, transform as window_transform
from shapely.geometry import Point

from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from tqdm import tqdm
from queue import Queue
import uuid
import time
import concurrent.futures
import threading
import pandas as pd

device = 'mps'

matcher = KF.LoFTR(pretrained='outdoor').to(device).eval()

def get_bbox_bounds(point_gdf, width_length, height_length=None, buffer=0):
    """
    Given a GeoDataFrame with a single point geometry, return (xmin, xmax, ymin, ymax)
    for a rectangular bounding box of given width and height centered at the point.
    
    Args:
        point_gdf: GeoDataFrame with a single point
        width_length: Width of the chip in CRS units
        height_length: Height of the chip in CRS units (defaults to 0.75*width_length if None)
        buffer: Additional buffer around the chip in CRS units
    """
    if height_length is None:
        height_length = width_length * 0.75
        
    half_width = width_length / 2 + buffer
    half_height = height_length / 2 + buffer
    x = point_gdf.geometry.iloc[0].x
    y = point_gdf.geometry.iloc[0].y
    xmin = x - half_width
    xmax = x + half_width
    ymin = y - half_height
    ymax = y + half_height
    return xmin, ymin, xmax, ymax

def load_chips(
    point_gdf,
    unregistered_path,
    registered_path,
    width_length,
    height_length=None,
    buffer=0
) -> tuple:
    """
    Read matching window chips from an unregistered and a registered raster,
    storing pixel arrays, profiles, and windows for GCP generation.

    Args:
        point_gdf: GeoDataFrame with a single point
        unregistered_path: Path to the poorly-referenced raster
        registered_path: Path to the well-referenced raster
        width_length: Width of the chip in CRS units
        height_length: Height of the chip in CRS units (defaults to 0.75*width_length if None)
        buffer: Additional buffer around the chip in CRS units
    
    Returns:
        Two dicts: un_chip and reg_chip, each with keys 'array', 'profile', 'window', and 'bounds'
    """
    if height_length is None:
        height_length = width_length * 0.75
        
    # Determine bounding box around the point with buffer
    bbox = get_bbox_bounds(point_gdf, width_length, height_length, buffer)

    # Unregistered chip
    with rasterio.open(unregistered_path) as un_ds:
        un_window = from_bounds(*bbox, un_ds.transform)
        un_array = un_ds.read(window=un_window)
        un_profile = un_ds.profile.copy()
        un_crs = un_ds.crs
        
        # Update profile with window transform
        win_transform = window_transform(un_window, un_ds.transform)
        un_profile.update({
            'height': un_window.height,
            'width': un_window.width,
            'transform': win_transform
        })

    un_chip = dict(array=un_array, profile=un_profile, window=un_window, bounds=bbox)

    # Registered chip: warp to unregistered CRS
    with rasterio.open(registered_path) as reg_ds:
        with WarpedVRT(
            reg_ds,
            crs=un_crs,
            resampling=Resampling.bilinear
        ) as vrt:
            reg_window = from_bounds(*bbox, vrt.transform)
            reg_array = vrt.read(window=reg_window)
            reg_profile = vrt.profile.copy()
            
            # Update profile with window transform
            win_transform = window_transform(reg_window, vrt.transform)
            reg_profile.update({
                'height': reg_window.height,
                'width': reg_window.width,
                'transform': win_transform
            })

    reg_chip = dict(array=reg_array, profile=reg_profile, window=reg_window, bounds=bbox)

    return un_chip, reg_chip

def rasterio_to_torch_tensor(numpy_array: np.ndarray, bands: int = [0, 1, 2]) -> torch.Tensor:
    """
    Converts a rasterio NumPy array from rasterio window output (C X H x W)
    to a PyTorch tensor of shape (1, C, H, W).

    Args:
        numpy_array: Input NumPy array with shape (C, H, W).

    Returns:
        PyTorch tensor with shape (1, C, H, W).
    """
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(numpy_array[bands]).unsqueeze(0)

    return tensor

def resize_for_matching(img_array, target_size=(512, 512)):
    """
    Resize an image array to target size for feature matching.
    Preserves original dimensions for coordinate rescaling.
    
    Args:
        img_array: Input image array of shape (C, H, W)
        target_size: Target size as (height, width) tuple
        
    Returns:
        Tuple of (resized_array, scale_factors)
    """
    # Get original dimensions
    _, orig_h, orig_w = img_array.shape
    target_h, target_w = target_size
    
    # Calculate scale factors
    h_scale = orig_h / target_h
    w_scale = orig_w / target_w
    
    # Create resized array
    resized_array = np.zeros((img_array.shape[0], target_h, target_w), dtype=img_array.dtype)
    
    # Resize each band
    for i in range(img_array.shape[0]):
        band = img_array[i]
        resized = cv2.resize(band, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_array[i] = resized
    
    return resized_array, (w_scale, h_scale)

# def apply_clahe_to_grayscale_batch(img_tensor: torch.Tensor,
#                                     clip_limit: float = 2.0,
#                                     tile_grid_size: tuple = (8, 8)) -> torch.Tensor:
#     """
#     Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a batch of RGB images
#     after converting them to grayscale using Kornia.

#     Args:
#         img_tensor (torch.Tensor): RGB image batch tensor of shape (B, 3, H, W), float32 in [0, 1].
#         clip_limit (float): CLAHE clip limit.
#         tile_grid_size (tuple): Size of grid for histogram equalization (tiles in x and y).

#     Returns:
#         torch.Tensor: CLAHE-enhanced grayscale image batch tensor of shape (B, 1, H, W), float32 in [0, 1].
#     """
#     assert img_tensor.dim() == 4 and img_tensor.shape[1] == 3, "Input must be RGB batch (B, 3, H, W)"

#     # Convert to grayscale with Kornia
#     grayscale = K.color.rgb_to_grayscale(img_tensor)  # (B, 1, H, W)

#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     enhanced = []

#     for img in grayscale:
#         img_np = (img.squeeze().cpu().numpy() * 255).astype('uint8')  # shape: (H, W)
#         cl_img = clahe.apply(img_np)
#         cl_tensor = torch.tensor(cl_img, dtype=torch.float32).div(255).unsqueeze(0)  # shape: (1, H, W)
#         enhanced.append(cl_tensor)

#     return torch.stack(enhanced)  # shape: (B, 1, H, W)

def batch_get_loftr_matches(img1_batch, img2_batch, device: str = 'cpu', target_size=(480, 640), #note H X W
                            loftr_space_reproj_threshold_levels: list[float] = [0.5, 0.75, 1.0, 2.0],
                            batch_size: int = None):
    """
    Process multiple image pairs in a single batch for efficient GPU usage.
    Uses multiple confidence levels and multiple reprojection thresholds for robust fundamental matrix estimation.
    Resizes images to target_size for matching, then rescales keypoints back.
    
    Args:
        img1_batch: List of registered image arrays
        img2_batch: List of unregistered image arrays
        device: Device to run inference on
        target_size: Target size for resizing images before matching (default: 480x640; note H X W)
        loftr_space_reproj_threshold_levels: List of RANSAC reprojection thresholds in LoFTR's 
                                             resized image space to try (e.g., [0.5, 0.75, 1.0] pixels)
        batch_size: Maximum number of image pairs to process at once on GPU (defaults to input batch size if None)
    
    Returns:
        List of tuples (mkpts0, mkpts1, inliers) for each image pair
    """
    # Check for empty batches
    if not img1_batch or not img2_batch:
        return []
    
    # Initialize LoFTR matcher once for the batch
    #matcher = KF.LoFTR(pretrained='outdoor').to(device).eval()
    input_batch_size = len(img1_batch)
    results = []
    
    # Preprocess all images in parallel with ThreadPoolExecutor for CPU operations
    with ThreadPoolExecutor() as executor:
        # Function to preprocess single image
        def preprocess_image(img, scale=True):
            if np.all(img == 0):
                return None, None
            
            if scale:
                img_resized, img_scale_factors = resize_for_matching(img, target_size)
            else:
                img_resized, img_scale_factors = img, (1.0, 1.0)
                
            # Convert to tensor
            img_tensor = rasterio_to_torch_tensor(img_resized).float().div(255.0)
            # Convert to grayscale
            img_gray = K.color.rgb_to_grayscale(img_tensor)
            #img_gray = apply_clahe_to_grayscale_batch(img_tensor)
            return img_gray, img_scale_factors
        
        # Submit all preprocessing tasks
        img1_futures = [executor.submit(preprocess_image, img) for img in img1_batch]
        img2_futures = [executor.submit(preprocess_image, img) for img in img2_batch]
        
        # Collect results
        processed_pairs = []
        for i in range(input_batch_size):
            img1_gray, img1_scale = img1_futures[i].result()
            img2_gray, img2_scale = img2_futures[i].result()
            
            if img1_gray is None or img2_gray is None:
                # Handle all-black images
                empty_pts = np.empty((0, 2), dtype=float)
                empty_inliers = np.empty((0,), dtype=bool)
                results.append((empty_pts, empty_pts, empty_inliers))
            else:
                processed_pairs.append((img1_gray, img2_gray, img1_scale, img2_scale))
    
    # Skip further processing if all images were invalid
    if not processed_pairs:
        return results
    
    # Process valid image pairs in sub-batches for optimal GPU memory usage
    MAX_GPU_BATCH = batch_size if batch_size is not None else input_batch_size  # Adjust based on your GPU memory
    
    for batch_start in range(0, len(processed_pairs), MAX_GPU_BATCH):
        batch_end = min(batch_start + MAX_GPU_BATCH, len(processed_pairs))
        current_batch = processed_pairs[batch_start:batch_end]
        
        # Prepare batched tensors for GPU processing
        img1_tensors = torch.cat([pair[0] for pair in current_batch]).to(device)
        img2_tensors = torch.cat([pair[1] for pair in current_batch]).to(device)
        
        # Process batch on GPU
        with torch.inference_mode():
            # Reshape to batch dimension
            img1_tensors = img1_tensors.reshape(-1, 1, target_size[0], target_size[1])
            img2_tensors = img2_tensors.reshape(-1, 1, target_size[0], target_size[1])
            
            # Process each pair in the batch
            batch_results = []
            for i in range(len(current_batch)):
                input_dict = {
                    "image0": img1_tensors[i:i+1], 
                    "image1": img2_tensors[i:i+1]
                }
                correspondences = matcher(input_dict)
                
                # Get keypoints
                mkpts0_resized = correspondences["keypoints0"].cpu().numpy()
                mkpts1_resized = correspondences["keypoints1"].cpu().numpy()
                
                # Get scale factors for this pair
                img1_scale = current_batch[i][2]
                img2_scale = current_batch[i][3]
                
                # Rescale keypoints back to original image dimensions
                mkpts0 = mkpts0_resized * np.array(img1_scale)
                mkpts1 = mkpts1_resized * np.array(img2_scale)
                
                # Fundamental matrix inlier filtering
                inliers = None # Initialize inliers for this pair
                
                if mkpts0.shape[0] >= 7:
                    confidence_levels = [0.999, 0.95]
                    avg_img1_scale_factor = (img1_scale[0] + img1_scale[1]) / 2.0

                    for current_loftr_thresh_px in loftr_space_reproj_threshold_levels:
                        adaptive_ransac_reproj_threshold = current_loftr_thresh_px * avg_img1_scale_factor
                        
                        for confidence in confidence_levels:
                            try:
                                Fm, inliers_cv = cv2.findFundamentalMat(
                                    mkpts0, mkpts1,
                                    method=cv2.USAC_MAGSAC,
                                    ransacReprojThreshold=adaptive_ransac_reproj_threshold,
                                    confidence=confidence,
                                    maxIters=100000
                                )
                                
                                if Fm is not None and Fm.shape[0] == 3 and not np.all(Fm == 0):
                                    inliers = inliers_cv.ravel() > 0
                                    break  # Found inliers, break from confidence loop
                            except cv2.error:
                                continue
                        
                        if inliers is not None:
                            break # Found inliers, break from reprojection threshold loop
                
                if inliers is None: # If no inliers found after all attempts
                    inliers = np.zeros(mkpts0.shape[0], dtype=bool)
                
                batch_results.append((mkpts0, mkpts1, inliers))
            
            # Add batch results to the overall results
            results.extend(batch_results[:(batch_end - batch_start)])
    
    # Fill in any missing results (should match the original batch size)
    while len(results) < input_batch_size:
        empty_pts = np.empty((0, 2), dtype=float)
        empty_inliers = np.empty((0,), dtype=bool)
        results.append((empty_pts, empty_pts, empty_inliers))
    
    return results

def generate_chip_gcps(un_chip, reg_chip, mkpts_un, mkpts_reg, inliers):
    """Generate GCPs for a single chip pair"""
    if mkpts_reg.size == 0 or mkpts_un.size == 0 or not np.any(inliers):
        return []
        
    # Pick only inlier points
    reg_pts = mkpts_reg[inliers]
    un_pts = mkpts_un[inliers]
    
    # Extract transforms & offsets
    tf_reg = reg_chip['profile']['transform']
    tf_un = un_chip['profile']['transform']
    
    # Build GCP list
    gcps = []
    for (cx, ry), (ux, uy) in zip(reg_pts, un_pts):
        geo_x, geo_y = tf_reg * (float(cx), float(ry))
        pixel = float(ux)
        line = float(uy)
        gcps.append(gdal.GCP(geo_x, geo_y, 0.0, pixel, line))
    
    return gcps

def warp_chip(un_chip, gcps, temp_dir, polynomial_order=3, resampling="cubic"):
    """Warp a single chip using GCPs and save to temporary file"""
    # Generate unique filename
    chip_filename = os.path.join(temp_dir, f"chip_{uuid.uuid4().hex}.tif")
    
    # Create a memory file from the chip array
    mem_file = f"/vsimem/temp_chip_{uuid.uuid4().hex}.tif"
    with rasterio.open(
        mem_file, 'w',
        driver='GTiff',
        height=un_chip['array'].shape[1],
        width=un_chip['array'].shape[2],
        count=un_chip['array'].shape[0],
        dtype=un_chip['array'].dtype,
        transform=un_chip['profile']['transform'],
        crs=un_chip['profile']['crs'],
        nodata=0  # Set nodata value explicitly
    ) as dst:
        dst.write(un_chip['array'])
    
    # Enable GDAL exceptions
    gdal.UseExceptions()
    
    try:
        # Create VRT with GCPs
        vrt_ds = gdal.Translate(
            "",  # in memory
            mem_file,
            options=gdal.TranslateOptions(format="VRT", GCPs=gcps)
        )
        
        # Get source CRS
        crs_wkt = un_chip['profile']['crs'].wkt
        
        # Warp the chip with nodata handling
        warp_opts = gdal.WarpOptions(
            format="GTiff",
            polynomialOrder=polynomial_order,
            resampleAlg=resampling,
            dstSRS=crs_wkt,
            srcNodata=0,      # Source nodata value
            dstNodata=0,      # Destination nodata value 
            warpOptions=[
                "SOURCE_EXTRA=5",  # Read extra pixels from source for better edge handling
                "WRITE_FLUSH=YES"   # Write data as it's processed
            ]
        )
        gdal.Warp(chip_filename, vrt_ds, options=warp_opts)
        
        # Cleanup
        vrt_ds = None
        gdal.Unlink(mem_file)
        
        return chip_filename
    except Exception as e:
        # If there's an error, clean up and re-raise
        gdal.Unlink(mem_file)
        raise e

def generate_grid_points(raster_path, width_length, height_length=None):
    """
    Build a GeoDataFrame of points that tile the raster with centers 
    spaced at width_length intervals in X direction and height_length in Y direction
    """
    if height_length is None:
        height_length = width_length * 0.75
        
    with rasterio.open(raster_path) as ds:
        left, bottom, right, top = ds.bounds
        crs = ds.crs

    # Generate grid points spaced at width_length and height_length
    xs = np.arange(left + width_length/2, right, width_length)
    ys = np.arange(bottom + height_length/2, top, height_length)

    pts = [Point(x, y) for y in ys for x in xs]
    return gpd.GeoDataFrame({'geometry': pts}, crs=crs)

def process_batch(batch_id, points, unreg_path, reg_path, width_length, temp_dir, device, height_length=None, buffer=2, polynomial_order=3, resampling="cubic", batch_size=None):
    """Process a batch of points, generating and warping chips"""
    if height_length is None:
        height_length = width_length * 0.75
        
    start_time = time.time()
    warped_files = []
    chip_stats = []  # Add this to collect chip statistics
    original_res = None  # To store the original resolution
    try:
        # Load all chips for the batch in parallel
        un_chips_list = []
        reg_chips_list = []
        original_indices_list = [] # To store original unique indices of loaded chips
        
        def load_chip_for_point(row_data):
            idx, row = row_data
            try:
                # Create a proper GeoDataFrame for each point
                pt_gdf = gpd.GeoDataFrame({'geometry': [row.geometry]}, crs=points.crs)
                un_chip, reg_chip = load_chips(pt_gdf, unreg_path, reg_path, width_length, height_length, buffer)
                return idx, (un_chip, reg_chip, True)
            except Exception as e:
                print(f"Error loading chip at point {idx}: {str(e)}")
                return idx, (None, None, False)
        
        # Use ThreadPoolExecutor for parallel I/O operations
        with ThreadPoolExecutor() as executor:
            # Submit all point loading tasks
            futures_load_chips = [executor.submit(load_chip_for_point, (idx, row)) 
                      for idx, row in points.iterrows()]
            
            # Collect results as they complete
            for future in as_completed(futures_load_chips):
                idx, (un_chip, reg_chip, success) = future.result()
                if success:
                    un_chips_list.append(un_chip)
                    reg_chips_list.append(reg_chip)
                    original_indices_list.append(idx) # Store the original unique index
                    
                    # Store the original resolution from the first valid chip
                    if original_res is None and un_chip is not None:
                        transform = un_chip['profile']['transform']
                        original_res = (abs(transform[0]), abs(transform[4]))
        
        if not un_chips_list or not reg_chips_list:
            print(f"No valid chips found in batch {batch_id}")
            return batch_id, [], [], time.time() - start_time
        
        # Extract arrays for batch processing
        reg_arrays = [reg_chip['array'] for reg_chip in reg_chips_list]
        un_arrays = [un_chip['array'] for un_chip in un_chips_list]
        
        # Get matches for all pairs in batch
        batch_match_results = batch_get_loftr_matches(reg_arrays, un_arrays, device, batch_size=batch_size)
        
        # Process each chip in parallel
        def process_chip(original_idx, match_data, un_chip, reg_chip):
            mkpts_reg, mkpts_un, inliers = match_data
            
            try:
                # Count inliers
                inlier_count = np.sum(inliers)
                
                # Generate GCPs for the chip
                gcps = generate_chip_gcps(un_chip, reg_chip, mkpts_un, mkpts_reg, inliers)
                gcp_count = len(gcps)
                
                # Track if warping succeeded
                warp_success = False
                warped_file = None
                
                if gcp_count >= 10:  # Only warp if we have enough GCPs
                    # Warp the chip and save to temp file
                    warped_file = warp_chip(un_chip, gcps, temp_dir, polynomial_order, resampling)
                    warp_success = True if warped_file else False
                
                # Get chip centroid
                xmin, ymin, xmax, ymax = un_chip['bounds']
                centroid_x = (xmin + xmax) / 2
                centroid_y = (ymin + ymax) / 2
                chip_info = {
                    "idx": original_idx, # Use the original unique idx
                    "inlier_count": int(inlier_count),
                    "gcp_count": gcp_count,
                    "warp_success": warp_success,
                    "centroid": (centroid_x, centroid_y)
                }
                
                # Print chip stats
                print(f"Chip {original_idx} (batch {batch_id}): inliers={inlier_count}, GCPs={gcp_count}, warped={warp_success}")
                
                return warped_file, un_chip['bounds'], chip_info
            except Exception as e:
                print(f"Error processing chip {original_idx} (batch {batch_id}): {str(e)}")
                xmin, ymin, xmax, ymax = un_chip['bounds']
                centroid_x = (xmin + xmax) / 2
                centroid_y = (ymin + ymax) / 2
                chip_info = {
                    "idx": original_idx, # Use the original unique idx
                    "inlier_count": 0,
                    "gcp_count": 0,
                    "warp_success": False,
                    "centroid": (centroid_x, centroid_y)
                }
                return None, un_chip['bounds'], chip_info
        
        # Process chips in parallel
        with ThreadPoolExecutor() as executor:
            futures_chip_processing = []
            for i in range(len(original_indices_list)): # Iterate over successfully loaded chips
                future = executor.submit(
                    process_chip,
                    original_indices_list[i],      # Pass the unique original_idx
                    batch_match_results[i],        # Corresponding match data
                    un_chips_list[i],              # Corresponding un_chip
                    reg_chips_list[i]              # Corresponding reg_chip
                )
                futures_chip_processing.append(future)
            
            # Collect results
            for future in as_completed(futures_chip_processing):
                warped_file_path, chip_bounds, chip_info_dict = future.result()
                if warped_file_path:
                    # original_res is from the process_batch scope
                    warped_files.append((warped_file_path, chip_bounds, original_res))
                chip_stats.append(chip_info_dict)  # Store chip info (already has unique idx)
        
        duration = time.time() - start_time
        return batch_id, warped_files, chip_stats, duration
    
    except Exception as e:
        print(f"Error in batch {batch_id}: {str(e)}")
        return batch_id, [], [], time.time() - start_time

def merge_warped_chips(warped_files, output_path, buffer):
    """
    Merge warped chips, properly removing buffer areas without losing any data.
    Uses parallel processing for cropping operations.
    """
    # Check if there are any warped files to merge
    if not warped_files:
        print("No warped chips to merge. Cannot create output.")
        return
    
    # Extract the original resolution from the chip data
    # Each warped_file entry is a tuple of (file_path, bounds, original_res)
    pixel_width, pixel_height = warped_files[0][2]
    
    # Create temporary directory for cropped chips
    with tempfile.TemporaryDirectory() as temp_crop_dir:
        cropped_files = []
        
        # Define function to crop a single chip
        def crop_chip(chip_data):
            i, (file_path, bounds, _) = chip_data
            
            # Extract bounds
            xmin, ymin, xmax, ymax = bounds
            
            # Calculate the non-buffer area (core chip)
            core_xmin = xmin + buffer
            core_xmax = xmax - buffer
            core_ymin = ymin + buffer
            core_ymax = ymax - buffer
            
            # Create output path for cropped chip
            crop_path = os.path.join(temp_crop_dir, f"crop_{i}.tif")
            
            try:
                # Use gdal_warp with exact georeferenced bounds for cropping
                warp_options = gdal.WarpOptions(
                    format="GTiff",
                    outputBounds=[core_xmin, core_ymin, core_xmax, core_ymax],
                    xRes=pixel_width,
                    yRes=pixel_height,
                    multithread=True,
                    warpOptions=["OPTIMIZE_SIZE=YES"],
                    dstNodata=0
                )
                
                gdal.Warp(crop_path, file_path, options=warp_options)
                return i, crop_path
            except Exception as e:
                print(f"Error cropping chip {i}: {str(e)}")
                return i, None
        
        # Process chips in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Submit all cropping tasks
            futures = [executor.submit(crop_chip, (i, file_data)) 
                      for i, file_data in enumerate(warped_files)]
            
            # Process results as they complete
            with tqdm(total=len(futures), desc="Cropping chips") as pbar:
                for future in as_completed(futures):
                    i, crop_path = future.result()
                    if crop_path:
                        cropped_files.append(crop_path)
                    pbar.update(1)
                
        print(f"Successfully cropped all {len(cropped_files)} chips")
        
        # Build VRT from all cropped chips
        vrt_path = "/vsimem/merged_cropped.vrt"
        
        vrt_options = gdal.BuildVRTOptions(
            xRes=pixel_width,
            yRes=pixel_height
        )
            
        gdal.BuildVRT(vrt_path, cropped_files, options=vrt_options)
        
        # Translate to final output
        gdal.Translate(
            output_path,
            vrt_path,
            options=gdal.TranslateOptions(
                format="GTiff",
                creationOptions=[
                    "COMPRESS=LZW", 
                    "PREDICTOR=2",
                    "BIGTIFF=YES",
                    "TILED=YES"
                ]
            )
        )
        
        # Build overviews
        ds = gdal.Open(output_path, gdal.GA_Update)
        if ds:
            gdal.SetConfigOption('COMPRESS_OVERVIEW', 'LZW')
            gdal.SetConfigOption('PREDICTOR_OVERVIEW', '2')
            overview_levels = [2, 4, 8, 16, 32]
            ds.BuildOverviews("NEAREST", overview_levels)
            ds = None  # Close the dataset
        
        # Clean up
        gdal.Unlink(vrt_path)

def register_raster_with_chips(unreg_path, reg_path, output_path, width_length, height_length=None, buffer=2,
                              device='cpu', max_workers=4, batch_size=8, 
                              polynomial_order=3, resampling="cubic"):
    """Main function to register a raster using chip-based approach"""
    if height_length is None:
        height_length = width_length * 0.75
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Create temporary directory for chips
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1) Build grid of points spaced at width_length and height_length
        grid_gdf = generate_grid_points(unreg_path, width_length, height_length)
        n_tiles = len(grid_gdf)
        
        # Group points into batches - preserve geometry by using proper GeoDataFrame slicing
        point_batches = []
        for i in range(0, n_tiles, batch_size):
            # Create proper GeoDataFrame slice that preserves geometry column
            batch = grid_gdf.iloc[i:i+batch_size].copy()
            point_batches.append(batch)
            
        print(f"Created {len(point_batches)} batches from {n_tiles} points")
        
        # Setup for monitoring
        warped_files = []
        all_chip_stats = []  # Add this to collect all chip statistics
        processed_batches = 0
        total_batches = len(point_batches)
        processing_times = []
        running = True
        
        # Heartbeat thread for progress reporting
        def heartbeat():
            while running:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Progress: {processed_batches}/{total_batches} batches " 
                      f"({processed_batches/total_batches*100:.1f}%) - {len(warped_files)} chips processed")
                if processing_times and processed_batches > 0:
                    avg_time = sum(processing_times) / len(processing_times)
                    est_remaining = avg_time * (total_batches - processed_batches) / max_workers
                    print(f"    Avg batch time: {avg_time:.1f}s, Est. remaining: {est_remaining/60:.1f} min")
                time.sleep(10)
        
        # Start heartbeat thread
        monitor = threading.Thread(target=heartbeat)
        monitor.daemon = True
        monitor.start()
        
        # Submit batches with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            
            # Submit initial batch of jobs
            for batch_id, batch in enumerate(point_batches):
                future = exe.submit(
                    process_batch, batch_id, batch, unreg_path, reg_path,
                    width_length, temp_dir, device, height_length, buffer, polynomial_order, resampling,
                    batch_size=batch_size
                )
                futures[future] = batch_id
            
            # Process results with timeouts
            for future in as_completed(futures):
                try:
                    # Get result with 5-minute timeout
                    batch_id, batch_files, batch_stats, duration = future.result()
                    
                    # Update tracking
                    processed_batches += 1
                    processing_times.append(duration)
                    if batch_files:
                        warped_files.extend(batch_files)
                    
                    # Add batch stats to all_chip_stats
                    all_chip_stats.extend(batch_stats)
                        
                    # Log individual batch completion
                    print(f"Batch {batch_id} completed in {duration:.1f}s with {len(batch_files)} warped chips")
                    
                except Exception as e:
                    # Handle any other errors
                    print(f"Error processing batch {futures[future]}: {str(e)}")
                    processed_batches += 1
        
        # Stop heartbeat thread
        running = False
        monitor.join(timeout=1)
        
        print(f"Completed all batches. Merging {len(warped_files)} warped chips...")
        
        # Merge all warped chips
        merge_warped_chips(warped_files, output_path, buffer)
        
        # Save chip statistics as GeoDataFrame
        stats_df = pd.DataFrame([
            {
                "idx": stat["idx"],
                "inlier_count": stat["inlier_count"],
                "gcp_count": stat["gcp_count"],
                "warp_success": stat["warp_success"],
                "geometry": Point(stat["centroid"])
            }
            for stat in all_chip_stats
        ])
        
        # Convert to GeoDataFrame
        with rasterio.open(unreg_path) as src:
            crs = src.crs
        
        stats_gdf = gpd.GeoDataFrame(stats_df, geometry="geometry", crs=crs)
        
        # Save to GeoJSON instead of GeoPackage
        stats_output_path = os.path.splitext(output_path)[0] + "_chip_stats.geojson"
        stats_gdf.to_file(stats_output_path, driver="GeoJSON")
        
        print(f"Registration complete. Output written to {output_path}")
        print(f"Chip statistics saved to {stats_output_path}")

def calculate_chip_crs_parameters(
    parent_raster_path: str,
    target_chip_width_px: int = 640,  # Default width
    target_chip_height_px: int = 480, # Default height
    buffer_fraction: float = 0.1      # Default buffer fraction
) -> tuple[float, float, float, float, float]:
    """
    Calculates the core dimensions and buffer in CRS units for chipping,
    and returns the raster resolution.

    Args:
        parent_raster_path: Path to the raster from which to derive resolution.
        target_chip_width_px: Desired total width of the chip in pixels.
        target_chip_height_px: Desired total height of the chip in pixels.
        buffer_fraction: Fraction of the target_chip_width_px to use as buffer
                         (applied to CRS units of total width).

    Returns:
        A tuple containing:
            - base_core_width_crs: Core width in CRS units (for size_scale=1).
            - base_core_height_crs: Core height in CRS units (for size_scale=1).
            - base_buffer_crs: Buffer amount in CRS units (for size_scale=1).
            - res_x: Pixel width in CRS units.
            - res_y: Pixel height in CRS units.
    """
    with rasterio.open(parent_raster_path) as src:
        res_x, res_y = (abs(src.transform[0]), abs(src.transform[4]))

    # Base total dimensions in CRS units
    base_total_width_crs = target_chip_width_px * res_x
    base_total_height_crs = target_chip_height_px * res_y

    # Base buffer amount in CRS units (for one side)
    base_buffer_crs = buffer_fraction * base_total_width_crs

    # Base core dimensions in CRS units
    base_core_width_crs = base_total_width_crs - (2 * base_buffer_crs)
    base_core_height_crs = base_total_height_crs - (2 * base_buffer_crs)

    if base_core_height_crs <= 0:
        raise ValueError(
            f"Calculated base_core_height_crs ({base_core_height_crs:.2f}) is not positive. "
            "Buffer might be too large relative to target height or pixel resolutions differ significantly."
        )
    
    return base_core_width_crs, base_core_height_crs, base_buffer_crs, res_x, res_y


# Example usage
#device = 'mps'
max_workers = 10
batch_size = 20

#unreg_path = '/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_test.tif'
unreg_path = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/multispectral/front_country-batch_5-MS.tif'
reg_path = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/front_country_2024.tif'

(
    width_length,
    height_length,
    buffer,
    res_x,
    res_y
) = calculate_chip_crs_parameters(parent_raster_path=unreg_path, buffer_fraction=0.1)

scaling_factor = 1.0

actual_total_chip_width_px = ((width_length + 2 * buffer) / res_x ) * scaling_factor
actual_total_chip_height_px = ((height_length + 2 * buffer) / res_y ) * scaling_factor
width_length = width_length * scaling_factor
height_length = height_length * scaling_factor
buffer = buffer * scaling_factor

print(f"Processing with base size_scale 1:")
print(f"  Target total chip pixel dimensions: {actual_total_chip_width_px:.0f}w x {actual_total_chip_height_px:.0f}h")
print(f"  Call `register_raster_with_chips` with:")
print(f"    width_length: {width_length:.2f} (CRS units)")
print(f"    height_length: {height_length:.2f} (CRS units)")
print(f"    buffer: {buffer:.2f} (CRS units)")

out_tif = f'/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_corrected.tif'

register_raster_with_chips(
    unreg_path=unreg_path,
    reg_path=reg_path,
    output_path=out_tif,
    width_length=width_length,
    height_length=height_length,
    buffer=buffer,
    device=device,
    max_workers=max_workers,
    batch_size=batch_size
)
import os
import sys
import time
import tempfile
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling as RioResamplingEnum
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal
from affine import Affine

import cv2
import torch
import kornia as K
import kornia.feature as KF
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class Chip:
    array: np.ndarray
    profile: dict
    window: object # rasterio.windows.Window
    bounds: Tuple[float, float, float, float]

def resolve_height_chip(w: float, h: Optional[float]) -> float:
    """Helper to resolve height for chip processing if not provided."""
    return h if h is not None else 0.75 * w

def get_bbox_bounds_chip(pt: Point, width: float, height: float, buffer_w: float, buffer_h: float) -> Tuple[float, float, float, float]:
    """Calculates bounding box with buffers for chip extraction."""
    hw = width / 2 + buffer_w
    hh = height / 2 + buffer_h
    return (pt.x - hw, pt.y - hh, pt.x + hw, pt.y + hh)

def generate_grid_points_chip(raster_path: str, width: float, height: float) -> gpd.GeoDataFrame:
    """Generates a grid of points over a raster."""
    with rasterio.open(raster_path) as src:
        left, bottom, right, top = src.bounds
        crs = src.crs
    xs = np.arange(left + width / 2, right, width)
    ys = np.arange(bottom + height / 2, top, height)
    pts = [Point(x, y) for y in ys for x in xs]
    return gpd.GeoDataFrame(geometry=pts, crs=crs)

def load_chip_gdal(pt: Point, raster_path: str, width: float, height: float, buffer_w: float, buffer_h: float) -> Chip:
    """Loads a single chip from a raster."""
    bounds = get_bbox_bounds_chip(pt, width, height, buffer_w, buffer_h)
    with rasterio.open(raster_path) as src:
        win = rasterio.windows.from_bounds(*bounds, src.transform)
        arr = src.read(window=win)
        prof = src.profile.copy()
        prof.update({
            "height": win.height,
            "width": win.width,
            "transform": rasterio.windows.transform(win, src.transform)
        })
    return Chip(arr, prof, win, bounds)

def load_chips_gdal(pt: Point, unreg_raster_path: str, reg_raster_path: str,
                    width: float, height: float, buffer_w: float, buffer_h: float,
                    ref_resampling_method: RioResamplingEnum = RioResamplingEnum.bilinear
                    ) -> Tuple[Chip, Chip]:
    """Loads a pair of chips, one from unregistered and one from registered (warped to unregistered CRS)."""
    un_chip = load_chip_gdal(pt, unreg_raster_path, width, height, buffer_w, buffer_h)
    
    with rasterio.open(reg_raster_path) as src_reg, \
         WarpedVRT(src_reg, crs=un_chip.profile["crs"], resampling=ref_resampling_method) as vrt_reg:
        
        win_reg = rasterio.windows.from_bounds(*un_chip.bounds, vrt_reg.transform)
        arr_reg = vrt_reg.read(window=win_reg)
        prof_reg = vrt_reg.profile.copy()
        prof_reg.update({
            "height": win_reg.height,
            "width": win_reg.width,
            "transform": rasterio.windows.transform(win_reg, vrt_reg.transform)
        })
    reg_chip = Chip(arr_reg, prof_reg, win_reg, un_chip.bounds)
    return un_chip, reg_chip

def raster_to_tensor_chip(img_array: np.ndarray, bands: Optional[List[int]] = None) -> torch.Tensor:
    """Converts a raster numpy array to a PyTorch tensor for LoFTR."""
    bands_to_use = bands or list(range(img_array.shape[0]))
    # Ensure bands_to_use are valid for the array
    if not all(0 <= b < img_array.shape[0] for b in bands_to_use):
        raise ValueError(f"Invalid band selection for image with shape {img_array.shape}")
    if not bands_to_use: # Handle empty band list if it occurs
        raise ValueError("Band selection cannot be empty.")
        
    selected_bands_array = img_array[bands_to_use]
    return torch.from_numpy(selected_bands_array).unsqueeze(0)


def resize_image_chip(img_array_chw: np.ndarray, target_size_hw: Tuple[int, int] = (480, 640)
                      ) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Resizes an image (C, H, W) to target_size (H, W) and returns scales."""
    C, H, W = img_array_chw.shape
    TH, TW = target_size_hw
    
    if H == 0 or W == 0: # Cannot resize an empty image
        # Return an empty array of the target shape and scales of 1 to avoid division by zero
        # This signals downstream that the image was problematic.
        return np.zeros((C, TH, TW), dtype=img_array_chw.dtype), (1.0, 1.0)

    resized_chw = np.stack([
        cv2.resize(img_array_chw[i], (TW, TH), interpolation=cv2.INTER_LINEAR)
        for i in range(C)
    ])
    h_scale = H / TH if TH > 0 else 1.0
    w_scale = W / TW if TW > 0 else 1.0
    return resized_chw, (h_scale, w_scale)

def batch_get_loftr_matches_chip(
    chip_data_list: List[dict], # Each dict: {'id':, 'reg_img': np.ndarray, 'un_img': np.ndarray}
    device_str: str = 'cpu',
    batch_size_config: int = 16,
    loftr_pretrained_model: str = 'outdoor',
    min_matches_for_fm: int = 7,
    loftr_reproj_thresh_px_levels: List[float] = [0.5, 1.0, 2.0], # RANSAC reproj threshold in LoFTR's input image space
    ransac_confidence_levels: List[float] = [0.999, 0.95], # Confidence levels for RANSAC
    target_size_hw_loftr: Tuple[int, int] = (480, 640) # Target H, W for LoFTR preprocessing
) -> List[dict]:
    """
    Performs batched LoFTR matching on pairs of image chips.
    Initializes LoFTR model internally.
    Each dictionary in the output list will contain:
    'id': original chip id
    'mkpts_reg': keypoints in registered chip (original pixel space)
    'mkpts_un': keypoints in unregistered chip (original pixel space)
    'inliers': boolean mask for inlier keypoints
    'used_reproj_thresh': LoFTR reprojection threshold (resized space) used for successful RANSAC, or np.nan
    'used_ransac_confidence': RANSAC confidence level used for successful RANSAC, or np.nan
    """
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA specified but not available, falling back to CPU for LoFTR.")
        actual_device = torch.device('cpu')
    elif device_str == 'mps' and not torch.backends.mps.is_available():
        print("MPS specified but not available, falling back to CPU for LoFTR.")
        actual_device = torch.device('cpu')
    else:
        actual_device = torch.device(device_str)

    matcher = KF.LoFTR(pretrained=loftr_pretrained_model).to(actual_device).eval()

    N = len(chip_data_list)
    all_pairs_results: List[Optional[dict]] = [None] * N
    valid_pairs_for_loftr: List[Tuple[int, torch.Tensor, Tuple[float, float], torch.Tensor, Tuple[float, float]]] = []

    def prep_for_loftr(img_array_chw: np.ndarray):
        if img_array_chw.sum() == 0 or img_array_chw.shape[1] == 0 or img_array_chw.shape[2] == 0: # Check for empty spatial dims
            return None
        # Ensure at least 3 bands for RGB conversion, pad if necessary or take first band if single.
        # LoFTR typically expects grayscale, but preprocessing often starts from RGB.
        # The original script takes first 3 bands. We'll adapt this.
        c = img_array_chw.shape[0]
        if c == 0: return None # No channels
        
        bands_for_loftr_input = img_array_chw
        if c >= 3:
            bands_for_loftr_input = img_array_chw[[0,1,2], :, :] # Take first 3 for RGB assumption
        elif c == 2: # Pad to 3 channels with zeros
            padding = np.zeros((1, img_array_chw.shape[1], img_array_chw.shape[2]), dtype=img_array_chw.dtype)
            bands_for_loftr_input = np.concatenate((img_array_chw, padding), axis=0)
        elif c == 1: # Replicate to 3 channels
            bands_for_loftr_input = np.repeat(img_array_chw, 3, axis=0)

        resized_img, scales = resize_image_chip(bands_for_loftr_input, target_size_hw_loftr)
        
        # Normalize: LoFTR expects [0,1] float
        # Convert to tensor, then normalize
        tensor_img_u8 = torch.from_numpy(resized_img).float() # (C,H,W)
        if tensor_img_u8.max() > 1.0: # Assume 0-255 if max > 1
             tensor_img_norm = tensor_img_u8 / 255.0
        else: # Assume already in [0,1] or similar small range
             tensor_img_norm = tensor_img_u8
        tensor_img_norm = torch.clamp(tensor_img_norm, 0.0, 1.0)

        # RGB to Grayscale for LoFTR
        # K.color.rgb_to_grayscale expects (B,C,H,W) or (C,H,W)
        # If it's already (C,H,W) and C=3, it should work.
        grayscale_img_tensor = K.color.rgb_to_grayscale(tensor_img_norm) # (H,W) or (1,H,W)
        if grayscale_img_tensor.ndim == 2: # ensure (1,H,W)
            grayscale_img_tensor = grayscale_img_tensor.unsqueeze(0)
        
        return grayscale_img_tensor.unsqueeze(0), scales # Return (1,1,H,W), scales

    preprocessed_chips_info = []
    for i, chip_data in enumerate(tqdm(chip_data_list, desc="Preprocessing chips for LoFTR", unit="chip", leave=False)):
        original_id = chip_data['id']
        processed_reg = prep_for_loftr(chip_data['reg_img'])
        processed_unreg = prep_for_loftr(chip_data['un_img'])
        preprocessed_chips_info.append({'id': original_id, 
                                       'idx_in_list': i, 
                                       'processed_reg': processed_reg, 'processed_unreg': processed_unreg})

    for chip_info in preprocessed_chips_info:
        original_id, list_idx = chip_info['id'], chip_info['idx_in_list']
        prep_reg_k, prep_unreg_k = chip_info['processed_reg'], chip_info['processed_unreg']

        if prep_reg_k is not None and prep_unreg_k is not None:
            r_tensor, r_scales = prep_reg_k # r_tensor is (1,1,H,W)
            u_tensor, u_scales = prep_unreg_k # u_tensor is (1,1,H,W)
            valid_pairs_for_loftr.append((original_id, r_tensor, r_scales, u_tensor, u_scales))
        else:
            all_pairs_results[list_idx] = {'id': original_id, 'mkpts_reg': np.empty((0,2)),
                                           'mkpts_un': np.empty((0,2)), 'inliers': np.zeros(0,bool),
                                           'used_reproj_thresh': np.nan, 
                                           'used_ransac_confidence': np.nan}
            
    if not valid_pairs_for_loftr:
        tqdm.write("No valid pairs for LoFTR after preprocessing.")
        for i in range(N):
            if all_pairs_results[i] is None:
                 all_pairs_results[i] = {'id': chip_data_list[i]['id'], 'mkpts_reg': np.empty((0,2)),
                                           'mkpts_un': np.empty((0,2)), 'inliers': np.zeros(0,bool),
                                           'used_reproj_thresh': np.nan,
                                           'used_ransac_confidence': np.nan}
        return [res for res in all_pairs_results if res is not None]

    loftr_processed_results_map = {}
    for i in tqdm(range(0, len(valid_pairs_for_loftr), batch_size_config), desc="LoFTR Matching", unit="batch", leave=False):
        current_gpu_batch_data = valid_pairs_for_loftr[i : i + batch_size_config]
        
        # image0 is registered (reference), image1 is unregistered (target for warping)
        # Script had: image0 reg, image1 unreg. corr['keypoints0'] are for reg, corr['keypoints1'] for unreg.
        # mkpts_reg, mkpts_un. This seems consistent.
        img0_tensors_reg = torch.cat([item[1] for item in current_gpu_batch_data]).to(actual_device) # Registered tensors
        img1_tensors_un = torch.cat([item[3] for item in current_gpu_batch_data]).to(actual_device)  # Unregistered tensors

        with torch.inference_mode():
            corr = matcher({"image0": img0_tensors_reg, "image1": img1_tensors_un})

        all_kpts0_reg_batch_resized = corr['keypoints0'].cpu().numpy() 
        all_kpts1_un_batch_resized = corr['keypoints1'].cpu().numpy()
        batch_indices = corr['batch_indexes'].cpu().numpy()

        del corr, img0_tensors_reg, img1_tensors_un # Clear GPU memory

        for j_batch_idx in range(len(current_gpu_batch_data)):
            original_id = current_gpu_batch_data[j_batch_idx][0]
            reg_img_scales = current_gpu_batch_data[j_batch_idx][2] # (h_scale, w_scale)
            un_img_scales  = current_gpu_batch_data[j_batch_idx][4] # (h_scale, w_scale)

            mask_j = (batch_indices == j_batch_idx)
            mkpts_reg_resized = all_kpts0_reg_batch_resized[mask_j] # Points in LoFTR's resized registered image
            mkpts_un_resized = all_kpts1_un_batch_resized[mask_j]   # Points in LoFTR's resized unregistered image

            # Scale keypoints back to original chip dimensions
            mkpts_reg_orig = np.zeros_like(mkpts_reg_resized)
            if mkpts_reg_resized.shape[0] > 0:
                mkpts_reg_orig[:, 0] = mkpts_reg_resized[:, 0] * reg_img_scales[1] # W scale (x-coord)
                mkpts_reg_orig[:, 1] = mkpts_reg_resized[:, 1] * reg_img_scales[0] # H scale (y-coord)

            mkpts_un_orig = np.zeros_like(mkpts_un_resized)
            if mkpts_un_resized.shape[0] > 0:
                mkpts_un_orig[:, 0] = mkpts_un_resized[:, 0] * un_img_scales[1]  # W scale
                mkpts_un_orig[:, 1] = mkpts_un_resized[:, 1] * un_img_scales[0]  # H scale
            del mkpts_reg_resized, mkpts_un_resized

            inliers = np.zeros(mkpts_reg_orig.shape[0], dtype=bool)
            Fm, inliers_mask_cv = None, None
            current_used_reproj_thresh = np.nan
            current_used_ransac_confidence = np.nan

            if mkpts_reg_orig.shape[0] >= min_matches_for_fm:
                # Average scale factor for RANSAC threshold conversion
                # Using reg_img_scales as reference for threshold scaling to original pixels
                avg_reg_img_scale_factor_for_ransac = (reg_img_scales[0] + reg_img_scales[1]) / 2.0
                found_inliers_for_pair = False
                for loftr_thresh_px_input_space in loftr_reproj_thresh_px_levels:
                    ransac_thresh_orig_chip_px = loftr_thresh_px_input_space * avg_reg_img_scale_factor_for_ransac
                    for confidence_val in ransac_confidence_levels:
                        try:
                            # Points for findFundamentalMat: (mkpts_imgA, mkpts_imgB)
                            # Here, reg is imgA (image0), un is imgB (image1)
                            Fm, inliers_mask_cv = cv2.findFundamentalMat(
                                mkpts_reg_orig, mkpts_un_orig, 
                                method=cv2.USAC_MAGSAC, # Robust method
                                ransacReprojThreshold=ransac_thresh_orig_chip_px,
                                confidence=confidence_val,
                                maxIters=100000 # Ample iterations
                            )
                            if Fm is not None and Fm.shape == (3,3) and not np.all(Fm == 0):
                                if inliers_mask_cv is not None: # Ensure mask is not None
                                    inliers = inliers_mask_cv.ravel() > 0
                                    current_used_reproj_thresh = loftr_thresh_px_input_space
                                    current_used_ransac_confidence = confidence_val
                                    found_inliers_for_pair = True
                                    break 
                        except cv2.error:
                            Fm, inliers_mask_cv = None, None 
                            continue # Try next parameters
                        if found_inliers_for_pair: break
                    if found_inliers_for_pair: break
            
            loftr_processed_results_map[original_id] = {
                'id': original_id, 'mkpts_reg': mkpts_reg_orig, 
                'mkpts_un': mkpts_un_orig, 'inliers': inliers,
                'used_reproj_thresh': current_used_reproj_thresh,
                'used_ransac_confidence': current_used_ransac_confidence
            }
            del Fm, inliers_mask_cv

        if actual_device.type == 'mps':
            torch.mps.empty_cache()
        elif actual_device.type == 'cuda':
            torch.cuda.empty_cache()

    for i_orig_list_idx in range(N):
        if all_pairs_results[i_orig_list_idx] is None: # Not an initially invalid chip
            original_id_from_input = chip_data_list[i_orig_list_idx]['id']
            if original_id_from_input in loftr_processed_results_map:
                all_pairs_results[i_orig_list_idx] = loftr_processed_results_map[original_id_from_input]
            else:
                tqdm.write(f"Warning: Missing LoFTR result for original_id {original_id_from_input}. Using empty.")
                all_pairs_results[i_orig_list_idx] = {'id': original_id_from_input, 'mkpts_reg': np.empty((0,2)), 
                                                      'mkpts_un': np.empty((0,2)), 'inliers': np.zeros(0,bool),
                                                      'used_reproj_thresh': np.nan,
                                                      'used_ransac_confidence': np.nan}
    
    final_output = []
    for i_res, res_dict in enumerate(all_pairs_results):
        if res_dict is not None:
            final_output.append(res_dict)
        else: # Should have been filled, but as a fallback
            final_output.append({'id': chip_data_list[i_res]['id'], 'mkpts_reg': np.empty((0,2)),
                                 'mkpts_un': np.empty((0,2)), 'inliers': np.zeros(0,bool),
                                 'used_reproj_thresh': np.nan,
                                 'used_ransac_confidence': np.nan})
    return final_output


def generate_chip_gcps_gdal(un_chip: Chip, reg_chip: Chip,
                            mkpts_un_chip_coords: np.ndarray, # Keypoints in unreg chip's pixel space
                            mkpts_reg_chip_coords: np.ndarray, # Keypoints in reg chip's pixel space
                            inliers_mask: np.ndarray # Boolean mask for inlier points
                           ) -> List[gdal.GCP]:
    """Generates GDAL GCPs from matched keypoints between unregistered and registered chips."""
    if not inliers_mask.any():
        return []
    
    # Transform for registered chip: converts reg_chip pixel coords to its CRS coords
    # The GCP's target (x,y) are geographic coordinates from the registered chip
    # The GCP's source (pixelX, pixelY) are pixel coordinates from the unregistered chip
    reg_chip_transform = Affine.from_gdal(*reg_chip.profile["transform"].GetGeoTransform()) \
        if isinstance(reg_chip.profile["transform"], gdal.Dataset) \
        else reg_chip.profile["transform"] # Should be Affine or rasterio transform
    
    if not isinstance(reg_chip_transform, Affine): # Convert if rasterio.Affine
        try:
            reg_chip_transform = Affine(reg_chip_transform.a, reg_chip_transform.b, reg_chip_transform.c,
                                        reg_chip_transform.d, reg_chip_transform.e, reg_chip_transform.f)
        except AttributeError:
            raise ValueError("Registered chip profile 'transform' is not a recognized Affine type.")


    gcps = []
    for (ux_px, uy_px), (rx_px, ry_px), is_inlier in zip(mkpts_un_chip_coords, mkpts_reg_chip_coords, inliers_mask):
        if not is_inlier:
            continue
        
        # Convert registered chip pixel coordinates (rx_px, ry_px) to its geographic coordinates
        # These become the GCP's (GCPX, GCPY, GCPZ)
        gx_geo, gy_geo = reg_chip_transform * (float(rx_px), float(ry_px))
        
        # Unregistered chip pixel coordinates (ux_px, uy_px) are the (GCPPixel, GCPLine)
        gcps.append(gdal.GCP(gx_geo, gy_geo, 0, float(ux_px), float(uy_px))) # Assume Z=0 for GCPs
    return gcps

def warp_chip_gdal(un_chip: Chip, gcps: List[gdal.GCP], out_chip_path: str,
                   polynomial_order: int = 3, 
                   gdal_resample_algorithm: str = "cubic",
                   src_nodata_val = 0, # Nodata value in source chip (un_chip)
                   dst_nodata_val = 0  # Nodata value for output warped chip
                   ):
    """Warps a single unregistered chip using GDAL GCPs, saving to out_chip_path."""
    gdal.UseExceptions() # Ensure exceptions are on for this GDAL-heavy part

    # Create an in-memory GTiff dataset from the un_chip numpy array
    # un_chip.profile["transform"] should be an Affine object or GDAL-style tuple
    un_chip_affine_transform = un_chip.profile["transform"]
    if not isinstance(un_chip_affine_transform, tuple): # If it's Affine, convert
        un_chip_gdal_transform = un_chip_affine_transform.to_gdal()
    else:
        un_chip_gdal_transform = un_chip_affine_transform

    with MemoryFile() as mem_file:
        # Correctly get rasterio driver for in-memory dataset

        with mem_file.open(
            driver='GTiff', # Use 'GTiff' for broader compatibility
            height=un_chip.array.shape[1], # H from (C,H,W)
            width=un_chip.array.shape[2],  # W from (C,H,W)
            count=un_chip.array.shape[0],  # C from (C,H,W)
            dtype=str(un_chip.array.dtype),
            transform=un_chip_affine_transform, # rasterio transform
            crs=un_chip.profile["crs"],
            nodata=src_nodata_val 
        ) as mem_rasterio_dst:
            mem_rasterio_dst.write(un_chip.array)
        
        # mem_file.name is the path to the in-memory dataset that GDAL can open
        vrt_ds = None
        try:
            # Translate to VRT with GCPs
            # Output to "" means in-memory GDAL dataset
            translate_opts = gdal.TranslateOptions(format="VRT", GCPs=gcps)
            vrt_ds = gdal.Translate("", mem_file.name, options=translate_opts)
            if vrt_ds is None:
                raise RuntimeError(f"GDAL Translate to VRT failed for chip data from {mem_file.name}")

            warp_opts = gdal.WarpOptions(
                format="GTiff",
                outputBounds=un_chip.bounds, # Geographic bounds of the original un_chip
                width=int(un_chip.profile["width"]), # Pixel width of original un_chip
                height=int(un_chip.profile["height"]),# Pixel height of original un_chip
                polynomialOrder=polynomial_order,
                resampleAlg=gdal_resample_algorithm,
                dstSRS=str(un_chip.profile["crs"]), # Target SRS is same as un_chip's
                srcNodata=src_nodata_val,
                dstNodata=dst_nodata_val,
                # warpOptions=["SOURCE_EXTRA=5", "WRITE_FLUSH=YES"] # From script
                # Forcing output type might be needed if issues with dtype
                # outputType=gdal.GDT_Byte if un_chip.array.dtype == np.uint8 else gdal.GDT_UInt16 # Example
            )
            gdal.Warp(out_chip_path, vrt_ds, options=warp_opts)
        finally:
            vrt_ds = None # Dereference to allow GDAL to free VRT resources

def merge_warped_chips_gdal(
    warped_chips_info: List[Tuple[str, Tuple[float,float,float,float], Tuple[float,float]]], # (fpath, bounds, (pix_w, pix_h))
    final_output_path: str, 
    buffer_width_crs: float, # Buffer width in CRS units to crop from each side
    buffer_height_crs: float # Buffer height in CRS units to crop from each side
):
    """Merges cropped warped chips into a final raster."""
    gdal.UseExceptions()
    if not warped_chips_info:
        print("No warped chips to merge.")
        return

    with tempfile.TemporaryDirectory() as crop_temp_dir:
        cropped_chip_paths = []
        print("Cropping buffered edges from warped chips...")
        for i, (chip_fpath, chip_bounds_buffered, (pix_res_w, pix_res_h)) in \
            enumerate(tqdm(warped_chips_info, desc="Cropping Warped Chips", unit="chip", leave=False)):
            
            # chip_bounds_buffered = (xmin_buf, ymin_buf, xmax_buf, ymax_buf)
            # Core bounds after removing buffer
            core_bounds = [
                chip_bounds_buffered[0] + buffer_width_crs,  # new xmin
                chip_bounds_buffered[1] + buffer_height_crs,  # new ymin
                chip_bounds_buffered[2] - buffer_width_crs,  # new xmax
                chip_bounds_buffered[3] - buffer_height_crs   # new ymax
            ]
            
            # Ensure core bounds are valid (xmin < xmax, ymin < ymax)
            if core_bounds[0] >= core_bounds[2] or core_bounds[1] >= core_bounds[3]:
                tqdm.write(f"Skipping chip {i} due to invalid core bounds after buffer removal.")
                continue

            cropped_chip_out_path = os.path.join(crop_temp_dir, f"crop_{os.path.basename(chip_fpath)}")
            
            warp_to_crop_opts = gdal.WarpOptions(
                format="GTiff",
                outputBounds=core_bounds, # Target geographic extent is the core area
                xRes=pix_res_w, 
                yRes=pix_res_h,
                multithread=True,
                # warpOptions=["OPTIMIZE_SIZE=YES"], # From script
                dstNodata=0 # Assuming 0 is a safe nodata for intermediate cropped chips
            )
            gdal.Warp(cropped_chip_out_path, chip_fpath, options=warp_to_crop_opts)
            cropped_chip_paths.append(cropped_chip_out_path)

        if not cropped_chip_paths:
            print("No chips remaining after cropping. Merge aborted.")
            return

        print("Building VRT for merging cropped chips...")
        merged_vrt_path = os.path.join(crop_temp_dir, "merged_all_cropped.vrt")
        # Use the resolution from the first successfully cropped chip as reference for VRT
        # This assumes all chips should roughly align to this resolution after cropping.
        ref_pix_w_for_vrt, ref_pix_h_for_vrt = warped_chips_info[0][2] # pix_res_w, pix_res_h from first chip
        
        vrt_build_opts = gdal.BuildVRTOptions(
            xRes=ref_pix_w_for_vrt, 
            yRes=ref_pix_h_for_vrt,
            # Add other options like srcNodata, VRTNodata if needed
        )
        gdal.BuildVRT(merged_vrt_path, cropped_chip_paths, options=vrt_build_opts)

        print(f"Translating merged VRT to final COG: {final_output_path} ...")
        output_dir = os.path.dirname(final_output_path)
        if output_dir: # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
        
        translate_to_cog_opts = gdal.TranslateOptions(
            format="COG", # Use the COG driver
            creationOptions=[
                "COMPRESS=LZW", 
                "PREDICTOR=2", 
                "BIGTIFF=YES", 
                "OVERVIEWS=AUTO",
                "OVERVIEW_RESAMPLING=NEAREST", # Resampling for overviews
                "OVERVIEW_COMPRESS=LZW",       # Compression for overviews
                "OVERVIEW_PREDICTOR=2"         # Predictor for overview compression
            ],
            callback=gdal.TermProgress_nocb,
        )
        gdal.Translate(final_output_path, merged_vrt_path, options=translate_to_cog_opts)
        
        try: # Clean up VRT
            gdal.Unlink(merged_vrt_path) 
        except Exception: # nosemgrep
            pass # nosemgrep
            tqdm.write(f"Note: Could not unlink intermediate VRT {merged_vrt_path}")

    print("Chip merging complete.")


def calculate_chip_crs_parameters_gdal(
    parent_raster_path: str,
    target_total_chip_width_px: int, # e.g., TARGET_SIZE[1] from script (TOTAL width)
    target_total_chip_height_px: int, # e.g., TARGET_SIZE[0] from script (TOTAL height)
    buffer_fraction_of_core: float # e.g., BUFFER_FRAC from script
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculates chip core dimensions and buffer sizes in CRS units, plus pixel resolutions.
    Args:
        parent_raster_path: Path to the main raster (e.g., unregistered survey).
        target_total_chip_width_px: Desired total width of the chip (core + 2*buffer) in pixels.
        target_total_chip_height_px: Desired total height of the chip (core + 2*buffer) in pixels.
        buffer_fraction_of_core: Fraction of core dimension to use as buffer on each side.
    Returns:
        Tuple: (actual_core_w_crs, actual_core_h_crs,
                buffer_to_add_w_crs, buffer_to_add_h_crs,
                res_x, res_y)
    """
    with rasterio.open(parent_raster_path) as src:
        # abs() for pixel resolution as it can be negative for north-up images where origin is top-left
        res_x = abs(src.transform.a) # Pixel width in CRS units
        res_y = abs(src.transform.e) # Pixel height in CRS units

    if res_x == 0 or res_y == 0:
        raise ValueError("Raster resolution (x or y) is zero. Cannot calculate chip dimensions.")

    # Calculate core dimensions in pixels based on total and buffer fraction
    # T_w = C_w + 2 * (C_w * B_f) = C_w * (1 + 2 * B_f) => C_w = T_w / (1 + 2 * B_f)
    core_chip_width_px = target_total_chip_width_px / (1 + 2 * buffer_fraction_of_core)
    core_chip_height_px = target_total_chip_height_px / (1 + 2 * buffer_fraction_of_core)

    if core_chip_width_px <= 0 or core_chip_height_px <= 0:
        raise ValueError(
            f"Calculated non-positive core pixel dimensions: {core_chip_width_px:.2f}w x {core_chip_height_px:.2f}h. "
            f"Check target_total_chip dimensions ({target_total_chip_width_px}w, {target_total_chip_height_px}h) "
            f"and buffer_fraction ({buffer_fraction_of_core}). Ensure total is large enough relative to buffer."
        )

    actual_core_w_crs = core_chip_width_px * res_x
    actual_core_h_crs = core_chip_height_px * res_y

    if actual_core_w_crs <= 0:
        raise ValueError("Target chip width results in non-positive core CRS width. Check target_total_chip_width_px and raster resolution.")
    if actual_core_h_crs <= 0:
        raise ValueError("Target chip height results in non-positive core CRS height. Check target_total_chip_height_px and raster resolution.")

    buffer_to_add_w_crs = buffer_fraction_of_core * actual_core_w_crs
    buffer_to_add_h_crs = buffer_fraction_of_core * actual_core_h_crs

    return actual_core_w_crs, actual_core_h_crs, buffer_to_add_w_crs, buffer_to_add_h_crs, res_x, res_y

def register_survey_by_chips(
    unreg_survey_path: str,
    reg_reference_path: str,
    output_registered_survey_path: str,
    # NEW: Parameters for chip dimension calculation
    target_total_chip_width_px: int = 640,
    target_total_chip_height_px: int = 480,
    buffer_fraction_of_core: float = 0.1,
    # Processing parameters
    device_for_loftr: str = 'cpu',
    max_loader_workers: int = 4, # For ThreadPoolExecutor loading chips
    loftr_batch_size: int = 8,
    processing_chunk_size: int = 32, # Number of grid points per major processing cycle
    # LoFTR & GCP parameters with defaults from script
    loftr_model_name: str = 'outdoor',
    target_size_hw_for_loftr_preprocessing: Tuple[int, int] = (480, 640), # (H,W) for LoFTR input
    min_loftr_matches_for_fundamental_matrix: int = 7,
    loftr_reproj_threshold_px_levels_in_resized_space: List[float] = [0.5, 1.0, 2.0],
    ransac_confidence_levels_for_fm: List[float] = [0.999, 0.95],
    min_gcps_for_warp: int = 10, # Minimum number of valid GCPs required to attempt warp_chip
    # GDAL warp parameters
    gdal_polynomial_order: int = 3,
    gdal_resampling_algorithm: str = "cubic", # e.g., "cubic", "bilinear", "near"
    gdal_src_nodata: Optional[float]=0, # Nodata in source chips before warp
    gdal_dst_nodata: Optional[float]=0,  # Nodata for warped chips and final output
    output_stats_raster_path: Optional[str] = None, # Path for the 3-band stats raster (inliers, reproj_thresh, ransac_conf)
    debug_output_dir_for_warped_chips: Optional[str] = None # NEW: Directory to save individual warped chips before merge for debugging
):
    """
    Registers an unregistered survey raster to a registered reference raster using a chip-based
    approach with LoFTR for feature matching and GDAL for warping and merging.

    Args:
        unreg_survey_path: Path to the unregistered survey raster.
        reg_reference_path: Path to the registered reference raster.
        output_registered_survey_path: Path to save the final registered survey.
        target_total_chip_width_px: Desired total width of the chip (core + 2*buffer) in pixels.
        target_total_chip_height_px: Desired total height of the chip (core + 2*buffer) in pixels.
        buffer_fraction_of_core: Fraction of core dimension to use as buffer on each side.
        device_for_loftr: Device for LoFTR model ('cpu', 'cuda', 'mps').
        max_loader_workers: Maximum number of workers for loading chips concurrently.
        loftr_batch_size: Batch size for LoFTR inference.
        processing_chunk_size: Number of grid points to process in one major cycle.
        loftr_model_name: Pretrained LoFTR model name (e.g., 'outdoor').
        target_size_hw_for_loftr_preprocessing: Target (H,W) for LoFTR input image resizing.
        min_loftr_matches_for_fundamental_matrix: Minimum matches needed for Fundamental Matrix estimation.
        loftr_reproj_threshold_px_levels_in_resized_space: RANSAC reprojection thresholds for LoFTR (in resized space).
        ransac_confidence_levels_for_fm: Confidence levels for RANSAC Fundamental Matrix estimation.
        min_gcps_for_warp: Minimum number of GCPs to attempt warping a chip (Note: effective minimum is based on polynomial_order).
        gdal_polynomial_order: Polynomial order for GDAL warp.
        gdal_resampling_algorithm: GDAL resampling algorithm for warp.
        gdal_src_nodata: Nodata value in source chips.
        gdal_dst_nodata: Nodata value for warped chips and final output.
        output_stats_raster_path: Optional path to save a 3-band raster with LoFTR match statistics.
        debug_output_dir_for_warped_chips: If provided, individual warped chips (before cropping and merging)
                                           will be saved to this directory for inspection. The directory will be created if it doesn't exist.
                                           These chips will not be automatically deleted.
    """
    gdal.UseExceptions() # Ensure GDAL exceptions are enabled

    print(f"Starting chip-based survey registration:")
    print(f"  Unregistered: {unreg_survey_path}")
    print(f"  Reference: {reg_reference_path}")
    print(f"  Output: {output_registered_survey_path}")
    if output_stats_raster_path:
        print(f"  Stats Raster: {output_stats_raster_path}")

    # Handle debug directory for warped chips
    if debug_output_dir_for_warped_chips:
        os.makedirs(debug_output_dir_for_warped_chips, exist_ok=True)
        print(f"  Debug: Individual warped chips (pre-cropping) will be saved to: {debug_output_dir_for_warped_chips}")

    # Calculate chip CRS parameters internally
    print("Calculating chip CRS parameters...")
    try:
        core_chip_width_crs, core_chip_height_crs, \
        buffer_width_crs, buffer_height_crs, \
        pix_x_res, pix_y_res = calculate_chip_crs_parameters_gdal(
            parent_raster_path=unreg_survey_path,
            target_total_chip_width_px=target_total_chip_width_px,
            target_total_chip_height_px=target_total_chip_height_px,
            buffer_fraction_of_core=buffer_fraction_of_core
        )
        print("Computed chip dimensions (CRS units):")
        print(f"    Core size: {core_chip_width_crs:.2f}w x {core_chip_height_crs:.2f}h")
        print(f"    Buffer to add (each side): {buffer_width_crs:.2f}w x {buffer_height_crs:.2f}h")
        print(f"    Pixel resolution: {pix_x_res:.4f} (x), {pix_y_res:.4f} (y)\\n")
    except FileNotFoundError:
        print(f"Error: Unregistered raster not found at {unreg_survey_path}. Cannot calculate parameters.")
        print("Please ensure the unreg_survey_path is a valid path or URL accessible by rasterio.")
        return
    except Exception as e:
        print(f"Error calculating chip CRS parameters: {e}")
        return


    # Generate grid over the unregistered survey based on core chip dimensions
    grid_gdf = generate_grid_points_chip(unreg_survey_path, core_chip_width_crs, core_chip_height_crs)
    if grid_gdf.empty:
        print("No grid points generated. Check chip dimensions and survey extent. Aborting.")
        return

    # Determine actual chip grid dimensions from the generated points
    num_chip_cols = 0
    num_chip_rows = 0
    if not grid_gdf.empty:
        # Sort unique coordinates to ensure consistent ordering for num_chip_rows/cols
        # unique_x_coords = np.sort(grid_gdf.geometry.x.unique())
        # unique_y_coords = np.sort(grid_gdf.geometry.y.unique()) # Sorted bottom-to-top
        # num_chip_cols = len(unique_x_coords)
        # num_chip_rows = len(unique_y_coords)

        # A more direct way to get counts if grid is regular and fully populated by generate_grid_points_chip
        # This assumes generate_grid_points_chip fills out a complete grid based on its arange steps.
        with rasterio.open(unreg_survey_path) as src_main_raster: # Re-open to get bounds for arange logic
            left_bound, bottom_bound, right_bound, top_bound = src_main_raster.bounds
        
        # Estimate number of x points (columns)
        xs_for_count = np.arange(left_bound + core_chip_width_crs / 2, right_bound, core_chip_width_crs)
        num_chip_cols = len(xs_for_count)
        
        # Estimate number of y points (rows)
        ys_for_count = np.arange(bottom_bound + core_chip_height_crs / 2, top_bound, core_chip_height_crs)
        num_chip_rows = len(ys_for_count)

    if num_chip_cols == 0 or num_chip_rows == 0:
        print("Warning: Effective chip grid has 0 columns or 0 rows. Stats raster will not be generated.")
        output_stats_raster_path = None # Disable stats raster

    # Initialize stats raster if path is provided
    stats_grid = None
    stats_raster_transform = None
    stats_raster_crs = None
    # num_grid_cols_for_stats_raster = 0 # Replaced by num_chip_cols

    if output_stats_raster_path: # Check if still enabled
        try:
            with rasterio.open(unreg_survey_path) as src:
                unreg_bounds_for_origin = src.bounds # Used for left origin, top might be adjusted
                stats_raster_crs = src.crs
                
                # Use num_chip_rows and num_chip_cols for stats_grid dimensions
                if num_chip_cols <= 0 or num_chip_rows <= 0: # Should have been caught above
                    print("Warning: Calculated zero or negative dimensions for stats raster based on chip grid. Skipping its creation.")
                    output_stats_raster_path = None # Disable further processing
                else:
                    # Initialize 3-band float32 grid with np.nan as nodata
                    # Band 0: Inlier count
                    # Band 1: Used LoFTR reprojection threshold
                    # Band 2: Used RANSAC confidence
                    stats_grid = np.full((3, num_chip_rows, num_chip_cols),
                                           np.nan, dtype=np.float32)
                    
                    # Calculate the actual top Y-coordinate of the chip grid.
                    # ys_for_count contains y-centroids from bottom to top.
                    # The top-most y-centroid is ys_for_count[num_chip_rows - 1].
                    # The top edge of this top-most row of chips is centroid_y + half_cell_height.
                    actual_grid_top_y = ys_for_count[num_chip_rows - 1] + (core_chip_height_crs / 2.0)
                    
                    # Transform defines top-left of cell (0,0) of stats_grid and pixel size
                    stats_raster_transform = Affine(core_chip_width_crs, 0.0, unreg_bounds_for_origin.left,
                                                     0.0, -core_chip_height_crs, actual_grid_top_y) # Use actual_grid_top_y
                    print(f"Initialized stats raster: 3 bands, {num_chip_rows}h x {num_chip_cols}w, cell_size=({core_chip_width_crs:.2f}, {core_chip_height_crs:.2f}) CRS units, top_y_origin={actual_grid_top_y:.2f}")

        except Exception as e:
            print(f"Error initializing stats raster: {e}. It will not be created.")
            output_stats_raster_path = None 
            stats_grid = None

    # List to collect paths and metadata of successfully warped (and buffered) chips for merging
    warped_chips_for_merge_all_chunks: List[Tuple[str, Tuple[float,float,float,float], Tuple[float,float]]] = [] 

    # Statistics counters
    total_successful_warps = 0
    total_skipped_no_img_data = 0
    total_skipped_no_loftr_inliers = 0
    total_skipped_insufficient_gcps = 0
    total_errors_in_processing = 0

    output_dir = os.path.dirname(output_registered_survey_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as main_processing_tmpdir:
        num_chunks = (len(grid_gdf) + processing_chunk_size - 1) // processing_chunk_size

        # Determine the actual path to save individual warped chips
        if debug_output_dir_for_warped_chips:
            path_to_save_individual_warped_chips = debug_output_dir_for_warped_chips
        else:
            path_to_save_individual_warped_chips = main_processing_tmpdir

        for chunk_idx in tqdm(range(num_chunks), desc="Total Progress (Chunks)", unit="chunk", position=0):
            chunk_start_idx = chunk_idx * processing_chunk_size
            chunk_end_idx = min((chunk_idx + 1) * processing_chunk_size, len(grid_gdf))
            current_grid_points_chunk_df = grid_gdf.iloc[chunk_start_idx:chunk_end_idx]

            if current_grid_points_chunk_df.empty:
                continue
            
            tqdm.write(f"--- Processing Chunk {chunk_idx + 1}/{num_chunks} (Grid Indices {chunk_start_idx} to {chunk_end_idx-1}) ---")

            # 1. Load chip pairs for the current chunk
            loaded_chips_data_current_chunk: List[dict] = [] # Store {'id': original_idx, 'un_chip': Chip, 'reg_chip': Chip}
            chip_load_futures = {}
            with ThreadPoolExecutor(max_workers=max_loader_workers) as executor:
                for original_grid_idx, grid_row in current_grid_points_chunk_df.iterrows():
                    future = executor.submit(load_chips_gdal, grid_row.geometry,
                                             unreg_survey_path, reg_reference_path,
                                             core_chip_width_crs, core_chip_height_crs,
                                             buffer_width_crs, buffer_height_crs)
                    chip_load_futures[future] = original_grid_idx
                
                progress_loading = tqdm(as_completed(chip_load_futures), total=len(chip_load_futures), 
                                        desc=f"  Loading chips (Chunk {chunk_idx+1})", unit="chip", leave=False, position=1)
                for fut_load in progress_loading:
                    original_idx = chip_load_futures[fut_load]
                    try:
                        un_chip_obj, reg_chip_obj = fut_load.result()
                        loaded_chips_data_current_chunk.append({'id': original_idx, 
                                                               'un_chip': un_chip_obj, 
                                                               'reg_chip': reg_chip_obj})
                    except Exception as e:
                        tqdm.write(f"Load failure for chip id {original_idx} in chunk {chunk_idx+1}: {e}")
                        total_errors_in_processing +=1
            
            if not loaded_chips_data_current_chunk:
                tqdm.write(f"No chips successfully loaded for chunk {chunk_idx+1}. Skipping.")
                continue
            
            # Sort by ID to maintain order if needed, though map lookup is used later
            loaded_chips_data_current_chunk.sort(key=lambda x: x['id'])

            # 2. Batch LoFTR matching for the current chunk's loaded chips
            loftr_input_for_chunk = [{'id': lcd['id'], 
                                      'un_img': lcd['un_chip'].array, # Unregistered is 'target' for warping
                                      'reg_img': lcd['reg_chip'].array  # Registered is 'reference' space
                                     } for lcd in loaded_chips_data_current_chunk]
            
            loftr_matches_results_chunk = batch_get_loftr_matches_chip(
                loftr_input_for_chunk,
                device_str=device_for_loftr,
                batch_size_config=loftr_batch_size,
                loftr_pretrained_model=loftr_model_name,
                min_matches_for_fm=min_loftr_matches_for_fundamental_matrix,
                loftr_reproj_thresh_px_levels=loftr_reproj_threshold_px_levels_in_resized_space,
                ransac_confidence_levels=ransac_confidence_levels_for_fm,
                target_size_hw_loftr=target_size_hw_for_loftr_preprocessing
            )
            # Create a map for quick lookup of match results by original_id
            matches_map_for_chunk = {match_res['id']: match_res for match_res in loftr_matches_results_chunk}

            # Populate stats grid for this chunk if enabled
            if output_stats_raster_path and stats_grid is not None and num_chip_cols > 0 and num_chip_rows > 0:
                for loaded_chip_info_for_stats in loaded_chips_data_current_chunk:
                    original_idx = loaded_chip_info_for_stats['id'] # Index in the full grid_gdf
                    match_data = matches_map_for_chunk.get(original_idx)
                    
                    # original_idx corresponds to the chip's position in the grid generated by
                    # iterating y from bottom-to-top, then x from left-to-right.
                    # num_chip_cols is the number of x-points in each row of chips.
                    
                    chip_col_idx_from_left = original_idx % num_chip_cols
                    chip_row_idx_from_bottom = original_idx // num_chip_cols

                    # The stats_grid numpy array is indexed (bands, row_from_top, col_from_left).
                    # Convert chip_row_idx_from_bottom to stats_grid_row_idx_from_top.
                    stats_grid_row_idx = (num_chip_rows - 1) - chip_row_idx_from_bottom
                    stats_grid_col_idx = chip_col_idx_from_left


                    if 0 <= stats_grid_row_idx < num_chip_rows and \
                       0 <= stats_grid_col_idx < num_chip_cols:
                        if match_data:
                            num_inliers = float(match_data['inliers'].sum())
                            used_thresh = match_data.get('used_reproj_thresh', np.nan)
                            used_conf = match_data.get('used_ransac_confidence', np.nan)

                            stats_grid[0, stats_grid_row_idx, stats_grid_col_idx] = num_inliers
                            stats_grid[1, stats_grid_row_idx, stats_grid_col_idx] = used_thresh
                            stats_grid[2, stats_grid_row_idx, stats_grid_col_idx] = used_conf
                        # Else: cells remain np.nan (initialized value) if no match_data or chip failed earlier
                    else:
                        # This warning indicates an issue with indexing logic or num_chip_rows/cols calculation relative to original_idx range.
                        tqdm.write(f"Warning: Calculated out-of-bounds index for stats grid. Chip ID: {original_idx}, "
                                   f"Target stats_grid_idx: ({stats_grid_row_idx}, {stats_grid_col_idx}). "
                                   f"Stats grid dims: ({num_chip_rows}h, {num_chip_cols}w). "
                                   f"Chip row_from_bottom: {chip_row_idx_from_bottom}, chip_col_from_left: {chip_col_idx_from_left}.")


            # 3. Generate GCPs and Warp chips for the current chunk
            chunk_successful_warps = 0
            chunk_skipped_no_data = 0
            chunk_skipped_no_inliers = 0
            chunk_skipped_low_gcps = 0
            chunk_errors = 0

            # Minimum GCPs required by GDAL for different polynomial orders
            MIN_GCPS_TABLE = {1: 3, 2: 6, 3: 10}

            progress_warping = tqdm(loaded_chips_data_current_chunk, desc=f"  Warping chips (Chunk {chunk_idx+1})", unit="chip", leave=False, position=1)
            for loaded_chip_info in progress_warping:
                original_idx = loaded_chip_info['id']
                un_chip_to_warp = loaded_chip_info['un_chip']
                reg_chip_reference = loaded_chip_info['reg_chip']
                
                match_data_for_chip = matches_map_for_chunk.get(original_idx)

                try:
                    if match_data_for_chip is None:
                        # This case should ideally be handled by batch_get_loftr_matches_chip returning empty arrays for this id
                        tqdm.write(f"Chip {original_idx}: Skipped (no LoFTR match data found).")
                        chunk_errors +=1
                        continue

                    # Check for empty image arrays before proceeding
                    if un_chip_to_warp.array.sum() == 0 or reg_chip_reference.array.sum() == 0 or \
                       un_chip_to_warp.array.size == 0 or reg_chip_reference.array.size == 0:
                        chunk_skipped_no_data += 1
                        continue
                    
                    # Keypoints from LoFTR: mkpts_reg are in reg_chip_reference's pixel space,
                    # mkpts_un are in un_chip_to_warp's pixel space.
                    mkpts_in_reg_chip = match_data_for_chip['mkpts_reg'] 
                    mkpts_in_un_chip = match_data_for_chip['mkpts_un']
                    inliers_from_loftr = match_data_for_chip['inliers']

                    num_inliers = int(inliers_from_loftr.sum())
                    if num_inliers == 0:
                        chunk_skipped_no_inliers += 1
                        continue
                    
                    # Generate GCPs: un_chip (source pixels), reg_chip (target geo coords)
                    gcp_list_for_warp = generate_chip_gcps_gdal(un_chip_to_warp, reg_chip_reference,
                                                                mkpts_in_un_chip, mkpts_in_reg_chip,
                                                                inliers_from_loftr)
                    
                    num_gcps = len(gcp_list_for_warp)
                    poly_order_for_this_chip = None
                    chosen_order_found = False

                    # Try polynomial orders from the configured gdal_polynomial_order down to 1
                    # to find the highest one that satisfies its GCP requirement.
                    # Assumes gdal_polynomial_order is a sensible positive integer (e.g., 1, 2, or 3)
                    for order_to_try in range(gdal_polynomial_order, 0, -1): 
                        min_req_for_order = MIN_GCPS_TABLE.get(order_to_try)
                        if min_req_for_order is not None: # Ensure the order is defined in our table
                            if num_gcps >= min_req_for_order:
                                poly_order_for_this_chip = order_to_try
                                chosen_order_found = True
                                break # Found the highest possible order
                    
                    if not chosen_order_found:
                        # Not enough GCPs even for the lowest considered/supported order (typically 1)
                        min_gcp_for_lowest_supported_order = MIN_GCPS_TABLE.get(1, 3) # Default to 3 for order 1 if not in table
                        tqdm.write(f"Chip {original_idx}: Skipped. Insufficient GCPs ({num_gcps}) for any supported polynomial order (e.g., order 1 needs {min_gcp_for_lowest_supported_order}).")
                        chunk_skipped_low_gcps += 1
                        continue
                    else:
                        # An order was found. Log if it's different from the initially configured/max one.
                        if poly_order_for_this_chip != gdal_polynomial_order:
                            tqdm.write(f"Chip {original_idx}: Using polynomial order {poly_order_for_this_chip} ({num_gcps} GCPs). Configured/max order was {gdal_polynomial_order}.")
                        # else: Using the configured/max order as it met requirements, no special message needed.

                    # Define output path for this warped chip (in the main temporary directory)
                    # Base name from original grid index to ensure uniqueness
                    warped_chip_filename = f"warped_chip_{original_idx}.tif"
                    output_path_for_this_warped_chip = os.path.join(path_to_save_individual_warped_chips, warped_chip_filename)
                    
                    warp_chip_gdal(un_chip_to_warp, gcp_list_for_warp, output_path_for_this_warped_chip,
                                   polynomial_order=poly_order_for_this_chip, # Use the dynamically determined order
                                   gdal_resample_algorithm=gdal_resampling_algorithm,
                                   src_nodata_val=gdal_src_nodata,
                                   dst_nodata_val=gdal_dst_nodata
                                   )
                    
                    chunk_successful_warps += 1
                    
                    # Get pixel resolution of the un_chip (source of warp) for merge step
                    # This assumes the warped chip maintains roughly this resolution in its core area
                    un_chip_transform = un_chip_to_warp.profile["transform"]
                    pix_width_un = abs(un_chip_transform.a)
                    pix_height_un = abs(un_chip_transform.e)
                    
                    # Store path, original (buffered) bounds of un_chip, and its pixel resolution
                    warped_chips_for_merge_all_chunks.append(
                        (output_path_for_this_warped_chip, un_chip_to_warp.bounds, (pix_width_un, pix_height_un))
                    )

                except Exception as e_warp:
                    tqdm.write(f"Chip {original_idx}: ERROR during warping in chunk {chunk_idx+1} - {e_warp}")
                    chunk_errors += 1
            
            # Aggregate chunk stats to totals
            total_successful_warps += chunk_successful_warps
            total_skipped_no_img_data += chunk_skipped_no_data
            total_skipped_no_loftr_inliers += chunk_skipped_no_inliers
            total_skipped_insufficient_gcps += chunk_skipped_low_gcps
            total_errors_in_processing += chunk_errors
            
            tqdm.write(f"  Chunk {chunk_idx+1} Summary: Warped: {chunk_successful_warps}, "
                  f"Skipped (NoData: {chunk_skipped_no_data}, NoInliers: {chunk_skipped_no_inliers}, LowGCPs: {chunk_skipped_low_gcps}), "
                  f"Errors: {chunk_errors}")

            # Clean up large chunk-specific data to free memory
            del loaded_chips_data_current_chunk, loftr_input_for_chunk, loftr_matches_results_chunk, matches_map_for_chunk
            if 'gc' in sys.modules: # If gc was imported (it is by original script)
                 import gc
                 gc.collect()
        
        # --- End of all chunk processing ---

        print("\nOverall Warping Summary:")
        print(f"  Successfully warped chips (pre-merge): {total_successful_warps}")
        print(f"  Skipped (no image data): {total_skipped_no_img_data}")
        print(f"  Skipped (no LoFTR inliers): {total_skipped_no_loftr_inliers}")
        print(f"  Skipped (insufficient GCPs): {total_skipped_insufficient_gcps}")
        print(f"  Errors during chip processing: {total_errors_in_processing}")

        # Merge all successfully warped (and buffered) chips
        if warped_chips_for_merge_all_chunks:
            print("\nStarting merge process for warped chips...")
            merge_warped_chips_gdal(warped_chips_for_merge_all_chunks,
                                    output_registered_survey_path,
                                    buffer_width_crs, # Buffer to remove (same as added for processing)
                                    buffer_height_crs # Buffer to remove
                                   )
            print(f"Registration process complete. Output at: {output_registered_survey_path}")
        else:
            print("No chips were successfully warped. Final output raster will not be created.")
            pass # Or raise an error / return a status

    # Save the stats raster if it was generated
    if output_stats_raster_path and stats_grid is not None and stats_raster_transform and stats_raster_crs:
        print(f"\nSaving stats raster to: {output_stats_raster_path}")
        try:
            with rasterio.open(
                output_stats_raster_path,
                'w',
                driver='GTiff',
                height=stats_grid.shape[1],
                width=stats_grid.shape[2],
                count=3, # 3 bands
                dtype=stats_grid.dtype, # np.float32
                crs=stats_raster_crs,
                transform=stats_raster_transform,
                nodata=np.nan, # Nodata value for all bands
                compress='lzw',
                predictor=2
            ) as dst:
                dst.write(stats_grid) # Writes all 3 bands
                dst.set_band_description(1, "Inlier Count")
                dst.set_band_description(2, "LoFTR Reprojection Threshold (px in resized space)")
                dst.set_band_description(3, "RANSAC Confidence")
            print("Stats raster saved successfully.")
        except Exception as e:
            print(f"Error saving stats raster: {e}")
    elif output_stats_raster_path: # Path was given, but something went wrong
        print(f"Stats raster was requested but not generated/saved due to earlier issues.")


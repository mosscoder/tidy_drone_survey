#!/usr/bin/env python3
"""
Chip-based raster registration pipeline using LoFTR + GDAL warp.

Verbose tqdm-based progress reporting; no logging module or plain prints.
"""

import os
import time
import tempfile
from dataclasses import dataclass
from typing import Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import kornia as K
import kornia.feature as KF
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.io import MemoryFile
from shapely.geometry import Point, Polygon
from concurrent.futures import ThreadPoolExecutor, as_completed
from osgeo import gdal
from tqdm import tqdm
import sys

gdal.UseExceptions() # Enable GDAL exceptions

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEVICE      = 'mps'
MATCHER     = KF.LoFTR(pretrained='outdoor').to(DEVICE).eval()
MAX_WORKERS = 10
BATCH_SIZE  = 20
POLY_ORDER  = 3
RESAMPLE_ALG= "cubic"
TARGET_SIZE = (480, 640)       # (H, W)
BUFFER_FRAC = 0.1

# Constants for LoFTR RANSAC
LOFTR_REPROJ_THRESH_LEVELS_PX = [1.0, 0.5]  # RANSAC reprojection threshold in LoFTR's input image space (pixels)
CONFIDENCE_LEVELS = [0.999, 0.95]              # Confidence levels for RANSAC
MIN_LOFTR_MATCHES_FOR_FUNDAMENTAL_MATRIX = 7   # Minimum number of matches required for cv2.findFundamentalMat

UNREG_URL = '/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_test.tif'
# UNREG_URL   = "https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/multispectral/front_country-batch_5-MS.tif"
REG_URL     = "https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/front_country_2024.tif"
OUTPUT_PATH = "/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_corrected.tif"

@dataclass
class Chip:
    array: np.ndarray
    profile: dict
    window: object
    bounds: Tuple[float, float, float, float]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged logic)
# ──────────────────────────────────────────────────────────────────────────────

def resolve_height(w: float, h: Optional[float]) -> float:
    return h if h is not None else 0.75 * w

def get_bbox_bounds(pt: Point, width: float, height: float, buffer_w: float, buffer_h: float):
    hw = width/2 + buffer_w
    hh = height/2 + buffer_h
    return (pt.x - hw, pt.y - hh, pt.x + hw, pt.y + hh)

def generate_grid_points(raster_path: str, width: float, height: float) -> gpd.GeoDataFrame:
    with rasterio.open(raster_path) as src:
        left, bottom, right, top = src.bounds
        crs = src.crs
    xs = np.arange(left + width/2, right, width)
    ys = np.arange(bottom + height/2, top, height)
    pts = [Point(x, y) for y in ys for x in xs]
    return gpd.GeoDataFrame(geometry=pts, crs=crs)

def load_chip(pt: Point, raster_path: str, width: float, height: float, buffer_w: float, buffer_h: float) -> Chip:
    bounds = get_bbox_bounds(pt, width, height, buffer_w, buffer_h)
    with rasterio.open(raster_path) as src:
        win = from_bounds(*bounds, src.transform)
        arr = src.read(window=win)
        prof = src.profile.copy()
        prof.update({
            "height": win.height,
            "width": win.width,
            "transform": window_transform(win, src.transform)
        })
    return Chip(arr, prof, win, bounds)

def load_chips(pt: Point, unreg: str, reg: str, width: float, height: float, buffer_w: float, buffer_h: float):
    un_chip = load_chip(pt, unreg, width, height, buffer_w, buffer_h)
    with rasterio.open(reg) as src, \
         WarpedVRT(src, crs=un_chip.profile["crs"], resampling=Resampling.bilinear) as vrt:
        win = from_bounds(*un_chip.bounds, vrt.transform)
        arr = vrt.read(window=win)
        prof = vrt.profile.copy()
        prof.update({
            "height": win.height,
            "width": win.width,
            "transform": window_transform(win, vrt.transform)
        })
    reg_chip = Chip(arr, prof, win, un_chip.bounds)
    return un_chip, reg_chip

def raster_to_tensor(img: np.ndarray, bands: Optional[List[int]] = None):
    bands = bands or list(range(img.shape[0]))
    return torch.from_numpy(img[bands]).unsqueeze(0)

def resize_image(img: np.ndarray, target_size=TARGET_SIZE):
    C, H, W = img.shape
    TH, TW = target_size
    resized = np.stack([
        cv2.resize(img[i], (TW, TH), interpolation=cv2.INTER_LINEAR)
        for i in range(C)
    ])
    h_scale, w_scale = H/TH, W/TW
    return resized, (h_scale, w_scale)

def batch_get_loftr_matches(chip_data_list: List[dict], device=DEVICE, batch_size=BATCH_SIZE) -> List[dict]:
    N = len(chip_data_list)
    # Initialize results with a structure that includes the id
    all_pairs_results: List[Optional[dict]] = [None] * N

    # Stores (original_id, registered_tensor, registered_scales, unregistered_tensor, unregistered_scales)
    valid_pairs_for_loftr: List[Tuple[int, torch.Tensor, Tuple[float, float], torch.Tensor, Tuple[float, float]]] = []

    def prep(img_array):
        if img_array.sum() == 0:
            return None
        resized_img, scales = resize_image(img_array)
        # Assuming RGB, take first 3 bands for LoFTR
        tensor_img = raster_to_tensor(resized_img, [0, 1, 2]).float().div(255.0)
        grayscale_img = K.color.rgb_to_grayscale(tensor_img)
        return grayscale_img, scales

    # Preprocess all chips
    # Store tuples of (original_id, processed_reg_data, processed_unreg_data)
    preprocessed_chips_info = []
    for i, chip_data in enumerate(tqdm(chip_data_list, desc="Preprocessing chips", unit="chip")):
        original_id = chip_data['id']
        reg_img_array = chip_data['reg_img']
        un_img_array = chip_data['un_img']
        
        processed_reg = prep(reg_img_array)
        processed_unreg = prep(un_img_array)
        preprocessed_chips_info.append({'id': original_id, 'idx_in_list': i, 'processed_reg': processed_reg, 'processed_unreg': processed_unreg})

    for chip_info in preprocessed_chips_info:
        original_id = chip_info['id']
        list_idx = chip_info['idx_in_list'] # The original index in chip_data_list / all_pairs_results
        prep_reg_k = chip_info['processed_reg']
        prep_unreg_k = chip_info['processed_unreg']

        if prep_reg_k is not None and prep_unreg_k is not None:
            r_tensor, r_scales = prep_reg_k
            u_tensor, u_scales = prep_unreg_k
            # Store original_id for later association
            valid_pairs_for_loftr.append((original_id, r_tensor, r_scales, u_tensor, u_scales))
        else:
            # For chips that can't be processed, store empty results associated with their original id
            all_pairs_results[list_idx] = {'id': original_id, 'mkpts_reg': np.empty((0,2)), 'mkpts_un': np.empty((0,2)), 'inliers': np.zeros(0,bool)}
            
    if not valid_pairs_for_loftr:
        tqdm.write("No valid pairs for LoFTR after preprocessing.")
        # Fill any remaining None slots if all were invalid from the start
        for i in range(N):
            if all_pairs_results[i] is None: # Should have been pre-filled if invalid
                 original_id = chip_data_list[i]['id'] # Get id from original list
                 all_pairs_results[i] = {'id': original_id, 'mkpts_reg': np.empty((0,2)), 'mkpts_un': np.empty((0,2)), 'inliers': np.zeros(0,bool)}
        return [res for res in all_pairs_results if res is not None] # Ensure no None is returned

    # This list will store dicts: {'id': original_id, 'mkpts_reg': ..., 'mkpts_un': ..., 'inliers': ...}
    loftr_processed_results_map = {}


    for i in tqdm(range(0, len(valid_pairs_for_loftr), batch_size), desc="LoFTR Matching", unit="batch"):
        current_gpu_batch_data = valid_pairs_for_loftr[i : i + batch_size]
        
        # image0 is registered, image1 is unregistered
        img0_tensors_reg = torch.cat([item[1] for item in current_gpu_batch_data]).to(device) # Registered tensors
        img1_tensors_un = torch.cat([item[3] for item in current_gpu_batch_data]).to(device)  # Unregistered tensors

        with torch.inference_mode():
            corr = MATCHER({"image0": img0_tensors_reg, "image1": img1_tensors_un})

        # keypoints0 are from registered, keypoints1 are from unregistered
        all_kpts0_reg_batch_resized = corr['keypoints0'].cpu().numpy() 
        all_kpts1_un_batch_resized = corr['keypoints1'].cpu().numpy()
        batch_indices = corr['batch_indexes'].cpu().numpy()

        for j in range(len(current_gpu_batch_data)):
            original_id = current_gpu_batch_data[j][0] # Get the original_id
            reg_img_scales = current_gpu_batch_data[j][2]
            un_img_scales  = current_gpu_batch_data[j][4]

            mask_j = (batch_indices == j)
            mkpts_reg_resized = all_kpts0_reg_batch_resized[mask_j]
            mkpts_un_resized = all_kpts1_un_batch_resized[mask_j]

            mkpts_reg_orig = np.zeros_like(mkpts_reg_resized)
            if mkpts_reg_resized.shape[0] > 0:
                mkpts_reg_orig[:, 0] = mkpts_reg_resized[:, 0] * reg_img_scales[1] # W scale
                mkpts_reg_orig[:, 1] = mkpts_reg_resized[:, 1] * reg_img_scales[0] # H scale

            mkpts_un_orig = np.zeros_like(mkpts_un_resized)
            if mkpts_un_resized.shape[0] > 0:
                mkpts_un_orig[:, 0] = mkpts_un_resized[:, 0] * un_img_scales[1]  # W scale
                mkpts_un_orig[:, 1] = mkpts_un_resized[:, 1] * un_img_scales[0]  # H scale
            
            inliers = np.zeros(mkpts_reg_orig.shape[0], dtype=bool)
            if mkpts_reg_orig.shape[0] >= MIN_LOFTR_MATCHES_FOR_FUNDAMENTAL_MATRIX:
                # Using registered image scales for RANSAC threshold, as it's the 'query' image in findFundamentalMat
                avg_reg_img_scale_factor = (reg_img_scales[0] + reg_img_scales[1]) / 2.0
                found_inliers_for_pair = False
                for loftr_thresh_px in LOFTR_REPROJ_THRESH_LEVELS_PX:
                    # RANSAC threshold is applied to the 'query' keypoints, which are mkpts_reg_orig
                    ransac_thresh_orig_px = loftr_thresh_px # * avg_reg_img_scale_factor (LoFTR threshold is in its input image space)

                    for confidence_val in CONFIDENCE_LEVELS:
                        try:
                            # cv2.findFundamentalMat(query, train, ...)
                            # Here, query=mkpts_reg_orig, train=mkpts_un_orig based on LoFTR's image0/image1
                            Fm, inliers_mask = cv2.findFundamentalMat(
                                mkpts_reg_orig, mkpts_un_orig, # Order matters: query (reg), train (unreg)
                                method=cv2.USAC_MAGSAC,
                                ransacReprojThreshold=ransac_thresh_orig_px,
                                confidence=confidence_val,
                                maxIters=100000
                            )
                            if Fm is not None and Fm.shape[0] == 3 and not np.all(Fm == 0):
                                inliers = inliers_mask.ravel() > 0
                                found_inliers_for_pair = True
                                break
                        except cv2.error:
                            continue
                    if found_inliers_for_pair:
                        break
            
            loftr_processed_results_map[original_id] = {
                'id': original_id,
                'mkpts_reg': mkpts_reg_orig, 
                'mkpts_un': mkpts_un_orig, 
                'inliers': inliers
            }

    # Populate all_pairs_results using the original list order and the map
    for i in range(N):
        if all_pairs_results[i] is None: # If it wasn't an invalid chip initially
            original_id_from_input = chip_data_list[i]['id']
            if original_id_from_input in loftr_processed_results_map:
                all_pairs_results[i] = loftr_processed_results_map[original_id_from_input]
            else:
                # This case should ideally not happen if all valid_pairs_for_loftr were processed
                tqdm.write(f"Warning: Missing LoFTR result for original_id {original_id_from_input}. Using empty result.")
                all_pairs_results[i] = {'id': original_id_from_input, 'mkpts_reg': np.empty((0,2)), 'mkpts_un': np.empty((0,2)), 'inliers': np.zeros(0,bool)}
    
    # Ensure no None values are in the final list
    final_output = [res if res is not None else {'id': chip_data_list[i]['id'], 'mkpts_reg': np.empty((0,2)), 'mkpts_un': np.empty((0,2)), 'inliers': np.zeros(0,bool)} for i, res in enumerate(all_pairs_results)]
    return final_output

def generate_chip_gcps(un_chip: Chip, reg_chip: Chip,
                       mkpts_un, mkpts_reg, inliers):
    if not inliers.any():
        return []
    tf = reg_chip.profile["transform"]
    gcps = []
    for (ux, uy), (rx, ry), ok in zip(mkpts_un, mkpts_reg, inliers):
        if not ok: continue
        gx, gy = tf * (float(rx), float(ry))
        gcps.append(gdal.GCP(gx, gy, 0, float(ux), float(uy)))
    return gcps

def warp_chip(un_chip: Chip, gcps: List[gdal.GCP], out_path: str):
    with MemoryFile() as mem:
        with mem.open(driver="GTiff",
                      height=un_chip.array.shape[1],
                      width=un_chip.array.shape[2],
                      count=un_chip.array.shape[0],
                      dtype=un_chip.array.dtype,
                      transform=un_chip.profile["transform"],
                      crs=un_chip.profile["crs"],
                      nodata=0) as dst:
            dst.write(un_chip.array)
        vrt = gdal.Translate("", mem.name,
                              options=gdal.TranslateOptions(format="VRT", GCPs=gcps))
        warp_opts = gdal.WarpOptions(
            format="GTiff",
            outputBounds=un_chip.bounds,
            width=int(un_chip.profile["width"]),
            height=int(un_chip.profile["height"]),
            polynomialOrder=POLY_ORDER,
            resampleAlg=RESAMPLE_ALG,
            dstSRS=un_chip.profile["crs"].to_wkt(),
            srcNodata=0,
            dstNodata=0,
            warpOptions=["SOURCE_EXTRA=5", "WRITE_FLUSH=YES"]
        )
        gdal.Warp(out_path, vrt, options=warp_opts)

def merge_warped_chips(warped, output_path: str, buffer_w: float, buffer_h: float):
    if not warped:
        print("⚠️  No warped chips to merge.")
        return

    with tempfile.TemporaryDirectory() as crop_dir:
        cropped = []
        print("Cropping chips...")
        for i, (fpath, bounds, (pw, ph)) in enumerate(warped):
            xmin, ymin, xmax, ymax = bounds
            core = [xmin + buffer_w, ymin + buffer_h, xmax - buffer_w, ymax - buffer_h]
            outc = os.path.join(crop_dir, f"crop_{i}.tif")
            opts = gdal.WarpOptions(
                format="GTiff",
                outputBounds=core,
                xRes=pw, yRes=ph,
                multithread=True,
                warpOptions=["OPTIMIZE_SIZE=YES"],
                dstNodata=0
            )
            gdal.Warp(outc, fpath, options=opts)
            cropped.append(outc)

        print("Building VRT for merge...")
        vrt = os.path.join(crop_dir, "merged.vrt")
        gdal.BuildVRT(vrt, cropped, options=gdal.BuildVRTOptions(xRes=pw, yRes=ph))

        print(f"Translating merged COG to {output_path} …")
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdal.Translate(
            output_path, vrt,
            options=gdal.TranslateOptions(
                format="GTiff",
                creationOptions=[
                    "COMPRESS=LZW", "PREDICTOR=2", "BIGTIFF=YES", "TILED=YES"
                ]
            )
        )
        ds = gdal.Open(output_path, gdal.GA_Update)
        gdal.SetConfigOption("COMPRESS_OVERVIEW", "LZW")
        gdal.SetConfigOption("PREDICTOR_OVERVIEW", "2")
        ds.BuildOverviews("NEAREST", [2,4,8,16,32])
        ds = None
        gdal.Unlink(vrt)
    print("Merge complete.")

def calculate_chip_crs_parameters(parent_raster_path: str,
                                  target_chip_width_px: int = TARGET_SIZE[1],
                                  target_chip_height_px: int = TARGET_SIZE[0],
                                  buffer_fraction: float = BUFFER_FRAC):
    with rasterio.open(parent_raster_path) as src:
        res_x, res_y = abs(src.transform[0]), abs(src.transform[4])

    # These are the CRS dimensions of the actual core area of each chip.
    # This will be used for grid generation and will be the size of the final cropped chip.
    actual_core_w_crs = target_chip_width_px * res_x
    actual_core_h_crs = target_chip_height_px * res_y
    
    if actual_core_w_crs <= 0:
        raise ValueError("Target chip width results in non-positive core CRS width. Check TARGET_SIZE and raster resolution.")
    if actual_core_h_crs <= 0:
        raise ValueError("Target chip height results in non-positive core CRS height. Check TARGET_SIZE and raster resolution.")
        
    # This is the buffer width/height to ADD to EACH SIDE of the actual_core_crs dimensions.
    buffer_to_add_w_crs = buffer_fraction * actual_core_w_crs
    buffer_to_add_h_crs = buffer_fraction * actual_core_h_crs
    
    return actual_core_w_crs, actual_core_h_crs, buffer_to_add_w_crs, buffer_to_add_h_crs, res_x, res_y

def register_raster_with_chips(unreg_path: str, reg_path: str,
                               output_path: str, width: float, height: float,
                               buffer_w: float, buffer_h: float, device=DEVICE,
                               max_workers=MAX_WORKERS, batch_size=BATCH_SIZE):
    grid = generate_grid_points(unreg_path, width, height)
    warped = []
    # loaded_chips will store dicts: {'id': original_idx, 'un_chip': Chip, 'reg_chip': Chip}
    loaded_chips_data: List[dict] = []
    results = []

    # 1) Load chips
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, row in grid.iterrows():
            # Pass idx as the identifier
            futures[ex.submit(load_chips, row.geometry,
                              unreg_path, reg_path, width, height, buffer_w, buffer_h)] = idx
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading chips", unit="chip"):
            original_idx = futures[fut] # This is the original grid index
            try:
                u_chip, r_chip = fut.result()
                loaded_chips_data.append({'id': original_idx, 'un_chip': u_chip, 'reg_chip': r_chip})
            except Exception as e:
                tqdm.write(f"⚠️  Load failure for chip id {original_idx}: {e}")

    if not loaded_chips_data:
        print("❌  No chips loaded; aborting.")
        return

    # Sort loaded_chips_data by 'id' to ensure original order if it was jumbled by async loading,
    # though it might not be strictly necessary if batch_get_loftr_matches handles arbitrary order.
    # However, for consistency with how all_pairs_results is structured, this is good.
    loaded_chips_data.sort(key=lambda x: x['id'])

    # 2) LoFTR batch matching
    # Prepare input for batch_get_loftr_matches
    loftr_input_data = [{'id': lcd['id'], 'un_img': lcd['un_chip'].array, 'reg_img': lcd['reg_chip'].array} for lcd in loaded_chips_data]

    tmpdir = tempfile.mkdtemp()
    
    # matches will be a list of dicts: {'id': original_idx, 'mkpts_reg': ..., 'mkpts_un': ..., 'inliers': ...}
    # The order of this list will correspond to the order of loftr_input_data
    matches_results = batch_get_loftr_matches(loftr_input_data,
                                              device=device, batch_size=batch_size)

    # Create a dictionary for quick lookup of matches by id
    matches_map = {match_data['id']: match_data for match_data in matches_results}

    # 3) warp & collect stats
    success_warp_count = 0
    skipped_no_data_count = 0
    skipped_no_inliers_count = 0
    skipped_low_gcps_count = 0
    error_count = 0

    # pairs = list(zip(loaded, matches))
    print("Warping chips...")

    # Iterate through loaded_chips_data to process each chip
    for chip_load_info in tqdm(loaded_chips_data, desc="Warping chips", unit="chip"):
        original_idx = chip_load_info['id']
        un_chip = chip_load_info['un_chip']
        reg_chip = chip_load_info['reg_chip']

        # Retrieve the match data for this chip using its original_idx
        match_data = matches_map.get(original_idx)

        # Initialize default values for stats in case of early exit or error
        cnt_in = 0
        gcps_list = []
        warp_success_flag = False

        try:
            if match_data is None:
                # This should ideally not happen if batch_get_loftr_matches returns a result for every input
                print(f"Chip {original_idx}: skipped (no match data found).")
                error_count +=1 # Or a new counter for missing match data
                # Log to results with failure
                results.append({
                    "idx": original_idx, "inlier_count": 0, "gcp_count": 0, "warp_success": False,
                    "geometry": Polygon([ # Placeholder geometry or calculate actual core
                        (un_chip.bounds[0] + buffer_w, un_chip.bounds[1] + buffer_h),
                        (un_chip.bounds[0] + buffer_w, un_chip.bounds[3] - buffer_h),
                        (un_chip.bounds[2] - buffer_w, un_chip.bounds[3] - buffer_h),
                        (un_chip.bounds[2] - buffer_w, un_chip.bounds[1] + buffer_h),
                        (un_chip.bounds[0] + buffer_w, un_chip.bounds[1] + buffer_h)
                    ])
                })
                continue

            mkpts_reg_matched = match_data['mkpts_reg']
            mkpts_un_matched = match_data['mkpts_un']
            inliers_matched = match_data['inliers']
            
            # 1) no‐data check (on original chip arrays)
            if un_chip.array.sum() == 0 or reg_chip.array.sum() == 0:
                skipped_no_data_count += 1
                # tqdm.write(f"Chip {original_idx}: skipped (no data)") # tqdm.write to avoid breaking bar
            else:
                # 2) count inliers
                cnt_in = int(inliers_matched.sum())
                # generate_chip_gcps expects (un_chip, reg_chip, mkpts_un, mkpts_reg, inliers)
                gcps_list = generate_chip_gcps(un_chip, reg_chip, mkpts_un_matched, mkpts_reg_matched, inliers_matched)

                if cnt_in == 0:
                    skipped_no_inliers_count += 1
                    # tqdm.write(f"Chip {original_idx}: skipped (no inliers)")
                elif len(gcps_list) < 10: # Check actual GCPs generated, not just inliers
                    skipped_low_gcps_count += 1
                    # tqdm.write(f"Chip {original_idx}: skipped ({len(gcps_list)} GCPs only)")
                else:
                    # 3) warp it!
                    out_chip_path = os.path.join(tmpdir, f"chip_{original_idx}.tif")
                    warp_chip(un_chip, gcps_list, out_chip_path)
                    success_warp_count += 1
                    # tqdm.write(f"Chip {original_idx}: warped ✅ (inliers = {cnt_in}, GCPs = {len(gcps_list)})")
                    pw = abs(un_chip.profile["transform"][0])
                    ph = abs(un_chip.profile["transform"][4])
                    warped.append((out_chip_path, un_chip.bounds, (pw, ph)))
                    warp_success_flag = True
            
            # 4) record stats for every chip
            xmin_buff, ymin_buff, xmax_buff, ymax_buff = un_chip.bounds
            core_xmin = xmin_buff + buffer_w
            core_ymin = ymin_buff + buffer_h
            core_xmax = xmax_buff - buffer_w
            core_ymax = ymax_buff - buffer_h
            
            core_polygon = Polygon([
                (core_xmin, core_ymin),  # bottom-left
                (core_xmin, core_ymax),  # top-left
                (core_xmax, core_ymax),  # top-right
                (core_xmax, core_ymin),  # bottom-right
                (core_xmin, core_ymin)   # close polygon
            ])

            results.append({
                "idx": original_idx,
                "inlier_count": cnt_in,
                "gcp_count": len(gcps_list),
                "warp_success": warp_success_flag,
                "geometry": core_polygon
            })

        except Exception as e:
            error_count += 1
            print(f"Chip {original_idx}: ERROR — {e}")
            
            # Calculate core bounds even in case of error for consistent GeoJSON structure
            xmin_buff, ymin_buff, xmax_buff, ymax_buff = un_chip.bounds
            core_xmin = xmin_buff + buffer_w
            core_ymin = ymin_buff + buffer_h
            core_xmax = xmax_buff - buffer_w
            core_ymax = ymax_buff - buffer_h

            core_polygon = Polygon([
                (core_xmin, core_ymin),  # bottom-left
                (core_xmin, core_ymax),  # top-left
                (core_xmax, core_ymax),  # top-right
                (core_xmax, core_ymin),  # bottom-right
                (core_xmin, core_ymin)   # close polygon
            ])
            
            results.append({
                "idx": original_idx,
                "inlier_count": 0,
                "gcp_count": 0,
                "warp_success": False,
                "geometry": core_polygon
            })

    print("\nWarping summary:")
    print(f"  Successfully warped: {success_warp_count}")
    print(f"  Skipped (no data): {skipped_no_data_count}")
    print(f"  Skipped (no inliers): {skipped_no_inliers_count}")
    print(f"  Skipped (low GCPs): {skipped_low_gcps_count}")
    print(f"  Errors: {error_count}\n")

    # 4) Merge
    merge_warped_chips(warped, output_path, buffer_w, buffer_h)

    # 5) Write stats
    df  = pd.DataFrame(results)
    gdf = gpd.GeoDataFrame(df, geometry="geometry",
                           crs=rasterio.open(unreg_path).crs)
    stats_path = output_path.replace(".tif", "_chip_stats.geojson")
    gdf.to_file(stats_path, driver="GeoJSON")
    print(f"✅  Stats written to: {stats_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    core_width_length, core_height_length, buffer_width_val, buffer_height_val, pix_res_x, pix_res_y = calculate_chip_crs_parameters(
        parent_raster_path=UNREG_URL,
        target_chip_width_px=TARGET_SIZE[1],
        target_chip_height_px=TARGET_SIZE[0],
        buffer_fraction=BUFFER_FRAC
    )

    print("💡  Computed chip dimensions:")
    print(f"    Core size (px) ≈ {core_width_length/pix_res_x:.0f}w × {core_height_length/pix_res_y:.0f}h")
    print(f"    Buffer (px) ≈ {buffer_width_val/pix_res_x:.1f}w × {buffer_height_val/pix_res_y:.1f}h\n")

    register_raster_with_chips(
        unreg_path=str(UNREG_URL),
        reg_path=str(REG_URL),
        output_path=OUTPUT_PATH,
        width=core_width_length,
        height=core_height_length,
        buffer_w=buffer_width_val,
        buffer_h=buffer_height_val,
        device=DEVICE,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE
    )
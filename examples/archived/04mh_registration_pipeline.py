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

# Constants that user
DEVICE      = 'mps'
MAX_WORKERS = 10
BATCH_SIZE  = 16
CHUNK_SIZE  = 32 # Number of grid points to process in one full load-match-warp cycle
TARGET_SIZE = (480, 640)       # (H, W)
BUFFER_FRAC = 0.1

#UNREG_URL = '/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_test.tif'
UNREG_URL   = "https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/multispectral/front_country-batch_5-MS.tif"
REG_URL     = "https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/front_country_2024.tif"
OUTPUT_PATH = "/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_corrected.tif"

# More baked in constants
MATCHER     = KF.LoFTR(pretrained='outdoor').to(DEVICE).eval()
LOFTR_REPROJ_THRESH_LEVELS_PX = [0.5, 1.0, 2.0]  # RANSAC reprojection threshold in LoFTR's input image space (pixels)
CONFIDENCE_LEVELS = [0.999, 0.95]              # Confidence levels for RANSAC
MIN_LOFTR_MATCHES_FOR_FUNDAMENTAL_MATRIX = 7   # Minimum number of matches required for cv2.findFundamentalMat
RESAMPLE_ALG= "cubic"
POLY_ORDER  = 3


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

        # Explicitly delete GPU tensors now that we have CPU copies or derived data
        del corr['keypoints0'] # These are tensors on the GPU
        del corr['keypoints1']
        del corr['batch_indexes']
        del corr # Delete the whole dictionary, which might hold other tensors
        del img0_tensors_reg
        del img1_tensors_un

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
            
            # Explicitly delete intermediate resized arrays once original scale arrays are computed
            del mkpts_reg_resized
            del mkpts_un_resized

            inliers = np.zeros(mkpts_reg_orig.shape[0], dtype=bool)
            Fm = None
            inliers_mask = None
            if mkpts_reg_orig.shape[0] >= MIN_LOFTR_MATCHES_FOR_FUNDAMENTAL_MATRIX:
                avg_reg_img_scale_factor = (reg_img_scales[0] + reg_img_scales[1]) / 2.0
                found_inliers_for_pair = False
                for loftr_thresh_px in LOFTR_REPROJ_THRESH_LEVELS_PX:
                    ransac_thresh_orig_px = loftr_thresh_px * avg_reg_img_scale_factor
                    for confidence_val in CONFIDENCE_LEVELS:
                        try:
                            Fm, inliers_mask = cv2.findFundamentalMat(
                                mkpts_reg_orig, mkpts_un_orig, 
                                method=cv2.USAC_MAGSAC,
                                ransacReprojThreshold=ransac_thresh_orig_px,
                                confidence=confidence_val,
                                maxIters=100000
                            )
                            if Fm is not None and Fm.shape[0] == 3 and not np.all(Fm == 0):
                                inliers = inliers_mask.ravel() > 0
                                found_inliers_for_pair = True
                                break
                        except cv2.error: # cv2.error can be raised if no fundamental matrix is found
                            # Explicitly delete potentially partially assigned Fm or inliers_mask on error
                            del Fm
                            del inliers_mask
                            Fm, inliers_mask = None, None # Ensure they are reset
                            continue
                        if found_inliers_for_pair:
                            break
                    if found_inliers_for_pair:
                        break
            
            loftr_processed_results_map[original_id] = {
                'id': original_id,
                'mkpts_reg': mkpts_reg_orig, 
                'mkpts_un': mkpts_un_orig, 
                'inliers': inliers
            }
            # Explicitly delete intermediate arrays from RANSAC
            del Fm
            del inliers_mask

        # Attempt to clear PyTorch MPS cache after each GPU batch processing
        if device == 'mps':
            torch.mps.empty_cache()

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
        
        vrt_ds = None  # Initialize to None
        try:
            # Create an in-memory VRT dataset from the source chip with GCPs
            vrt_ds = gdal.Translate("", mem.name, # Output to in-memory dataset
                                  options=gdal.TranslateOptions(format="VRT", GCPs=gcps))
            if vrt_ds is None:
                # This can happen if Translate fails for some reason (e.g. no GCPs, invalid GCPs)
                # Although generate_chip_gcps and the check for len(gcps_list) < 10 should prevent empty/bad GCPs.
                raise RuntimeError(f"GDAL Translate to VRT failed for chip data originally from {mem.name}")

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
            # Warp the in-memory VRT dataset to the output file path
            # The return value of gdal.Warp when outputting to a file is typically a Dataset object
            # representing the *output* dataset, or None/True on failure/success without dataset return.
            # We don't strictly need to keep the output dataset object from gdal.Warp here.
            gdal.Warp(out_path, vrt_ds, options=warp_opts)
        finally:
            # Crucially, dereference the in-memory VRT dataset to allow GDAL to free resources.
            # This is important because vrt_ds is an actual GDAL Dataset object.
            vrt_ds = None

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
    
    # This list will collect warped chip file paths and metadata for merging
    warped_all_chunks = [] 

    # Statistics counters aggregated across chunks
    total_success_warp_count = 0
    total_skipped_no_data_count = 0
    total_skipped_no_inliers_count = 0
    total_skipped_low_gcps_count = 0
    total_error_count = 0

    # Ensure the output directory for the final merged raster exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Temporary directory for warped chip files
    with tempfile.TemporaryDirectory() as main_tmpdir:
        num_chunks = (len(grid) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_idx in tqdm(range(num_chunks), desc="Total Progress (Chunks)", unit="chunk"):
            chunk_start_idx = chunk_idx * CHUNK_SIZE
            chunk_end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(grid))
            current_grid_chunk_df = grid.iloc[chunk_start_idx:chunk_end_idx]

            if current_grid_chunk_df.empty:
                continue

            tqdm.write(f"--- Processing Chunk {chunk_idx + 1}/{num_chunks} (Grid Indices {chunk_start_idx} to {chunk_end_idx-1}) ---")

            # Per-chunk data structures
            loaded_chips_data_chunk: List[dict] = []
            
            # 1) Load chips for the current chunk
            futures_chunk = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for original_idx, row in current_grid_chunk_df.iterrows():
                    futures_chunk[ex.submit(load_chips, row.geometry,
                                            unreg_path, reg_path, width, height, buffer_w, buffer_h)] = original_idx
                
                progress_bar_loading = tqdm(as_completed(futures_chunk), total=len(futures_chunk), desc=f"  Loading chips (Chunk {chunk_idx+1})", unit="chip", leave=False)
                for fut in progress_bar_loading:
                    original_idx_from_future = futures_chunk[fut]
                    try:
                        u_chip, r_chip = fut.result()
                        loaded_chips_data_chunk.append({'id': original_idx_from_future, 'un_chip': u_chip, 'reg_chip': r_chip})
                    except Exception as e:
                        tqdm.write(f"⚠️  Load failure for chip id {original_idx_from_future} in chunk {chunk_idx+1}: {e}")

            if not loaded_chips_data_chunk:
                tqdm.write(f"❌  No chips loaded for chunk {chunk_idx+1}; skipping further processing for this chunk.")
                continue

            loaded_chips_data_chunk.sort(key=lambda x: x['id'])

            # 2) LoFTR batch matching for the current chunk
            loftr_input_data_chunk = [{'id': lcd['id'], 'un_img': lcd['un_chip'].array, 'reg_img': lcd['reg_chip'].array} for lcd in loaded_chips_data_chunk]
            
            matches_results_chunk = batch_get_loftr_matches(loftr_input_data_chunk,
                                                            device=device, batch_size=batch_size)
            matches_map_chunk = {match_data['id']: match_data for match_data in matches_results_chunk}

            # 3) Warp & collect stats for the current chunk
            chunk_success_warp_count = 0
            chunk_skipped_no_data_count = 0
            chunk_skipped_no_inliers_count = 0
            chunk_skipped_low_gcps_count = 0
            chunk_error_count = 0

            progress_bar_warping = tqdm(loaded_chips_data_chunk, desc=f"  Warping chips (Chunk {chunk_idx+1})", unit="chip", leave=False)
            for chip_load_info in progress_bar_warping:
                original_idx = chip_load_info['id']
                un_chip = chip_load_info['un_chip']
                reg_chip = chip_load_info['reg_chip']
                match_data = matches_map_chunk.get(original_idx)

                try:
                    if match_data is None:
                        tqdm.write(f"Chip {original_idx}: skipped (no match data found in chunk {chunk_idx+1}).")
                        chunk_error_count +=1 
                        continue

                    mkpts_reg_matched = match_data['mkpts_reg']
                    mkpts_un_matched = match_data['mkpts_un']
                    inliers_matched = match_data['inliers']
                    
                    if un_chip.array.sum() == 0 or reg_chip.array.sum() == 0:
                        chunk_skipped_no_data_count += 1
                    else:
                        cnt_in = int(inliers_matched.sum())
                        gcps_list = generate_chip_gcps(un_chip, reg_chip, mkpts_un_matched, mkpts_reg_matched, inliers_matched)
                        if cnt_in == 0:
                            chunk_skipped_no_inliers_count += 1
                        elif len(gcps_list) < 10:
                            chunk_skipped_low_gcps_count += 1
                        else:
                            out_chip_path = os.path.join(main_tmpdir, f"chip_{original_idx}.tif")
                            warp_chip(un_chip, gcps_list, out_chip_path)
                            chunk_success_warp_count += 1
                            pw = abs(un_chip.profile["transform"][0])
                            ph = abs(un_chip.profile["transform"][4])
                            warped_all_chunks.append((out_chip_path, un_chip.bounds, (pw, ph)))

                except Exception as e:
                    chunk_error_count += 1
                    tqdm.write(f"Chip {original_idx}: ERROR during warping in chunk {chunk_idx+1} — {e}")
            
            # Aggregate chunk stats to total stats
            total_success_warp_count += chunk_success_warp_count
            total_skipped_no_data_count += chunk_skipped_no_data_count
            total_skipped_no_inliers_count += chunk_skipped_no_inliers_count
            total_skipped_low_gcps_count += chunk_skipped_low_gcps_count
            total_error_count += chunk_error_count

            tqdm.write(f"  Chunk {chunk_idx+1} Summary: Warped: {chunk_success_warp_count}, Skipped (NoData: {chunk_skipped_no_data_count}, NoInliers: {chunk_skipped_no_inliers_count}, LowGCPs: {chunk_skipped_low_gcps_count}), Errors: {chunk_error_count}")

            # Explicitly delete large chunk-specific data structures to help free memory
            del loaded_chips_data_chunk
            del loftr_input_data_chunk
            del matches_results_chunk
            del matches_map_chunk
            import gc
            gc.collect()
        
        # --- End of chunk processing loop ---

        print("Overall Warping Summary:")
        print(f"  Successfully warped: {total_success_warp_count}")
        print(f"  Skipped (no data): {total_skipped_no_data_count}")
        print(f"  Skipped (no inliers): {total_skipped_no_inliers_count}")
        print(f"  Skipped (low GCPs): {total_skipped_low_gcps_count}")
        print(f"  Errors during processing: {total_error_count}")

        # Merge (operates on files in main_tmpdir)
        if warped_all_chunks:
            merge_warped_chips(warped_all_chunks, output_path, buffer_w, buffer_h)
        else:
            print("⚠️  No chips were successfully warped. Output raster will not be created.")

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
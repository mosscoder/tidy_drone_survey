import os
import cv2
import geopandas as gpd
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from shapely.geometry import Point

# --- Configuration - Adjust these as per your setup ---
GEOJSON_PATH = '/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_corrected_chip_stats.geojson'
UNREG_RASTER_PATH = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/multispectral/front_country-batch_5-MS.tif'
REG_RASTER_PATH = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/front_country_2024.tif'

CHIP_WIDTH_CRS = 30.27
CHIP_HEIGHT_CRS = 20.81
CHIP_BUFFER_CRS = 3.78

LOFTR_DEVICE = 'mps'
LOFTR_TARGET_SIZE = (480, 640) # As used in batch_get_loftr_matches
LOFTR_REPROJ_THRESHOLDS = [0.5, 0.75, 1.0, 2.0] # As used in batch_get_loftr_matches

TARGET_FEATURE_INDEX = 3140 # Set this to an integer (e.g., 0, 1, 2...) to debug a specific feature by its index in the GeoJSON
TARGET_IDX_VALUE = 2100 # Set this to an integer to debug a specific feature by its 'idx' field value

# --- Helper Functions (Copied/adapted from registration_pipeline_batch.py) ---

def get_bbox_bounds(point_gdf, width_length, height_length, buffer=0):
    """
    Given a GeoDataFrame with a single point geometry, return (xmin, xmax, ymin, ymax)
    for a rectangular bounding box of given width and height centered at the point.
    """
    half_width = width_length / 2 + buffer
    half_height = height_length / 2 + buffer
    x = point_gdf.geometry.iloc[0].x
    y = point_gdf.geometry.iloc[0].y
    xmin = x - half_width
    xmax = x + half_width
    ymin = y - half_height
    ymax = y + half_height
    return xmin, ymin, xmax, ymax

def load_chips_for_debug(
    point_gdf,
    unregistered_path,
    registered_path,
    width_length,
    height_length,
    buffer=0
) -> tuple:
    """
    Simplified from load_chips for debugging: loads arrays and profiles.
    """
    bbox = get_bbox_bounds(point_gdf, width_length, height_length, buffer)

    # Unregistered chip
    with rasterio.open(unregistered_path) as un_ds:
        un_window = from_bounds(*bbox, un_ds.transform)
        un_array = un_ds.read(window=un_window)
        un_profile = un_ds.profile.copy()
        un_crs = un_ds.crs
        win_transform = window_transform(un_window, un_ds.transform)
        un_profile.update({
            'height': un_window.height,
            'width': un_window.width,
            'transform': win_transform
        })

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
            win_transform = window_transform(reg_window, vrt.transform)
            reg_profile.update({
                'height': reg_window.height,
                'width': reg_window.width,
                'transform': win_transform
            })
    return (un_array, un_profile), (reg_array, reg_profile)

def rasterio_to_torch_tensor(numpy_array: np.ndarray, bands: int = [0, 1, 2]) -> torch.Tensor:
    tensor = torch.from_numpy(numpy_array[bands]).unsqueeze(0)
    return tensor

def resize_for_matching(img_array, target_size=(512, 512)):
    _, orig_h, orig_w = img_array.shape
    target_h, target_w = target_size
    h_scale = orig_h / target_h
    w_scale = orig_w / target_w
    resized_array = np.zeros((img_array.shape[0], target_h, target_w), dtype=img_array.dtype)
    for i in range(img_array.shape[0]):
        band = img_array[i]
        resized = cv2.resize(band, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_array[i] = resized
    return resized_array, (w_scale, h_scale)

def apply_clahe_to_grayscale_batch(img_tensor: torch.Tensor,
                                    clip_limit: float = 2.0,
                                    tile_grid_size: tuple = (8, 8)) -> torch.Tensor:
    assert img_tensor.dim() == 4 and img_tensor.shape[1] == 3, "Input must be RGB batch (B, 3, H, W)"
    grayscale = K.color.rgb_to_grayscale(img_tensor)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = []
    for img in grayscale:
        img_np = (img.squeeze().cpu().numpy() * 255).astype('uint8')
        cl_img = clahe.apply(img_np)
        cl_tensor = torch.tensor(cl_img, dtype=torch.float32).div(255).unsqueeze(0)
        enhanced.append(cl_tensor)
    return torch.stack(enhanced)

def save_matches_visualization(
    img1_arr_raw, # Registered array (bands, H, W)
    img2_arr_raw, # Unregistered array (bands, H, W)
    mkpts1,       # Keypoints for img1 (N, 2) e.g., mkpts_reg
    mkpts2,       # Keypoints for img2 (N, 2) e.g., mkpts_unreg
    inliers_mask, # Boolean array (N,)
    filename_prefix,
    bands_to_display=[0, 1, 2] # Assuming these are R, G, B or similar
):
    """Saves a side-by-side visualization of matches between two images."""
    # 1. Select bands and transpose for OpenCV (H, W, C)
    # Ensure we don't go out of bounds if an image has fewer bands than expected
    num_bands_img1 = img1_arr_raw.shape[0]
    num_bands_img2 = img2_arr_raw.shape[0]
    
    actual_bands_to_display_img1 = [b for b in bands_to_display if b < num_bands_img1]
    actual_bands_to_display_img2 = [b for b in bands_to_display if b < num_bands_img2]

    if not actual_bands_to_display_img1 or not actual_bands_to_display_img2:
        print(f"  Warning: Not enough bands to display for {filename_prefix}. Skipping visualization.")
        return

    img1_selected_bands = img1_arr_raw[actual_bands_to_display_img1, :, :]
    img2_selected_bands = img2_arr_raw[actual_bands_to_display_img2, :, :]

    img1_vis = img1_selected_bands.transpose(1, 2, 0)
    img2_vis = img2_selected_bands.transpose(1, 2, 0)

    # 2. Normalize to 0-255 and convert to uint8
    def normalize_to_uint8(img_float_raw_hwc): # Expects (H, W, C)
        img_float = img_float_raw_hwc.astype(np.float32)
        if np.all(img_float == 0):
            return np.zeros(img_float.shape, dtype=np.uint8)

        normalized_bands_list = []
        for i in range(img_float.shape[2]): # Iterate over channels
            band = img_float[:, :, i]
            min_val = np.percentile(band, 2)
            max_val = np.percentile(band, 98)

            if max_val <= min_val: # Handles flat or near-flat bands
                scaled_band = np.zeros(band.shape, dtype=np.float32)
            else:
                band_clipped = np.clip(band, min_val, max_val)
                scaled_band = (band_clipped - min_val) / (max_val - min_val) * 255.0
            normalized_bands_list.append(scaled_band.astype(np.uint8))
        
        if not normalized_bands_list: # Should not happen if input shape is correct
             return np.zeros(img_float_raw_hwc.shape, dtype=np.uint8)

        scaled_img = np.stack(normalized_bands_list, axis=-1)
        return scaled_img

    img1_u8 = normalize_to_uint8(img1_vis.copy())
    img2_u8 = normalize_to_uint8(img2_vis.copy())

    # 3. Convert to BGR for OpenCV drawing functions
    # Handle cases: 3-channel (assume RGB -> BGR), 1-channel (GRAY -> BGR)
    def convert_to_bgr_for_draw(img_u8_in):
        if img_u8_in.ndim == 3 and img_u8_in.shape[2] == 3:
            return cv2.cvtColor(img_u8_in, cv2.COLOR_RGB2BGR)
        elif img_u8_in.ndim == 2 or (img_u8_in.ndim == 3 and img_u8_in.shape[2] == 1):
            return cv2.cvtColor(img_u8_in, cv2.COLOR_GRAY2BGR)
        # else: # e.g. 2 or 4+ channels, not directly supported by simple conversion to BGR
        # print(f"  Warning: Image for {filename_prefix} has {img_u8_in.shape[2]} channels, cannot convert to BGR easily. Using as is.")
        return img_u8_in # Return as is, may or may not work with drawMatches

    img1_bgr = convert_to_bgr_for_draw(img1_u8)
    img2_bgr = convert_to_bgr_for_draw(img2_u8)

    # 4. Filter keypoints by inliers
    mkpts1_inliers = mkpts1[inliers_mask]
    mkpts2_inliers = mkpts2[inliers_mask]

    # 5. Convert keypoints to cv2.KeyPoint objects
    kp1 = [cv2.KeyPoint(p[0], p[1], 10) for p in mkpts1_inliers] # size=10 for visibility
    kp2 = [cv2.KeyPoint(p[0], p[1], 10) for p in mkpts2_inliers]

    # 6. Create DMatch objects
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

    if not matches:
        print(f"  No inlier matches to draw for {filename_prefix}.")
        # Optionally, save individual images if no matches
        # cv2.imwrite(f"{filename_prefix}_img1_no_matches.png", img1_bgr)
        # cv2.imwrite(f"{filename_prefix}_img2_no_matches.png", img2_bgr)
        return

    # 7. Draw matches
    drawn_matches_img = cv2.drawMatches(
        img1_bgr, kp1, img2_bgr, kp2, matches, None,
        matchColor=(0, 255, 0), # Green lines for matches
        singlePointColor=(255, 0, 0), # Blue for single points (not drawn due to flag)
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 8. Save the image
    output_filename = f"examples/{filename_prefix}_matches.png"
    try:
        cv2.imwrite(output_filename, drawn_matches_img)
        print(f"  Saved matches visualization to {output_filename}")
    except Exception as e:
        print(f"  Error saving matches visualization {output_filename}: {e}")

def debug_loftr_on_pair(img1_raw_array, img2_raw_array, device_str, target_size_tuple, reproj_threshold_levels_list):
    """
    Runs LoFTR matching on a single pair of images with detailed output.
    img1_raw_array: Registered image (typically)
    img2_raw_array: Unregistered image (typically)
    """
    matcher = KF.LoFTR(pretrained='outdoor').to(device_str).eval() # Ensure eval mode

    # Preprocess image 1 (Registered)
    if np.all(img1_raw_array == 0):
        print("Image 1 (Registered) is all black. Skipping.")
        return None, None, None
    img1_resized, img1_scale = resize_for_matching(img1_raw_array, target_size_tuple)
    img1_tensor = rasterio_to_torch_tensor(img1_resized).float().div(255.0)
    img1_gray_tensor = apply_clahe_to_grayscale_batch(img1_tensor).to(device_str)
    # Reshape for LoFTR: (1, 1, H, W)
    img1_gray_tensor = img1_gray_tensor.reshape(-1, 1, target_size_tuple[0], target_size_tuple[1])


    # Preprocess image 2 (Unregistered)
    if np.all(img2_raw_array == 0):
        print("Image 2 (Unregistered) is all black. Skipping.")
        return None, None, None
    img2_resized, img2_scale = resize_for_matching(img2_raw_array, target_size_tuple)
    img2_tensor = rasterio_to_torch_tensor(img2_resized).float().div(255.0)
    img2_gray_tensor = apply_clahe_to_grayscale_batch(img2_tensor).to(device_str)
    # Reshape for LoFTR: (1, 1, H, W)
    img2_gray_tensor = img2_gray_tensor.reshape(-1, 1, target_size_tuple[0], target_size_tuple[1])

    print(f"  Image 1 (Reg) original shape: {img1_raw_array.shape}, resized to: {img1_resized.shape}, scale: {img1_scale}")
    print(f"  Image 2 (Unreg) original shape: {img2_raw_array.shape}, resized to: {img2_resized.shape}, scale: {img2_scale}")
    print(f"  Image 1 tensor shape for LoFTR: {img1_gray_tensor.shape}")
    print(f"  Image 2 tensor shape for LoFTR: {img2_gray_tensor.shape}")

    # LoFTR Matching
    with torch.inference_mode():
        input_dict = {"image0": img1_gray_tensor, "image1": img2_gray_tensor}
        correspondences = matcher(input_dict)

    mkpts0_resized = correspondences["keypoints0"].cpu().numpy() # In LoFTR's resized space
    mkpts1_resized = correspondences["keypoints1"].cpu().numpy() # In LoFTR's resized space
    confidences = correspondences["confidence"].cpu().numpy()

    print(f"  LoFTR found {mkpts0_resized.shape[0]} raw correspondences.")
    print(f"  Confidences: {confidences}")

    # Rescale keypoints
    mkpts0_orig_scale = mkpts0_resized * np.array(img1_scale)
    mkpts1_orig_scale = mkpts1_resized * np.array(img2_scale)

    # Fundamental Matrix RANSAC - REVISED LOGIC
    inliers_mask = None # Will store the boolean mask
    Fm_final = None     # Will store the chosen Fundamental Matrix
    
    if mkpts0_orig_scale.shape[0] >= 7:
        print(f"  Attempting Fundamental Matrix estimation with {mkpts0_orig_scale.shape[0]} points.")
        confidence_levels_cv = [0.999, 0.95]
        avg_img1_scale_factor = (img1_scale[0] + img1_scale[1]) / 2.0

        # Outer loop: Iterate through reprojection thresholds
        for reproj_thresh_px_loftr_space in reproj_threshold_levels_list:
            adaptive_ransac_reproj_thresh_orig_space = reproj_thresh_px_loftr_space * avg_img1_scale_factor
            print(f"    Trying LoFTR space reproj threshold: {reproj_thresh_px_loftr_space} px (scaled to {adaptive_ransac_reproj_thresh_orig_space:.2f} px in original img1 space)")

            # Inner loop: Iterate through confidence levels
            for conf_cv in confidence_levels_cv:
                print(f"      Using OpenCV RANSAC confidence: {conf_cv}")
                try:
                    Fm_cand, current_inliers_cv_mask = cv2.findFundamentalMat(
                        mkpts0_orig_scale, mkpts1_orig_scale,
                        method=cv2.USAC_MAGSAC,
                        ransacReprojThreshold=adaptive_ransac_reproj_thresh_orig_space,
                        confidence=conf_cv,
                        maxIters=100000
                    )
                    
                    # Check if a valid Fundamental Matrix was found (as per original script)
                    if Fm_cand is not None and Fm_cand.shape == (3, 3) and not np.all(Fm_cand == 0):
                        inliers_mask = current_inliers_cv_mask.ravel() > 0 # Store this result
                        Fm_final = Fm_cand # Store the matrix
                        num_inliers_this_attempt = np.sum(inliers_mask)
                        print(f"        Valid Fm found. Inliers for THIS attempt: {num_inliers_this_attempt}. Storing this result and breaking from confidence loop.")
                        break  # Break from confidence loop (inner break)
                    else:
                        print(f"        cv2.findFundamentalMat returned None or invalid/all-zero Fm for this attempt.")
                        
                except cv2.error as e:
                    print(f"        cv2.error in findFundamentalMat: {e}")
                    continue # Continue to next confidence level
            
            # If inliers_mask is now set (meaning a valid Fm was found in the inner loop),
            # then break from the outer (reprojection threshold) loop as well.
            if inliers_mask is not None:
                print(f"    A valid Fm was processed in the confidence loop. Breaking from reprojection threshold loop.")
                break 
    
    # If after all attempts, inliers_mask is still None, it means no valid Fm was ever found.
    if inliers_mask is None:
        print("  Failed to find any valid Fundamental Matrix after all attempts (all Fm were None or invalid).")
        inliers_mask = np.zeros(mkpts0_orig_scale.shape[0], dtype=bool) # Ensure it's a boolean array
    elif not np.any(inliers_mask): # A valid Fm was found, but it resulted in zero inliers
        print(f"  A valid Fundamental Matrix (Fm_final) was found, but it resulted in 0 inliers with the chosen parameters.")


    final_inlier_count = np.sum(inliers_mask)
    print(f"  Final inlier count based on replicated logic: {final_inlier_count}")
    if Fm_final is not None:
        print(f"  Final Fundamental Matrix (Fm_final):\n{Fm_final}")
    else:
        print(f"  No final Fundamental Matrix was selected.")

    return mkpts0_orig_scale, mkpts1_orig_scale, inliers_mask

# --- Main script ---
if __name__ == "__main__":
    print("Loading GeoJSON stats...")
    stats_gdf = gpd.read_file(GEOJSON_PATH)
    print(f"Loaded {len(stats_gdf)} chip records.")

    chips_to_process_gdf = None

    if TARGET_IDX_VALUE is not None:
        print(f"Attempting to debug specific feature with idx: {TARGET_IDX_VALUE}")
        # Find the chip by its 'idx' field value
        target_chip_gdf = stats_gdf[stats_gdf['idx'] == TARGET_IDX_VALUE]
        if not target_chip_gdf.empty:
            chips_to_process_gdf = target_chip_gdf
            selected_chip_data = chips_to_process_gdf.iloc[0] # Get the Series for checking inlier count
            # The gdf_idx for this chip will be its original index in stats_gdf
            original_gdf_idx = chips_to_process_gdf.index[0]
            print(f"  Found chip with idx {TARGET_IDX_VALUE} at GeoJSON feature index {original_gdf_idx}.")
            if selected_chip_data['inlier_count'] > 0:
                print(f"  INFO: Chip with idx {TARGET_IDX_VALUE} (original script idx: {selected_chip_data['idx']}) has {selected_chip_data['inlier_count']} inliers, not 0. Proceeding with debug anyway.")
        else:
            print(f"  ERROR: Feature with idx {TARGET_IDX_VALUE} not found in the GeoJSON.")
            chips_to_process_gdf = gpd.GeoDataFrame([]) # Empty GeoDataFrame
    # Fallback to old TARGET_FEATURE_INDEX logic if TARGET_IDX_VALUE is not set or not found,
    # or remove this if TARGET_IDX_VALUE is the sole method for specific targeting.
    # For now, let's prioritize TARGET_IDX_VALUE and then fall back to 0-inlier search if not used.
    elif TARGET_FEATURE_INDEX is not None: # This block can be kept for legacy or removed
        print(f"Attempting to debug specific feature at index: {TARGET_FEATURE_INDEX} (TARGET_IDX_VALUE not set or not found)")
        if 0 <= TARGET_FEATURE_INDEX < len(stats_gdf):
            chips_to_process_gdf = stats_gdf.iloc[[TARGET_FEATURE_INDEX]]
            selected_chip_data = chips_to_process_gdf.iloc[0]
            if selected_chip_data['inlier_count'] > 0:
                print(f"  INFO: Chip at feature index {TARGET_FEATURE_INDEX} (original script idx: {selected_chip_data['idx']}) has {selected_chip_data['inlier_count']} inliers, not 0. Proceeding with debug anyway.")
        else:
            print(f"  ERROR: Feature index {TARGET_FEATURE_INDEX} is out of bounds (0 to {len(stats_gdf) - 1}).")
            chips_to_process_gdf = gpd.GeoDataFrame([])
    else:
        print("No specific feature targeted by idx or index. Looking for chips with 0 inliers...")
        failing_chips = stats_gdf[stats_gdf['inlier_count'] == 0]
        print(f"Found {len(failing_chips)} chips with 0 inliers.")
        if failing_chips.empty:
            print("No failing chips with 0 inliers found to debug.")
            chips_to_process_gdf = gpd.GeoDataFrame([]) # Empty GeoDataFrame
        else:
            # Debug the first few failing chips if no specific target
            num_chips_to_debug = min(3, len(failing_chips))
            chips_to_process_gdf = failing_chips.head(num_chips_to_debug)
            print(f"Will debug the first {len(chips_to_process_gdf)} of them.")


    if chips_to_process_gdf.empty:
        print("No chips selected for debugging.")
    else:
        for gdf_idx, chip_data_series in chips_to_process_gdf.iterrows():
            # gdf_idx is the index from the GeoDataFrame (which corresponds to TARGET_FEATURE_INDEX if set)
            # chip_data_series is the Series containing data for that row
            
            chip_original_script_idx = chip_data_series['idx'] # The 'idx' field from your stats file
            centroid_geom = chip_data_series.geometry
            
            print(f"\n--- Debugging Chip (Feature Index in GeoJSON: {gdf_idx}, Original Script Idx: {chip_original_script_idx}) ---")
            print(f"Centroid: {centroid_geom.x}, {centroid_geom.y}")
            print(f"Reported inlier_count in GeoJSON: {chip_data_series['inlier_count']}")

            # Create a GeoDataFrame for the single point for load_chips_for_debug
            point_gdf = gpd.GeoDataFrame({'geometry': [centroid_geom]}, crs=stats_gdf.crs)

            print("Loading chip image data...")
            try:
                (unreg_array, unreg_profile), (reg_array, reg_profile) = load_chips_for_debug(
                    point_gdf,
                    UNREG_RASTER_PATH,
                    REG_RASTER_PATH,
                    CHIP_WIDTH_CRS,
                    CHIP_HEIGHT_CRS,
                    CHIP_BUFFER_CRS
                )
                print(f"  Unregistered chip array shape: {unreg_array.shape}, dtype: {unreg_array.dtype}")
                print(f"  Registered chip array shape: {reg_array.shape}, dtype: {reg_array.dtype}")

                # You might want to save these arrays as images to inspect them visually
                # e.g., using rasterio to write them or matplotlib to show them
                # For example, to save the first band of the unregistered chip:
                # with rasterio.open(f'debug_unreg_chip_{chip_original_script_idx}_band0.tif', 'w', **unreg_profile) as dst:
                #     dst.write(unreg_array[0], 1)


                print("Running LoFTR analysis...")
                mkpts_reg, mkpts_unreg, inliers = debug_loftr_on_pair(
                    reg_array, # Typically img1 for LoFTR
                    unreg_array, # Typically img2 for LoFTR
                    LOFTR_DEVICE,
                    LOFTR_TARGET_SIZE,
                    LOFTR_REPROJ_THRESHOLDS
                )

                if mkpts_reg is not None:
                    print(f"  LoFTR analysis done. Total inliers: {np.sum(inliers) if inliers is not None else 'N/A'}")
                    # Here you can add more code to visualize matches if you have inliers,
                    # or inspect mkpts_reg[~inliers] and mkpts_unreg[~inliers] if you want to see outliers.
                    if inliers is not None and np.sum(inliers) > 0:
                        save_matches_visualization(
                            reg_array,     # Corresponds to img1_raw_array / mkpts0 in LoFTR / mkpts_reg
                            unreg_array,   # Corresponds to img2_raw_array / mkpts1 in LoFTR / mkpts_unreg
                            mkpts_reg,
                            mkpts_unreg,
                            inliers,
                            f"debug_chip_{chip_original_script_idx}"
                        )
                    elif inliers is not None:
                         print(f"  No inliers found for chip {chip_original_script_idx}, skipping match visualization.")

                else:
                    print("  LoFTR analysis skipped due to black image or other error.")

            except Exception as e:
                print(f"  ERROR processing chip {chip_original_script_idx}: {e}")
                import traceback
                traceback.print_exc()

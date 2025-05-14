#!/usr/bin/env python3
"""
Example script for chip-based raster registration using the tidysurvey.registration module.
"""
import os
from tidysurvey.registration import calculate_chip_crs_parameters_gdal, register_survey_by_chips

from tqdm import tqdm # For a consistent progress bar style if desired in example

def main():
    # --------------------------------------------------------------------------
    # User-defined constants for a specific registration run
    # --------------------------------------------------------------------------
    #UNREG_URL   = "https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/multispectral/front_country-batch_5-MS.tif"
    UNREG_URL   = "/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_test.tif"
    REG_URL     = "https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/front_country_2024.tif"
    OUTPUT_PATH = "/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_module_corrected.tif"

    # Processing parameters (configurable by user for the run)
    DEVICE      = 'mps'      # 'cpu', 'cuda', 'mps'
    MAX_WORKERS = 10       # For concurrent chip loading
    BATCH_SIZE  = 16       # LoFTR batch size
    CHUNK_SIZE  = 32       # Number of grid points to process in one full load-match-warp cycle
    TARGET_SIZE_PX = (480, 640) # Target (H, W) for LoFTR preprocessing & core chip pixel size
    BUFFER_FRAC = 0.1      # Buffer fraction around core chip area

    # --------------------------------------------------------------------------
    # Calculate chip CRS parameters
    # --------------------------------------------------------------------------

    print("⏳ Calculating chip CRS parameters...")
    try:
        core_w_crs, core_h_crs, buf_w_crs, buf_h_crs, pix_x, pix_y = calculate_chip_crs_parameters_gdal(
            parent_raster_path=UNREG_URL,
            target_chip_width_px=TARGET_SIZE_PX[1],  # Width from (H,W)
            target_chip_height_px=TARGET_SIZE_PX[0], # Height from (H,W)
            buffer_fraction_of_core=BUFFER_FRAC
        )
    except FileNotFoundError:
        print(f"Error: Unregistered raster not found at {UNREG_URL}. Cannot calculate parameters.")
        print("Please ensure the UNREG_URL is a valid path or URL accessible by rasterio.")
        return
    except Exception as e:
        print(f"Error calculating chip CRS parameters: {e}")
        return

    print("💡 Computed chip dimensions (CRS units):")
    print(f"    Core size: {core_w_crs:.2f}w × {core_h_crs:.2f}h")
    print(f"    Buffer to add (each side): {buf_w_crs:.2f}w × {buf_h_crs:.2f}h")
    print(f"    Pixel resolution: {pix_x:.4f} (x), {pix_y:.4f} (y)\n")

    # --------------------------------------------------------------------------
    # Run the registration
    # --------------------------------------------------------------------------
    print(f"🚀 Starting registration process...")
    print(f"   Output will be saved to: {OUTPUT_PATH}\n")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    try:
        register_survey_by_chips(
            unreg_survey_path=UNREG_URL,
            reg_reference_path=REG_URL,
            output_registered_survey_path=OUTPUT_PATH,
            core_chip_width_crs=core_w_crs,
            core_chip_height_crs=core_h_crs,
            buffer_width_crs=buf_w_crs,
            buffer_height_crs=buf_h_crs,
            device_for_loftr=DEVICE,
            max_loader_workers=MAX_WORKERS,
            loftr_batch_size=BATCH_SIZE,
            processing_chunk_size=CHUNK_SIZE,
            target_size_hw_for_loftr_preprocessing=TARGET_SIZE_PX,
        )
        print(f"✅ Registration pipeline finished. Output saved to: {OUTPUT_PATH}")
    except FileNotFoundError as e:
        print(f"Error during registration: A file was not found. {e}")
        print("Please check UNREG_URL and REG_URL.")
    except Exception as e:
        print(f"An error occurred during the registration pipeline: {e}")

if __name__ == "__main__":
    main() 
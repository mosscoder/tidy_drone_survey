#!/usr/bin/env python3
"""
Example script for chip-based raster registration using the tidysurvey.registration module.
"""
import os
from tidysurvey.registration import register_survey_by_chips

def main():
    # --------------------------------------------------------------------------
    # User-defined constants for a specific registration run
    # --------------------------------------------------------------------------
    #UNREG_URL   = "/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_test.tif"
    UNREG_URL   = "https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/multispectral/front_country-batch_5-MS.tif"
    REG_URL     = "https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/front_country_2024.tif"
    OUTPUT_PATH = "/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_corrected.tif"
    STATS_PATH = "/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_stats.tif"
    #DEBUG_WARPED_CHIPS_DIR = "/Users/kdoherty/tidy_drone_survey/data/debug_warped_chips" # Directory to save individual warped chips

    # Processing parameters (configurable by user for the run)
    DEVICE      = 'mps'      # 'cpu', 'cuda', 'mps'
    MAX_WORKERS = 10       # For concurrent chip loading
    BATCH_SIZE  = 16       # LoFTR batch size
    CHUNK_SIZE  = 32       # Number of grid points to process in one full load-match-warp cycle
    
    # --------------------------------------------------------------------------
    # Run the registration
    # --------------------------------------------------------------------------
    print(f"   Starting registration process...")
    print(f"   Output will be saved to: {OUTPUT_PATH}\n")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    try:
        register_survey_by_chips(
            unreg_survey_path=UNREG_URL,
            reg_reference_path=REG_URL,
            output_registered_survey_path=OUTPUT_PATH,
            output_stats_raster_path=STATS_PATH,
            device_for_loftr=DEVICE,
            max_loader_workers=MAX_WORKERS,
            loftr_batch_size=BATCH_SIZE,
            processing_chunk_size=CHUNK_SIZE,
            #debug_output_dir_for_warped_chips=DEBUG_WARPED_CHIPS_DIR # Pass the debug directory
        )

        print(f"Registration pipeline finished. Output saved to: {OUTPUT_PATH}")
    except FileNotFoundError as e:
        print(f"Error during registration: A file was not found. {e}")
        print("Please check UNREG_URL and REG_URL.")
    except Exception as e:
        print(f"An error occurred during the registration pipeline: {e}")

if __name__ == "__main__":
    main() 
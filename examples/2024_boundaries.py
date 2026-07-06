#!/usr/bin/env python3
"""
Generate non-overlapping mission boundaries for 2024 front country visible light data.

This script:
1. Scans the GCS bucket for all 2024 visible light orthomosaics
2. Extracts mission names from filenames (e.g., 'batch_5')
3. Generates non-overlapping boundaries using distance transforms
4. Outputs a combined GeoJSON with one feature per mission
"""

import subprocess
import re
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tidysurvey import generate_combined_boundaries


# 2024-specific configuration
GCS_BUCKET = "gs://mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/visible/"
OUTPUT_GEOJSON = "data/vector/2024_mission_boundaries.geojson"

# Processing parameters
TOL = 0.5  # Boundary precision (meters)
N_WORKERS = 8  # Parallel workers


def list_rasters_from_gcs(bucket_path: str) -> list:
    """List all .tif files from a GCS bucket."""
    print(f"Scanning GCS bucket: {bucket_path}")
    result = subprocess.run(
        ["gsutil", "ls", bucket_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"gsutil failed: {result.stderr}")

    raster_paths = [
        line.strip() for line in result.stdout.strip().split('\n')
        if line.strip().endswith('.tif')
    ]

    print(f"Found {len(raster_paths)} rasters")
    return raster_paths


def extract_mission_name(raster_path: str) -> str:
    """Extract mission name from 2024 filename pattern."""
    # Pattern: front_country-batch_5.tif -> batch_5
    filename = os.path.basename(raster_path)
    match = re.search(r'batch_(\d+)', filename)
    if match:
        return f"batch_{match.group(1)}"
    # Fallback to filename without extension
    return os.path.splitext(filename)[0]


def main():
    # Step 1: Discover rasters from GCS
    raster_paths = list_rasters_from_gcs(GCS_BUCKET)

    if not raster_paths:
        print("No rasters found. Exiting.")
        return

    # Step 2: Extract mission names
    mission_names = [extract_mission_name(p) for p in raster_paths]

    print("\nMissions found:")
    for name, path in zip(mission_names, raster_paths):
        print(f"  {name}: {path}")

    # Step 3: Generate combined boundaries
    print(f"\nGenerating boundaries with tol={TOL}m...")
    output = generate_combined_boundaries(
        raster_paths=raster_paths,
        mission_names=mission_names,
        output_geojson=OUTPUT_GEOJSON,
        tol=TOL,
        n_workers=N_WORKERS
    )

    print(f"\nDone! Output saved to: {output}")


if __name__ == "__main__":
    main()

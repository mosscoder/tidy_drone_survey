import os
from tidysurvey.merge import meta_mosaic

# Parameters
n_workers = 10
tol = 1.0

output_final_cog = 'data/raster/batch_5&10.tif'
# Ensure the output directory for the final COG exists
os.makedirs(os.path.dirname(output_final_cog), exist_ok=True)

# Define input orthomosaics (URLs or local paths)
# Using the example URLs from the original script
base_url = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/multispectral/front_country-batch_{BATCH}-MS.tif'
input_orthos = [
    base_url.format(BATCH=batch_num) for batch_num in [5, 10] # Example for 2 batches
]

print(f"Starting mosaic pipeline for {len(input_orthos)} orthomosaics.")
print(f"Output will be saved to: {output_final_cog}")

# Call the centralized meta_mosaic function
final_mosaic_path = meta_mosaic(
    orthos_in=input_orthos, 
    output_path=output_final_cog,
    keep_bands=[1, 2, 3, 4], # Example: keeping the first four bands (e.g., R, G, B, NIR)
    intermediary_dir=None, 
    tol=tol,
    n_workers=n_workers
)

print(f"Pipeline completed. Final mosaic saved to: {final_mosaic_path}")
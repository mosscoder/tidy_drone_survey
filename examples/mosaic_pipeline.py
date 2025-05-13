import os
import shutil
from tidysurvey import define_hull, find_seamlines, clip, mosaic
import joblib

n_workers = 10
tol = 1.0
tmp_dir = 'tempdir'
output_final_cog = 'data/raster/front_country_2024_ms.tif'
os.makedirs(tmp_dir, exist_ok=True)

url = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/multispectral/front_country-batch_{BATCH}-MS.tif'

def extract_hull_batch(batch_num):
    input_geotiff = url.format(BATCH=batch_num)
    output_hull_geojson = os.path.join(tmp_dir, os.path.basename(input_geotiff).replace('.tif', '.geojson'))
 
    file_in, file_out = define_hull(
        geotiff_path_in=input_geotiff,
        hull_geojson_path=output_hull_geojson,
        tol=tol
    )

    return file_in, file_out

results = joblib.Parallel(n_jobs=n_workers)(
    joblib.delayed(extract_hull_batch)(batch_num) for batch_num in range(1, 11)
)

orthos_in = [r[0] for r in results]
hulls_out = [r[1] for r in results]

output_seamlines_geojson = os.path.join(tmp_dir, f'seamlines.geojson')

seamline_files = find_seamlines(
    input_hulls=hulls_out,
    output_dir=tmp_dir,
    tol=tol
)

print('Input orthomosaics:')
for f in orthos_in:
    print(f)

print('Seamlines:')
for seamline_file in seamline_files:
    print(seamline_file)

def process_orthomosaics(orthos_in, seamline_files, output_final_cog, keep_bands=None):

    def _clip_single_raster(ortho, seamline_geojson, output_raster, keep_bands):
        return clip(ortho, seamline_geojson, output_raster, keep_bands=keep_bands)

    cut_rasters = joblib.Parallel(n_jobs=n_workers)(
        joblib.delayed(_clip_single_raster)(
            ortho,
            seamline_files[i],
            os.path.join(tmp_dir, f"clipped_{i+1}.tif"),
            keep_bands
        )
        for i, ortho in enumerate(orthos_in)
    )

    mosaic(cut_rasters, output_final_cog, n_jobs=n_workers-1)

process_orthomosaics(orthos_in, seamline_files, output_final_cog, keep_bands=[1,2,3,4])

shutil.rmtree(tmp_dir)
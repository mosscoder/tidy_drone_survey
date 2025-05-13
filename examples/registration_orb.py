import rasterio
import numpy as np
import cv2
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
import os
import rasterio
from osgeo import gdal
import multiprocessing
from functools import partial
import itertools
from tqdm import tqdm
import glob
import psutil
import subprocess
def register_ms_wrt_rgb(rgb_path, ms_path, output_dir, x, y, side_length=1024, buffer_frac=0.1, drop_bands=None):
    buffer_size = int(side_length * buffer_frac)
    buffered_side_length = side_length + 2 * buffer_size

    ms_output_path = os.path.join(output_dir, f'registered_ms_window_{round(x)}_{round(y)}.tif')
    if os.path.exists(ms_output_path):
        print(f'Skipping tile {x}_{y} due to already registered.')
        return ms_output_path

    with rasterio.open(rgb_path) as rgb_src, rasterio.open(ms_path) as ms_src:
        # Calculate the window extent in the RGB raster
        rgb_row, rgb_col = rgb_src.index(x, y)
        half_side = buffered_side_length // 2
        rgb_window = Window(rgb_col - half_side, rgb_row - half_side, buffered_side_length, buffered_side_length)

        # Read the RGB window and extend if necessary
        rgb_data = rgb_src.read(window=rgb_window, boundless=True)
        rgb_transform = rgb_src.window_transform(rgb_window)

        # Calculate the corresponding window in the MS raster
        ms_bounds = rasterio.windows.bounds(rgb_window, rgb_src.transform)
        ms_window = rasterio.windows.from_bounds(*ms_bounds, ms_src.transform)

        # Read and resample the MS window to match RGB resolution
        ms_data = ms_src.read(window=ms_window, boundless=True)

        ms_resampled = np.zeros((ms_data.shape[0], buffered_side_length, buffered_side_length), dtype=ms_data.dtype)
        reproject(
            ms_data,
            ms_resampled,
            src_transform=ms_src.window_transform(ms_window),
            src_crs=ms_src.crs,
            dst_transform=rgb_transform,
            dst_crs=rgb_src.crs,
            resampling=Resampling.bilinear
        )

    ms_resampled[-1] = rgb_data[-1]

    unbuffered_transform = rasterio.Affine(rgb_transform.a, rgb_transform.b, rgb_transform.c + buffer_size * rgb_transform.a,
                                    rgb_transform.d, rgb_transform.e, rgb_transform.f + buffer_size * rgb_transform.e)

    # Check if all RGB data is masked
    unreg_output_path = os.path.join(output_dir, f'unregistered_ms_window_{round(x)}_{round(y)}.tif')
    #if np.all(rgb_data[-1] == 0):
    if np.count_nonzero(rgb_data[-1] == 0) >= 0.90 * rgb_data[-1].size:
      print(f'Insufficient visible data at {round(x)}_{round(y)}, skipping.')
      return unreg_output_path

    # Prepare images for ORB
    rgb_gray = cv2.equalizeHist(cv2.cvtColor(rgb_data.transpose(1, 2, 0)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY))
    ms_preprocessed = cv2.equalizeHist(cv2.cvtColor(ms_resampled.transpose(1, 2, 0)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY))

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=int(1e4))

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(rgb_gray, None)
    kp2, des2 = orb.detectAndCompute(ms_preprocessed, None)

    print(f"Number of keypoints in RGB: {len(kp1)}")
    print(f"Number of keypoints in MS: {len(kp2)}")

    # Calculate adaptive min_matches
    min_matches = 10

    # Initialize parameters for iterative matching
    distance_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    good_matches = []
    for distance_threshold in distance_thresholds:
        try:
            # Match descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Apply distance threshold
            good_matches = [m for m in matches if m.distance < distance_threshold * max(m.distance for m in matches)]
            print(f"Matches found with threshold {distance_threshold}: {len(good_matches)}")

            if len(good_matches) >= min_matches:
                break
        except cv2.error as e:
            print(f"Matching failed with threshold {distance_threshold}: {str(e)}")
            continue  # Try the next distance threshold

    if len(good_matches) < min_matches:
        ms_data = ms_resampled[:, buffer_size:-buffer_size, buffer_size:-buffer_size]
        save_geotiff(unreg_output_path, ms_data, unbuffered_transform, rgb_src.crs, drop_bands)
        return print(f"Not enough good matches found. Best attempt: {len(good_matches)} matches. Required: {min_matches}. Saving unregistered MS data at {round(x)}_{round(y)}.")

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if M is None or M.shape != (3, 3):
        ms_data = ms_data[:, buffer_size:-buffer_size, buffer_size:-buffer_size]
        save_geotiff(unreg_output_path, ms_data, unbuffered_transform, rgb_src.crs, drop_bands)
        return print(f"Failed to find valid homography. Saving unregistered MS data at {round(x)}_{round(y)}.")

    # Ensure M is float32
    M = M.astype(np.float32)

    # Apply homography to register MS image
    registered_ms = cv2.warpPerspective(ms_resampled.transpose(1, 2, 0), M, (buffered_side_length, buffered_side_length))
    registered_ms = registered_ms.transpose(2, 0, 1)

    # Remove buffer
    registered_ms = registered_ms[:, buffer_size:-buffer_size, buffer_size:-buffer_size]

    # Save registered MS window if output directory is provided
    if output_dir:
        save_geotiff(ms_output_path, registered_ms, unbuffered_transform, rgb_src.crs, drop_bands)

    return print(f'Tile {os.path.basename(ms_output_path)} registered and saved.')

def save_geotiff(output_path, data, transform, crs, drop_bands):
    """
    Save a numpy array as a GeoTIFF file.

    Args:
    output_path (str): Path to save the GeoTIFF file.
    data (numpy.ndarray): The image data to save.
    transform (affine.Affine): The affine transform of the image.
    crs: The coordinate reference system of the image.
    drop_bands (list): A list of bands to drop from the data.
    """
    if drop_bands is not None:
        # Convert 1-indexed band numbers to 0-indexed
        drop_bands_zero_indexed = [band - 1 for band in drop_bands]

        # Create a list of bands to keep, excluding the last band (alpha mask)
        keep_bands = [i for i in range(data.shape[0] - 1) if i not in drop_bands_zero_indexed]

        # Always keep the last band (alpha mask)
        keep_bands.append(data.shape[0] - 1)

        data = data[keep_bands]

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

def add_overviews(geotiff_path):
    # Open the dataset
    dataset = gdal.Open(geotiff_path, gdal.GA_Update)

    if dataset is None:
        print(f"Unable to open {geotiff_path}")
        return

    # Define the overview levels
    overview_levels = [2, 4, 8, 16, 32]
    # Add overviews
    dataset.BuildOverviews("average", overview_levels)

    # Close the dataset
    dataset = None
    print(f"Overviews added successfully to {geotiff_path}")

def add_color_schema(geotiff_path):
    # Open the raster file
    ds = gdal.Open(geotiff_path, gdal.GA_Update)

    # Get the number of bands
    band_count = ds.RasterCount

    # Set color interpretation for each band
    for i in range(1, band_count + 1):
        band = ds.GetRasterBand(i)

        if i == 1:
            band.SetColorInterpretation(gdal.GCI_RedBand)
        elif i == 2:
            band.SetColorInterpretation(gdal.GCI_GreenBand)
        elif i == 3:
            band.SetColorInterpretation(gdal.GCI_BlueBand)
        elif i == band_count:
            band.SetColorInterpretation(gdal.GCI_AlphaBand)
        else:
            band.SetColorInterpretation(gdal.GCI_Undefined)

    # Close the dataset
    ds = None

def process_tile(args):
    func, x, y = args
    return func(x, y)

def register_orthomosaic(rgb_path, ms_path, output_dir, side_length=1024, buffer_frac=0.1, drop_bands=None):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(rgb_path) as rgb_src:
        rgb_bounds = rgb_src.bounds

        rgb_width, rgb_height = rgb_src.width, rgb_src.height

        # Generate tile coordinates
        x_coords = np.arange(rgb_bounds.left, rgb_bounds.right, side_length * rgb_src.transform.a)
        y_coords = np.arange(rgb_bounds.bottom, rgb_bounds.top, -side_length * rgb_src.transform.e)

        tiles = list(itertools.product(x_coords, y_coords))
        print(f'Number of tiles: {len(tiles)}')

        # Set up multiprocessing
        num_cores = multiprocessing.cpu_count() - 1
        print(f"Using {num_cores} cores for multiprocessing.")
        pool = multiprocessing.Pool(processes=num_cores)

        # Prepare the partial function for multiprocessing
        register_ms_wrt_rgb_partial = partial(register_ms_wrt_rgb, rgb_path, ms_path, output_dir,
                                              side_length=side_length, buffer_frac=buffer_frac,
                                              drop_bands=drop_bands)

        # Prepare arguments for process_tile
        args = [(register_ms_wrt_rgb_partial, x, y) for x, y in tiles]

        # Process tiles in parallel with progress bar
        with tqdm(total=len(tiles), desc="Processing tiles") as pbar:
            for _ in pool.imap_unordered(process_tile, args):
                pbar.update()

        pool.close()
        pool.join()

        #Glob all tif files in the output directory
        tif_files = glob.glob(os.path.join(output_dir, '*.tif'))

        if not tif_files:
            print("No valid tiles were processed.")
            return

        #Create VRT
        vrt_path = os.path.join(output_dir, 'merged_orthomosaic.vrt')
        gdal.BuildVRT(vrt_path, tif_files)

        # Translate VRT to Cloud Optimized GeoTIFF
        cog_path = os.path.join(output_dir, 'orthomosaic_cog.tif')

        # Calculate 75% of the total RAM in MB
        total_ram = psutil.virtual_memory().total  # Total RAM in bytes
        gdal_cachemax = int(total_ram * 0.75 / (1024 * 1024))  # Convert to MB

        gdal.SetConfigOption('GDAL_CACHEMAX', str(gdal_cachemax))
        gdal.Translate(cog_path, vrt_path, format='COG', creationOptions=[
            'COMPRESS=DEFLATE',
            'PREDICTOR=2',
            'ZLEVEL=5',
            'TILED=YES',
            'BLOCKSIZE=256',
            'BIGTIFF=IF_SAFER',
            'RESAMPLING=AVERAGE',
            'NUM_THREADS=ALL_CPUS'
        ])

        add_color_schema(cog_path)
        add_overviews(cog_path)
        print(f"Orthomosaic processing complete. Output saved to {cog_path}")

rgb_url = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/240606_uppersheepcamp/processing/dronedeploy/240606_uppersheepcamp-visible.tif'
ms_url = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/240606_uppersheepcamp/processing/dronedeploy/240606_uppersheepcamp-multispectral.tif'

output_dir = '/Users/kdoherty/tidy_drone_survey/data/raster/orb'
os.makedirs(output_dir, exist_ok=True)
drop_bands = [5] #five is a redundant NIR
register_orthomosaic(rgb_url, ms_url, output_dir, side_length=1024, buffer_frac=0.1, drop_bands=drop_bands)
subprocess.run(['gdalinfo', output_dir + '/orthomosaic_cog.tif'])
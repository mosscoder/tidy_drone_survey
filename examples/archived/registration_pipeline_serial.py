import os

import cv2
import geopandas as gpd
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from IPython.display import Image
from osgeo import gdal
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds, transform as window_transform
from shapely.geometry import Point

from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from tqdm import tqdm


def get_bbox_bounds(point_gdf, side_length):
    """
    Given a GeoDataFrame with a single point geometry, return (xmin, xmax, ymin, ymax)
    for a square bounding box of given side_length centered at the point.
    """
    half_size = side_length / 2
    x = point_gdf.geometry.iloc[0].x
    y = point_gdf.geometry.iloc[0].y
    xmin = x - half_size
    xmax = x + half_size
    ymin = y - half_size
    ymax = y + half_size
    return xmin, ymin, xmax, ymax

def load_chips(
    point_gdf,
    unregistered_path,
    registered_path,
    side_length
) -> tuple:
    """
    Read matching window chips from an unregistered and a registered raster,
    storing pixel arrays, profiles, and windows for GCP generation.

    Args:
        point_gdf (GeoDataFrame): single-point geometry in the SRC of unregistered image.
        unregistered_path (str): path/URL to the poorly-referenced raster.
        registered_path (str): path/URL to the well-referenced raster.
        side_length (float): length of the square side (in CRS units) around the point.

    Returns:
        tuple: (un_chip, reg_chip), each a Chip(array, profile, window).
    """
    # Determine bounding box around the point
    bbox = get_bbox_bounds(point_gdf, side_length)

    # Unregistered chip
    with rasterio.open(unregistered_path) as un_ds:
        un_window = from_bounds(*bbox, un_ds.transform)
        un_array = un_ds.read(window=un_window)
        un_profile = un_ds.profile
        un_crs = un_ds.crs

    un_chip = dict(array=un_array, profile=un_profile, window=un_window)

    # Registered chip: warp to unregistered CRS
    with rasterio.open(registered_path) as reg_ds:
        with WarpedVRT(
            reg_ds,
            crs=un_crs,
            resampling=Resampling.bilinear
        ) as vrt:
            reg_window = from_bounds(*bbox, vrt.transform)
            reg_array = vrt.read(window=reg_window)
            reg_profile = vrt.profile

    reg_chip = dict(array=reg_array, profile=reg_profile, window=reg_window)

    return un_chip, reg_chip

def rasterio_to_torch_tensor(numpy_array: np.ndarray, bands: int = [0, 1, 2]) -> torch.Tensor:
    """
    Converts a rasterio NumPy array from rasterio window output (C X H x W)
    to a PyTorch tensor of shape (1, C, H, W).

    Args:
        numpy_array: Input NumPy array with shape (C, H, W).

    Returns:
        PyTorch tensor with shape (1, C, H, W).
    """
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(numpy_array[bands]).unsqueeze(0)

    return tensor

def get_loftr_matches(img1, img2, device: str = 'cpu'):
    """
    Finds keypoint matches between two images using LoFTR.

    Returns empty arrays if either image is all black.
    Prints status messages throughout.
    """
    # 0) check for all-black images
    if np.all(img1 == 0):
        print("Skipping LoFTR: image1 is all-black.")
        empty_pts = np.empty((0, 2), dtype=float)
        empty_inliers = np.empty((0,), dtype=bool)
        return empty_pts, empty_pts, empty_inliers

    if np.all(img2 == 0):
        print("Skipping LoFTR: image2 is all-black.")
        empty_pts = np.empty((0, 2), dtype=float)
        empty_inliers = np.empty((0,), dtype=bool)
        return empty_pts, empty_pts, empty_inliers

    # 1) load & cast to float, normalize to [0,1]
    img1_tensor = rasterio_to_torch_tensor(img1).float().div(255.0).to(device)
    img2_tensor = rasterio_to_torch_tensor(img2).float().div(255.0).to(device)

    # 2) Convert to grayscale
    img1_gray = K.color.rgb_to_grayscale(img1_tensor)
    img2_gray = K.color.rgb_to_grayscale(img2_tensor)

    # 3) LoFTR matching
    matcher = KF.LoFTR(pretrained='outdoor').to(device)
    input_dict = {"image0": img1_gray, "image1": img2_gray}
    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()

    # 4) Fundamental matrix inlier filtering
    if mkpts0.shape[0] >= 7:
        Fm, inliers_cv = cv2.findFundamentalMat(
            mkpts0, mkpts1,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=0.5,
            confidence=0.999,
            maxIters=100000
        )
        inliers = inliers_cv.ravel() > 0
        print(f"LoFTR: found {len(mkpts0)} keypoints, {inliers.sum()} inliers.")
    else:
        print(f"Warning: Not enough keypoints to compute Fundamental Matrix (found {mkpts0.shape[0]}).")
        inliers = np.zeros(mkpts0.shape[0], dtype=bool)
        print("Inliers count: 0")

    return mkpts0, mkpts1, inliers

def generate_gcps(un_chip: dict,
                  reg_chip: dict,
                  device: str = 'cpu') -> list:
    """
    Generate GDAL GCPs from inlier LoFTR matches between a poorly-referenced chip
    (un_chip) and a well-referenced chip (reg_chip).

    Only matches marked True in the inliers mask get turned into GCPs.

    Args:
        un_chip: dict with keys 'array', 'profile', 'window'
        reg_chip: same for the registered chip
        device: torch device for LoFTR matching

    Returns:
        List of osgeo.gdal.GCP objects with pixel/line in the full unregistered image
        and X/Y from the registered image. Returns an empty list if no reliable matches.
    """
    # 1) get all matches + boolean mask
    mkpts_reg, mkpts_un, inliers = get_loftr_matches(
        reg_chip['array'],
        un_chip['array'],
        device
    )
    mkpts_reg = np.asarray(mkpts_reg)
    mkpts_un  = np.asarray(mkpts_un)
    inliers   = np.asarray(inliers)

    # 2) if no matches at all, or no inliers, bail out early
    if mkpts_reg.size == 0 or mkpts_un.size == 0 or not np.any(inliers):
        # Optionally log or warn here
        print("No valid GCPs could be generated (no inliers).")
        return []

    # 3) pick only the inlier points
    reg_pts = mkpts_reg[inliers]
    un_pts  = mkpts_un[inliers]

    # 4) prepare transforms & offsets
    tf_reg     = window_transform(reg_chip['window'], reg_chip['profile']['transform'])
    col_off_un = un_chip['window'].col_off
    row_off_un = un_chip['window'].row_off

    # 5) build GCP list
    gcps: list = []
    for (cx, ry), (ux, uy) in zip(reg_pts, un_pts):
        geo_x, geo_y = tf_reg * (float(cx), float(ry))
        pixel = float(ux + col_off_un)
        line  = float(uy + row_off_un)
        gcps.append(gdal.GCP(geo_x, geo_y, 0.0, pixel, line))

    return gcps

def generate_grid_points(raster_path: str,
                         side_length: float,
                         overlap_fraction: float = 0.1) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame of points that tile the raster with the given side_length
    and fractional overlap.
    """
    half = side_length / 2.0
    stride = side_length * (1 - overlap_fraction)

    with rasterio.open(raster_path) as ds:
        left, bottom, right, top = ds.bounds
        crs = ds.crs

    # x: left+half ... right-half, y: bottom+half ... top-half
    xs = np.arange(left + half, right - half + 1e-8, stride)
    ys = np.arange(bottom + half, top - half + 1e-8, stride)

    pts = [Point(x, y) for y in ys for x in xs]
    return gpd.GeoDataFrame({'geometry': pts}, crs=crs)

def collect_all_gcps(unreg_path: str,
                     reg_path: str,
                     side_length: float,
                     overlap_fraction: float = 0.1,
                     device: str = 'cpu',
                     max_workers: int = 4) -> list:
    """
    Tile the unregistered raster, generate chips and GCPs for each tile in parallel,
    and show a tqdm progress bar.
    """
    # 1) build grid of points
    grid_gdf = generate_grid_points(unreg_path, side_length, overlap_fraction)
    n_tiles = len(grid_gdf)

    all_gcps = []

    def process_point(pt):
        pt_gdf = gpd.GeoDataFrame({'geometry': [pt]}, crs=grid_gdf.crs)
        un_chip, reg_chip = load_chips(pt_gdf, unreg_path, reg_path, side_length)
        return generate_gcps(un_chip, reg_chip, device)

    # 2) submit all jobs
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(process_point, pt) for pt in grid_gdf.geometry]

        # 3) iterate as they complete, updating progress
        for future in tqdm(as_completed(futures),
                           total=n_tiles,
                           desc="Tiles processed",
                           unit="tile"):
            gcp_list = future.result()
            if gcp_list:
                all_gcps.extend(gcp_list)

    return all_gcps

def reproject_with_gcps(unreg_path: str,
                        out_tif: str,
                        gcps: list,
                        polynomial_order: int = 3,
                        resampling: str = "cubic") -> None:
    # 1) Read the source CRS
    src_ds = gdal.Open(unreg_path)
    crs_wkt = src_ds.GetProjection() or src_ds.GetProjectionRef()
    src_ds = None
    if not crs_wkt:
        raise ValueError(f"No CRS found in {unreg_path}")

    # 2) Create an in‐memory VRT with the GCPs attached
    vrt_ds = gdal.Translate(
        "",  # empty string means 'in memory'
        unreg_path,
        options=gdal.TranslateOptions(format="VRT", GCPs=gcps)
    )

    gdal.UseExceptions()

    warp_opts = gdal.WarpOptions(
        format="GTiff",
        polynomialOrder=polynomial_order,
        resampleAlg=resampling,
        dstSRS=crs_wkt,
    )
    gdal.Warp(out_tif, vrt_ds, options=warp_opts)

    # 4) Clean up
    vrt_ds = None

device = 'mps'
max_workers = 8

#unreg_path = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/multispectral/front_country-batch_5-MS.tif'
unreg_path = '/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_test.tif'
reg_path = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/2024_front_country/processing/dronedeploy/front_country_2024.tif'
out_tif = '/Users/kdoherty/tidy_drone_survey/data/raster/batch_5_test_corrected.tif'

side_length = 10
overlap_fraction = 0.05

gcps = collect_all_gcps(unreg_path, reg_path, side_length, overlap_fraction, device, max_workers)

print(f"Found {len(gcps)} GCPs")

reproject_with_gcps(
    unreg_path=unreg_path,
    out_tif  =out_tif,
    gcps     =gcps,             # your list of osgeo.gdal.GCP
    polynomial_order=3,         # same as "-order 3"
    resampling="cubic"          # same as "-r cubic"
)
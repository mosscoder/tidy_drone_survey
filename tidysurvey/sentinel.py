"""
Sentinel-2 band download utilities for survey calibration.

Downloads Sentinel-2 bands from Google Earth Engine, matched to survey dates.
"""
import json
import os
from datetime import datetime, timedelta

import ee
import rasterio
import requests
from google.oauth2 import service_account
from pyproj import CRS, Transformer


def download_sentinel2_bands(
    bounds_raster: str,
    target_date: str,
    output_path: str,
    bands: list[str],
    gee_credentials_path: str,
    band_names: list[str] = None,
    buffer_days: int = 7,
    output_resolution: int = 10,
    collection: str = 'COPERNICUS/S2_SR_HARMONIZED',
) -> str:
    """
    Download Sentinel-2 bands for a survey area.

    Finds the nearest Sentinel-2 image to the target date and downloads
    the specified bands, reprojected to match the input raster's CRS.
    Applies pixel-level cloud masking using QA60 band.

    Args:
        bounds_raster: Path to raster for extracting bounds and CRS
        target_date: Target date as 'YYYY-MM-DD' string
        output_path: Output path for downloaded GeoTIFF.
            If path contains '{date}', it will be replaced with actual
            S2 capture date (YYMMDD format).
        bands: List of Sentinel-2 band IDs (e.g., ['B4', 'B3', 'B8', 'B5'])
        gee_credentials_path: Path to GEE service account credentials JSON
        band_names: Optional list of output band names. Defaults to band IDs.
        buffer_days: Search window +/- days around target date
        output_resolution: Output resolution in meters (default 10m)
        collection: Earth Engine collection ID

    Returns:
        Path to downloaded file (with actual date if {date} placeholder used)

    Raises:
        ValueError: If no images found in search window
        FileNotFoundError: If bounds_raster or credentials not found
    """
    if not bands:
        raise ValueError("bands list cannot be empty")

    if band_names is None:
        band_names = bands.copy()
    elif len(band_names) != len(bands):
        raise ValueError("band_names must have same length as bands")

    if not gee_credentials_path or not os.path.exists(gee_credentials_path):
        raise FileNotFoundError(f"GEE credentials not found: {gee_credentials_path}")

    # Extract bounds and CRS from reference raster
    print(f"Reading bounds from: {bounds_raster}")
    with rasterio.open(bounds_raster) as src:
        bounds = src.bounds
        crs = src.crs
        export_epsg = crs.to_epsg()

    print(f"  Bounds: {bounds}")
    print(f"  CRS: EPSG:{export_epsg}")

    # Initialize Earth Engine
    print("Initializing Earth Engine...")
    scopes = ['https://www.googleapis.com/auth/earthengine']
    credentials = service_account.Credentials.from_service_account_file(
        gee_credentials_path,
        scopes=scopes
    )
    ee.Initialize(credentials)

    # Transform bounds to WGS84 for Earth Engine
    transformer = Transformer.from_crs(
        CRS.from_epsg(export_epsg),
        CRS.from_epsg(4326),
        always_xy=True
    )
    left_lon, top_lat = transformer.transform(bounds.left, bounds.top)
    right_lon, _ = transformer.transform(bounds.right, bounds.top)
    _, bottom_lat = transformer.transform(bounds.right, bounds.bottom)

    region = [
        [left_lon, top_lat],
        [right_lon, top_lat],
        [right_lon, bottom_lat],
        [left_lon, bottom_lat],
        [left_lon, top_lat]
    ]
    ee_region = ee.Geometry.Polygon([region])

    # Search for images (no scene-level cloud filter)
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    start_date = (target_dt - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    end_date = (target_dt + timedelta(days=buffer_days)).strftime('%Y-%m-%d')

    print(f"Searching for Sentinel-2 images from {start_date} to {end_date}")

    image_collection = ee.ImageCollection(collection) \
        .filterDate(start_date, end_date) \
        .filterBounds(ee_region)

    collection_size = image_collection.size().getInfo()
    print(f"  Found {collection_size} images")

    if collection_size == 0:
        raise ValueError(
            f"No Sentinel-2 images found between {start_date} and {end_date}. "
            "Try increasing buffer_days."
        )

    # List available images
    image_list = image_collection.getInfo()['features']
    print("Available images:")
    for img in image_list:
        img_date = datetime.fromtimestamp(
            img['properties']['system:time_start'] / 1000
        ).strftime('%Y-%m-%d')
        scene_cloud = img['properties'].get('CLOUDY_PIXEL_PERCENTAGE', 'N/A')
        if isinstance(scene_cloud, (int, float)):
            print(f"  - {img_date} (scene cloud: {scene_cloud:.1f}%)")
        else:
            print(f"  - {img_date}")

    # Find nearest image to target date
    ee_target_date = ee.Date(target_date)

    def add_days_diff(img):
        diff = ee.Number(img.date().difference(ee_target_date, 'day')).abs()
        return img.set('days_diff', diff)

    image_collection = image_collection.map(add_days_diff)
    image = image_collection.sort('days_diff').first()

    # Get selected image info
    image_info = image.getInfo()
    selected_date = datetime.fromtimestamp(
        image_info['properties']['system:time_start'] / 1000
    )
    selected_date_str = selected_date.strftime('%Y-%m-%d')
    date_yymmdd = selected_date.strftime('%y%m%d')
    days_from_target = abs((selected_date.date() - target_dt.date()).days)

    print(f"Selected image: {selected_date_str} ({days_from_target} days from target)")

    # Calculate AOI cloud percentage using QA60
    qa = image.select('QA60')
    clouds = qa.bitwiseAnd(1 << 10)  # Cloud bit
    cirrus = qa.bitwiseAnd(1 << 11)  # Cirrus bit
    cloud_mask = clouds.Or(cirrus)

    # Calculate cloud percentage within AOI
    cloud_stats = cloud_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee_region,
        scale=60,
        maxPixels=1e9
    )
    aoi_cloud_pct = cloud_stats.get('QA60').getInfo()
    if aoi_cloud_pct is not None:
        aoi_cloud_pct = aoi_cloud_pct * 100
    else:
        aoi_cloud_pct = 0.0

    print(f"  AOI cloud cover: {aoi_cloud_pct:.1f}%")

    # Build output path with date
    final_output_path = output_path.replace('{date}', date_yymmdd)
    print(f"Output: {final_output_path}")

    # Select and stack bands
    print(f"Selecting bands: {bands}")
    band_images = [image.select(band) for band in bands]
    result = band_images[0]
    for band_img in band_images[1:]:
        result = result.addBands(band_img)

    # Rename bands
    result = result.rename(band_names)

    # Apply cloud mask (cloudy pixels become nodata)
    result = result.updateMask(cloud_mask.Not())

    # Reproject to target CRS
    result = result.reproject(crs=f'EPSG:{export_epsg}', scale=output_resolution)

    # Convert to int16
    result = result.toInt16()

    # Download
    print(f"Requesting download (EPSG:{export_epsg}, {output_resolution}m)...")
    url = result.getDownloadURL({
        'scale': output_resolution,
        'crs': f'EPSG:{export_epsg}',
        'region': region,
        'format': 'GEO_TIFF'
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    print("Downloading...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Download failed with status {response.status_code}")

    with open(final_output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Update metadata
    print("Updating metadata...")
    with rasterio.open(final_output_path, 'r+') as dst:
        dst.nodata = -32768
        for i, name in enumerate(band_names, 1):
            dst.set_band_description(i, name)

    # Write JSON sidecar with stats
    json_path = os.path.splitext(final_output_path)[0] + '.json'
    stats = {
        'target_date': target_date,
        'selected_date': selected_date_str,
        'days_from_target': days_from_target,
        'aoi_cloud_pct': round(aoi_cloud_pct, 2),
        'bands': bands,
        'band_names': band_names,
        'collection': collection,
        'output_resolution_m': output_resolution,
    }
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to: {json_path}")

    # Print summary
    with rasterio.open(final_output_path) as src:
        print(f"\nDownload complete: {final_output_path}")
        print(f"  Bands: {src.count}")
        print(f"  Resolution: {src.res[0]}m")
        print(f"  CRS: {src.crs}")
        print(f"  Dimensions: {src.width} x {src.height}")
        print(f"  Cloud-masked pixels: {aoi_cloud_pct:.1f}% of AOI")
        for i in range(1, src.count + 1):
            print(f"  Band {i}: {src.descriptions[i-1]}")

    return final_output_path


# =========================================================================== #
# pick_scene — stage 0: settle the calibration reference BEFORE any heavy work
# =========================================================================== #
def _median_date(dates):
    """Median of 'YYYY-MM-DD' strings; the latest drops first on even counts,
    so a late re-fly can't drag the target forward."""
    ds = sorted(datetime.strptime(d, "%Y-%m-%d") for d in dates)
    return ds[(len(ds) - 1) // 2].strftime("%Y-%m-%d")


def pick_scene(
    bounds_raster: str = None,
    bounds: tuple = None,
    bounds_crs_epsg: int = None,
    mission_dates: list = None,
    target_date: str = None,
    search_days: int = 14,
    max_scene_cloud_pct: float = 20.0,
    gee_credentials_path: str = None,
    interactive: bool = True,
    collection: str = 'COPERNICUS/S2_SR_HARMONIZED',
    log=print,
) -> dict:
    """Search Sentinel-2 around the campaign and choose the calibration scene.

    Calibration is the LAST stage, but its scene is chosen FIRST — otherwise
    hours of stitching can end at a cloudy reference. target_date=None derives
    the target from the median of mission_dates (per-mission `date` fields in
    the survey TOML). Candidates are ranked by |days from target|; each gets
    an AOI cloud fraction (QA60 over YOUR footprint — the number that matters,
    not the granule figure). Interactive: prints the menu, pre-selects the
    nearest clean scene, reads a choice from stdin (Enter = accept).
    Headless: auto-accepts the suggestion and logs it.

    Returns {date, days_from_target, aoi_cloud_pct, scene_cloud_pct, id} —
    lock it into the run manifest so stage 4 downloads exactly this scene.
    """
    if target_date in (None, "auto"):
        if not mission_dates:
            raise ValueError(
                'date = "auto" needs mission dates: give each mission a '
                'date = "YYYY-MM-DD" in the TOML, or set an explicit calibrate.date')
        target_date = _median_date(mission_dates)
        log(f"  {len(mission_dates)} mission dates -> median {target_date}")

    # ---- bounds -> WGS84 region ------------------------------------------- #
    if bounds_raster is not None:
        with rasterio.open(bounds_raster) as src:
            bounds = src.bounds
            bounds_crs_epsg = src.crs.to_epsg()
    if bounds is None or bounds_crs_epsg is None:
        raise ValueError("pick_scene needs bounds_raster, or bounds + bounds_crs_epsg")
    transformer = Transformer.from_crs(CRS.from_epsg(bounds_crs_epsg),
                                       CRS.from_epsg(4326), always_xy=True)
    l, t = transformer.transform(bounds[0], bounds[3])
    r, _ = transformer.transform(bounds[2], bounds[3])
    _, b = transformer.transform(bounds[2], bounds[1])
    region = [[l, t], [r, t], [r, b], [l, b], [l, t]]

    if not gee_credentials_path or not os.path.exists(gee_credentials_path):
        raise FileNotFoundError(
            f"GEE credentials not found: {gee_credentials_path!r} — the config's "
            "credentials_env names an environment variable whose value must be the "
            "path to a service-account JSON; set it in the shell or in a .env "
            "beside the survey TOML")
    scopes = ['https://www.googleapis.com/auth/earthengine']
    credentials = service_account.Credentials.from_service_account_file(
        gee_credentials_path, scopes=scopes)
    ee.Initialize(credentials)
    ee_region = ee.Geometry.Polygon([region])

    tgt = datetime.strptime(target_date, "%Y-%m-%d")
    start = (tgt - timedelta(days=search_days)).strftime("%Y-%m-%d")
    end = (tgt + timedelta(days=search_days + 1)).strftime("%Y-%m-%d")
    log(f"  searching {collection}, {start} .. {end} (±{search_days} d)")
    coll = ee.ImageCollection(collection).filterDate(start, end).filterBounds(ee_region)
    feats = coll.getInfo().get("features", [])
    if not feats:
        raise ValueError(f"no Sentinel-2 scenes {start}..{end} — widen search_days")

    # one candidate per DAY (nearest granule), AOI cloud via QA60
    by_day = {}
    for f in feats:
        d = datetime.fromtimestamp(f["properties"]["system:time_start"] / 1000)
        by_day.setdefault(d.strftime("%Y-%m-%d"), f)
    cands = []
    for day, f in sorted(by_day.items()):
        img = ee.Image(f["id"])
        qa = img.select("QA60")
        cloud = qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))
        st = cloud.reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_region,
                                scale=60, maxPixels=1e9).get("QA60").getInfo()
        aoi = (st or 0.0) * 100
        scn = f["properties"].get("CLOUDY_PIXEL_PERCENTAGE")
        delta = (datetime.strptime(day, "%Y-%m-%d").date() - tgt.date()).days
        cands.append(dict(date=day, days_from_target=delta, aoi_cloud_pct=round(aoi, 1),
                          scene_cloud_pct=(round(scn, 1) if scn is not None else None),
                          id=f["id"], over_limit=aoi > max_scene_cloud_pct))
    cands.sort(key=lambda c: (abs(c["days_from_target"]), c["aoi_cloud_pct"]))

    clean = [c for c in cands if not c["over_limit"]]
    suggested = clean[0] if clean else cands[0]

    log("\n    #   date          Δ target    cloud over survey    scene cloud")
    for i, c in enumerate(cands, 1):
        mark = "›" if c is suggested else " "
        flag = "   (over cloud limit)" if c["over_limit"] else \
               ("      ← suggested" if c is suggested else "")
        scn = f"{c['scene_cloud_pct']:.1f}%" if c["scene_cloud_pct"] is not None else "n/a"
        log(f"  {mark} {i:<3} {c['date']}   {c['days_from_target']:+4d} d        "
            f"{c['aoi_cloud_pct']:5.1f}%             {scn}{flag}")

    chosen = suggested
    if interactive:
        try:
            raw = input(f"\n  scene [{cands.index(suggested) + 1}] (Enter = accept): ").strip()
            if raw:
                chosen = cands[int(raw) - 1]
        except (EOFError, KeyboardInterrupt):
            pass
    log(f"  selected -> {chosen['date']}  (AOI cloud {chosen['aoi_cloud_pct']:.1f}%)")
    return chosen

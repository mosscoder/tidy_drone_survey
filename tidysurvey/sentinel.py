"""
Sentinel-2 band download utilities for survey calibration.

Downloads Sentinel-2 bands from Google Earth Engine, matched to survey dates.
"""
import json
import os
from datetime import datetime, timedelta

try:
    import ee
except ImportError:                       # the [sentinel] extra; the GPU registration image omits it
    class _MissingEE:
        def __getattr__(self, name):
            raise ImportError("tidysurvey.sentinel needs earthengine-api "
                              "(pip install 'tidysurvey[sentinel]')")
    ee = _MissingEE()
import rasterio
import requests
from google.oauth2 import service_account
from pyproj import CRS, Transformer


def _aoi_cloud_pct(img, ee_region) -> float:
    """AOI cloud fraction (%) — QA60 cloud+cirrus bits meaned over the survey
    footprint at 60 m. The SINGLE definition of 'cloud over the AOI': the scene
    picker and the downloader both call it, so they can never again disagree
    about the same granule the way they once did (0.0% vs 46.5% — identical
    scene, but measured over one mission's clear corner vs the full survey)."""
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))
    st = cloud.reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_region,
                            scale=60, maxPixels=1e9).get("QA60").getInfo()
    return (st or 0.0) * 100


def _choose_scene(cands: list, max_cloud_pct: float) -> dict:
    """Pick the calibration scene from candidate dicts (each carrying
    aoi_cloud_pct + days_from_target): the NEAREST-to-target among those AT OR
    UNDER the cloud limit; if none qualify, the single least-cloudy scene (the
    caller warns). Pure + deterministic — the offline-testable core shared by
    both selectors, so 'which scene is clean' lives in exactly one place."""
    clean = [c for c in cands if c["aoi_cloud_pct"] <= max_cloud_pct]
    pool = clean or [min(cands, key=lambda c: c["aoi_cloud_pct"])]
    return min(pool, key=lambda c: (abs(c["days_from_target"]), c["aoi_cloud_pct"]))


def download_sentinel2_bands(
    bounds_raster: str,
    target_date: str,
    output_path: str,
    bands: list[str],
    gee_credentials_path: str,
    band_names: list[str] = None,
    buffer_days: int = 7,
    output_resolution: int = 10,
    scene_id: str = None,
    max_scene_cloud_pct: float = 20.0,
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

    target_dt = datetime.strptime(target_date, '%Y-%m-%d')

    if scene_id:
        # Exact granule chosen by pick_scene and locked in the manifest — download
        # THIS scene, never a date re-search. (The re-search below once re-picked a
        # 46%-cloud scene over the clean one pick_scene intended, because it ranked
        # purely by date proximity and ignored cloud entirely.)
        print(f"Using locked scene: {scene_id}")
        image = ee.Image(scene_id)
    else:
        # No locked id (manual/legacy call): search the window and choose the
        # NEAREST scene AT OR UNDER the AOI cloud limit — cloud first, then date.
        start_date = (target_dt - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        end_date = (target_dt + timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        print(f"Searching for Sentinel-2 images from {start_date} to {end_date}")
        coll = ee.ImageCollection(collection) \
            .filterDate(start_date, end_date).filterBounds(ee_region)
        feats = coll.getInfo().get('features', [])
        if not feats:
            raise ValueError(
                f"No Sentinel-2 images found between {start_date} and {end_date}. "
                "Try increasing buffer_days.")
        by_day = {}
        for f in feats:
            d = datetime.fromtimestamp(f['properties']['system:time_start'] / 1000)
            by_day.setdefault(d.strftime('%Y-%m-%d'), f)
        cands = []
        for day, f in sorted(by_day.items()):
            aoi = _aoi_cloud_pct(ee.Image(f['id']), ee_region)
            delta = (datetime.strptime(day, '%Y-%m-%d').date() - target_dt.date()).days
            cands.append(dict(date=day, days_from_target=delta,
                              aoi_cloud_pct=round(aoi, 1), id=f['id']))
            print(f"  - {day} (AOI cloud: {aoi:.1f}%)")
        pick = _choose_scene(cands, max_scene_cloud_pct)
        print(f"  -> {pick['date']} (AOI cloud {pick['aoi_cloud_pct']:.1f}%)")
        image = ee.Image(pick['id'])

    # Selected image info (same for both paths)
    image_info = image.getInfo()
    selected_date = datetime.fromtimestamp(
        image_info['properties']['system:time_start'] / 1000)
    selected_date_str = selected_date.strftime('%Y-%m-%d')
    date_yymmdd = selected_date.strftime('%y%m%d')
    days_from_target = abs((selected_date.date() - target_dt.date()).days)
    print(f"Selected image: {selected_date_str} ({days_from_target} days from target)")

    # AOI cloud cover for the sidecar + a loud guard: over the limit means the
    # calibration targets are cloud-masked/hazy over that fraction of the survey.
    aoi_cloud_pct = _aoi_cloud_pct(image, ee_region)
    print(f"  AOI cloud cover: {aoi_cloud_pct:.1f}%")
    if aoi_cloud_pct > max_scene_cloud_pct:
        print(f"  ⚠ WARNING: AOI cloud {aoi_cloud_pct:.1f}% exceeds the "
              f"{max_scene_cloud_pct:.0f}% limit — ~{aoi_cloud_pct:.0f}% of the "
              f"survey will calibrate against cloud-masked/hazy targets. "
              f"Re-pick with `tidysurvey scenes --rescan` before trusting this.")

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

    # Apply pixel-level cloud mask (QA60 cloud+cirrus → nodata)
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))
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
        aoi = _aoi_cloud_pct(ee.Image(f["id"]), ee_region)
        scn = f["properties"].get("CLOUDY_PIXEL_PERCENTAGE")
        delta = (datetime.strptime(day, "%Y-%m-%d").date() - tgt.date()).days
        cands.append(dict(date=day, days_from_target=delta, aoi_cloud_pct=round(aoi, 1),
                          scene_cloud_pct=(round(scn, 1) if scn is not None else None),
                          id=f["id"], over_limit=aoi > max_scene_cloud_pct))
    cands.sort(key=lambda c: (abs(c["days_from_target"]), c["aoi_cloud_pct"]))
    suggested = _choose_scene(cands, max_scene_cloud_pct)

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

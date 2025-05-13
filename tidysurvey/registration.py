import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling as RioResampling
from rasterio.vrt import WarpedVRT
from affine import Affine
import geopandas as gpd
from shapely.geometry import box
import torch
import kornia.feature as KF
import cv2
from tqdm import tqdm

def plan_grid(target_ortho_path: str, cell_size_m: float) -> list[tuple[float, float]]:
    """
    Plan a grid of cell center coordinates over a target orthomosaic.

    Args:
        target_ortho_path (str): Path to the target orthomosaic GeoTIFF.
        cell_size_m (float): Desired size of each cell in meters (unbuffered).
        
    Returns:
        list[tuple[float, float]]: List of (x_center, y_center) coordinates for each grid cell
                                   in the CRS of the target_ortho_path.
    """
    with rasterio.open(target_ortho_path) as src:
        bounds = src.bounds

    min_x, min_y, max_x, max_y = bounds.left, bounds.bottom, bounds.right, bounds.top

    n_cells_x = int(np.floor((max_x - min_x) / cell_size_m))
    n_cells_y = int(np.floor((max_y - min_y) / cell_size_m))

    if n_cells_x <= 0 or n_cells_y <= 0:
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        return [(center_x, center_y)]

    grid_coords = []
    for i in range(n_cells_x):
        x_center = min_x + (i + 0.5) * cell_size_m
        for j in range(n_cells_y):
            y_center = min_y + (j + 0.5) * cell_size_m
            grid_coords.append((x_center, y_center))

    return grid_coords

def make_paired_tiles(
    target_ortho_path: str,
    reference_ortho_path: str,
    x_center: float, 
    y_center: float, 
    cell_size_m: float, 
    buffer_m: float,
    resampling_method: RioResampling = RioResampling.bilinear
) -> tuple[np.ndarray | None, np.ndarray | None, dict | None, dict | None]:
    """
    Creates paired, buffered image tiles from a target and reference orthomosaic.
    The reference tile is warped to the CRS of the target tile.

    Args:
        target_ortho_path (str): Path to the target orthomosaic.
        reference_ortho_path (str): Path to the reference orthomosaic.
        x_center (float): X coordinate of the unbuffered cell center (in CRS of target_ortho_path).
        y_center (float): Y coordinate of the unbuffered cell center (in CRS of target_ortho_path).
        cell_size_m (float): Side length of the unbuffered cell in meters.
        buffer_m (float): Buffer distance in meters to add around the cell for tile extraction.
        resampling_method (RioResampling): Resampling method for warping the reference tile.
                                            Defaults to RioResampling.bilinear.
    
    Returns:
        tuple:
            - img_target_buffered (np.ndarray | None): Target image tile (bands, height, width).
            - img_reference_buffered (np.ndarray | None): Reference image tile (bands, height, width),
                                                       warped to target's CRS.
            - meta_target_buffered (dict | None): Rasterio metadata for the target tile.
            - meta_reference_buffered (dict | None): Rasterio metadata for the (warped) reference tile.
        Returns (None, None, None, None) if a tile cannot be extracted or warping fails.
    """
    
    buffered_side_length_m = cell_size_m + 2 * buffer_m
    half_buffered_side = buffered_side_length_m / 2

    # Define the geographic bounding box for the buffered tile in the target's CRS
    left = x_center - half_buffered_side
    bottom = y_center - half_buffered_side
    right = x_center + half_buffered_side
    top = y_center + half_buffered_side
    buffered_bounds_in_target_crs = (left, bottom, right, top)

    img_target_buffered, meta_target_buffered = None, None
    img_reference_buffered, meta_reference_buffered = None, None

    try:
        with rasterio.open(target_ortho_path) as src_target:
            target_crs = src_target.crs
            
            # Extract Target Tile (in its native CRS)
            window_target = rasterio.windows.from_bounds(
                *buffered_bounds_in_target_crs, transform=src_target.transform
            )
            img_target_buffered = src_target.read(window=window_target)
            meta_target_buffered = src_target.profile.copy()
            meta_target_buffered.update({
                'height': img_target_buffered.shape[1],
                'width': img_target_buffered.shape[2],
                'transform': src_target.window_transform(window_target)
            })

            # Extract Reference Tile (warped to Target CRS)
            with rasterio.open(reference_ortho_path) as src_reference:
                with WarpedVRT(src_reference, crs=target_crs, resampling=resampling_method) as vrt_reference:
                    
                    window_ref = rasterio.windows.from_bounds(
                        *buffered_bounds_in_target_crs, transform=vrt_reference.transform
                    )
                    img_reference_buffered = vrt_reference.read(window=window_ref)
                    
                    meta_reference_buffered = vrt_reference.profile.copy() # Profile of the VRT
                    meta_reference_buffered.update({
                        'height': img_reference_buffered.shape[1],
                        'width': img_reference_buffered.shape[2],
                        'transform': vrt_reference.window_transform(window_ref),
                        'crs': target_crs 
                    })

    except Exception as e:
        print(f"Error in make_paired_tiles for center ({x_center}, {y_center}): {e}")
        return None, None, None, None
        
    return img_target_buffered, img_reference_buffered, meta_target_buffered, meta_reference_buffered

def check_valid_mask(
    img_target: np.ndarray, 
    meta_target: dict, 
    img_reference: np.ndarray, 
    meta_reference: dict, 
    tol_na_frac: float = 0.25
) -> tuple[bool, float, float]:
    """
    Assess what fraction of pixels are NA in the target and reference images.

    Args:
        img_target (np.ndarray): Target image tile (bands, height, width).
        meta_target (dict): Rasterio metadata for the target tile, used to get nodata value.
        img_reference (np.ndarray): Reference image tile (bands, height, width).
        meta_reference (dict): Rasterio metadata for the reference tile, used to get nodata value.
        tol_na_frac (float): Tolerance for the fraction of NA pixels. If NA fraction in *either*
                             image exceeds this, the mask is considered invalid.
    
    Returns:
        tuple:
            - is_valid (bool): True if NA fraction in both images is <= tol_na_frac, False otherwise.
            - targ_na_frac (float): Fraction of NA pixels in the target image.
            - ref_na_frac (float): Fraction of NA pixels in the reference image.
    """
    if img_target is None or img_reference is None:
        return False, 1.0, 1.0 # Consider fully NA if image is None

    def calculate_na_fraction(img: np.ndarray, meta: dict) -> float:
        if img.size == 0:
            return 1.0 # Empty image is fully NA
        
        nodata_val = meta.get('nodata')
        
        if nodata_val is not None:
            if np.isnan(nodata_val):
                na_count = np.isnan(img).sum()
            else:
                na_count = (img == nodata_val).sum()
        else:
            # If no nodata value is defined in metadata, assume no pixels are NA by this definition.
            # This might need adjustment if NA is represented differently (e.g., all zeros for some sensors).
            na_count = 0 
            
        return na_count / img.size

    targ_na_frac = calculate_na_fraction(img_target, meta_target)
    ref_na_frac = calculate_na_fraction(img_reference, meta_reference)

    is_valid = (targ_na_frac <= tol_na_frac) and (ref_na_frac <= tol_na_frac)
    
    return is_valid, targ_na_frac, ref_na_frac

def rasterio_to_torch_tensor(numpy_array: np.ndarray, target_bands: int = 3) -> torch.Tensor:
    """
    Converts a NumPy array from rasterio window output (expected as B, H, W or H, W)
    to a PyTorch tensor of shape (1, C, H, W), suitable for models like LoFTR.
    It handles single-band, multi-band, and ensures the output has `target_bands` channels,
    either by selecting/padding or converting to grayscale if target_bands is 1.

    Args:
        numpy_array (np.ndarray): Input NumPy array from rasterio.read().
                                  Expected shapes: (bands, height, width) or (height, width).
        target_bands (int): Desired number of channels for the output tensor (e.g., 1 for grayscale, 3 for RGB).

    Returns:
        torch.Tensor: PyTorch tensor with shape (1, target_bands, H, W), normalized to [0, 1].
    """
    if not isinstance(numpy_array, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(numpy_array)}")

    # Ensure it's a float tensor for processing
    tensor = torch.from_numpy(numpy_array.astype(np.float32))

    # Handle (H, W) -> (1, H, W) for single band images from rasterio
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0) # Add channel dimension: (H, W) -> (1, H, W)
    
    if tensor.ndim != 3: # Should be (C, H, W) at this point
        raise ValueError(f"Input tensor must be 2D (H, W) or 3D (C, H, W), got {tensor.shape}")

    # Normalize to [0, 1] - assuming input is in typical image range (e.g., 0-255, 0-65535, etc.)
    # A more robust normalization might require min/max from data or specific sensor ranges.
    # For now, simple division assuming 0-255 range for typical visual bands.
    # If data is already float [0,1] this might be an issue, or if it's other int types.
    # Consider a more adaptive normalization if value ranges vary widely.
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val > 1.0: # Heuristic: if max is already > 1, assume it needs normalization from common integer ranges
        if max_val <= 255.0 and min_val >=0:
            tensor = tensor / 255.0
        elif max_val <= 65535.0 and min_val >=0:
            tensor = tensor / 65535.0
        # else: # Potentially already normalized or unknown range, leave as is or add warning
            # print(f"Warning: Tensor value range ({min_val.item()}-{max_val.item()}) not typical for 8/16 bit. Check normalization.")
    tensor = torch.clamp(tensor, 0.0, 1.0) # Ensure it's in [0,1] after normalization

    # Adjust channels to target_bands
    c, h, w = tensor.shape
    if target_bands == 1:
        if c == 1:
            pass # Already grayscale
        elif c == 3: # RGB to Grayscale
            # Using Kornia for consistency if available, otherwise standard weights
            # tensor = KF.rgb_to_grayscale(tensor.unsqueeze(0)).squeeze(0) # Requires Kornia
            # Standard ITU-R BT.601 weights: R*0.299 + G*0.587 + B*0.114
            # Assuming tensor is (3, H, W) for RGB input
            if c >=3: # take first 3 bands if more are present
                tensor = 0.299 * tensor[0:1, :, :] + 0.587 * tensor[1:2, :, :] + 0.114 * tensor[2:3, :, :]
            else: # if less than 3 bands, just take the first one (e.g. if it was 2 bands)
                tensor = tensor[0:1, :,:]
        elif c > 1: # Multi-band (not 3) to Grayscale (e.g. take first band)
            tensor = tensor[0:1, :, :] 
        # else c==1, already handled
    elif target_bands == 3:
        if c == 3:
            pass # Already RGB
        elif c == 1: # Grayscale to RGB (replicate channel)
            tensor = tensor.repeat(3, 1, 1)
        elif c > 3: # More than 3 channels, take first 3
            tensor = tensor[:3, :, :]
        else: # c == 2 or other, pad with zeros to 3 channels
            padding = torch.zeros(target_bands - c, h, w, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=0)
    # Else, if target_bands is not 1 or 3, the behavior is undefined by typical models
    # For now, we only explicitly handle target_bands = 1 or 3.
    # If other numbers are needed, this logic should be extended.
    if tensor.shape[0] != target_bands:
         # Fallback: if still not matching (e.g. target_bands=2), take first band and replicate if needed or error
         print(f"Warning: Could not achieve target_bands={target_bands}. Resulting channels: {tensor.shape[0]}")

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    tensor = tensor.unsqueeze(0)
    return tensor

def get_loftr_matches(
    img_target_np: np.ndarray, 
    img_reference_np: np.ndarray, 
    pretrained_model: str = "outdoor", 
    device_str: str = 'cpu'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds keypoint matches between two images using LoFTR and computes an inlier mask.
    Input images are NumPy arrays (bands, H, W) from rasterio.

    Args:
        img_target_np (np.ndarray): NumPy array of the target image (bands, H, W).
        img_reference_np (np.ndarray): NumPy array of the reference image (bands, H, W).
        pretrained_model (str): Name of the pretrained LoFTR model (e.g., "outdoor").
        device_str (str): Device to run LoFTR on ('cpu', 'cuda', 'mps', etc.).

    Returns:
        A tuple containing:
            - mkpts0 (np.ndarray): Matched keypoints in target image (N, 2).
            - mkpts1 (np.ndarray): Matched keypoints in reference image (N, 2).
            - inliers_mask (np.ndarray): Boolean array indicating inlier matches (N,).
                                        Returns empty arrays if matching or inlier detection fails.
    """
    
    if device_str == 'cuda' and torch.cuda.is_available():
        current_device = torch.device("cuda")
    elif device_str == 'mps' and torch.backends.mps.is_available():
        current_device = torch.device("mps")
    else:
        current_device = torch.device("cpu")
    # print(f"Using device: {current_device} for LoFTR") # Reduced verbosity

    img_target_tensor = rasterio_to_torch_tensor(img_target_np, target_bands=1).to(current_device)
    img_reference_tensor = rasterio_to_torch_tensor(img_reference_np, target_bands=1).to(current_device)

    if img_target_tensor.shape[2] < 20 or img_target_tensor.shape[3] < 20 or \
       img_reference_tensor.shape[2] < 20 or img_reference_tensor.shape[3] < 20:
        print("Warning: One or both images are too small for LoFTR. Skipping matching.")
        return np.array([]).reshape(0,2), np.array([]).reshape(0,2), np.array([], dtype=bool)

    matcher = KF.LoFTR(pretrained=pretrained_model).to(current_device).eval()

    input_dict = {
        "image0": img_target_tensor,
        "image1": img_reference_tensor,
    }

    try:
        with torch.inference_mode():
            correspondences = matcher(input_dict)
    except Exception as e:
        print(f"Error during LoFTR matching: {e}")
        return np.array([]).reshape(0,2), np.array([]).reshape(0,2), np.array([], dtype=bool)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    # mconf = correspondences["confidence"].cpu().numpy() # LoFTR confidence, not directly returned now

    if mkpts0.shape[0] == 0:
        print("LoFTR found no matches.")
        return np.array([]).reshape(0,2), np.array([]).reshape(0,2), np.array([], dtype=bool)

    # Use cv2.findFundamentalMat to determine inliers, as per typical pipeline structure
    # This was the previous approach and aligns with having an "inlier_mask"
    inliers_mask = np.zeros(mkpts0.shape[0], dtype=bool)
    if mkpts0.shape[0] >= 7: # findFundamentalMat requires at least 7 points
        # Parameters for findFundamentalMat can be tuned. Using USAC_MAGSAC as it's generally robust.
        # Thresholds like 0.5 for reprojection error and 0.999 for confidence are common starts.
        # The maxIterations (e.g., 100000) is to give it ample chance to find a good model.
        try:
            _, cv_inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
            if cv_inliers is not None:
                inliers_mask = cv_inliers.ravel().astype(bool)
            else:
                print("Warning: findFundamentalMat returned None for inliers.")
        except cv2.error as e:
            print(f"cv2.error in findFundamentalMat: {e}. mkpts0: {mkpts0.shape}, mkpts1: {mkpts1.shape}")
            # Keep inliers_mask as all False
    else:
        print(f"Warning: Not enough keypoints ({mkpts0.shape[0]}) to compute Fundamental Matrix. Returning no inliers.")
        # inliers_mask remains all False

    return mkpts0, mkpts1, inliers_mask

def register_target(
    img_target_buffered_np: np.ndarray, 
    meta_reference_buffered: dict, # Metadata of the space to warp into
    mkpts_target: np.ndarray, 
    mkpts_reference: np.ndarray, 
    inliers_mask: np.ndarray, # Boolean mask from get_loftr_matches
    min_matches_for_homography: int = 10,
    ransac_reproj_threshold: float = 5.0
) -> tuple[np.ndarray | None, dict | None]:
    """
    Registers the target image tile to the reference image tile using an inlier mask
    derived from LoFTR keypoints and findFundamentalMat.
    The output image is warped to align with the reference tile's pixel grid and georeferencing.

    Args:
        img_target_buffered_np (np.ndarray): Target image tile (bands, H, W) including buffer.
        meta_reference_buffered (dict): Rasterio metadata for the reference tile.
                                          This defines the target coordinate system for warping.
        mkpts_target (np.ndarray): Matched keypoints in target image tile (N, 2), from LoFTR.
        mkpts_reference (np.ndarray): Matched keypoints in reference image tile (N, 2), from LoFTR.
        inliers_mask (np.ndarray): Boolean array (N,) indicating inlier matches.
        min_matches_for_homography (int): Minimum number of inlier matches to attempt homography.
        ransac_reproj_threshold (float): RANSAC reprojection threshold for cv2.findHomography.

    Returns:
        tuple:
            - registered_target_img (np.ndarray | None): The target image warped to the reference
                                                         tile's coordinate system (bands, H, W).
                                                         Shape matches reference tile.
            - registered_meta (dict | None): Rasterio metadata for the registered target image.
                                             Transform, CRS, width, height match meta_reference_buffered.
                                             Returns None, None if registration fails.
    """
    if img_target_buffered_np is None or meta_reference_buffered is None:
        print("Error: Target image or reference metadata is None. Cannot register.")
        return None, None
    if inliers_mask is None:
        print("Error: Inliers mask is None. Cannot register.")
        return None, None

    # Filter points using the provided inliers_mask
    if not np.any(inliers_mask):
        print("Warning: No inliers provided by the mask. Cannot compute homography.")
        return None, None
        
    pts_target = mkpts_target[inliers_mask]
    pts_reference = mkpts_reference[inliers_mask]

    if pts_target.shape[0] < min_matches_for_homography:
        print(f"Warning: Not enough inlier matches ({pts_target.shape[0]}) for homography. Need at least {min_matches_for_homography}.")
        return None, None

    # Estimate Homography
    homography_matrix, h_mask = cv2.findHomography(
        pts_target, pts_reference, cv2.RANSAC, ransac_reproj_threshold
    )

    if homography_matrix is None:
        print("Warning: Homography estimation failed.")
        return None, None

    ref_height = meta_reference_buffered['height']
    ref_width = meta_reference_buffered['width']
    ref_bands = img_target_buffered_np.shape[0]
    ref_dtype = img_target_buffered_np.dtype

    img_target_hwc = np.moveaxis(img_target_buffered_np, 0, -1)
    
    registered_target_hwc = cv2.warpPerspective(
        img_target_hwc, 
        homography_matrix, 
        (ref_width, ref_height)
    )
    
    if registered_target_hwc.ndim == 2 and ref_bands == 1:
        registered_target_hwc = registered_target_hwc[..., np.newaxis]
        
    registered_target_img = np.moveaxis(registered_target_hwc, -1, 0)

    registered_meta = meta_reference_buffered.copy()
    registered_meta['count'] = ref_bands
    registered_meta['dtype'] = str(ref_dtype) 

    return registered_target_img, registered_meta

def tidy_target(registered_target_img: np.ndarray, metadata_target: dict, buffer_pixels: int):
    """
    Removes a buffer from an image and updates its metadata (transform, width, height).

    Args:
        registered_target_img (np.ndarray): The image data (bands, height, width) including the buffer.
        metadata_target (dict): Rasterio-like metadata for registered_target_img.
                                Must include 'transform' (affine.Affine), 'width', 'height'.
                                Other keys like 'crs', 'count', 'dtype' will be passed through.
        buffer_pixels (int): The buffer size in pixels to remove from each side.

    Returns:
        tuple: (tidied_image_data (np.ndarray), tidied_metadata (dict))
               tidied_metadata contains updated 'transform', 'width', 'height'.
    """
    if not isinstance(metadata_target, dict) or 'transform' not in metadata_target:
        raise ValueError("metadata_target must be a dict with a 'transform' key.")
    if not isinstance(metadata_target['transform'], Affine):
        raise ValueError("metadata_target['transform'] must be an affine.Affine object.")
    if buffer_pixels < 0:
        raise ValueError("buffer_pixels must be non-negative.")

    original_transform = metadata_target['transform']
    
    # Calculate new transform: top-left corner shifts by (buffer_pixels, buffer_pixels)
    # relative to the old pixel grid.
    new_top_left_transform = original_transform * Affine.translation(buffer_pixels, buffer_pixels)

    # Crop the image data
    # Assuming image is (bands, height, width) or (height, width)
    if registered_target_img.ndim == 3:
        h_buffered, w_buffered = registered_target_img.shape[1], registered_target_img.shape[2]
        tidied_image_data = registered_target_img[
            :,
            buffer_pixels : h_buffered - buffer_pixels,
            buffer_pixels : w_buffered - buffer_pixels
        ]
        new_height, new_width = tidied_image_data.shape[1], tidied_image_data.shape[2]
    elif registered_target_img.ndim == 2:
        h_buffered, w_buffered = registered_target_img.shape
        tidied_image_data = registered_target_img[
            buffer_pixels : h_buffered - buffer_pixels,
            buffer_pixels : w_buffered - buffer_pixels
        ]
        new_height, new_width = tidied_image_data.shape
    else:
        raise ValueError("registered_target_img must be a 2D or 3D array.")

    if new_height <= 0 or new_width <= 0:
        raise ValueError(f"Buffer ({buffer_pixels}px) is too large for image dimensions ({h_buffered}x{w_buffered}).")

    tidied_metadata = metadata_target.copy()
    tidied_metadata['transform'] = new_top_left_transform
    tidied_metadata['width'] = new_width
    tidied_metadata['height'] = new_height

    return tidied_image_data, tidied_metadata


def make_recipient_ortho(
    target_ortho_path: str,
    recipient_ortho_path: str,
    dtype_override: str = None,
    nodata_override = None,
    fill_value = np.nan # Default fill if nodata is not specified or not applicable for dtype
):
    """
    Creates an empty recipient orthomosaic GeoTIFF with the same geospatial
    properties (dimensions, CRS, transform, bounds) as a model orthomosaic.
    The created raster is filled with a specified nodata value or NaN.

    Args:
        target_ortho_path (str): Path to the model orthomosaic (e.g., original target).
        recipient_ortho_path (str): Path for the new recipient orthomosaic.
        dtype_override (str, optional): Override data type for the recipient.
                                      If None, uses dtype from target_ortho_path.
        nodata_override (any, optional): Override NoData value for the recipient.
                                       If None, uses nodata from target_ortho_path if available.
        fill_value (any, optional): Value to fill the raster with if nodata cannot be set
                                   (e.g. nodata_override is None and target has no nodata, or chosen dtype).
                                   Defaults to np.nan, which implies a float dtype if not overridden.

    Returns:
        dict: The Rasterio profile of the created recipient orthomosaic.
    """
    with rasterio.open(target_ortho_path) as src:
        profile = src.profile.copy()

    if dtype_override:
        profile['dtype'] = dtype_override
    
    # If fill_value is nan and dtype is not float, this will be problematic.
    # Ensure dtype is float if nan is the fill_value and no nodata is explicitly set.
    current_fill_value = fill_value
    if nodata_override is not None:
        profile['nodata'] = nodata_override
        current_fill_value = nodata_override # Fill with the specified nodata value
    elif profile.get('nodata') is not None:
        current_fill_value = profile['nodata'] # Fill with source nodata
    elif np.isnan(current_fill_value) and not np.issubdtype(np.dtype(profile['dtype']), np.floating):
        # If default fill is nan, but dtype is integer, change dtype to float32
        profile['dtype'] = 'float32'
        if nodata_override is None: # Only set nodata if not already overridden
             profile['nodata'] = np.nan

    # Ensure blockysize is a power of 2 for COG compatibility if not present
    if 'blockxsize' not in profile or not (profile['blockxsize'] > 0 and (profile['blockxsize'] & (profile['blockxsize'] - 1) == 0)):
        profile['blockxsize'] = 256 # Default block size
    if 'blockysize' not in profile or not (profile['blockysize'] > 0 and (profile['blockysize'] & (profile['blockysize'] - 1) == 0)):
        profile['blockysize'] = 256 # Default block size
    profile['tiled'] = True

    with rasterio.open(recipient_ortho_path, 'w', **profile) as dst:
        # Efficiently fill the raster if a fill value is determined
        # For very large rasters, writing in chunks might be more memory-efficient
        # but rasterio handles reasonably sized ones well with a full write.
        if profile.get('nodata') is not None:
            pass # No need to explicitly fill if nodata is set, it's implicitly that value
        else:
            # If no nodata value, fill explicitly, e.g. for float types with NaN
            # This part is tricky because just opening with nodata doesn't fill.
            # For true emptiness, relying on nodata is best.
            # If we must fill (e.g. no nodata concept for the dtype), do it block by block.
            fill_block = np.full((profile['blockysize'], profile['blockxsize']), 
                                 current_fill_value, dtype=profile['dtype'])
            for ji, window in dst.block_windows(1):
                # Adjust fill_block shape for partial blocks at edges
                current_block_shape = (window.height, window.width)
                if fill_block.shape != current_block_shape:
                    block_data = np.full(current_block_shape, current_fill_value, dtype=profile['dtype'])
                    dst.write(block_data, window=window, indexes=dst.count)
                else:
                    dst.write(fill_block, window=window, indexes=dst.count)
                    
    return profile

def fill_recipient_ortho(
    list_of_tidy_target_imgs: list[np.ndarray],
    list_of_metadata_targets: list[dict],
    recipient_ortho_path: str
):
    """
    Fills the recipient orthomosaic with tidied target images.

    Each image in list_of_tidy_target_imgs is written to the recipient_ortho_path
    according to its corresponding metadata in list_of_metadata_targets.
    Assumes recipient_ortho_path exists and is writable.

    Args:
        list_of_tidy_target_imgs (list[np.ndarray]): List of tidied target image data arrays.
                                                     Each array is (bands, height, width) or (height, width).
        list_of_metadata_targets (list[dict]): List of metadata dicts for each target image.
                                              Each dict must have 'transform', 'width', 'height',
                                              and optionally 'count' (bands).
        recipient_ortho_path (str): Path to the recipient orthomosaic.
    """
    if len(list_of_tidy_target_imgs) != len(list_of_metadata_targets):
        raise ValueError("Mismatch between number of images and metadata entries.")

    with rasterio.open(recipient_ortho_path, 'r+') as dst:
        recipient_transform = dst.transform
        recipient_crs = dst.crs

        for img_data, meta in zip(list_of_tidy_target_imgs, list_of_metadata_targets):
            if not isinstance(meta, dict) or not all(k in meta for k in ['transform', 'width', 'height']):
                raise ValueError("Invalid metadata entry: must be dict with transform, width, height.")
            if not isinstance(meta['transform'], Affine):
                raise ValueError("Metadata 'transform' must be an Affine object.")

            # Ensure image data is in (bands, height, width) or (height, width) format
            if img_data.ndim == 2:
                # Add a band dimension for single-band images
                img_data_to_write = img_data.reshape(1, *img_data.shape)
            elif img_data.ndim == 3:
                img_data_to_write = img_data
            else:
                raise ValueError("Image data must be 2D or 3D.")
            
            # Check CRS consistency if available in source meta
            if 'crs' in meta and meta['crs'] != recipient_crs:
                # This is a simplistic check. True reprojection is complex.
                # For now, we assume CRSs are compatible if provided and matching.
                # Consider raising a warning or error if they don't match.
                print(f"Warning: CRS mismatch for a tile ({meta['crs']}) and recipient ({recipient_crs}). Assuming compatibility.")

            # Calculate the bounds of the current tidied image in its own CRS
            # meta['transform'] is the transform for img_data
            # img_data_to_write.shape[2] is width, img_data_to_write.shape[1] is height
            img_bounds = array_bounds(
                height=img_data_to_write.shape[1],
                width=img_data_to_write.shape[2],
                transform=meta['transform']
            )

            # Calculate the window in the recipient raster
            try:
                window_in_recipient = Window.from_slices(
                    *rasterio.windows.transform(img_bounds, recipient_transform).round_offsets().round_lengths().toranges()
                )
                # Alternative: window_in_recipient = rasterio.windows.from_bounds(*img_bounds, transform=recipient_transform)
                # from_bounds can be sensitive, let's try to be precise by converting from world to pixel coords carefully
                row_start, row_stop, col_start, col_stop = rasterio.warp.transform_bounds(
                    meta.get('crs', recipient_crs), # Use tile's CRS if available, else recipient's
                    recipient_crs, 
                    *img_bounds
                )
                
                # Convert geographic bounds to pixel window
                # This is a more robust way if transforms are slightly different
                # or CRSs are involved (though full reprojection is not done here)
                top_left = rasterio.transform.rowcol(recipient_transform, xs=img_bounds[0], ys=img_bounds[3])
                bottom_right = rasterio.transform.rowcol(recipient_transform, xs=img_bounds[2], ys=img_bounds[1])

                window_col_off = top_left[1]
                window_row_off = top_left[0]
                window_width = bottom_right[1] - top_left[1]
                window_height = bottom_right[0] - top_left[0]
                
                # Ensure window dimensions match the data to write, adjusting if necessary due to rounding
                # This can happen if tile transform is not perfectly aligned with recipient grid
                # For simplicity, we expect the window to match img_data_to_write dimensions
                # If not, the user might need to resample/reproject the tile first.
                
                # Create the window object
                window_to_write = Window(window_col_off, window_row_off, window_width, window_height)
                
                # Check that window dimensions approximately match data dimensions
                if not (abs(window_to_write.width - img_data_to_write.shape[2]) <=1 and \
                        abs(window_to_write.height - img_data_to_write.shape[1]) <=1 ):
                     print(f"Warning: Calculated window {window_to_write} dimensions differ significantly from image data {img_data_to_write.shape[1:]}. Tile may be skewed or scaled differently.")
                     # Potentially crop/pad img_data_to_write or skip if too different.
                     # For now, proceed with calculated window.

            except Exception as e:
                print(f"Could not calculate window for a tile. Bounds: {img_bounds}, Error: {e}")
                continue # Skip this tile

            # Write the data
            # Ensure number of bands matches
            if img_data_to_write.shape[0] != dst.count:
                 # Attempt to write to the first band if single band image and multi-band recipient, or vice-versa
                 # This is a common scenario, but be careful.
                if img_data_to_write.shape[0] == 1 and dst.count > 1:
                    print(f"Warning: Writing single-band image to multi-band ({dst.count}) recipient. Writing to band 1.")
                    dst.write(img_data_to_write[0], window=window_to_write, indexes=1)
                elif dst.count == 1 and img_data_to_write.shape[0] > 1:
                    # Example: take first band of image data if recipient is single band
                    print(f"Warning: Writing multi-band image ({img_data_to_write.shape[0]}) to single-band recipient. Writing first band of image.")
                    dst.write(img_data_to_write[0], window=window_to_write, indexes=1)
                else:
                    print(f"Error: Band count mismatch. Image has {img_data_to_write.shape[0]} bands, recipient has {dst.count}. Skipping tile.")
                    continue
            else:
                dst.write(img_data_to_write, window=window_to_write)

def register_ortho(
    target_ortho_path: str,
    reference_ortho_path: str,
    output_recipient_path: str,
    cell_size_m: float,
    buffer_m: float,
    min_matches_for_homography: int = 10,
    ransac_reproj_threshold: float = 5.0,
    loftr_pretrained_model: str = "outdoor",
    loftr_device_str: str = "cpu",
    tile_resampling_method: RioResampling = RioResampling.bilinear,
    nodata_tolerance_fraction: float = 0.25,
    recipient_dtype_override: str = None,
    recipient_nodata_override = None
) -> str:
    """
    Orchestrates the entire survey registration pipeline.

    Steps include:
    1. Planning a grid over the target orthomosaic.
    2. Creating an empty recipient orthomosaic.
    3. For each grid cell:
        a. Extracting paired target and reference tiles (with buffering).
        b. Validating tiles for NoData content.
        c. If valid, attempting feature matching (LoFTR) and registration (Homography).
        d. Tidying the processed tile (removing buffer).
        e. If registration fails or tile is invalid, the original target tile (tidied) is used.
    4. Filling the recipient orthomosaic with all processed (and tidied) tiles.

    Args:
        target_ortho_path: Path to the target orthomosaic GeoTIFF.
        reference_ortho_path: Path to the reference orthomosaic GeoTIFF.
        output_recipient_path: Path to save the final registered mosaic.
        cell_size_m: Size of each grid cell in meters (unbuffered).
        buffer_m: Buffer to add around each cell for tile processing, in meters.
        min_matches_for_homography: Minimum number of inlier matches to attempt homography.
        ransac_reproj_threshold: RANSAC reprojection threshold for findHomography.
        loftr_pretrained_model: LoFTR pretrained model name.
        loftr_device_str: Device for LoFTR ('cpu', 'cuda', 'mps').
        tile_resampling_method: Rasterio resampling method for warping reference tiles.
        nodata_tolerance_fraction: Max allowed NA fraction in tiles for registration attempt.
        recipient_dtype_override: Optional dtype for the output mosaic.
        recipient_nodata_override: Optional NoData value for the output mosaic.

    Returns:
        Path to the created registered orthomosaic.
    """

    print(f"Starting survey registration process...")
    print(f"  Target: {target_ortho_path}")
    print(f"  Reference: {reference_ortho_path}")
    print(f"  Output: {output_recipient_path}")

    # 1. Plan Grid
    print(f"Planning grid with cell size {cell_size_m}m...")
    grid_coords = plan_grid(target_ortho_path, cell_size_m)
    if not grid_coords:
        print("Error: No grid cells planned. Aborting.")
        return None
    print(f"Planned {len(grid_coords)} grid cells.")

    # 2. Create Recipient Ortho
    print(f"Creating recipient orthomosaic: {output_recipient_path}")
    try:
        make_recipient_ortho(
            target_ortho_path=target_ortho_path, 
            recipient_ortho_path=output_recipient_path,
            dtype_override=recipient_dtype_override,
            nodata_override=recipient_nodata_override
        )
    except Exception as e:
        print(f"Error creating recipient orthomosaic: {e}. Aborting.")
        return None

    processed_tiles_data = []
    processed_tiles_metadata = []

    # 3. Process Each Tile
    print(f"Processing {len(grid_coords)} tiles with {buffer_m}m buffer...")
    for x_center, y_center in tqdm(grid_coords, desc="Processing Tiles"):
        img_t_buf, img_r_buf, meta_t_buf, meta_r_buf = make_paired_tiles(
            target_ortho_path,
            reference_ortho_path,
            x_center, y_center,
            cell_size_m,
            buffer_m,
            resampling_method=tile_resampling_method
        )

        if img_t_buf is None or meta_t_buf is None: # Indicates failure in make_paired_tiles for this cell
            # tqdm.write(f"Skipping cell ({x_center:.2f}, {y_center:.2f}) due to tile extraction failure.")
            continue
        
        current_tile_data = None
        current_tile_meta = None
        registration_successful = False

        # Validate tiles
        is_valid_for_reg, targ_na, ref_na = check_valid_mask(
            img_t_buf, meta_t_buf, img_r_buf, meta_r_buf, 
            tol_na_frac=nodata_tolerance_fraction
        )
        # tqdm.write(f"  Cell ({x_center:.2f}, {y_center:.2f}): Valid for Reg: {is_valid_for_reg} (Target NA: {targ_na:.2f}, Ref NA: {ref_na:.2f})")

        if is_valid_for_reg:
            # Attempt matching
            mkpts_t, mkpts_r, inliers_mask = get_loftr_matches(
                img_t_buf, img_r_buf, 
                pretrained_model=loftr_pretrained_model, 
                device_str=loftr_device_str
            )

            if inliers_mask.sum() >= min_matches_for_homography:
                # Attempt registration
                # tqdm.write(f"    Attempting registration with {inliers_mask.sum()} inlier points.")
                reg_t_buf, reg_meta_buf = register_target(
                    img_t_buf, 
                    meta_r_buf, # Warp target into reference tile's space and metadata
                    mkpts_t, 
                    mkpts_r, 
                    inliers_mask,
                    min_matches_for_homography=min_matches_for_homography,
                    ransac_reproj_threshold=ransac_reproj_threshold
                )

                if reg_t_buf is not None and reg_meta_buf is not None:
                    # Registration successful, now tidy this registered tile
                    try:
                        buffer_px_for_tidy = int(round(buffer_m / abs(reg_meta_buf['transform'].a)))
                        current_tile_data, current_tile_meta = tidy_target(
                            reg_t_buf, reg_meta_buf, buffer_px_for_tidy
                        )
                        registration_successful = True
                        tqdm.write(f"      Registration and tidying successful.")
                    except Exception as e:
                        tqdm.write(f"      Error tidying registered tile: {e}")
                
        if not registration_successful:
            tqdm.write(f"    Using original target tile (pass-through). Attempting to tidy.")
            try:
                buffer_px_for_tidy = int(round(buffer_m / abs(meta_t_buf['transform'].a)))
                current_tile_data, current_tile_meta = tidy_target(
                    img_t_buf, meta_t_buf, buffer_px_for_tidy
                )
                tqdm.write(f"      Tidying of original target successful.")
            except Exception as e:
                tqdm.write(f"      Error tidying original target tile: {e}")
        
        if current_tile_data is not None and current_tile_meta is not None:
            processed_tiles_data.append(current_tile_data)
            processed_tiles_metadata.append(current_tile_meta)
        # else:
            # tqdm.write(f"    Failed to produce a tidied tile for cell ({x_center:.2f}, {y_center:.2f}).")

    # 4. Fill Recipient Ortho
    if processed_tiles_data:
        print(f"Assembling final orthomosaic from {len(processed_tiles_data)} processed tiles...")
        try:
            fill_recipient_ortho(
                processed_tiles_data, 
                processed_tiles_metadata, 
                output_recipient_path
            )
            print(f"Successfully created registered orthomosaic: {output_recipient_path}")
            return output_recipient_path
        except Exception as e:
            print(f"Error filling recipient orthomosaic: {e}")
            return None
    else:
        print("No tiles were successfully processed to fill the orthomosaic.")
        # Optionally, delete the empty recipient file created by make_recipient_ortho if desired.
        # For now, it will remain as an empty (or NaN-filled) file.
        return None

    



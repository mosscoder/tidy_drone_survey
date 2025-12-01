# Make functions/classes from submodules available at the package level
from .merge import define_hull
from .merge import find_seamlines
from .merge import clip
from .merge import mosaic
from .merge import meta_mosaic
from .registration import Chip
from .registration import resolve_height_chip
from .registration import get_bbox_bounds_chip
from .registration import generate_grid_points_chip
from .registration import load_chip_gdal
from .registration import load_chips_gdal
from .registration import raster_to_tensor_chip
from .registration import resize_image_chip
from .registration import batch_get_loftr_matches_chip
from .registration import generate_chip_gcps_gdal
from .registration import warp_chip_gdal
from .registration import merge_warped_chips_gdal
from .registration import calculate_chip_crs_parameters_gdal
from .registration import register_survey_by_chips

# Optionally, define __all__ to control what `from tidysurvey import *` imports
__all__ = ['define_hull', 'find_seamlines', 'clip', 'mosaic', 'meta_mosaic',
           'Chip', 'resolve_height_chip', 'get_bbox_bounds_chip', 'generate_grid_points_chip',
           'load_chip_gdal', 'load_chips_gdal', 'raster_to_tensor_chip', 'resize_image_chip',
           'batch_get_loftr_matches_chip', 'generate_chip_gcps_gdal', 'warp_chip_gdal',
           'merge_warped_chips_gdal', 'calculate_chip_crs_parameters_gdal', 'register_survey_by_chips'] 
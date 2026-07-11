# tidysurvey — the config-driven survey pipeline (refactor, branch dev).
#
# Lean surface: one module per stage + the shared engine. Retired first-gen
# machinery (per-chip registration internals, hard-cut mosaic + meta_mosaic,
# clip-and-fill, geo neighbour helpers) lives on branch `main` / git history.
# One compatibility promise is kept: register_survey_by_chips retains its
# exact signature, running the audited dense internals.

# --- remote-source I/O config (audit shared/config.json gcs_env + m2_full.py) --
# Set BEFORE any GDAL/numpy import so it takes at library init; setdefault means
# an explicit shell value still wins. GDAL_NUM_THREADS=1 pins warp READS
# single-threaded (a 9-thread pool per 640x480 read is pure overhead — audit
# 02_registration §7); every writer passes NUM_THREADS per-dataset, so writes
# stay multithreaded. The /vsicurl + HTTP flags drop the per-open bucket LIST
# and pipeline range requests for the GCS mission COGs (registration reads them
# per tile); the OMP/BLAS pins stop loky workers oversubscribing in calibrate.
import os as _os
for _k, _v in {
    "GDAL_NUM_THREADS": "1",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
    "GDAL_CACHEMAX": "8192",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}.items():
    _os.environ.setdefault(_k, _v)

from . import config
from . import fields
from . import cog
from . import validate
from . import report
from . import calibrate
from . import tiles

# stitch (seam-walk blend) + mission-boundary products
from .merge import (
    seam_merge,
    define_hull_tiled,
    find_seamlines,
    generate_combined_boundaries,
)

# align (dense corroborated field)
from .registration import register_survey_dense
from .registration import register_survey_by_chips   # same signature, dense internals

# satellite reference
from .sentinel import download_sentinel2_bands, pick_scene

__all__ = [
    # modules
    'config', 'fields', 'cog', 'validate', 'report', 'calibrate', 'tiles',
    # stitch + boundaries
    'seam_merge', 'define_hull_tiled', 'find_seamlines',
    'generate_combined_boundaries',
    # align
    'register_survey_dense', 'register_survey_by_chips',
    # sentinel
    'download_sentinel2_bands', 'pick_scene',
]

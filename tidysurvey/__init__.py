# tidysurvey — the config-driven survey pipeline (refactor, branch dev).
#
# Lean surface: one module per stage + the shared engine. Retired first-gen
# machinery (per-chip registration internals, hard-cut mosaic + meta_mosaic,
# clip-and-fill, geo neighbour helpers) lives on branch `main` / git history.
# One compatibility promise is kept: register_survey_by_chips retains its
# exact signature, running the audited dense internals.

from . import config
from . import fields
from . import cog
from . import validate
from . import report
from . import calibrate

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
    'config', 'fields', 'cog', 'validate', 'report', 'calibrate',
    # stitch + boundaries
    'seam_merge', 'define_hull_tiled', 'find_seamlines',
    'generate_combined_boundaries',
    # align
    'register_survey_dense', 'register_survey_by_chips',
    # sentinel
    'download_sentinel2_bands', 'pick_scene',
]

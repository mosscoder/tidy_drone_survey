# tidysurvey

Config-driven pipeline for drone survey processing: **stitch** flight orthos
into one mosaic, **align** multispectral (and GCP-less seasonal) imagery to a
ground-surveyed reference, **calibrate** to Sentinel-2 surface reflectance,
and document every run with a **quality report**.

One TOML per survey drives everything — the command line never decides
pipeline shape:

```
tidysurvey run --config 2024.toml       # scene pick → stitch → align → stitch → calibrate → report
```

or stage by stage:

```
tidysurvey scenes    --config 2024.toml                       # preview the Sentinel-2 scene menu
tidysurvey stitch    --config 2024.toml --product visible     # the base map (the survey's anchor)
tidysurvey align     --config 2024.toml                       # each MS mission → one smooth field
tidysurvey stitch    --config 2024.toml --product multispectral
tidysurvey calibrate --config 2024.toml                       # reflectance + reliability rasters
tidysurvey report    --config 2024.toml                       # regenerate the quality report
```

Every run leaves the same shape on disk:

```
<run_dir>/products/            the deliverables — copy this one folder = shipped
<run_dir>/products/quality/    reliability rasters + quality_report.html
<run_dir>/work/                intermediates, deletable once the run is accepted
```

Geometric truth is declared in the config, exactly one of `georeferencing =
"gcp"` (summer: the orthos carry ground control; their stitched product
becomes the anchor) or `anchor = "<prior base map>"` (spring/fall: no GCPs;
the visible layer is aligned to a previous summer's product first).

Hard gates exist only where a threshold is physical (seams ≤ 20 cm, interiors
byte-identical). Registration agreement and calibration MAE ship as evidence
rasters beside the products — read per survey, no fixed thresholds.

## Install

```
pip install -e .                      # core
pip install -e ".[match,sentinel,report]"   # LoFTR matching, GEE download, report thumbnails
```

Python ≥ 3.10 (`tomli` is pulled in below 3.11).

## Provenance

The methods are the audited implementations from the 2024 front-country
audit (seam texture-r 0.40 → 0.62 with interiors byte-identical; registration
agreement with the anchor 0.685 → 0.863; calibration held-out band R² 0.864
with the 7-MAE reliability raster). Design documents: `production_refactor.md`
and `refactor_walkthrough_2024ms.md` in the audit repository. First-generation
machinery (per-chip registration, hard-cut mosaic, clip-and-fill) was retired
on this branch and lives in git history; `register_survey_by_chips` keeps its
signature and now runs the dense engine.

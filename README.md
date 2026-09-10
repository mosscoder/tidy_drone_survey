# tidysurvey

Config-driven pipeline for drone survey processing: **stitch** flight orthos
into one mosaic, **align** multispectral (and GCP-less seasonal) imagery to a
ground-surveyed reference, **calibrate** to Sentinel-2 surface reflectance,
and document every run with a **quality report**.

One TOML per survey drives everything — the command line never decides
pipeline shape:

```
tidysurvey run --config 2024.toml            # scene → stitch → align → stitch → calibrate → tiles → report
tidysurvey run --config 2024.toml --detach   # same, fire-and-forget → tail -f <run_dir>/run.log
```

`run` is **rerunnable**: completed stages leave durable outputs and skip
themselves next time (delete an output to redo its stage), aligned missions
are kept individually, and a chosen Sentinel-2 scene stays chosen. After any
interruption the refire command is just the same `run` again. `--detach`
launches it immune to a closed terminal or session cleanup, logging to
`<run_dir>/run.log` with a timestamp on every line.

or stage by stage:

```
tidysurvey scenes    --config 2024.toml                       # preview the Sentinel-2 scene menu
tidysurvey stitch    --config 2024.toml --product visible     # the base map (the survey's anchor)
tidysurvey align     --config 2024.toml                       # each MS mission → one smooth field
tidysurvey stitch    --config 2024.toml --product multispectral
tidysurvey calibrate --config 2024.toml                       # reflectance + reliability rasters
tidysurvey report    --config 2024.toml                       # regenerate the quality report
tidysurvey tiles     --config 2024.toml                       # visible base -> .pmtiles web map
tidysurvey serve     --config 2024.toml                       # view products/ locally (opens the web map)
```

`tiles` runs inside every plan by default (`web_map = true` under
`[visible]`; set false to skip) and renders the visible COG into a single-file
[PMTiles](https://docs.protomaps.com/pmtiles/) archive (WebMercator XYZ,
512 px WEBP tiles, max zoom derived from the GSD) plus a Leaflet viewer
(`*_map.html`) beside it — a slippy map servable from any static host or
bucket that supports byte-range requests, no tile server. View locally with
`tidysurvey serve`, which opens the web map in your browser (`--open report`
for the quality report, `--open none` to suppress) and byte-serves it —
python's stock `http.server` ignores Range headers and cannot serve PMTiles.
It is a viewing artifact like the COG's overviews; analysis stays on the COG.

Every run leaves the same shape on disk:

```
<run_dir>/products/            the deliverables — copy this one folder = shipped
<run_dir>/products/quality/    reliability rasters + quality_report.html
<run_dir>/work/                intermediates, deletable once the run is accepted
```

## Publish (optional): run on fast local disk, ship to durable storage

By default everything lands under `run_dir` and stays there. A survey may
instead run on a local SSD and, **only after a fully successful run**, move its
bulky finished data off to durable NAS homes — leaving the lightweight quality
record behind. Opt in with a `[publish]` section:

```toml
[publish]
# the two COGs, renamed to the consumer basemap names:
#   <survey>_visible_<gsd>.tif        -> basemap_dir/visible.tif
#   <survey>_ms_calibrated_<gsd>.tif  -> basemap_dir/multispectral.tif
basemap_dir = "/Volumes/GIS/MPG_basemap/Raster/imagery/drone/2024/summer"
# the .pmtiles web map (+ its _map.html) and the whole work/ tree:
archive_dir = "/Volumes/home/nas_data/tidy_survey/2024_front_country_summer"
```

Omit the section entirely and nothing moves (the historical behaviour). What
**never** moves is the survey's durable record, kept in the local `run_dir`:
`products/quality/` (report + reliability rasters + calibration model) and
`products/_run_manifest.json`.

The move is idempotent and crash-safe — each file copies to a `.partial`, is
size-verified, then atomically swapped into place, so an interrupted publish
resumes rather than corrupts. On completion a `products/_published.json` marker
**seals** the run: `tidysurvey run` on a published survey is a no-op that prints
where the data went (and no longer depends on the remote inputs still existing).
Clear the marker to rebuild.

After publish the web map lives on the NAS — view it with `tidysurvey serve
--dir <archive_dir>`; `tidysurvey serve` from the run_dir still serves the local
quality report.

Geometric truth is declared in the config, exactly one of `georeferencing =
"gcp"` (summer: the orthos carry ground control; their stitched product
becomes the anchor) or `anchor = "<prior base map>"` (spring/fall: no GCPs;
the visible layer is aligned to a previous summer's product first).

Hard gates exist only where a threshold is physical (seams ≤ 20 cm, interiors
byte-identical). Registration agreement and calibration MAE ship as evidence
rasters beside the products — read per survey, no fixed thresholds.

## Bands and validity (the band law)

`tidysurvey/bands.py` is the one place that decides which bands are data and
which is validity. DroneDeploy's exports, as verified on the bucket:

| export | bands | validity |
|---|---|---|
| visible | RGBA | band 4, tagged alpha |
| multispectral 2024 | R, G, NIR, RE, NIR again | none (no alpha, nodata unset) |
| multispectral 2025+ | R, G, NIR, RE, NIR again, alpha | band 6, tagged alpha |

The colour tags on the multispectral files are wrong (band 3 is tagged "blue"
and is NIR), so only the alpha tag is trusted. `[multispectral] bands` in the
config names the first N non-alpha bands, in order; the duplicate and any
untagged trailing band are dropped, never carried. Validity is the tagged
alpha where one exists; without one it is derived from the data
(border-connected zero = nodata, interior zero islands = data). Every writer
tags its alpha band and names its bands, so a registered mission is
`Red, Green, NIR, RedEdge, alpha`, never the source alpha resampled as data.

## Per-cell evidence

Every registration writes `<out>.cells.tif` beside the registered mission:
one float32 raster on the engine's 64-pixel cell lattice (NaN = nodata),
bands `d_cm, dx_cm, dy_cm, n_matches, field_x_cm, field_y_cm, coverage`
(`registration.CELL_BANDS`). The first three are the RAW pooled displacements
the matches measured, the honest error signal; `field_*` is the smooth
correction the warp applied, sampled at the cell centres; `coverage` is 1
where a cell matched, 0 where it was expected to and did not, NaN where no
match was expected. `validate.registration_r_cells(..., like=<cells.tif>)`
scores agreement with the anchor on that same lattice, so the two stack band
for band and across seasons by georeference.

Every stitch writes an ownership raster (`ownership_out`): band 1 is the
owner index with the seam band marked `N + 1`, band 2 the owner index
everywhere, and the `names` tag maps values to inputs. A later stitch can
recycle it (`seam_merge(owner_in=...)`, or `geometry_only=True` for the
seams from the alphas alone): the multispectral stitch of a season reuses
the visible stitch's partition when the missions are the same flights, and
falls back to the distance rule wherever the recycled owner is absent or not
valid. `tidysurvey stitch --product multispectral` does this by itself when
the visible ownership exists and the mission names match.

## Install

```
pip install -e .                      # core
pip install -e ".[match,sentinel,report,tiles]"   # LoFTR, GEE, report thumbnails, web map
```

Python ≥ 3.10 (`tomli` is pulled in below 3.11).

## Credentials

The TOML never holds credentials — it names an environment variable
(`credentials_env = "MPG_PROJECTS_CREDENTIALS"`) whose value is the path to a
GEE service-account JSON. Export it in the shell, or put it in a `.env` file
**beside the survey TOML** (plain `KEY=VALUE` lines); loading a config applies
that file with setdefault semantics, so an exported variable always wins.

## Provenance

The methods are the audited implementations from the 2024 front-country
audit (seam texture-r 0.40 → 0.62 with interiors byte-identical; registration
agreement with the anchor 0.685 → 0.863; calibration held-out band R² 0.864
with the 7-MAE reliability raster). Design documents: `production_refactor.md`
and `refactor_walkthrough_2024ms.md` in the audit repository. First-generation
machinery (per-chip registration, hard-cut mosaic, clip-and-fill) was retired
on this branch and lives in git history; `register_survey_by_chips` keeps its
signature and now runs the dense engine.

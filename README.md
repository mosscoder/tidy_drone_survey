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
rasters beside the products — read per survey, no fixed thresholds. A
registration is also flagged for a look when more than 0.5 percent of the map
moved over 50 cm or when agreement did not rise. The flag does not stop a run.

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

## Registration and its numbers

Registration uses eFOLKI, a dense optical flow, in `tidysurvey/efolki.py`. It
measures the shift at every pixel of the reference map's own grid, with the
mission averaged onto that grid. Nothing is measured on a coarser copy. Each
block of the output is then warped once, cubic, from the mission's own pixels,
onto a grid whose corner is on the lattice of its pixel size, so the season
stitch copies it without resampling.
The settings are window radii 32, 24, 16 and 8, eight iterations per window,
rank radius 4 and three pyramid levels. The sources, the tests and what is our
own choice are in the audit's `02_registration/METHOD.md`.

Every registration reports three numbers, measured on the reference's pixels.
They are agreement with the reference before and after, as Pearson r of the
grey images, the median move, and the area that moved more than 50 cm. The
summary returned by `register_survey_dense` and its `qa_json` carry them as
`r_before`, `r_after`, `move_median_cm`, `moved_over_50cm_ha` and
`moved_over_50cm_pct`, with `look` and a one-line verdict in `line`.

Every registration writes `<out>.quality.tif` beside the registered mission.
It is one float32 raster with NaN as no-data, on cells of 64 reference pixels,
which is 2.1 m on a 3.3 cm map. Its bands are `r_before`, `r_after`, `move_cm`,
`move_east_cm` and `move_north_cm`, named in `registration.QUALITY_BANDS`. The
move bands say how far the mission's content moved, east and north.

The seam merge measures its seams with eFOLKI too, inside the render, on the
merge grid. Each pair of flights that meet in a render block is measured in the
box around their faultline, padded by twice eFOLKI's reach. The offsets are
split over the flights at every pixel of the 1 m band. Outside the band nothing
moves or blends. The stage report lists each seam with its length, the median
offset on the faultline, the metres of seam over 50 cm and the median misfit of
the split.

Every stitch writes an ownership raster (`ownership_out`): band 1 is the
owner index with the seam band marked `N + 1`, band 2 the owner index
everywhere, and the `names` tag maps values to inputs. A later stitch can
recycle it (`seam_merge(owner_in=...)`, or `geometry_only=True` for the
seams from the alphas alone): the multispectral stitch of a season reuses
the visible stitch's partition when the missions are the same flights, and
falls back to the distance rule wherever the recycled owner is absent or not
valid. `tidysurvey stitch --product multispectral` does this by itself when
the visible ownership exists and the mission names match.

## Changes

- **0.4.2** (2026-11-01). The registered map's edges. DroneDeploy's no-data mask
  decides which output pixels have data, pixel by pixel at full resolution. Until
  now the decision was made on squares of 8 by 8 output pixels, so a square more
  than half data was all data, and its empty pixels came out as black data along
  every edge. A shift counts as measured only on the intersection of the two maps'
  data masks with a negative buffer of eFOLKI's largest window, 32 px. Every other
  pixel takes the nearest measured shift, so an edge moves as one piece with the
  ground beside it. Beyond an edge eFOLKI matched fill, not ground, and placed the
  edge about 4 px wrong, with junk shifts of thousands of pixels in the corners
  where neither map has data. One of those sized the warp's read of the native
  pixels past OpenCV's limit, and the register job died on a segmentation fault
  five times out of five on the 2025 spring mission 00. The read is now sized only
  by the pixels the output keeps, and a shift past eFOLKI's reach is dropped. The
  context read around each block is 512 px instead of 1024, which the settings
  allow and which cuts eFOLKI's work by a third for the same result to within
  floating-point noise. On a CPU eFOLKI's window sums are cumulative sums instead
  of convolutions, the same sums in another order, which halves a seam block on
  the season VM. Evidence for the edge rule is the audit's known-warp test,
  `02_registration/matcher_ablation/poc/edge_known_warp.py`. Four new tests.
- **0.4.1** (2026-11-01). One resample per pixel, cubic, everywhere it can be.
  The season stitch reads each source onto its grid cubic instead of `average`.
  A source on the grid's lattice is still copied exactly, and one off it, an
  unregistered reference map, keeps more of its fine detail. The interiors gate
  reads the same way. The seam band warps cubic instead of bilinear, and stays
  exact where the taper is zero. Registration rounds its cubic warp to the
  nearest DN where it used to truncate, which left every registered pixel half
  a DN dark on average. `register_survey_dense` takes `blocks_at_once`, or
  `TIDYSURVEY_BLOCKS_AT_ONCE`, the number of blocks it works on at once.
  eFOLKI still runs one block at a time while the others read and warp. New
  tests check each of these. A map registered to itself comes back byte for
  byte, a known shift comes back as one cubic sample of the native pixels, and
  the stitch copies on-lattice maps exactly and reads off-lattice ones cubic.
- **0.4.0** (2026-10-31). eFOLKI replaces LoFTR in registration and in the seam
  merge, as settled in the audit's `02_registration/METHOD.md`. Registration
  measures on the reference's own pixels in blocks of 4096 with 1024 pixels of
  context, warps once from the mission's own pixels, and scores in the same
  pass. It writes `<out>.quality.tif` with five bands in place of
  `<out>.cells.tif`. Its summary carries the three numbers and `look`, and the
  keys `d_med_cm`, `cells_tif`, `tiles` and `match` are gone. The seam merge
  measures each seam block by block during the render and splits the offsets
  at each band pixel. The LoFTR walk, its checkpoint and the node solve are
  gone. The stage report's seams carry `seam_m`, `med_offset_cm`, `over50_m`
  and `residual_cm`. The `match` extra is torch alone. The interiors gate skips
  a sample when the clamp into one composite block moves its window off the
  owner's clear ground. On the 2025 fall mission_11 against the 2025 summer map
  at 3.3 cm, 0.4.0 gave r 0.44 to 0.51, a median move of 11 cm and 0.03 ha over
  50 cm, which matches the audit, in 18 minutes on an Apple GPU. tidysurvey is
  now GPL-3.0, because `efolki.py` ports the authors' GPL-3.0 code.
- **0.3.4** (2026-09-23) — two fixes found by the first season stitch on a cloud VM. `cog.finalize_cog`
  fetches the GDAL bindings through the lazy import (0.3.3 still named the module-level `gdal` that
  the lazy-import change removed, so every COG finalize raised `NameError`). `tiles.build_pmtiles`
  checks the archive against the zooms it actually wrote: a survey smaller than a pixel at the lowest
  zooms writes no tiles there, the pmtiles writer stamps the range it holds, and 0.3.3 failed the
  build (the run loop logs "web map failed" and carries on without the viewer). Tests write a real
  COG and a real archive for a 160 m survey (`tests/test_cog_tiles.py`).
- **0.3.3** (2026-09-17) — heartbeat hook: `TIDYSURVEY_HEARTBEAT=<path>` makes the scorer's block
  loop, the match chunk's tile loop and the warp's block loop touch that file (≤ 1/s); a caller's
  watchdog can detect a stalled loop independent of map size. Off unless the variable is set.
  Motivation: two cloud registrations of the 2026-09-08 collaring multispectral map blocked in the
  second scoring pass until the one-hour cap; the cause is still unknown and this makes the next
  hang fail fast with a stack dump (the dump is the caller's job). `tidysurvey/heartbeat.py`,
  `tests/test_heartbeat.py`.
- **0.3.2** — registration memory independent of map size. The dense
  registration no longer materialises the native-resolution validity mask
  (one byte per output pixel, 4.8 GB for a 92k × 52k ortho) or the
  FIELD_DS correction field; each warp block gathers its validity window from
  the coarse mask through the exact nearest-neighbour offsets `cv2.resize`
  would have used, and pulls its field from a `fields.StreamedField` that
  holds only the horizontal pass of the same resize and serves column bands
  bit-for-bit. Registered map, `cells.tif` and QA are byte-identical to
  0.3.1 (verified with the real matcher on a synthetic pair with a varying
  field); peak memory is a function of block size, halo and worker count,
  and every array a warp worker touches is allocated once per worker. A
  20k × 20k synthetic mission with two workers: 2421 MiB peak on 0.3.1,
  1420 on 0.3.2, and the same at 10k × 10k (`tests/test_field_streamed.py`,
  `TIDYSURVEY_SLOW_TESTS=1`).
- **0.3.1** — `calibrate.reference = "none"`: a plan that ends at the
  stitched mosaic (no scene selection, no calibration).
- **0.3.0** — per-cell evidence raster, the scorer on the engine lattice,
  ownership recycling between the visible and multispectral stitches.

## Install

```
pip install -e .                      # core
pip install -e ".[match,sentinel,report,tiles]"   # eFOLKI (torch), GEE, report thumbnails, web map
```

Python ≥ 3.10 (`tomli` is pulled in below 3.11).

## Credentials

The TOML never holds credentials — it names an environment variable
(`credentials_env = "MPG_PROJECTS_CREDENTIALS"`) whose value is the path to a
GEE service-account JSON. Export it in the shell, or put it in a `.env` file
**beside the survey TOML** (plain `KEY=VALUE` lines); loading a config applies
that file with setdefault semantics, so an exported variable always wins.

## Licence

GPL-3.0, in `LICENSE`. `tidysurvey/efolki.py` is a port of the eFOLKI reference
code by Plyer and colleagues at github.com/aplyer/gefolki, which is GPL-3.0, so
tidysurvey is too.

## Provenance

The methods are the audited implementations from the 2024 front-country
audit (seam texture-r 0.40 → 0.62 with interiors byte-identical; registration
agreement with the anchor 0.685 → 0.863; calibration held-out band R² 0.864
with the 7-MAE reliability raster). Design documents: `production_refactor.md`
and `refactor_walkthrough_2024ms.md` in the audit repository. First-generation
machinery (per-chip registration, hard-cut mosaic, clip-and-fill) was retired
on this branch and lives in git history; `register_survey_by_chips` keeps its
signature and now runs the eFOLKI engine. Since 0.4.0 registration and the
seams use eFOLKI, and the LoFTR engine is at tag `v0.3.4`.

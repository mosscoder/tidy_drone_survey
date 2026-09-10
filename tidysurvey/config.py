"""One TOML per survey drives the whole pipeline.

The config declares WHAT the survey is (named inputs, bands, the source of
geometric truth); the pipeline derives WHERE everything lands (`Paths`) and
WHICH stages run (`Config.plan()`). File paths are not settings — the run
layout under `run_dir` is a guarantee of the tool, identical for every survey:

    <run_dir>/products/                  the deliverables (copy this = shipped)
    <run_dir>/products/quality/          reliability rasters + quality report
    <run_dir>/work/                      intermediates, deletable once accepted

Geometric truth is declared explicitly, exactly one of:
    georeferencing = "gcp"               orthos carry ground control; their
                                         stitched product becomes the anchor
    anchor = "<path to prior base map>"  no GCPs; the visible layer is aligned
                                         to this before stitching (spring/fall)

Resolutions may be "auto": the median of the named inputs' native pixel
sizes, rounded to the millimetre (resolved once, at load). Mission dates
resolve the same way: an explicit per-mission date = "YYYY-MM-DD" wins,
otherwise the date is read from the raster's own metadata tags at load
(DroneDeploy's acquisitionStartDate, then TIFFTAG_DATETIME).

Credentials never live in the TOML — it only NAMES an environment variable
(credentials_env). A `.env` beside the TOML may supply that variable's value
(the path to a key file); variables already exported in the shell always win.
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field as dc_field
from datetime import date as _iso_date
from pathlib import Path
from typing import List, Optional, Union

try:  # tomllib is stdlib from 3.11; tomli is the same parser for 3.10
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml


# --------------------------------------------------------------------------- #
# pieces
# --------------------------------------------------------------------------- #
@dataclass
class NamedInput:
    """Every input is named; the name follows it through registered files,
    seam tables and report rows, so every number traces to an input."""
    name: str
    path: str
    date: Optional[str] = None      # 'YYYY-MM-DD'; omitted -> read from raster
                                    # tags at load (explicit value = override)


@dataclass
class VisibleCfg:
    resolution_m: Union[float, str] = "auto"
    web_map: bool = True                # ship .pmtiles + Leaflet viewer with the base
    orthos: List[NamedInput] = dc_field(default_factory=list)


@dataclass
class MultispectralCfg:
    resolution_m: Union[float, str] = "auto"
    bands: List[str] = dc_field(default_factory=lambda: ["Red", "Green", "NIR", "RedEdge"])
    missions: List[NamedInput] = dc_field(default_factory=list)


@dataclass
class StitchCfg:
    band_width_m: float = 1.0        # blend width; the one aesthetic knob
    gauge: str = "free"              # "free" = split evenly; "anchored" = pin to the first input
    seam_tripwire_cm: float = 20.0   # hard gate on post-stitch seams


@dataclass
class CalibrateCfg:
    reference: str = "sentinel-2"    # "none" = no calibration: the plan ends at the stitched mosaic
    date: str = "auto"               # "auto" = median mission date + scene menu; or 'YYYY-MM-DD'
    search_days: int = 14
    max_scene_cloud_pct: float = 20.0
    band_map: dict = dc_field(default_factory=lambda: {"B4": "Red", "B3": "Green",
                                                       "B8": "NIR", "B5": "RedEdge"})
    block_m: float = 100.0           # tile size — a reasoned choice, not tuned
    model: str = "ridge"             # per-tile cross-band linear fit
    blend: str = "bilinear"          # coefficient interpolation; exact for a linear fit
    clip: str = "scene"              # clamp output to the reference scene's range


@dataclass
class PublishCfg:
    """The one sanctioned output-destination knob: an OPT-IN terminal step
    that, after a successful run, moves the bulky finished data off the local
    run_dir to durable NAS homes. Omit the [publish] section entirely and the
    pipeline behaves exactly as before — everything stays in run_dir.

    Two destinations, because finished data splits by consumer:
      basemap_dir  the visible + calibrated-MS COGs, renamed to the consumer
                   basemap names (visible.tif / multispectral.tif)
      archive_dir  the bulk that is not a basemap consumable: the .pmtiles web
                   map (+ its _map.html) and the whole work/ tree
    The quality report, reliability rasters, calibration model and manifest are
    NEVER moved — they stay in the local run_dir as the survey's durable record.
    """
    basemap_dir: Optional[str] = None
    archive_dir: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.basemap_dir and self.archive_dir)


def gsd_slug(res_m: float) -> str:
    """0.032 -> '3p2cm', 0.059 -> '5p9cm', 0.10 -> '10cm'."""
    cm = res_m * 100.0
    txt = f"{cm:g}"
    return txt.replace(".", "p") + "cm"


class Paths:
    """The run layout, derived from run_dir + survey. Every stage writes
    through this object, so the on-disk tree is a guarantee, not a habit."""

    def __init__(self, run_dir: str, survey: str,
                 visible_res: Optional[float] = None, ms_res: Optional[float] = None):
        self.run_dir = Path(run_dir)
        self.survey = survey
        self._vres = visible_res
        self._mres = ms_res

        self.products = self.run_dir / "products"
        self.quality = self.products / "quality"
        self.work = self.run_dir / "work"
        self.registered = self.work / "registered"
        self.reports = self.work / "reports"

    # -- deliverables (self-describing names: survey slug + resolved GSD) --- #
    def _need(self, res, what):
        if res is None:
            raise ValueError(f"{what} resolution not resolved yet — "
                             "load the config with resolve_auto=True")
        return res

    @property
    def visible_base(self) -> Path:
        r = self._need(self._vres, "visible")
        return self.products / f"{self.survey}_visible_{gsd_slug(r)}.tif"

    @property
    def ms_calibrated(self) -> Path:
        r = self._need(self._mres, "multispectral")
        return self.products / f"{self.survey}_ms_calibrated_{gsd_slug(r)}.tif"

    @property
    def visible_pmtiles(self) -> Path:
        r = self._need(self._vres, "visible")
        return self.products / f"{self.survey}_visible_{gsd_slug(r)}.pmtiles"

    @property
    def manifest(self) -> Path:
        return self.products / "_run_manifest.json"

    # -- quality (shipped evidence) ----------------------------------------- #
    @property
    def reg_qa(self) -> Path:
        return self.quality / "ms_registration_reliability_7p55m.tif"

    @property
    def visible_reg_qa(self) -> Path:
        return self.quality / "visible_registration_reliability_7p55m.tif"

    @property
    def calib_qa(self) -> Path:
        return self.quality / "ms_calibrated_reliability_100m.tif"

    @property
    def calib_model(self) -> Path:
        return self.quality / "calibration_model.json"

    @property
    def report_html(self) -> Path:
        return self.quality / "quality_report.html"

    @property
    def report_json(self) -> Path:
        return self.quality / "quality_report.json"

    # -- work (intermediates) ------------------------------------------------ #
    @property
    def ms_mosaic(self) -> Path:
        return self.work / "ms_uncalibrated.tif"

    @property
    def ms_mosaic_ownership(self) -> Path:
        return self.work / "ms_uncalibrated.ownership.tif"

    @property
    def visible_ownership(self) -> Path:
        return self.work / "visible_base.ownership.tif"

    def stage_report(self, stage: str) -> Path:
        return self.reports / f"{stage}.json"

    def sentinel_scene(self, date_yymmdd: str = "{date}") -> Path:
        return self.work / f"sentinel_2_bands_{date_yymmdd}.tif"

    def ensure(self):
        for d in (self.products, self.quality, self.work, self.registered, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class Config:
    survey: str
    crs: str
    run_dir: str
    georeferencing: Optional[str] = None    # "gcp" | None
    anchor: Optional[str] = None            # path to a prior base map | None
    credentials_env: Optional[str] = None
    visible: VisibleCfg = dc_field(default_factory=VisibleCfg)
    ms: MultispectralCfg = dc_field(default_factory=MultispectralCfg)
    stitch: StitchCfg = dc_field(default_factory=StitchCfg)
    calibrate: CalibrateCfg = dc_field(default_factory=CalibrateCfg)
    publish: PublishCfg = dc_field(default_factory=PublishCfg)
    paths: Paths = None
    source: Optional[str] = None            # the TOML this came from
    dotenv_loaded: int = 0                  # vars applied from a .env beside the TOML

    # ---------------------------------------------------------------- plan -- #
    def plan(self) -> List[str]:
        """The stage list is DERIVED from the config — the command line never
        decides pipeline shape. A borrowed anchor inserts the visible-align
        pass; a multispectral layer appends align/stitch/calibrate."""
        stages = []
        calibrating = bool(self.ms.missions) and self.calibrate.reference not in (None, "", "none")
        if calibrating:
            stages.append("scene")                       # settled before heavy work
        if self.anchor:
            stages.append("align/visible")               # borrowed truth: align first
        if self.visible.orthos:
            stages.append("stitch/visible")
        if self.ms.missions:
            stages += ["align/ms", "stitch/ms"] + (["calibrate"] if calibrating else [])
        if self.visible.orthos and self.visible.web_map:
            stages.append("tiles")              # the web map, from the finished base
        stages.append("report")
        if self.publish.enabled:
            stages.append("publish")            # opt-in: move finished data to the NAS
        return stages

    def credentials_path(self) -> Optional[str]:
        if not self.credentials_env:
            return None
        return os.environ.get(self.credentials_env)

    def snapshot(self) -> dict:
        """The resolved config (autos filled in), for the manifest + report."""
        return {
            "survey": self.survey, "crs": self.crs, "run_dir": self.run_dir,
            "georeferencing": self.georeferencing, "anchor": self.anchor,
            "visible": {"resolution_m": self.visible.resolution_m,
                        "web_map": self.visible.web_map,
                        "orthos": [vars(o) for o in self.visible.orthos]},
            "multispectral": {"resolution_m": self.ms.resolution_m, "bands": self.ms.bands,
                              "missions": [vars(m) for m in self.ms.missions]},
            "stitch": vars(self.stitch), "calibrate": dict(vars(self.calibrate)),
            "publish": vars(self.publish),
            "plan": self.plan(), "source": self.source,
        }


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def _load_dotenv_beside(config_path) -> int:
    """Apply a `.env` sitting next to the survey TOML: plain KEY=VALUE lines
    (# comment lines and an `export ` prefix tolerated, matching quotes
    stripped). Setdefault semantics — a variable already exported in the
    shell ALWAYS wins, so the file is a fallback, never an override. This is
    the ergonomic home for credentials_env's variable; the value is still a
    path to a key file, never key material."""
    envf = Path(config_path).resolve().parent / ".env"
    if not envf.is_file():
        return 0
    applied = 0
    for line in envf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not key or any(ch.isspace() for ch in key):
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key not in os.environ:
            os.environ[key] = val
            applied += 1
    return applied


def _named(items) -> List[NamedInput]:
    out = []
    for it in items or []:
        out.append(NamedInput(name=it["name"], path=it["path"], date=it.get("date")))
    return out


def _median_native_gsd(inputs: List[NamedInput]) -> float:
    """The 'auto' rule: median of the inputs' native pixel sizes, to the mm."""
    import rasterio
    res = []
    for it in inputs:
        with rasterio.open(it.path) as s:
            res.append(abs(s.res[0]))
    return round(statistics.median(res), 3)


_DATE_TAGS = ("acquisitionStartDate", "acquisitionEndDate", "TIFFTAG_DATETIME")


def _date_from_tags(tags: dict) -> Optional[str]:
    """'YYYY-MM-DD' from raster metadata: DroneDeploy's acquisitionStartDate
    (ISO), else acquisitionEndDate, else baseline TIFFTAG_DATETIME (which
    writes 'YYYY:MM:DD HH:MM:SS'). None when nothing parseable is present."""
    for key in _DATE_TAGS:
        val = tags.get(key)
        if not val:
            continue
        d = val[:10].replace(":", "-") if key == "TIFFTAG_DATETIME" else val[:10]
        try:
            _iso_date.fromisoformat(d)
            return d
        except ValueError:
            continue
    return None


def _scan_missions(missions: List[NamedInput], need_gsd: bool) -> Optional[float]:
    """One header-open per mission serves both autos: native GSDs (when the
    resolution is 'auto') and acquisition dates from tags for missions
    without an explicit date. Missions whose header holds no usable date
    keep date=None — pick_scene medians over whatever dates exist."""
    import rasterio
    res = []
    for m in missions:
        need_date = m.date is None
        if not (need_gsd or need_date):
            continue
        with rasterio.open(m.path) as s:
            if need_gsd:
                res.append(abs(s.res[0]))
            if need_date:
                m.date = _date_from_tags(s.tags())
    return round(statistics.median(res), 3) if res else None


def survey_bounds(inputs: List[NamedInput], out_epsg: int) -> tuple:
    """Union footprint of a set of rasters as (minx, miny, maxx, maxy) in
    EPSG:out_epsg. The scene picker needs the WHOLE survey AOI for its cloud
    estimate: fed a single mission (as it once was), it can read 0.0% cloud from
    that mission's clear corner of a scene that is 46% clouded over the full
    survey, and lock that scene as the calibration reference. One header-open
    per input (bounds only), each reprojected to the survey CRS and unioned."""
    import rasterio
    from rasterio.warp import transform_bounds
    box = None
    for it in inputs:
        with rasterio.open(it.path) as s:
            b, src_epsg = s.bounds, s.crs.to_epsg()
        l, bm, r, t = ((b.left, b.bottom, b.right, b.top) if src_epsg == out_epsg
                       else transform_bounds(f"EPSG:{src_epsg}", f"EPSG:{out_epsg}",
                                             b.left, b.bottom, b.right, b.top))
        box = (l, bm, r, t) if box is None else (
            min(box[0], l), min(box[1], bm), max(box[2], r), max(box[3], t))
    if box is None:
        raise ValueError("survey_bounds: no readable inputs")
    return box


def load(path: str, resolve_auto: bool = True) -> Config:
    """Parse + validate a survey TOML. A `.env` beside it is applied first
    (variables already in the environment win). With resolve_auto (default),
    "auto" resolutions and missing mission dates are resolved by opening each
    named input (any GDAL-openable path: local, gs://, https://, /vsicurl/...);
    dates come from the rasters' own metadata tags."""
    dotenv_n = _load_dotenv_beside(path)
    raw = _toml.loads(Path(path).read_text())

    cfg = Config(
        survey=raw["survey"],
        crs=raw.get("crs", "EPSG:6514"),
        run_dir=raw.get("run_dir", "."),
        georeferencing=raw.get("georeferencing"),
        anchor=raw.get("anchor"),
        credentials_env=raw.get("credentials_env"),
        source=str(path),
        dotenv_loaded=dotenv_n,
    )

    # exactly one source of geometric truth — neither/both is a loud error
    has_gcp = cfg.georeferencing is not None
    has_anchor = cfg.anchor is not None
    if has_gcp and cfg.georeferencing != "gcp":
        raise ValueError(f'georeferencing = "{cfg.georeferencing}" — the only '
                         'accepted value is "gcp" (or set anchor = "<path>" instead)')
    if has_gcp == has_anchor:
        raise ValueError(
            'declare where geometric truth comes from: exactly ONE of '
            'georeferencing = "gcp" (orthos carry ground control) or '
            'anchor = "<path to a prior GCP-stitched base map>". '
            f'Got georeferencing={cfg.georeferencing!r}, anchor={cfg.anchor!r}.')
    if has_anchor and resolve_auto and not Path(cfg.anchor).exists() \
            and "://" not in cfg.anchor and not cfg.anchor.startswith("/vsi"):
        raise FileNotFoundError(f"anchor not found: {cfg.anchor}")

    v = raw.get("visible", {})
    cfg.visible = VisibleCfg(resolution_m=v.get("resolution_m", "auto"),
                             web_map=bool(v.get("web_map", True)),
                             orthos=_named(v.get("orthos")))
    m = raw.get("multispectral", {})
    cfg.ms = MultispectralCfg(resolution_m=m.get("resolution_m", "auto"),
                              bands=m.get("bands", ["Red", "Green", "NIR", "RedEdge"]),
                              missions=_named(m.get("missions")))
    s = raw.get("stitch", {})
    cfg.stitch = StitchCfg(band_width_m=s.get("band_width_m", 1.0),
                           gauge=s.get("gauge", "free"),
                           seam_tripwire_cm=s.get("seam_tripwire_cm", 20.0))
    c = raw.get("calibrate", {})
    cfg.calibrate = CalibrateCfg(
        reference=c.get("reference", "sentinel-2"),
        date=c.get("date", "auto"),
        search_days=c.get("search_days", 14),
        max_scene_cloud_pct=c.get("max_scene_cloud_pct", 20.0),
        band_map=c.get("band_map", {"B4": "Red", "B3": "Green", "B8": "NIR", "B5": "RedEdge"}),
        block_m=c.get("block_m", 100.0),
        model=c.get("model", "ridge"),
        blend=c.get("blend", "bilinear"),
        clip=c.get("clip", "scene"),
    )
    pb = raw.get("publish", {})
    cfg.publish = PublishCfg(basemap_dir=pb.get("basemap_dir"),
                             archive_dir=pb.get("archive_dir"))

    vres = cfg.visible.resolution_m
    mres = cfg.ms.resolution_m
    if resolve_auto:
        if vres == "auto" and cfg.visible.orthos:
            vres = _median_native_gsd(cfg.visible.orthos)
            cfg.visible.resolution_m = vres
        if cfg.ms.missions:
            scanned = _scan_missions(cfg.ms.missions, need_gsd=(mres == "auto"))
            if mres == "auto":
                mres = scanned
                cfg.ms.resolution_m = mres
    cfg.paths = Paths(cfg.run_dir, cfg.survey,
                      visible_res=vres if isinstance(vres, float) else None,
                      ms_res=mres if isinstance(mres, float) else None)
    return cfg


def write_manifest(cfg: Config, extra: dict = None) -> Path:
    """Config snapshot + resolved autos (+ scene lock, versions, timings)."""
    cfg.paths.ensure()
    manifest = {"config": cfg.snapshot()}
    if cfg.paths.manifest.exists():
        try:
            manifest = json.loads(cfg.paths.manifest.read_text())
            manifest["config"] = cfg.snapshot()
        except Exception:
            pass
    if extra:
        manifest.update(extra)
    cfg.paths.manifest.write_text(json.dumps(manifest, indent=2))
    return cfg.paths.manifest

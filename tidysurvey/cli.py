"""tidysurvey — four commands driven by one config file.

    tidysurvey scenes    --config config/2024.toml          # preview the scene menu
    tidysurvey stitch    --config ... --product visible     # 1. base map
    tidysurvey align     --config ... [--product ms]        # 2. align missions
    tidysurvey stitch    --config ... --product multispectral
    tidysurvey calibrate --config ...                       # 4. reflectance + reliability
    tidysurvey report    --config ...                       # regenerate the quality report
    tidysurvey tiles     --config ...                       # visible base -> .pmtiles web map
    tidysurvey serve     --config ...                       # view products/ locally (byte ranges)
    tidysurvey run       --config ...                       # the whole plan, in order
    #   plan: scene -> stitch/visible -> align/ms -> stitch/ms -> calibrate -> tiles -> report
    tidysurvey run       --config ... --detach               # same, fire-and-forget

`run` is RERUNNABLE: completed stages leave durable outputs and skip
themselves on the next run (delete an output to redo its stage), so after
any interruption the refire command is simply the same `run` again.
--detach launches it immune to terminal close / session cleanup and
appends everything to <run_dir>/run.log.

The PLAN is derived from the TOML, never from flags: a borrowed `anchor`
inserts the visible-align pass; a multispectral layer appends align/stitch/
calibrate; the run always ends with the quality report — which is written
even when a gate halts the run.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from . import config as _config
from . import mem as _mem


def say(msg=""):
    """Every pipeline line is timestamped — hours-long stages must leave a
    log that can answer 'when did it stop and why' on its own."""
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def _install_death_rattle():
    """A killed run should confess in its own log. SIGTERM/SIGHUP get logged
    with a timestamp before dying (SIGKILL can't be caught — but then the
    log's last timestamp still dates the kill); faulthandler dumps a
    traceback on hard crashes."""
    faulthandler.enable()

    def _die(signum, frame):
        name = signal.Signals(signum).name
        say(f"!! received {name} (pid {os.getpid()}) — terminating")
        try:
            say(f"   mem at signal: {_mem.line()}")
        except Exception:
            pass
        sys.exit(128 + signum)

    for s in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(s, _die)
        except (ValueError, OSError):    # non-main thread / unsupported
            pass


def _banner(stage, i, n):
    say("")
    say(f"── stage {i}/{n} · {stage} " + "─" * max(1, 60 - len(stage)))


# --------------------------------------------------------------------------- #
# stage executors (thin wrappers over the package — the CLI holds no logic)
# --------------------------------------------------------------------------- #
def _scene(cfg, args):
    from . import sentinel
    rescan = bool(getattr(args, "rescan", False))
    if cfg.calibrate.date not in (None, "auto") and not rescan:
        say(f"  scene fixed in config: {cfg.calibrate.date} (no search)")
        _config.write_manifest(cfg, {"scene": {"date": cfg.calibrate.date, "source": "config"}})
        return {"date": cfg.calibrate.date}
    if not rescan and cfg.paths.manifest.exists():
        try:
            locked = (json.loads(cfg.paths.manifest.read_text()).get("scene") or {}).get("date")
        except Exception:
            locked = None
        if locked:
            say(f"  scene already locked: {locked} (manifest; --rescan to search again)")
            return {"date": locked}
    chosen = sentinel.pick_scene(
        bounds_raster=None if not cfg.ms.missions else cfg.ms.missions[0].path,
        mission_dates=[m.date for m in cfg.ms.missions if m.date] or None,
        target_date=None if cfg.calibrate.date == "auto" else cfg.calibrate.date,
        search_days=cfg.calibrate.search_days,
        max_scene_cloud_pct=cfg.calibrate.max_scene_cloud_pct,
        gee_credentials_path=cfg.credentials_path(),
        interactive=not getattr(args, "non_interactive", False),
        log=say,
    )
    _config.write_manifest(cfg, {"scene": chosen})
    return chosen


def _stitch(cfg, product):
    from . import cog, merge, validate
    paths = cfg.paths.ensure()
    if product == "visible":
        inputs = [(o.name, o.path) for o in cfg.visible.orthos]
        if cfg.anchor:  # borrowed truth: stitch the ALIGNED orthos, not the raw ones
            inputs = [(o.name, str(paths.registered / f"visible_{o.name}.tif"))
                      for o in cfg.visible.orthos]
        # stream the master into work/; only the finalized COG lands in products/
        out = paths.work / "_visible_stitch_master.tif"
        final = paths.visible_base
        ownership = paths.visible_ownership
        rep_path = paths.stage_report("stitch_visible")
        res = cfg.visible.resolution_m
    else:
        inputs = [(m.name, str(paths.registered / f"{m.name}.tif")) for m in cfg.ms.missions]
        out = paths.ms_mosaic          # intermediate: calibrate finalizes its product
        final = None
        ownership = paths.ms_mosaic_ownership
        rep_path = paths.stage_report("stitch_ms")
        res = cfg.ms.resolution_m
    rep = merge.seam_merge(inputs, str(out), band_width_m=cfg.stitch.band_width_m,
                           gauge=cfg.stitch.gauge,
                           res=res if isinstance(res, float) else None,
                           out_crs=cfg.crs, ownership_out=str(ownership),
                           report_json=str(rep_path), log=say)
    validate.seam_tripwire(rep, max_cm=cfg.stitch.seam_tripwire_cm).assert_ok()
    validate.assert_interiors_unchanged(inputs, str(out), str(ownership),
                                        band_width_m=cfg.stitch.band_width_m,
                                        log=say).assert_ok()
    if final is not None:
        # gates passed on the master; deliver as COG — lossless ZSTD base,
        # WEBP overviews (the visible viewing recipe). Base pixels unchanged.
        cog.finalize_cog(str(out), str(final), lossless=False, log=say)
        out.unlink()
        rep["out"] = str(final)
        rep_path.write_text(json.dumps(rep, indent=2))
    return rep


def _align(cfg, product):
    from . import registration, validate
    paths = cfg.paths.ensure()
    if product == "visible":
        if not cfg.anchor:
            raise SystemExit("align --product visible needs a borrowed anchor in the "
                             "config (this survey's orthos carry GCPs)")
        items = cfg.visible.orthos
        reference = cfg.anchor
        qa_out = paths.visible_reg_qa
        prefix = "visible_"
        rep_path = paths.stage_report("align_visible")
        scored = paths.visible_base          # scored after stitch; per-mission here
    else:
        items = cfg.ms.missions
        reference = str(cfg.paths.visible_base)
        qa_out = paths.reg_qa
        prefix = ""
        rep_path = paths.stage_report("align_ms")
        scored = paths.ms_mosaic
    summaries = []
    for it in items:
        out = paths.registered / f"{prefix}{it.name}.tif"
        qa = paths.reports / f"align_{prefix}{it.name}.json"
        if out.exists() and qa.exists():
            # resume: a mission is ~an hour of dense matching; keep completed
            # ones (the qa json only lands after a successful write — a killed
            # run leaves no qa, so partial outputs are redone, not trusted)
            say(f"    {it.name}: already registered, skipping "
                f"(delete {out.name} to redo)")
            s = json.loads(qa.read_text())
        else:
            s = registration.register_survey_dense(
                it.path, reference, str(out), qa_json=str(qa), log=say)
        s["name"] = it.name
        summaries.append(s)
    rep = dict(kind="align", product=product, reference=str(reference),
               missions=summaries)
    rep_path.write_text(json.dumps(rep, indent=2))
    # the promoted evidence map: per-cell r of the stitched product vs the
    # reference — written after the matching stitch exists; score now if it does
    if scored.exists():
        sc = validate.registration_r_cells(str(scored), str(reference), str(qa_out), log=say)
        (paths.reports / f"reg_r_cells_{product}.json").write_text(json.dumps(sc, indent=2))
    else:
        say(f"  (r map deferred: {scored.name} not built yet — "
              "it is scored right after its stitch)")
    return rep


def _score_after_stitch(cfg, product):
    """The per-cell agreement map for the freshly stitched product."""
    from . import validate
    paths = cfg.paths
    if product == "visible" and cfg.anchor:
        sc = validate.registration_r_cells(str(paths.visible_base), cfg.anchor,
                                           str(paths.visible_reg_qa), log=say)
        (paths.reports / "reg_r_cells_visible.json").write_text(json.dumps(sc, indent=2))
    if product == "multispectral":
        sc = validate.registration_r_cells(str(paths.ms_mosaic), str(paths.visible_base),
                                           str(paths.reg_qa), log=say)
        (paths.reports / "reg_r_cells_ms.json").write_text(json.dumps(sc, indent=2))


def _calibrate(cfg, args):
    from . import calibrate, sentinel, validate, cog
    paths = cfg.paths.ensure()
    manifest = json.loads(paths.manifest.read_text()) if paths.manifest.exists() else {}
    scene_date = (manifest.get("scene") or {}).get("date") or cfg.calibrate.date
    if scene_date in (None, "auto"):
        raise SystemExit("no scene locked — run `tidysurvey scenes` (or `run`) first, "
                         "or set an explicit calibrate.date")
    s2 = sentinel.download_sentinel2_bands(
        bounds_raster=str(paths.ms_mosaic), target_date=scene_date,
        output_path=str(paths.sentinel_scene()), bands=list(cfg.calibrate.band_map.keys()),
        band_names=list(cfg.calibrate.band_map.values()),
        gee_credentials_path=cfg.credentials_path())
    model = calibrate.fit_block_ridge(str(paths.ms_mosaic), s2, log=say,
                                      block_m=cfg.calibrate.block_m,
                                      band_names=tuple(cfg.calibrate.band_map.values()))
    tmp = paths.work / "_ms_calibrated_master.tif"
    calibrate.apply_bilinear(str(paths.ms_mosaic), model, str(tmp), log=say)
    cog.finalize_cog(str(tmp), str(paths.ms_calibrated), lossless=True, log=say)
    tmp.unlink(missing_ok=True)
    qa_summary = calibrate.perblock_qa(model, str(paths.calib_qa), crs=cfg.crs, log=say)
    pooled = calibrate.pooled_oof_r2(model, log=say)
    calibrate.save_model(model, str(paths.calib_model))
    rep = dict(kind="calibrate", scene_date=scene_date, qa_summary=qa_summary,
               pooled_oof=pooled, out=str(paths.ms_calibrated))
    paths.stage_report("calibrate").write_text(json.dumps(rep, indent=2))
    return rep


def _stage_done(cfg, stage):
    """A completed stage leaves its product AND its stage report (reports are
    written last), so a rerun of `run` can skip it. Delete the product to
    redo a stage. align/* resume per mission; scene honors the manifest."""
    p = cfg.paths
    if stage == "stitch/visible" and p.visible_base.exists() \
            and p.stage_report("stitch_visible").exists():
        return f"{p.visible_base.name} exists"
    if stage == "stitch/ms" and p.ms_mosaic.exists() \
            and p.stage_report("stitch_ms").exists():
        return f"{p.ms_mosaic.name} exists"
    if stage == "calibrate" and p.ms_calibrated.exists() \
            and p.stage_report("calibrate").exists():
        return f"{p.ms_calibrated.name} exists"
    if stage == "tiles" and p.visible_pmtiles.exists() \
            and p.stage_report("tiles").exists():
        return f"{p.visible_pmtiles.name} exists"
    return None


def _score_if_missing(cfg, product):
    """Backfill the r map when a stitch is skipped but a prior death landed
    between the stitch and its scoring."""
    qa = cfg.paths.visible_reg_qa if product == "visible" else cfg.paths.reg_qa
    if not qa.exists():
        _score_after_stitch(cfg, product)


def _tiles(cfg, args):
    from . import tiles
    paths = cfg.paths.ensure()
    rep = tiles.build_pmtiles(str(paths.visible_base), str(paths.visible_pmtiles),
                              min_zoom=args.min_zoom, max_zoom=args.max_zoom,
                              log=say)
    paths.stage_report("tiles").write_text(json.dumps(rep, indent=2))
    return rep


def _serve(cfg, args):
    from .serve import serve_dir
    root = Path(args.dir) if args.dir else cfg.paths.products
    base = f"http://localhost:{args.port}"
    say(f"serving {root}")
    maps = sorted(root.glob("*_map.html"))
    for m in maps:
        say(f"  map:    {base}/{m.name}")
    rep_html = root / "quality" / "quality_report.html"
    if rep_html.exists():
        say(f"  report: {base}/quality/quality_report.html")
    say(f"  root:   {base}/   (Ctrl-C stops)")

    # what to open in the browser on start (default: the map — it's why serve exists)
    targets = {
        "map": f"{base}/{maps[0].name}" if maps else None,
        "report": f"{base}/quality/quality_report.html" if rep_html.exists() else None,
        "root": f"{base}/",
    }
    want = getattr(args, "open_target", "map")
    open_url = None
    if want != "none":
        open_url = targets.get(want) or targets["map"] or targets["report"] or targets["root"]
        say(f"  opening {open_url} in your browser …")
    serve_dir(root, args.port, open_url=open_url, log=say)
    return True


def _report(cfg, failed_stage=None):
    from . import report
    return report.build(cfg, failed_stage=failed_stage, log=say)


# --------------------------------------------------------------------------- #
# publish — the opt-in terminal step: move the bulky finished data to the NAS,
# leaving only the quality record local. Idempotent + resumable via the
# products/_published.json marker; runs only on a fully successful run.
# --------------------------------------------------------------------------- #
def _published_status(cfg):
    """'done' | 'in_progress' | None — read from the run_dir's publish marker."""
    marker = cfg.paths.products / "_published.json"
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text()).get("status")
    except Exception:
        return None


def _crash_safe_move(src, dst, log=say):
    """Move a file so a crash leaves either the source or a COMPLETE
    destination, never a half-file. Same-share = atomic rename; cross-share
    (e.g. local SSD -> /Volumes/GIS) = copy to <dst>.partial, verify byte size,
    atomic os.replace into place, then drop the source. Idempotent: a source
    already gone with the destination present counts as done."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        if dst.exists():
            return "already"
        raise FileNotFoundError(f"publish: source vanished before move: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)                 # same filesystem: atomic + instant
        return "renamed"
    except OSError:
        pass                                 # cross-device — copy via a .partial
    tmp = dst.parent / (dst.name + ".partial")
    shutil.copy2(src, tmp)
    if tmp.stat().st_size != src.stat().st_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"publish: size mismatch copying {src} -> {dst}")
    os.replace(tmp, dst)                      # atomic swap into place
    src.unlink()
    return "copied"


def _move_tree(src, dst, log=say):
    """Move a whole directory (work/). Same-share = instant rename; cross-share
    = shutil.move (copytree + rmtree). Refuses to merge into an existing dst."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return "already" if dst.exists() else "missing"
    if dst.exists():
        raise FileExistsError(f"publish: destination already present, refusing "
                              f"to merge: {dst} (resolve by hand)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)                  # same filesystem: instant
        return "renamed"
    except OSError:
        shutil.move(str(src), str(dst))      # cross-device: copytree + rmtree
        return "copied"


def _rewrite_report_maplink(cfg, map_name):
    """The web map moved to the NAS archive; the quality report stays local.
    Point its map reference at the new home (both substrings are literals
    emitted by report.build, so this is survey-independent)."""
    rep = cfg.paths.report_html
    if not map_name or not rep.exists():
        return
    archive = cfg.publish.archive_dir
    html = rep.read_text()
    html = html.replace(f'href="../{map_name}"',
                        f'href="file://{archive}/{map_name}"')
    html = html.replace(
        "(lives beside the archive in products/)",
        f"(published to the NAS archive — view with: "
        f"tidysurvey serve --dir {archive})")
    rep.write_text(html)


def _publish(cfg):
    """Terminal, opt-in: move the bulky finished data off the local run_dir to
    its NAS homes — the visible + calibrated-MS COGs -> basemap_dir under the
    consumer names visible.tif / multispectral.tif; the .pmtiles web map (+ its
    _map.html) and work/ -> archive_dir — leaving the quality report,
    reliability rasters and manifest in the local run_dir. Driven by
    products/_published.json so it is idempotent and resumable after an
    interrupted move (on resume it needs no resolved config — the move list is
    self-contained)."""
    p = cfg.paths
    pub = cfg.publish
    marker_path = p.products / "_published.json"
    m = json.loads(marker_path.read_text()) if marker_path.exists() else None
    if m and m.get("status") == "done":
        say(f"  already published → {pub.basemap_dir} · {pub.archive_dir}")
        return m

    if m is None:
        # first publish: build the move list from the resolved product paths
        basemap, archive = Path(pub.basemap_dir), Path(pub.archive_dir)
        moves = []

        def add(src, dst, tree=False):
            moves.append({"src": str(src), "dst": str(dst),
                          "tree": tree, "done": False})

        add(p.visible_base, basemap / "visible.tif")
        vaux = Path(str(p.visible_base) + ".aux.xml")
        if vaux.exists():
            add(vaux, basemap / "visible.tif.aux.xml")
        add(p.ms_calibrated, basemap / "multispectral.tif")
        maux = Path(str(p.ms_calibrated) + ".aux.xml")
        if maux.exists():
            add(maux, basemap / "multispectral.tif.aux.xml")
        add(p.visible_pmtiles, archive / p.visible_pmtiles.name)
        map_html = p.visible_pmtiles.with_name(p.visible_pmtiles.stem + "_map.html")
        map_name = map_html.name if map_html.exists() else None
        if map_name:
            add(map_html, archive / map_name)
        add(p.work, archive / "work", tree=True)
        m = {"status": "in_progress",
             "published_at": time.strftime("%Y-%m-%d %H:%M"),
             "basemap_dir": pub.basemap_dir, "archive_dir": pub.archive_dir,
             "map_html": map_name, "moves": moves}
        marker_path.write_text(json.dumps(m, indent=2))

    for mv in m["moves"]:
        if mv.get("done"):
            continue
        res = (_move_tree(mv["src"], mv["dst"]) if mv.get("tree")
               else _crash_safe_move(mv["src"], mv["dst"]))
        say(f"  {Path(mv['src']).name} → {mv['dst']}  [{res}]")
        mv["done"] = True
        marker_path.write_text(json.dumps(m, indent=2))

    _rewrite_report_maplink(cfg, m.get("map_html"))
    mani = json.loads(p.manifest.read_text()) if p.manifest.exists() else {}
    mani["published"] = {"at": m["published_at"], "basemap_dir": pub.basemap_dir,
                         "archive_dir": pub.archive_dir}
    p.manifest.write_text(json.dumps(mani, indent=2))
    m["status"] = "done"
    marker_path.write_text(json.dumps(m, indent=2))
    return m


# --------------------------------------------------------------------------- #
def main(argv=None):
    # progress must stream through pipes/tee live, not sit in an 8 KB buffer
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    _install_death_rattle()
    ap = argparse.ArgumentParser(prog="tidysurvey", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "scenes", "stitch", "align",
                                        "calibrate", "report", "tiles", "serve"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--product", choices=["visible", "multispectral", "ms"],
                    default=None, help="for stitch/align")
    ap.add_argument("--date", default=None, help="override calibrate.date")
    ap.add_argument("--non-interactive", action="store_true",
                    help="headless: auto-accept the suggested scene")
    ap.add_argument("--detach", action="store_true",
                    help="run detached: survives terminal close/session cleanup; "
                         "appends to <run_dir>/run.log")
    ap.add_argument("--rescan", action="store_true",
                    help="scenes: search again even when a scene is already locked")
    ap.add_argument("--min-zoom", type=int, default=8, help="tiles: lowest zoom")
    ap.add_argument("--max-zoom", type=int, default=None,
                    help="tiles: highest zoom (default: derived from the GSD)")
    ap.add_argument("--port", type=int, default=8080, help="serve: port")
    ap.add_argument("--dir", default=None, help="serve: directory (default products/)")
    ap.add_argument("--open", dest="open_target",
                    choices=["map", "report", "root", "none"], default="map",
                    help="serve: open this in the browser on start (default map; "
                         "none = don't open)")
    args = ap.parse_args(argv)

    if args.detach:
        import subprocess
        lite = _config.load(args.config, resolve_auto=False)
        log_path = Path(lite.run_dir) / "run.log"
        raw = list(argv) if argv is not None else sys.argv[1:]
        child = [sys.executable, "-u", "-m", "tidysurvey"] + \
                [a for a in raw if a != "--detach"]
        if "--non-interactive" not in child:
            child.append("--non-interactive")      # a detached run has no keyboard
        with open(log_path, "a") as lf:
            proc = subprocess.Popen(child, stdout=lf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        say(f"detached: pid {proc.pid} · follow with  tail -f {log_path}")
        return 0

    # a published run is TERMINAL: check the seal before the (remote) resolve so
    # re-running a shipped survey never depends on its inputs still existing
    if args.command == "run":
        lite = _config.load(args.config, resolve_auto=False)
        if lite.publish.enabled:
            sealed = _published_status(lite)
            if sealed == "done":
                say(f"tidysurvey · {lite.survey} · already published")
                say(f"  basemap {lite.publish.basemap_dir}")
                say(f"  archive {lite.publish.archive_dir}")
                say(f"  nothing to do — clear "
                    f"{lite.paths.products / '_published.json'} to rebuild")
                return 0
            if sealed == "in_progress":
                say(f"tidysurvey · {lite.survey} · resuming interrupted publish")
                _publish(lite)
                return 0

    # serve only needs run_dir/products — skip the remote-header resolve
    cfg = _config.load(args.config, resolve_auto=(args.command != "serve"))
    if args.date:
        cfg.calibrate.date = args.date
    product = {"ms": "multispectral"}.get(args.product, args.product)

    say(f"tidysurvey · {cfg.survey} · {cfg.crs} · pid {os.getpid()}")
    truth = (f'anchor = {cfg.anchor}' if cfg.anchor
             else 'georeferencing = "gcp" (stitched visible becomes the anchor)')
    say(f"  config OK — {truth}")
    if cfg.dotenv_loaded:
        say(f"  env     .env beside config applied ({cfg.dotenv_loaded} var"
            f"{'s' if cfg.dotenv_loaded != 1 else ''}; exported shell vars win)")
    say(f"  plan    {' → '.join(cfg.plan())}")
    say(f"  run_dir {cfg.run_dir}  (products/ = deliverables · work/ = intermediates)")

    # memory telemetry: a breadcrumb every ~30s so a silent SIGKILL (jetsam on a
    # memory-starved box) can be told apart from an external kill by the last
    # line before the log goes dark. Daemon thread; harmless to every command.
    sampler = _mem.Sampler(say).start()
    sampler.sample_now("·start")

    if args.command == "scenes":
        return _scene(cfg, args) and 0
    if args.command == "stitch":
        return _stitch(cfg, product or "visible") and 0
    if args.command == "align":
        return _align(cfg, product or "multispectral") and 0
    if args.command == "calibrate":
        return _calibrate(cfg, args) and 0
    if args.command == "report":
        return _report(cfg) and 0
    if args.command == "tiles":
        return _tiles(cfg, args) and 0
    if args.command == "serve":
        return _serve(cfg, args) and 0

    # ---- run: the whole plan, report always written ------------------------- #
    plan = cfg.plan()
    t0 = time.time()
    failed = None
    try:
        for i, stage in enumerate(plan, 1):
            if stage == "publish":
                continue                    # terminal move — run at the seam, below
            _banner(stage, i, len(plan))
            sampler.set_stage(stage)
            sampler.sample_now()
            done = _stage_done(cfg, stage)
            if done:
                say(f"  {done} — skipping (delete it to redo this stage)")
                if stage == "stitch/visible":
                    _score_if_missing(cfg, "visible")
                elif stage == "stitch/ms":
                    _score_if_missing(cfg, "multispectral")
                continue
            if stage == "scene":
                _scene(cfg, args)
            elif stage == "align/visible":
                _align(cfg, "visible")
            elif stage == "stitch/visible":
                _stitch(cfg, "visible")
                _score_after_stitch(cfg, "visible")
            elif stage == "align/ms":
                _align(cfg, "multispectral")
            elif stage == "stitch/ms":
                _stitch(cfg, "multispectral")
                _score_after_stitch(cfg, "multispectral")
            elif stage == "calibrate":
                _calibrate(cfg, args)
            elif stage == "tiles":
                try:
                    _tiles(cfg, args)
                except ImportError as e:        # optional extra; a missing
                    say(f"  web map skipped ({e}) — "   # viewer must not kill
                        "pip install -e '.[tiles]'")    # a 15-hour run
                except Exception as e:
                    say(f"  web map failed ({e}) — continuing; "
                        "rerun `tidysurvey tiles` after the run")
            elif stage == "report":
                pass                                    # built in finally, below
    except Exception as e:
        failed = f"{stage}: {e}"
        say(f"✗ {failed}")
        sampler.sample_now("·FAILED")
        raise
    finally:
        _banner("report", plan.index("report") + 1, len(plan))
        sampler.set_stage("report")
        _report(cfg, failed_stage=failed)
        _config.write_manifest(cfg, {"finished": time.strftime("%Y-%m-%d %H:%M"),
                                     "seconds": round(time.time() - t0, 1),
                                     "failed_stage": failed})
    # success only (a failed stage re-raised above): the opt-in move to the NAS
    if cfg.publish.enabled:
        _banner("publish", len(plan), len(plan))
        sampler.set_stage("publish")
        sampler.sample_now()
        _publish(cfg)
        say(f"✓ run complete — {time.time() - t0:.0f}s · published →")
        say(f"    basemap  {cfg.publish.basemap_dir}")
        say(f"    archive  {cfg.publish.archive_dir}")
        say(f"    local record kept: {cfg.paths.products}")
    else:
        say(f"✓ run complete — {time.time() - t0:.0f}s · to ship this survey, "
            f"copy one folder: {cfg.paths.products}")
    sampler.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""tidysurvey — four commands driven by one config file.

    tidysurvey scenes    --config config/2024.toml          # preview the scene menu
    tidysurvey stitch    --config ... --product visible     # 1. base map
    tidysurvey align     --config ... [--product ms]        # 2. align missions
    tidysurvey stitch    --config ... --product multispectral
    tidysurvey calibrate --config ...                       # 4. reflectance + reliability
    tidysurvey report    --config ...                       # regenerate the quality report
    tidysurvey run       --config ...                       # the whole plan, in order

The PLAN is derived from the TOML, never from flags: a borrowed `anchor`
inserts the visible-align pass; a multispectral layer appends align/stitch/
calibrate; the run always ends with the quality report — which is written
even when a gate halts the run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import config as _config


def _banner(stage, i, n):
    print(f"\n── stage {i}/{n} · {stage} " + "─" * max(1, 60 - len(stage)), flush=True)


# --------------------------------------------------------------------------- #
# stage executors (thin wrappers over the package — the CLI holds no logic)
# --------------------------------------------------------------------------- #
def _scene(cfg, args):
    from . import sentinel
    if cfg.calibrate.date not in (None, "auto") and not getattr(args, "rescan", False):
        print(f"  scene fixed in config: {cfg.calibrate.date} (no search)")
        _config.write_manifest(cfg, {"scene": {"date": cfg.calibrate.date, "source": "config"}})
        return {"date": cfg.calibrate.date}
    chosen = sentinel.pick_scene(
        bounds_raster=None if not cfg.ms.missions else cfg.ms.missions[0].path,
        mission_dates=[m.date for m in cfg.ms.missions if m.date] or None,
        target_date=None if cfg.calibrate.date == "auto" else cfg.calibrate.date,
        search_days=cfg.calibrate.search_days,
        max_scene_cloud_pct=cfg.calibrate.max_scene_cloud_pct,
        gee_credentials_path=cfg.credentials_path(),
        interactive=not getattr(args, "non_interactive", False),
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
                           report_json=str(rep_path))
    validate.seam_tripwire(rep, max_cm=cfg.stitch.seam_tripwire_cm).assert_ok()
    validate.assert_interiors_unchanged(inputs, str(out), str(ownership),
                                        band_width_m=cfg.stitch.band_width_m).assert_ok()
    if final is not None:
        # gates passed on the master; deliver as COG — lossless ZSTD base,
        # WEBP overviews (the visible viewing recipe). Base pixels unchanged.
        cog.finalize_cog(str(out), str(final), lossless=False)
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
            print(f"    {it.name}: already registered, skipping "
                  f"(delete {out.name} to redo)")
            s = json.loads(qa.read_text())
        else:
            s = registration.register_survey_dense(
                it.path, reference, str(out), qa_json=str(qa))
        s["name"] = it.name
        summaries.append(s)
    rep = dict(kind="align", product=product, reference=str(reference),
               missions=summaries)
    rep_path.write_text(json.dumps(rep, indent=2))
    # the promoted evidence map: per-cell r of the stitched product vs the
    # reference — written after the matching stitch exists; score now if it does
    if scored.exists():
        sc = validate.registration_r_cells(str(scored), str(reference), str(qa_out))
        (paths.reports / f"reg_r_cells_{product}.json").write_text(json.dumps(sc, indent=2))
    else:
        print(f"  (r map deferred: {scored.name} not built yet — "
              "it is scored right after its stitch)")
    return rep


def _score_after_stitch(cfg, product):
    """The per-cell agreement map for the freshly stitched product."""
    from . import validate
    paths = cfg.paths
    if product == "visible" and cfg.anchor:
        sc = validate.registration_r_cells(str(paths.visible_base), cfg.anchor,
                                           str(paths.visible_reg_qa))
        (paths.reports / "reg_r_cells_visible.json").write_text(json.dumps(sc, indent=2))
    if product == "multispectral":
        sc = validate.registration_r_cells(str(paths.ms_mosaic), str(paths.visible_base),
                                           str(paths.reg_qa))
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
    model = calibrate.fit_block_ridge(str(paths.ms_mosaic), s2,
                                      block_m=cfg.calibrate.block_m,
                                      band_names=tuple(cfg.calibrate.band_map.values()))
    tmp = paths.work / "_ms_calibrated_master.tif"
    calibrate.apply_bilinear(str(paths.ms_mosaic), model, str(tmp))
    cog.finalize_cog(str(tmp), str(paths.ms_calibrated), lossless=True)
    tmp.unlink(missing_ok=True)
    qa_summary = calibrate.perblock_qa(model, str(paths.calib_qa), crs=cfg.crs)
    pooled = calibrate.pooled_oof_r2(model)
    calibrate.save_model(model, str(paths.calib_model))
    rep = dict(kind="calibrate", scene_date=scene_date, qa_summary=qa_summary,
               pooled_oof=pooled, out=str(paths.ms_calibrated))
    paths.stage_report("calibrate").write_text(json.dumps(rep, indent=2))
    return rep


def _report(cfg, failed_stage=None):
    from . import report
    return report.build(cfg, failed_stage=failed_stage)


# --------------------------------------------------------------------------- #
def main(argv=None):
    # progress must stream through pipes/tee live, not sit in an 8 KB buffer
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(prog="tidysurvey", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "scenes", "stitch", "align",
                                        "calibrate", "report"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--product", choices=["visible", "multispectral", "ms"],
                    default=None, help="for stitch/align")
    ap.add_argument("--date", default=None, help="override calibrate.date")
    ap.add_argument("--non-interactive", action="store_true",
                    help="headless: auto-accept the suggested scene")
    args = ap.parse_args(argv)

    cfg = _config.load(args.config)
    if args.date:
        cfg.calibrate.date = args.date
    product = {"ms": "multispectral"}.get(args.product, args.product)

    print(f"tidysurvey · {cfg.survey} · {cfg.crs}")
    truth = (f'anchor = {cfg.anchor}' if cfg.anchor
             else 'georeferencing = "gcp" (stitched visible becomes the anchor)')
    print(f"  config OK — {truth}")
    if cfg.dotenv_loaded:
        print(f"  env     .env beside config applied ({cfg.dotenv_loaded} var"
              f"{'s' if cfg.dotenv_loaded != 1 else ''}; exported shell vars win)")
    print(f"  plan    {' → '.join(cfg.plan())}")
    print(f"  run_dir {cfg.run_dir}  (products/ = deliverables · work/ = intermediates)")

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

    # ---- run: the whole plan, report always written ------------------------- #
    plan = cfg.plan()
    t0 = time.time()
    failed = None
    try:
        for i, stage in enumerate(plan, 1):
            _banner(stage, i, len(plan))
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
            elif stage == "report":
                pass                                    # built in finally, below
    except Exception as e:
        failed = f"{stage}: {e}"
        print(f"\n✗ {failed}", file=sys.stderr)
        raise
    finally:
        _banner("report", len(plan), len(plan))
        _report(cfg, failed_stage=failed)
        _config.write_manifest(cfg, {"finished": time.strftime("%Y-%m-%d %H:%M"),
                                     "seconds": round(time.time() - t0, 1),
                                     "failed_stage": failed})
    print(f"\n✓ run complete — {time.time() - t0:.0f}s · to ship this survey, "
          f"copy one folder: {cfg.paths.products}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

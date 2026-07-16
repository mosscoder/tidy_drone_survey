"""End-to-end checks for the spring-2025 hiccup fixes (A: graceful scorer,
B: VI-R2 report honesty, C: local-union-anchor scoring).

No real config/rasters needed — cli._score_after_stitch and report.build both
duck-type `cfg`, so a SimpleNamespace with the paths they touch exercises the
real code paths. Synthetic 4-band RGBA rasters stand in for the visible base and
the union anchor.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import ColorInterp

from tidysurvey import cli, report

TMP = Path(tempfile.mkdtemp(prefix="hiccup_"))
W, H, RES = 1280, 896, 0.1                       # multiples of cell_px=128 -> 10x7 cells
TR = from_origin(0, H * RES, RES, RES)


def _rgba(path, seed):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(40, 210, (3, H, W)).astype(np.uint8)
    prof = dict(driver="GTiff", width=W, height=H, count=4, dtype="uint8",
                crs="EPSG:32611", transform=TR, tiled=True, compress="zstd")
    with rasterio.open(path, "w", **prof) as d:
        for b in range(3):
            d.write(rgb[b], b + 1)
        d.write(np.full((H, W), 255, np.uint8), 4)
        d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]


# ------------------------------------------------------------------ A + C -----
# scorer prefers the LOCAL union anchor (scores fine despite a bogus cfg.anchor),
# drops the union after, and degrades gracefully when there is nothing to score on.
def test_scorer():
    run = TMP / "run"
    work = run / "work"; reports = work / "reports"; qa = run / "quality"
    for d in (work, reports, qa):
        d.mkdir(parents=True, exist_ok=True)
    vbase = run / "visible_base.tif"
    union = work / "_anchor_union.tif"
    _rgba(vbase, 1); _rgba(union, 2)             # overlapping (identical grid)
    vqa = qa / "visible_registration_reliability.tif"
    vjson = reports / "reg_r_cells_visible.json"

    paths = SimpleNamespace(work=work, reports=reports, visible_base=vbase,
                            visible_reg_qa=vqa, reg_qa=qa / "ms.tif",
                            ms_mosaic=run / "ms.tif")
    cfg = SimpleNamespace(anchor="/bogus/does_not_exist.tif", paths=paths)

    # part 1: bogus NAS anchor + a present local union -> must score via the union
    cli._score_after_stitch(cfg, "visible")
    assert vqa.exists(), "FAIL: visible reliability raster not written (union not used)"
    assert vjson.exists(), "FAIL: reg_r_cells_visible.json not written"
    assert not union.exists(), "FAIL: union prewarp not dropped after scoring"
    print("A/C part1 PASS: scored via LOCAL union despite bogus cfg.anchor; union dropped")

    # part 2: no union + bogus anchor -> must SKIP gracefully (no raise, no output)
    vqa.unlink(missing_ok=True); vjson.unlink(missing_ok=True)
    cli._score_after_stitch(cfg, "visible")      # must not raise
    assert not vqa.exists(), "FAIL: wrote a reliability raster despite a failed score"
    assert not vjson.exists(), "FAIL: wrote json despite a failed score"
    print("A/C part2 PASS: bogus anchor + no union -> graceful skip (no raise, no output)")


# -------------------------------------------------------------------- B --------
# report excludes VI R2 (both the OLD flat shape and the NEW nested shape),
# keeps raw-band + mean R2, adds the footnote, and shows VI reliability as MAE.
def _build_report(pooled):
    run = TMP / f"rep_{abs(hash(json.dumps(pooled, sort_keys=True))) % 9999}"
    reports = run / "work" / "reports"; reports.mkdir(parents=True, exist_ok=True)
    (reports / "calibrate.json").write_text(json.dumps(dict(
        kind="calibrate", scene_date="2025-05-02", pooled_oof=pooled,
        qa_summary={"tiles": 3307,
                    "MAE_Red_x1e4": {"median": 54.67, "p90": 114.82},
                    "MAE_NDVI": {"median": 0.0231, "p90": 0.0594},
                    "MAE_NDRE": {"median": 0.0195, "p90": 0.0505},
                    "MAE_CIre": {"median": 0.0664, "p90": 0.2535}})))
    nope = run / "none.tif"
    paths = SimpleNamespace(reports=reports, manifest=run / "none.json",
                            visible_base=nope, ms_calibrated=nope, reg_qa=nope,
                            visible_reg_qa=nope, calib_qa=nope,
                            report_html=run / "quality_report.html",
                            report_json=run / "quality_report.json",
                            ensure=lambda: None)
    cfg = SimpleNamespace(survey="test", source=None, paths=paths, snapshot=lambda: {})
    report.build(cfg, log=lambda m: None)
    html = (run / "quality_report.html").read_text()
    return html.split("calibrate — held-out agreement")[1].split("</section>")[0], html


def test_report_vi_r2():
    # OLD flat shape — the exact spring numbers (negatives at the top level)
    old = {"Red": 0.897, "Green": 0.884, "NIR": 0.864, "RedEdge": 0.854,
           "NDVI": -0.334, "NDRE": -0.754, "CIre": 0.516, "mean_band_R2": 0.875}
    sec, html = _build_report(old)
    assert "-0.334" not in html and "-0.754" not in html, "FAIL: VI R2 leaked into the HTML"
    assert "0.897" in sec and "0.875" in sec, "FAIL: raw-band / mean R2 missing"
    assert "variance" in sec.lower(), "FAIL: variance-suppression footnote missing"
    assert "MAE_NDVI" in sec, "FAIL: VI reliability (MAE) missing"
    print("B (old flat shape) PASS: VI R2 excluded; raw+mean R2, footnote, VI-MAE present")

    # NEW nested shape — vi_r2 must be excluded (it's a dict), no crash
    new = {"Red": 0.897, "Green": 0.884, "NIR": 0.864, "RedEdge": 0.854,
           "mean_band_R2": 0.875, "vi_r2": {"NDVI": -0.334, "NDRE": -0.754, "CIre": 0.516}}
    sec2, html2 = _build_report(new)
    assert "-0.334" not in html2, "FAIL: nested VI R2 leaked into the HTML"
    assert "0.875" in sec2 and "vi_r2" not in sec2, "FAIL: nested shape rendered wrong"
    print("B (new nested shape) PASS: vi_r2 excluded cleanly")


if __name__ == "__main__":
    test_scorer()
    test_report_vi_r2()
    print("\nALL HICCUP-FIX CHECKS PASSED")

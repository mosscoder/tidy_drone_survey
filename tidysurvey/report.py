"""The run quality report: one self-contained HTML page + a JSON twin.

Assembled from the per-stage numbers in <work>/reports/*.json and the shipped
rasters — so it is REGENERABLE without re-processing (`tidysurvey report`).
Written even when a run fails: a failed run is exactly when the evidence
should be organized (the cover shows the breach, sections cover the stages
that ran). Thumbnails are inlined as base64; nothing external to break when
the file is copied.

Design reference: production_refactor_assets/quality_report_sample.html in
the audit repo (built from the real 2024 rasters).
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling


# --------------------------------------------------------------------------- #
# thumbnails (matplotlib optional — report degrades to tables-only without it)
# --------------------------------------------------------------------------- #
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _fig64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _overview(path, bands, w=640, resampling=Resampling.average):
    with rasterio.open(path) as src:
        h = int(w * src.height / src.width)
        return src.read(bands, out_shape=(len(bands), h, w),
                        resampling=resampling).astype(float)


def _stretch(a):
    m = np.isfinite(a) & (a > 0)
    if not m.any():
        return np.zeros_like(a)
    p1, p2 = np.percentile(a[m], [2, 98])
    return np.clip((a - p1) / max(p2 - p1, 1e-9), 0, 1)


def _thumb_product(path, bands, w=640):
    plt = _plt()
    if plt is None or not Path(path).exists():
        return None
    arr = _overview(path, bands, w)
    arr[arr == 0] = np.nan
    rgb = np.dstack([_stretch(np.nan_to_num(a)) for a in arr])
    rgb[~np.isfinite(arr[0])] = 1.0
    fig, ax = plt.subplots(figsize=(6, 6 * rgb.shape[0] / rgb.shape[1]), dpi=110)
    ax.imshow(rgb); ax.axis("off")
    return _fig64(fig)


def _thumb_r_map(path):
    plt = _plt()
    if plt is None or not Path(path).exists():
        return None
    with rasterio.open(path) as s:
        r = s.read(1).astype(float)
    cmap = plt.cm.Spectral.copy(); cmap.set_bad("#e3e3e3")
    fig, ax = plt.subplots(figsize=(5.2, 5.2 * r.shape[0] / r.shape[1]), dpi=110)
    im = ax.imshow(np.ma.masked_invalid(r), cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("texture-masked r vs anchor", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    return _fig64(fig)


def _thumb_qa_panels(path):
    plt = _plt()
    if plt is None or not Path(path).exists():
        return None
    with rasterio.open(path) as s:
        names = list(s.descriptions)
        qa = s.read().astype(float)
    fig, axs = plt.subplots(2, 4, figsize=(10.4, 6.2), dpi=110)
    for ax, name, band in zip(axs.flat, names, qa):
        v = band[np.isfinite(band)]
        if v.size == 0:
            ax.axis("off"); continue
        cm = (plt.cm.Spectral if name == "n_px" else plt.cm.magma).copy()
        cm.set_bad("#e8e8e8")
        im = ax.imshow(np.ma.masked_invalid(band), cmap=cm, interpolation="nearest",
                       vmin=float(np.floor(v.min())), vmax=float(np.percentile(v, 98)))
        med = np.median(v)
        ax.set_title(f"{name} — median {med:.0f}" if med >= 5 else
                     f"{name} — median {med:.3f}", fontsize=8.5)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle("Calibration reliability — cross-validated MAE per tile "
                 "(each panel on its own scale) + n_px", fontsize=9.5)
    return _fig64(fig)


# --------------------------------------------------------------------------- #
# html assembly
# --------------------------------------------------------------------------- #
_CSS = """
* { box-sizing:border-box; margin:0; }
body { background:#f4f5f7; color:#1a202c; font:14px/1.55 -apple-system,'Segoe UI',
       Roboto,Helvetica,Arial,sans-serif; font-variant-numeric:tabular-nums;
       padding:28px 16px 60px; }
.wrap { max-width:980px; margin:0 auto; }
header { background:#1f2937; color:#fff; border-radius:12px; padding:20px 24px;
         display:flex; justify-content:space-between; align-items:center; gap:16px;
         flex-wrap:wrap; }
header h1 { font-size:19px; font-weight:650; }
header .meta { color:#aeb6c2; font:12px/1.7 ui-monospace,Menlo,monospace; }
.badge { color:#fff; font-weight:700; font-size:13px; padding:9px 16px;
         border-radius:999px; white-space:nowrap; }
.pass { background:#1a7f37; } .fail { background:#b42318; }
section { background:#fff; border:1px solid #e3e6ea; border-radius:12px;
          padding:18px 22px; margin-top:16px; }
h2 { font-size:12px; font-weight:700; letter-spacing:1.4px; text-transform:uppercase;
     color:#5b6472; border-bottom:1px solid #e3e6ea; padding-bottom:8px;
     margin-bottom:12px; }
table { width:100%; border-collapse:collapse; margin-top:6px; }
th { font-size:11px; text-transform:uppercase; letter-spacing:.8px; color:#5b6472;
     text-align:left; padding:6px 8px; border-bottom:1.5px solid #e3e6ea; }
td { padding:6px 8px; border-bottom:1px solid #eef0f2; font-size:13px; }
.num { text-align:right; } .mono { font-family:ui-monospace,Menlo,monospace;
     font-size:12.5px; }
.ok { color:#1a7f37; font-weight:700; } .bad { color:#b42318; font-weight:700; }
figure { margin-top:12px; } figure img { max-width:100%; border:1px solid #e3e6ea;
     border-radius:8px; }
figcaption { font-size:12px; color:#5b6472; margin-top:6px; }
.cards { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
@media (max-width:760px){ .cards { grid-template-columns:1fr; } }
.note { color:#5b6472; font-size:12.5px; margin-top:8px; }
pre { background:#f8f9fa; border:1px solid #e3e6ea; border-radius:8px; padding:12px;
      font:12px/1.65 ui-monospace,Menlo,monospace; overflow-x:auto; margin-top:8px; }
footer { margin-top:20px; color:#5b6472; font-size:12px; line-height:1.7; }
"""


def _img(b64, caption=""):
    if not b64:
        return f'<p class="note">(thumbnail unavailable{": " + caption if caption else ""})</p>'
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="data:image/png;base64,{b64}">{cap}</figure>'


def _row(cells, tag="td"):
    return "<tr>" + "".join(f"<{tag}{' class=num' if isinstance(c, (int, float)) else ''}>"
                            f"{c}</{tag}>" for c in cells) + "</tr>"


def build(cfg, failed_stage=None, log=print):
    """Assemble quality_report.html + quality_report.json for this run from
    the stage reports and shipped rasters. Safe to call on partial runs."""
    t0 = time.time()
    paths = cfg.paths
    paths.ensure()

    stages = {}
    for p in sorted(Path(paths.reports).glob("*.json")):
        try:
            rep = json.loads(p.read_text())
        except Exception:
            continue
        # per-mission align checkpoints (align_batch_*) are resume state, not
        # report sections; the combined align report carries the missions table
        if rep.get("kind") == "align" and "missions" not in rep:
            continue
        stages[p.stem] = rep

    manifest = {}
    if paths.manifest.exists():
        try:
            manifest = json.loads(paths.manifest.read_text())
        except Exception:
            pass

    ok = failed_stage is None
    badge = ("✓ ALL CHECKS PASSED" if ok else f"✗ FAILED AT {failed_stage.upper()}")

    # ---- cover rows -------------------------------------------------------- #
    cover = []
    for name, rep in stages.items():
        kind = rep.get("kind", name)
        if kind == "stitch":
            seams = rep.get("seams", [])
            res = [s.get("residual_cm") for s in seams if s.get("residual_cm") is not None]
            cover.append((name, f"{len(rep.get('inputs', []))} inputs · "
                          f"{rep.get('n_seams', 0)} seams"
                          + (f" · worst residual {max(res):.1f} cm" if res else "")))
        elif kind == "align":
            shifts = sorted(m["d_med_cm"] for m in rep.get("missions", [])
                            if m.get("d_med_cm") is not None)
            n = len(shifts)
            med = (shifts[n // 2] if n % 2 else (shifts[n // 2 - 1] + shifts[n // 2]) / 2) \
                if shifts else None
            cover.append((name, f"{len(rep.get('missions', []))} missions"
                          + (f" · median shift {med:.1f} cm" if med is not None else "")))
        elif kind == "registration_r_cells":
            cover.append((name, f"median cell r {rep.get('median_r')} "
                                f"({rep.get('cells')} cells)"))
        elif kind == "calibrate":
            oof = rep.get("pooled_oof", {})
            cover.append((name, f"scene {rep.get('scene_date', '?')} · "
                                f"mean band R² {oof.get('mean_band_R2', '?')}"))
        elif kind == "tiles":
            cover.append((name, f"{rep.get('tiles_written', '?')} web tiles · "
                                f"z{rep.get('min_zoom', '?')}–z{rep.get('max_zoom', '?')} · "
                                f"{rep.get('size_mb', '?')} MB"))
        else:
            cover.append((name, ", ".join(f"{k}={v}" for k, v in list(rep.items())[:4]
                                          if not isinstance(v, (dict, list)))))
    cover_html = "".join(_row([f'<span class="{"ok" if ok else "ok"}">✓</span>', n, d])
                         for n, d in cover) or _row(["", "no stage reports found", ""])

    # ---- thumbnails --------------------------------------------------------- #
    thumbs = []
    try:
        if paths.visible_base.exists():
            thumbs.append(_img(_thumb_product(paths.visible_base, [1, 2, 3]),
                               f"{paths.visible_base.name} — true color"))
    except ValueError:
        pass
    try:
        if paths.ms_calibrated.exists():
            thumbs.append(_img(_thumb_product(paths.ms_calibrated, [3, 1, 2]),
                               f"{paths.ms_calibrated.name} — false color (NIR·Red·Green)"))
    except ValueError:
        pass
    reg_html = _img(_thumb_r_map(paths.reg_qa),
                    "Registration reliability — per-cell texture-masked r vs the anchor. "
                    "No pass/fail threshold: r is scene-dependent; this map is the "
                    "evidence, read per survey.") if paths.reg_qa.exists() else ""
    vis_reg_html = _img(_thumb_r_map(paths.visible_reg_qa),
                        "Visible-layer alignment vs the borrowed anchor (cross-season r "
                        "reads lower for phenology reasons, not geometry).") \
        if paths.visible_reg_qa.exists() else ""
    qa_html = _img(_thumb_qa_panels(paths.calib_qa),
                   "The 7 MAE values + n_px. Per-band MAE is raw and un-normalised — "
                   "read each against its OWN band's magnitude, never across bands.") \
        if paths.calib_qa.exists() else ""

    # ---- stage detail tables ------------------------------------------------ #
    detail = []
    for name, rep in stages.items():
        if rep.get("kind") == "stitch" and rep.get("seams"):
            rows = "".join(_row([s["pair"], s.get("med_shift_cm", ""),
                                 s.get("residual_cm", "")]) for s in rep["seams"])
            detail.append(f"<section><h2>{name} — seams</h2><table>"
                          + _row(["seam", "measured shift (cm)", "post-solve residual (cm)"], "th")
                          + rows + "</table></section>")
        if rep.get("kind") == "align":
            rows = "".join(_row([m.get("name") or m.get("mission"), m.get("tiles"),
                                 m.get("matches"), m.get("cells"), m.get("d_med_cm")])
                           for m in rep.get("missions", []))
            detail.append(
                f"<section><h2>{name}</h2><table>"
                + _row(["mission", "tiles", "matches", "cells", "median shift (cm)"], "th")
                + rows + "</table></section>")
        if rep.get("kind") == "calibrate":
            oof = rep.get("pooled_oof", {})
            mae = rep.get("qa_summary", {})
            head = _row(list(oof.keys()), "th") + _row(list(oof.values())) if oof else ""
            mrows = "".join(_row([k, v.get("median"), v.get("p90")])
                            for k, v in mae.items() if isinstance(v, dict))
            detail.append(f"<section><h2>calibrate — held-out agreement</h2>"
                          f"<table>{head}</table>"
                          "<table>" + _row(["reliability band", "median", "p90"], "th")
                          + mrows + "</table></section>")

    # ---- web map (pmtiles + leaflet viewer, when the tiles stage ran) ------- #
    tiles_html = ""
    trep = next((r for r in stages.values() if r.get("kind") == "tiles"), None)
    if trep:
        pm_name = Path(trep.get("out", "")).name
        map_name = Path(trep["map_html"]).name if trep.get("map_html") else None
        tiles_html = (
            f"<p><b>{pm_name}</b> — single-file PMTiles archive: "
            f"{trep.get('tiles_written', '?')} WEBP tiles, "
            f"z{trep.get('min_zoom', '?')}–z{trep.get('max_zoom', '?')}, "
            f"{trep.get('size_mb', '?')} MB. Serve it from any static host or "
            f"bucket; byte-range requests do the rest — no tile server.</p>")
        if map_name:
            serve_cmd = (f"tidysurvey serve --config {cfg.source}"
                         if cfg.source else "tidysurvey serve --config <survey>.toml")
            tiles_html += (
                f'<p>Interactive map: <a href="../{map_name}">{map_name}</a> '
                f"(lives beside the archive in products/). View locally — needs "
                f"byte-range HTTP, which file:// and python's stock http.server "
                f"don't provide:</p>"
                f'<pre class="mono">{serve_cmd}</pre>')

    snapshot = json.dumps(manifest.get("config", cfg.snapshot()), indent=2)
    stamp = time.strftime("%Y-%m-%d %H:%M")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{cfg.survey} — run quality report</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header><div><h1>{cfg.survey} — run quality report</h1>
<div class="meta">generated {stamp} · tidysurvey · config {cfg.source or ""}</div></div>
<div class="badge {'pass' if ok else 'fail'}">{badge}</div></header>
<section><h2>Run verdict</h2><table>{cover_html}</table>
{'' if ok else f'<p class="note bad">Run halted at {failed_stage}; sections below cover the stages that ran.</p>'}
</section>
<section><h2>Products</h2><div class="cards">{''.join(thumbs) or '<p class="note">no products yet</p>'}</div></section>
{f'<section><h2>Web map</h2>{tiles_html}</section>' if tiles_html else ''}
{f'<section><h2>Alignment — visible vs borrowed anchor</h2>{vis_reg_html}</section>' if vis_reg_html else ''}
{f'<section><h2>Alignment — multispectral vs base map</h2>{reg_html}</section>' if reg_html else ''}
{''.join(detail)}
{f'<section><h2>Calibration reliability</h2>{qa_html}</section>' if qa_html else ''}
<section><h2>Configuration snapshot</h2><pre>{snapshot}</pre></section>
<footer>Assembled by <span class="mono">tidysurvey report</span> — regenerable without
re-processing from <span class="mono">work/reports/</span> and the shipped rasters.
Machine twin: <span class="mono">quality_report.json</span>.</footer>
</div></body></html>"""

    paths.report_html.write_text(html)
    paths.report_json.write_text(json.dumps(
        dict(survey=cfg.survey, generated=stamp, ok=ok, failed_stage=failed_stage,
             stages=stages, manifest=manifest), indent=2))
    log(f"[report] -> {paths.report_html} ({time.time() - t0:.1f}s)")
    return dict(kind="report", html=str(paths.report_html),
                json=str(paths.report_json), ok=ok)

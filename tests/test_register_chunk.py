"""Smoke test for the chunked-subprocess dense match (the MPS-decay fix).

register_survey_dense now runs the match loop in fresh subprocess chunks. This
exercises the full orchestration — geometry, per-chunk subprocess spawn, part
collection, field fit, warp, output cleanup — with a FAKE matcher (env flag) so
no LoFTR is needed, and a tiny chunk size so several subprocesses run. The fake
matcher lives in the subprocess too (inherited env), which is the whole point:
if the spawn/collect wiring is wrong, this fails.
"""
import os
import json
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import ColorInterp

os.environ["TIDYSURVEY_REG_FAKE"] = "1"     # workers use the synthetic matcher
os.environ["TIDYSURVEY_REG_CHUNK"] = "4"    # small chunks -> several subprocesses

import tidysurvey.registration as R

TMP = tempfile.mkdtemp(prefix="reg_chunk_")
W, H, res = 2000, 1500, 0.1
tr = from_origin(0, H * res, res, res)
crs = "EPSG:32611"


def _rgba(path, seed):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(40, 210, (3, H, W)).astype(np.uint8)
    alpha = np.full((H, W), 255, np.uint8)
    prof = dict(driver="GTiff", width=W, height=H, count=4, dtype="uint8",
                crs=crs, transform=tr, tiled=True, compress="zstd")
    with rasterio.open(path, "w", **prof) as d:
        for b in range(3):
            d.write(rgb[b], b + 1)
        d.write(alpha, 4)
        d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue,
                         ColorInterp.alpha]


mission = f"{TMP}/mission.tif"
anchor = f"{TMP}/anchor.tif"
out = f"{TMP}/registered.tif"
qa = f"{TMP}/qa.json"
_rgba(mission, 1)
_rgba(anchor, 2)

logs = []
summary = R.register_survey_dense(mission, anchor, out, resolution_m=res,
                                  qa_json=qa, log=lambda m: logs.append(m))

chunk_done = [m for m in logs if "chunk" in m and "done" in m]
announce = [m for m in logs if "subprocess chunks" in m]
have_out = os.path.exists(out)
with rasterio.open(out) as d:
    bands, ow, oh = d.count, d.width, d.height
qa_j = json.load(open(qa))
cleaned = not os.path.exists(out + ".regwork")

print(f"announce: {announce[:1]}")
print(f"chunks run: {len(chunk_done)}  (expect 3 with 9 tiles / CHUNK=4)")
print(f"output: exists={have_out} {ow}x{oh} bands={bands}")
print(f"qa: tiles={qa_j['tiles']} matches={qa_j['matches']} cells={qa_j['cells']} "
      f"d_med_cm={qa_j['d_med_cm']}")
print(f"workdir cleaned: {cleaned}")

ok = (have_out and len(chunk_done) > 1 and bands == 4 and ow == W and oh == H
      and qa_j["matches"] > 0 and qa_j["cells"] > 0 and cleaned)
print("VERDICT:", "PASS" if ok else "FAIL",
      f"(chunked subprocess register reproduces a valid warp; {len(chunk_done)} chunks)")

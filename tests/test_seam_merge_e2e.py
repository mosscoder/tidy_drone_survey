"""End-to-end integration + resume test for the checkpointed low-memory
seam_merge. The LoFTR matcher is monkeypatched to synthetic correspondences so
the test is deterministic and exercises the NEW code (checkpoint write/load,
sparse efield, lattice solve, lattice composite, resume) rather than the
(unchanged, proven) matcher. Also a real-dims solve-memory probe.
"""
import os, glob, resource, numpy as np, rasterio
from rasterio.transform import from_origin
from rasterio.enums import ColorInterp
import tidysurvey.fields as F
import tidysurvey.merge as M

import tempfile
TMP = tempfile.mkdtemp(prefix="seam_e2e_")


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


# ---- synthetic inputs: 3 overlapping RGBA rasters (triple junction) ----------
W, H, res = 4000, 3000, 0.1
tr = from_origin(0, H * res, res, res)
crs = "EPSG:32611"
cols = np.arange(W)[None, :].repeat(H, 0)
rows = np.arange(H)[:, None].repeat(W, 1)
masks = {"m0": cols < 2400,
         "m1": (cols > 1600) & (rows < 1600),
         "m2": (cols > 1600) & (rows > 1400)}
paths = []
for i, (name, mask) in enumerate(masks.items()):
    rng = np.random.default_rng(i)
    rgb = (rng.integers(40, 210, (3, H, W)) * mask).astype(np.uint8)
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    p = f"{TMP}/{name}.tif"
    prof = dict(driver="GTiff", width=W, height=H, count=4, dtype="uint8",
                crs=crs, transform=tr, tiled=True, compress="zstd")
    with rasterio.open(p, "w", **prof) as d:
        for b in range(3):
            d.write(rgb[b], b + 1)
        d.write(alpha, 4)
        d.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
    paths.append((name, p))


# ---- monkeypatch the matcher: synthetic dense correspondences ----------------
import torch
_orig_build, _orig_match, _orig_rel = F.build_matcher, F.match_tile, F.release_matcher_cache


def fake_build(device=None):
    return object(), torch.device("cpu")


def fake_match(matcher, dev, ga, gb, reject_px, min_tile):
    h, w = ga.shape[-2:]
    ys, xs = np.mgrid[30:h - 30:40, 30:w - 30:40]
    kp = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)         # local (x,y)
    rng = np.random.default_rng(int(kp.sum()) % 9999)
    disp = np.column_stack([np.full(len(kp), 0.6), np.full(len(kp), -0.4)]) \
        + rng.normal(0, 0.05, (len(kp), 2))                              # ~sub-px shift
    return kp, disp


F.build_matcher, F.match_tile, F.release_matcher_cache = fake_build, fake_match, (lambda d: None)


def run(tag, ckpt_survivors=None):
    out = f"{TMP}/master.tif"
    ckdir = f"{TMP}/master_seam_ckpt"
    if ckpt_survivors is not None and os.path.isdir(ckdir):
        for f in glob.glob(f"{ckdir}/seam_*.npz"):
            if os.path.basename(f) not in ckpt_survivors:
                os.remove(f)
    if os.path.exists(out):
        os.remove(out)
    rep = M.seam_merge(paths, out, res=res, ownership_out=f"{TMP}/own.tif",
                       report_json=f"{TMP}/stitch.json", log=lambda m: None)
    with rasterio.open(out) as d:
        arr = d.read()
        cov = (arr[-1] > 0).mean()
        rgb_mean = arr[:3][:, arr[-1] > 0].mean() if cov else 0
    cks = sorted(os.path.basename(x) for x in glob.glob(f"{ckdir}/seam_*.npz"))
    print(f"[{tag}] out {d.width}x{d.height} cov={cov*100:.0f}% rgbμ={rgb_mean:.0f}  "
          f"seams={rep['n_seams']} residual_cm={[s.get('residual_cm') for s in rep['seams']]}")
    print(f"       checkpoints: {cks}")
    return rep, arr


print("=" * 74)
print("TEST A — full run (walk writes checkpoints -> solve -> composite)")
r1, a1 = run("fresh")
pk1 = peak_gb()

print("\nTEST B — resume with ALL checkpoints present (must SKIP the walk)")
# keep all ckpts, delete output -> should load every pair and reproduce output
r2, a2 = run("resume-all", ckpt_survivors=set(
    os.path.basename(x) for x in glob.glob(f"{TMP}/master_seam_ckpt/seam_*.npz")))
same = np.array_equal(a1, a2)
print(f"       output identical to fresh run: {same}")

print("\nTEST C — resume after a mid-walk 'crash' (only seam_0_1 survived)")
r3, a3 = run("resume-partial", ckpt_survivors={"seam_0_1.npz"})
print(f"       completed, seams={r3['n_seams']}")

# ---- restore matcher; probe solve memory at REAL coarse dims -----------------
F.build_matcher, F.match_tile, F.release_matcher_cache = _orig_build, _orig_match, _orig_rel
print("\nTEST D — solve memory at REAL coarse dims (N=3), no dense field built")
Hc, Wc = 29637, 22936
base = peak_gb()
valid = [np.ones((Hc, Wc), bool) for _ in range(3)]        # 3 x 680 MB inputs
corridor = np.zeros((Hc, Wc), bool)
corridor[7:3007:15, 7:3007:15] = True                      # aligned to node grid -> ~40k nodes
rng = np.random.default_rng(0)
eff = {}
for pr in [(0, 1), (0, 2), (1, 2)]:
    pts = rng.uniform(0, [Wc, Hc], (400, 2))
    eff[pr] = (pts, rng.normal(0, 1, 400), rng.normal(0, 1, 400))
b4 = peak_gb()
lat, sol, resid, nn, nv = F.solve_free_gauge(valid, eff, corridor, 15, 3)
pk = peak_gb()
print(f"       {nn} nodes, {nv} unknowns; peak added by the SOLVE = {pk - b4:.2f} G "
      f"(old dense path would add ~{3*2*Hc*Wc*4/1024**3:.0f} G of fields)")

# =====================================================================
# TEST E — coarse-pass reuse ;  TEST F — composite block-resume
# both new resume paths must reproduce the clean Test A output byte-for-byte
# =====================================================================
import json, shutil as _sh
F.build_matcher, F.match_tile, F.release_matcher_cache = fake_build, fake_match, (lambda d: None)


def _clear(out):
    base = out.rsplit(".", 1)[0]
    _sh.rmtree(base + "_coarse", ignore_errors=True)
    _sh.rmtree(base + "_seam_ckpt", ignore_errors=True)
    for f in (out, base + "_render_ckpt.json"):
        if os.path.exists(f):
            os.remove(f)


print("\nTEST E — coarse-pass reuse (crash after coarse -> resume reuses memmaps)")
outE = f"{TMP}/masterE.tif"; baseE = outE.rsplit(".", 1)[0]; _clear(outE)
_orig_bld = F.build_matcher


def _boom(device=None):
    raise RuntimeError("injected: crash right after the coarse pass")


F.build_matcher = _boom                      # the walk builds the matcher first -> dies post-coarse
crashedE = False
try:
    M.seam_merge(paths, outE, res=res, log=lambda m: None)
except RuntimeError as e:
    crashedE = "injected" in str(e)
_arts = ["ds.dat", "valid.dat", "D.dat", "owner.npy", "cov2.npy", "_coarse_done.json"]
have_coarse = all(os.path.exists(f"{baseE}_coarse/{a}") for a in _arts)
F.build_matcher = _orig_bld
logsE = []
M.seam_merge(paths, outE, res=res, log=lambda m: logsE.append(m))
reusedE = any("reusing complete geometry" in m for m in logsE)
with rasterio.open(outE) as d:
    aE = d.read()
identicalE = np.array_equal(aE, a1)
coarse_gone_E = not os.path.exists(f"{baseE}_coarse")
print(f"       crashed after coarse={crashedE}  artifacts persisted={have_coarse}")
print(f"       resume logged reuse={reusedE}  output==clean={identicalE}  "
      f"coarse cleaned on success={coarse_gone_E}")

print("\nTEST F — composite block-resume (crash mid-render -> resume completes)")
outF = f"{TMP}/masterF.tif"; baseF = outF.rsplit(".", 1)[0]; _clear(outF)
_orig_wja = M._write_json_atomic
_rck = {"n": 0}


def _crashing_wja(path, obj):
    if "render_ckpt" in str(path):
        _rck["n"] += 1
        if _rck["n"] == 2:                    # let chunk 0 checkpoint, die on chunk 1
            raise RuntimeError("injected: crash mid-render after one checkpoint")
    return _orig_wja(path, obj)


M._write_json_atomic = _crashing_wja
crashedF = False
try:                                         # render_chunk=1 -> one block per checkpoint
    M.seam_merge(paths, outF, res=res, render_chunk=1, log=lambda m: None)
except RuntimeError as e:
    crashedF = "injected" in str(e)
M._write_json_atomic = _orig_wja
rckF = f"{baseF}_render_ckpt.json"
partial_done = json.load(open(rckF))["done"] if os.path.exists(rckF) else None
logsF = []
M.seam_merge(paths, outF, res=res, render_chunk=1, log=lambda m: logsF.append(m))
resumedF = any("composite: resuming" in m for m in logsF)
with rasterio.open(outF) as d:
    aF = d.read()
identicalF = np.array_equal(aF, a1)
rck_gone = not os.path.exists(rckF)
print(f"       crashed mid-render={crashedF}  partial ckpt done={partial_done}")
print(f"       resume logged resume={resumedF}  output==clean={identicalF}  "
      f"render ckpt cleaned={rck_gone}")

print("=" * 74)
ok = (same and r1['n_seams'] == 3 and r3['n_seams'] == 3 and (pk - b4) < 4.0
      and crashedE and have_coarse and reusedE and identicalE and coarse_gone_E
      and crashedF and partial_done == [0] and resumedF and identicalF and rck_gone)
print("VERDICT:", "PASS" if ok else "FAIL",
      f"(walk-resume + coarse-reuse + render-resume all byte-identical to clean; "
      f"solve adds {pk-b4:.1f}G)")

"""TESTER (run before refactoring): does a LOW-MEMORY sparse-sampling solve
reproduce the current DENSE-field solve?

Current dense path (fields.fill_field): griddata_fill on an 8-px lattice ->
cv2.resize INTER_LINEAR to the full (Hc,Wc) field (2.7 GB at real dims) ->
solve samples field[r,c] at corridor nodes.

Proposed sparse path: build the SAME 8-px lattice (42 MB), sample it at the
nodes with cv2.remap (same INTER_LINEAR kernel) -- never materialize the full
field. If the node values match, the solve RHS matches, so the lsqr solution
matches by construction.

Corners are the stress case: griddata nearest-extrapolation past the hull +
cv2 border handling are worst where a mission footprint edge meets others.
"""
import numpy as np, cv2, resource
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from tidysurvey.fields import fill_field, griddata_fill

RES_M, GEOM_DS = 0.033, 8            # for a ground-distance read-out


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3   # macOS: bytes


def sample_dense_at(pts, vals, Wc, Hc, nodes):
    """CURRENT path: full dense field, then index at nodes."""
    F = fill_field(np.asarray(pts, float), np.asarray(vals), Wc, Hc)      # (Hc,Wc) full
    rr = np.array([r for r, c in nodes]); cc = np.array([c for r, c in nodes])
    out = F[rr, cc].copy()
    del F
    return out


def sample_sparse_at(pts, vals, Wc, Hc, nodes, sample_ds=8):
    """PROPOSED path: the SHIPPED fields.sample_field_nodes (what goes in the solve)."""
    from tidysurvey.fields import sample_field_nodes
    nr = np.array([r for r, c in nodes]); nc = np.array([c for r, c in nodes])
    return sample_field_nodes(pts, vals, Wc, Hc, nr, nc, sample_ds)


# =================== PART 1: node-value equivalence, incl. corners ===================
def part1(Wc, Hc, label):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(1)
    n = 400
    # correspondences clustered along a diagonal seam corridor (like real seams),
    # NOT spread over the whole grid
    t = rng.uniform(0, 1, n)
    pts = np.column_stack([t * Wc, t * Hc]) + rng.normal(0, min(Wc, Hc) * 0.01, (n, 2))
    pts = np.clip(pts, 0, [Wc - 1, Hc - 1])
    # REALISTIC field: bounded to a few px regardless of grid size (real seam shifts ~1-2 px)
    truth = lambda x, y: 2.0 * np.sin(2 * np.pi * x / Wc) + 1.5 * np.cos(2 * np.pi * y / Hc)
    vx = truth(pts[:, 0], pts[:, 1]) + rng.normal(0, 0.05, n)
    spacing = 15
    grid = [(r, c) for r in range(spacing // 2, Hc, spacing)
            for c in range(spacing // 2, Wc, spacing)]
    corners = [(0, 0), (0, Wc - 1), (Hc - 1, 0), (Hc - 1, Wc - 1),
               (0, Wc // 2), (Hc - 1, Wc // 2), (Hc // 2, 0), (Hc // 2, Wc - 1)]
    nodes = corners + grid
    base = peak_gb()
    a = sample_dense_at(pts, vx, Wc, Hc, nodes)     # builds the full field (2.7 GB at real dims)
    pk_dense = peak_gb()
    b = sample_sparse_at(pts, vx, Wc, Hc, nodes)
    d = np.abs(a - b)
    # distance (coarse px) from each node to the nearest correspondence = corridor proxy.
    # the real solve only uses nodes with D_c<=4m ~ 15 coarse px of the faultline.
    dist = cKDTree(pts).query(np.array(nodes, float)[:, ::-1])[0]   # nodes are (r,c)->(x=c,y=r)
    corridor = dist <= 30
    j = int(np.argmax(d))
    print(f"[{label}]  grid {Wc}x{Hc}  ({len(nodes)} nodes)")
    print(f"    peak RSS building the DENSE full field : {pk_dense:.2f} G "
          f"(+{pk_dense - base:.2f} G)   sparse lattice stays tiny")
    print(f"    ALL nodes         : max|Δ| = {d.max():.3e}  mean = {d.mean():.3e} px")
    print(f"    CORRIDOR nodes    : max|Δ| = {d[corridor].max():.3e}  "
          f"(within 30 coarse px of a correspondence; {corridor.sum()} nodes)  <- the solve's regime")
    print(f"    corner/edge nodes : max|Δ| = {d[:len(corners)].max():.3e}")
    print(f"    worst outlier node: |Δ|={d[j]:.3f} px  at dist={dist[j]:.0f} coarse px "
          f"from nearest data (deep extrapolation, excluded by corridor_c)")
    print(f"    -> worst CORRIDOR error on the ground = {d[corridor].max() * RES_M * 1000:.5f} mm")
    return d[corridor].max()


# =================== PART 3: end-to-end solve at a 3-mission corner ===================
def assemble_solve(valid_c, corridor_c, spacing, N, node_vals, nodes):
    """faithful copy of fields.solve_free_gauge assembly; node_vals[(i,j)]=(fxn,fyn)."""
    var = {}
    for oi in range(N):
        V = valid_c[oi]
        for (r, c) in nodes:
            if V[r, c]:
                var[(oi, r, c)] = len(var)
    rows, cols, data, bx, by = [], [], [], [], []; nrow = [0]

    def add(terms, rx, ry):
        for vi, co in terms:
            rows.append(nrow[0]); cols.append(vi); data.append(co)
        bx.append(rx); by.append(ry); nrow[0] += 1

    for (oi, oj), (fxn, fyn) in node_vals.items():
        co = valid_c[oi] & valid_c[oj]
        for k, (r, c) in enumerate(nodes):
            if not co[r, c]:
                continue
            ti = var.get((oi, r, c)); tj = var.get((oj, r, c))
            if ti is not None and tj is not None:
                add([(tj, 1.0), (ti, -1.0)], float(fxn[k]), float(fyn[k]))
    for (oi, r, c), vi in var.items():
        add([(vi, 0.05)], 0.0, 0.0)                  # gauge ridge (as in solve_free_gauge)
    A = sp.coo_matrix((data, (rows, cols)), shape=(nrow[0], len(var))).tocsr()
    cx = lsqr(A, np.array(bx), atol=1e-8, btol=1e-8)[0]
    cy = lsqr(A, np.array(by), atol=1e-8, btol=1e-8)[0]
    return cx, cy, var


def part3():
    Wc = Hc = 1600; spacing = 15; N = 3
    xx, yy = np.meshgrid(np.arange(Wc), np.arange(Hc))
    m0 = xx < 900                       # left
    m1 = (xx > 700) & (yy < 900)        # right-top
    m2 = (xx > 700) & (yy > 700)        # right-bottom  -> triple junction ~ (800,800)
    valid_c = [m0, m1, m2]
    corridor_c = (m0.astype(int) + m1 + m2) >= 2
    rng = np.random.default_rng(7)

    def corr(mask, fxf, fyf, n=300):
        ys, xs = np.where(mask)
        idx = rng.choice(len(xs), min(n, len(xs)), replace=False)
        pts = np.column_stack([xs[idx], ys[idx]]).astype(float)
        return pts, (fxf(pts[:, 0], pts[:, 1]) + rng.normal(0, 0.05, len(idx))), \
                    (fyf(pts[:, 0], pts[:, 1]) + rng.normal(0, 0.05, len(idx)))

    pairs = {(0, 1): corr(m0 & m1, lambda x, y: 0.010 * (x - 800), lambda x, y: -0.008 * (y - 800)),
             (0, 2): corr(m0 & m2, lambda x, y: 0.012 * (x - 800), lambda x, y:  0.006 * (y - 800)),
             (1, 2): corr(m1 & m2, lambda x, y: -0.009 * (y - 800), lambda x, y: 0.011 * (x - 800))}

    rr = list(range(spacing // 2, Hc, spacing)); cc = list(range(spacing // 2, Wc, spacing))
    nodes = [(r, c) for r in rr for c in cc if corridor_c[r, c]]
    rrn = np.array([r for r, c in nodes]); ccn = np.array([c for r, c in nodes])

    nvA = {}
    for pr, (pts, vx, vy) in pairs.items():
        Fx = fill_field(pts, vx, Wc, Hc); Fy = fill_field(pts, vy, Wc, Hc)   # DENSE reference
        nvA[pr] = (Fx[rrn, ccn], Fy[rrn, ccn])
    cxA, cyA, varA = assemble_solve(valid_c, corridor_c, spacing, N, nvA, nodes)   # old dense path
    # the REAL shipped function (sparse in, lattice out):
    from tidysurvey.fields import solve_free_gauge
    lat, sol, resid, nn, nv = solve_free_gauge(valid_c, {pr: v for pr, v in pairs.items()},
                                               corridor_c, spacing, N)
    dreal = 0.0
    for oi in range(N):
        if sol[oi] is None:
            continue
        P, cxo, cyo = sol[oi]
        for (c, r), vxv, vyv in zip(P.astype(int), cxo, cyo):
            key = (oi, int(r), int(c))
            if key in varA:
                dreal = max(dreal, abs(cxA[varA[key]] - vxv), abs(cyA[varA[key]] - vyv))
    print("[part3]  3-mission corner (triple junction ~ (800,800))")
    print(f"    {len(varA)} unknowns, {len(nodes)} corridor nodes")
    print(f"    REAL solve_free_gauge vs OLD dense solve: max|Δ solution| = {dreal:.3e} px "
          f"({dreal * RES_M * 1000:.5f} mm)")
    return dreal


def part4():
    """The composite's ONE new approximation: the solved field reaches native
    pixels by upsampling the sample_ds LATTICE (x GEOM_DS*sample_ds = x64) rather
    than the old full coarse field (x GEOM_DS = x8). Same lattice, one-stage vs
    two-stage bilinear -- check it's sub-mm at a far corner block."""
    from tidysurvey.fields import fill_field, griddata_fill, upsample_block
    GEOM_DS, sample_ds = 8, 8
    Wc = Hc = 1600
    rng = np.random.default_rng(3)
    P = rng.uniform(0, [Wc, Hc], (150, 2))
    cx = 2.0 * np.sin(P[:, 0] / Wc * 2 * np.pi) + 1.5 * np.cos(P[:, 1] / Hc * 2 * np.pi) \
        + rng.normal(0, 0.02, 150)
    Fx = fill_field(P, cx, Wc, Hc)                                    # OLD: full coarse field
    r0, c0, bh, bw = (Hc - 50) * GEOM_DS, (Wc - 50) * GEOM_DS, 300, 300   # far corner, native px
    dense_native = upsample_block(Fx, GEOM_DS, r0, c0, bh, bw)
    ex = np.arange(0, Wc, sample_ds); ey = np.arange(0, Hc, sample_ds); EX, EY = np.meshgrid(ex, ey)
    lat = np.nan_to_num(griddata_fill(P, cx, EX, EY)).astype(np.float32)   # NEW: lattice
    lat_native = upsample_block(lat, GEOM_DS * sample_ds, r0, c0, bh, bw)
    d = np.abs(dense_native - lat_native)
    print(f"[part4]  composite solved-field at a corner block {bh}x{bw} (native)")
    print(f"    dense x8  vs  lattice x64 :  max|Δ| = {d.max():.3e}  mean = {d.mean():.3e} px "
          f"-> {d.max() * RES_M * 1000:.4f} mm")
    return d.max()


if __name__ == "__main__":
    print("=" * 74)
    m1 = part1(2400, 2400, "PART 1a small")
    print()
    m2 = part1(22936, 29637, "PART 1b REAL coarse dims (dense field = 2.7 GB)")
    print()
    w = part3()
    print()
    c = part4()
    print("=" * 74)
    ok = (m1 < 1e-2) and (m2 < 1e-2) and (w < 1e-2) and (c < 1e-2)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(all differences {'<' if ok else 'NOT <'} 1e-2 px = {1e-2*RES_M*1000:.2f} mm; "
          f"uint8 output rounds identically)")

#!/usr/bin/env python3
"""Print-ready figurine of a toddler girl, modelled from a reference photo.

The figure is sculpted with signed distance fields (SDFs) and meshed with
marching cubes, so every output is a closed, watertight solid.

Outfit taken from the photo: two little pigtails tied with yellow bows, a high
ruffled collar, a shiny long-sleeve jacket with ruffled cuffs, hands clasped in
front, flared trousers with a bow at each knee, and sneakers.

Units are millimetres. Z is up and the figure faces -Y.

Outputs (in --out):
  girl_figurine.stl            one-piece model for single-colour printing
  multicolor/<part>.stl        non-overlapping parts for multi-material printing
  girl_figurine_color.glb      vertex-coloured model for previewing
"""
import argparse
import os
import time

import fast_simplification
import numpy as np
import trimesh
from skimage import measure

BIG = 1e3


def V(*a):
    return np.array(a, dtype=np.float64)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def length(q):
    return np.sqrt(np.einsum("ij,ij->i", q, q))


# ---------------------------------------------------------------- primitives

def sphere(P, c, r):
    return length(P - c) - r


def ellipsoid(P, c, r, axes=None):
    """Approximate ellipsoid SDF; `axes` columns are the local unit axes."""
    q = P - c
    if axes is not None:
        q = q @ axes
    r = np.asarray(r, dtype=np.float64)
    k0 = length(q / r)
    k1 = length(q / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def round_cone(P, a, b, r1, r2):
    """Capsule from a (radius r1) to b (radius r2)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = P - a
    y = pa @ ba
    z = y - l2
    w = pa * l2 - np.outer(y, ba)
    x2 = np.einsum("ij,ij->i", w, w)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d_end = np.sqrt(x2 + z2) * il2 - r2
    d_start = np.sqrt(x2 + y2) * il2 - r1
    d_side = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d_end,
                    np.where(np.sign(y) * a2 * y2 < k, d_start, d_side))


def frame(axis, ref=(0.0, 0.0, 1.0)):
    a = unit(axis)
    r = np.asarray(ref, dtype=np.float64)
    if abs(a @ r) > 0.95:
        r = np.array([1.0, 0.0, 0.0])
    e1 = unit(np.cross(r, a))
    e2 = np.cross(a, e1)
    return a, e1, e2


def band(P, origin, axis, u0, u1, r0, r1, amp0=0.0, amp1=None, n=0,
         phase=0.0, rnd=0.3):
    """Solid frustum around `axis`, optionally with a ruffled (wavy) rim.

    u0/u1 are positions along the axis measured from `origin`; the radius goes
    from r0 to r1 and the ruffle amplitude from amp0 to amp1 along the way.
    """
    a, e1, e2 = frame(axis)
    q = P - origin
    u = q @ a
    x1 = q @ e1
    x2 = q @ e2
    rho = np.sqrt(x1 * x1 + x2 * x2)
    t = np.clip((u - u0) / (u1 - u0), 0.0, 1.0)
    R = r0 + (r1 - r0) * t
    lip = 1.0 / np.sqrt(1.0 + ((r1 - r0) / (u1 - u0)) ** 2)
    if n:
        amp1 = amp0 if amp1 is None else amp1
        R = R + (amp0 + (amp1 - amp0) * t) * np.sin(n * np.arctan2(x2, x1) + phase)
        lip /= np.sqrt(1.0 + (max(amp0, amp1) * n / min(r0, r1)) ** 2)
    return smax((rho - R) * lip, np.maximum(u0 - u, u - u1), rnd)


def torus(P, c, axis, R, r, amp=0.0, n=0):
    a, e1, e2 = frame(axis)
    q = P - c
    x1 = q @ e1
    x2 = q @ e2
    rho = np.sqrt(x1 * x1 + x2 * x2)
    rr = r + amp * np.sin(n * np.arctan2(x2, x1)) if n else r
    return np.sqrt((rho - R) ** 2 + (q @ a) ** 2) - rr


def ellip_frustum(P, cy, z0, z1, rx0, ry0, rx1, ry1, rnd):
    x, y, z = P[:, 0], P[:, 1] - cy, P[:, 2]
    t = np.clip((z - z0) / (z1 - z0), 0.0, 1.0)
    rx = rx0 + (rx1 - rx0) * t
    ry = ry0 + (ry1 - ry0) * t
    k = np.sqrt((x / rx) ** 2 + (y / ry) ** 2)
    return smax((k - 1.0) * np.minimum(rx, ry), np.maximum(z0 - z, z - z1), rnd)


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


# ------------------------------------------------------------ key positions

SIDES = (-1.0, 1.0)
HEAD_C = V(0.0, 0.8, 90.0)       # cranium centre
HAIR_C = V(0.0, 1.2, 89.9)       # hair cap centre
HAND_Y = -9.1


def wrist(s):
    return V(s * 6.4, -8.9, 47.6)


def elbow(s):
    return V(s * 15.4, -0.8, 59.0)


def shoulder(s):
    return V(s * 11.6, 0.6, 71.0)


def pigtail(s):
    pts = [V(s * 7.2, 1.8, 97.0), V(s * 11.4, 2.4, 98.9),
           V(s * 14.0, 3.0, 97.0), V(s * 14.9, 3.4, 93.9)]
    return pts, [2.7, 2.7, 2.1, 1.3]


def knee(s):
    return V(s * 7.2, 0.0, 21.6)


# ----------------------------------------------------------------- materials

def skin(P):
    d = sphere(P, HEAD_C, 10.0)
    d = smin(d, ellipsoid(P, V(0, -1.6, 86.2), (8.2, 8.0, 8.4)), 2.0)
    for s in SIDES:
        d = smin(d, sphere(P, V(s * 4.4, -6.0, 84.6), 2.8), 1.5)             # cheeks
        d = smin(d, ellipsoid(P, V(s * 9.3, 0.6, 86.8), (1.3, 2.0, 2.7)), 0.5)  # ears
    d = smin(d, sphere(P, V(0, -9.55, 85.4), 0.9), 0.6)                      # nose
    for s in SIDES:                                                          # eye sockets
        d = smax(d, -ellipsoid(P, V(s * 3.2, -9.45, 87.6), (1.75, 1.0, 1.35)), 0.35)
    d = smax(d, -ellipsoid(P, V(0, -9.0, 82.9), (1.35, 0.5, 0.32)), 0.25)    # mouth
    d = smin(d, round_cone(P, (0, 0.6, 73.0), (0, 0.4, 82.0), 4.3, 4.0), 1.0)  # neck
    hands = BIG
    for s in SIDES:
        h = ellipsoid(P, V(s * 2.7, HAND_Y, 43.4), (2.7, 2.4, 3.2))
        h = smin(h, round_cone(P, wrist(s), (s * 2.9, HAND_Y, 44.2), 1.9, 2.2), 1.0)
        thumb = round_cone(P, (s * 3.6, HAND_Y - 2.0, 45.4), (s * 1.2, HAND_Y - 2.2, 45.9), 0.75, 0.7)
        hands = smin(hands, smin(h, thumb, 0.4), 0.8)
    return np.minimum(d, hands)


def eyes(P):
    return np.minimum(ellipsoid(P, V(-3.2, -7.85, 87.5), (1.3, 1.2, 1.25)),
                      ellipsoid(P, V(3.2, -7.85, 87.5), (1.3, 1.2, 1.25)))


def hair(P):
    d = sphere(P, HAIR_C, 11.1)
    k = 0.85                                         # hairline: high at the forehead, low at the nape
    f = (P[:, 2] - 93.0) + k * (P[:, 1] + 8.0)
    d = smax(d, -f / np.sqrt(1.0 + k * k), 0.5)
    for s in SIDES:
        d = smin(d, round_cone(P, (s * 6.0, -6.9, 93.3), (s * 7.6, -4.7, 85.2), 1.3, 0.9), 0.8)
        pts, rs = pigtail(s)
        for i in range(3):
            d = smin(d, round_cone(P, pts[i], pts[i + 1], rs[i], rs[i + 1]), 0.6)

    # strands combed towards each pigtail, plus a centre parting
    q = P - HAIR_C
    disp = np.zeros(len(P))
    for s in SIDES:
        _, e1, e2 = frame(pigtail(s)[0][0] - HAIR_C, ref=(0, 1, 0))
        psi = np.arctan2(q @ e2, q @ e1)
        disp = np.where(np.sign(P[:, 0] + 1e-9) == s, 0.17 * np.sin(26 * psi), disp)
    part = np.clip((P[:, 2] - 92.0) / 2.0, 0, 1) * np.clip((6.0 - P[:, 1]) / 2.0, 0, 1)
    disp += 0.3 * np.exp(-(P[:, 0] / 0.5) ** 2) * part
    return d + disp


def ribbons(P):
    d = BIG
    z_hat = V(0, 0, 1)
    for s in SIDES:                                  # hair ties with bows
        pts, _ = pigtail(s)
        axis = unit(pts[1] - pts[0])
        c = pts[0] + 1.2 * axis
        d = np.minimum(d, torus(P, c, axis, 2.4, 1.0, amp=0.22, n=12))
        up = unit(z_hat - (z_hat @ axis) * axis)
        e = unit(np.cross(axis, up))
        T = c + 1.6 * up
        bow = ellipsoid(P, T, (1.1, 1.1, 1.0), axes=np.column_stack([e, up, axis]))
        for t in SIDES:
            L = unit(t * e + 0.3 * up)
            U = unit(up - (up @ L) * L)
            loop = ellipsoid(P, T + 2.1 * L - 0.25 * up, (2.3, 1.45, 0.9),
                             axes=np.column_stack([L, U, axis]))
            bow = smin(bow, loop, 0.5)
        d = np.minimum(d, bow)
    for s in SIDES:                                  # bows on the knees
        kc = knee(s)
        bow = ellipsoid(P, kc + V(0, -6.4, 0), (1.15, 0.85, 1.05))
        for t in SIDES:
            phi = t * 0.42
            tang = V(np.cos(phi), np.sin(phi), 0)
            nrm = V(np.sin(phi), -np.cos(phi), 0)
            loop = ellipsoid(P, kc + 6.65 * nrm + V(0, 0, 0.3), (2.2, 0.8, 1.35),
                             axes=np.column_stack([tang, nrm, z_hat]))
            bow = smin(bow, loop, 0.4)
            tail = round_cone(P, kc + V(t * 0.5, -6.5, -0.6), kc + V(t * 1.8, -6.1, -4.0), 0.55, 0.45)
            bow = smin(bow, tail, 0.3)
        d = np.minimum(d, bow)
    return d


def jacket(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    d = ellipsoid(P, V(0, 0.2, 65.0), (11.5, 8.3, 11.2))
    for s in SIDES:
        d = smin(d, sphere(P, V(s * 10.2, 0.4, 70.6), 4.6), 3.0)
    d = smin(d, ellip_frustum(P, 0.0, 48.8, 63.0, 13.2, 9.8, 11.4, 8.4, 0.6), 3.0)
    # soft pleats towards the hem, front opening and a hem seam
    d = d + 0.3 * np.clip((60.0 - z) / 10.0, 0, 1) * np.sin(7 * np.arctan2(y, x) + 0.4)
    front = np.clip((-y - 3.0) / 2.0, 0, 1) * np.clip(z - 49.5, 0, 1) * np.clip(74.0 - z, 0, 1)
    d = d + 0.4 * np.exp(-(x / 0.45) ** 2) * front
    d = d + 0.22 * np.exp(-((z - 50.6) / 0.35) ** 2)
    for s in SIDES:
        S, E, W = shoulder(s), elbow(s), wrist(s)
        arm = smin(round_cone(P, S, E, 4.4, 4.0), round_cone(P, E, W, 4.0, 3.3), 1.2)
        cuff = band(P, W, W - E, -3.6, 0.8, 3.5, 4.9, amp0=0.45, amp1=0.6, n=11, rnd=0.35)
        d = smin(d, np.minimum(arm, cuff), 1.0)
    collar = band(P, V(0, 0.4, 0), (0, 0, 1), 74.0, 78.2, 6.0, 7.0,
                  amp0=0.25, amp1=0.45, n=16, rnd=0.45)
    frill = band(P, V(0, 0.4, 0), (0, 0, 1), 77.4, 79.6, 7.0, 8.4,
                 amp0=0.4, amp1=0.8, n=18, phase=0.5, rnd=0.4)
    collar = np.minimum(collar, frill)
    return np.minimum(d, collar)


def pants(P):
    d = ellipsoid(P, V(0, 0.3, 45.8), (11.0, 8.0, 6.6))
    for s in SIDES:
        d = smin(d, round_cone(P, (s * 5.9, 0.3, 45.0), (s * 7.2, 0.0, 22.0), 6.4, 5.7), 1.5)
        d = np.minimum(d, band(P, V(s * 7.2, 0, 0), (0, 0, 1), 20.6, 22.6, 6.1, 6.1, rnd=0.4))
        bottom, top = V(s * 9.0, -0.4, -0.5), V(s * 7.2, 0.0, 21.5)
        bell = band(P, bottom, top - bottom, 0.0, np.linalg.norm(top - bottom), 7.8, 5.8,
                    amp0=0.6, amp1=0.0, n=9, phase=0.7 if s > 0 else 2.1, rnd=0.5)
        d = smin(d, bell, 0.6)
    return d


def shoes(P):
    d = BIG
    for s in SIDES:
        shoe = ellipsoid(P, V(s * 9.0, -7.0, 2.0), (3.9, 5.6, 3.0))
        sole = ellipsoid(P, V(s * 9.0, -7.0, 0.6), (4.3, 6.0, 1.1))
        d = np.minimum(d, smin(shoe, sole, 0.3))
    return d


def base(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    k = np.sqrt((x / 27.0) ** 2 + ((y + 1.5) / 22.5) ** 2)
    return smax((k - 1.0) * 22.5, np.maximum(-4.5 - z, z), 1.0)


# name -> (sdf, preview colour)
MATERIALS = {
    "skin":    (skin,    (0.96, 0.80, 0.69)),
    "eyes":    (eyes,    (0.10, 0.07, 0.06)),
    "hair":    (hair,    (0.24, 0.15, 0.10)),
    "ribbons": (ribbons, (0.95, 0.86, 0.33)),
    "jacket":  (jacket,  (0.76, 0.80, 0.35)),
    "pants":   (pants,   (0.68, 0.76, 0.30)),
    "shoes":   (shoes,   (0.93, 0.93, 0.93)),
    "base":    (base,    (0.42, 0.40, 0.40)),
}
# odd offsets keep grid samples off flat faces such as the top of the base
BOUNDS = (V(-29.03, -26.07, -6.11), V(29.0, 23.0, 103.5))


# ----------------------------------------------------------------- sampling

def grid_axes(h):
    lo, hi = BOUNDS
    return [np.arange(lo[i], hi[i] + h, h) for i in range(3)]


def eval_grid(fn, axes, chunk=600_000):
    xs, ys, zs = axes
    out = np.empty((len(xs), len(ys), len(zs)), dtype=np.float32)
    plane = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    nz = max(1, chunk // len(plane))
    for k in range(0, len(zs), nz):
        zz = zs[k:k + nz]
        P = np.concatenate([np.repeat(plane, len(zz), 0),
                            np.tile(zz, len(plane))[:, None]], 1)
        out[:, :, k:k + nz] = fn(P).reshape(len(xs), len(ys), len(zz))
    return out


def eval_material(fn, h, coarse=1.0, thresh=3.0):
    """Evaluate on the fine grid, but only inside the region the material can occupy."""
    ca = grid_axes(coarse)
    cd = eval_grid(fn, ca)
    idx = np.argwhere(cd < thresh)
    fa = grid_axes(h)
    out = np.full([len(a) for a in fa], thresh, dtype=np.float32)
    if len(idx) == 0:
        return out
    sub = []
    for i in range(3):
        lo = ca[i][idx[:, i].min()] - thresh
        hi = ca[i][idx[:, i].max()] + thresh
        sub.append(np.where((fa[i] >= lo) & (fa[i] <= hi))[0])
    sl = tuple(slice(s[0], s[-1] + 1) for s in sub)
    out[sl] = eval_grid(fn, [fa[i][sub[i]] for i in range(3)])
    return out


def mesh_from_field(field, h, smooth=True, target_faces=None):
    lo = BOUNDS[0]
    # pad so every surface closes
    f = np.pad(field, 1, constant_values=max(1.0, float(field.max())))
    f[f == 0] = 1e-6
    verts, faces, _, _ = measure.marching_cubes(f, level=0.0, spacing=(h, h, h))
    verts += lo - h
    m = trimesh.Trimesh(verts, faces, process=True)
    # drop sub-voxel specks that marching cubes can leave at razor-thin contacts
    bodies = m.split(only_watertight=False)
    if len(bodies) > 1:
        m = trimesh.util.concatenate([b for b in bodies if abs(b.volume) > 0.5])
    if m.volume < 0:
        m.invert()
    if smooth:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=6)
    if target_faces and target_faces < 1:      # a fraction of the marching-cubes count
        target_faces = int(len(m.faces) * target_faces)
    if target_faces and len(m.faces) > target_faces:
        v, fcs = fast_simplification.simplify(
            m.vertices.astype(np.float32), m.faces.astype(np.int64),
            target_reduction=1.0 - target_faces / len(m.faces))
        dec = trimesh.Trimesh(v, fcs, process=True)
        trimesh.repair.fix_normals(dec)
        if dec.is_watertight:                # otherwise keep the full-detail mesh
            m = dec
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--res", type=float, default=0.25, help="voxel size in mm")
    ap.add_argument("--height", type=float, default=0.0,
                    help="rescale so the whole model (with base) is this tall, in mm")
    ap.add_argument("--faces", type=int, default=400_000,
                    help="triangle budget of the one-piece STL")
    ap.add_argument("--no-parts", action="store_true", help="skip the multi-colour parts")
    args = ap.parse_args()

    t0 = time.time()
    fields = {}
    for name, (fn, _) in MATERIALS.items():
        fields[name] = eval_material(fn, args.res)
        print(f"  sampled {name:8s} {time.time() - t0:6.1f}s", flush=True)

    names = list(fields)
    stack = np.stack([fields[n] for n in names])
    total = stack.min(0)

    # base bottom sits at z = -4.5; move it to the build plate, then scale
    lift = 4.5
    whole = mesh_from_field(total, args.res, target_faces=args.faces)
    scale = args.height / (whole.extents[2]) if args.height else 1.0

    def place(m):
        m.apply_translation([0, 0, lift])
        m.apply_scale(scale)
        return m

    place(whole)
    os.makedirs(args.out, exist_ok=True)
    stl = os.path.join(args.out, "girl_figurine.stl")
    whole.export(stl)
    print(f"wrote {stl}: {len(whole.faces)} triangles, watertight={whole.is_watertight}, "
          f"bodies={len(whole.split(only_watertight=False))}, size={np.round(whole.extents, 1)} mm")

    # colour the preview by the material that owns each vertex
    probe = whole.vertices / scale - [0, 0, lift] - whole.vertex_normals * 0.05
    owner = np.argmin(np.stack([MATERIALS[n][0](probe) for n in names]), 0)
    rgb = np.array([MATERIALS[n][1] for n in names])[owner]
    colored = whole.copy()
    colored.visual = trimesh.visual.ColorVisuals(
        colored, vertex_colors=(np.c_[rgb, np.ones(len(rgb))] * 255).astype(np.uint8))
    glb = os.path.join(args.out, "girl_figurine_color.glb")
    colored.export(glb)
    print(f"wrote {glb}")

    if args.no_parts:
        return
    part_dir = os.path.join(args.out, "multicolor")
    os.makedirs(part_dir, exist_ok=True)
    for i, name in enumerate(names):
        others = np.delete(stack, i, 0).min(0)
        own = np.maximum(stack[i], stack[i] - others)   # where this material is the closest one
        if (own < 0).sum() == 0:
            continue
        part = place(mesh_from_field(own, args.res, smooth=False, target_faces=0.35))
        path = os.path.join(part_dir, f"{i + 1}_{name}.stl")
        part.export(path)
        print(f"wrote {path}: {len(part.faces)} triangles, watertight={part.is_watertight}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

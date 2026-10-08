#!/usr/bin/env python3
"""Turn a photo into a standing lithophane (a plate that shows the photo when lit from behind).

Dark pixels become thick plastic and bright pixels thin plastic. The plate has a
frame and a foot so it stands on its own. The front is flat and the relief is on
the back, so the picture reads correctly from the front.

Example:
  python lithophane.py photo.jpg -o photo_lithophane.stl --height 120
"""
import argparse

import cv2
import fast_simplification
import manifold3d as mf
import numpy as np
import trimesh


def heightfield_solid(T, px):
    """Closed mesh with a flat face at y=0 and thickness T[row, col] towards +y.

    Row 0 is the top of the picture; columns run along +x, rows run down -z.
    """
    rows, cols = T.shape
    u, v = np.meshgrid(np.arange(cols) * px, (rows - 1 - np.arange(rows)) * px)
    front = np.c_[u.ravel(), np.zeros(T.size), v.ravel()]
    back = np.c_[u.ravel(), T.ravel(), v.ravel()]
    verts = np.vstack([front, back])
    n = T.size

    idx = np.arange(n).reshape(rows, cols)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    back_f = np.vstack([np.c_[a, b, c], np.c_[b, d, c]]) + n   # normals +y
    front_f = back_f[:, ::-1] - n                               # normals -y

    ring = np.r_[idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]]
    nxt = np.roll(ring, -1)
    walls = np.vstack([np.c_[ring, nxt, ring + n], np.c_[nxt, nxt + n, ring + n]])
    m = trimesh.Trimesh(verts, np.vstack([front_f, back_f, walls]), process=True)
    if m.volume < 0:
        m.invert()
    return m


def to_manifold(m):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                               tri_verts=np.asarray(m.faces, np.uint32)))


def from_manifold(M):
    mesh = M.to_mesh()
    return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("-o", "--out", default="lithophane.stl")
    ap.add_argument("--height", type=float, default=120.0, help="picture height in mm")
    ap.add_argument("--px", type=float, default=0.2, help="mm per pixel")
    ap.add_argument("--tmin", type=float, default=0.8, help="thickness of white, mm")
    ap.add_argument("--tmax", type=float, default=3.0, help="thickness of black, mm")
    ap.add_argument("--border", type=float, default=3.0, help="frame width, mm")
    ap.add_argument("--no-foot", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.photo, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {args.photo}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = int(round(args.height / args.px))
    cols = int(round(rows * gray.shape[1] / gray.shape[0]))
    gray = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    # lift local contrast (old prints are flat and vignetted), then denoise a touch
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 25, 5)
    lum = gray.astype(np.float64) / 255.0

    T = args.tmin + (1.0 - lum) * (args.tmax - args.tmin)
    b = int(round(args.border / args.px))
    T = np.pad(T, b, constant_values=args.tmax + 0.2)          # frame
    plate = heightfield_solid(T, args.px)

    solid = to_manifold(plate)
    if not args.no_foot:
        w = plate.extents[0]
        foot = mf.Manifold.cube([w, 16.0, 5.0]).translate([0.0, -6.5, -4.5])
        solid = solid + foot
    out = from_manifold(solid)

    v, f = fast_simplification.simplify(out.vertices.astype(np.float32),
                                        out.faces.astype(np.int64), target_reduction=0.75)
    slim = trimesh.Trimesh(v, f, process=True)
    if slim.is_watertight:
        out = slim
    out.apply_translation([-out.bounds.mean(0)[0], -out.bounds.mean(0)[1], -out.bounds[0][2]])
    out.export(args.out)
    print(f"wrote {args.out}: {len(out.faces)} triangles, watertight={out.is_watertight}, "
          f"size={np.round(out.extents, 1)} mm")


if __name__ == "__main__":
    main()

"""Run DA3 with the 3D Gaussian head on a directory of images.

Usage:
    pixi run python run_gs_inference.py [IMAGE_DIR] [EXPORT_DIR] [--model MODEL_ID]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import trimesh

from depth_anything_3.api import DepthAnything3


def convert_glb_to_ply(glb_path: str) -> list[str]:
    """Write PLY siblings of ``glb_path`` so MeshLab can open the scene.

    MeshLab's glTF importer asserts on non-triangle primitives (POINTS / LINES);
    the DA3 GLB contains exactly those. We split into a point-cloud PLY and an
    edge PLY for the camera frustums.
    """
    scene = trimesh.load(glb_path, force="scene")
    stem, _ = os.path.splitext(glb_path)
    written: list[str] = []

    pcs, paths = [], []
    for g in scene.geometry.values():
        if isinstance(g, trimesh.points.PointCloud):
            pcs.append(g)
        elif isinstance(g, trimesh.path.Path3D):
            paths.append(g)

    if pcs:
        pts = np.concatenate([p.vertices for p in pcs], axis=0)
        cols = []
        for p in pcs:
            c = p.colors
            if c is None or len(c) != len(p.vertices):
                c = np.full((len(p.vertices), 4), 200, dtype=np.uint8)
            cols.append(c)
        out = f"{stem}_points.ply"
        trimesh.PointCloud(vertices=pts, colors=np.concatenate(cols, axis=0)).export(out)
        written.append(out)

    if paths:
        verts, edges = [], []
        offset = 0
        for path in paths:
            v = np.asarray(path.vertices)
            verts.append(v)
            for ent in path.entities:
                pts_i = np.asarray(ent.points)
                for a, b in zip(pts_i[:-1], pts_i[1:]):
                    edges.append([a + offset, b + offset])
            offset += len(v)
        V = np.concatenate(verts, axis=0)
        E = np.asarray(edges, dtype=np.int32)
        out = f"{stem}_cameras.ply"
        with open(out, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(V)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element edge {len(E)}\n")
            f.write("property int vertex1\nproperty int vertex2\n")
            f.write("end_header\n")
            for x, y, z in V:
                f.write(f"{x} {y} {z}\n")
            for a, b in E:
                f.write(f"{a} {b}\n")
        written.append(out)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", nargs="?", default="/home/andy/concant0520/pb2_imgs")
    parser.add_argument("export_dir", nargs="?", default="/home/andy/concant0520/da3_gs_out")
    parser.add_argument("--model", default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument(
        "--ext",
        default="jpg,jpeg,png",
        help="Comma-separated image extensions (case-insensitive)",
    )
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    images: list[str] = []
    for ext in args.ext.split(","):
        ext = ext.strip().lstrip(".")
        images.extend(glob.glob(os.path.join(args.image_dir, f"*.{ext}")))
        images.extend(glob.glob(os.path.join(args.image_dir, f"*.{ext.upper()}")))
    images = sorted(set(images))
    assert images, f"no images found in {args.image_dir}"
    print(f"[info] {len(images)} images", flush=True)

    print(f"[info] loading model {args.model} ...", flush=True)
    t0 = time.time()
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained(args.model).to(device)
    print(f"[info] model loaded in {time.time() - t0:.1f}s", flush=True)

    print("[info] running inference (infer_gs=True, export=glb-gs_ply) ...", flush=True)
    t0 = time.time()
    pred = model.inference(
        image=images,
        infer_gs=True,
        export_dir=args.export_dir,
        export_format="glb-gs_ply",
    )
    print(f"[info] inference + export done in {time.time() - t0:.1f}s", flush=True)

    print(f"[info] depth shape:      {pred.depth.shape}", flush=True)
    print(f"[info] extrinsics shape: {pred.extrinsics.shape}", flush=True)
    print(f"[info] intrinsics shape: {pred.intrinsics.shape}", flush=True)
    print(f"[info] peak gpu mem:     {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"[info] results in:       {args.export_dir}", flush=True)

    glb_path = os.path.join(args.export_dir, "scene.glb")
    if os.path.exists(glb_path):
        try:
            written = convert_glb_to_ply(glb_path)
            for p in written:
                print(f"[info] wrote {p}", flush=True)
        except Exception as e:
            print(f"[warn] glb->ply conversion failed: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

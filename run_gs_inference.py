"""Run DA3 with the 3D Gaussian head on a directory of images.

Usage:
    pixi run python run_gs_inference.py [IMAGE_DIR] [EXPORT_DIR] [--model MODEL_ID]
"""

import argparse
import glob
import os
import sys
import time

import torch

from depth_anything_3.api import DepthAnything3


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


if __name__ == "__main__":
    sys.exit(main())

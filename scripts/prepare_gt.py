#!/usr/bin/env python3
"""Stage 02: build the ground-truth evaluation set.

1. Extract the 29 ground-truth frames (Images/ minus masks/).
2. Locate the pot in each one (Segment Anything with a box prompt; GrabCut fallback).
3. Generate the paper's four mask types per frame (seeded -> reproducible).
4. Write readable preview sheets (pot outline + zoomed crops with each mask).

Per-frame corrections: if the pot outline is wrong for some frame, add its relative
box to <out>/vase_boxes.json, e.g. {"frame_00090": [0.30, 0.03, 0.85, 0.68]}, and re-run.

Resumable. Stored masks carry a version/method stamp; if it differs from the current
settings the stored masks are discarded and rebuilt (so stale results are never reused).
Output under <drive>/results/gt/: images/ vase_masks/ masks/<type>/ previews/ manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from heritage.gt import extract_gt, split_frames  # noqa: E402
from heritage.paths import get_paths, load_config  # noqa: E402
from heritage.synth_masks import (  # noqa: E402
    MASK_TYPES, crop_box, make_masks, overlay, vase_mask_grabcut, vase_mask_sam,
)

ROWS_PER_SHEET = 8
CYAN, MAGENTA = (0, 255, 255), (255, 0, 255)


def read_rgb(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"cannot read {p}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def sync_stamp(out: Path, stamp: dict) -> bool:
    """Discard stored vase/masks if they were made with different settings. True if discarded."""
    f = out / "masks_meta.json"
    old = json.loads(f.read_text()) if f.exists() else None
    stale = old != stamp
    if stale:
        for d in ("vase_masks", "masks", "previews"):
            shutil.rmtree(out / d, ignore_errors=True)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(stamp, indent=2))
    return stale and old is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--images", default=None, help="override Images folder")
    ap.add_argument("--masks", default=None, help="override masks folder")
    ap.add_argument("--out", default=None, help="override output folder")
    ap.add_argument("--method", choices=["sam", "grabcut"], default="sam")
    ap.add_argument("--box", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                    help="relative box containing the pot (default from config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = get_paths(cfg).ensure()
    ds, seg = cfg["dataset"], cfg["segmentation"]
    images = Path(args.images) if args.images else P.data / ds["images_subdir"]
    masks = Path(args.masks) if args.masks else P.data / ds["masks_subdir"]
    out = Path(args.out) if args.out else P.stage("gt")
    box = tuple(args.box) if args.box else tuple(seg["vase_box"])

    manifest = extract_gt(images, masks, out, ds["expected_gt_frames"], ds["expected_damaged_frames"])
    print(f"GT frames: {manifest['n_gt']}  runs: {manifest['gt_runs']}  (damaged: {manifest['n_damaged']})")

    overrides_file = out / "vase_boxes.json"
    overrides = json.loads(overrides_file.read_text()) if overrides_file.exists() else {}
    stamp = {"version": seg["gt_mask_version"], "method": args.method, "box": list(box),
             "sam_model": seg["sam_model"] if args.method == "sam" else None, "overrides": overrides,
             "seed": cfg["project"]["seed"]}
    if sync_stamp(out, stamp):
        print("Settings changed since the last run -> stored masks discarded and rebuilt.")

    predictor = None
    if args.method == "sam":
        from heritage.models import fetch_sam_checkpoint, load_sam_predictor

        ckpt = fetch_sam_checkpoint(P.checkpoints, seg["sam_model"])
        predictor = load_sam_predictor(ckpt, seg["sam_model"])
        print(f"SAM {seg['sam_model']} loaded")

    (out / "vase_masks").mkdir(exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    for t in MASK_TYPES:
        (out / "masks" / t).mkdir(parents=True, exist_ok=True)

    split = split_frames(images, masks)
    rows, report = [], []
    for n, src in split["gt"]:
        name = f"frame_{n:05d}.png"
        key = f"frame_{n:05d}"
        img = read_rgb(out / "images" / f"frame_{n:05d}{src.suffix.lower()}")
        done = all((out / "masks" / t / name).exists() for t in MASK_TYPES) and (out / "vase_masks" / name).exists()
        if done:
            vase = cv2.imread(str(out / "vase_masks" / name), 0) > 127
            ms = {t: cv2.imread(str(out / "masks" / t / name), 0) for t in MASK_TYPES}
        else:
            b = tuple(overrides.get(key, box))
            vase = vase_mask_sam(img, predictor, b) if predictor else vase_mask_grabcut(img, b)
            ms = make_masks(vase, seed=cfg["project"]["seed"] + n)
            cv2.imwrite(str(out / "vase_masks" / name), vase.astype(np.uint8) * 255)
            for t, m in ms.items():
                cv2.imwrite(str(out / "masks" / t / name), m)

        report.append({"frame": n, "vase_frac_of_image": round(float(vase.mean()), 3),
                       "mask_frac_of_vase": {t: round(float((ms[t] > 0).sum()) / max(1, vase.sum()), 3) for t in MASK_TYPES}})

        x0, y0, x1, y1 = crop_box(vase)
        full = cv2.resize(overlay(img, vase, CYAN), (384, 216), interpolation=cv2.INTER_AREA)
        crops = [cv2.resize(overlay(img, ms[t] > 0, MAGENTA)[y0:y1, x0:x1], (216, 216), interpolation=cv2.INTER_AREA)
                 for t in MASK_TYPES]
        row = np.hstack([full] + crops)
        cv2.putText(row, key[-3:], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        rows.append(row)

    sheets = []
    for i in range(0, len(rows), ROWS_PER_SHEET):
        f = out / "previews" / f"gt_masks_sheet_{i // ROWS_PER_SHEET + 1}.jpg"
        cv2.imwrite(str(f), cv2.cvtColor(np.vstack(rows[i:i + ROWS_PER_SHEET]), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        sheets.append(f.name)
    (out / "mask_report.json").write_text(json.dumps(report, indent=2))
    fr = [r["vase_frac_of_image"] for r in report]
    print(f"pot covers {min(fr):.0%}-{max(fr):.0%} of the frame; sheets: {sheets}")
    print("columns: full frame with pot outline (cyan) | zoomed crops with mask (magenta): " + " | ".join(MASK_TYPES))


if __name__ == "__main__":
    main()

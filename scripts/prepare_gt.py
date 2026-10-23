#!/usr/bin/env python3
"""Stage 02: build the ground-truth evaluation set.

1. Extract the 29 ground-truth frames (Images/ minus masks/).
2. Estimate the vase region in each (GrabCut, seeded by the nearest damaged
   frame's annotated mask when available).
3. Generate the paper's four mask types per frame (seeded -> reproducible).
4. Write a preview contact sheet so the result can be checked by eye.

Resumable and idempotent. Output under <drive>/results/gt/:
    images/  vase_masks/  masks/<type>/  previews/  manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from heritage.gt import extract_gt, split_frames  # noqa: E402
from heritage.paths import get_paths, load_config  # noqa: E402
from heritage.synth_masks import MASK_TYPES, make_masks, overlay, to_bool_mask, vase_mask_grabcut  # noqa: E402


def read_rgb(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"cannot read {p}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--images", default=None, help="override Images folder")
    ap.add_argument("--masks", default=None, help="override masks folder")
    ap.add_argument("--out", default=None, help="override output folder")
    ap.add_argument("--no-prior", action="store_true", help="do not seed GrabCut with neighbouring annotated masks")
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = get_paths(cfg).ensure()
    ds = cfg["dataset"]
    images = Path(args.images) if args.images else P.data / ds["images_subdir"]
    masks = Path(args.masks) if args.masks else P.data / ds["masks_subdir"]
    out = Path(args.out) if args.out else P.stage("gt")

    manifest = extract_gt(images, masks, out, ds["expected_gt_frames"], ds["expected_damaged_frames"])
    print(f"GT frames: {manifest['n_gt']}  runs: {manifest['gt_runs']}  (damaged: {manifest['n_damaged']})")

    split = split_frames(images, masks)
    damaged = {n: m for n, _, m in split["damaged"]}
    damaged_nums = np.array(sorted(damaged))
    seed0 = cfg["project"]["seed"]

    (out / "vase_masks").mkdir(exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    for t in MASK_TYPES:
        (out / "masks" / t).mkdir(parents=True, exist_ok=True)

    sheet_rows, report = [], []
    for n, src in split["gt"]:
        name = f"frame_{n:05d}.png"
        done = all((out / "masks" / t / name).exists() for t in MASK_TYPES) and (out / "vase_masks" / name).exists()
        img = read_rgb(out / "images" / f"frame_{n:05d}{src.suffix.lower()}")

        if done:
            vase = cv2.imread(str(out / "vase_masks" / name), 0) > 127
            ms = {t: cv2.imread(str(out / "masks" / t / name), 0) for t in MASK_TYPES}
        else:
            prior = None
            if not args.no_prior:
                nearest = int(damaged_nums[np.argmin(np.abs(damaged_nums - n))])
                prior = to_bool_mask(cv2.imread(str(damaged[nearest]), cv2.IMREAD_UNCHANGED))
                if prior.shape != img.shape[:2]:
                    prior = cv2.resize(prior.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            vase = vase_mask_grabcut(img, prior=prior)
            ms = make_masks(vase, seed=seed0 + n)
            cv2.imwrite(str(out / "vase_masks" / name), vase.astype(np.uint8) * 255)
            for t, m in ms.items():
                cv2.imwrite(str(out / "masks" / t / name), m)

        area = {t: round(float((ms[t] > 0).sum()) / max(1, vase.sum()), 3) for t in MASK_TYPES}
        report.append({"frame": n, "vase_frac_of_image": round(float(vase.mean()), 3), "mask_frac_of_vase": area})

        thumbs = [img, overlay(img, vase, (0, 160, 255))] + [overlay(img, ms[t]) for t in MASK_TYPES]
        sheet_rows.append(np.hstack([cv2.resize(x, (256, 144), interpolation=cv2.INTER_AREA) for x in thumbs]))

    sheet = np.vstack(sheet_rows)
    cv2.imwrite(str(out / "previews" / "gt_masks_contact_sheet.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "mask_report.json").write_text(json.dumps(report, indent=2))
    fr = [r["vase_frac_of_image"] for r in report]
    print(f"vase covers {min(fr):.0%}-{max(fr):.0%} of the frame; sheet: {out/'previews'/'gt_masks_contact_sheet.jpg'}")
    print("columns: original | vase estimate | " + " | ".join(MASK_TYPES))


if __name__ == "__main__":
    main()

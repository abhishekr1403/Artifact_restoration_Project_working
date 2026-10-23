"""Ground-truth frame extraction.

The Figshare archive ships only ``Images/`` (334 frames) and ``masks/`` (305
masks). The paper's 29 "ground truth" frames are the ones where the damage is
NOT visible, so the authors never drew a mask for them. Therefore:

    ground truth frames  =  frames in Images/ that have NO file in masks/

This module derives that set, validates it against the expected counts, and
copies the frames into their own folder together with a manifest.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .ids import build_index, pick


def split_frames(images_dir: str | Path, masks_dir: str | Path) -> Dict[str, object]:
    """Split frames into ground-truth (no mask) and damaged (has mask).

    Returns a dict with:
      gt            sorted [(frame_number, image_path)]
      damaged       sorted [(frame_number, image_path, mask_path)]
      orphan_masks  frame numbers that have a mask but no image (should be empty)
    """
    img_idx = build_index(images_dir)
    msk_idx = build_index(masks_dir)
    if not img_idx:
        raise FileNotFoundError(f"No images found in {images_dir}")
    if not msk_idx:
        raise FileNotFoundError(f"No masks found in {masks_dir}")

    gt, damaged = [], []
    for n in sorted(img_idx):
        img = pick(img_idx[n])
        if n in msk_idx:
            damaged.append((n, img, pick(msk_idx[n])))
        else:
            gt.append((n, img))
    return {"gt": gt, "damaged": damaged, "orphan_masks": sorted(set(msk_idx) - set(img_idx))}


def contiguous_runs(numbers: List[int]) -> List[List[int]]:
    """Group sorted integers into [start, end] runs, e.g. [84..112] -> [[84, 112]]."""
    runs: List[List[int]] = []
    for n in sorted(numbers):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return runs


def extract_gt(
    images_dir: str | Path,
    masks_dir: str | Path,
    out_dir: str | Path,
    expected_gt: Optional[int] = 29,
    expected_damaged: Optional[int] = 305,
) -> Dict[str, object]:
    """Copy the ground-truth frames to ``out_dir`` and write ``manifest.json``.

    Idempotent: files already present with the same size are skipped, so it is
    safe to re-run after a Colab disconnect. Raises if the counts do not match
    the paper, because every later metric depends on this split being right.
    """
    s = split_frames(images_dir, masks_dir)
    gt, damaged = s["gt"], s["damaged"]
    problems = []
    if expected_gt is not None and len(gt) != expected_gt:
        problems.append(f"expected {expected_gt} GT frames, found {len(gt)}")
    if expected_damaged is not None and len(damaged) != expected_damaged:
        problems.append(f"expected {expected_damaged} damaged frames, found {len(damaged)}")
    if s["orphan_masks"]:
        problems.append(f"{len(s['orphan_masks'])} masks have no matching image: {s['orphan_masks'][:5]}")
    if problems:
        raise ValueError("Ground-truth split does not match the paper: " + "; ".join(problems))

    out_dir = Path(out_dir)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for n, src in gt:
        dst = img_out / f"frame_{n:05d}{src.suffix.lower()}"
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst)
        copied += 1

    numbers = [n for n, _ in gt]
    manifest = {
        "rule": "ground-truth frame = image in Images/ with no matching file in masks/",
        "n_images": len(gt) + len(damaged),
        "n_gt": len(gt),
        "n_damaged": len(damaged),
        "gt_frames": numbers,
        "gt_runs": contiguous_runs(numbers),
        "damaged_runs": contiguous_runs([n for n, _, _ in damaged]),
        "newly_copied": copied,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

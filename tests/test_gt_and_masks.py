import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heritage.gt import contiguous_runs, extract_gt, split_frames  # noqa: E402
from heritage.synth_masks import MASK_TYPES, make_masks, overlay, to_bool_mask, vase_mask_grabcut  # noqa: E402

H, W = 108, 192


def vase_img(shift=0, rng=None):
    """Synthetic frame: textured grey background, orange ellipse 'vase'."""
    rng = rng or np.random.default_rng(0)
    img = np.full((H, W, 3), 110, np.uint8) + rng.integers(0, 12, (H, W, 3), dtype=np.uint8)
    cv2.ellipse(img, (W // 2 + shift, H // 2), (30, 46), 0, 0, 360, (200, 110, 40), -1)
    vase = np.zeros((H, W), np.uint8)
    cv2.ellipse(vase, (W // 2 + shift, H // 2), (30, 46), 0, 0, 360, 255, -1)
    return img, vase


def make_dataset(root: Path, n=334, gt=range(84, 113)):
    (root / "Images").mkdir(parents=True)
    (root / "masks").mkdir()
    for i in range(1, n + 1):
        img, vase = vase_img(shift=i % 5)
        cv2.imwrite(str(root / "Images" / f"frame_{i:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if i not in gt:
            cv2.imwrite(str(root / "masks" / f"frame_{i:05d}_mask.png"), vase)  # white = region


def test_split_derives_gt_as_frames_without_masks(tmp_path):
    make_dataset(tmp_path)
    s = split_frames(tmp_path / "Images", tmp_path / "masks")
    assert len(s["gt"]) == 29 and len(s["damaged"]) == 305 and s["orphan_masks"] == []
    assert [n for n, _ in s["gt"]] == list(range(84, 113))
    assert contiguous_runs([n for n, _ in s["gt"]]) == [[84, 112]]


def test_extract_gt_copies_and_is_idempotent(tmp_path):
    make_dataset(tmp_path / "d")
    out = tmp_path / "gt"
    m1 = extract_gt(tmp_path / "d/Images", tmp_path / "d/masks", out)
    assert m1["n_gt"] == 29 and m1["newly_copied"] == 29 and m1["gt_runs"] == [[84, 112]]
    assert len(list((out / "images").glob("*.png"))) == 29
    m2 = extract_gt(tmp_path / "d/Images", tmp_path / "d/masks", out)
    assert m2["newly_copied"] == 0
    assert json.loads((out / "manifest.json").read_text())["gt_frames"][0] == 84


def test_extract_gt_rejects_wrong_counts(tmp_path):
    make_dataset(tmp_path / "d", gt=range(84, 110))  # only 26 GT
    with pytest.raises(ValueError, match="expected 29"):
        extract_gt(tmp_path / "d/Images", tmp_path / "d/masks", tmp_path / "gt")


def test_orphan_mask_is_reported(tmp_path):
    make_dataset(tmp_path / "d")
    (tmp_path / "d/Images/frame_00400.png").unlink(missing_ok=True)
    cv2.imwrite(str(tmp_path / "d/masks/frame_00400_mask.png"), np.zeros((H, W), np.uint8))
    with pytest.raises(ValueError, match="no matching image"):
        extract_gt(tmp_path / "d/Images", tmp_path / "d/masks", tmp_path / "gt")


def test_mask_polarity_is_detected():
    _, vase = vase_img()
    assert (to_bool_mask(vase) == (vase > 0)).all()          # white on black
    assert (to_bool_mask(255 - vase) == (vase > 0)).all()    # black on white (inverted file)
    assert (to_bool_mask(np.dstack([vase] * 3)) == (vase > 0)).all()


def test_grabcut_finds_the_vase_with_and_without_prior():
    img, vase = vase_img()
    v = vase > 0
    iou = lambda a, b: (a & b).sum() / (a | b).sum()  # noqa: E731
    assert iou(vase_mask_grabcut(img), v) > 0.85
    _, shifted = vase_img(shift=3)                       # neighbouring frame's mask as prior
    assert iou(vase_mask_grabcut(img, prior=shifted > 0), v) > 0.85


def test_make_masks_types_geometry_and_determinism():
    _, vase = vase_img()
    v = vase > 0
    a, b = make_masks(v, seed=7), make_masks(v, seed=7)
    assert set(a) == set(MASK_TYPES)
    for t in MASK_TYPES:
        assert (a[t] == b[t]).all()                                    # reproducible
        m = a[t] > 0
        assert set(np.unique(a[t])) <= {0, 255} and m.sum() > 50       # binary, non-empty
        loose = cv2.dilate(vase, np.ones((7, 7), np.uint8)) > 0
        assert (m & ~loose).sum() == 0                                 # only on the vase
    area = {t: (a[t] > 0).sum() / v.sum() for t in MASK_TYPES}
    assert area["large_marginal"] > area["marginal"]
    assert (make_masks(v, seed=8)["multiple"] != a["multiple"]).any()  # seed matters
    # rectangle sits in the interior, marginal touches the edge
    interior = cv2.erode(vase, np.ones((9, 9), np.uint8)) > 0
    rect = a["rectangle"] > 0
    assert (rect & interior).sum() / rect.sum() > 0.6
    edge = v & ~(cv2.erode(vase, np.ones((9, 9), np.uint8)) > 0)
    assert (a["marginal"] > 0)[edge].sum() > 0


def test_make_masks_rejects_empty_vase():
    with pytest.raises(ValueError):
        make_masks(np.zeros((H, W), bool), seed=1)


def test_overlay_keeps_shape_and_changes_masked_pixels():
    img, vase = vase_img()
    o = overlay(img, vase)
    assert o.shape == img.shape and o.dtype == np.uint8 and (o != img).any()


def test_prepare_gt_script_end_to_end(tmp_path):
    make_dataset(tmp_path / "d")
    env = {"HERITAGE_DRIVE": str(tmp_path / "drive"), "HERITAGE_LOCAL": str(tmp_path / "local"),
           "PATH": "/usr/bin:/bin:/usr/local/bin"}
    cmd = [sys.executable, str(ROOT / "scripts/prepare_gt.py"), "--images", str(tmp_path / "d/Images"),
           "--masks", str(tmp_path / "d/masks")]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    out = tmp_path / "drive/results/gt"
    assert len(list((out / "images").glob("*.png"))) == 29
    for t in MASK_TYPES:
        assert len(list((out / "masks" / t).glob("*.png"))) == 29
    assert (out / "previews/gt_masks_contact_sheet.jpg").exists()
    rep = json.loads((out / "mask_report.json").read_text())
    assert len(rep) == 29 and all(0.02 < x["vase_frac_of_image"] < 0.9 for x in rep)
    before = (out / "masks/multiple/frame_00090.png").read_bytes()
    r2 = subprocess.run(cmd, env=env, capture_output=True, text=True)       # resume run
    assert r2.returncode == 0 and (out / "masks/multiple/frame_00090.png").read_bytes() == before

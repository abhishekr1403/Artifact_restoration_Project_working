"""Synthetic test masks for the 29 ground-truth frames.

The paper evaluates inpainting where the answer is known: on undamaged frames
it hides part of the vase with a free-hand mask, inpaints it, and compares the
result with the original. Four mask types are used per image:

  rectangle       one rectangle-like region in the INTERIOR of the vase
  marginal        one small blob at the EDGE of the vase
  large_marginal  one larger blob at the edge of the vase
  multiple        several blobs of different sizes at various positions

"All masks are applied exclusively to the vase" and have to be custom for every
frame because the vase moves a little between frames. The authors drew them by
hand; here they are generated reproducibly from a vase mask (seeded by frame
number), so results can be re-created exactly on any machine.

Mask convention written by this module: uint8, 255 = region to inpaint
(the convention LaMa and Stable-Diffusion inpainting expect), 0 = keep.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

MASK_TYPES = ("rectangle", "marginal", "large_marginal", "multiple")


# ----------------------------------------------------------------- mask I/O
def to_bool_mask(mask_img: np.ndarray) -> np.ndarray:
    """Convert a loaded mask image to bool (True = region to inpaint).

    Polarity differs between tools, so it is detected instead of assumed: an
    inpainting mask never covers the outer border of a frame, so if the border
    is mostly "on" the mask is inverted.
    """
    m = mask_img
    if m.ndim == 3:
        m = m.max(axis=2)
    on = m > 127
    h, w = on.shape
    b = max(1, int(0.03 * min(h, w)))
    border = np.concatenate([on[:b].ravel(), on[-b:].ravel(), on[:, :b].ravel(), on[:, -b:].ravel()])
    return ~on if border.mean() > 0.5 else on


# --------------------------------------------------------- vase localisation
# NOTE: the 305 annotated masks that ship with the dataset mark only the MISSING
# part of the pot (the broken rim), not the whole pot. They must not be used to
# find the pot itself (an earlier version did, and the "vase" collapsed onto the rim).

# Relative box (x0, y0, x1, y1) that contains the whole pot in the ground-truth
# frames (pot roughly mid-frame, resting on a black stool just below).
DEFAULT_BOX = (0.28, 0.02, 0.92, 0.66)


def _clean_vase(fg: np.ndarray, open_px: int = 5) -> np.ndarray:
    """Largest connected component, holes filled, speckle removed. uint8 0/1 in and out."""
    fg = fg.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))


def choose_mask(masks: np.ndarray, scores: np.ndarray, min_frac: float = 0.04, max_frac: float = 0.5) -> np.ndarray:
    """Pick the vase among SAM's candidate masks (shape (K, H, W)).

    SAM returns several nested guesses (part / object / bigger region). Keep the
    best-scoring one whose size is plausible for a pot and that does not reach
    the left/right image border (that would be the room, not the pot).
    """
    W = masks.shape[2]
    best, best_score = None, -np.inf
    for m, sc in zip(masks, scores):
        frac = float(m.mean())
        cols = np.where(m.any(axis=0))[0]
        touches_side = cols.size > 0 and (cols[0] == 0 or cols[-1] == W - 1)
        if min_frac <= frac <= max_frac and not touches_side and sc > best_score:
            best, best_score = m, sc
    if best is None:
        best = masks[int(np.argmax(scores))]
    return best.astype(bool)


def vase_mask_sam(img_rgb: np.ndarray, predictor, box_rel=DEFAULT_BOX) -> np.ndarray:
    """Segment the pot with Segment Anything (box + centre-point prompt).

    ``predictor`` is a ``segment_anything.SamPredictor`` (injected so the logic can
    be tested without the model). Returns a bool mask at the image resolution.
    """
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = box_rel
    box = np.array([x0 * W, y0 * H, x1 * W, y1 * H], dtype=np.float32)
    centre = np.array([[(x0 + x1) / 2 * W, (y0 + y1) / 2 * H]], dtype=np.float32)
    predictor.set_image(img_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=centre, point_labels=np.array([1]), box=box, multimask_output=True
    )
    m = choose_mask(np.asarray(masks), np.asarray(scores))
    k = max(5, int(0.004 * min(H, W))) | 1
    return _clean_vase(m, open_px=k).astype(bool)


def vase_mask_grabcut(
    img_rgb: np.ndarray,
    box_rel=DEFAULT_BOX,
    work_height: int = 540,
    iters: int = 6,
) -> np.ndarray:
    """CPU-only fallback: GrabCut inside ``box_rel`` (everything outside is background).

    Cheaper but less reliable than SAM on a dark pot against a cluttered lab; use it
    when SAM is unavailable. Always check the preview sheet.
    """
    H, W = img_rgb.shape[:2]
    s = min(1.0, work_height / H)
    small = cv2.resize(img_rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gc = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    x0, y0, x1, y1 = box_rel
    gc[int(y0 * sh):int(y1 * sh), int(x0 * sw):int(x1 * sw)] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    out = _clean_vase(fg)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def crop_box(mask: np.ndarray, pad: float = 0.15) -> Tuple[int, int, int, int]:
    """Square-ish crop (x0, y0, x1, y1) around a bool mask, padded and clipped to the image."""
    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, 0, W, H
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * (1 + 2 * pad)
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    return x0, y0, x1, y1


# --------------------------------------------------------------- generators
def _blob(shape, cx, cy, rx, ry, rot, rng) -> np.ndarray:
    """Irregular free-hand-looking blob (low-frequency wobbly ellipse)."""
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    wob = 1 + 0.18 * (rng.uniform(0.5, 1) * np.sin(2 * t + rng.uniform(0, 6.28))
                      + rng.uniform(0.5, 1) * np.sin(3 * t + rng.uniform(0, 6.28)))
    x, y = rx * np.cos(t) * wob, ry * np.sin(t) * wob
    c, s = np.cos(rot), np.sin(rot)
    pts = np.stack([cx + c * x - s * y, cy + s * x + c * y], axis=1).astype(np.int32)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return m


def _rect_blob(shape, cx, cy, w, h, rot, rng) -> np.ndarray:
    """Rectangle-like free-hand region (rounded, slightly rotated)."""
    half = np.array([[-w, -h], [w, -h], [w, h], [-w, h]], np.float64) / 2
    half += rng.normal(0, 0.03 * min(w, h), half.shape)  # hand jitter
    c, s = np.cos(rot), np.sin(rot)
    pts = (half @ np.array([[c, s], [-s, c]]) + [cx, cy]).astype(np.int32)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return cv2.GaussianBlur(m, (0, 0), 0.01 * min(w, h)) > 127


def make_masks(vase: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Create the four paper-style masks for one frame.

    ``vase`` is a bool mask of the vase. Deterministic for a given (vase, seed);
    use the frame number as the seed so every machine produces the same masks.
    Returns {type: uint8 mask (255 = inpaint)}.
    """
    if vase.sum() < 100:
        raise ValueError("vase mask is (almost) empty - segmentation failed for this frame")
    rng = np.random.default_rng(seed)
    shape = vase.shape
    ys, xs = np.where(vase)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    unit = float(min(bw, bh))
    cy, cx = ys.mean(), xs.mean()

    v8 = vase.astype(np.uint8)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(r)) + 1,) * 2)  # noqa: E731
    loose = cv2.dilate(v8, k(0.02 * unit)).astype(bool)        # may poke "slightly beyond the edge"
    interior = cv2.erode(v8, k(0.12 * unit)).astype(bool)
    edge = vase & ~cv2.erode(v8, k(0.03 * unit)).astype(bool)

    def pts(region, near=None, radius=None):
        yy, xx = np.where(region)
        if len(yy) == 0:
            yy, xx = ys, xs
        if near is not None:
            d = np.hypot(yy - near[0], xx - near[1])
            sel = d < radius
            if sel.any():
                yy, xx = yy[sel], xx[sel]
        i = rng.integers(len(yy))
        return float(xx[i]), float(yy[i])

    out: Dict[str, np.ndarray] = {}

    x, y = pts(interior, near=(cy, cx), radius=0.25 * unit)
    w, h = 0.30 * bw * rng.uniform(0.8, 1.2), 0.22 * bh * rng.uniform(0.8, 1.2)
    out["rectangle"] = _rect_blob(shape, x, y, w, h, np.deg2rad(rng.uniform(-6, 6)), rng)

    for name, r in (("marginal", 0.11), ("large_marginal", 0.25)):
        x, y = pts(edge)
        rr = r * unit * rng.uniform(0.85, 1.15)
        out[name] = _blob(shape, x, y, rr * rng.uniform(0.8, 1.2), rr, rng.uniform(0, np.pi), rng) > 0

    m = np.zeros(shape, bool)
    for _ in range(int(rng.integers(3, 6))):
        x, y = pts(vase)
        rr = rng.uniform(0.05, 0.14) * unit
        m |= _blob(shape, x, y, rr * rng.uniform(0.8, 1.3), rr, rng.uniform(0, np.pi), rng) > 0
    out["multiple"] = m

    final = {}
    for name, m in out.items():
        m = m & loose                       # exclusively on the vase
        if m.sum() < 50:                    # degenerate clip -> fall back to a blob at the centre
            m = _blob(shape, cx, cy, 0.1 * unit, 0.1 * unit, 0, rng) > 0
            m &= loose
        final[name] = (m.astype(np.uint8) * 255)
    return final


# ----------------------------------------------------------------- previews
def overlay(img_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 255), alpha: float = 0.5, thickness: int = 0) -> np.ndarray:
    """Blend a coloured mask over an image (for visual QA)."""
    m = (mask > 0) if mask.dtype != bool else mask
    out = img_rgb.astype(np.float32).copy()
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.ascontiguousarray(out.clip(0, 255).astype(np.uint8))
    cv2.drawContours(out, cnts, -1, color, thickness or max(2, img_rgb.shape[1] // 400))
    return out

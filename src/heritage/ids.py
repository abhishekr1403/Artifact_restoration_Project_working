"""Frame-ID helpers.

Lesson learned from the first implementation: files that belong to the same
frame carry different decorations depending on which stage produced them, e.g.

    frame_00084.png            ground truth / raw frame
    frame_00084_mask001.png    LaMa output
    frame_00084_mean.png       mean of 10 Stable Diffusion runs
    frame_00084_sd_run03.png   one Stable Diffusion run

and the ground-truth frames start at frame_00084 while other folders start at
frame_00001. Matching by "digits in the name" in ONE place avoids the empty
DataFrame / KeyError crashes we hit before.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_FRAME_RE = re.compile(r"frame[_\-]?(\d+)", re.IGNORECASE)
_DIGITS_RE = re.compile(r"(\d+)")


def frame_number(name: str) -> Optional[int]:
    """Return the frame number encoded in a file name, or None.

    Prefers the digits that follow the word "frame"; otherwise falls back to
    the FIRST run of digits (so ``00084.png`` and ``img_84.jpg`` also work).
    Trailing decorations such as ``_mask001`` or ``_run03`` are ignored
    because the frame token always comes first.
    """
    stem = Path(name).stem
    m = _FRAME_RE.search(stem)
    if m:
        return int(m.group(1))
    m = _DIGITS_RE.search(stem)
    return int(m.group(1)) if m else None


def list_images(folder: str | Path, recursive: bool = False) -> List[Path]:
    """Sorted list of image files in ``folder``."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_index(folder: str | Path, recursive: bool = False) -> Dict[int, List[Path]]:
    """Map frame number -> all image files for that frame in ``folder``."""
    index: Dict[int, List[Path]] = defaultdict(list)
    for p in list_images(folder, recursive=recursive):
        n = frame_number(p.name)
        if n is not None:
            index[n].append(p)
    return dict(index)


def pick(candidates: Iterable[Path], prefer: Iterable[str] = ()) -> Optional[Path]:
    """Choose one file from ``candidates``.

    ``prefer`` is a list of substrings (e.g. ``["_mean"]``); the first
    candidate containing the earliest-listed substring wins. With no
    preference, the shortest file name wins (the undecorated original).
    """
    cands = sorted(candidates, key=lambda p: (len(p.name), p.name))
    if not cands:
        return None
    for token in prefer:
        for p in cands:
            if token in p.stem:
                return p
    return cands[0]


def pair_by_frame(
    a_dir: str | Path,
    b_dir: str | Path,
    prefer_a: Iterable[str] = (),
    prefer_b: Iterable[str] = (),
) -> List[tuple[int, Path, Path]]:
    """Pair files from two folders by frame number.

    Returns ``[(frame_number, path_a, path_b), ...]`` sorted by frame number.
    Frames missing from either side are skipped silently; callers that need to
    report them should compare ``build_index`` key sets themselves.
    """
    ia, ib = build_index(a_dir), build_index(b_dir)
    pairs = []
    for n in sorted(set(ia) & set(ib)):
        pa, pb = pick(ia[n], prefer_a), pick(ib[n], prefer_b)
        if pa and pb:
            pairs.append((n, pa, pb))
    return pairs

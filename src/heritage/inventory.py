"""Dataset inventory.

Walks a directory tree and reports, for every folder that contains files:
file counts per extension, frame-number range, sample file names and the image
size of a few samples. Used by notebook 01 to learn what the Figshare archive
actually contains before any processing code is written against it.

CLI:
    python -m heritage.inventory <root> [--out report.json] [--max-depth N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .ids import IMAGE_EXTS, frame_number


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        return None


def inventory(root: str | Path, max_depth: int = 6, sample_sizes: int = 3) -> List[Dict[str, Any]]:
    """Return one record per folder (relative path) that directly holds files."""
    root = Path(root)
    records: List[Dict[str, Any]] = []
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    for folder in sorted([root, *[p for p in root.rglob("*") if p.is_dir()]]):
        rel = folder.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
        if not files:
            continue
        ext_counts = Counter(p.suffix.lower() or "<none>" for p in files)
        images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
        nums = sorted({n for n in (frame_number(p.name) for p in images) if n is not None})
        sizes = Counter()
        for p in images[:sample_sizes]:
            s = _image_size(p)
            if s:
                sizes[f"{s[0]}x{s[1]}"] += 1
        records.append(
            {
                "folder": rel.as_posix() or ".",
                "n_files": len(files),
                "total_mb": round(sum(p.stat().st_size for p in files) / 1e6, 1),
                "extensions": dict(ext_counts),
                "n_images": len(images),
                "frame_min": nums[0] if nums else None,
                "frame_max": nums[-1] if nums else None,
                "n_unique_frames": len(nums),
                "first_files": [p.name for p in files[:3]],
                "last_file": files[-1].name,
                "sample_image_sizes": dict(sizes),
            }
        )
    return records


def format_markdown(records: List[Dict[str, Any]]) -> str:
    lines = ["| folder | files | MB | images | frames (min-max, unique) | sample size | first files |",
             "|---|---|---|---|---|---|---|"]
    for r in records:
        frames = f"{r['frame_min']}-{r['frame_max']} ({r['n_unique_frames']})" if r["frame_min"] is not None else "-"
        size = ", ".join(r["sample_image_sizes"]) or "-"
        lines.append(
            f"| `{r['folder']}` | {r['n_files']} | {r['total_mb']} | {r['n_images']} | {frames} | {size} | "
            f"{', '.join(r['first_files'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--max-depth", type=int, default=6)
    args = ap.parse_args()
    recs = inventory(args.root, max_depth=args.max_depth)
    print(format_markdown(recs))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(recs, indent=2))
        print(f"\nJSON report written to {args.out}")


if __name__ == "__main__":
    main()

"""Model download / loading helpers (weights are never stored in the repo)."""

from __future__ import annotations

from pathlib import Path

import requests

SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}
_MIN_BYTES = {"vit_b": 300_000_000, "vit_l": 1_000_000_000, "vit_h": 2_000_000_000}


def fetch_sam_checkpoint(dest_dir: str | Path, model_type: str = "vit_b") -> Path:
    """Download the SAM checkpoint to ``dest_dir`` (skipped if already complete)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = SAM_URLS[model_type]
    dest = dest_dir / url.rsplit("/", 1)[1]
    if dest.exists() and dest.stat().st_size >= _MIN_BYTES[model_type]:
        return dest
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if tmp.stat().st_size < _MIN_BYTES[model_type]:
        tmp.unlink()
        raise RuntimeError(f"SAM download looks truncated ({url}); re-run.")
    tmp.rename(dest)
    return dest


def load_sam_predictor(checkpoint: str | Path, model_type: str = "vit_b", device: str | None = None):
    """Return a ready ``SamPredictor`` (needs ``pip install segment-anything``)."""
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as e:  # pragma: no cover
        raise ImportError("segment-anything is not installed: pip install segment-anything") from e
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    return SamPredictor(sam)

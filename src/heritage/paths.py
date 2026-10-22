"""Single source of truth for where things live.

Two roots:
  * ``drive``  - persistent. Final images, metrics, checkpoints you want to keep.
  * ``local``  - fast scratch on the Colab VM (wiped on reset). Big downloads and
                 intermediate files live here and are re-creatable.

Both can be overridden with environment variables ``HERITAGE_DRIVE`` and
``HERITAGE_LOCAL`` (used by the tests and by anyone running outside Colab).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


def repo_root() -> Path:
    """Directory that contains ``configs/`` (works from notebooks and scripts)."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "configs" / "default.yaml").exists():
            return p
    return Path.cwd()


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Paths:
    drive: Path
    local: Path

    # --- raw data (local scratch; re-downloadable) ---
    @property
    def raw(self) -> Path:
        return self.local / "raw"

    @property
    def data(self) -> Path:
        """Extracted dataset root."""
        return self.local / "data"

    # --- persistent products (Drive) ---
    @property
    def results(self) -> Path:
        return self.drive / "results"

    @property
    def checkpoints(self) -> Path:
        return self.drive / "checkpoints"

    @property
    def reports(self) -> Path:
        return self.drive / "reports"

    @property
    def figures(self) -> Path:
        return self.drive / "figures"

    def stage(self, name: str) -> Path:
        """Per-stage output folder on Drive, e.g. ``stage('lama')``."""
        return self.results / name

    def ensure(self) -> "Paths":
        for p in (self.raw, self.data, self.results, self.checkpoints, self.reports, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(cfg: dict | None = None) -> Paths:
    cfg = cfg or load_config()
    drive = os.environ.get("HERITAGE_DRIVE", cfg["workspace"]["drive_dir"])
    local = os.environ.get("HERITAGE_LOCAL", cfg["workspace"]["local_dir"])
    return Paths(drive=Path(drive), local=Path(local))

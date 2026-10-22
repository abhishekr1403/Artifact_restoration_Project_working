#!/usr/bin/env python3
"""Download and extract the Figshare dataset (Stoean et al., CC BY 4.0).

* Resumable: re-running continues a partial download (HTTP Range).
* Verifies the byte size against configs/default.yaml.
* Extracts only if not already extracted (marker file), safe to re-run.

Usage:
    python scripts/download_dataset.py            # uses configs/default.yaml paths
    python scripts/download_dataset.py --force    # re-extract
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

from heritage.paths import get_paths, load_config  # noqa: E402

URL_TMPL = "https://ndownloader.figshare.com/files/{fid}"
CHUNK = 1 << 20


def download(url: str, dest: Path, expected: int | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if expected and have == expected:
        print(f"Already downloaded: {dest} ({have/1e6:.0f} MB)")
        return
    if expected and have > expected:
        dest.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    t0 = time.time()
    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
        if r.status_code == 200 and have:  # server ignored Range -> restart
            have = 0
        elif r.status_code not in (200, 206):
            r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        done = have
        with open(dest, mode) as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if expected and (done // CHUNK) % 50 == 0:
                    pct = 100 * done / expected
                    mbs = (done - have) / 1e6 / max(time.time() - t0, 1e-6)
                    print(f"\r  {done/1e6:7.0f}/{expected/1e6:.0f} MB ({pct:4.1f}%)  {mbs:5.1f} MB/s", end="")
    print()
    got = dest.stat().st_size
    if expected and got != expected:
        raise RuntimeError(f"Size mismatch: got {got} bytes, expected {expected}. Re-run to resume.")


def extract(zip_path: Path, out_dir: Path, force: bool) -> None:
    marker = out_dir / ".extracted"
    if marker.exists() and not force:
        print(f"Already extracted: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"Corrupt member in zip: {bad}. Delete the zip and re-download.")
        z.extractall(out_dir)
    marker.write_text("ok")
    print(f"Extracted to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-extract even if marker exists")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg).ensure()
    ds = cfg["dataset"]
    zip_path = paths.raw / "3D.zip"
    download(URL_TMPL.format(fid=ds["figshare_file_id"]), zip_path, ds.get("expected_zip_bytes"))
    extract(zip_path, paths.data, args.force)
    print(f"\nDataset ready at: {paths.data}\nLicense: {ds['license']}. Please cite: {ds['citation']}")


if __name__ == "__main__":
    main()

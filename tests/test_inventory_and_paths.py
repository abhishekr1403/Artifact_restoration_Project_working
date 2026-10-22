import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heritage.inventory import format_markdown, inventory  # noqa: E402
from heritage.paths import get_paths, load_config  # noqa: E402


def _png(p: Path, size=(32, 18)):
    from PIL import Image

    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(p)


def test_inventory_reports_ranges_sizes_and_nested_folders(tmp_path):
    for n in (84, 85, 86):
        _png(tmp_path / "ground_truth" / "frames" / f"frame_{n:05d}.png")
    for n in (1, 2):
        _png(tmp_path / "gt_inpainted" / "lama" / "marginal" / f"frame_{n:05d}_mask001.png", (64, 36))
    (tmp_path / "gt_inpainted" / "readme.txt").write_text("hello")

    recs = {r["folder"]: r for r in inventory(tmp_path)}
    gt = recs["ground_truth/frames"]
    assert gt["n_images"] == 3 and (gt["frame_min"], gt["frame_max"]) == (84, 86)
    assert gt["sample_image_sizes"] == {"32x18": 3}
    assert recs["gt_inpainted/lama/marginal"]["frame_min"] == 1
    assert recs["gt_inpainted"]["extensions"] == {".txt": 1}
    md = format_markdown(list(recs.values()))
    assert "ground_truth/frames" in md and md.count("\n") >= 3
    json.dumps(list(recs.values()))  # must be JSON serialisable


def test_inventory_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        inventory(tmp_path / "nope")


def test_paths_env_override_and_ensure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERITAGE_DRIVE", str(tmp_path / "drive"))
    monkeypatch.setenv("HERITAGE_LOCAL", str(tmp_path / "local"))
    p = get_paths(load_config()).ensure()
    assert p.drive == tmp_path / "drive" and p.raw == tmp_path / "local" / "raw"
    assert p.stage("lama") == tmp_path / "drive" / "results" / "lama"
    assert p.results.is_dir() and p.data.is_dir()


def test_config_has_expected_keys():
    cfg = load_config()
    assert cfg["dataset"]["figshare_file_id"] == 43296708
    assert cfg["inpainting"]["sd_runs"] == 10
    assert cfg["colab"]["numpy_pin_if_smoke_test_fails"] == "<2"

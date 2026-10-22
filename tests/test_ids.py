import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heritage.ids import build_index, frame_number, pair_by_frame, pick  # noqa: E402


def test_frame_number_variants():
    assert frame_number("frame_00084.png") == 84
    assert frame_number("frame_00084_mask001.png") == 84
    assert frame_number("frame_00084_mean.png") == 84
    assert frame_number("frame_00084_sd_run03.png") == 84
    assert frame_number("frame-12.jpg") == 12
    assert frame_number("00084.png") == 84
    assert frame_number("img_84.jpg") == 84
    assert frame_number("notes.txt") is None


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_pairing_gt_vs_lama_vs_sd_mean(tmp_path):
    gt = tmp_path / "gt"
    lama = tmp_path / "lama"
    sd = tmp_path / "sd"
    for n in (84, 85, 86):
        _touch(gt / f"frame_{n:05d}.png")
    # LaMa has an extra frame the GT does not, and lacks frame 86
    for n in (1, 84, 85):
        _touch(lama / f"frame_{n:05d}_mask001.png")
    for n in (84, 85, 86):
        _touch(sd / f"frame_{n:05d}_mean.png")

    pairs = pair_by_frame(gt, lama)
    assert [p[0] for p in pairs] == [84, 85]
    pairs = pair_by_frame(gt, sd, prefer_b=["_mean"])
    assert [p[0] for p in pairs] == [84, 85, 86]


def test_pick_prefers_token_then_shortest(tmp_path):
    a, b = tmp_path / "frame_00001.png", tmp_path / "frame_00001_mean.png"
    assert pick([b, a]) == a
    assert pick([a, b], prefer=["_mean"]) == b
    assert pick([]) is None


def test_build_index_groups_runs(tmp_path):
    for r in range(1, 4):
        _touch(tmp_path / f"frame_00010_sd_run{r:02d}.png")
    idx = build_index(tmp_path)
    assert list(idx) == [10] and len(idx[10]) == 3

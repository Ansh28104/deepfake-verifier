from __future__ import annotations

from pathlib import Path

from scripts.train_finetune import collect_items, collect_items_from_roots


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_collect_items_generic_real_fake_layout(tmp_path) -> None:
    root = tmp_path / "faceforensics"
    _touch(root / "real" / "a.mp4")
    _touch(root / "real" / "nested" / "b.mov")
    _touch(root / "fake" / "c.mkv")
    _touch(root / "fake" / "nested" / "d.avi")
    _touch(root / "real" / "ignore.txt")

    items = collect_items(root)

    reals = [item for item in items if item.label == 0]
    fakes = [item for item in items if item.label == 1]

    assert len(reals) == 2
    assert len(fakes) == 2


def test_collect_items_celeb_df_v2_layout(tmp_path) -> None:
    root = tmp_path / "celeb_df_v2"
    _touch(root / "Celeb-real" / "r1.mp4")
    _touch(root / "YouTube-real" / "r2.mp4")
    _touch(root / "Celeb-synthesis" / "f1.mp4")

    items = collect_items(root)

    reals = [item for item in items if item.label == 0]
    fakes = [item for item in items if item.label == 1]

    assert len(reals) == 2
    assert len(fakes) == 1


def test_collect_items_from_roots_combines_sources(tmp_path) -> None:
    ff_root = tmp_path / "faceforensics"
    celeb_root = tmp_path / "celeb_df_v2"

    _touch(ff_root / "real" / "ff_real.mp4")
    _touch(ff_root / "fake" / "ff_fake.mp4")
    _touch(celeb_root / "Celeb-real" / "c_real.mp4")
    _touch(celeb_root / "Celeb-synthesis" / "c_fake.mp4")

    items = collect_items_from_roots([ff_root, celeb_root])

    assert len(items) == 4
    assert sum(1 for item in items if item.label == 0) == 2
    assert sum(1 for item in items if item.label == 1) == 2

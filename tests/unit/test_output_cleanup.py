"""Testes da limpeza de diretórios de saída órfãos."""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from backend.app.services.output_cleanup import (
    delete_orphan_outputs,
    find_orphan_outputs,
)

KNOWN = "20260101_120000_aaaaaaaa"
ORPHAN = "20260102_120000_bbbbbbbb"
RECENT = "20260103_120000_cccccccc"


def _make_output(base: Path, job_id: str, size: int = 10) -> Path:
    directory = base / job_id
    directory.mkdir()
    (directory / "classification.png").write_bytes(b"x" * size)
    return directory


def test_finds_only_generated_orphan_directories(tmp_path):
    _make_output(tmp_path, KNOWN)
    orphan = _make_output(tmp_path, ORPHAN, size=20)
    (tmp_path / "not-a-job").mkdir()
    (tmp_path / "ensemble_weights.json").write_text("{}")

    orphans = find_orphan_outputs(tmp_path, {KNOWN})

    assert [item.path for item in orphans] == [orphan]
    assert orphans[0].size_bytes == 20


def test_respects_older_than_days(tmp_path):
    old = _make_output(tmp_path, ORPHAN)
    recent = _make_output(tmp_path, RECENT)
    old_time = (datetime.now(UTC) - timedelta(days=40)).timestamp()
    os.utime(old, (old_time, old_time))

    orphans = find_orphan_outputs(
        tmp_path, set(), older_than_days=30
    )

    assert [item.path for item in orphans] == [old]
    assert recent.exists()


def test_missing_base_returns_empty(tmp_path):
    assert find_orphan_outputs(tmp_path / "nao-existe", set()) == []


def test_delete_removes_only_orphans(tmp_path):
    known = _make_output(tmp_path, KNOWN)
    orphan = _make_output(tmp_path, ORPHAN)
    stray = tmp_path / "not-a-job"
    stray.mkdir()

    orphans = find_orphan_outputs(tmp_path, {KNOWN})
    deleted = delete_orphan_outputs(tmp_path, orphans)

    assert deleted == 1
    assert not orphan.exists()
    assert known.exists()
    assert stray.exists()


def test_delete_ignores_paths_outside_base(tmp_path):
    outside = tmp_path / "fora"
    outside.mkdir()
    base = tmp_path / "outputs"
    base.mkdir()

    from backend.app.services.output_cleanup import OrphanOutput

    deleted = delete_orphan_outputs(
        base,
        [
            OrphanOutput(
                path=outside,
                size_bytes=0,
                modified_at=datetime.now(UTC),
            )
        ],
    )

    assert deleted == 0
    assert outside.exists()

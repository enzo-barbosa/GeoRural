"""Testes de health check (liveness e memória)."""

import asyncio

from backend.app.services.health import check_liveness, memory_snapshot


def test_memory_snapshot_has_expected_keys():
    snapshot = memory_snapshot()
    assert snapshot["rss_mb"] > 0
    assert snapshot["system_available_mb"] > 0


def test_liveness_does_not_depend_on_gee():
    result = asyncio.run(check_liveness())
    assert result["status"] == "ok"
    assert result["rss_mb"] > 0

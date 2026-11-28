"""Testes de retry, streaming e pré-alocação dos downloads do GEE."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from backend.app.services import classification_service as service


@pytest.fixture()
def retry_settings(monkeypatch):
    settings = SimpleNamespace(
        GEE_MAX_ATTEMPTS=3,
        GEE_RETRY_BACKOFF_SECONDS=1.5,
        DOWNLOAD_TIMEOUT_SECONDS=300,
        MAX_DOWNLOAD_BYTES=40_000_000,
    )
    monkeypatch.setattr(service, "get_settings", lambda: settings)
    return settings


class FakeResponse:
    def __init__(self, status_code=200, chunks=(), error=None):
        self.status_code = status_code
        self._chunks = list(chunks)
        self._error = error
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}", response=self
            )

    def iter_content(self, chunk_size=1024):
        yield from self._chunks
        if self._error is not None:
            raise self._error


class FakeImage:
    def __init__(self, bands):
        self._bands = bands
        self.selected = []

    def select(self, bands):
        self.selected = list(bands)
        return self

    def bandNames(self):
        return self

    def getInfo(self):
        return self._bands

    def getDownloadURL(self, params):
        return f"https://example.test/{'-'.join(self.selected)}.tif"


def _fake_get(responses):
    calls: list[dict] = []

    def get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return responses.pop(0)

    return get, calls


def test_gee_call_retries_transient_error(monkeypatch, retry_settings):
    sleeps: list[float] = []
    monkeypatch.setattr(service.time, "sleep", sleeps.append)
    calls = []

    def call():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("503 Service Unavailable")
        return "ok"

    assert service._gee_call("teste", call) == "ok"
    assert len(calls) == 3
    assert sleeps == [1.5, 3.0]


def test_gee_call_does_not_retry_permanent_error(
    monkeypatch, retry_settings
):
    sleeps: list[float] = []
    monkeypatch.setattr(service.time, "sleep", sleeps.append)
    calls = []

    def call():
        calls.append(1)
        raise ValueError("geometria inválida")

    with pytest.raises(ValueError, match="geometria inválida"):
        service._gee_call("teste", call)

    assert len(calls) == 1
    assert sleeps == []


def test_download_geotiff_retries_503(monkeypatch, retry_settings, tmp_path):
    sleeps: list[float] = []
    monkeypatch.setattr(service.time, "sleep", sleeps.append)
    fake_get, calls = _fake_get(
        [
            FakeResponse(503),
            FakeResponse(200, chunks=[b"conteudo"]),
        ]
    )
    monkeypatch.setattr(service.requests, "get", fake_get)

    path = service._download_geotiff("https://example.test/img.tif")
    try:
        assert service.Path(path).read_bytes() == b"conteudo"
    finally:
        os.unlink(path)

    assert len(calls) == 2
    assert calls[0]["stream"] is True
    assert sleeps == [1.5]


def test_download_geotiff_stops_on_404(monkeypatch, retry_settings):
    sleeps: list[float] = []
    monkeypatch.setattr(service.time, "sleep", sleeps.append)
    fake_get, calls = _fake_get([FakeResponse(404)])
    monkeypatch.setattr(service.requests, "get", fake_get)

    with pytest.raises(requests.HTTPError):
        service._download_geotiff("https://example.test/img.tif")

    assert len(calls) == 1
    assert sleeps == []


def test_download_geotiff_cleans_temp_on_stream_error(
    monkeypatch, retry_settings, tmp_path
):
    retry_settings.GEE_MAX_ATTEMPTS = 1
    target = tmp_path / "parcial.tif"

    def fake_mkstemp(suffix):
        fd = os.open(target, os.O_CREAT | os.O_WRONLY, 0o600)
        return fd, str(target)

    monkeypatch.setattr(service.tempfile, "mkstemp", fake_mkstemp)
    fake_get, _ = _fake_get(
        [
            FakeResponse(
                200,
                chunks=[b"parcial"],
                error=requests.ConnectionError("caiu no meio"),
            )
        ]
    )
    monkeypatch.setattr(service.requests, "get", fake_get)

    with pytest.raises(requests.ConnectionError):
        service._download_geotiff("https://example.test/img.tif")

    assert not target.exists()


def test_download_ee_image_preallocates_multi_band(
    monkeypatch, retry_settings
):
    monkeypatch.setattr(
        service, "bands_per_download", lambda pixels, max_bytes: 1
    )
    monkeypatch.setattr(
        service, "_download_geotiff", lambda url: f"/tmp/{url[-5:]}"
    )

    chunks = [
        (np.full((1, 2, 2), value, dtype=np.int32), {"chunk": value})
        for value in (1, 2, 3)
    ]
    monkeypatch.setattr(
        service, "read_geotiff", lambda path: chunks.pop(0)
    )
    monkeypatch.setattr(
        service.np,
        "concatenate",
        lambda *args, **kwargs: pytest.fail("não deveria concatenar"),
    )

    data, meta = service._download_ee_image(
        FakeImage(["B1", "B2", "B3"]),
        geometry=None,
        scale=10,
        estimated_pixels=1000,
    )

    assert data.shape == (3, 2, 2)
    np.testing.assert_array_equal(data[:, 0, 0], [1, 2, 3])
    assert meta == {"chunk": 1}


def test_download_ee_image_without_bands(monkeypatch, retry_settings):
    data, meta = service._download_ee_image(
        FakeImage([]),
        geometry=None,
        scale=10,
        estimated_pixels=1000,
    )

    assert data is None
    assert meta is None


def test_download_ee_image_cleans_temp_when_read_fails(
    monkeypatch, retry_settings, tmp_path
):
    target = tmp_path / "baixado.tif"
    target.write_bytes(b"tif")
    monkeypatch.setattr(
        service, "_download_geotiff", lambda url: str(target)
    )

    def fail_read(path):
        raise RuntimeError("geotiff corrompido")

    monkeypatch.setattr(service, "read_geotiff", fail_read)

    with pytest.raises(RuntimeError, match="corrompido"):
        service._download_ee_image(
            FakeImage(["B1"]),
            geometry=None,
            scale=10,
            estimated_pixels=1000,
        )

    assert not target.exists()

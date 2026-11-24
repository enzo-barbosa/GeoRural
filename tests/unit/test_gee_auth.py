"""Testes da autenticação do Google Earth Engine."""

import pytest

from src.gee import auth


def test_project_id_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.delenv("GEE_PROJECT_ID", raising=False)
    with pytest.raises(RuntimeError, match="GEE_PROJECT_ID"):
        auth._project_id()

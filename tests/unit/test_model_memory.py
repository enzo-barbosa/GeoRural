"""Testes da liberação de modelos carregados em memória."""

from backend.app.dependencies import models


def test_release_model_cache_clears_cached_models(monkeypatch, tmp_path):
    model_path = tmp_path / "brazil_rf.pkl"
    model_path.write_bytes(b"modelo-falso")

    monkeypatch.setattr(models, "get_model_path", lambda name: model_path)
    monkeypatch.setattr(
        "src.classifiers.random_forest.load_model",
        lambda path: object(),
    )
    trimmed: list[bool] = []
    monkeypatch.setattr(
        models, "_trim_process_memory", lambda: trimmed.append(True)
    )

    models.get_cached_model.cache_clear()
    assert models.get_cached_model("brazil_rf") is not None
    assert models.get_cached_model.cache_info().currsize == 1

    models.release_model_cache()

    assert models.get_cached_model.cache_info().currsize == 0
    assert trimmed == [True]


def test_trim_process_memory_never_raises(monkeypatch):
    monkeypatch.setattr(models.sys, "platform", "win32")

    models._trim_process_memory()

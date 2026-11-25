#!/usr/bin/env python3
"""Benchmark de memória e tempo da inferência (RF, SVM, U-Net ou ensemble).

Usa uma imagem sintética e os modelos locais em ``models/``, sem chamar o GEE.
Serve para comparar o pico de memória antes/depois de otimizações.

Exemplos:
    uv run python scripts/benchmark_inference.py --classifier random_forest
    uv run python scripts/benchmark_inference.py --classifier unet_cnn --size 512
    uv run python scripts/benchmark_inference.py --classifier auto --json-out /tmp/bench.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GEE_PROJECT_ID", "benchmark")


class MemorySampler(threading.Thread):
    """Amostra o RSS do processo em uma thread para capturar o pico."""

    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mb = 0.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            rss = process.memory_info().rss / (1024 * 1024)
            self.peak_mb = max(self.peak_mb, rss)
            self._stop_event.wait(self.interval)

    def stop(self) -> float:
        self._stop_event.set()
        self.join(timeout=2)
        return round(self.peak_mb, 1)


def _hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier",
        choices=["random_forest", "svm", "unet_cnn", "auto"],
        default="unet_cnn",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Lado da imagem sintética (size x size x 15 canais)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    # Importa depois de definir GEE_PROJECT_ID (Settings exige o campo).
    from backend.app.services import classification_service as service

    process = psutil.Process(os.getpid())
    rss_start = process.memory_info().rss / (1024 * 1024)

    rng = np.random.default_rng(args.seed)
    image = rng.random((args.size, args.size, 15), dtype=np.float32)

    sampler = MemorySampler()
    sampler.start()
    started = time.perf_counter()
    try:
        pred_map, probabilities, used, weights = service._predict_probability_maps(
            image,
            satellite="sentinel2",
            coverage="brazil",
            classifier_name=args.classifier,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Falhou: {exc}")
        return 1
    elapsed = time.perf_counter() - started
    peak_mb = sampler.stop()
    rss_end = process.memory_info().rss / (1024 * 1024)

    result = {
        "classifier": args.classifier,
        "size": args.size,
        "pixels": args.size * args.size,
        "used": used,
        "weights": weights,
        "elapsed_s": round(elapsed, 3),
        "rss_start_mb": round(rss_start, 1),
        "rss_end_mb": round(rss_end, 1),
        "peak_rss_mb": peak_mb,
        "pred_hash": _hash_array(pred_map),
        "proba_hash": _hash_array(probabilities),
        "pred_shape": list(pred_map.shape),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

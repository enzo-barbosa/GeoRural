#!/usr/bin/env python3
"""Baixa os modelos treinados a partir de uma GitHub Release.

O manifesto (``models.lock.json``) lista os arquivos, tamanhos e hashes.
O script é idempotente: arquivos já baixados e válidos são ignorados. Downloads
interrompidos são retomados usando ``.part`` + HTTP Range.

Uso:
    python scripts/download_models.py
    python scripts/download_models.py --only brazil_unet.pth
    python scripts/download_models.py --force --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CHUNK_SIZE = 1024 * 1024
USER_AGENT = "GeoRural-model-downloader/1.0"


def human_size(num_bytes: float) -> str:
    """Formata bytes em MB/GB."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def sha256_of(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, expected_size: int | None, expected_sha: str | None) -> bool:
    """Verifica tamanho e hash (quando disponíveis)."""
    if not path.exists():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return expected_sha is None or sha256_of(path) == expected_sha


def download(url: str, destination: Path, expected_size: int | None) -> None:
    """Baixa com retomada (HTTP Range) para um arquivo ``.part``."""
    part = destination.with_name(destination.name + ".part")
    offset = part.stat().st_size if part.exists() else 0

    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and part.exists():
            part.replace(destination)
            return
        raise

    status = getattr(response, "status", 200)
    mode = "ab" if offset and status == 206 else "wb"
    if mode == "wb":
        offset = 0

    downloaded = offset
    total = None
    content_length = response.headers.get("Content-Length")
    if content_length:
        total = int(content_length) + offset
    elif expected_size:
        total = expected_size

    print(
        f"  baixando {destination.name} ({human_size(offset)} de "
        f"{human_size(total) if total else '?'})"
    )

    with part.open(mode) as file:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            file.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                print(f"\r  {percent:5.1f}%  {human_size(downloaded)}", end="")
    print()

    part.replace(destination)


def load_manifest(path: Path) -> dict:
    """Carrega e valida o formato do manifesto."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(
            f"Manifesto não encontrado: {path}. "
            "Rode o script na raiz do repositório."
        ) from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Manifesto inválido ({path}): {exc}") from exc

    if not manifest.get("files"):
        raise SystemExit(f"Manifesto sem arquivos: {path}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models.lock.json"),
        help="Caminho do manifesto (default: models.lock.json)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("models"),
        help="Diretório de destino (default: models)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Baixa apenas o(s) arquivo(s) informado(s); pode repetir",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Baixa novamente mesmo se o arquivo já estiver válido",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Apenas valida os arquivos existentes; não baixa",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    base_url = manifest.get("base_url", "").rstrip("/")
    destination_dir: Path = args.dest
    destination_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.only or [])
    files = [
        item for item in manifest["files"] if not selected or item["name"] in selected
    ]
    if not files:
        raise SystemExit("Nenhum arquivo corresponde a --only.")

    failures: list[str] = []
    for item in files:
        name = item["name"]
        expected_size = item.get("size")
        expected_sha = item.get("sha256")
        path = destination_dir / name

        if not args.force and is_valid(path, expected_size, expected_sha):
            print(f"[ok] {name} já está válido")
            continue
        if args.check:
            failures.append(name)
            print(f"[falhou] {name} ausente ou inválido")
            continue

        url = item.get("url") or f"{base_url}/{name}"
        print(f"[download] {name}")
        try:
            download(url, path, expected_size)
        except urllib.error.HTTPError as exc:
            failures.append(name)
            print(f"  erro HTTP {exc.code}: {url}", file=sys.stderr)
            continue
        except urllib.error.URLError as exc:
            failures.append(name)
            print(f"  erro de rede: {exc.reason}", file=sys.stderr)
            continue

        if expected_size is not None and path.stat().st_size != expected_size:
            failures.append(name)
            print(
                f"  tamanho inválido: {path.stat().st_size} != {expected_size}",
                file=sys.stderr,
            )
            continue
        if expected_sha and sha256_of(path) != expected_sha:
            failures.append(name)
            print("  SHA-256 inválido", file=sys.stderr)
            continue
        print(f"[ok] {name}")

    if failures:
        print(
            "\nFalhas: " + ", ".join(failures) + "\n"
            "Confira se a release de modelos já foi publicada.",
            file=sys.stderr,
        )
        return 1

    print("\nModelos prontos em:", destination_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

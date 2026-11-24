#!/usr/bin/env python3
"""Publica os modelos treinados em uma GitHub Release.

O script valida tamanho e SHA-256, cria a release (se necessário) e envia os
assets com `gh release upload`. Ele não altera o manifesto: use
``--update-manifest`` para gravar os hashes calculados em ``models.lock.json``
antes do upload.

Exemplos:
    uv run python scripts/publish_models.py --dry-run
    uv run python scripts/publish_models.py --upload
    uv run python scripts/publish_models.py --upload --update-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_size: int | None, expected_sha: str | None) -> None:
    if not path.exists():
        raise SystemExit(f"Arquivo ausente: {path}")
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise SystemExit(f"Tamanho divergente em {path.name}: {size} != {expected_size}")
    actual_sha = sha256_of(path)
    if expected_sha and actual_sha != expected_sha:
        raise SystemExit(
            f"SHA-256 divergente em {path.name}:\n"
            f"  esperado: {expected_sha}\n"
            f"  atual:    {actual_sha}"
        )
    print(f"[ok] {path.name} ({size / (1024**2):.1f} MB)")


def run(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def release_exists(tag: str) -> bool:
    result = subprocess.run(
        ["gh", "release", "view", tag],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "models.lock.json",
        help="Manifesto dos modelos (default: models.lock.json)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Diretório local com os modelos (default: models/)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Envia de fato para o GitHub (sem isso, apenas simula)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas simula (padrão quando --upload não é informado)",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Recalcula e grava tamanho/SHA-256 no manifesto",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Não valida os arquivos antes do upload",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tag = manifest["release"]
    base_url = manifest.get("base_url", "")
    if tag not in base_url:
        raise SystemExit(
            f"O base_url do manifesto não contém a tag {tag!r}: {base_url}"
        )

    files: list[Path] = []
    for item in manifest["files"]:
        path = args.models_dir / item["name"]
        if not args.skip_verify:
            verify_file(path, item.get("size"), item.get("sha256"))
        files.append(path)

    if args.update_manifest:
        for item, path in zip(manifest["files"], files, strict=True):
            item["size"] = path.stat().st_size
            item["sha256"] = sha256_of(path)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[ok] manifesto atualizado: {args.manifest}")

    if shutil.which("gh") is None:
        raise SystemExit(
            "GitHub CLI (gh) não encontrado. Instale em https://cli.github.com/ "
            "ou faça upload manual pela interface do GitHub."
        )
    subprocess.run(["gh", "auth", "status"], cwd=ROOT, check=True)

    if not release_exists(tag):
        if args.upload:
            run(
                [
                    "gh",
                    "release",
                    "create",
                    tag,
                    "--title",
                    f"Modelos {tag}",
                    "--notes",
                    (
                        "Modelos treinados (Random Forest, SVM, U-Net e cultivo) "
                        "usados pelo GeoRural."
                    ),
                ]
            )
        else:
            print(
                f"[dry-run] criaria a release {tag} (gh release create {tag} ...)"
            )
    else:
        print(f"[ok] release {tag} já existe")

    if args.upload:
        run(["gh", "release", "upload", tag, *map(str, files), "--clobber"])
        print(f"\nModelos publicados em {base_url}")
    else:
        print(
            "[dry-run] upload não executado. Rode com --upload para publicar:\n"
            f"  gh release upload {tag} " + " ".join(str(p) for p in files)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

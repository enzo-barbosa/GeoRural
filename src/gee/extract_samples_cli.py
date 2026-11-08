"""Extração de amostras multi-região/multi-ano (Brasil e global).

Exemplos:
    python src/gee/extract_samples_cli.py --domain brazil --regions all \
        --years 2019-2023 --label-source mapbiomas
    python src/gee/extract_samples_cli.py --domain global --regions all \
        --years 2021 --label-source worldcover
    python src/gee/extract_samples_cli.py --task crop --domain brazil \
        --regions all --years 2019-2023

Cada região/ano gera um CSV parcial e uma entrada em
``data/manifests/{task}_{domain}.json``. Use ``--resume`` para pular o que já
foi concluído.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from src.gee.auth import initialize_gee
from src.gee.regions import BRAZIL_YEARS, GLOBAL_YEAR, resolve_regions
from src.gee.sampling import (
    extract_crop_region_samples,
    extract_region_samples,
    load_manifest,
    merge_csvs,
    update_manifest,
)


def parse_years(value: str | None, domain: str) -> list[int]:
    """Interpreta '2019-2023' ou '2021,2023'."""
    if not value:
        return BRAZIL_YEARS if domain == "brazil" else [GLOBAL_YEAR]
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


def run(args) -> None:
    initialize_gee()

    if args.task == "crop" and args.domain != "brazil":
        raise ValueError("O tipo de cultivo é Brasil-only (use --domain brazil).")

    regions = resolve_regions(args.domain, args.regions or ["all"])
    years = parse_years(args.years, args.domain)

    label_source = args.label_source
    if label_source is None:
        label_source = "mapbiomas" if args.domain == "brazil" else "worldcover"

    default_out = (
        "data/samples/crop" if args.task == "crop"
        else os.path.join("data/samples", args.domain)
    )
    out_dir = args.out_dir or default_out
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(
        "data/manifests", f"{args.task}_{args.domain}.json"
    )
    manifest = (
        load_manifest(manifest_path)
        if args.resume and not args.force
        else {}
    )

    completed_paths: list[str] = []

    for region in regions:
        for year in years:
            key = f"{args.domain}:{region.name}:{year}"
            partial_path = os.path.join(out_dir, f"{region.name}_{year}.csv")

            if (
                args.resume
                and not args.force
                and manifest.get(key, {}).get("status") == "completed"
                and os.path.exists(partial_path)
            ):
                print(f"[skip] {region.name} {year} (já concluído)")
                completed_paths.append(partial_path)
                continue

            print(f"\n[{args.task}] {region.name} ({region.biome}) — {year}")
            try:
                if args.task == "crop":
                    df, num_images = extract_crop_region_samples(
                        region,
                        year,
                        satellite=args.satellite,
                        max_cloud_pct=args.max_cloud,
                        scale=args.scale,
                        seed=args.seed,
                        samples_per_crop=args.samples_per_crop,
                        tile_scale=args.tile_scale,
                    )
                    id_col = "crop_id"
                else:
                    df, num_images = extract_region_samples(
                        region,
                        year,
                        label_source=label_source,
                        satellite=args.satellite,
                        max_cloud_pct=args.max_cloud,
                        scale=args.scale,
                        seed=args.seed,
                        samples_per_class=args.samples_per_class,
                        tile_scale=args.tile_scale,
                    )
                    id_col = "class_id"

                if df.empty:
                    print(f"  sem amostras (imagens={num_images})")
                    update_manifest(
                        manifest_path,
                        {
                            "domain": args.domain,
                            "region_id": region.name,
                            "biome": region.biome,
                            "year": year,
                            "status": "no_data",
                            "rows": 0,
                            "num_images": num_images,
                            "task": args.task,
                        },
                    )
                    continue

                if id_col not in df.columns:
                    raise RuntimeError(
                        f"Coluna {id_col} ausente. Colunas: {df.columns.tolist()}"
                    )

                df.to_csv(partial_path, index=False)
                completed_paths.append(partial_path)

                dist = df[id_col].value_counts().sort_index()
                print(f"  {len(df)} amostras | imagens={num_images}")
                print("  distribuição:", dict(dist))

                update_manifest(
                    manifest_path,
                    {
                        "domain": args.domain,
                        "region_id": region.name,
                        "biome": region.biome,
                        "year": year,
                        "status": "completed",
                        "rows": len(df),
                        "num_images": int(num_images),
                        "path": partial_path,
                        "task": args.task,
                        "label_source": label_source,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERRO: {exc}")
                update_manifest(
                    manifest_path,
                    {
                        "domain": args.domain,
                        "region_id": region.name,
                        "biome": region.biome,
                        "year": year,
                        "status": "error",
                        "error": str(exc),
                        "task": args.task,
                    },
                )

    if args.combined_out:
        combined_path = args.combined_out
    elif args.task == "crop":
        combined_path = os.path.join("data/samples", "crop_samples.csv")
    else:
        combined_path = os.path.join(
            "data/samples", f"{args.domain}_samples.csv"
        )

    if completed_paths:
        merged = merge_csvs(completed_paths, combined_path)
        print(f"\nCSV combinado: {combined_path} ({len(merged)} amostras)")
    else:
        print("\nNenhum CSV parcial para combinar.")

    # Mantém o caminho legado usado pelos notebooks/testes.
    legacy = os.path.join("data/samples", "training_samples.csv")
    if (
        not args.no_legacy
        and args.domain == "brazil"
        and args.task == "landcover"
        and os.path.exists(combined_path)
    ):
        pd.read_csv(combined_path).to_csv(legacy, index=False)
        print(f"Atualizado também: {legacy}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai amostras multi-região para treino"
    )
    parser.add_argument("--task", default="landcover", choices=["landcover", "crop"])
    parser.add_argument("--domain", default="brazil", choices=["brazil", "global"])
    parser.add_argument("--regions", nargs="+", default=["all"])
    parser.add_argument("--years", default=None, help="'2019-2023' ou '2021,2023'")
    parser.add_argument(
        "--label-source", default=None, choices=["mapbiomas", "worldcover"]
    )
    parser.add_argument("--satellite", default="sentinel2", choices=["sentinel2", "landsat8"])
    parser.add_argument("--max-cloud", type=int, default=80)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--samples-per-crop", type=int, default=500)
    parser.add_argument("--tile-scale", type=int, default=4)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--combined-out", default=None)
    parser.add_argument("--no-legacy", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refaz regiões/anos já concluídos (ignora o manifesto).",
    )
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()

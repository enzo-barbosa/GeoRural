"""Valida e repara os tiles da U-Net sem acessar o GEE.

Uso:
    python scripts/validate_unet_dataset.py --dir data/processed/unet_full
    python scripts/validate_unet_dataset.py --dir data/processed/unet_full --fix --move-invalid
"""

import argparse
import glob
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.config import NUM_CLASSES
from src.processing.dataset_validation import (
    analyze_tile,
    patch_coverage,
    repair_tile,
    write_report,
)


def main():
    parser = argparse.ArgumentParser(description="Valida/repara tiles da U-Net")
    parser.add_argument("--dir", default="data/processed/unet_full")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Sanitiza NDVI/NaN e regrava os arrays.",
    )
    parser.add_argument(
        "--move-invalid",
        action="store_true",
        help="Move tiles inválidos para <dir>/invalid/.",
    )
    parser.add_argument("--min-valid-fraction", type=float, default=0.01)
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Tamanho do patch para medir cobertura por classe.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Stride da grade para medir cobertura por classe.",
    )
    args = parser.parse_args()

    directory = os.path.join(PROJECT_ROOT, args.dir)
    invalid_dir = os.path.join(directory, "invalid")
    reports = {}
    aggregate = {"total_patches": 0, "counts": {i: 0 for i in range(NUM_CLASSES)}}

    for image_path in sorted(glob.glob(os.path.join(directory, "*_image.npy"))):
        name = os.path.basename(image_path).replace("_image.npy", "")
        labels_path = os.path.join(directory, f"{name}_labels.npy")
        if not os.path.exists(labels_path):
            print(f"[sem labels] {name}")
            continue

        image = np.load(image_path)
        labels = np.load(labels_path)

        if args.fix:
            image, labels, report = repair_tile(image, labels)
            np.save(image_path, image)
            np.save(labels_path, labels)
        else:
            report = analyze_tile(
                image,
                labels,
                min_valid_fraction=args.min_valid_fraction,
                min_classes=args.min_classes,
            )

        reports[name] = report
        coverage = patch_coverage(
            labels, patch_size=args.patch_size, stride=args.stride
        )
        report["patch_coverage"] = coverage
        if report["valid"]:
            aggregate["total_patches"] += coverage["total_patches"]
            for class_id, count in coverage["patches_with_class"].items():
                aggregate["counts"][class_id] += count

        status = "OK" if report["valid"] else "INVÁLIDO"
        print(
            f"{status:<8} {name:<28} "
            f"fração={report['valid_fraction']:.3f} "
            f"classes={report['classes']}"
        )

        if not report["valid"] and args.move_invalid:
            os.makedirs(invalid_dir, exist_ok=True)
            for path in (
                image_path,
                labels_path,
                os.path.join(directory, f"{name}_meta.json"),
            ):
                if os.path.exists(path):
                    shutil.move(path, os.path.join(invalid_dir, os.path.basename(path)))

    report_path = os.path.join(directory, "dataset_report.json")
    total_patches = aggregate["total_patches"]
    coverage_summary = {
        "patch_size": args.patch_size,
        "stride": args.stride,
        "total_patches": total_patches,
        "fraction": {
            class_id: (
                round(count / total_patches, 4) if total_patches else 0.0
            )
            for class_id, count in aggregate["counts"].items()
        },
        "patches_with_class": aggregate["counts"],
    }
    write_report(
        {
            "valid_tiles": [
                name for name, report in reports.items() if report["valid"]
            ],
            "patch_coverage": coverage_summary,
            "tiles": reports,
        },
        report_path,
    )
    print(f"\nRelatório: {report_path}")

    print("\nCobertura de patches por classe (tiles válidos):")
    for class_id in range(NUM_CLASSES):
        count = coverage_summary["patches_with_class"][class_id]
        fraction = coverage_summary["fraction"][class_id]
        print(
            f"  classe {class_id}: {count:>6}/{total_patches} "
            f"patches ({100 * fraction:.2f}%)"
        )

    valid = [name for name, report in reports.items() if report["valid"]]
    print(f"Tiles válidos: {len(valid)}/{len(reports)}")
    for tile in (
        "unet_piracicaba_t0",
        "unet_caatinga_t2",
        "unet_rio_verde_t2",
        "unet_cerrado_t0",
        "unet_solo_quadrilatero",
        "unet_solo_paracatu",
    ):
        if tile not in valid:
            print(f"ATENÇÃO: tile de validação multi-tile ausente: {tile}")


if __name__ == "__main__":
    main()

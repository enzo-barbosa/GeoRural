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

from src.processing.dataset_validation import (
    analyze_tile,
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
    args = parser.parse_args()

    directory = os.path.join(PROJECT_ROOT, args.dir)
    invalid_dir = os.path.join(directory, "invalid")
    reports = {}

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
    write_report(
        {
            "valid_tiles": [
                name for name, report in reports.items() if report["valid"]
            ],
            "tiles": reports,
        },
        report_path,
    )
    print(f"\nRelatório: {report_path}")

    valid = [name for name, report in reports.items() if report["valid"]]
    print(f"Tiles válidos: {len(valid)}/{len(reports)}")
    if "unet_piracicaba_t0" not in valid:
        print("ATENÇÃO: o tile de validação unet_piracicaba_t0 não está válido.")


if __name__ == "__main__":
    main()

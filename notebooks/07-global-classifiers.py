"""Atalho para treinar os classificadores da base global.

Equivalente a:
    python notebooks/03-classifiers-test.py global
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.training import train_domain


def main():
    csv_path = os.path.join(PROJECT_ROOT, "data/samples/global_samples.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "data/samples/global_samples.csv não encontrado. Rode:\n"
            "  python src/gee/extract_samples_cli.py --domain global "
            "--regions all --years 2021 --label-source worldcover"
        )

    print("=" * 60)
    print("TREINO — GLOBAL (ESA WorldCover 2021)")
    print("=" * 60)
    train_domain(csv_path, prefix="global")


if __name__ == "__main__":
    main()

"""Atalho para treinar os classificadores da base global."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.training import train_domain


def main():
    parser = argparse.ArgumentParser(description="Treina RF/SVM globais")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--svm-max", type=int, default=20_000)
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(
        PROJECT_ROOT, "data/samples/global_samples.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} não encontrado. Rode a extração global antes."
        )

    train_domain(
        csv_path,
        prefix="global",
        svm_max_samples=args.svm_max,
        models_dir=args.models_dir or os.path.join(PROJECT_ROOT, "models"),
        output_dir=args.output_dir or os.path.join(PROJECT_ROOT, "outputs"),
    )


if __name__ == "__main__":
    main()

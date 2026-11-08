"""Treino e avaliação dos classificadores clássicos por domínio.

Uso:
    python notebooks/03-classifiers-test.py
    python notebooks/03-classifiers-test.py global
    python notebooks/03-classifiers-test.py brazil \
        --csv data/samples/full/brazil_samples.csv \
        --models-dir models/full --output-dir outputs/full
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.training import train_domain


def main():
    parser = argparse.ArgumentParser(description="Treina RF/SVM por domínio")
    parser.add_argument(
        "domain", nargs="?", default="brazil", choices=["brazil", "global", "sp"]
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--svm-max", type=int, default=20_000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--filter-regions", nargs="+", default=None)
    parser.add_argument(
        "--group-strategy",
        default="auto",
        choices=["auto", "block", "region", "biome"],
    )
    args = parser.parse_args()

    domain = args.domain
    prefix = args.prefix or domain
    models_dir = args.models_dir or os.path.join(PROJECT_ROOT, "models")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs")

    candidates = [os.path.join(PROJECT_ROOT, f"data/samples/{prefix}_samples.csv")]
    if args.csv:
        candidates.insert(0, os.path.join(PROJECT_ROOT, args.csv))
    if prefix == "brazil":
        candidates.append(
            os.path.join(PROJECT_ROOT, "data/samples/training_samples.csv")
        )

    csv_path = next((path for path in candidates if os.path.exists(path)), None)
    if csv_path is None:
        raise FileNotFoundError(
            "CSV de amostras não encontrado. Rode a extração antes:\n"
            f"  python src/gee/extract_samples_cli.py --domain {domain} --regions all"
        )

    print("=" * 60)
    print(f"TREINO — {prefix.upper()}")
    print("=" * 60)
    print(f"CSV: {csv_path}")
    print(f"Modelos: {models_dir}")
    print(f"Saídas: {output_dir}")

    train_domain(
        csv_path,
        prefix=prefix,
        n_splits=args.n_splits,
        svm_max_samples=args.svm_max,
        models_dir=models_dir,
        output_dir=output_dir,
        filter_regions=args.filter_regions,
        group_strategy=args.group_strategy,
    )


if __name__ == "__main__":
    main()

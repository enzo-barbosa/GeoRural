"""Treino e avaliação dos classificadores clássicos por domínio.

Uso:
    python notebooks/03-classifiers-test.py            # Brasil
    python notebooks/03-classifiers-test.py global     # base global
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.training import train_domain


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "brazil"
    prefix = "global" if domain == "global" else "brazil"

    candidates = [
        os.path.join(PROJECT_ROOT, f"data/samples/{prefix}_samples.csv")
    ]
    if prefix == "brazil":
        candidates.append(
            os.path.join(PROJECT_ROOT, "data/samples/training_samples.csv")
        )
    csv_path = next((path for path in candidates if os.path.exists(path)), None)
    if csv_path is None:
        raise FileNotFoundError(
            "CSV de amostras não encontrado. Rode a extração antes:\n"
            "  python src/gee/extract_samples_cli.py --domain "
            f"{domain} --regions all"
        )

    print("=" * 60)
    print(f"TREINO — {prefix.upper()}")
    print("=" * 60)
    print(f"CSV: {csv_path}")

    train_domain(csv_path, prefix=prefix)


if __name__ == "__main__":
    main()

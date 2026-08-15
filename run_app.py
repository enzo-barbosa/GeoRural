"""Entrypoint para rodar a aplicação Streamlit com o path correto."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.main import main

if __name__ == "__main__":
    main()

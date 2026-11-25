"""Configuração compartilhada dos testes.

O CI não tem `.env`, mas o backend exige `GEE_PROJECT_ID` para criar Settings.
Definimos um valor fictício apenas para testes; nenhum teste chama o GEE real
(sem a marca `gee`).
"""

import os

os.environ.setdefault("GEE_PROJECT_ID", "test-project")
# Os testes não devem tocar no Postgres real durante o lifespan do app.
os.environ["RECOVER_STALE_JOBS"] = "false"

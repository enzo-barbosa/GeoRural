# Manutenção

## Limpeza de arquivos órfãos

Cada classificação cria uma pasta em `outputs/<job_id>`. Quando a análise é
excluída pelo endpoint `DELETE /api/analyses/{job_id}`, os arquivos são
removidos junto.

Ainda podem sobrar pastas de:

- análises geradas antes da persistência no banco;
- jobs interrompidos antes de gravar a linha;
- remoções manuais no banco;
- falhas ao apagar arquivos.

O script compara os diretórios de `outputs/` com os `id` da tabela `analyses`
e lista os que não têm análise correspondente.

```bash
# só mostra o que seria removido (padrão)
uv run python scripts/cleanup_outputs.py --dry-run

# remove de verdade
uv run python scripts/cleanup_outputs.py --yes

# considera apenas órfãos com mais de 30 dias
uv run python scripts/cleanup_outputs.py --yes --older-than-days 30

# outro diretório de saída
uv run python scripts/cleanup_outputs.py --dry-run --outputs-dir /caminho/outputs
```

Regras de segurança:

- só entram diretórios com o formato `AAAAMMDD_HHMMSS_xxxxxxxx`;
- arquivos soltos (`ensemble_weights.json`, `metrics_*.json`) são ignorados;
- diretórios de análises existentes nunca são tocados;
- sem `--yes`, nada é removido;
- `--older-than-days` evita apagar algo recém-criado.

## Backup do banco

O Postgres roda no volume `pgdata` do Compose. Para um dump simples:

```bash
docker compose exec -T db pg_dump -U georural georural > backup.sql
```

Restauração:

```bash
docker compose exec -T db psql -U georural georural < backup.sql
```

Ajuste usuário/banco se `POSTGRES_USER`/`POSTGRES_DB` forem diferentes no
`.env`.

## Modelos

Os modelos ficam em `models/` (volume), não na imagem Docker. Para conferir o
que está disponível:

```bash
uv run python - <<'PY'
from backend.app.dependencies.models import get_models_status

for name, exists in get_models_status().items():
    print(f"{name}: {'ok' if exists else 'faltando'}")
PY
```

Detalhes de download e versões em [`MODELS.md`](MODELS.md).

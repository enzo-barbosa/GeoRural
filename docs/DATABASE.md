# Banco de dados

O GeoRural usa PostgreSQL (container `db` do Compose) através do SQLAlchemy 2.0
e do Alembic. Tipos JSON são genéricos para permitir testes em SQLite.

## Tabelas

```text
users
  id, email (único), name, password_hash, is_active, created_at

areas
  id, user_id -> users.id, name, description, geojson (JSON),
  area_ha, created_at, updated_at

analyses
  id (job_id), user_id -> users.id, area_id -> areas.id,
  name, geojson (JSON), area_ha,
  satellite, classifier, coverage, label_source, start_date, end_date,
  max_cloud, include_topography, include_crop, postprocess,
  status, stage, progress, error, num_images, confidence_mean,
  models_used (JSON), weights (JSON), result (JSON),
  created_at, updated_at
```

- `password_hash` guarda o hash Argon2 gerado no cadastro.
- `user_id`/`area_id` usam `ON DELETE SET NULL` para preservar o histórico.
- `analyses.result` guarda o payload completo da classificação (stats, camadas,
  medições, topografia etc.).
- `analyses.name`, `geojson` e `area_ha` guardam um snapshot da execução: a
  análise continua reabrível mesmo se a área salva for excluída.

## Migrações

```bash
# com o banco do Compose rodando
uv run python scripts/dev.py migrate

# alternativa dentro do container
docker compose exec backend alembic upgrade head

# criar uma nova migração (após alterar os modelos)
uv run alembic revision --autogenerate -m "descricao"
uv run alembic upgrade head
```

O CI roda `alembic upgrade head` contra um Postgres real antes do pytest.

## Testes

Os testes unitários usam SQLite em memória:

```bash
uv run pytest tests/unit/test_db_models.py -q
```

## Configuração

`DATABASE_URL` fica no `.env`:

```env
# host (backend fora do Docker)
DATABASE_URL=postgresql+psycopg://georural:troque-esta-senha@localhost:5434/georural
```

Dentro do Docker o Compose substitui o host por `db:5432`.

# API

Base: `http://localhost:8000/api`. Autenticação por cookie httpOnly
(`georural_access`) criado em `/api/auth/register` e `/api/auth/login`.
Detalhes em [`AUTH.md`](AUTH.md).

## Autenticação

| Método | Rota | Descrição |
|---|---|---|
| POST | `/api/auth/register` | cria conta e inicia sessão |
| POST | `/api/auth/login` | inicia sessão |
| POST | `/api/auth/logout` | encerra sessão |
| GET | `/api/auth/me` | usuário autenticado |

## Áreas

Todas as rotas exigem sessão e são isoladas por usuário.

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/areas` | lista áreas salvas |
| POST | `/api/areas` | salva uma área |
| GET | `/api/areas/{id}` | detalhe |
| PATCH | `/api/areas/{id}` | renomeia/edita descrição |
| DELETE | `/api/areas/{id}` | remove (histórico preservado) |

```json
{
  "name": "Fazenda Boa Vista",
  "description": "Talhão norte",
  "polygon": {
    "type": "Polygon",
    "coordinates": [[[-50.3, -20.25], [-50.2, -20.25], [-50.2, -20.3], [-50.3, -20.3], [-50.3, -20.25]]]
  }
}
```

`area_ha` é calculado no backend.

## Análises

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/analyses?limit=20&offset=0` | histórico do usuário |
| GET | `/api/analyses/{job_id}` | detalhe com `result` completo |

Cada chamada a `POST /api/classify` cria automaticamente um registro em
`analyses`, com status `running` e depois `completed` ou `failed`.

## Classificação

| Método | Rota | Descrição |
|---|---|---|
| POST | `/api/classify` | classifica um polígono |
| GET | `/api/classify/models` | disponibilidade dos modelos |
| POST | `/api/classify/coverage` | cobertura Brasil/global |

`POST /api/classify` aceita, além dos parâmetros de satélite/período:

```json
{
  "area_id": 1,
  "name": "Safra 2023"
}
```

`area_id` é opcional e precisa pertencer ao usuário. O endpoint continua
síncrono; o job assíncrono com progresso real virá no próximo PR.

## Imagens

`GET /outputs/{job_id}/{filename}.png` serve os PNGs da classificação. Exige
sessão e só responde para o dono da análise; qualquer outro acesso retorna
404 (ou 401 sem sessão).

## Públicas

`GET /api/health`, `GET /api/health/live` e `GET /api/geocode` não exigem
autenticação.

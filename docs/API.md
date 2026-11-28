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
| GET | `/api/analyses/{job_id}/report.pdf` | relatório PDF da análise |
| POST | `/api/analyses/{job_id}/retry` | reprocessa com os mesmos parâmetros |

Cada chamada a `POST /api/classify` cria automaticamente um registro em
`analyses` com status `queued`. O worker muda para `running`, atualiza
`stage`/`progress` e termina em `completed` ou `failed`.

`GET /api/analyses/{job_id}` é o endpoint de polling: consulte a cada ~2 s
enquanto `status` for `queued` ou `running`.

`POST /api/analyses/{job_id}/retry` cria uma nova análise (novo `job_id`) com
o polígono e os parâmetros da original, sem alterar a antiga. Retorna 409 se a
original ainda estiver em andamento ou não tiver polígono salvo.

### Relatório PDF

`GET /api/analyses/{job_id}/report.pdf` gera o relatório sob demanda e devolve
`application/pdf` com `Content-Disposition: attachment`. Inclui resumo,
medidas, mapa de classificação, distribuição das classes, imagens da área,
topografia e cultivo quando disponíveis.

- `404` para análise inexistente ou de outro usuário;
- `409` se a análise ainda não estiver `completed` ou não tiver resultado;
- imagens ausentes são omitidas sem quebrar o PDF.

## Classificação

| Método | Rota | Descrição |
|---|---|---|
| POST | `/api/classify` | enfileira um polígono (202) |
| GET | `/api/classify/models` | disponibilidade dos modelos |
| POST | `/api/classify/coverage` | cobertura Brasil/global |

`POST /api/classify` aceita, além dos parâmetros de satélite/período:

```json
{
  "area_id": 1,
  "name": "Safra 2023"
}
```

`area_id` é opcional e precisa pertencer ao usuário. A resposta é imediata:

```json
{
  "job_id": "20231125_120000_ab12cd34",
  "status": "pending"
}
```

O processamento roda em segundo plano, uma análise por vez. Acompanhe pelo
detalhe da análise; detalhes da fila em [`JOBS.md`](JOBS.md).

## Imagens

`GET /outputs/{job_id}/{filename}.png` serve os PNGs da classificação. Exige
sessão e só responde para o dono da análise; qualquer outro acesso retorna
404 (ou 401 sem sessão).

## Públicas

`GET /api/health`, `GET /api/health/live` e `GET /api/geocode` não exigem
autenticação.

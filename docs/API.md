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

O polígono é validado antes de salvar: precisa ter pelo menos 3 pontos
distintos, anel externo fechado (o backend fecha automaticamente), coordenadas
dentro dos limites geográficos e geometria válida, sem auto-interseção.
Polígonos inválidos retornam 422 com uma mensagem explicando o problema.

## Análises

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/analyses?limit=20&offset=0` | histórico do usuário |
| GET | `/api/analyses/{job_id}` | detalhe com `result` completo |
| GET | `/api/analyses/{job_id}/report.pdf` | relatório PDF da análise |
| POST | `/api/analyses/{job_id}/retry` | reprocessa com os mesmos parâmetros |
| DELETE | `/api/analyses/{job_id}` | remove a análise e seus arquivos |

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

### Excluir análise

`DELETE /api/analyses/{job_id}` remove a análise do histórico e a pasta
`outputs/<job_id>`.

- `completed`/`failed`: responde 204 e apaga linha + arquivos;
- `queued`: tenta cancelar o job ainda na fila; se cancelar, responde 204;
- `running`: retorna 409, pois não há cancelamento seguro no meio do pipeline;
- análise de outro usuário: 404;
- falha ao remover arquivos: 500 e a linha permanece para nova tentativa.

Jobs em execução precisam terminar antes de serem excluídos.

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

Antes de enfileirar, o backend valida:

- geometria do polígono (mesmas regras de `/api/areas`);
- `start_date <= end_date` e data final não futura;
- classificadores U-Net e o ensemble regional SP apenas com Sentinel-2;
- área compatível com `MAX_CLASSIFY_PIXELS` na resolução do satélite.

Falhas de validação retornam 422 sem criar análise nem consumir a fila.

O processamento roda em segundo plano, uma análise por vez. Acompanhe pelo
detalhe da análise; detalhes da fila em [`JOBS.md`](JOBS.md).

## Imagens

`GET /outputs/{job_id}/{filename}.png` serve os PNGs da classificação. Exige
sessão e só responde para o dono da análise; qualquer outro acesso retorna
404 (ou 401 sem sessão).

## Públicas

`GET /api/health`, `GET /api/health/live` e `GET /api/geocode` não exigem
autenticação.

# Fila de classificação

O GeoRural processa classificações em segundo plano para que o usuário não
precise manter a página aberta por 30–120 s.

## Como funciona

- `POST /api/classify` valida a área, cria a linha em `analyses`
  (`status=queued`) e enfileira o `job_id`.
- A resposta é `202` imediata com `{ job_id, status: "pending" }`.
- Um único worker em processo executa a classificação. Durante a execução, o
  worker grava `stage` e `progress` no banco.
- O frontend consulta `GET /api/analyses/{job_id}` a cada 2 s enquanto o status
  for `queued` ou `running`.
- Ao final, a análise vira `completed` (com `result`) ou `failed` (com
  `error`).

## Etapas

| `stage` | Progresso |
|---|---|
| `queued` | 0 |
| `starting` | 5 |
| `validating` | 10 |
| `coverage` | 15 |
| `searching_images` | 20–25 |
| `downloading` | 35 |
| `preprocessing` | 45 |
| `features` | 55 |
| `inference` | 65 |
| `postprocessing` | 75 |
| `saving_layers` | 82 |
| `topography` | 88 |
| `crop` | 94 |
| `finalizing` | 98 |
| `done` | 100 |

## Por que um worker só

- Evita duas inferências pesadas ao mesmo tempo (pico de RAM).
- Respeita melhor a cota do Google Earth Engine.
- É suficiente para o uso local do TCC.

Jobs enviados enquanto outro roda ficam em `queued` e começam em seguida.

## Reprocessar

`POST /api/analyses/{job_id}/retry` copia o polígono e os parâmetros da análise
original e enfileira um novo job. A análise antiga é preservada.

## Restart e jobs órfãos

A fila vive na memória do processo. Se o servidor reiniciar:

- jobs ainda na fila são perdidos;
- análises que ficaram em `queued`/`running` são marcadas como `failed` com
  `stage=interrupted` no próximo startup;
- o usuário pode usar **Reprocessar** para rodar de novo.

Esse comportamento é controlado por `RECOVER_STALE_JOBS` (padrão `true`) e
pressupõe **um processo Uvicorn** (`API_WORKERS=1`). Com múltiplos workers,
apenas o processo que roda o startup faz a recuperação.

## Limitações

- Sem Redis/Celery: a fila não é compartilhada entre processos e não sobrevive
  a restart.
- Não há cancelamento de job em andamento.
- Em desenvolvimento, o `--reload` do Uvicorn interrompe o job ao salvar
  arquivos.

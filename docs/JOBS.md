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
| `loading_models` | 62 |
| `inference` | 70 |
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

## Excluir e cancelar

`DELETE /api/analyses/{job_id}`:

- análise `completed`/`failed`: remove a linha e a pasta de outputs;
- análise `queued`: cancela o `Future` ainda não iniciado e remove a linha;
- análise `running`: retorna 409. O pipeline só pode ser interrompido com um
  cancelamento cooperativo, que ainda não existe;
- o job é isolado por usuário: outra conta recebe 404.

Se o worker já tiver pegado o job no instante do cancelamento, o endpoint
retorna 409 e a análise continua normalmente.

## Restart e jobs órfãos

A fila vive na memória do processo. Se o servidor reiniciar:

- jobs ainda na fila são perdidos;
- análises que ficaram em `queued`/`running` são marcadas como `failed` com
  `stage=interrupted` no próximo startup;
- o usuário pode usar **Reprocessar** para rodar de novo.

Esse comportamento é controlado por `RECOVER_STALE_JOBS` (padrão `true`) e
pressupõe **um processo Uvicorn** (`API_WORKERS=1`). Com múltiplos workers,
apenas o processo que roda o startup faz a recuperação.

## Memória dos modelos

O Random Forest do Brasil tem ~1,8 GB em disco e ocupa vários GB quando
carregado pelo scikit-learn. Para não deixar essa memória presa entre análises,
o worker limpa o cache de modelos e devolve o heap ao sistema ao final de cada
job (`completed` ou `failed`).

- Controlado por `RELEASE_MODELS_AFTER_JOB` (padrão `true`).
- A próxima análise recarrega os modelos; esse tempo aparece na etapa
  `loading_models`.
- O processo fica com um consumo ocioso bem menor (o `torch` importado pela
  U-Net continua residente, mas some o grosso do RF).
- Para priorizar velocidade em máquinas com RAM sobrando, defina
  `RELEASE_MODELS_AFTER_JOB=false`.

Acompanhe o consumo em tempo real:

```bash
curl -s http://localhost:8000/api/health/live
```

## Limitações

- Sem Redis/Celery: a fila não é compartilhada entre processos e não sobrevive
  a restart.
- Não há cancelamento de job em execução; jobs ainda na fila podem ser
  excluídos.
- Em desenvolvimento, o `--reload` do Uvicorn interrompe o job ao salvar
  arquivos.

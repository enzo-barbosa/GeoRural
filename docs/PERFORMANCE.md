# Performance da inferência

O gargalo de memória do GeoRural é o ensemble **RF + SVM + U-Net** rodando ao
mesmo tempo. Este documento registra as otimizações e como medi-las.

## Como medir

O script usa uma imagem sintética e os modelos locais em `models/`, sem chamar
o GEE:

```bash
# um modelo
uv run python scripts/benchmark_inference.py --classifier random_forest --size 256

# ensemble completo (RF + SVM + U-Net)
uv run python scripts/benchmark_inference.py --classifier auto --size 256 \
  --json-out /tmp/bench.json
```

Ele imprime tempo, RSS inicial/final, pico de RSS e hashes do mapa/da
probabilidade para comparar execuções.

## Otimizações aplicadas

- **Imports lazy:** `torch` e `sklearn` só são importados quando o modelo
  correspondente é usado. Um backend que só importa o serviço sai de ~455 MB
  para ~177 MB de RSS.
- **Inferência em chunks:** RF/SVM processam ~200 mil pixels por vez e somam as
  probabilidades ponderadas em um único mapa `float32`, em vez de manter um
  mapa H×W×6 por modelo.
- **U-Net em lotes:** patches de 64×64 processados em lotes (`UNET_BATCH_SIZE`,
  padrão 8) com `torch.inference_mode()`.
- **Liberação de intermediários:** a imagem de 15 canais e o mapa de
  probabilidades são liberados antes de gerar PNGs/topografia/cultivo.
- **cv2 lazy** em medições e pós-processamento.

## Configuração

```env
# .env (opcional)
INFERENCE_CHUNK_SIZE=200000
UNET_BATCH_SIZE=8
```

- Chunks menores reduzem mais o pico de memória, ao custo de mais chamadas.
- Lotes maiores aceleram a U-Net, mas aumentam um pouco a memória.

## Resultado de referência

Imagem sintética 256×256 (65.536 px), modelos full locais:

| Cenário | RSS inicial | Pico antes | Pico depois | Tempo antes | Tempo depois | Saída |
|---|---:|---:|---:|---:|---:|---|
| U-Net | 177 MB | 819 MB | 785 MB | 4,47 s | 4,08 s | argmax idêntico |
| RF | 177 MB | 3.968 MB | 3.794 MB | 4,93 s | 4,92 s | idêntica |
| SVM | 177 MB | 472 MB | 267 MB | 25,4 s | 26,3 s | idêntica |
| Auto (RF+SVM+U-Net) | 177 MB | 3.995 MB | 3.781 MB | 35,1 s | 33,1 s | argmax idêntico |

O pico do RF é dominado pelo carregamento do `brazil_rf.pkl` (~1,8 GB). A
economia cresce com a área: em 6M pixels, cada mapa H×W×6 ocupa ~144 MB, e o
ensemble deixa de manter três mapas simultâneos.

> Para reduzir ainda mais o pico do RF seria necessário mudar o formato do
> modelo (ONNX/Treelite) ou reduzir árvores/profundidade — ambos mudam o
> modelo, então ficam fora do escopo.

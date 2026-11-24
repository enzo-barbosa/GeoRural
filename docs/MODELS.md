# Modelos

Os modelos treinados **não ficam no Git** (são ~3,8 GB). Eles são distribuídos
como assets de uma GitHub Release e baixados por
`scripts/download_models.py`, que valida tamanho e SHA-256.

## Arquivos

| Arquivo | Uso |
|---|---|
| `brazil_rf.pkl` | Random Forest Brasil (ensemble) |
| `brazil_svm.pkl` | SVM Brasil (ensemble) |
| `brazil_unet.pth` | U-Net Brasil (6 classes, 15 canais) |
| `global_rf.pkl` | Random Forest fora do Brasil |
| `global_svm.pkl` | SVM fora do Brasil |
| `sp_rf.pkl` | Random Forest regional de SP |
| `sp_svm.pkl` | SVM regional de SP |
| `crop_classifier.pkl` | Classificador de cultivo (Brasil) |

O manifesto `models.lock.json` define a release (`models-v1`), o `base_url`, o
tamanho e o SHA-256 de cada arquivo.

## Baixar

```bash
# via Docker (não exige Python instalado)
docker compose --profile tools run --rm download-models

# ou com o ambiente uv
uv run python scripts/download_models.py
uv run python scripts/download_models.py --check
```

Downloads interrompidos são retomados a partir de um `.part`.

## Publicar uma nova versão

1. Treine no Google Colab (`notebooks/04-unet-training-colab.ipynb`) e baixe o
   `brazil_unet.pth`.
2. Coloque os arquivos em `models/`.
3. Atualize `release` e `base_url` no `models.lock.json` (ex.: `models-v2`).
4. Publique:

```bash
uv run python scripts/publish_models.py --dry-run
uv run python scripts/publish_models.py --upload --update-manifest
```

5. Faça commit do `models.lock.json` atualizado e abra um PR.

## Compatibilidade

Os `.pkl` são sensíveis às versões de `scikit-learn`/`numpy`. Use o `uv.lock`
e a imagem Docker do projeto para garantir o mesmo ambiente do treino. Se o
modelo for treinado com outra versão, registre isso no PR e teste o carregamento
com `scripts/check_unet_tile.py`/testes antes de promover.

## Limitações

- Solo Exposto e Área Construída ainda têm F1 baixo no tile de validação da
  U-Net; o ensemble RF/SVM ajuda a compensar.
- A release é pública; não coloque credenciais ou dados privados nela.

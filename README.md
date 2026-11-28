# GeoRural

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![React](https://img.shields.io/badge/react-19-61dafb)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c)
[![CI](https://github.com/enzo-barbosa/GeoRural/actions/workflows/ci.yml/badge.svg)](https://github.com/enzo-barbosa/GeoRural/actions/workflows/ci.yml)
[![CodeQL](https://github.com/enzo-barbosa/GeoRural/actions/workflows/codeql.yml/badge.svg)](https://github.com/enzo-barbosa/GeoRural/actions/workflows/codeql.yml)

Aplicação para classificação do uso e cobertura do solo em propriedades rurais utilizando imagens de satélite e aprendizado de máquina.

## Funcionalidades

- Download de imagens Sentinel-2 e Landsat 8 via Google Earth Engine
- **Cobertura híbrida:** Brasil com MapBiomas (detalhado) e exterior com ESA WorldCover, roteada automaticamente
- **6 classes:** Vegetação Nativa, Pastagem, Agricultura, Solo Exposto, Área Construída e Água
- Classificação com Random Forest, SVM e U-Net CNN, incluindo **ensemble por votação suave**
- **Modelo regional de São Paulo** (`sp_ensemble`) e índices locais (NDWI, MNDWI, SAVI, EVI, NBR)
- **Pós-processamento** por maioria 3×3 e **relatório de calibração** (ECE/Brier/NLL)
- Medidas reais (área/perímetro), topografia (NASADEM/GLO-30) e **mapa de relevo** com hipsometria + hillshade
- Tipo de cultivo (Brasil) e interface web React com mapa Leaflet
- **Busca por endereço ou coordenadas** (Nominatim/OSM) que centraliza o mapa no local
- **Login/cadastro** com sessão em cookie httpOnly e senhas com Argon2
- **Áreas salvas e histórico** de análises por usuário, com reabertura de resultados
- **Relatório PDF completo** por análise, com mapas, distribuição das classes e topografia
- **Classificação em segundo plano** com progresso por etapa e botão de reprocessar
- **Exclusão de análises** com remoção dos mapas e limpeza de arquivos órfãos
- **Memória controlada:** modelos liberados ao fim de cada análise (configurável)
- Imagens geradas em `/outputs/...` protegidas por dono da análise

## Requisitos

- Python 3.12+ (o projeto usa 3.13 via `.python-version`; não precisa instalar manualmente, o `uv` baixa)
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- Conta Google Earth Engine
- GPU apenas para treinar a CNN localmente (o fluxo oficial usa Google Colab)

## Instalação

```bash
git clone https://github.com/enzo-barbosa/GeoRural.git
cd GeoRural
uv sync --all-extras
cp .env.example .env             # Windows: Copy-Item .env.example .env
```

Para rodar scripts e ferramentas dentro do ambiente gerenciado pelo `uv`:

```bash
uv run python -m pytest -q
uv run ruff check src backend tests scripts
uv run python src/gee/extract_samples_cli.py --help
```

## Configuração

1. Crie um projeto no [Google Cloud Console](https://console.cloud.google.com/) e habilite a Earth Engine API
   (neste projeto usa-se o GCP **`georural-project`** com a conta **`georural92@gmail.com`**).
2. Autentique o GEE localmente (na conta do projeto):
   ```bash
   earthengine authenticate
   ```
3. Copie o arquivo `.env.example` para `.env` e preencha com o ID do seu projeto:
   ```
   GEE_PROJECT_ID=georural-project
   ```

## Rodando com Docker (recomendado)

O ambiente completo (Postgres + backend + frontend) sobe com Docker Compose e
funciona igual no Linux e no Windows.

```bash
# 1. Configure as variáveis
cp .env.example .env             # Windows: Copy-Item .env.example .env

# 2. Autentique o Earth Engine no host (uma vez por máquina)
uv run earthengine authenticate

# 3. Baixe os modelos da release
docker compose --profile tools run --rm download-models
# ou: uv run python scripts/dev.py models

# 4. Suba tudo
docker compose up --build
# ou: uv run python scripts/dev.py up

# 5. Aplique as migrações do banco
uv run python scripts/dev.py migrate
```

- Frontend: http://localhost:5173
- API: http://localhost:8000
- Health (sem GEE): http://localhost:8000/api/health/live
- Postgres: `localhost:5434`

O `scripts/dev.py` detecta as credenciais do GEE e expõe atalhos:

```bash
uv run python scripts/dev.py up -d      # sobe em background
uv run python scripts/dev.py logs backend
uv run python scripts/dev.py test       # pytest dentro do container
uv run python scripts/dev.py lint       # ruff dentro do container
uv run python scripts/dev.py migrate    # alembic upgrade head
uv run python scripts/dev.py down       # derruba
uv run python scripts/dev.py rebuild    # reconstrói imagens e containers
```

Para produção local (FastAPI servindo o build do React):

```bash
docker compose -f compose.prod.yml up --build
# acesse http://localhost:8000
```

> Os modelos e `outputs/` são montados como volumes. O build da imagem não os
> inclui. Sem a release `models-v1`, rode o app apenas com os modelos que você
> já tem em `models/`. Detalhes de versões e publicação em
> [`docs/MODELS.md`](docs/MODELS.md).

Documentação da API, autenticação e banco:

- [`docs/API.md`](docs/API.md) — endpoints de áreas, análises e classificação
- [`docs/AUTH.md`](docs/AUTH.md) — sessão, cookie e configuração do JWT
- [`docs/DATABASE.md`](docs/DATABASE.md) — schema e migrações
- [`docs/JOBS.md`](docs/JOBS.md) — fila em processo, etapas e recuperação
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — memória, inferência e downloads
- [`docs/MAINTENANCE.md`](docs/MAINTENANCE.md) — limpeza de outputs e backup

## Uso

1. Extraia as amostras por domínio (cada região/ano gera CSV parcial + manifesto):
   ```bash
   # Brasil, 2019–2023, 6 biomas + 6 polos agrícolas, 500/classe
   python src/gee/extract_samples_cli.py --domain brazil --regions all --years 2019-2023 \
     --samples-per-class 500 --out-dir data/samples/full/brazil \
     --combined-out data/samples/full/brazil_samples.csv --no-legacy --resume

   # Global, WorldCover 2021, 15 regiões, 500/classe
   python src/gee/extract_samples_cli.py --domain global --regions all --years 2021 \
     --label-source worldcover --samples-per-class 500 \
     --out-dir data/samples/full/global \
     --combined-out data/samples/full/global_samples.csv --no-legacy --resume

   # Cultivo (Brasil-only), 500/cultivo
   python src/gee/extract_samples_cli.py --task crop --domain brazil --regions all \
     --years 2019-2023 --samples-per-crop 500 \
     --out-dir data/samples/full/crop \
     --combined-out data/samples/full/crop_samples.csv --no-legacy --resume
   ```
2. Treine RF/SVM (validação leave-one-bioma-out) e cultivo:
   ```bash
   python notebooks/03-classifiers-test.py --csv data/samples/full/brazil_samples.csv \
     --models-dir models/full --output-dir outputs/full
   python notebooks/03-classifiers-test.py global --csv data/samples/full/global_samples.csv \
     --models-dir models/full --output-dir outputs/full
   python notebooks/05-crop-classifier.py --csv data/samples/full/crop_samples.csv \
     --models-dir models/full --output-dir outputs/full
   ```
   Gera: `models/full/brazil_rf.pkl`, `brazil_svm.pkl`, `global_rf.pkl`,
   `global_svm.pkl`, `crop_classifier.pkl` e `outputs/full/metrics_*.json`.

3. Avalie o ensemble (gera `outputs/full/ensemble_weights.json`):
   ```bash
   python notebooks/06-ensemble-eval.py --csv data/samples/full/brazil_samples.csv \
     --models-dir models/full --output-dir outputs/full
   ```

4. Modelo regional SP, calibração e avaliação do pós-processamento:
   ```bash
   python notebooks/03-classifiers-test.py sp \
     --csv data/samples/full/brazil_samples.csv \
     --filter-regions piracicaba ribeirao_preto barretos \
     --group-strategy block --prefix sp \
     --models-dir models/v2/sp --output-dir outputs/v2/sp

   python notebooks/11-calibration-report.py \
     --csv data/samples/full/brazil_samples.csv --prefix sp \
     --filter-regions piracicaba ribeirao_preto barretos \
     --group-strategy block --output-dir outputs/v2/calibration

   python notebooks/12-sp-raster-eval.py \
     --holdout-region piracicaba --output-dir outputs/v2/sp
   ```
   Gera `models/v2/sp/sp_rf.pkl`, `sp_svm.pkl`, relatórios de calibração
   (ECE/Brier/NLL/reliability) e o ganho do filtro de maioria no raster.

5. Rasters da U-Net (**38 válidos**: 26 tiles dos 8 biomas + 12 tiles
   dirigidos — 4 água, 4 solo exposto e 4 urbanos):
   ```bash
   python src/gee/build_unet_dataset.py --regions all --tiles-per-region 4 \
     --tile-size 0.08 --out data/processed/unet_full \
     --year 2023 --start 2023-01-01 --end 2023-12-31 --skip-existing

   python src/gee/build_unet_dataset.py \
     --regions unet_agua_ilha_solteira unet_agua_itaipu \
       unet_agua_balbina unet_agua_manaus \
       unet_solo_carajas unet_solo_quadrilatero \
       unet_solo_lencois unet_solo_paracatu \
       unet_urbano_sao_paulo unet_urbano_campinas \
       unet_urbano_goiania unet_urbano_belo_horizonte \
     --tiles-per-region 1 --out data/processed/unet_full \
     --year 2023 --start 2023-01-01 --end 2023-12-31 --skip-existing
   ```
   A composição passou de 18,7M para 23,7M pixels válidos: Solo 0,3%→5,2%,
   Água 0,5%→4,9% e Área Construída 2,0%→10,0%. A cobertura de patches
   (patch 64/stride 32) das raras subiu de ~4% para ~10%.
   Valide/repare antes de zipar:
   ```bash
   python scripts/validate_unet_dataset.py --dir data/processed/unet_full --fix --move-invalid
   python scripts/check_unet_tile.py --tile unet_piracicaba_t0
   ```
   Treine no **Colab (T4)** com `notebooks/04-unet-training-colab.ipynb`
   (validação **multi-tile**, 6 classes, amostragem de patches centrada nas
   classes raras 3/4/5, `FocalLoss(γ=2)` + pesos `1/sqrt(freq)`, augmentation,
   15 canais com índices locais, patches sob demanda, checkpoints no Drive).
   O pacote pronto para upload está em `data/colab_upload/` (inclui
   `LEIA-ME.md` e checksums).
   Gera `models/brazil_unet.pth`. Modelo atual (antes da Etapa A): acurácia
   0,824, mIoU 0,449 e macro F1 0,543 no tile de validação. No ensemble com
   maioria 3×3, o tile de Piracicaba chegou a 0,843 (Brasil) e 0,870 (SP).
   Cobertura de patches por classe (38 tiles): Veg 78,1%, Past 57,1%,
   Agri 35,0%, **Solo 10,4%, Urbano 13,1%, Água 10,5%** (antes: 4,1/5,3/3,8).
   Áreas pequenas (ex.: 24×83 px) são aceitas: as bordas são replicadas até
   64×64 e o resultado é recortado. Para números confiáveis, use áreas maiores
   que 640×640 m (~41 ha).

   Para testar a cota do GEE antes de gerar novos tiles:
   ```bash
   python scripts/check_gee_quota.py --lon -48.55 --lat -22.49 --size-km 2
   ```

6. Execute backend + frontend:
   ```bash
   uvicorn backend.app.main:app --reload
   cd frontend && npm install && npm run dev
   ```

> Features unificadas (`get_feature_bands`): 6 bandas (B2, B3, B4, B8, B11, B12)
> + NDVI + 2 texturas GLCM. Os índices locais (NDWI, MNDWI, SAVI, EVI, NBR,
> brilho) são calculados localmente; o modelo SP usa 15 features. O
> classificador de cultivo usa 24 features NDVI/EVI mensais.

## Estrutura do Projeto

```
georural/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── ground_truth/
│   └── samples/
├── models/
├── outputs/
├── src/
│   ├── gee/
│   ├── processing/
│   ├── classifiers/
│   ├── evaluation/
│   ├── geo/
│   └── app/
├── notebooks/
├── tests/
├── pyproject.toml
└── README.md
```

## Contribuindo

Leia o [`CONTRIBUTING.md`](CONTRIBUTING.md). O fluxo é **um PR por vez**: abra o
PR, espere todos os checks verdes, o dono revisa e faz o squash merge, e só
depois o próximo PR é iniciado. PRs empilhados não são usados.

## Autores

- Arthur Zocal Ribeiro da Silva
- Enzo Barbosa Dourado de Almeida
- Felipe Zocal Medes
- José Pedro Visotti

## Orientador

[Nome do orientador]

# GeoRural

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

## Requisitos

- Python 3.10+
- Conta Google Earth Engine
- GPU (recomendado para treinamento da CNN)

## Instalação

```bash
git clone https://github.com/SEU-USER/georural.git
cd georural
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,backend]"
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

5. Rasters da U-Net (32 gerados; 26 v�lidos ap�s valida��o: regiões × 4):
   ```bash
   python src/gee/build_unet_dataset.py --regions all --tiles-per-region 4 \
     --tile-size 0.08 --out data/processed/unet_full \
     --year 2023 --start 2023-01-01 --end 2023-12-31 --skip-existing
   ```
   Valide/repare antes de zipar:
   ```bash
   python scripts/validate_unet_dataset.py --dir data/processed/unet_full --fix --move-invalid
   python scripts/check_unet_tile.py --tile unet_piracicaba_t0
   ```
   Treine no **Colab (T4)** com `notebooks/04-unet-training-colab.ipynb`
   (leave-one-tile-out, 6 classes, augmentation, 15 canais com índices locais,
   patches sob demanda, checkpoints no Drive). O pacote pronto para upload está
   em `data/colab_upload/` (inclui `LEIA-ME.md` e checksums).
   Gera `models/brazil_unet.pth`. J� treinada (08/11/2026): acur�cia 0,824, mIoU 0,449 e macro F1 0,543 no tile de valida��o. No ensemble com maioria 3�3, o tile de Piracicaba chegou a 0,843 (Brasil) e 0,870 (regional SP).
   �reas pequenas (ex.: 24�83 px) s�o aceitas: as bordas s�o replicadas at�
   64�64 e o resultado � recortado. Para n�meros confi�veis, use �reas
   maiores que 640�640 m (~41 ha).

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

## Autores

- Arthur Zocal Ribeiro da Silva
- Enzo Barbosa Dourado de Almeida
- Felipe Zocal Medes
- José Pedro Visotti

## Orientador

[Nome do orientador]

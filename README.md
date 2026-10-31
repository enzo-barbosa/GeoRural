# GeoRural

Aplicação para classificação do uso e cobertura do solo em propriedades rurais utilizando imagens de satélite e aprendizado de máquina.

## Funcionalidades

- Download de imagens Sentinel-2 e Landsat 8 via Google Earth Engine
- **Cobertura híbrida:** Brasil com MapBiomas (detalhado) e exterior com ESA WorldCover, roteada automaticamente
- **6 classes:** Vegetação Nativa, Pastagem, Agricultura, Solo Exposto, Área Construída e Água
- Classificação com Random Forest, SVM e U-Net CNN, incluindo **ensemble por votação suave**
- Medidas reais (área/perímetro), topografia (NASADEM) e tipo de cultivo (Brasil)
- Interface web React com mapa Leaflet e camadas georreferenciadas

## Requisitos

- Python 3.10+
- Conta Google Earth Engine
- GPU (recomendado para treinamento da CNN)

## Instalação

```bash
git clone https://github.com/SEU-USER/georural.git
cd georural
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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
   # Brasil, 2019–2023, 6 biomas + polos agrícolas
   python src/gee/extract_samples_cli.py --domain brazil --regions all --years 2019-2023 --samples-per-class 200 --resume

   # Global, WorldCover 2021, 15 regiões
   python src/gee/extract_samples_cli.py --domain global --regions all --years 2021 --label-source worldcover --samples-per-class 200 --resume

   # Cultivo (Brasil-only)
   python src/gee/extract_samples_cli.py --task crop --domain brazil --regions all --years 2019-2023 --samples-per-crop 300 --resume
   ```
2. Treine RF/SVM (validação leave-one-bioma-out) e cultivo:
   ```bash
   python notebooks/03-classifiers-test.py          # Brasil
   python notebooks/03-classifiers-test.py global   # global (ou 07-global-classifiers.py)
   python notebooks/05-crop-classifier.py
   ```
   Gera: `models/brazil_rf.pkl`, `models/brazil_svm.pkl`, `models/global_rf.pkl`,
   `models/global_svm.pkl`, `models/crop_classifier.pkl` e `outputs/metrics_*.json`.

3. Avalie o ensemble (gera `outputs/ensemble_weights.json`):
   ```bash
   python notebooks/06-ensemble-eval.py
   ```

4. Rasters da U-Net (tiles pequenos por bioma):
   ```bash
   python src/gee/build_unet_dataset.py --regions all --year 2023 --start 2023-01-01 --end 2023-12-31
   ```
   Treine no **Colab (T4)** com `notebooks/04-unet-training-colab.ipynb`
   (leave-one-tile-out, 6 classes, pesos de classe). Gera `models/brazil_unet.pth`.

5. Execute backend + frontend:
   ```bash
   uvicorn backend.app.main:app --reload
   cd frontend && npm install && npm run dev
   ```

> Features unificadas (`get_feature_bands`): 6 bandas (B2, B3, B4, B8, B11, B12)
> + NDVI + 2 texturas GLCM. O classificador de cultivo usa 24 features
> NDVI/EVI mensais.

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

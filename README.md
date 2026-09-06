# GeoRural

Aplicação para classificação do uso e cobertura do solo em propriedades rurais utilizando imagens de satélite e aprendizado de máquina.

## Funcionalidades

- Download de imagens Sentinel-2 e Landsat 8 via Google Earth Engine
- Classificação com Random Forest, SVM e U-Net CNN
- Interface web interativa com mapa
- Identificação de áreas ociosas

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

1. Extraia os dados (CSV balanceado para RF/SVM e rasters para a U-Net):
   ```bash
   python src/gee/extract_samples_cli.py --satellite sentinel2 --start 2023-01-01 --end 2023-12-31 --max-cloud 80
   python src/gee/build_unet_dataset.py --satellite sentinel2 --start 2023-01-01 --end 2023-12-31 --max-cloud 80
   ```
2. Treine os classificadores clássicos (Random Forest e SVM):
   ```bash
   python notebooks/03-classifiers-test.py
   ```
   Gera: `models/random_forest.pkl` e `models/svm.pkl`

3. Treine a U-Net (PyTorch — TensorFlow NÃO é usado; GPU recomendada):
   ```bash
   python notebooks/04-unet-training.py
   # ou
   jupyter notebook notebooks/04-unet-training.ipynb
   ```
   Gera: `models/unet_model.pth`

4. Execute a aplicação:
   ```bash
   streamlit run run_app.py
   ```

> RF/SVM e U-Net usam a **mesma definição de features** (`get_feature_bands`):
> 6 bandas (B2, B3, B4, B8, B11, B12) + NDVI + 2 texturas GLCM.

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

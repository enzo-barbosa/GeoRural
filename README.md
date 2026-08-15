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

1. Crie um projeto no [Google Cloud Console](https://console.cloud.google.com/) e habilite a Earth Engine API.
2. Autentique o GEE localmente:
   ```bash
   earthengine authenticate
   ```
3. Copie o arquivo `.env.example` para `.env` e preencha com o ID do seu projeto:
   ```
   GEE_PROJECT_ID=seu-project-id-aqui
   ```

## Uso

```bash
streamlit run run_app.py
```

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

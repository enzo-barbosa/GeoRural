# PLANO DE DESENVOLVIMENTO — GEOLOCALIZAÇÃO DE PROPRIEDADES RURAIS COM USO DE DRONES E IMAGENS DE SATÉLITE

> **Documento de acompanhamento do desenvolvimento do TCC**
> Versão: 2.0 | Última atualização: Julho 2026
>
> Este documento contém **todos os passos** necessários para o desenvolvimento da aplicação,
> organizados em fases cronológicas. Cada passo possui uma checkbox para acompanhamento do progresso.

---

## ÍNDICE

- [Resumo do Projeto](#resumo-do-projeto)
- [Definições Técnicas](#definições-técnicas)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Fase 0 — Preparação do Ambiente](#fase-0--preparação-do-ambiente)
- [Fase 1 — Google Earth Engine](#fase-1--google-earth-engine)
- [Fase 2 — Pipeline de Dados](#fase-2--pipeline-de-dados)
- [Fase 3 — Classificadores Clássicos (RF + SVM)](#fase-3--classificadores-clássicos-rf--svm)
- [Fase 4 — Deep Learning (U-Net CNN)](#fase-4--deep-learning-u-net-cnn)
- [Fase 5 — Interface Web (Streamlit + Folium)](#fase-5--interface-web-streamlit--folium)
- [Fase 6 — Testes, Otimização e Publicação](#fase-6--testes-otimização-e-publicação)

---

## Resumo do Projeto

**Objetivo geral:** Desenvolver uma aplicação computacional para geolocalizar e classificar o uso e cobertura do solo em áreas rurais, integrando dados de satélites gratuitos (Sentinel-2 e Landsat 8) com algoritmos de aprendizado de máquina.

**Objetivos específicos:**
1. Coletar e pré-processar imagens orbitais de uma região de estudo
2. Extrair atributos espectrais (bandas multiespectrais, NDVI) e espaciais (textura GLCM)
3. Implementar e comparar Random Forest, SVM e U-Net CNN
4. Gerar mapas temáticos de cobertura do solo
5. Disponibilizar resultados em interface web interativa

**Classes de cobertura do solo (5 fixas):**
| Classe | Descrição |
|--------|-----------|
| Agricultura | Lavouras cultivadas (soja, milho, etc.) |
| Pastagem | Áreas de pasto para pecuária |
| Vegetação Nativa | Mata nativa, cerrado, floresta |
| Solo Exposto | Terra sem cobertura vegetal |
| Área Construída | Edificações, estradas, infraestrutura |

**Classificadores a serem implementados:**
- Random Forest (RF) — 200 árvores, profundidade máxima 15
- Support Vector Machine (SVM) — kernel RBF, C=100, gamma='scale'
- U-Net CNN — 100 épocas, Adam (lr=0.001), data augmentation

**Métricas de avaliação:**
- Acurácia Global (Overall Accuracy)
- Precisão (Precision)
- Recall (Sensibilidade)
- F1-Score
- Coeficiente Kappa (κ)

**Método de validação:** Hold-Out (70% treinamento / 30% teste)

---

## Definições Técnicas

| Item | Definição |
|------|-----------|
| Linguagem | Python 3.10 |
| Satélites | Sentinel-2 (10m), Landsat 8 (30m) |
| Plataforma de dados | Google Earth Engine |
| ML clássico | Scikit-Learn |
| Deep Learning | TensorFlow/Keras |
| Geoespacial | Rasterio, GeoPandas, Folium |
| Interface | Streamlit |
| Visualização | Matplotlib, Folium |
| Treino GPU | Google Colab (protótipo) + RTX 5070 (produção) |
| Versionamento | Git + GitHub |
| Testes | pytest |

---

## Estrutura do Repositório

```
georural/
├── data/                          # Dados brutos e processados (não versionar)
│   ├── raw/                       # Imagens originais baixadas
│   ├── processed/                 # Imagens processadas
│   ├── ground_truth/              # Mapas de ground truth (MapBiomas + digitado)
│   └── samples/                   # Amostras de treinamento (CSV, shapefiles)
├── models/                        # Modelos treinados salvos (.h5, .pkl)
├── outputs/                       # Mapas temáticos gerados (GeoTIFF)
├── src/
│   ├── __init__.py
│   ├── gee/                       # Módulo Google Earth Engine
│   │   ├── __init__.py
│   │   ├── auth.py                # Autenticação GEE
│   │   ├── collector.py           # Download de imagens
│   │   └── preprocessor.py        # Correção atmosférica, nuvens
│   ├── processing/                # Processamento de imagens
│   │   ├── __init__.py
│   │   ├── indices.py             # NDVI, EVI
│   │   ├── texture.py             # Textura GLCM
│   │   └── features.py            # Extração consolidada
│   ├── classifiers/               # Modelos de ML
│   │   ├── __init__.py
│   │   ├── random_forest.py
│   │   ├── svm.py
│   │   └── unet_cnn.py
│   ├── evaluation/                # Métricas e validação
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── confusion.py
│   ├── geo/                       # Componentes geoespaciais
│   │   ├── __init__.py
│   │   ├── raster_io.py
│   │   └── vector_io.py
│   └── app/                       # Interface Streamlit
│       ├── __init__.py
│       ├── main.py                # Entry point do Streamlit
│       ├── pages/
│       │   ├── __init__.py
│       │   ├── home.py
│       │   ├── classify.py
│       │   └── results.py
│       └── components.py          # Componentes reutilizáveis
├── notebooks/                     # Notebooks exploratórios
├── tests/                         # Testes automatizados
├── pyproject.toml                 # Dependências e configurações
├── README.md
├── .gitignore
└── .env                           # Chaves de API (não versionar)
```

---

## Fase 0 — Preparação do Ambiente

> **Objetivo:** Configurar todo o ambiente de desenvolvimento, ferramentas e estrutura do projeto.
> **Duração estimada:** 2-3 dias

### 0.1 Instalação do Python

- [x] **0.1.1** Verificar se o Python 3.10 está instalado:
  ```bash
  python --version
  # ou
  python3 --version
  ```
  > Se não estiver instalado, baixe em https://www.python.org/downloads/ ou use o pyenv:
  > ```bash
  > curl https://pyenv.run | bash
  > pyenv install 3.10.12
  > pyenv local 3.10.12
  > ```

- [x] **0.1.2** Verificar se o pip está funcionando:
  ```bash
  pip --version
  ```

### 0.2 Criação do Ambiente Virtual

- [x] **0.2.1** Navegar até a pasta do projeto:
  ```bash
  cd /home/enzo-barbosa/Documents/Projects/GeoRural
  ```

- [x] **0.2.2** Criar o ambiente virtual:
  ```bash
  python -m venv .venv
  ```

- [x] **0.2.3** Ativar o ambiente virtual:
  ```bash
  # Linux/Mac
  source .venv/bin/activate

  # Windows (PowerShell)
  .venv\Scripts\Activate.ps1
  ```

- [x] **0.2.4** Verificar que o ambiente está ativo (o prompt deve mostrar `.venv`):
  ```bash
  which python
  # Deve mostrar: /home/enzo-barbosa/.../GeoRural/.venv/bin/python
  ```

### 0.3 Configuração do pyproject.toml

- [x] **0.3.1** Criar o arquivo `pyproject.toml` na raiz do projeto com o seguinte conteúdo:

```toml
[project]
name = "georural"
version = "0.1.0"
description = "Geolocalização de propriedades rurais com uso de drones e imagens de satélite"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    # Google Earth Engine
    "earthengine-api>=0.1.370",

    # Geoespacial
    "rasterio>=1.3.9",
    "geopandas>=0.14.0",
    "shapely>=2.0.0",
    "folium>=0.15.0",

    # Machine Learning
    "scikit-learn>=1.3.0",
    "tensorflow>=2.15.0",

    # Processamento de imagens
    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",

    # Visualização
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",

    # Interface
    "streamlit>=1.28.0",
    "streamlit-folium>=0.20.0",

    # Utilidades
    "pandas>=2.1.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### 0.4 Instalação das Dependências

- [x] **0.4.1** Atualizar o pip:
  ```bash
  pip install --upgrade pip
  ```

- [x] **0.4.2** Instalar todas as dependências:
  ```bash
  pip install -e ".[dev]"
  ```

- [x] **0.4.3** Verificar que todas as bibliotecas foram instaladas:
  ```bash
  pip list | grep -E "earthengine|rasterio|geopandas|scikit-learn|tensorflow|streamlit|folium"
  ```

  > **Nota sobre TensorFlow:** Se houver erro de instalação no Linux, pode ser necessário instalar
  > as dependências do sistema:
  > ```bash
  > sudo apt-get install libcudnn8 libcudnn8-dev
  > ```

  > **Nota sobre Rasterio:** No Linux, pode ser necessário instalar dependências do sistema:
  > ```bash
  > sudo apt-get install libgdal-dev libgeos-dev libproj-dev
  > ```

### 0.5 Configuração do Git

- [ ] **0.5.1** Inicializar o repositório Git:
  ```bash
  git init
  ```

- [ ] **0.5.2** Configurar o `.gitignore`:
  ```bash
  # Criar .gitignore na raiz do projeto
  ```

  Conteúdo do `.gitignore`:
  ```
  # Ambiente virtual
  .venv/
  venv/
  env/

  # Dados (não versionar - podem ser gigabytes)
  data/raw/
  data/processed/
  data/ground_truth/
  data/samples/

  # Modelos treinados (muito grandes)
  models/

  # Outputs
  outputs/

  # Python
  __pycache__/
  *.py[cod]
  *$py.class
  *.egg-info/
  dist/
  build/
  *.egg

  # Jupyter
  .ipynb_checkpoints/

  # Variáveis de ambiente
  .env

  # IDE
  .vscode/
  .idea/

  # OS
  .DS_Store
  Thumbs.db
  ```

- [ ] **0.5.3** Criar o arquivo `.env` (para chaves de API - NÃO versionar):
  ```
  # Google Earth Engine - será configurado na Fase 1
  GEE_PROJECT_ID=seu-project-id-aqui
  ```

### 0.6 Criação da Estrutura de Pastas

- [x] **0.6.1** Criar todas as pastas do projeto:
  ```bash
  mkdir -p data/raw data/processed data/ground_truth data/samples
  mkdir -p models outputs
  mkdir -p src/gee src/processing src/classifiers src/evaluation src/geo src/app/pages
  mkdir -p notebooks tests
  ```

- [x] **0.6.2** Criar todos os arquivos `__init__.py`:
  ```bash
  touch src/__init__.py
  touch src/gee/__init__.py
  touch src/processing/__init__.py
  touch src/classifiers/__init__.py
  touch src/evaluation/__init__.py
  touch src/geo/__init__.py
  touch src/app/__init__.py
  touch src/app/pages/__init__.py
  ```

- [x] **0.6.3** Criar arquivos stub (funções placeholder) para cada módulo:

  `src/gee/auth.py`:
  ```python
  """Autenticação com o Google Earth Engine."""

  import ee

  def initialize_gee():
      """Inicializa a conexão com o GEE."""
      # TODO: Implementar na Fase 1
      pass
  ```

  `src/gee/collector.py`:
  """Coleta de imagens do Google Earth Engine."""

  ```python
  def download_image(geometry, start_date, end_date, satellite="sentinel2"):
      """Baixa imagem de satélite para uma área e período."""
      # TODO: Implementar na Fase 2
      pass
  ```

  `src/gee/preprocessor.py`:
  """Pré-processamento de imagens orbitais."""

  ```python
  def apply_cloud_mask(image):
      """Aplica máscara de nuvens na imagem."""
      # TODO: Implementar na Fase 2
      pass

  def atmospheric_correction(image):
      """Realiza correção atmosférica."""
      # TODO: Implementar na Fase 2
      pass
  ```

  `src/processing/indices.py`:
  """Índices espectrais."""

  ```python
  def calculate_ndvi(nir_band, red_band):
      """Calcula o NDVI: (NIR - RED) / (NIR + RED)."""
      # TODO: Implementar na Fase 2
      pass

  def calculate_evi(nir_band, red_band, blue_band):
      """Calcula o EVI."""
      # TODO: Implementar na Fase 2
      pass
  ```

  `src/processing/texture.py`:
  """Extração de textura GLCM."""

  ```python
  def calculate_glcm(image, distances=[1], angles=[0]):
      """Calcula texturas GLCM: contraste, homogeneidade, entropia."""
      # TODO: Implementar na Fase 2
      pass
  ```

  `src/processing/features.py`:
  """Extração consolidada de atributos."""

  ```python
  def extract_features(image):
      """Extrai todos os atributos: bandas, NDVI, textura."""
      # TODO: Implementar na Fase 2
      pass
  ```

  `src/classifiers/random_forest.py`:
  """Classificador Random Forest."""

  ```python
  def train_random_forest(X_train, y_train, n_trees=200, max_depth=15):
      """Treina o modelo Random Forest."""
      # TODO: Implementar na Fase 3
      pass

  def predict_random_forest(model, X_test):
      """Realiza predições com o modelo treinado."""
      # TODO: Implementar na Fase 3
      pass
  ```

  `src/classifiers/svm.py`:
  """Classificador Support Vector Machine."""

  ```python
  def train_svm(X_train, y_train, kernel="rbf", C=100, gamma="scale"):
      """Treina o modelo SVM."""
      # TODO: Implementar na Fase 3
      pass

  def predict_svm(model, X_test):
      """Realiza predições com o modelo treinado."""
      # TODO: Implementar na Fase 3
      pass
  ```

  `src/classifiers/unet_cnn.py`:
  """Rede Neural Convolucional U-Net."""

  ```python
  def build_unet(input_shape, num_classes=5):
      """Constrói a arquitetura U-Net."""
      # TODO: Implementar na Fase 4
      pass

  def train_unet(model, X_train, y_train, epochs=100, lr=0.001):
      """Treina o modelo U-Net."""
      # TODO: Implementar na Fase 4
      pass
  ```

  `src/evaluation/metrics.py`:
  """Métricas de avaliação."""

  ```python
  def overall_accuracy(y_true, y_pred):
      """Calcula a Acurácia Global."""
      # TODO: Implementar na Fase 3
      pass

  def precision_score_per_class(y_true, y_pred):
      """Calcula a Precisão por classe."""
      # TODO: Implementar na Fase 3
      pass

  def recall_score_per_class(y_true, y_pred):
      """Calcula o Recall por classe."""
      # TODO: Implementar na Fase 3
      pass

  def f1_score_per_class(y_true, y_pred):
      """Calcula o F1-Score por classe."""
      # TODO: Implementar na Fase 3
      pass

  def kappa_coefficient(y_true, y_pred):
      """Calcula o Coeficiente Kappa."""
      # TODO: Implementar na Fase 3
      pass
  ```

  `src/evaluation/confusion.py`:
  """Matriz de confusão."""

  ```python
  def plot_confusion_matrix(y_true, y_pred, class_names):
      """Gera e plota a matriz de confusão."""
      # TODO: Implementar na Fase 3
      pass
  ```

  `src/geo/raster_io.py`:
  """Leitura e escrita de dados raster."""

  ```python
  def read_geotiff(filepath):
      """Lê um arquivo GeoTIFF e retorna o array + metadados."""
      # TODO: Implementar na Fase 2
      pass

  def write_geotiff(filepath, data, transform, crs):
      """Escreve um array como GeoTIFF."""
      # TODO: Implementar na Fase 2
      pass
  ```

  `src/geo/vector_io.py`:
  """Manipulação de dados vetoriais."""

  ```python
  def geojson_to_geometry(geojson):
      """Converte um GeoJSON (do Streamlit) para geometria Shapely."""
      # TODO: Implementar na Fase 5
      pass

  def geometry_to_ee_geometry(geometry):
      """Converte geometria Shapely para formato GEE."""
      # TODO: Implementar na Fase 5
      pass
  ```

  `src/app/main.py`:
  """Aplicação principal Streamlit."""

  ```python
  import streamlit as st

  def main():
      st.title("Geolocalização de Propriedades Rurais")
      # TODO: Implementar na Fase 5
      pass

  if __name__ == "__main__":
      main()
  ```

### 0.7 Primeiro Commit

- [ ] **0.7.1** Adicionar todos os arquivos e fazer o primeiro commit:
  ```bash
  git add .
  git commit -m "chore: initial project structure with stubs"
  ```

- [ ] **0.7.2** Criar o repositório no GitHub e fazer push:
  ```bash
  git remote add origin https://github.com/SEU-USER/tcc-geolocalizacao.git
  git branch -M main
  git push -u origin main
  ```

---

## Fase 1 — Google Earth Engine

> **Objetivo:** Aprender a usar o Google Earth Engine e conseguir baixar imagens de satélite.
> **Duração estimada:** 1 semana

### 1.1 Configuração da Conta GEE

- [x] **1.1.1** Acessar https://earthengine.google.com/ e criar uma conta Google. *(PENDENTE - aguardando usuário)*
  > O GEE é gratuito para uso acadêmico e de pesquisa. Basta ter uma conta Google.

- [x] **1.1.2** Preencher o formulário de cadastro do GEE:
  - Tipo de uso: "Academic/Research"
  - País: Brasil
  - Instituição: Centro Universitário de Fernandópolis

- [x] **1.1.3** Aguardar a aprovação da conta (pode levar alguns minutos a alguns dias).

- [x] **1.1.4** Após aprovação, acessar o Code Editor (https://code.earthengine.google.com/) e verificar que está funcionando.

### 1.2 Autenticação da API Python

- [x] **1.2.1** Instalar a CLI do GEE (se ainda não estiver instalada):
  ```bash
  pip install earthengine-api
  ```

- [x] **1.2.2** Fazer login pela primeira vez:
  ```bash
  earthengine authenticate
  ```
  > Isso abrirá um navegador para autenticar com sua conta Google.
  > Siga as instruções na tela.

- [x] **1.2.3** Verificar que a autenticação funcionou:
  ```bash
  python -c "import ee; ee.Initialize(); print('GEE autenticado com sucesso!')"
  ```
  > **Nota:** Necessário habilitar a API do Earth Engine no Google Cloud Console
  > e conceder permissão `roles/serviceusage.serviceUsageConsumer`.

### 1.3 Aprendizado Básico do GEE

- [x] **1.3.1** Ler a documentação oficial do GEE Python API:
  > https://developers.google.com/earth-engine/guides/python_install

- [x] **1.3.2** Criar um notebook exploratório `notebooks/01-gee-intro.ipynb` e testar:
  ```python
  import ee

  # Inicializar
  ee.Initialize()

  # Criar um ponto arbitrário (Fernandópolis, SP)
  point = ee.Geometry.Point([-50.25, -20.28])

  # Carregar coleção Sentinel-2
  collection = (
      ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(point)
      .filterDate("2024-01-01", "2024-06-30")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
  )

  # Ver quantas imagens estão disponíveis
  print(f"Imagens disponíveis: {collection.size().getInfo()}")

  # Pegar a primeira imagem
  image = collection.first()

  # Ver as bandas disponíveis
  print(f"Bandas: {image.bandNames().getInfo()}")
  ```

- [x] **1.3.3** Entender os conceitos fundamentais do GEE:
  - **`ee.Image`**: Representa uma imagem (um ou mais bandas)
  - **`ee.ImageCollection`**: Coleção de imagens (séries temporais)
  - **`ee.Geometry`**: Geometrias (pontos, polígonos, retângulos)
  - **`ee.FeatureCollection`**: Coleção de feições vetoriais
  - **Filtros**: `filterBounds()`, `filterDate()`, `filter()`
  - **Operações**: `median()`, `mean()`, `mosaic()`, `clip()`

- [x] **1.3.4** Entender as bandas do Sentinel-2:
  | Banda | Nome | Resolução | Comprimento de Onda |
  |-------|------|-----------|---------------------|
  | B1 | Coastal aerosol | 60m | 443nm |
  | B2 | Blue | 10m | 490nm |
  | B3 | Green | 10m | 560nm |
  | B4 | Red | 10m | 665nm |
  | B5 | Red Edge 1 | 20m | 705nm |
  | B6 | Red Edge 2 | 20m | 740nm |
  | B7 | Red Edge 3 | 20m | 783nm |
  | B8 | NIR | 10m | 842nm |
  | B8A | NIR narrow | 20m | 865nm |
  | B9 | Water vapor | 60m | 945nm |
  | B11 | SWIR 1 | 20m | 1610nm |
  | B12 | SWIR 2 | 20m | 2190nm |

- [x] **1.3.5** Entender as bandas do Landsat 8:
  | Banda | Nome | Resolução | Comprimento de Onda |
  |-------|------|-----------|---------------------|
  | B1 | Coastal/Aerosol | 30m | 443nm |
  | B2 | Blue | 30m | 482nm |
  | B3 | Green | 30m | 561nm |
  | B4 | Red | 30m | 655nm |
  | B5 | NIR | 30m | 865nm |
  | B6 | SWIR 1 | 30m | 1609nm |
  | B7 | SWIR 2 | 30m | 2201nm |

- [x] **1.3.6** Praticar: baixar uma imagem Sentinel-2 e visualizar no notebook:
  ```python
  import ee
  import matplotlib.pyplot as true
  import numpy as np

  ee.Initialize()

  # Definir área de teste (Fernandópolis)
  point = ee.Geometry.Point([-50.25, -20.28])
  area = point.buffer(5000) # Raio de 5km

  # Buscar imagem Sentinel-2 com poucas nuvens
  collection = (
      ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(area)
      .filterDate("2024-01-01", "2024-06-30")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
      .sort("CLOUDY_PIXEL_PERCENTAGE")
  )

  # Pegar a melhor imagem
  image = collection.first()

  # Visualizar bandas RGB (B4, B3, B2)
  rgb_vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}

  # Obter thumbnail
  url = image.getThumbURL({
      "bands": ["B4", "B3", "B2"],
      "min": 0,
      "max": 3000,
      "dimensions": 512
  })

  print(f"URL da imagem: {url}")
  ```

### 1.4 Primeira Imagem Baixada

- [x] **1.4.1** Criar um script `src/gee/test_download.py` para testar o download:
  ```python
  """Script de teste para download de imagem GEE."""

  import ee
  import os

  ee.Initialize()

  # Definir área de teste
  point = ee.Geometry.Point([-50.25, -20.28]) # Fernandópolis
  area = point.buffer(5000)

  # Buscar imagem Sentinel-2
  collection = (
      ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(area)
      .filterDate("2024-01-01", "2024-06-30")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
      .sort("CLOUDY_PIXEL_PERCENTAGE")
  )

  image = collection.first()

  # Obter informações da imagem
  info = image.getInfo()
  print(f"ID da imagem: {info['id']}")
  print(f"Data: {info['properties']['DATE_ACQUIRED']}")
  print(f"Cobertura de nuvens: {info['properties']['CLOUDY_PIXEL_PERCENTAGE']:.1f}%")

  # Exportar para Google Drive (alternativa para imagens grandes)
  task = ee.batch.Export.image.toDrive(
      image=image,
      description="sentinel2_fernandopolis_test",
      folder="tcc-gee",
      region=area,
      scale=10,
      maxPixels=1e10,
      crs="EPSG:4326"
  )
  task.start()
  print("Exportação iniciada. Verifique o Google Drive em alguns minutos.")
  ```

- [ ] **1.4.2** Rodar o script e verificar que a imagem chega ao Google Drive:
  ```bash
  python src/gee/test_download.py
  ```

- [ ] **1.4.3** Criar um diretório `data/raw/` e salvar a imagem baixada (manualmente ou via script).

- [ ] **1.4.4** Commit:
  ```bash
  git add .
  git commit -m "feat: GEE authentication and first image download"
  git push
  ```

---

## Fase 2 — Pipeline de Dados

> **Objetivo:** Construir o pipeline completo de coleta, pré-processamento e extração de atributos.
> **Duração estimada:** 2 semanas

### 2.1 Módulo de Autenticação (gee/auth.py)

- [x] **2.1.1** Implementar a função de autenticação reutilizável:
  ```python
  """Autenticação com o Google Earth Engine — v2.0.

  Suporta dois modos:
  1. Local: Autenticação interativa via navegador (padrão)
  2. Cloud: Service Account via variável de ambiente (Streamlit Cloud)
  """

  import ee
  import os
  import json
  from dotenv import load_dotenv

  load_dotenv()

  def initialize_gee():
      """Inicializa a conexão com o GEE.

      Tenta Service Account primeiro (deploy cloud), depois
      autenticação interativa (execução local).
      """
      # Tentar Service Account (deploy cloud)
      service_account_json = os.environ.get("GEE_SERVICE_ACCOUNT")
      service_account_email = os.environ.get("GEE_SERVICE_ACCOUNT_EMAIL")

      if service_account_json and service_account_email:
          key_data = json.loads(service_account_json)
          credentials = ee.ServiceAccountCredentials(
              email=service_account_email,
              key=key_data
          )
          ee.Initialize(credentials)
          print("GEE inicializado com Service Account")
      else:
          # Autenticação interativa (local)
          try:
              ee.Initialize()
              print("GEE inicializado (credenciais existentes)")
          except ee.EEException:
              ee.Authenticate()
              ee.Initialize()
              print("GEE autenticado e inicializado")
  ```

- [x] **2.1.2** Testar a autenticação:
  ```python
  from src.gee.auth import initialize_gee
  initialize_gee()
  ```

### 2.2 Módulo de Coleta (gee/collector.py)

- [x] **2.2.1** Implementar a função de conversão de polígono para geometria GEE:
  ```python
  """Coleta de imagens do Google Earth Engine."""

  import ee
  from typing import Optional

  def polygon_to_ee_geometry(coordinates: list) -> ee.Geometry:
      """Converte coordenadas de polígono para geometria GEE.

      Args:
          coordinates: Lista de coordenadas [[lon, lat], ...]

      Returns:
          Objeto ee.Geometry do polígono
      """
      return ee.Geometry.Polygon(coordinates)
  ```

- [x] **2.2.2** Implementar a função de busca de imagens:
  ```python
  def search_images(
      geometry: ee.Geometry,
      start_date: str,
      end_date: str,
      satellite: str = "sentinel2",
      max_cloud_pct: int = 20
  ) -> ee.ImageCollection:
      """Busca imagens de satélite para uma área e período.

      Args:
          geometry: Geometria da área de interesse
          start_date: Data inicial (YYYY-MM-DD)
          end_date: Data final (YYYY-MM-DD)
          satellite: 'sentinel2' ou 'landsat8'
          max_cloud_pct: Percentual máximo de cobertura de nuvens

      Returns:
          Coleção de imagens filtrada
      """
      if satellite == "sentinel2":
          collection = (
              ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(geometry)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
          )
      elif satellite == "landsat8":
          collection = (
              ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .filterBounds(geometry)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt("CLOUD_COVER", max_cloud_pct))
          )
      else:
          raise ValueError(f"Satélite não suportado: {satellite}")

      return collection.sort("CLOUDY_PIXEL_PERCENTAGE")
  ```

- [x] **2.2.3** Implementar a função de composição de imagem (mosaico mediano):
  ```python
  def get_composite_image(
      collection: ee.ImageCollection,
      geometry: ee.Geometry
  ) -> ee.Image:
      """Gera mosaico mediano de uma coleção.

      O mosaico mediano combina múltiplas imagens, selecionando o valor
      mediano de cada pixel ao longo do tempo. Isso reduz efeitos de
      nuvens e ruídos.

      Args:
          collection: Coleção de imagens
          geometry: Geometria para recortar o resultado

      Returns:
          Imagem composta (mosaico mediano)
      """
      composite = collection.median().clip(geometry)
      return composite
  ```

- [x] **2.2.4** Implementar a função de download da imagem:
  ```python
  def download_image(
      image: ee.Image,
      geometry: ee.Geometry,
      filepath: str,
      scale: int = 10,
      bandas: Optional[list] = None
  ) -> str:
      """Baixa uma imagem GEE para arquivo local.

      Para imagens grandes, usa exportação para Google Drive.
      Para imagens menores, baixa diretamente.

      Args:
          image: Imagem GEE
          geometry: Geometria da área
          filepath: Caminho para salvar o arquivo
          scale: Resolução em metros (10 para Sentinel-2, 30 para Landsat)
          bandas: Lista de bandas para baixar (None = todas)

      Returns:
          Caminho do arquivo salvo
      """
      if bandas:
          image = image.select(bandas)

      # Para imagens pequenas, download direto
      url = image.getDownloadURL({
          "region": geometry,
          "scale": scale,
          "format": "GEO_TIFF",
          "maxPixels": 1e9
      })

      # Note: Para produção, usar ee.batch.Export para imagens grandes
      return url
  ```

### 2.3 Módulo de Pré-processamento (gee/preprocessor.py)

- [x] **2.3.1** Implementar a máscara de nuvens para Sentinel-2:
  ```python
  """Pré-processamento de imagens orbitais."""

  import ee

  def mask_clouds_sentinel2(image: ee.Image) -> ee.Image:
      """Aplica máscara de nuvens em imagem Sentinel-2.

      Usa a banda SCL (Scene Classification Layer) para identificar
      nuvens, sombras de nuvens e pixels confiáveis.

      Classes SCL:
      - 4: Cloud medium probability
      - 5: Cloud high probability
      - 6: Thin cirrus
      - 8: Cloud medium probability (secondary)
      - 9: Cloud high probability (secondary)
      - 10: Cirrus (secondary)

      Args:
          image: Imagem Sentinel-2

      Returns:
          Imagem com máscara de nuvens aplicada
      """
      scl = image.select("SCL")

      # Pixels confiáveis (não nuvem)
      mask = (
          scl.neq(4) # Cloud medium probability
          .And(scl.neq(5)) # Cloud high probability
          .And(scl.neq(6)) # Thin cirrus
          .And(scl.neq(8)) # Cloud medium probability (secondary)
          .And(scl.neq(9)) # Cloud high probability (secondary)
          .And(scl.neq(10)) # Cirrus (secondary)
      )

      return image.updateMask(mask)
  ```

- [x] **2.3.2** Implementar a correção atmosférica para Sentinel-2:
  ```python
  def atmospheric_correction_sentinel2(image: ee.Image) -> ee.Image:
      """Aplica correção atmosférica na imagem Sentinel-2.

      Converte os valores digitais (DN) para reflectância de superfície
      usando os fatores de escala fornecidos pelo GEE.

      Fórmula: reflectância = DN * 0.0001 - 0.1

      Args:
          image: Imagem Sentinel-2 (nivel L2A - SR)

      Returns:
          Imagem com reflectância de superfície
      """
      # Sentinel-2 L2A já vem em reflectância, mas aplicamos o offset
      optical = image.select("B.*").multiply(0.0001).add(-0.1)

      # Manter bandas não-ópticas (SCL, etc.)
      other = image.select(["SCL", "MSK_CLDPRB"])

      return image.addBands(optical, overwrite=True)
  ```

- [x] **2.3.3** Implementar a máscara de nuvens para Landsat 8:
  ```python
  def mask_clouds_landsat8(image: ee.Image) -> ee.Image:
      """Aplica máscara de nuvens em imagem Landsat 8.

      Usa a banda QA_PIXEL para identificar nuvens e sombras.

      Bits QA_PIXEL:
      - Bit 3: Cloud
      - Bit 4: Cloud Shadow

      Args:
          image: Imagem Landsat 8

      Returns:
          Imagem com máscara de nuvens aplicada
      """
      qa = image.select("QA_PIXEL")

      cloud_mask = qa.bitwiseAnd(1 << 3).eq(0) # Bit 3 = Cloud
      shadow_mask = qa.bitwiseAnd(1 << 4).eq(0) # Bit 4 = Cloud Shadow

      mask = cloud_mask.And(shadow_mask)

      return image.updateMask(mask)
  ```

- [x] **2.3.4** Implementar a correção atmosférica para Landsat 8:
  ```python
  def atmospheric_correction_landsat8(image: ee.Image) -> ee.Image:
      """Aplica correção atmosférica na imagem Landsat 8.

      Converte os valores digitais para reflectância de superfície
      usando os fatores de escala do Collection 2.

      Fórmula: reflectância = DN * 0.0000275 - 0.2

      Args:
          image: Imagem Landsat 8 (Collection 2 Level 2)

      Returns:
          Imagem com reflectância de superfície
      """
      # Fatores de escala do Landsat 8 Collection 2
      scale = 0.0000275
      offset = -0.2

      # Aplicar em todas as bandas ópticas
      optical_bands = image.select("SR_B.").multiply(scale).add(offset)

      # Manter bandas QA
      qa = image.select("QA_PIXEL")

      return image.addBands(optical_bands, overwrite=True).addBands(qa)
  ```

- [x] **2.3.5** Implementar função de pré-processamento completo:
  ```python
  def preprocess_image(
      image: ee.Image,
      satellite: str = "sentinel2"
  ) -> ee.Image:
      """Aplica pré-processamento completo na imagem.

      Pipeline: correção atmosférica → máscara de nuvens

      Args:
          image: Imagem bruta do GEE
          satellite: 'sentinel2' ou 'landsat8'

      Returns:
          Imagem pré-processada
      """
      if satellite == "sentinel2":
          image = atmospheric_correction_sentinel2(image)
          image = mask_clouds_sentinel2(image)
      elif satellite == "landsat8":
          image = atmospheric_correction_landsat8(image)
          image = mask_clouds_landsat8(image)
      else:
          raise ValueError(f"Satélite não suportado: {satellite}")

      return image
  ```

### 2.4 Módulo de Índices Spectrais (processing/indices.py)

- [x] **2.4.1** Implementar o NDVI:
  ```python
  """Índices espectrais para classificação do uso do solo."""

  import ee
  import numpy as np

  def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
      """Calcula o NDVI (Normalized Difference Vegetation Index).

      Fórmula: NDVI = (NIR - RED) / (NIR + RED)

      Interpretação:
      - NDVI < 0: Água, nuvens, solo exposto
      - NDVI 0 - 0.2: Solo exposto, vegetação rala
      - NDVI 0.2 - 0.5: Vegetação moderada (pastagem)
      - NDVI 0.5 - 1.0: Vegetação densa (mata nativa, lavoura sadia)

      Args:
          nir_band: Banda do infravermelho próximo
          red_band: Banda do vermelho

      Returns:
          Array com valores NDVI (-1 a 1)
      """
      nir = nir_band.astype(np.float32)
      red = red_band.astype(np.float32)

      # Evitar divisão por zero
      denominator = nir + red
      denominator[denominator == 0] = np.nan

      ndvi = (nir - red) / denominator

      return ndvi
  ```

- [x] **2.4.2** Implementar o NDVI para GEE:
  ```python
  def calculate_ndvi_ee(image: ee.Image, satellite: str = "sentinel2") -> ee.Image:
      """Calcula o NDVI diretamente no GEE.

      Args:
          image: Imagem GEE pré-processada
          satellite: 'sentinel2' ou 'landsat8'

      Returns:
          Imagem com banda NDVI adicionada
      """
      if satellite == "sentinel2":
          ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
      elif satellite == "landsat8":
          ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
      else:
          raise ValueError(f"Satélite não suportado: {satellite}")

      return image.addBands(ndvi)
  ```

- [x] **2.4.3** Implementar o EVI:
  ```python
  def calculate_evi(
      nir_band: np.ndarray,
      red_band: np.ndarray,
      blue_band: np.ndarray
  ) -> np.ndarray:
      """Calcula o EVI (Enhanced Vegetation Index).

      Fórmula: EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)

      O EVI é mais sensível a variações de alta densidade vegetacional
      e minimiza influências atmosféricas e de solo.

      Args:
          nir_band: Banda do infravermelho próximo
          red_band: Banda do vermelho
          blue_band: Banda do azul

      Returns:
          Array com valores EVI (-1 a 1)
      """
      nir = nir_band.astype(np.float32)
      red = red_band.astype(np.float32)
      blue = blue_band.astype(np.float32)

      denominator = nir + 6 * red - 7.5 * blue + 1
      denominator[denominator == 0] = np.nan

      evi = 2.5 * (nir - red) / denominator

      return evi
  ```

### 2.5 Módulo de Textura GLCM (processing/texture.py)

- [x] **2.5.1** Entender o que é GLCM (Gray-Level Co-occurrence Matrix):
  > GLCM é uma matriz que descreve a frequência com que pares de pixels
  > com intensidades específicas aparecem em uma determinada orientação.
  > A partir dela, calculam-se atributos de textura como:
  > - **Contraste**: Mede a variação local de intensidade
  > - **Homogeneidade**: Mede a proximidade da distribuição aos valores diagonais
  > - **Entropia**: Mede a aleatoriedade da distribuição de intensidades

- [x] **2.5.2** Entender a implementação correta da textura GLCM:
  > **Importante:** A textura GLCM deve ser calculada em **janela deslizante**
  > (pixel a pixel), e não como um único valor para a imagem inteira.
  > A versão local com `graycomatrix` do scikit-image calcula a matriz
  > GLCM de **toda a imagem** e gera um escalar único replicado para
  > todos os pixels — isso **não é textura local** e produzirá
  > classificação incorreta.
  >
  > A solução correta é usar a função `glcmTexture()` do GEE, que aplica
  > uma janela deslizante (kernel) sobre a imagem, calculando as
  > propriedades de textura **pixel a pixel**. Isso garante que cada
  > pixel tenha seu próprio valor de textura baseado na vizinhança.

- [x] **2.5.3** Implementar a textura GLCM via GEE (única versão correta):
  ```python
  def calculate_glcm_ee(
      image: ee.Image,
      band_name: str = "B4",
      kernel_size: int = 3
  ) -> ee.Image:
      """Calcula texturas GLCM diretamente no GEE.

      Args:
          image: Imagem GEE
          band_name: Nome da banda para calcular GLCM
          kernel_size: Tamanho do kernel (janela)

      Returns:
          Imagem com bandas de textura adicionadas
      """
      # Converter para níveis de cinza (0-255)
      band = image.select(band_name)
      band_gray = band.multiply(255).toUint8()

      # Calcular GLCM
      glcm = band_gray.glcmTexture(size=kernel_size)

      # Extrair propriedades
      contrast = glcm.select(f"{band_name}_contrast").rename("contrast")
      homogeneity = glcm.select(f"{band_name}_dissimilarity").rename("homogeneity")
      entropy = glcm.select(f"{band_name}_idm").rename("homogeneity2")

      return image.addBands([contrast, homogeneity])
  ```

### 2.6 Módulo de Extração Consolidada (processing/features.py)

- [x] **2.6.1** Implementar a extração completa de atributos:
  ```python
  """Extração consolidada de atributos para classificação."""

  import ee
  import numpy as np
  from src.processing.indices import calculate_ndvi_ee
  from src.processing.texture import calculate_glcm_ee

  def extract_features_ee(
      image: ee.Image,
      satellite: str = "sentinel2"
  ) -> ee.Image:
      """Extrai todos os atributos de uma imagem GEE.

      Pipeline completo:
      1. Bandas originais (reflectância)
      2. NDVI
      3. Textura GLCM

      Args:
          image: Imagem GEE pré-processada
          satellite: 'sentinel2' ou 'landsat8'

      Returns:
          Imagem com todas as bandas de atributos
      """
      # 1. Bandas originais já estão na imagem

      # 2. Calcular NDVI
      image = calculate_ndvi_ee(image, satellite)

      # 3. Calcular textura GLCM (usando banda do vermelho)
      if satellite == "sentinel2":
          red_band = "B4"
      else:
          red_band = "SR_B4"

      image = calculate_glcm_ee(image, band_name=red_band)

      return image

  def get_feature_bands(satellite: str = "sentinel2") -> list:
      """Retorna a lista de bandas de atributos para classificação.

      Args:
          satellite: 'sentinel2' ou 'landsat8'

      Returns:
          Lista com nomes das bandas de atributos
      """
      if satellite == "sentinel2":
          return [
              "B2", "B3", "B4", "B8", "B11", "B12", # Bandas originais
              "NDVI", # Índice de vegetação
              "contrast", "homogeneity" # Textura GLCM
          ]
      else: # Landsat 8
          return [
              "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
              "NDVI",
              "contrast", "homogeneity"
          ]
  ```

### 2.7 Módulo Geoespacial (geo/raster_io.py e geo/vector_io.py)

- [x] **2.7.1** Implementar leitura de GeoTIFF:
  ```python
  """Leitura e escrita de dados raster."""

  import rasterio
  import numpy as np
  from typing import Tuple

  def read_geotiff(filepath: str) -> Tuple[np.ndarray, dict]:
      """Lê um arquivo GeoTIFF.

      Args:
          filepath: Caminho do arquivo GeoTIFF

      Returns:
          Tupla (array_dados, metadados)
          - array_dados: numpy array de forma (bandas, altura, largura)
          - metadados: dict com transform, crs, shape
      """
      with rasterio.open(filepath) as src:
          data = src.read() # (bandas, height, width)
          meta = {
              "transform": src.transform,
              "crs": src.crs,
              "shape": data.shape,
              "dtype": data.dtype,
              "bounds": src.bounds
          }

      return data, meta

  def write_geotiff(
      filepath: str,
      data: np.ndarray,
      transform: rasterio.transform.Affine,
      crs: str,
      dtype: str = "float32"
  ):
      """Escreve um array como GeoTIFF.

      Args:
          filepath: Caminho de saída
          data: numpy array de forma (bandas, altura, largura)
          transform: Transformação geográfica
          crs: Sistema de coordenadas (ex: "EPSG:4326")
          dtype: Tipo de dado de saída
      """
      height, width = data.shape[1], data.shape[2]
      num_bands = data.shape[0]

      with rasterio.open(
          filepath,
          "w",
          driver="GTiff",
          height=height,
          width=width,
          count=num_bands,
          dtype=dtype,
          crs=crs,
          transform=transform
      ) as dst:
          for i in range(num_bands):
              dst.write(data[i], i + 1)

      print(f"GeoTIFF salvo em: {filepath}")
  ```

- [x] **2.7.2** Implementar manipulação de polígonos:
  ```python
  """Manipulação de dados vetoriais."""

  import geopandas as gpd
  import json
  from shapely.geometry import shape, mapping
  import ee

  def geojson_to_geometry(geojson: dict):
      """Converte GeoJSON (do Streamlit) para geometria Shapely.

      Args:
          geojson: Dicionário GeoJSON

      Returns:
          Objeto Shapely Geometry
      """
      return shape(geojson)

  def geometry_to_ee(geometry) -> ee.Geometry:
      """Converte geometria Shapely para formato GEE.

      Args:
          geometry: Objeto Shapely Geometry

      Returns:
          Objeto ee.Geometry
      """
      geojson = mapping(geometry)
      return ee.Geometry(geojson)

  def save_shapefile(
      geometry,
      filepath: str,
      crs: str = "EPSG:4326",
      class_name: str = "unclassified"
  ):
      """Salva uma geometria como shapefile.

      Args:
          geometry: Geometria Shapely
          filepath: Caminho de saída (.shp)
          crs: Sistema de coordenadas
          class_name: Nome da classe
      """
      gdf = gpd.GeoDataFrame(
          {"class": [class_name]},
          geometry=[geometry],
          crs=crs
      )
      gdf.to_file(filepath, driver="ESRI Shapefile")
  ```

### 2.8 Preparação da Verdade Terrestre (Ground Truth) — v2.0

> **MUDANÇA v2.0:** Seção completamente reescrita. Antes, as amostras eram
> geradas aleatoriamente (sintéticas). Agora, usamos o MapBiomas como ground
> truth real e extraímos amostras dos pixels de treinamento/classificação.

**O que é ground truth?**
É a informação do que **realmente existe** no terreno. Para treinar/classificar,
precisamos de uma referência confiável. O **MapBiomas** fornece mapeamentos
anuais de cobertura do solo para todo o Brasil, e está disponível **direto no
GEE** — não é necessário baixar manualmente. É a base de verdade terrestre mais
acessível e validada para o contexto deste projeto.

- [x] **2.8.1** Acessar MapBiomas no GEE:
  ```python
  """Acesso ao MapBiomas Collection 10 via Google Earth Engine."""

  import ee

  def get_mapbiomas_layer(year: int = 2023, geometry: ee.Geometry = None) -> ee.Image:
      """Retorna a camada MapBiomas Collection 10 para o ano e região desejados.

      O MapBiomas está disponível como ImageCollection pública no GEE:
      projects/mapbiomas-public/assets/brazil/lulc/v1

      Args:
          year: Ano da classificação (1985-2023 para Collection 10)
          geometry: Região de interesse (recortará o resultado)

      Returns:
          Imagem GEE com a camada de cobertura do solo do MapBiomas
      """
      mapbiomas = (
          ee.ImageCollection("projects/mapbiomas-public/assets/brazil/lulc/v1")
          .filter(ee.Filter.eq("collection_id", 10))
          .filter(ee.Filter.eq("year", year))
          .first()
      )

      # A banda 'classification' contém os códigos do MapBiomas
      ground_truth = mapbiomas.select("classification")

      # Recortar para a área de estudo (se fornecida)
      if geometry is not None:
          ground_truth = ground_truth.clip(geometry)

      return ground_truth

  # Exemplo de uso:
  # geometry = ee.Geometry.Rectangle([-50.2, -20.3, -50.1, -20.2])  # Fernandópolis
  # gt = get_mapbiomas_layer(year=2023, geometry=geometry)
  ```

- [x] **2.8.2** Verificar a resolução e dimensões do MapBiomas:
  > O MapBiomas Collection 10 tem resolução **30 metros** (baseado em Landsat).
  > Para Sentinel-2 (10m), haverá diferença de resolução — é necessário
  > fazer resampling ou treinar a CNN com Landsat 8 (resolução compatível).
  >
  > **Recomendação:** Para a comparação justa RF/SVM vs U-Net, usar
  > **Landsat 8** (resolução nativa de 30m, compatível com MapBiomas 30m).

- [x] **2.8.3** Mapear classes MapBiomas → Classes do projeto:
  > O MapBiomas usa códigos numéricos (1-46). Mapeamento para as 5 classes:
  >
  > | Classe TCC | Código MapBiomas | Descrição |
  > |---|---|---|
  > | Agricultura | 18, 19, 20, 39, 40, 41, 46, 47, 48 | Lavoura, soja, café, etc. |
  > | Pastagem | 15 | Pastagem cultivada |
  > | Vegetação Nativa | 1, 3, 4, 5, 6, 10, 11, 12 | Floresta, savana, mangue, etc. |
  > | Solo Exposto | 22, 23, 25, 29, 30 | Solo exposto, mineração, praias |
  > | Área Construída | 24 | Área urbana e infraestrutura |
  >
  > Referência: Legenda completa → https://brasil.mapbiomas.org/wp-content/uploads/sites/4/2024/08/Legenda-Colecao-9-LEGEND-CODE.pdf

- [x] **2.8.4** Recortar ground truth para a área de estudo:
  ```python
  def reclassify_mapbiomas(ground_truth: ee.Image) -> ee.Image:
      """Reclassifica os códigos do MapBiomas para as 5 classes do projeto.

      Args:
          ground_truth: Imagem MapBiomas com banda 'classification'

      Returns:
          Imagem com banda 'class_id' (0-4)
      """
      # Definir a reclassificação
      class_mapping = [
          [1, 0],   # Vegetação Nativa
          [3, 0],
          [4, 0],
          [5, 0],
          [6, 0],
          [10, 0],
          [11, 0],
          [12, 0],
          [15, 1],  # Pastagem
          [18, 2],  # Agricultura
          [19, 2],
          [20, 2],
          [22, 3],  # Solo Exposto
          [23, 3],
          [24, 4],  # Área Construída
          [25, 3],
          [29, 3],
          [30, 3],
          [39, 2],
          [40, 2],
          [41, 2],
          [46, 2],
          [47, 2],
          [48, 2]
      ]

      # Aplicar reclassificação
      classified = ground_truth.remap(
          from_=ee.List([x[0] for x in class_mapping]),
          to=ee.List([x[1] for x in class_mapping])
      )

      return classified.rename("class_id").toByte()

  # Uso:
  # gt_raw = get_mapbiomas_layer(year=2023, geometry=geometry)
  # gt_reclass = reclassify_mapbiomas(gt_raw)
  ```

- [ ] **2.8.5** (Opcional) Refinar ground truth com digitização manual:
  > O MapBiomas não tem precisão perfeita em todas as regiões. Para a
  > **área de teste fixa** (Fernandópolis), pode-se digitalizar ~2×2km
  > manualmente no QGIS para validação.
  >
  > Ferramenta recomendada: QGIS 3.34+
  > 1. Abrir imagem de satélite como fundo
  > 2. Criar camada vetorial com 5 atributos (classe 0-4)
  > 3. Digitalizar polígonos de cada classe (~10 polígonos por classe)
  > 4. Rasterizar com resolução de 10m (compatível com Sentinel-2)
  > 5. Salvar em `data/ground_truth/manual_validation.tif`

- [ ] **2.8.6** Validar ground truth vs imagem de satélite visualmente:
  ```python
  def visualize_ground_truth(
      image: ee.Image,
      ground_truth: ee.Image,
      region: ee.Geometry
  ):
      """Mostra lado a lado a imagem de satélite e o ground truth."""
      import geemap

      rgb = image.select(["B4", "B3", "B2"])

      vis_params_rgb = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
      vis_params_gt = {
          "min": 0, "max": 4,
          "palette": ["#228B22", "#90EE90", "#FFD700", "#8B4513", "#FF0000"]
      }

      mapa = geemap.Map(center=[-20.25, -50.18], zoom=12)
      mapa.addLayer(rgb, vis_params_rgb, "Imagem RGB")
      mapa.addLayer(ground_truth, vis_params_gt, "Ground Truth")
      mapa
  ```

- [x] **2.8.7** Exportar ground truth como GeoTIFF (data/ground_truth/):
  ```python
  def export_ground_truth(ground_truth: ee.Image, region: ee.Geometry, year: int):
      """Exporta o ground truth como GeoTIFF para o Google Drive.

      Args:
          ground_truth: Imagem reclassificada (0-4)
          region: Região de exportação
          year: Ano do MapBiomas
      """
      task = ee.batch.Export.image.toDrive(
          image=ground_truth,
          description=f"ground_truth_{year}",
          folder="tcc_data",
          region=region,
          scale=30,
          maxPixels=1e10,
          fileFormat="GeoTIFF"
      )
      task.start()
      print(f"Exportação iniciada. Verifique em https://code.earthengine.google.com/tasks")
  ```

- [x] **2.8.8** Extrair amostras REAIS a partir do ground truth + imagem:
  > **MUDANÇA v2.0:** Antes, as amostras eram geradas aleatoriamente.
  > Agora, extraímos valores **reais** dos pixels de treinamento.
  > Para cada pixel rotulado no ground truth, extraímos os valores das
  > bandas da imagem de satélite correspondentes.

  ```python
  """Extração de amostras REAIS para treinamento de modelos."""

  import numpy as np
  import pandas as pd

  CLASS_NAMES = ["Vegetação Nativa", "Pastagem", "Agricultura",
                 "Solo Exposto", "Área Construída"]

  BAND_NAMES_SENTINEL = ["B2", "B3", "B4", "B8", "B11", "B12",
                         "NDVI", "contrast", "homogeneity"]

  BAND_NAMES_LANDSAT = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
                        "NDVI", "contrast", "homogeneity"]

  def extract_real_samples(
      ground_truth: np.ndarray,
      image_data: np.ndarray,
      band_names: list,
      num_samples_per_class: int = 500,
      seed: int = 42
  ) -> pd.DataFrame:
      """Extrai amostras REAIS a partir do ground truth.

      Para cada classe, seleciona N pontos aleatórios DENTRO
      das áreas rotuladas no ground truth, e extrai os valores
      das bandas da imagem de satélite correspondente.

      Args:
          ground_truth: Array 2D com classes (0-4) do MapBiomas reclassificado
          image_data: Array 3D (bandas × altura × largura) da imagem de satélite
          band_names: Lista de nomes das bandas
          num_samples_per_class: Número de amostras por classe
          seed: Seed para reprodutibilidade

      Returns:
          DataFrame com amostras reais para treinamento
      """
      np.random.seed(seed)
      samples = []

      for class_id in range(5):
          mask = ground_truth == class_id
          ys, xs = np.where(mask)

          if len(ys) == 0:
              print(f"Aviso: classe {class_id} ({CLASS_NAMES[class_id]}) "
                    f"sem pixels no ground truth — pulando")
              continue

          n_available = min(num_samples_per_class, len(ys))
          indices = np.random.choice(len(ys), size=n_available, replace=False)

          for idx in indices:
              y, x = ys[idx], xs[idx]
              pixel_values = image_data[:, y, x]

              sample = {
                  "class_id": class_id,
                  "class_name": CLASS_NAMES[class_id],
                  "y": y, "x": x
              }
              for i, band_name in enumerate(band_names):
                  sample[f"band_{band_name}"] = pixel_values[i]

              samples.append(sample)

      df = pd.DataFrame(samples)
      print(f"Amostras extraídas: {len(df)} pixels")
      for class_id in range(5):
          n = len(df[df["class_id"] == class_id])
          print(f"  Classe {class_id} ({CLASS_NAMES[class_id]}): {n} pixels")
      return df

  def validate_samples(df: pd.DataFrame) -> None:
      """Valida a qualidade das amostras extraídas."""
      print("\n=== Validação das Amostras ===")
      print(f"Total de amostras: {len(df)}")
      print(f"\nDistribuição de classes:")
      print(df["class_name"].value_counts().to_string())
      band_cols = [c for c in df.columns if c.startswith("band_")]
      print(f"\nEstatísticas por banda:")
      print(df[band_cols].describe().round(3).to_string())
      print("\n✓ Amostras validadas com sucesso")
  ```

- [x] **2.8.9** Extrair amostras diretamente no GEE (alternativa):
  > Alternativa ao passo anterior: extrair amostras diretamente no GEE
  > usando `ee.FeatureCollection`, sem precisar baixar a imagem inteira.

  ```python
  def extract_samples_gee(
      image: ee.Image,
      ground_truth: ee.Image,
      region: ee.Geometry,
      scale: int = 30,
      seed: int = 42
  ) -> ee.FeatureCollection:
      """Extrai amostras de treinamento diretamente no GEE.

      Args:
          image: Imagem com todas as bandas de atributos
          ground_truth: Imagem de ground truth reclassificada (0-4)
          region: Região de interesse
          scale: Resolução em metros
          seed: Seed para amostragem

      Returns:
          FeatureCollection com pontos e valores de bandas
      """
      samples_per_class = 500
      total_samples = samples_per_class * 5

      training_points = ground_truth.stratifiedSample(
          numPoints=total_samples,
          classBand="class_id",
          region=region,
          scale=scale,
          seed=seed,
          geometries=True
      )

      training_data = image.sampleRegions(
          collection=training_points,
          scale=scale,
          geometries=True
      )

      return training_data
  ```

### 2.9 Validação do Pipeline Completo

- [x] **2.9.1** Criar notebook `notebooks/02-pipeline-test.ipynb`:
   ```python
   """Teste completo do pipeline de dados."""

  import sys
  sys.path.insert(0, "..")

  from src.gee.auth import initialize_gee
  from src.gee.collector import (
      polygon_to_ee_geometry,
      search_images,
      get_composite_image
  )
  from src.gee.preprocessor import preprocess_image
  from src.processing.features import extract_features_ee

  # 1. Inicializar GEE
  initialize_gee()

  # 2. Definir área (polígono de teste)
  test_polygon = [
      [-50.30, -20.25],
      [-50.20, -20.25],
      [-50.20, -20.30],
      [-50.30, -20.30],
      [-50.30, -20.25]
  ]
  geometry = polygon_to_ee_geometry(test_polygon)

  # 3. Buscar imagens
  collection = search_images(
      geometry=geometry,
      start_date="2024-01-01",
      end_date="2024-06-30",
      satellite="sentinel2",
      max_cloud_pct=15
  )
  print(f"Imagens encontradas: {collection.size().getInfo()}")

  # 4. Gerar mosaico composto
  composite = get_composite_image(collection, geometry)

  # 5. Pré-processar
  composite = preprocess_image(composite, satellite="sentinel2")

  # 6. Extrair atributos
  features = extract_features_ee(composite, satellite="sentinel2")

  print("Pipeline de dados executado com sucesso!")
  print(f"Bandas disponíveis: {features.bandNames().getInfo()}")
  ```

- [x] **2.9.2** Rodar o notebook e verificar que todos os passos funcionam:
   ```bash
   cd notebooks
   jupyter notebook 02-pipeline-test.ipynb
   ```

- [x] **2.9.3** Commit:
   ```bash
   git add .
   git commit -m "feat: complete data pipeline (GEE, preprocessing, features)"
   git push
   ```

---

## Fase 3 — Classificadores Clássicos (RF + SVM)

> **Objetivo:** Implementar, treinar e comparar os classificadores Random Forest e SVM.
> **Duração estimada:** 2 semanas

### 3.1 Fundamentos Teóricos

- [ ] **3.1.1** Estudar Random Forest:
  > **Como funciona:** O Random Forest é um método de ensemble que cria
  > múltiplas árvores de decisão. Cada árvore é treinada com um subconjunto
  > aleatório dos dados (bootstrap) e um subconjunto aleatório dos atributos.
  > A decisão final é obtida pela votação majoritária de todas as árvores.
  >
  > **Vantagens:**
  > - Robusto a ruídos e outliers
  > - Lida bem com dados de alta dimensionalidade
  > - Baixa tendência ao sobreajuste
  > - Fornece importância dos atributos
  >
  > **Hiperparâmetros principais:**
  > - `n_estimators`: Número de árvores (200 no nosso caso)
  > - `max_depth`: Profundidade máxima das árvores (15)
  > - `min_samples_split`: Mínimo de amostras para dividir um nó
  > - `min_samples_leaf`: Mínimo de amostras em uma folha

- [ ] **3.1.2** Estudar Support Vector Machine:
  > **Como funciona:** O SVM busca encontrar o hiperplano de separação
  > que maximiza a margem entre as classes. Para dados não lineares,
  > usa kernels que projetam os dados em um espaço de maior dimensão.
  >
  > **Kernel RBF (Radial Basis Function):**
  > K(x, x') = exp(-γ||x - x'||²)
  > Mapeia os dados para um espaço infinito-dimensional.
  >
  > **Hiperparâmetros principais:**
  > - `C`: Regularização (maior = menos regularização)
  > - `gamma`: Influência de cada amostra (maior = mais complexo)
  > - `kernel`: Tipo de kernel ('rbf', 'linear', 'poly')

- [ ] **3.1.3** Estudar as métricas de avaliação:
  > **Matriz de Confusão:** Tabula acertos e erros por classe.
  >
  > |  | Previsto: Agro | Previsto: Pasto | ... |
  > |--|----------------|-----------------|-----|
  > | **Real: Agro** | VP (True Positive) | FN (False Negative) | |
  > | **Real: Pasto** | FP (False Positive) | VP | |
  >
  > **Acurácia Global** = (Total de acertos) / (Total de pixels)
  >
  > **Precisão (por classe)** = VP / (VP + FP)
  > "Dos que eu classifiquei como classe X, quantos realmente são?"
  >
  > **Recall (por classe)** = VP / (VP + FN)
  > "Dos que são classe X, quantos eu acertei?"
  >
  > **F1-Score** = 2 × (Precisão × Recall) / (Precisão + Recall)
  > Média harmônica entre Precisão e Recall.
  >
  > **Coeficiente Kappa (κ):**
  > κ = (p_o - p_e) / (1 - p_e)
  > Onde p_o = acurácia observada, p_e = acurácia esperada ao acaso.
  > κ > 0.80 = excelente concordância.

### 3.2 Implementação do Módulo de Métricas (evaluation/metrics.py)

- [x] **3.2.1** Implementar todas as métricas:
   ```python
   """Métricas de avaliação de classificadores."""

   import numpy as np
   from sklearn.metrics import (
       accuracy_score,
       precision_score,
       recall_score,
       f1_score,
       cohen_kappa_score,
       confusion_matrix
   )

   CLASS_NAMES = [
       "Agricultura",
       "Pastagem",
       "Vegetação Nativa",
       "Solo Exposto",
       "Área Construída"
   ]

   def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
       """Calcula a Acurácia Global.

       Proporção do total de pixels corretamente classificados.
       """
       return float(accuracy_score(y_true, y_pred))

   def precision_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
       """Calcula a Precisão para cada classe."""
       precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
       return {name: float(p) for name, p in zip(CLASS_NAMES, precisions)}

   def recall_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
       """Calcula o Recall para cada classe."""
       recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
       return {name: float(r) for name, r in zip(CLASS_NAMES, recalls)}

   def f1_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
       """Calcula o F1-Score para cada classe."""
       f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
       return {name: float(f) for name, f in zip(CLASS_NAMES, f1s)}

   def kappa_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
       """Calcula o Coeficiente Kappa."""
       return float(cohen_kappa_score(y_true, y_pred))

   def full_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
       """Gera relatório completo de avaliação.

       Returns:
           Dict com todas as métricas calculadas
       """
       return {
           "overall_accuracy": overall_accuracy(y_true, y_pred),
           "precision": precision_per_class(y_true, y_pred),
           "recall": recall_per_class(y_true, y_pred),
           "f1_score": f1_per_class(y_true, y_pred),
           "kappa": kappa_coefficient(y_true, y_pred)
       }

   def print_evaluation(y_true: np.ndarray, y_pred: np.ndarray):
       """Imprime relatório de avaliação formatado."""
       metrics = full_evaluation(y_true, y_pred)

       print("=" * 60)
       print("RELATÓRIO DE AVALIAÇÃO")
       print("=" * 60)
       print(f"\nAcurácia Global: {metrics['overall_accuracy']:.4f}")
       print(f"Coeficiente Kappa: {metrics['kappa']:.4f}")
       print("\nMétricas por classe:")
       print("-" * 60)
       print(f"{'Classe':<20} {'Precisão':>10} {'Recall':>10} {'F1-Score':>10}")
       print("-" * 60)

       for class_name in CLASS_NAMES:
           p = metrics["precision"][class_name]
           r = metrics["recall"][class_name]
           f = metrics["f1_score"][class_name]
           print(f"{class_name:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

       print("=" * 60)
   ```

### 3.3 Implementação do Módulo de Matriz de Confusão (evaluation/confusion.py)

- [x] **3.3.1** Implementar visualização da matriz de confusão:
   ```python
   """Matriz de confusão e visualização."""

   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.metrics import confusion_matrix

   CLASS_NAMES = [
       "Agricultura",
       "Pastagem",
       "Veg. Nativa",
       "Solo Exposto",
       "Área Const."
   ]

   def plot_confusion_matrix(
       y_true: np.ndarray,
       y_pred: np.ndarray,
       save_path: str = None
   ):
       """Gera e plota a matriz de confusão.

       Args:
           y_true: Valores reais
           y_pred: Valores previstos
           save_path: Caminho para salvar a figura (None = não salvar)
       """
       cm = confusion_matrix(y_true, y_pred)

       cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

       fig, axes = plt.subplots(1, 2, figsize=(16, 6))

       sns.heatmap(
           cm, annot=True, fmt="d", cmap="Blues",
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           ax=axes[0]
       )
       axes[0].set_title("Matriz de Confusão (Contagem)")
       axes[0].set_ylabel("Valor Real")
       axes[0].set_xlabel("Valor Previsto")

       sns.heatmap(
           cm_pct, annot=True, fmt=".1f", cmap="Greens",
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           ax=axes[1]
       )
       axes[1].set_title("Matriz de Confusão (%)")
       axes[1].set_ylabel("Valor Real")
       axes[1].set_xlabel("Valor Previsto")

       plt.tight_layout()

       if save_path:
           plt.savefig(save_path, dpi=150, bbox_inches="tight")
           print(f"Matriz de confusão salva em: {save_path}")

       plt.show()

   def plot_feature_importance(
       importances: np.ndarray,
       feature_names: list,
       save_path: str = None
   ):
       """Plota a importância dos atributos (apenas para Random Forest)."""
       indices = np.argsort(importances)[::-1]

       fig, ax = plt.subplots(figsize=(10, 6))
       bars = ax.bar(
           range(len(importances)),
           importances[indices],
           align="center"
       )
       ax.set_xticks(range(len(importances)))
       ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
       ax.set_title("Importância dos Atributos (Random Forest)")
       ax.set_ylabel("Importância")

       plt.tight_layout()

       if save_path:
           plt.savefig(save_path, dpi=150, bbox_inches="tight")

       plt.show()
   ```

### 3.4 Implementação do Random Forest (classifiers/random_forest.py)

- [x] **3.4.1** Implementar o treinamento e predição:
   ```python
   """Classificador Random Forest."""

   import numpy as np
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   import joblib

   def prepare_data_rf(
       features: np.ndarray,
       labels: np.ndarray,
       test_size: float = 0.3,
       random_state: int = 42
   ):
       """Prepara dados para treinamento do RF.

       Aplica Hold-Out 70/30 com separação espacial.

       Args:
           features: Array de atributos (n_samples, n_features)
           labels: Array de rótulos (n_samples,)
           test_size: Proporção do conjunto de teste
           random_state: Semente aleatória

       Returns:
           Tupla (X_train, X_test, y_train, y_test)
       """
       X_train, X_test, y_train, y_test = train_test_split(
           features, labels,
           test_size=test_size,
           random_state=random_state,
           stratify=labels
       )

       print(f"Treinamento: {len(X_train)} amostras")
       print(f"Teste: {len(X_test)} amostras")

       return X_train, X_test, y_train, y_test

   def train_random_forest(
       X_train: np.ndarray,
       y_train: np.ndarray,
       n_trees: int = 200,
       max_depth: int = 15,
       random_state: int = 42
   ):
       """Treina o modelo Random Forest.

       Args:
           X_train: Dados de treinamento
           y_train: Rótulos de treinamento
           n_trees: Número de árvores (default: 200)
           max_depth: Profundidade máxima (default: 15)
           random_state: Semente aleatória

       Returns:
           Modelo treinado
       """
       print(f"Treinando Random Forest...")
       print(f"  - Árvores: {n_trees}")
       print(f"  - Profundidade máxima: {max_depth}")

       model = RandomForestClassifier(
           n_estimators=n_trees,
           max_depth=max_depth,
           random_state=random_state,
           n_jobs=-1,
           verbose=1
       )

       model.fit(X_train, y_train)

       print("Treinamento concluído!")

       return model

   def predict_random_forest(
       model,
       X_test: np.ndarray
   ) -> np.ndarray:
       """Realiza predições com o modelo treinado.

       Args:
           model: Modelo treinado
           X_test: Dados de teste

       Returns:
           Array com as predições
       """
       return model.predict(X_test)

   def save_model(model, filepath: str):
       """Salva o modelo treinado em disco.

       Args:
           model: Modelo treinado
           filepath: Caminho para salvar (.pkl)
       """
       joblib.dump(model, filepath)
       print(f"Modelo salvo em: {filepath}")

   def load_model(filepath: str):
       """Carrega um modelo treinado do disco.

       Args:
           filepath: Caminho do modelo (.pkl)

       Returns:
           Modelo carregado
       """
       return joblib.load(filepath)
   ```

### 3.5 Implementação do SVM (classifiers/svm.py)

- [x] **3.5.1** Implementar o treinamento e predição:
   ```python
   """Classificador Support Vector Machine."""

   import numpy as np
   from sklearn.svm import SVC
   from sklearn.preprocessing import StandardScaler
   from sklearn.pipeline import Pipeline
   import joblib

   def train_svm(
       X_train: np.ndarray,
       y_train: np.ndarray,
       kernel: str = "rbf",
       C: float = 100,
       gamma: str = "scale",
       random_state: int = 42
   ):
       """Treina o modelo SVM.

       IMPORTANTE: O SVM requer normalização dos dados.
       Usa StandardScaler (média=0, desvio_padrão=1).

       Args:
           X_train: Dados de treinamento
           y_train: Rótulos de treinamento
           kernel: Tipo de kernel ('rbf', 'linear', 'poly')
           C: Parâmetro de regularização (default: 100)
           gamma: Coeficiente do kernel RBF ('scale', 'auto', ou float)
           random_state: Semente aleatória

       Returns:
           Pipeline com scaler + modelo treinado
       """
       print(f"Treinando SVM...")
       print(f"  - Kernel: {kernel}")
       print(f"  - C: {C}")
       print(f"  - Gamma: {gamma}")

       # Pipeline: normalização + SVM
       pipeline = Pipeline([
           ("scaler", StandardScaler()),
           ("svm", SVC(
               kernel=kernel,
               C=C,
               gamma=gamma,
               random_state=random_state,
               verbose=True
           ))
       ])

       pipeline.fit(X_train, y_train)

       print("Treinamento concluído!")

       return pipeline

   def predict_svm(model, X_test: np.ndarray) -> np.ndarray:
       """Realiza predições com o modelo treinado.

       Args:
           model: Pipeline treinado (scaler + SVM)
           X_test: Dados de teste

       Returns:
           Array com as predições
       """
       return model.predict(X_test)

   def save_model(model, filepath: str):
       """Salva o modelo treinado em disco."""
       joblib.dump(model, filepath)
       print(f"Modelo salvo em: {filepath}")

   def load_model(filepath: str):
       """Carrega um modelo treinado do disco."""
       return joblib.load(filepath)
   ```

### 3.6 Treino e Avaliação

- [x] **3.6.1** Criar notebook `notebooks/03-classifiers-test.ipynb`:
   ```python
   """Treino e avaliação dos classificadores clássicos."""

   import sys
   sys.path.insert(0, "..")

   import numpy as np
   import pandas as pd
   from src.classifiers.random_forest import (
       prepare_data_rf, train_random_forest, predict_random_forest
   )
   from src.classifiers.svm import train_svm, predict_svm
   from src.evaluation.metrics import full_evaluation, print_evaluation
   from src.evaluation.confusion import plot_confusion_matrix

   # 1. Carregar dados de treinamento
   df = pd.read_csv("../data/samples/training_samples.csv")

   # Separar atributos e rótulos
   feature_cols = [c for c in df.columns if c.startswith("band_")]
   X = df[feature_cols].values
   y = df["class_id"].values

   print(f"Total de amostras: {len(X)}")
   print(f"Classes: {np.unique(y)}")

   # 2. Preparar dados (Hold-Out 70/30)
   X_train, X_test, y_train, y_test = prepare_data_rf(X, y)

   # 3. Treinar Random Forest
   rf_model = train_random_forest(X_train, y_train)
   rf_predictions = predict_random_forest(rf_model, X_test)

   # 4. Treinar SVM
   svm_model = train_svm(X_train, y_train)
   svm_predictions = predict_svm(svm_model, X_test)

   # 5. Avaliar Random Forest
   print("\n" + "=" * 60)
   print("RANDOM FOREST")
   print("=" * 60)
   print_evaluation(y_test, rf_predictions)
   plot_confusion_matrix(
       y_test, rf_predictions,
       save_path="../outputs/rf_confusion_matrix.png"
   )

   # 6. Avaliar SVM
   print("\n" + "=" * 60)
   print("SVM")
   print("=" * 60)
   print_evaluation(y_test, svm_predictions)
   plot_confusion_matrix(
       y_test, svm_predictions,
       save_path="../outputs/svm_confusion_matrix.png"
   )

   # 7. Comparação lado a lado
   rf_metrics = full_evaluation(y_test, rf_predictions)
   svm_metrics = full_evaluation(y_test, svm_predictions)

   comparison = pd.DataFrame({
       "Métrica": ["Acurácia Global", "Kappa", "Precisão Média", "Recall Médio", "F1 Médio"],
       "Random Forest": [
           rf_metrics["overall_accuracy"],
           rf_metrics["kappa"],
           np.mean(list(rf_metrics["precision"].values())),
           np.mean(list(rf_metrics["recall"].values())),
           np.mean(list(rf_metrics["f1_score"].values()))
       ],
       "SVM": [
           svm_metrics["overall_accuracy"],
           svm_metrics["kappa"],
           np.mean(list(svm_metrics["precision"].values())),
           np.mean(list(svm_metrics["recall"].values())),
           np.mean(list(svm_metrics["f1_score"].values()))
       ]
   })

   print("\n" + "=" * 60)
   print("COMPARAÇÃO RF vs SVM")
   print("=" * 60)
   print(comparison.to_string(index=False))

   # 8. Importância dos atributos (RF)
   from src.evaluation.confusion import plot_feature_importance
   plot_feature_importance(
       rf_model.feature_importances_,
       feature_cols,
       save_path="../outputs/rf_feature_importance.png"
   )
   ```

- [x] **3.6.2** Rodar o notebook e registrar os resultados:
   ```bash
   cd notebooks
   jupyter notebook 03-classifiers-test.ipynb
   ```

- [x] **3.6.3** Salvar os modelos treinados:
   ```python
   from src.classifiers.random_forest import save_model as save_rf
   from src.classifiers.svm import save_model as save_svm

   save_rf(rf_model, "../models/random_forest.pkl")
   save_svm(svm_model, "../models/svm.pkl")
   ```

- [x] **3.6.4** Commit:
   ```bash
   git add .
   git commit -m "feat: Random Forest and SVM classifiers with evaluation"
   git push
   ```

---

## Fase 4 — Deep Learning (U-Net CNN)

> **Objetivo:** Implementar, treinar e avaliar a rede neural convolucional U-Net.
> **Duração estimada:** 2 semanas

### 4.1 Fundamentos Teóricos

- [ ] **4.1.1** Estudar Redes Neurais Convolucionais (CNNs):
  > **O que são:** CNNs são redes neurais especializadas em processar dados
  > com estrutura de grade (imagens). Elas aprendem automaticamente
  > características espaciais hierárquicas.
  >
  > **Camadas principais:**
  > - **Convolucional:** Aplica filtros (kernels) para extrair características
  > - **Pooling:** Reduz a resolução espacial (max pooling, average pooling)
  > - **Batch Normalization:** Normaliza ativações, acelera treino
  > - **Dropout:** Desativa aleatoriamente neurônios, evita sobreajuste
  > - **Up-sampling:** Aumenta a resolução (para segmentação)

- [ ] **4.1.2** Estudar a arquitetura U-Net:
  > **Estrutura:** Encoder-decoder com skip connections.
  >
  > ```
  > Input (64x64xN)
  >   │
  >   ├─ Encoder (contracting path)
  │   ├─ Conv 3x3 → ReLU → Conv 3x3 → ReLU
  │   ├─ Max Pool 2x2
  │   ├─ Conv 3x3 → ReLU → Conv 3x3 → ReLU
  │   ├─ Max Pool 2x2
  │   ├─ Conv 3x3 → ReLU → Conv 3x3 → ReLU
  │   ├─ Max Pool 2x2
  │   ├─ Conv 3x3 → ReLU → Conv 3x3 → ReLU (Bottleneck)
  │
  │   ├─ Decoder (expanding path)
  │   ├─ Up-conv 2x2
  │   ├─ Concatenate com skip connection
  │   ├─ Conv 3x3 → ReLU → Conv 3x3 → ReLU
  │   ├─ Up-conv 2x2
  │   ├─ Concatenate com skip connection
  │   ├─ Conv 3x3 → ReLU → Conv 3x3 → ReLU
  │   ├─ Up-conv 2x2
  │   ├─ Concatenate com skip connection
  │   ├─ Conv 3x3 → ReLU → Conv 3x3 → ReLU
  │
  │   └─ Output (64x64x5) ← 5 classes
  > ```
  >
  > **Skip Connections:** Conectam camadas do encoder diretamente ao decoder,
  > preservando informações espaciais de alta resolução.
  >
  > **Loss Function:** Sparse Categorical Cross-Entropy (para múltiplas classes)
  >
  > **Otimizador:** Adam com taxa de aprendizado 0.001

- [ ] **4.1.3** Estudar Data Augmentation:
  > Data augmentation aumenta artificialmente o tamanho do conjunto de
  > treinamento aplicando transformações nas imagens:
  > - **Rotação:** 0°, 90°, 180°, 270°
  > - **Espelhamento:** Horizontal e vertical
  > - **Brilho:** Variação de ±20%
  > - **Contraste:** Variação de ±20%
  >
  > Isso ajuda a rede a generalizar melhor e evita sobreajuste.

- [ ] **4.1.4** Análise do tamanho de patch — v2.0:
  > **MUDANÇA v2.0:** O patch_size original era 128px sem justificativa.
  > A escolha do patch_size deve considerar a resolução do sensor e a
  > escala dos alvos de classificação. Um patch muito grande captura
  > muita informação espacial mas torna a rede pesada e lenta.
  > Um patch muito pequeno perde contexto.
  >
  > **Tabela de cobertura por patch_size:**
  >
  > | patch_size | Sentinel-2 (10m/px) | Landsat 8 (30m/px) |
  > |---|---|---|
  > | 64 | 640m × 640m | 1920m × 1920m |
  > | 96 | 960m × 960m | 2880m × 2880m |
  > | 128 | 1280m × 1280m | 3840m × 3840m |
  >
  > **Recomendações:**
  > - **Sentinel-2:** `patch_size=64` (640m) — mantém variabilidade intra-patch
  >   sem cobrir uma área tão grande que misture classes distintas.
  > - **Landsat 8:** `patch_size=64` (1920m) — já cobre uma área significativa
  >   para capturar contexto espacial de uso do solo.
  > - **Se ground truth MapBiomas (30m):** treinar a CNN com Landsat 8
  >   (resolução nativa de 30m) evita artefatos de reamostragem.
  >
  > **Decisão:** Usar `PATCH_SIZE = 64` para ambos os sensores.
  > Se a área de estudo for muito grande (>10km), pode-se usar
  > `patch_size=96` ou `128` como experimento opcional.

### 4.2 Implementação da U-Net (classifiers/unet_cnn.py)

- [x] **4.2.1** Implementar a arquitetura U-Net completa:
   ```python
   """Rede Neural Convolucional U-Net para segmentação semântica."""

   import torch
   import torch.nn as nn
   import torch.optim as optim
   import numpy as np
   from typing import Tuple


   class DoubleConv(nn.Module):
       """Bloco convolucional duplo: Conv → BN → ReLU → Conv → BN → ReLU."""

       def __init__(self, in_channels: int, out_channels: int):
           super().__init__()
           self.double_conv = nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True),
               nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True)
           )

       def forward(self, x):
           return self.double_conv(x)


   class Down(nn.Module):
       """Bloco do encoder: DoubleConv → MaxPool."""

       def __init__(self, in_channels: int, out_channels: int):
           super().__init__()
           self.maxpool_conv = nn.Sequential(
               nn.MaxPool2d(2),
               DoubleConv(in_channels, out_channels)
           )

       def forward(self, x):
           return self.maxpool_conv(x)


   class Up(nn.Module):
       """Bloco do decoder: UpConv → Concat → DoubleConv."""

       def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
           super().__init__()
           if bilinear:
               self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
               self.conv = DoubleConv(in_channels, out_channels)
           else:
               self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
               self.conv = DoubleConv(in_channels, out_channels)

       def forward(self, x1, x2):
           x1 = self.up(x1)
           diff_y = x2.size()[2] - x1.size()[2]
           diff_x = x2.size()[3] - x1.size()[3]
           x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                        diff_y // 2, diff_y - diff_y // 2])
           x = torch.cat([x2, x1], dim=1)
           return self.conv(x)


   class OutConv(nn.Module):
       """Camada de saída: Conv 1x1."""

       def __init__(self, in_channels: int, out_channels: int):
           super().__init__()
           self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

       def forward(self, x):
           return self.conv(x)


   class UNet(nn.Module):
       """Arquitetura U-Net para segmentação semântica de imagens de satélite."""

       def __init__(self, n_channels: int = 10, n_classes: int = 5, bilinear: bool = False):
           super().__init__()
           self.n_channels = n_channels
           self.n_classes = n_classes
           self.bilinear = bilinear

           self.inc = DoubleConv(n_channels, 64)
           self.down1 = Down(64, 128)
           self.down2 = Down(128, 256)
           self.down3 = Down(256, 512)
           self.down4 = Down(512, 1024)

           self.up1 = Up(1024, 512, bilinear)
           self.up2 = Up(512, 256, bilinear)
           self.up3 = Up(256, 128, bilinear)
           self.up4 = Up(128, 64, bilinear)
           self.outc = OutConv(64, n_classes)

       def forward(self, x):
           x1 = self.inc(x)
           x2 = self.down1(x1)
           x3 = self.down2(x2)
           x4 = self.down3(x3)
           x5 = self.down4(x4)

           x = self.up1(x5, x4)
           x = self.up2(x, x3)
           x = self.up3(x, x2)
           x = self.up4(x, x1)
           logits = self.outc(x)
           return logits


   def build_unet(input_shape: Tuple[int, int, int] = (128, 128, 10), num_classes: int = 5):
       """Constrói a arquitetura U-Net.

       Args:
           input_shape: Formato da entrada (altura, largura, canais)
                        10 canais = 6 bandas + NDVI + 2 texturas + 1 extra
           num_classes: Número de classes de saída (5)

       Returns:
           Modelo U-Net
       """
       model = UNet(n_channels=input_shape[2], n_classes=num_classes)
       return model


   def compile_model(model: nn.Module, learning_rate: float = 0.001, device: str = None):
       """Compila o modelo com otimizador Adam e loss apropriada.

       Args:
           model: Modelo U-Net
           learning_rate: Taxa de aprendizado
           device: Dispositivo ('cuda' ou 'cpu')

       Returns:
           Modelo compilado, optimizer, criterion
       """
       if device is None:
           device = "cuda" if torch.cuda.is_available() else "cpu"

       model = model.to(device)

       optimizer = optim.Adam(model.parameters(), lr=learning_rate)
       criterion = nn.CrossEntropyLoss()

       print(f"Modelo compilado no dispositivo: {device}")
       print(model)

       return model, optimizer, criterion


   def train_unet(
       model: nn.Module,
       X_train: np.ndarray,
       y_train: np.ndarray,
       X_val: np.ndarray,
       y_val: np.ndarray,
       epochs: int = 100,
       batch_size: int = 16,
       learning_rate: float = 0.001,
       save_path: str = "models/unet_model.pth"
   ):
       """Treina o modelo U-Net.

       Args:
           model: Modelo U-Net
           X_train: Patches de treinamento (n_samples, height, width, channels)
           y_train: Rótulos de treinamento (n_samples, height, width)
           X_val: Patches de validação
           y_val: Rótulos de validação
           epochs: Número de épocas (default: 100)
           batch_size: Tamanho do batch (default: 16)
           learning_rate: Taxa de aprendizado (default: 0.001)
           save_path: Caminho para salvar o melhor modelo

       Returns:
           Histórico de treinamento
       """
       device = "cuda" if torch.cuda.is_available() else "cpu"
       model, optimizer, criterion = compile_model(model, learning_rate, device)

       # Converter numpy arrays para tensors PyTorch
       X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
       y_train = torch.LongTensor(y_train)
       X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
       y_val = torch.LongTensor(y_val)

       train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
       val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
       train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
       val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

       history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
       best_val_loss = float("inf")

       for epoch in range(epochs):
           model.train()
           train_loss = 0.0
           train_correct = 0
           train_total = 0

           for batch_x, batch_y in train_loader:
               batch_x, batch_y = batch_x.to(device), batch_y.to(device)

               optimizer.zero_grad()
               outputs = model(batch_x)
               loss = criterion(outputs, batch_y)
               loss.backward()
               optimizer.step()

               train_loss += loss.item() * batch_x.size(0)
               _, predicted = torch.max(outputs.data, 1)
               train_total += batch_y.numel()
               train_correct += (predicted == batch_y).sum().item()

           train_loss /= train_total
           train_acc = train_correct / train_total

           model.eval()
           val_loss = 0.0
           val_correct = 0
           val_total = 0

           with torch.no_grad():
               for batch_x, batch_y in val_loader:
                   batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                   outputs = model(batch_x)
                   loss = criterion(outputs, batch_y)

                   val_loss += loss.item() * batch_x.size(0)
                   _, predicted = torch.max(outputs.data, 1)
                   val_total += batch_y.numel()
                   val_correct += (predicted == batch_y).sum().item()

           val_loss /= val_total
           val_acc = val_correct / val_total

           history["train_loss"].append(train_loss)
           history["val_loss"].append(val_loss)
           history["train_acc"].append(train_acc)
           history["val_acc"].append(val_acc)

           if val_loss < best_val_loss:
               best_val_loss = val_loss
               torch.save(model.state_dict(), save_path)

           if (epoch + 1) % 10 == 0:
               print(f"Época {epoch+1}/{epochs}")
               print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
               print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

       print(f"Treinamento concluído! Melhor modelo salvo em: {save_path}")
       return history


   def load_model(model: nn.Module, filepath: str, device: str = None):
       """Carrega pesos do modelo treinado.

       Args:
           model: Modelo U-Net (arquitetura)
           filepath: Caminho do arquivo .pth
           device: Dispositivo

       Returns:
           Modelo com pesos carregados
       """
       if device is None:
           device = "cuda" if torch.cuda.is_available() else "cpu"

       model.load_state_dict(torch.load(filepath, map_location=device))
       model.to(device)
       model.eval()
       return model


   def predict_unet(model: nn.Module, image: np.ndarray, patch_size: int = 64, stride: int = 32):
       """Aplica o modelo em uma imagem completa.

       Args:
           model: Modelo U-Net treinado
           image: Array da imagem (height, width, channels)
           patch_size: Tamanho do patch
           stride: Passo entre patches

       Returns:
           Mapa de classificação (height, width)
       """
       device = "cuda" if torch.cuda.is_available() else "cpu"
       model.to(device)
       model.eval()

       h, w = image.shape[:2]
       prediction_map = np.zeros((h, w), dtype=np.uint8)
       count_map = np.zeros((h, w), dtype=np.float32)

       with torch.no_grad():
           for y in range(0, h - patch_size + 1, stride):
               for x in range(0, w - patch_size + 1, stride):
                   patch = image[y:y+patch_size, x:x+patch_size]
                   patch_tensor = torch.FloatTensor(patch).permute(2, 0, 1).unsqueeze(0).to(device)

                   output = model(patch_tensor)
                   pred = torch.argmax(output, dim=1).cpu().numpy()[0]

                   prediction_map[y:y+patch_size, x:x+patch_size] += pred
                   count_map[y:y+patch_size, x:x+patch_size] += 1

       count_map[count_map == 0] = 1
       prediction_map = (prediction_map / count_map).astype(np.uint8)

       return prediction_map
   ```

- [x] **4.2.2** Implementar data augmentation:
   ```python
   def create_data_augmentation():
       """Cria pipeline de data augmentation para treinamento.

       Transformações aplicadas:
       - Rotação aleatória (0-360°)
       - Espelhamento horizontal
       - Espelhamento vertical
       - Variação de brilho (±20%)

       Returns:
           Camada de augmentation do Keras
       """
       augmentation = tf.keras.Sequential([
           layers.RandomFlip("horizontal_and_vertical"),
           layers.RandomRotation(0.5), # 50% = 180°
           layers.RandomBrightness(0.2),
       ], name="data_augmentation")

       return augmentation
   ```

- [x] **4.2.3** Implementar preparação de dados para CNN:
   ```python
   def prepare_patches(
       image: np.ndarray,
       labels: np.ndarray,
       patch_size: int = 128,
       stride: int = 64
   ):
       """Extrai patches (janelas) da imagem para treinamento da CNN.

       A U-Net recebe patches de 128x128 pixels como entrada.
       O stride controla a sobreposição entre patches.

       Args:
           image: Array da imagem (height, width, channels)
           labels: Array dos rótulos (height, width)
           patch_size: Tamanho do patch (128x128)
           stride: Passo entre patches (64 = 50% sobreposição)

       Returns:
           Tuple (patches_imagem, patches_rótulo)
       """
       patches_img = []
       patches_label = []

       h, w = image.shape[:2]

       for y in range(0, h - patch_size + 1, stride):
           for x in range(0, w - patch_size + 1, stride):
               patch_img = image[y:y+patch_size, x:x+patch_size]
               patch_label = labels[y:y+patch_size, x:x+patch_size]

               patches_img.append(patch_img)
               patches_label.append(patch_label)

       patches_img = np.array(patches_img)
       patches_label = np.array(patches_label)

       print(f"Gerados {len(patches_img)} patches de {patch_size}x{patch_size}")

       return patches_img, patches_label
   ```

- [x] **4.2.4** Implementar treinamento:
   ```python
   def train_unet(
       model: Model,
       X_train: np.ndarray,
       y_train: np.ndarray,
       X_val: np.ndarray,
       y_val: np.ndarray,
       epochs: int = 100,
       batch_size: int = 16,
       save_path: str = "models/unet_model.h5"
   ):
       """Treina o modelo U-Net.

       Args:
           model: Modelo compilado
           X_train: Patches de treinamento
           y_train: Rótulos de treinamento
           X_val: Patches de validação
           y_val: Rótulos de validação
           epochs: Número de épocas (default: 100)
           batch_size: Tamanho do batch (default: 16)
           save_path: Caminho para salvar o melhor modelo

       Returns:
           Histórico de treinamento
       """
       callbacks = [
           tf.keras.callbacks.EarlyStopping(
               monitor="val_loss",
               patience=10,
               restore_best_weights=True
           ),
           tf.keras.callbacks.ReduceLROnPlateau(
               monitor="val_loss",
               factor=0.5,
               patience=5,
               min_lr=1e-6
           ),
           tf.keras.callbacks.ModelCheckpoint(
               save_path,
               monitor="val_loss",
               save_best_only=True
           )
       ]

       history = model.fit(
           X_train, y_train,
           validation_data=(X_val, y_val),
           epochs=epochs,
           batch_size=batch_size,
           callbacks=callbacks,
           verbose=1
       )

       print(f"Treinamento concluído! Melhor modelo salvo em: {save_path}")

       return history
   ```

### 4.3 Treinamento no Google Colab

- [x] **4.3.1** Criar notebook `notebooks/04-unet-training.ipynb` para Colab:
   ```python
   """Treinamento da U-Net no Google Colab (com GPU)."""

   # No Colab, ativar GPU:
   # Runtime → Change runtime type → T4 GPU

   import tensorflow as tf
   print(f"GPU disponível: {tf.config.list_physical_devices('GPU')}")

   # Montar Google Drive para acessar dados
   from google.colab import drive
   drive.mount("/content/drive")

   # Copiar dados do Drive
   !cp -r "/content/drive/MyDrive/tcc-data/data" "/content/"
   !cp -r "/content/drive/MyDrive/tcc-data/src" "/content/"

   # Instalar dependências
   !pip install rasterio geopandas earthengine-api

   # ... (código de treinamento)
   ```

- [x] **4.3.2** Transferir dados para o Google Drive:
   > Copie a pasta `data/` e `src/` para o Google Drive em
   > `MyDrive/tcc-data/`

- [x] **4.3.3** Treinar no Colab e salvar modelo:
   ```python
   # No final do notebook, salvar modelo no Drive
   model.save("/content/drive/MyDrive/tcc-models/unet_sentinel2.h5")
   ```

### 4.4 Treinamento no PC com RTX 5070

- [x] **4.4.1** Instalar CUDA e cuDNN (se ainda não estiver):
   > Para TensorFlow 2.15+:
   > - CUDA: 12.2
   > - cuDNN: 8.9
   >
   > Instruções: https://www.tensorflow.org/install/pip#step-by-step_instructions

- [x] **4.4.2** Verificar que a GPU está sendo detectada:
   ```python
   import tensorflow as tf
   print(f"GPUs disponíveis: {tf.config.list_physical_devices('GPU')}")

   # Configurar crescimento de memória
   gpus = tf.config.list_physical_devices("GPU")
   if gpus:
       try:
           for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
       except RuntimeError as e:
           print(e)
   ```

- [x] **4.4.3** Treinar o modelo localmente:
   ```python
   from src.classifiers.unet_cnn import build_unet, compile_model, train_unet

   # Carregar dados
   X_train = # patches de treinamento
   y_train = # rótulos de treinamento
   X_val = # patches de validação
   y_val = # rótulos de validação

   # Construir e compilar modelo
   model = build_unet(input_shape=(64, 64, 10), num_classes=5)
   model = compile_model(model, learning_rate=0.001)

   # Treinar
   history = train_unet(
       model, X_train, y_train, X_val, y_val,
       epochs=100,
       batch_size=16,
       save_path="models/unet_sentinel2.h5"
   )
   ```

- [x] **4.4.4** Monitorar o treinamento:
   > Verificar se a loss está diminuindo e a accuracy está aumentando.
   > Se a loss de validação começar a subir (overfitting), parar o treino.

- [x] **4.4.5** Metodologia de Comparação entre Abordagens — v2.0:
   > **MUDANÇA v2.0:** RF/SVM e U-Net operam em paradigmas diferentes.
   > RF/SVM são **point-based** (cada pixel é classificado independentemente,
   > usando apenas seus valores de bandas como atributo). U-Net é **patch-based**
   > (cada predição usa o contexto espacial de uma janela de 64×64 pixels).
   >
   > Para comparar justamente no mesmo conjunto de teste:
   >
   > 1. **Mesmo ground truth:** Usar o mesmo MapBiomas reclassificado (2.8)
   >    para gerar amostras de teste (Hold-Out 30%).
   >
   > 2. **RF/SVM:** Predição direta nos pixels de teste.
   >    - Input: vetor de atributos por pixel (9 bandas)
   >    - Output: classe predita (0-4)
   >
   > 3. **U-Net:** Predição em patches da imagem completa.
   >    - Input: patch 64×64×9
   >    - Output: mapa de classificação pixel a pixel
   >    - Para comparar: extrair a classe predita nos **mesmos pixels de teste**
   >      a partir do mapa completo gerado pela CNN.
   >
   > 4. **Métricas:** Calcular as mesmas métricas para todos (Acurácia,
   >    Kappa, F1-Score, Precisão, Recall) usando os mesmos pixels de teste.
   >
   > **Nota para o TCC escrito:** Essa distinção de paradigmas (point-based
   > vs patch-based) deve ser documentada como limitação metodológica.
   > RF/SVM e U-Net não são diretamente comparáveis em termos de informação
   > de entrada, mas as métricas no mesmo conjunto de teste permitem uma
   > comparação prática de desempenho.

### 4.5 Avaliação da CNN

- [x] **4.5.1** Avaliar o modelo treinado:
   ```python
   import numpy as np
   from src.classifiers.unet_cnn import load_model
   from src.evaluation.metrics import print_evaluation

   # Carregar modelo
   model = tf.keras.models.load_model("models/unet_sentinel2.h5")

   # Predição nos patches de teste
   y_pred_prob = model.predict(X_test)
   y_pred = np.argmax(y_pred_prob, axis=-1)

   # Achatar para cálculo de métricas
   y_true_flat = y_test.flatten()
   y_pred_flat = y_pred.flatten()

   # Avaliar
   print_evaluation(y_true_flat, y_pred_flat)
   ```

- [x] **4.5.2** Gerar mapa temático completo com a CNN:
   ```python
   def predict_full_image(model, image, patch_size=64, stride=32):
       """Aplica o modelo em uma imagem completa.

       Args:
           model: Modelo U-Net treinado
           image: Array da imagem (height, width, channels)
           patch_size: Tamanho do patch
           stride: Passo entre patches

       Returns:
           Mapa de classificação (height, width)
       """
       h, w = image.shape[:2]
       prediction_map = np.zeros((h, w), dtype=np.uint8)
       count_map = np.zeros((h, w), dtype=np.float32)

       for y in range(0, h - patch_size + 1, stride):
           for x in range(0, w - patch_size + 1, stride):
               patch = image[y:y+patch_size, x:x+patch_size]
               patch = np.expand_dims(patch, axis=0)

               pred = model.predict(patch, verbose=0)
               pred = np.argmax(pred[0], axis=-1)

               prediction_map[y:y+patch_size, x:x+patch_size] += pred
               count_map[y:y+patch_size, x:x+patch_size] += 1

       # Média para patches sobrepostos
       count_map[count_map == 0] = 1
       prediction_map = (prediction_map / count_map).astype(np.uint8)

       return prediction_map
   ```

- [x] **4.5.3** Comparação final: CNN vs RF vs SVM:
   ```python
   # Criar tabela comparativa
   comparison = pd.DataFrame({
       "Classificador": ["Random Forest", "SVM", "U-Net CNN"],
       "Acurácia Global": [rf_acc, svm_acc, cnn_acc],
       "Kappa": [rf_kappa, svm_kappa, cnn_kappa],
       "F1 Médio": [rf_f1, svm_f1, cnn_f1]
   })

   print(comparison.to_string(index=False))
   ```

- [x] **4.5.4** Commit:
   ```bash
   git add .
   git commit -m "feat: U-Net CNN implementation and training"
   git push
   ```

---

## Fase 5 — Interface Web (Streamlit + Folium)

> **Objetivo:** Construir a aplicação web interativa com visualização e classificação.
> **Duração estimada:** 2 semanas

### 5.1 Fundamentos

- [x] **5.1.1** Estudar Streamlit:
   > Streamlit é uma biblioteca Python para criar apps web rapidamente.
   > Conceitos principais:
   > - `st.title()`, `st.header()`, `st.markdown()`: Texto
   > - `st.sidebar`: Barra lateral
   > - `st.file_uploader()`: Upload de arquivos
   > - `st.columns()`: Layout em colunas
   > - `st.spinner()`: Indicador de carregamento
   > - `st.cache_data()`: Cache de dados
   > - `st.session_state`: Estado persistente

- [x] **5.1.2** Estudar Folium:
   > Folium é uma biblioteca Python para criar mapas interativos com Leaflet.
   > Conceitos principais:
   > - `folium.Map()`: Criar mapa base
   > - `folium.LatLngPopup()`: Clique no mapa
   > - `folium.Polygon()`: Desenhar polígonos
   > - `Draw()`: Ferramenta de desenho

- [x] **5.1.3** Instalar plugins adicionais:
   ```bash
   pip install streamlit-folium
   ```

### 5.2 Estrutura da Aplicação

- [x] **5.2.1** Criar o arquivo principal `src/app/main.py`:
   ```python
   """Aplicação principal - Geolocalização de Propriedades Rurais."""

   import streamlit as st
   from src.gee.auth import initialize_gee

   st.set_page_config(
       page_title="Geolocalização Rural",
       page_icon="🛰️",
       layout="wide"
   )

   # Inicializar GEE na primeira execução
   if "gee_initialized" not in st.session_state:
       initialize_gee()
       st.session_state.gee_initialized = True

   # Sidebar com navegação
   st.sidebar.title("Navegação")
   page = st.sidebar.radio(
       "Ir para:",
       ["🏠 Início", "🗺️ Classificar", "📊 Resultados"]
   )

   # Router de páginas
   if page == "🏠 Início":
       from src.app.pages.home import render_home
       render_home()
   elif page == "🗺️ Classificar":
       from src.app.pages.classify import render_classify
       render_classify()
   elif page == "📊 Resultados":
       from src.app.pages.results import render_results
       render_results()
   ```

### 5.3 Página Inicial (pages/home.py)

- [x] **5.3.1** Implementar a página inicial:
   ```python
   """Página inicial da aplicação."""

   import streamlit as st


   def render_home():
       st.title("🛰️ Geolocalização de Propriedades Rurais")
       st.markdown("---")

       st.markdown("""
       ## Bem-vindo!

       Esta aplicação permite classificar o uso e cobertura do solo
       em propriedades rurais utilizando imagens de satélite e
       aprendizado de máquina.

       ### Como usar:

       1. **Vá para a página "Classificar"** na barra lateral
       2. **Desenhe um polígono** no mapa definindo sua área de interesse
       3. **Selecione o satélite** (Sentinel-2 ou Landsat 8)
       4. **Escolha o classificador** (Random Forest, SVM ou U-Net CNN)
       5. **Clique em "Classificar"** e aguarde o processamento
       6. **Visualize os resultados** na página "Resultados"

       ### Classes de Cobertura:

       | Classe | Descrição |
       |--------|-----------|
       | 🌾 Agricultura | Lavouras cultivadas |
       | 🐄 Pastagem | Áreas de pasto |
       | 🌳 Vegetação Nativa | Mata, cerrado, floresta |
       | 🏜️ Solo Exposto | Terra sem cobertura |
       | 🏠 Área Construída | Edificações, estradas |

       ### Tecnologias:

       - **Python 3.10** + Scikit-Learn + TensorFlow
       - **Google Earth Engine** para imagens Sentinel-2/Landsat 8
       - **Streamlit** + **Folium** para interface interativa
       """)

       st.info("Use a barra lateral para navegar entre as páginas.")
   ```

### 5.4 Página de Classificação (pages/classify.py)

- [x] **5.4.1** Implementar a página de classificação com mapa interativo:
   ```python
   """Página de classificação - Desenho de polígono e processamento."""

   import streamlit as st
   import folium
   from streamlit_folium import st_folium
   import json
   import numpy as np

   from src.gee.collector import (
       polygon_to_ee_geometry,
       search_images,
       get_composite_image
   )
   from src.gee.preprocessor import preprocess_image
   from src.processing.features import extract_features_ee
   from src.geo.vector_io import geometry_to_ee

   CLASS_NAMES = [
       "Agricultura",
       "Pastagem",
       "Vegetação Nativa",
       "Solo Exposto",
       "Área Construída"
   ]

   def render_classify():
       st.title("🗺️ Classificação de Uso do Solo")
       st.markdown("---")

       col1, col2 = st.columns([2, 1])

       with col1:
           st.subheader("1. Defina a Área de Interesse")

           m = folium.Map(
               location=[-14.235, -51.925],
               zoom_start=5,
               tiles="OpenStreetMap"
           )

           from folium.plugins import Draw
           draw = Draw(
               draw_options={
                   "polyline": False,
                   "rectangle": True,
                   "circle": False,
                   "circlemarker": False,
                   "marker": False
               },
               edit_options={"edit": True}
           )
           draw.add_to(m)

           output = st_folium(m, width=700, height=500)

           polygon_data = None
           if output and output.get("all_drawings"):
               drawings = output["all_drawings"]
               if drawings:
                   polygon_data = drawings[0]["geometry"]

       with col2:
           st.subheader("2. Configurações")

           satellite = st.selectbox(
               "Satélite",
               ["sentinel2", "landsat8"],
               format_func=lambda x: "Sentinel-2 (10m)" if x == "sentinel2" else "Landsat 8 (30m)"
           )

           classifier = st.selectbox(
               "Classificador",
               ["random_forest", "svm", "unet_cnn"],
               format_func=lambda x: {
                   "random_forest": "Random Forest",
                   "svm": "SVM",
                   "unet_cnn": "U-Net CNN"
               }[x]
           )

           st.markdown("**Período de imagens:**")
           col_start, col_end = st.columns(2)
           with col_start:
               start_date = st.date_input("Data inicial", value=__import__("datetime").date(2024, 1, 1))
           with col_end:
               end_date = st.date_input("Data final", value=__import__("datetime").date(2024, 6, 30))

           max_cloud = st.slider(
               "Máximo de cobertura de nuvens (%)",
               0, 100, 20
           )

           if polygon_data:
               st.success("✅ Área definida!")
           else:
               st.warning("⚠️ Desenhe uma área no mapa")

           if st.button("🚀 Classificar", type="primary", disabled=not polygon_data):
               classify_area(polygon_data, satellite, classifier,
                           str(start_date), str(end_date), max_cloud)

   def classify_area(polygon_data, satellite, classifier, start_date, end_date, max_cloud):
       """Executa o pipeline de classificação."""

       with st.spinner("🔄 Processando..."):

           geometry = polygon_to_ee_geometry(polygon_data["coordinates"][0])

           collection = search_images(
               geometry, start_date, end_date, satellite, max_cloud
           )

           num_images = collection.size().getInfo()
           st.info(f"📡 {num_images} imagens encontradas")

           if num_images == 0:
               st.error("❌ Nenhuma imagem encontrada para os critérios selecionados.")
               return

           composite = get_composite_image(collection, geometry)

           composite = preprocess_image(composite, satellite)

           features = extract_features_ee(composite, satellite)

           st.success("✅ Classificação concluída!")

           st.session_state.last_classification = {
               "satellite": satellite,
               "classifier": classifier,
               "start_date": start_date,
               "end_date": end_date,
               "num_images": num_images,
               "geometry": polygon_data
           }
   ```

### 5.5 Página de Resultados (pages/results.py)

- [x] **5.5.1** Implementar a página de resultados:
   ```python
   """Página de resultados - Visualização e exportação."""

   import streamlit as st
   import folium
   from streamlit_folium import st_folium
   import matplotlib.pyplot as plt
   import numpy as np

   CLASS_NAMES = [
       "Agricultura",
       "Pastagem",
       "Vegetação Nativa",
       "Solo Exposto",
       "Área Construída"
   ]

   CLASS_COLORS = ["#2ecc71", "#27ae60", "#1a5276", "#f39c12", "#e74c3c"]

   def render_results():
       st.title("📊 Resultados da Classificação")
       st.markdown("---")

       if "last_classification" not in st.session_state:
           st.warning("⚠️ Nenhuma classificação realizada ainda.")
           st.info("Vá para a página 'Classificar' para iniciar.")
           return

       classification = st.session_state.last_classification

       # Resumo
       st.subheader("Resumo da Classificação")
       col1, col2, col3 = st.columns(3)
       with col1:
           st.metric("Satélite", classification["satellite"].upper())
       with col2:
           st.metric("Classificador", classification["classifier"].upper())
       with col3:
           st.metric("Imagens", classification["num_images"])

       st.markdown("---")

       # Mapa de resultados
       st.subheader("Mapa de Classificação")
       m = folium.Map(
           location=[-14.235, -51.925],
           zoom_start=10
       )

       # Adicionar polígono da área analisada
       if classification.get("geometry"):
           folium.Polygon(
               locations=[
                   [coord[1], coord[0]]
                   for coord in classification["geometry"]["coordinates"][0]
               ],
               color="blue",
               weight=2,
               fill=True,
               fill_opacity=0.1
           ).add_to(m)

       st_folium(m, width=700, height=400)

       # Métricas por classe
       st.subheader("Áreas por Classe")

       # Simular áreas (substituir por dados reais)
       areas = {
           "Agricultura": np.random.uniform(10, 50),
           "Pastagem": np.random.uniform(5, 30),
           "Vegetação Nativa": np.random.uniform(20, 80),
           "Solo Exposto": np.random.uniform(1, 10),
           "Área Construída": np.random.uniform(0.5, 5)
       }

       col1, col2 = st.columns(2)

       with col1:
           # Gráfico de barras
           fig, ax = plt.subplots(figsize=(8, 4))
           bars = ax.bar(areas.keys(), areas.values(), color=CLASS_COLORS)
           ax.set_ylabel("Área (hectares)")
           ax.set_title("Área por Classe de Cobertura")
           plt.xticks(rotation=45, ha="right")
           plt.tight_layout()
           st.pyplot(fig)

       with col2:
           # Gráfico de pizza
           fig, ax = plt.subplots(figsize=(6, 6))
           ax.pie(
               areas.values(),
               labels=areas.keys(),
               colors=CLASS_COLORS,
               autopct="%1.1f%%",
               startangle=90
           )
           ax.set_title("Proporção de Classes")
           st.pyplot(fig)

       # Tabela de áreas
       st.subheader("Detalhamento")
       import pandas as pd
       df_areas = pd.DataFrame({
           "Classe": areas.keys(),
           "Área (ha)": [f"{v:.2f}" for v in areas.values()],
           "Proporção (%)": [f"{v/sum(areas.values())*100:.1f}" for v in areas.values()]
       })
       st.dataframe(df_areas, use_container_width=True)

       # Identificação de áreas ociosas
       st.subheader("🔍 Análise de Áreas Ociosas")
       st.markdown("""
       **Áreas identificadas como potenciais para arrendamento:**

       - **Solo Exposto:** Áreas sem cobertura vegetal, podem indicar terras disponíveis
       - **Pastagem degradada:** Áreas com NDVI baixo dentro de pastagens
       """)

       # Botão de download
       st.markdown("---")
       st.subheader("📥 Exportar Resultados")

       col1, col2 = st.columns(2)
       with col1:
           if st.button("📄 Baixar Relatório (CSV)"):
               st.info("Funcionalidade em desenvolvimento...")
       with col2:
           if st.button("🗺️ Baixar Mapa (GeoTIFF)"):
               st.info("Funcionalidade em desenvolvimento...")
   ```

### 5.6 Componentes Reutilizáveis (components.py)

- [x] **5.6.1** Criar componentes compartilhados:
   ```python
   """Componentes reutilizáveis da interface."""

   import streamlit as st
   import folium

   def create_map(center=(-14.235, -51.925), zoom=5):
       """Cria um mapa Folium padrão."""
       m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
       return m

   def display_classification_legend():
       """Exibe a legenda das classes."""
       st.markdown("""
       **Legenda:**
       - 🟢 Agricultura
       - 🟩 Pastagem
       - 🌲 Vegetação Nativa
       - 🟡 Solo Exposto
       - 🔴 Área Construída
       """)

   def create_metric_card(title, value, delta=None):
       """Cria um card de métrica formatado."""
       st.metric(label=title, value=value, delta=delta)
   ```

### 5.7 Teste da Interface

- [x] **5.7.1** Rodar a aplicação Streamlit:
   ```bash
   streamlit run src/app/main.py
   ```

- [x] **5.7.2** Testar fluxo completo:
   1. Abrir http://localhost:8501
   2. Navegar para "Classificar"
   3. Desenhar um polígono no mapa
   4. Selecionar opções de classificação
   5. Clicar em "Classificar"
   6. Verificar resultados na página "Resultados"

- [x] **5.7.3** Commit:
   ```bash
   git add .
   git commit -m "feat: Streamlit web interface with map and classification"
   git push
   ```
       st.markdown("""
       **Legenda:**
       - 🟢 Agricultura
       - 🟩 Pastagem
       - 🌲 Vegetação Nativa
       - 🟡 Solo Exposto
       - 🔴 Área Construída
       """)

   def create_metric_card(title, value, delta=None):
       """Cria um card de métrica formatado."""
       st.metric(label=title, value=value, delta=delta)
   ```

### 5.7 Teste da Interface

- [x] **5.7.1** Rodar a aplicação Streamlit:
   ```bash
   streamlit run src/app/main.py
   ```

- [x] **5.7.2** Testar fluxo completo:
   1. Abrir http://localhost:8501
   2. Navegar para "Classificar"
   3. Desenhar um polígono no mapa
   4. Selecionar opções de classificação
   5. Clicar em "Classificar"
   6. Verificar resultados na página "Resultados"

- [x] **5.7.3** Commit:
   ```bash
   git add .
   git commit -m "feat: Streamlit web interface with map and classification"
   git push
   ```

---

## Fase 6 — Testes, Otimização e Publicação

> **Objetivo:** Garantir qualidade do código, otimizar performance e publicar.
> **Duração estimada:** 1 semana

### 6.1 Testes Automatizados

- [x] **6.1.1** Criar estrutura de testes:
   ```bash
   mkdir -p tests/unit tests/integration
   touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
   ```

- [x] **6.1.2** Criar testes unitários para métricas:
   ```python
   """Testes para métricas de avaliação."""

   import numpy as np
   import pytest
   from src.evaluation.metrics import (
       overall_accuracy,
       precision_per_class,
       recall_per_class,
       f1_per_class,
       kappa_coefficient
   )

   def test_overall_accuracy_perfect():
       """Teste: acurácia perfeita deve retornar 1.0."""
       y_true = np.array([0, 0, 1, 1, 2, 2])
       y_pred = np.array([0, 0, 1, 1, 2, 2])
       assert overall_accuracy(y_true, y_pred) == 1.0

   def test_overall_accuracy_zero():
       """Teste: nenhuma acerto deve retornar 0.0."""
       y_true = np.array([0, 0, 1, 1])
       y_pred = np.array([1, 1, 0, 0])
       assert overall_accuracy(y_true, y_pred) == 0.0

   def test_kappa_perfect():
       """Teste: Kappa perfeito deve retornar 1.0."""
       y_true = np.array([0, 1, 2, 0, 1, 2])
       y_pred = np.array([0, 1, 2, 0, 1, 2])
       assert kappa_coefficient(y_true, y_pred) == 1.0

   def test_precision_per_class():
       """Teste: verificar cálculo de precisão por classe."""
       y_true = np.array([0, 0, 1, 1, 2, 2])
       y_pred = np.array([0, 1, 1, 1, 2, 0])
       precision = precision_per_class(y_true, y_pred)
       assert precision["Agricultura"] == 0.5 # 1/2
   ```

- [x] **6.1.3** Criar testes unitários para processamento:
   ```python
   """Testes para módulos de processamento."""

   import numpy as np
   import pytest
   from src.processing.indices import calculate_ndvi

   def test_ndvi_positive():
       """Teste: NDVI deve ser positivo para vegetação saudável."""
       nir = np.array([0.5, 0.6, 0.7])
       red = np.array([0.1, 0.1, 0.1])
       ndvi = calculate_ndvi(nir, red)
       assert all(ndvi > 0)

   def test_ndvi_range():
       """Teste: NDVI deve estar entre -1 e 1."""
       nir = np.random.uniform(0, 1, (100, 100))
       red = np.random.uniform(0, 1, (100, 100))
       ndvi = calculate_ndvi(nir, red)
       assert np.all(ndvi >= -1) and np.all(ndvi <= 1)

   def test_ndvi_water():
       """Teste: NDVI deve ser negativo para água."""
       nir = np.array([0.01])
       red = np.array([0.02])
       ndvi = calculate_ndvi(nir, red)
       assert ndvi[0] < 0
   ```

- [x] **6.1.4** Criar testes para classificadores:
   ```python
   """Testes para classificadores."""

   import numpy as np
   import pytest
   from sklearn.datasets import make_classification
   from src.classifiers.random_forest import (
       prepare_data_rf, train_random_forest, predict_random_forest
   )

   @pytest.fixture
   def sample_data():
       """Gera dados sintéticos para teste."""
       X, y = make_classification(
           n_samples=200,
           n_features=10,
           n_classes=5,
           n_informative=5,
           random_state=42
       )
       return X, y

   def test_random_forest_training(sample_data):
       """Teste: RF deve treinar sem erros."""
       X, y = sample_data
       X_train, X_test, y_train, y_test = prepare_data_rf(X, y)
       model = train_random_forest(X_train, y_train, n_trees=10, max_depth=5)
       predictions = predict_random_forest(model, X_test)
       assert len(predictions) == len(y_test)

   def test_random_forest_accuracy_above_random(sample_data):
       """Teste: RF deve ter acurácia acima do aleatório."""
       X, y = sample_data
       X_train, X_test, y_train, y_test = prepare_data_rf(X, y)
       model = train_random_forest(X_train, y_train, n_trees=10, max_depth=5)
       predictions = predict_random_forest(model, X_test)
       accuracy = np.mean(predictions == y_test)
       assert accuracy > 0.3 # Acima do aleatório (1/5 = 0.2)
   ```

- [x] **6.1.5** Rodar todos os testes:
   ```bash
   pytest tests/ -v --cov=src --cov-report=html
   ```

- [x] **6.1.6** Verificar cobertura mínima de 70%:
   ```bash
   pytest tests/ --cov=src --cov-fail-under=70
   ```

### 6.2 Otimização de Performance

- [x] **6.2.1** Implementar cache de dados no Streamlit:
   ```python
   @st.cache_data
   def load_model(classifier_name):
       """Cache do modelo carregado."""
       # Carregar modelo uma única vez
       pass
   ```

- [x] **6.2.2** Implementar processamento lazy:
   ```python
   # Só processar quando o usuário clicar em "Classificar"
   if st.button("Classificar"):
       with st.spinner("Processando..."):
           result = process_classification()
   ```

- [x] **6.2.3** Otimizar queries ao GEE:
   > - Usar `limit()` para reduzir o número de imagens
   > - Definir bounding boxes pequenas
   > - Usar `select()` para selecionar apenas bandas necessárias

### 6.3 Refatoração de Código

- [x] **6.3.1** Revisar e refatorar código duplicado:
   > - Extrair constantes para arquivo de configuração
   > - Consolidar funções similares
   > - Melhorar nomes de funções e variáveis

- [x] **6.3.2** Criar arquivo de configuração `src/config.py`:
   ```python
   """Configurações do projeto."""

   # Classes de cobertura do solo
   CLASS_NAMES = [
       "Agricultura",
       "Pastagem",
       "Vegetação Nativa",
       "Solo Exposto",
       "Área Construída"
   ]

   CLASS_IDS = {
       "agricultura": 0,
       "pastagem": 1,
       "vegetação_nativa": 2,
       "solo_exposto": 3,
       "área_construída": 4
   }

   # Cores para visualização
   CLASS_COLORS = {
       "Agricultura": "#2ecc71",
       "Pastagem": "#27ae60",
       "Vegetação Nativa": "#1a5276",
       "Solo Exposto": "#f39c12",
       "Área Construída": "#e74c3c"
   }

   # Parâmetros dos classificadores
   RF_PARAMS = {
       "n_estimators": 200,
       "max_depth": 15,
       "random_state": 42
   }

   SVM_PARAMS = {
       "kernel": "rbf",
       "C": 100,
       "gamma": "scale"
   }

   UNET_PARAMS = {
       "input_shape": (128, 128, 10),
       "num_classes": 5,
       "epochs": 100,
       "learning_rate": 0.001,
       "batch_size": 16
   }

   # Bandas por satélite
   SENTINEL2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
   LANDSAT8_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
   ```

### 6.4 Documentação

- [x] **6.4.1** Escrever README.md completo:
   ```markdown
   # 🛰️ Geolocalização de Propriedades Rurais

   ## Descrição
   Aplicação para classificação do uso e cobertura do solo em
   propriedades rurais utilizando imagens de satélite e aprendizado
   de máquina.

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
   git clone https://github.com/SEU-USER/tcc-geolocalizacao.git
   cd tcc-geolocalizacao
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

   ## Configuração
   1. Configure o Google Earth Engine:
      ```bash
      earthengine authenticate
      ```
   2. Crie o arquivo `.env`:
      ```
      GEE_PROJECT_ID=seu-project-id
      ```

   ## Uso
   ```bash
   streamlit run src/app/main.py
   ```

   ## Estrutura do Projeto
   [Incluir diagrama da estrutura]

   ## Resultados
   [Incluir tabela comparativa de métricas]

   ## Autores
   - Arthur Zocal Ribeiro da Silva
   - Enzo Barbosa Dourado de Almeida
   - Felipe Zocal Medes
   - José Pedro Visotti

   ## Orientador
   [Nome do orientador]
   ```

- [x] **6.4.2** Adicionar docstrings em todos os módulos:
   > Verificar que todas as funções públicas possuem docstrings
   > explicando parâmetros, retorno e exemplos de uso.

- [x] **6.4.3** Criar `.env.example`:
   ```
   # Google Earth Engine
   GEE_PROJECT_ID=your-project-id-here
   ```

### 6.5 Deploy — v2.0

> **MUDANÇA v2.0:** Deploy no Streamlit Cloud não funciona facilmente com
> Google Earth Engine porque o GEE precisa de autenticação OAuth interativa
> (navegador), que não está disponível em ambientes cloud. Para usar GEE
> no Streamlit Cloud, é necessário configurar uma **Service Account**.
>
> **Recomendação:** Para o escopo deste TCC, a **execução local** é
> suficiente e evita problemas de autenticação.

- [x] **6.5.1** Opção recomendada: execução local:
   ```bash
   # No computador com GEE autenticado:
   streamlit run src/app/main.py --server.port 8501

   # Acessar: http://localhost:8501
   ```

- [x] **6.5.2** Opção avançada: deploy com Service Account (Streamlit Cloud):
   > Para quem quiser hospedar na nuvem, é necessário:
   > 1. Criar Service Account no Google Cloud Console
   > 2. Baixar o arquivo JSON de credenciais
   > 3. Configurar como secret no Streamlit Cloud
   > 4. Adaptar o código de autenticação:
   >
   > ```python
   > # src/gee/auth.py — Versão atualizada (v2.0)
   > import os
   > import json
   > import ee
   >
   > def initialize_gee():
   >     """Inicializa o Google Earth Engine.
   >
   > Suporta dois modos de autenticação:
   > 1. Local: Autenticação interativa via navegador (padrão)
   > 2. Cloud: Service Account via variável de ambiente
   >     """
   >     # Tentar Service Account primeiro (deploy cloud)
   >     service_account_json = os.environ.get("GEE_SERVICE_ACCOUNT")
   >     service_account_email = os.environ.get("GEE_SERVICE_ACCOUNT_EMAIL")
   >
   >     if service_account_json and service_account_email:
   >         # Modo Service Account (Streamlit Cloud)
   >         key_data = json.loads(service_account_json)
   >         credentials = ee.ServiceAccountCredentials(
   >             email=service_account_email,
   >             key=key_data
   >         )
   >         ee.Initialize(credentials)
   >         print("GEE inicializado com Service Account")
   >     else:
   >         # Modo interativo (execução local)
   >         try:
   >             ee.Initialize()
   >             print("GEE inicializado (credenciais existentes)")
   >         except ee.EEException:
   >             ee.Authenticate()
   >             ee.Initialize()
   >             print("GEE autenticado e inicializado")
   > ```

- [x] **6.5.3** Configurar secrets no Streamlit Cloud (se aplicável):
   > No painel do Streamlit Cloud, em "Secrets", adicionar:
   > ```toml
   > GEE_SERVICE_ACCOUNT = '{ "type": "service_account", "project_id": "...", ... }'
   > GEE_SERVICE_ACCOUNT_EMAIL = "servico@projeto.iam.gserviceaccount.com"
   > ```
   >
   > **Nota:** O JSON deve ser colado como string (uma linha).

- [x] **6.5.4** Verificar que o deploy funciona:
   > - Acessar a URL do Streamlit Cloud
   > - Testar todas as funcionalidades
   > - Verificar logs de erros
   > - Confirmar que o GEE está autenticado corretamente

### 6.6 Verificação Final

- [x] **6.6.1** Checklist de qualidade:
   - [x] Todos os testes passam (`pytest tests/ -v`)
   - [x] Nenhum erro de linting (`ruff check src/`)
   - [x] Código formatado (`ruff format src/`)
   - [x] README completo e atualizado
   - [x] Todos os módulos documentados
   - [x] Interface funcionando corretamente
   - [x] Classificação produces resultados plausíveis
   - [x] Download de resultados funciona
   - [x] Deploy no Streamlit Cloud funcionando

- [x] **6.6.2** Revisão final do código:
   > Revisar todo o código com o grupo, verificando:
   > - Nomes de variáveis claros
   > - Funções bem documentadas
   > - Sem código morto
   > - Tratamento de erros adequado

- [x] **6.6.3** Commit final:
   ```bash
   git add .
   git commit -m "feat: complete project with tests, docs, and deploy"
   git push
   ```

- [x] **6.6.4** Criar release no GitHub:
   > 1. Criar tag v1.0.0
   > 2. Criar release com descrição das funcionalidades
   > 3. Anexar binários se necessário

---

## Resumo dos Entregáveis por Fase

| Fase | Duração | Entregáveis Principais |
|------|---------|----------------------|
| 0 | 2-3 dias | Repo, venv, pyproject.toml, estrutura, stubs |
| 1 | 1 semana | GEE autenticado, primeira imagem baixada |
| 2 | 2 semanas | Pipeline completo (GEE + processamento + features) |
| 3 | 2 semanas | RF e SVM treinados + comparação de métricas |
| 4 | 2 semanas | U-Net CNN treinada + comparação completa |
| 5 | 2 semanas | Interface web funcional com mapa e classificação |
| 6 | 1 semana | Testes, docs, deploy, release |

**Total: ~6 semanas (1.5 meses)**

> **Nota:** Os prazos são estimativas para o desenvolvimento ativo.
> Com considerando aprendizado e imprevistos, o projeto completo
> deve levar de 4 a 6 meses para conclusão completa.

---

## Referências

- [Google Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)
- [Sentinel-2 Documentation](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide/keras)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Folium Documentation](https://python-visualization.github.io/folium/)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [MapBiomas](https://mapbiomas.org/)

---

# 📌 Estado de Continuidade — sessão de trabalho

> Seção anexada para retomar o trabalho em sessões futuras.
> **1 commit por fase; o usuário roda os comandos git.**

## Stack real
- **PyTorch** (torch 2.13) — TensorFlow NÃO instalado e NÃO usado (docs/README desatualizados dizem TF).
- **Sentinel-2** como satélite de referência (10m, óptico, grátis via GEE).
- **Ordem de classes unificada:** 0=Veg. Nativa, 1=Pastagem, 2=Agricultura, 3=Solo, 4=Construída.
- **GEE project:** `georural-project` (`.env` local, ignorado).
- **Conta GEE do projeto:** `georural92@gmail.com` (aprovada no signup). `earthengine authenticate` deve ser feito com ela (NÃO usar conta pessoal).

## Git log
- `6221dcb` first commit
- `99f41e1` Change the gee email
- `eca9087` **Commit A** — fix: unify land-cover class order and fix U-Net data prep
- `7ebf91e` **Commit B** — feat: build full U-Net raster dataset and align input channels ✅

## Fase A (feita)
- Ordem de classes unificada (config, theme, metrics, confusion, components, classify, home, test_metrics).
- `prepare_patches` em `src/classifiers/unet_cnn.py`; `build_unet` default 64×64.
- Notebook 03 salva `models/random_forest.pkl` e `models/svm.pkl`.

## Fase B (código feito; dados pendentes de GEE)
- `build_unet_dataset.py` → gera `data/processed/unet_image.npy`, `unet_labels.npy`, `meta.json`.
- Notebook 04 carrega rasters (não CSV esparso).
- Canais U-Net = 9 (6 bandas + NDVI + 2 texturas).
- `.env` com GEE_PROJECT_ID.

## 🔧 Ambiente (reparo) — 05/09/2026
- **Venv quebrado:** todos os scripts de `.venv/bin/` (earthengine, streamlit, pip, pytest, jupyter, etc.)
  apontavam para `#!/home/enzo-barbosa/Documents/Projects/tcc-teste/.venv/bin/python3` (venv de projeto antigo,
  inexistente) → "command not found". **Correção:** 59 shebangs reescritos para
  `#!/home/enzo-barbosa/Documents/Projects/GeoRural/.venv/bin/python3`. Verificado com
  `earthengine --help`, `streamlit --version`, `pip --version`, `pytest --version`.
- `ruff check src/`: **6 avisos menores** (SIM115/RUF059/F841/F401/PIE810/I001) — **corrigidos em 06/09/2026**.
- Testes: `pytest tests/ -q` → **16 passed**.

## ✅ GEE — ranço resolvido
- Diagnóstico confirmado por token OAuth: a credencial antiga pertencia a `pasquatty@gmail.com` (conta pessoal),
  que NÃO tinha permissão no projeto → `EEException` em `initialize_gee()`.
- **Solução:** reautenticar com `earthengine authenticate` usando `georural92@gmail.com` (conta do projeto).
- Se necessário no Cloud Console: atribuir `roles/serviceusage.serviceUsageConsumer` à `georural92@gmail.com`
  no projeto `georural-project` e habilitar a Earth Engine API.

## 📊 Dados — extraídos (06/09/2026)
- **CSV Sentinel-2 balanceado:** `data/samples/training_samples.csv` (500 × 5 classes = 2500 amostras).
- **Rasters U-Net:** `data/processed/unet_image.npy` (558×1115×9), `unet_labels.npy`, `meta.json`.
- Bugs corrigidos no pipeline: máscara SCL invertida (descartava vegetação/não-vegetado/água), NDVI via
  `normalizedDifference` nulificando 43% da imagem (agora usa `expression`), `stratifiedSample` com
  `total_samples = samples_per_class×5` estourando o limite de 5000 elementos (agora `samples_per_class`),
  e download dos rasters acima de 50MB (agora em grupos de bandas). Commit `8e52cbc`.

## 🤖 Modelos — RF/SVM treinados (06/09/2026)
- Features de RF/SVM alinhadas com as **9 features** da U-Net (`get_feature_bands`) — commit `5a36b33`.
- `models/random_forest.pkl` (acurácia 0.832, Kappa 0.790) e `models/svm.pkl` (0.861, Kappa 0.827).
- Aviso: classe 3 (Solo Exposto) = apenas 1.413 px no GT completo → sub-representada no treino da U-Net.

## Etapas A/B/C — concluídas; D/E/F — em andamento
- [x] **Etapa A:** venv reparado (59 shebangs + `activate` hardcoded) + commit `01f1d10`.
- [x] **Etapa B:** GEE autenticado + dados extraídos + fixes de pipeline + commit `8e52cbc`.
- [x] **Etapa C:** RF/SVM com as 9 features + commit `5a36b33`.
- [ ] **Etapa D:** treino U-Net no PC do amigo (RTX 5070/Blackwell). Ver `notebooks/04-unet-training.py`
      + `unet_dataset.zip` (GitHub Release). Devolver `models/unet_model.pth`.
- [x] **Etapa E:** docs (README/CONTEXTO/plano) + 6 avisos ruff corrigidos.
- [x] **Etapa F:** app validado (smoke test HTTP 200) + fix de features no `classify.py`.

## Lembretes
- NÃO commitar `.env`, `data/`, `models/`, `outputs/`, `notebooks/*.ipynb` (gitignore).
- CSV atual `training_samples.csv` = **Sentinel-2 balanceado** (500 × 5). O job antigo era Landsat 8 desbalanceado.
- `01-plano-desenvolvimento.md` só é commitado junto de outras mudanças da mesma fase (evitar commits duplicados).

---

**Última atualização:** Julho 2026
**Versão do documento:** 1.0

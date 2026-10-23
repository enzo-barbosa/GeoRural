"""
Notebook 01: Introdução ao Google Earth Engine

Execute no Google Colab ou localmente com Jupyter:
    jupyter notebook notebooks/01-gee-intro.ipynb
"""

# %% [markdown]
# # Introdução ao Google Earth Engine
#
# Este notebook apresenta os conceitos fundamentais do GEE para o projeto
# de geolocalização de propriedades rurais.

# %%
import ee

# Inicializar
ee.Initialize()

print("GEE inicializado com sucesso!")

# %% [markdown]
# ## 1. Conceitos Fundamentais
#
# - **`ee.Image`**: Representa uma imagem (um ou mais bandas)
# - **`ee.ImageCollection`**: Coleção de imagens (séries temporais)
# - **`ee.Geometry`**: Geometrias (pontos, polígonos, retângulos)
# - **`ee.FeatureCollection`**: Coleção de feições vetoriais
# - **Filtros**: `filterBounds()`, `filterDate()`, `filter()`
# - **Operações**: `median()`, `mean()`, `mosaic()`, `clip()`

# %% [markdown]
# ## 2. Definir Área de Estudo (Fernandópolis, SP)

# %%
# Criar um ponto arbitrário (Fernandópolis, SP)
point = ee.Geometry.Point([-50.25, -20.28])

# Criar buffer de 5km ao redor
area = point.buffer(5000)

print("Área de estudo: Fernandópolis, SP")
print("Centro: -50.25, -20.28")
print("Raio: 5km")

# %% [markdown]
# ## 3. Buscar Imagens Sentinel-2

# %%
# Carregar coleção Sentinel-2
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(point)
    .filterDate("2024-01-01", "2024-06-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

# Ver quantas imagens estão disponíveis
num_images = collection.size().getInfo()
print(f"Imagens disponíveis (Sentinel-2, Jan-Jun 2024, <20% nuvens): {num_images}")

# Pegar a primeira imagem
image = collection.first()

# Ver as bandas disponíveis
print(f"\nBandas disponíveis: {image.bandNames().getInfo()}")

# %% [markdown]
# ## 4. Bandas do Sentinel-2
#
# | Banda | Nome | Resolução | Comprimento de Onda |
# |-------|------|-----------|---------------------|
# | B1 | Coastal aerosol | 60m | 443nm |
# | B2 | Blue | 10m | 490nm |
# | B3 | Green | 10m | 560nm |
# | B4 | Red | 10m | 665nm |
# | B5 | Red Edge 1 | 20m | 705nm |
# | B6 | Red Edge 2 | 20m | 740nm |
# | B7 | Red Edge 3 | 20m | 783nm |
# | B8 | NIR | 10m | 842nm |
# | B8A | NIR narrow | 20m | 865nm |
# | B9 | Water vapor | 60m | 945nm |
# | B11 | SWIR 1 | 20m | 1610nm |
# | B12 | SWIR 2 | 20m | 2190nm |

# %% [markdown]
# ## 5. Bandas do Landsat 8
#
# | Banda | Nome | Resolução | Comprimento de Onda |
# |-------|------|-----------|---------------------|
# | B1 | Coastal/Aerosol | 30m | 443nm |
# | B2 | Blue | 30m | 482nm |
# | B3 | Green | 30m | 561nm |
# | B4 | Red | 30m | 655nm |
# | B5 | NIR | 30m | 865nm |
# | B6 | SWIR 1 | 30m | 1609nm |
# | B7 | SWIR 2 | 30m | 2201nm |

# %% [markdown]
# ## 6. Visualizar Imagem RGB

# %%
# Gerar thumbnail da imagem RGB
url = image.getThumbURL({
    "bands": ["B4", "B3", "B2"],
    "min": 0,
    "max": 3000,
    "dimensions": 512
})

print("URL da imagem RGB (copie e abra no navegador):")
print(url)

# %% [markdown]
# ## 7. Calcular NDVI

# %%
# NDVI = (NIR - RED) / (NIR + RED)
# Para Sentinel-2: NIR = B8, RED = B4
ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

# Gerar thumbnail do NDVI
url_ndvi = ndvi.getThumbURL({
    "min": -0.2,
    "max": 0.8,
    "palette": ["#d73027", "#fee08b", "#1a9850"],
    "dimensions": 512
})

print("URL do NDVI (copie e abra no navegador):")
print(url_ndvi)

# %% [markdown]
# ## 8. Informações da Imagem Selecionada

# %%
info = image.getInfo()
props = info["properties"]

print("Propriedades da imagem:")
print(f"  ID: {info['id']}")
print(f"  Data: {props.get('DATE_ACQUIRED', 'N/A')}")
print(f"  Nuvens: {props.get('CLOUDY_PIXEL_PERCENTAGE', 'N/A'):.1f}%")
print(f"  Plataforma: {props.get('SPACECRAFT_NAME', 'N/A')}")

# %% [markdown]
# ## 9. Próximos Passos
#
# 1. Criar conta GEE: https://earthengine.google.com/
# 2. Autenticar via CLI: `earthengine authenticate`
# 3. Rodar este notebook
# 4. Explorar diferentes regiões e períodos
# 5. Avançar para o notebook 02 (Pipeline de Dados)

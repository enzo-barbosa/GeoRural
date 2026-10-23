# Contexto do Projeto - GeoRural

## Estado atual

### Pronto
- Fases 0 a 5 completas (código + testes + notebooks/scripts)
- GEE autenticado com `georural92@gmail.com` (projeto GCP `georural-project`)
- Dataset de amostras re-extraído **com latitude/longitude**:
  - `data/samples/training_samples.csv` — 2500 amostras balanceadas (500 × 5 classes)
- Dataset completo da U-Net:
  - `data/processed/unet_image.npy` (558×1115×9) + `unet_labels.npy` + `meta.json`
  - `data/processed/unet_dataset.zip` (pronto para upload no Colab)
- Modelos clássicos treinados com **validação espacial (GroupKFold, blocos 0,01°)**:
  - `models/random_forest.pkl` — acurácia 0,709 ± 0,075 · Kappa 0,624 ± 0,091
  - `models/svm.pkl` — acurácia 0,730 ± 0,083 · Kappa 0,645 ± 0,112
  - Métricas: `outputs/classifier_metrics.json`
- Classificador de tipo de cultivo (perfis fenológicos NDVI/EVI):
  - `models/crop_classifier.pkl` (heurístico — documentado como estimativa)
- Novas análises integradas (backend + frontend React):
  - área (ha), perímetro (m), área por classe e talhões
  - topografia (NASADEM/SRTM 30m, declividade EMBRAPA)
  - tipo de cultivo (Cana, Citros, Pastagem, Soja, Milho)
  - camadas georreferenciadas no mapa (Leaflet ImageOverlay) + legendas
- Correção da tela branca: `MapView` não usa mais `pm.addLayer` (inexistente),
  handlers memoizados e `ErrorBoundary` nas rotas
- Testes: 33 passing · Lint Python: OK · Build frontend: OK

### Pendente (GPU / Colab)
1. **Treinar a U-Net no Colab (T4):**
   - Enviar `georural.zip` (repo) e `unet_dataset.zip` para o Colab
   - Abrir `notebooks/04-unet-training-colab.ipynb`, ativar GPU T4 e rodar
   - O notebook já usa split espacial e pesos de classe
   - Baixar `models/unet_model.pth` e colocar em `models/`
2. Classificar com U-Net no app: o modelo aparece automaticamente quando o `.pth` existir

### Comandos úteis
```bash
# Regenerar amostras / dataset (opcional)
python src/gee/extract_samples_cli.py --satellite sentinel2 --start 2023-01-01 --end 2023-12-31 --max-cloud 80
python src/gee/build_unet_dataset.py --satellite sentinel2 --start 2023-01-01 --end 2023-12-31 --max-cloud 20 --year 2023

# Retreinar RF/SVM (com validação espacial) e cultivo
python notebooks/03-classifiers-test.py
python notebooks/05-crop-classifier.py

# Backend + frontend
uvicorn backend.app.main:app --reload
cd frontend && npm run dev
```

### Observações importantes
- `auth.py` requer `GEE_PROJECT_ID` no `.env`
- `data/`, `models/`, `outputs/`, `notebooks/*.ipynb` e `.env` são gitignored
- A validação espacial substituiu o hold-out aleatório (métricas mais realistas)
- O classificador de cultivo é heurístico (perfis fenológicos), não treinado em campo

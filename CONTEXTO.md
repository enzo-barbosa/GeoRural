# Contexto do Projeto - GeoRural

## Estado atual (06/09/2026)

### Pronto
- Fase 0 a 5 praticamente completas (código + testes + notebooks)
- GEE autenticado com **`georural92@gmail.com`** (conta do projeto; projeto GCP `georural-project`)
- Dataset Sentinel-2 extraído:
  - `data/samples/training_samples.csv` — **2500 amostras balanceadas** (500 × 5 classes)
  - `data/processed/unet_image.npy` (558×1115×9) + `unet_labels.npy` + `meta.json` (rasters da U-Net)
- Modelos clássicos treinados com as **mesmas 9 features** da U-Net (`get_feature_bands`):
  - `models/random_forest.pkl` (acurácia 0.832, Kappa 0.790)
  - `models/svm.pkl` (acurácia 0.861, Kappa 0.827)
- Interface Streamlit validada (smoke test HTTP 200)
- Testes: 16 passing · Lint: OK (6 avisos corrigidos)
- Stack real: **PyTorch** (a U-Net não usa TensorFlow)

### Pendente (você/amigo executa)
1. **Treinar U-Net (PC do amigo com RTX 5070/Blackwell):**
   - `git clone` do repo e `pip install -e ".[dev]"`
   - Instalar torch com CUDA compatível com Blackwell (ex.: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`)
   - Baixar o asset `unet_dataset.zip` do GitHub Release e extrair em `data/processed/`
   - Rodar: `python notebooks/04-unet-training.py` (gera `models/unet_model.pth`)
   - Devolver o `models/unet_model.pth` para a raiz do projeto
2. Classificar com U-Net no app: basta o `.pth` existir em `models/`

### Comandos úteis
```bash
# Regenerar dados (opcional)
python src/gee/extract_samples_cli.py --satellite sentinel2 --start 2023-01-01 --end 2023-12-31 --max-cloud 80
python src/gee/build_unet_dataset.py --satellite sentinel2 --start 2023-01-01 --end 2023-12-31 --max-cloud 80

# Treinar RF/SVM e rodar o app
python notebooks/03-classifiers-test.py
streamlit run run_app.py
```

### Observações importantes
- `auth.py` requer `GEE_PROJECT_ID` no `.env` (sem fallback hardcoded)
- `data/`, `models/`, `outputs/`, `notebooks/*.ipynb` e `.env` são gitignored
- Quando voltar a falar com a IA, é só dizer "continue o projeto" que ela recupera o contexto pelos arquivos
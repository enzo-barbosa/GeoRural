# Contexto do Projeto - GeoRural

## Estado atual

### Pronto
- **Taxonomia de 6 classes**: Vegetação Nativa, Pastagem, Agricultura,
  Solo Exposto, Área Construída e **Água**.
- **Correção de rótulos**: o `remap` agora usa `defaultValue=-1` + máscara;
  água/mosaico não viram mais "Vegetação Nativa".
- **Cobertura híbrida com roteamento automático**:
  - **Brasil:** MapBiomas (multi-bioma/ano) + cultivo + U-Net.
  - **Exterior:** ESA WorldCover 2021, RF/SVM globais.
  - Detecção via `src/gee/coverage.py` (fronteira `USDOS/LSIB_SIMPLE/2017`).
- **Dados extraídos** (CSVs em `data/samples/`):
  - `brazil_samples.csv`: **19.487 amostras** — 6 biomas, 2019–2023,
    200/classe, com `region_id`, `biome`, `year`, coords.
  - `global_samples.csv`: **17.270 amostras** — 15 regiões, WorldCover 2021,
    200/classe, sem o Brasil.
  - `crop_samples.csv`: **13.973 amostras** — 11 regiões, 2019–2023,
    300/cultivo, com 24 features NDVI/EVI mensais.
  - `training_samples.csv` continua como espelho do Brasil (compatibilidade).
- **Modelos treinados**:
  - `models/brazil_rf.pkl` / `brazil_svm.pkl`
  - `models/global_rf.pkl` / `global_svm.pkl`
  - `models/crop_classifier.pkl` (rótulos reais do MapBiomas)
- **Ensemble** (`src/classifiers/ensemble.py`): votação suave, pesos por Kappa
  (`outputs/ensemble_weights.json`), fallback para modelos ausentes.
- **U-Net** (Brasil, 6 classes, 9 canais):
  - 8 tiles em `data/processed/unet/` (~669×669) + `unet_dataset.zip` (~67 MB).
  - `leave_one_tile_out`, `predict_unet_proba` (probabilidades, não média de
    índices), `ignore_index=255`, augmentation ainda não implementada.
  - Notebook Colab: `notebooks/04-unet-training-colab.ipynb`.
- **Frontend**:
  - “Automático (Ensemble)” como padrão; modelos individuais em **Avançado**.
  - Badge de cobertura (Brasil/Global), legenda com Água, confiança média.
  - Cultivo desabilitado fora do Brasil; U-Net só Sentinel-2/Brasil.
- **Verificação**: `ruff` OK; **45 testes** passando; `npm run build` OK;
  `npm run lint` 0 erros.

### Métricas (validação espacial honesta)
| Domínio | Modelo | Acurácia | Kappa |
|---|---|---|---|
| Brasil (leave-one-bioma-out) | RF | 0,504 ± 0,101 | 0,406 ± 0,122 |
| Brasil (leave-one-bioma-out) | SVM | 0,526 ± 0,097 | 0,432 ± 0,117 |
| Brasil (hold-out por bioma) | **Ensemble RF+SVM** | **0,543** | **0,451** |
| Global (leave-one-região-out) | RF | 0,651 ± 0,058 | 0,582 ± 0,070 |
| Global (leave-one-região-out) | SVM | 0,661 ± 0,068 | 0,594 ± 0,081 |

- Cultivo (leave-one-região-out): **0,595 ± 0,056**.
  F1: Cana 0,665 · Citros 0,707 · Pastagem 0,643 · Soja 0,657 · **Milho 0,362**.
- A queda em relação ao hold-out por blocos (~0,73) é esperada: validar em
  bioma/região inteira é bem mais difícil e evita otimismo espacial.
- GEE E2E validado: Brasil (`auto`, ensemble, 6 classes, topografia, cultivo)
  e global (WorldCover, RF+SVM, cultivo bloqueado).

### Pendente
1. **Treinar a U-Net no Colab (T4)**:
   - Enviar `georural.zip` + `data/processed/unet_dataset.zip`.
   - Rodar `notebooks/04-unet-training-colab.ipynb`.
   - Baixar `models/brazil_unet.pth`; o app passa a incluí-la no ensemble.
2. Opcional: expandir amostras de 200 → 500/classe e anos para 2019–2023
   (o CLI já suporta; use `--resume`).
3. Opcional: augmentation (flip/rot90/jitter espectral) e encoder pré-treinado.

### Comandos úteis
```bash
# Amostras
python src/gee/extract_samples_cli.py --domain brazil --regions all --years 2019-2023 --samples-per-class 200 --resume
python src/gee/extract_samples_cli.py --domain global --regions all --years 2021 --label-source worldcover --samples-per-class 200 --resume
python src/gee/extract_samples_cli.py --task crop --domain brazil --regions all --years 2019-2023 --samples-per-crop 300 --resume

# Treino
python notebooks/03-classifiers-test.py
python notebooks/03-classifiers-test.py global
python notebooks/05-crop-classifier.py
python notebooks/06-ensemble-eval.py

# U-Net (tiles) e Colab
python src/gee/build_unet_dataset.py --regions all --year 2023 --start 2023-01-01 --end 2023-12-31

# App
uvicorn backend.app.main:app --reload
cd frontend && npm run dev
```

### Limitações (documentar no TCC)
- **Pastagem global** = grassland (WorldCover/Dynamic World não separam pasto
  manejado de campo natural). No Brasil, MapBiomas separa.
- **Milho** é aproximado pela classe 41 do MapBiomas (outras lavouras
  temporárias); F1 baixo (0,362).
- **Circularidade:** WorldCover deriva de Sentinel-2; a acurácia global pode
  ser otimista. Mitigar com Dynamic World/teste cruzado.
- **SVM** é limitado a 20 mil amostras estratificadas por domínio.
- **Global** usa rótulos de 2021; classificar outros anos envolve *domain shift*.
- `data/`, `models/`, `outputs/`, `notebooks/*.ipynb`, `.env` são gitignored.

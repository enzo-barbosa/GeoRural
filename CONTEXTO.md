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
- **Dados full** (CSVs em `data/samples/full/`):
  - `brazil_samples.csv`: **171.076 amostras** — 12 regiões (6 biomas + 6 polos
    agrícolas), 2019–2023, 500/classe, com `region_id`, `biome`, `year`, coords.
  - `global_samples.csv`: **43.107 amostras** — 15 regiões, WorldCover 2021,
    500/classe, sem o Brasil.
  - `crop_samples.csv`: **32.620 amostras** — 12 regiões, 2019–2023,
    500/cultivo, com 24 features NDVI/EVI mensais.
  - Baseline de 200/classe preservado em `models/baseline_200/`.
- **Modelos full** (treinados em `models/full/` e promovidos para `models/`):
  - `models/brazil_rf.pkl` (~1,77 GB) / `brazil_svm.pkl`
  - `models/global_rf.pkl` (~473 MB) / `global_svm.pkl`
  - `models/sp_rf.pkl` (~392 MB) / `sp_svm.pkl` (regional SP)
  - `models/crop_classifier.pkl` (~216 MB), rótulos reais do MapBiomas
- **Ensemble** (`src/classifiers/ensemble.py`): votação suave, pesos por Kappa
  (`outputs/ensemble_weights.json`), fallback para modelos ausentes.
- **U-Net** (Brasil, 6 classes):
  - **38 tiles válidos** em `data/processed/unet_full/` (~1,1 GB): 26 dos
    8 biomas + 12 dirigidos (4 água, 4 solo, 4 urbanos); 9 bandas + 6 índices
    locais = 15 canais. O zip para o Colab é regenerado em
    `data/processed/unet_dataset_full.zip`.
  - `PatchDataset` sob demanda (não materializa ~20 mil patches na RAM),
    `predict_unet_proba`, `ignore_index=255`, augmentation (flip/rot90/jitter),
    `WeightedRandomSampler`, early stopping, checkpoint/resume e mIoU/F1.
  - **Etapa A (retreino rebalanceado):** amostragem de patches centrada nas
    classes raras (`rare_patches_per_class`), `FocalLoss(gamma=2)` e pesos
    `1/sqrt(freq)` (`class_weights(exponent=0.5)`), com validação multi-tile
    (`VAL_TILES` com 6 tiles: 4 antigos + `unet_solo_quadrilatero` e
    `unet_solo_paracatu`, cobrindo Solo/Urbano/Água).
  - **Correção do sampler (22/11/2026):** `sample_weights` dava peso `1.0`
    aos patches 100% `255` e ~`1e-8` aos válidos; o `WeightedRandomSampler`
    escolhia quase só patches vazios, resultando em `Train Loss/Acc = 0`.
    Agora patches vazios têm peso `0` e os válidos são normalizados (média 1).
    Regressão coberta por testes em `tests/unit/test_patch_dataset.py`.
  - **Cobertura de patches** (38 tiles válidos, patch 64/stride 32): Veg 78,1%,
    Past 57,1%, Agri 35,0%, Solo **10,4%**, Urbano **13,1%**, Água **10,5%**
    (antes: 4,1/5,3/3,8). O relatório `dataset_report.json` traz
    `patch_coverage` por tile e agregado.
  - **Distribuição de pixels (38 tiles, 23,7M válidos):** Veg 37,8%, Past 28,0%,
    Agri 14,1%, Solo **5,2%**, Urbano **10,0%**, Água **4,9%** (antes:
    44,4/34,9/17,9/0,3/2,0/0,5).
  - Notebook Colab: `notebooks/04-unet-training-colab.ipynb` (checkpoints no Drive).
  - Pacote de upload: `data/colab_upload/` (repo + tiles + notebook + checksums).
  - **Treinada no Colab T4 (08/11/2026)**: `models/brazil_unet.pth` (124 MB,
    15 canais, 6 classes); early stop na época 67, melhor época ~55.
  - O loader do backend infere os canais do `state_dict` (15) e carrega o modelo.
  - �reas menores que 64�64 px s�o aceitas: `_pad_to_grid` replica as bordas
    at� 64�64 e recorta a sa�da. Se a U-Net falhar, o ensemble segue com RF/SVM.
    Para resultados quantitativos, prefira �reas maiores que 640�640 m (~41 ha).
  - **Correção do dataset (nov/2026):** NDVI/EVI agora têm `clamp(-1,1)`;
    `sanitize_features`/`sanitize_labels` removem NaN/Inf/outliers; tiles sem
    labels válidos são descartados. Dos 32 originais, **26 são válidos**:
    Pampa estava 100% `255` (label vazio) e Amazônia t2/t3 só tinham classe 0.
    `PatchDataset` ignora tiles inválidos; `Train Loss=0` era causado por lotes
    totalmente ignorados/sem diversidade.
  - Validação: `scripts/validate_unet_dataset.py` e `scripts/check_unet_tile.py`;
    relatório em `data/processed/unet_full/dataset_report.json`.
  - Teste de cota do GEE (1 tile pequeno): `scripts/check_gee_quota.py`.
- **Modelo regional SP** (`models/sp_rf.pkl`, `models/sp_svm.pkl`):
  - Treinado só nos polos Piracicaba, Ribeirão Preto e Barretos, com 15 features
    (9 originais + NDWI/MNDWI/SAVI/EVI/NBR/brilho).
  - Opção **“Regional SP (experimental)”** no modo Avançado (`sp_ensemble`).
- **Pós-processamento**: filtro de maioria 3×3 opcional (`postprocess`), aplicado
  antes das métricas/camadas. Ganho medido: **+1,29 pp** e Kappa +0,017 no tile
  de Piracicaba.
- **Calibração**: scripts/relatórios com ECE, MCE, Brier, NLL e reliability
  diagram em `outputs/v2/calibration/` (SP, Brasil e global).
- **Topografia**: camada `relief` (hipsometria + hillshade) exibida junto dos
  dados no card de topografia; fonte NASADEM (GLO-30 via
  `GEORURAL_DEM_SOURCE=GLO30`).
- **Frontend**:
  - “Automático (Ensemble)” como padrão; modelos individuais em **Avançado**.
  - Badge de cobertura (Brasil/Global), legenda com Água, confiança média
    rotulada como “não calibrada”.
  - Cultivo desabilitado fora do Brasil; U-Net só Sentinel-2/Brasil.
  - **Busca por local**: aceita endereço (Nominatim/OSM via
    `GET /api/geocode`) ou coordenadas `lat, lon`; o mapa voa até o ponto e
    mostra um marcador. O backend faz proxy do Nominatim com `User-Agent`,
    throttle (~1 req/s), cache em memória e **deduplicação** de resultados
    (mesmo local a <10 km).
  - **Limite de área**: aceita até `MAX_CLASSIFY_PIXELS` (6M px; ~60 mil ha
    em 10 m). Acima disso retorna erro amigável. Downloads grandes usam
    menos bandas por requisição (1 banda quando necessário) para não
    estourar o limite de ~48 MB do `getDownloadURL`.
- **Verificação**: `ruff` OK; **87 testes** passando; `npm run build` OK;
  `npm run lint` 0 erros (1 warning pré-existente em `Results.tsx`).

### Métricas full (validação espacial honesta)
| Domínio | Modelo | Acurácia | Kappa |
|---|---|---|---|
| Brasil (leave-one-bioma-out) | RF | 0,538 ± 0,097 | 0,447 ± 0,116 |
| Brasil (leave-one-bioma-out) | SVM | 0,535 ± 0,109 | 0,443 ± 0,131 |
| Brasil (hold-out por bioma) | **Ensemble RF+SVM** | **0,634** | **0,561** |
| Global (leave-one-região-out) | RF | 0,654 ± 0,051 | 0,584 ± 0,061 |
| Global (leave-one-região-out) | SVM | 0,669 ± 0,068 | 0,603 ± 0,082 |

- Cultivo (leave-one-região-out): **0,599 ± 0,037**.
  F1: Cana 0,666 · Citros 0,664 · Pastagem 0,654 · Soja 0,680 · **Milho 0,355**.
- A validação em bioma/região inteira é bem mais difícil que o hold-out por
  blocos (~0,73) e evita otimismo espacial.
- GEE E2E validado com os modelos full: Brasil (`auto`, ensemble, 6 classes,
  topografia, cultivo), global (WorldCover, RF+SVM, cultivo bloqueado) e SP
  (`sp_ensemble`, com relevo).
- **Atenção:** o projeto GEE atingiu a cota de compute do tier não comercial e
  entrou em modo restrito; novas extrações grandes podem ser limitadas.

### Interior de São Paulo (3 polos, blocos espaciais de 0,02°)
| Modelo | Acurácia | Kappa |
|---|---|---|
| RF 15 features | 0,7759 ± 0,0226 | 0,7307 ± 0,0269 |
| SVM 15 features | 0,7837 ± 0,0165 | 0,7400 ± 0,0199 |
| **Ensemble hold-out** | **0,8160** | **0,7790** |
| Raster Piracicaba (cross-polo) | 0,6306 | 0,4051 |
| Raster + maioria 3×3 | 0,6434 | 0,4221 |

### Calibração (hold-out espacial)
| Domínio | Acurácia | Confiança média | ECE bruto | ECE calibrado | T |
|---|---|---|---|---|---|
| SP | 0,8160 | 0,7747 | 0,0430 | 0,0203 | 0,884 |
| Brasil | 0,6343 | 0,6621 | 0,0278 | 0,0084 | 1,131 |
| Global | 0,6496 | 0,7029 | 0,0536 | 0,0113 | 1,197 |

- O “66%” da tela é confiança, não acurácia: no Brasil a confiança média foi
  0,662 contra 0,634 de acurácia; no global, 0,703 contra 0,650.
- Relatórios: `outputs/v2/calibration/calibration_*.json` e
  `reliability_*.png`.

### U-Net treinada (validação no tile `unet_piracicaba_t0`)
| Métrica | Valor |
|---|---:|
| Acurácia | 0,8237 (treino) / 0,8315 (modelo final) |
| Kappa | 0,7078 |
| mIoU | 0,4487 |
| Macro F1 | 0,5428 (treino) / 0,5908 (modelo final) |
| F1 Vegetação | 0,7945 |
| F1 Pastagem | 0,8443 |
| F1 Agricultura | 0,8107 |
| F1 Solo Exposto | 0,0623 |
| F1 Área Construída | 0,0068 |
| F1 Água | 0,7383 |

- No mesmo tile, U-Net+maioria 3×3: **0,8355** de acurácia, Kappa 0,7146,
  macro F1 0,6019.
- Pesos adotados no app: **U-Net 0,60 · RF 0,20 · SVM 0,20** (Brasil e SP).
  Com maioria 3×3:
  - ensemble Brasil: **0,8431** de acurácia, Kappa 0,7278, macro F1 0,5907;
  - ensemble SP (sp_rf + sp_svm + U-Net): **0,8703**, Kappa 0,7737,
    macro F1 0,6115.
- **Cautela:** Solo Exposto e Área Construída têm pouquíssimos pixels no tile
  de validação; o IoU/F1 dessas classes é instável. A U-Net é forte nas classes
  dominantes (0,1,2,5). O resultado de 0,8703 é de **um tile**; ainda falta
  validação leave-one-tile-out nos 38 tiles.
- Comparação completa: `outputs/tile_model_comparison.json` e
  `outputs/unet_metrics.json`.

### Retreino Etapa A + B (rebalanceamento offline + expansão dirigida)
- **Problema original:** Solo (0,30%), Água (0,51%) e Urbano (2,04%) quase não
  apareciam; no nível de patch, as raras estavam em apenas 4,1%/3,8%/5,3%.
- **Etapa B concluída (cota GEE confirmada em 22/11/2026):** 12 tiles dirigidos
  gerados — Ilha Solteira, Itaipu, Balbina e Manaus (água); Carajás,
  Quadrilátero Ferrífero, Lençóis e Paracatu (solo); São Paulo, Campinas,
  Goiânia e Belo Horizonte (urbano). Total: **38 tiles**, 23,7M px válidos.
- **Código pronto:** amostragem centrada nas raras, `FocalLoss(γ=1.5)`, pesos
  `1/sqrt(freq)` e validação multi-tile no notebook do Colab.
- **Metas:** F1 Solo ≥ 0,40; F1 Urbano ≥ 0,35; F1 Água ≥ 0,80;
  Veg/Past/Agri ≥ 0,75; acurácia ≥ 0,80; mIoU ≥ 0,55.
- **Critério:** só promover o novo `brazil_unet.pth` se as metas forem atingidas
  sem derrubar as classes dominantes.

### Retreino #1 (resultado e ajuste)
- **Resultado (6 tiles de validação):** acurácia 0,742 · mIoU 0,508 · macro F1
  0,652. F1 Solo **0,672** (era 0,062) e F1 Urbano **0,731** (era 0,007), mas
  **Água caiu para 0,240** (era 0,738) e acurácia ficou abaixo de 0,80.
- **Diagnóstico:** a precisão de água desabou (14,5%) — 221 mil px de Solo
  viraram Água. A confusão está em **Paracatu** (mineração de ouro escura,
  NDVI≈0,24 vs água 0,17), que estava na validação mas **nunca foi vista no
  treino** (treino de solo = Carajás ferro + Lençóis dunas), e em **Lençóis**
  (dunas NDVI≈0,01 vs lagoas NDVI≈0,13, quase idênticas).
- **Ajustes aplicados:** (1) `unet_solo_lencois` excluído do treino;
  (2) `unet_solo_paracatu` movido da validação para o treino (valida em
  `unet_solo_quadrilatero`); (3) `rare_patches_per_class` 150→100 e
  `focal_gamma` 2.0→1.5 para reduzir o overpredict das raras.

### Pendente
1. **Rodar o retreino #2 no Colab (T4)**:
   - Substituir no Drive apenas o `georural.zip` (e o notebook); o
     `unet_dataset_full.zip` **não mudou**.
   - Apagar `brazil_unet*.pth` antigos; rodar o notebook (célula de setup
     re-extrai o repositório sempre).
   - Baixar `brazil_unet.pth` e colocar em `models/`; o app passa a usá-lo.
2. Opcional: encoder pré-treinado (SSL4EO-S12/Prithvi) para a U-Net.
3. Opcional: features multi-temporais para tentar superar o platô do LORO.

### Comandos úteis
```bash
# Amostras full
python src/gee/extract_samples_cli.py --domain brazil --regions all --years 2019-2023 --samples-per-class 500 --out-dir data/samples/full/brazil --combined-out data/samples/full/brazil_samples.csv --no-legacy --resume
python src/gee/extract_samples_cli.py --domain global --regions all --years 2021 --label-source worldcover --samples-per-class 500 --out-dir data/samples/full/global --combined-out data/samples/full/global_samples.csv --no-legacy --resume
python src/gee/extract_samples_cli.py --task crop --domain brazil --regions all --years 2019-2023 --samples-per-crop 500 --out-dir data/samples/full/crop --combined-out data/samples/full/crop_samples.csv --no-legacy --resume

# Treino full
python notebooks/03-classifiers-test.py --csv data/samples/full/brazil_samples.csv --models-dir models/full --output-dir outputs/full
python notebooks/03-classifiers-test.py global --csv data/samples/full/global_samples.csv --models-dir models/full --output-dir outputs/full
python notebooks/05-crop-classifier.py --csv data/samples/full/crop_samples.csv --models-dir models/full --output-dir outputs/full
python notebooks/06-ensemble-eval.py --csv data/samples/full/brazil_samples.csv --models-dir models/full --output-dir outputs/full

# Modelo regional SP
python notebooks/03-classifiers-test.py sp --csv data/samples/full/brazil_samples.csv --filter-regions piracicaba ribeirao_preto barretos --group-strategy block --prefix sp --models-dir models/v2/sp --output-dir outputs/v2/sp

# Calibração e raster/pós-processamento
python notebooks/11-calibration-report.py --csv data/samples/full/brazil_samples.csv --prefix sp --filter-regions piracicaba ribeirao_preto barretos --group-strategy block --output-dir outputs/v2/calibration
python notebooks/11-calibration-report.py --csv data/samples/full/brazil_samples.csv --prefix brazil --group-strategy auto --no-local-indices --output-dir outputs/v2/calibration
python notebooks/12-sp-raster-eval.py --holdout-region piracicaba --output-dir outputs/v2/sp

# U-Net (38 tiles) e Colab
python src/gee/build_unet_dataset.py --regions all --tiles-per-region 4 --tile-size 0.08 --out data/processed/unet_full --year 2023 --start 2023-01-01 --end 2023-12-31 --skip-existing
python src/gee/build_unet_dataset.py --regions unet_agua_ilha_solteira unet_agua_itaipu unet_agua_balbina unet_agua_manaus unet_solo_carajas unet_solo_quadrilatero unet_solo_lencois unet_solo_paracatu unet_urbano_sao_paulo unet_urbano_campinas unet_urbano_goiania unet_urbano_belo_horizonte --tiles-per-region 1 --out data/processed/unet_full --year 2023 --start 2023-01-01 --end 2023-12-31 --skip-existing

# App
uvicorn backend.app.main:app --reload
cd frontend && npm run dev
```

### Limitações (documentar no TCC)
- **Pastagem global** = grassland (WorldCover/Dynamic World não separam pasto
  manejado de campo natural). No Brasil, MapBiomas separa.
- **Milho** é aproximado pela classe 41 do MapBiomas (outras lavouras
  temporárias); F1 baixo (0,355).
- **Circularidade:** WorldCover deriva de Sentinel-2; a acurácia global pode
  ser otimista. Mitigar com Dynamic World/teste cruzado.
- **SVM** é limitado a 20 mil amostras estratificadas por domínio.
- **Global** usa rótulos de 2021; classificar outros anos envolve *domain shift*.
- **Modelo SP** cobre os três polos amostrados; não generaliza automaticamente
  para todo o estado. Use a opção “Regional SP (experimental)” com cautela.
- **Índices locais** (NDWI etc.) não substituem features temporais: o teto de
  acurácia em SP (~0,82) ainda está abaixo de 0,85 em raster.
- **Pós-processamento** melhora suavidade/Kappa, mas pode remover corpos d’água
  pequenos e detalhes de borda.
- **RF Brasil** tem ~1,77 GB (300 árvores, profundidade 25 em 171 mil amostras);
  o carregamento do modelo exige memória.
- **Cota GEE:** o projeto entrou em modo restrito por excesso de compute do
  tier não comercial; novas extrações grandes podem falhar.
- `data/`, `models/`, `outputs/`, `notebooks/*.ipynb`, `.env` são gitignored.

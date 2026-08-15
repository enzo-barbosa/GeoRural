# Contexto do Projeto - GeoRural

## Estado atual (30/07/2026)

### Pronto
- Fase 0 a 5 praticamente completas (código + testes + notebooks)
- Notebooks ajustados: 01, 02, 03, 04
- Script CLI: `src/gee/extract_samples_cli.py`
- Interface Streamlit funcionando
- Testes: 16 passing
- Lint: OK
- Tudo que foi implementado já está marcado com checkbox `[x]` em `01-plano-desenvolvimento.md`

### Próximos passos (você precisa executar)

1. **Autenticar GEE:**
   ```bash
   earthengine authenticate
   ```

2. **Gerar CSV de amostras:**
   ```bash
   python src/gee/extract_samples_cli.py --out data/samples/training_samples.csv --satellite sentinel2
   ```

3. **Treinar RF/SVM:**
   ```bash
   jupyter notebook notebooks/03-classifiers-test.ipynb
   ```
   Gera: `models/random_forest.pkl` e `models/svm.pkl`

4. **Treinar U-Net (PC do amigo com RTX 5070):**
   - Levar: `data/samples/training_samples.csv`, `notebooks/04-unet-training.ipynb`, pasta `src/`
   - No PC do amigo: instalar deps e rodar notebook 04
   - De volta: `models/unet_model.pth`

5. **Rodar interface:**
    ```bash
    streamlit run run_app.py
    ```

### Observações importantes
- O notebook 04 agora gera os `.npy` automaticamente a partir do CSV
- A interface carrega modelos automaticamente se existirem em `models/`
- Não precisa de git/github para continuar
- Quando voltar a falar com a IA, é só dizer "continue o projeto" que ela recupera o contexto pelos arquivos
- O `auth.py` agora requer `GEE_PROJECT_ID` no `.env` (sem fallback hardcoded)
- Interface melhorada com tema visual, CSS customizado e componentes reutilizáveis
- Entrypoint principal: `streamlit run run_app.py`

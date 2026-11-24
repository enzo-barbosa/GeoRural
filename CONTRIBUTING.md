# Como contribuir

Este documento descreve o fluxo de trabalho do GeoRural. Ele vale para os dois
sistemas operacionais usados no projeto: Linux e Windows.

## Pré-requisitos

- Python 3.10+
- Node.js 20+
- Conta com acesso ao Google Earth Engine (a mesma conta do projeto)
- Git

## Configuração local

```bash
git clone git@github.com:enzo-barbosa/GeoRural.git
cd GeoRural
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,backend]"
cp .env.example .env             # Windows: Copy-Item .env.example .env
earthengine authenticate
cd frontend && npm ci
```

Preencha o `.env` com o `GEE_PROJECT_ID` do projeto. Nunca commite o `.env`.

## Branches

Crie uma branch por tarefa, a partir da `main` atualizada:

```bash
git switch main
git pull
git switch -c feat/nome-curto
```

Prefixos usados:

| Prefixo | Uso |
|---|---|
| `feat/` | nova funcionalidade |
| `fix/` | correção de bug |
| `refactor/` | reorganização sem mudar comportamento |
| `perf/` | melhoria de desempenho |
| `test/` | testes |
| `docs/` | documentação |
| `chore/` | infraestrutura, dependências, configuração |

## Commits atômicos

Cada commit deve resolver **um assunto**. Não misture refatoração com feature,
formatação com correção de bug ou documentação com código.

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: salvar áreas classificadas no banco
fix: corrigir máscara de nuvens do Sentinel-2
perf: reduzir pico de memória da classificação
docs: documentar fluxo de PR
chore: configurar Docker Compose
test: cobrir autenticação JWT
```

Exemplos de commits ruins:

- `ajustes` (não diz o que mudou)
- `feat: várias coisas` (não é atômico)
- `fix: bug` (não descreve o problema)

## Pull Request

1. Faça push da branch:
   ```bash
   git push -u origin feat/nome-curto
   ```
2. Abra o PR usando o template. O **título do PR deve seguir Conventional
   Commits**, porque o merge é feito por squash e o título vira a mensagem do
   commit na `main`.
3. Descreva o que mudou, como testar e anexe screenshots quando houver mudança
   de interface.
4. Mantenha o PR pequeno. Quando possível, até ~400 linhas alteradas.
5. O CI precisa passar (Ubuntu e Windows).
6. O merge só acontece após aprovação do code owner. O merge é **squash** e a
   branch é apagada automaticamente.

Antes de abrir o PR:

```bash
ruff check src backend tests scripts
pytest -q
cd frontend && npm run lint && npm run build
```

## O que nunca commitar

- `.env` e qualquer credencial
- `models/` (modelos treinados são distribuídos por release)
- `data/`, `outputs/` e arquivos `.tif`, `.pkl`, `.pth`
- `.venv/`, `node_modules/`
- artefatos de cobertura (`.coverage`, `htmlcov/`)

O `.gitignore` já cobre esses casos. Em caso de dúvida, rode `git status` antes
de commitar.

## Windows

- O `.gitattributes` força LF no repositório. Não altere a configuração global
  de `core.autocrlf`; deixe o Git cuidar disso.
- Use PowerShell para os comandos equivalentes (`Copy-Item`, `Activate.ps1`).
- Caminhos de credenciais do Earth Engine ficam em
  `%USERPROFILE%\.config\earthengine`.

## Dúvidas

Abra uma issue com o template de bug ou feature. Para mudanças grandes, alinhe
antes com o dono do repositório para evitar retrabalho.

# Configuração do GitHub

Passo a passo para deixar o repositório com o fluxo profissional de PR, CI e
merge. Só o dono do repositório precisa fazer isso, uma vez.

## 1. Pull Requests

Em **Settings → General → Pull Requests**:

- [x] Allow squash merging
- [x] Default to pull request title (mensagem do squash = título do PR)
- [ ] Desabilitar **Allow merge commits**
- [ ] Desabilitar **Allow rebase merging**
- [x] Automatically delete head branches
- [x] Allow auto-merge (opcional)

Assim o histórico da `main` fica linear: **1 PR = 1 commit**.

## 2. Ruleset da branch main

Em **Settings → Rules → Rulesets → New branch ruleset**:

1. Nome: `main-protection`
2. Enforcement status: **Active**
3. Target branches: **Include default branch**
4. Bypass list: adicione **Repository admin** (necessário porque o GitHub não
   permite autoaprovação; o dono pode usar bypass nos próprios PRs).
5. Marque:
   - [x] Restrict deletions
   - [x] Require a pull request before merging
     - Required approvals: **1**
     - [x] Dismiss stale pull request approvals when new commits are pushed
     - [x] Require review from Code Owners
     - [x] Require conversation resolution before merging
   - [x] Require status checks to pass
     - Marque **Require branches to be up to date before merging**
     - Checks obrigatórios (aparecem depois do primeiro run):
       - `lint`
       - `test`
       - `frontend`
       - `windows`
       - `conventional-title`
       - `analyze (python)` (CodeQL)
       - `analyze (javascript-typescript)` (CodeQL)
   - [x] Block force pushes
   - [x] Require linear history
6. Salve.

> O `CODEOWNERS` já aponta para `@enzo-barbosa`. O amigo abre o PR, você
> aprova e faz o squash merge. Nos seus próprios PRs, use o bypass de admin ou
> peça a aprovação dele.

## 3. Actions

Em **Settings → Actions → General**:

- [x] Allow all actions and reusable workflows
- Workflow permissions: **Read repository contents and packages permissions**
  (os workflows que precisam de escrita declaram `permissions` próprias)
- [x] Allow GitHub Actions to create and approve pull requests (necessário para
  o Release Please)

## 4. Segurança

Em **Settings → Code security and analysis**:

- [x] Dependabot alerts
- [x] Dependabot security updates
- [x] Secret scanning
- [x] Push protection
- [x] CodeQL: **Advanced setup** (o workflow `codeql.yml` já está no repositório;
  não habilite o default setup para não duplicar)

## 5. Release de modelos

Publique a release `models-v1` com o script (mais seguro que o comando manual):

```bash
uv run python scripts/publish_models.py --dry-run
uv run python scripts/publish_models.py --upload --update-manifest
```

O `models.lock.json` aponta para
`https://github.com/enzo-barbosa/GeoRural/releases/download/models-v1/<arquivo>`
e já contém os SHA-256. O download (`scripts/download_models.py`) valida
tamanho e hash; confira com `--check`.

## 6. Desenvolvimento local

Instale os hooks uma vez:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## 7. Fluxo resumido

```bash
git switch main && git pull
git switch -c feat/minha-tarefa
# commits atômicos...
git push -u origin feat/minha-tarefa
gh pr create --fill
# CI roda; você revisa e aprova
# squash merge + delete branch
```

## 8. Regra de ouro: um PR por vez

O desenvolvimento é **sequencial**, sem PRs empilhados:

1. Abra **um PR** por tarefa.
2. Espere **todos os checks verdes** (lint, test, frontend, windows, título e
   CodeQL quando aplicável).
3. O dono revisa e faz o **squash merge**.
4. Atualize a `main` (`git switch main && git pull`) e só então abra o próximo.

Valem também para PRs automáticos (Dependabot e Release Please): um por vez,
CI verde e merge antes de seguir. Se um PR depende de outro, aguarde o merge e
rebaseie a branch sobre a `main`; nunca abra em cima de uma branch não mergeada.

## 9. Problemas comuns

- **Check obrigatório não aparece**: ele só fica disponível depois de rodar ao
  menos uma vez. Abra um PR de teste.
- **`windows` falhando**: veja o log do job; normalmente é dependência sem wheel
  ou caminho com separador errado.
- **Release Please sem permissão**: confira a opção “Allow GitHub Actions to
  create and approve pull requests”.
- **Docker não sobe no Windows**: confirme que o Docker Desktop está com WSL2
  ativo e que as portas 8000/5173/5434 estão livres.

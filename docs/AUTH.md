# Autenticação

O GeoRural usa sessão por **JWT em cookie httpOnly** e senhas com **Argon2**.
Não há recuperação de senha nem verificação de email (fora do escopo do TCC).

## Variáveis de ambiente

```env
JWT_SECRET_KEY=dev-insecure-secret-change-me-please-32-bytes
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080
AUTH_COOKIE_NAME=georural_access
AUTH_COOKIE_SECURE=false
```

Gere um segredo real:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Em produção/HTTPS, use `AUTH_COOKIE_SECURE=true`.

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| POST | `/api/auth/register` | cria conta e inicia a sessão |
| POST | `/api/auth/login` | valida email/senha e inicia a sessão |
| POST | `/api/auth/logout` | apaga o cookie |
| GET | `/api/auth/me` | retorna o usuário autenticado |

Exemplo com `curl`:

```bash
curl -c cookies.txt -X POST http://localhost:8000/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"name":"Teste","email":"teste@example.com","password":"senha1234"}'

curl -b cookies.txt http://localhost:8000/api/auth/me
```

O cookie é `httpOnly`, `SameSite=Lax` e enviado automaticamente pelo navegador
no mesmo site. As rotas de classificação exigem sessão; `/api/health` e
`/api/geocode` continuam públicos.

## Limitações

- JWT é stateless: o logout apaga o cookie, mas não revoga tokens já emitidos.
- Não há rate limiting no login.
- As imagens em `/outputs/...` ainda são servidas publicamente por URL; a
  proteção por usuário será feita junto com a persistência das análises.
- O frontend usa o proxy do Vite (`/api`), então o cookie funciona em
  `localhost:5173` sem configuração extra.

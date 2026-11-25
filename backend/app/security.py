"""Hash de senha e tokens JWT."""

from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash

from backend.app.config import get_settings

_password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Gera o hash Argon2 da senha."""
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """Verifica a senha contra o hash (False se o hash estiver vazio)."""
    if not password_hash:
        return False
    return _password_hash.verify(password, password_hash)


def create_access_token(user_id: int) -> str:
    """Cria um JWT com o id do usuário e expiração configurada."""
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(
        payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


def decode_access_token(token: str) -> int | None:
    """Decodifica o JWT e retorna o id do usuário, ou None se inválido."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.InvalidTokenError:
        return None

    subject = payload.get("sub")
    try:
        return int(subject)
    except (TypeError, ValueError):
        return None

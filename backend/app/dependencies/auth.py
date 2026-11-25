"""Dependência de usuário autenticado (cookie httpOnly ou Bearer)."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.db.models import User
from backend.app.db.session import get_db
from backend.app.security import decode_access_token

_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Não autenticado",
    headers={"WWW-Authenticate": "Bearer"},
)


def _token_from_request(request: Request) -> str | None:
    token = request.cookies.get(get_settings().AUTH_COOKIE_NAME)
    if token:
        return token

    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def get_current_user(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Retorna o usuário da sessão ou 401."""
    token = _token_from_request(request)
    if not token:
        raise _CREDENTIALS_ERROR

    user_id = decode_access_token(token)
    if user_id is None:
        raise _CREDENTIALS_ERROR

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _CREDENTIALS_ERROR
    return user

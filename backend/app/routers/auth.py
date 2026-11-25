"""Rotas de cadastro, login, logout e usuário atual."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.db.models import User
from backend.app.db.repositories import create_user, get_user_by_email
from backend.app.db.session import get_db
from backend.app.dependencies.auth import get_current_user
from backend.app.schemas.auth import (
    LoginRequest,
    LogoutResponse,
    RegisterRequest,
    UserRead,
)
from backend.app.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _set_auth_cookie(response: Response, user_id: int) -> None:
    settings = get_settings()

    token = create_access_token(user_id)
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        secure=settings.AUTH_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


@router.post("/register", response_model=UserRead, status_code=201)
def register(
    payload: RegisterRequest,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Cria uma conta e já inicia a sessão."""
    if get_user_by_email(db, payload.email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Este email já está cadastrado.",
        )

    try:
        user = create_user(
            db,
            email=payload.email,
            name=payload.name,
            password_hash=hash_password(payload.password),
        )
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Este email já está cadastrado.",
        ) from exc

    _set_auth_cookie(response, user.id)
    return user


@router.post("/login", response_model=UserRead)
def login(
    payload: LoginRequest,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Valida as credenciais e inicia a sessão."""
    user = get_user_by_email(db, payload.email)
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou senha inválidos.",
        )

    _set_auth_cookie(response, user.id)
    return user


@router.post("/logout", response_model=LogoutResponse)
def logout(response: Response) -> LogoutResponse:
    """Remove o cookie de sessão."""
    settings = get_settings()
    response.delete_cookie(
        key=settings.AUTH_COOKIE_NAME,
        path="/",
        secure=settings.AUTH_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )
    return LogoutResponse()


@router.get("/me", response_model=UserRead)
def me(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Retorna o usuário autenticado."""
    return current_user

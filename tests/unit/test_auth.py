"""Testes de autenticação (cadastro, login, sessão e proteção de rotas)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.models import User
from backend.app.db.session import get_db
from backend.app.dependencies.models import get_gee
from backend.app.main import create_app
from backend.app.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as session:
        yield session
    Base.metadata.drop_all(engine)


@pytest.fixture()
def client(db):
    app = create_app()

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_gee] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _register(client, email="user@example.com", password="senha1234"):
    return client.post(
        "/api/auth/register",
        json={"email": email, "name": "Usuário", "password": password},
    )


def test_register_logs_in_and_returns_user(client):
    response = _register(client)

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "user@example.com"
    assert body["name"] == "Usuário"
    assert "password" not in body
    assert client.cookies.get("georural_access")

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["id"] == body["id"]


def test_duplicate_email_returns_conflict(client):
    assert _register(client).status_code == 201
    client.post("/api/auth/logout")

    response = _register(client)

    assert response.status_code == 409


def test_login_with_wrong_password(client):
    assert _register(client).status_code == 201
    client.post("/api/auth/logout")

    response = client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "errada123"},
    )

    assert response.status_code == 401


def test_logout_clears_session(client):
    assert _register(client).status_code == 201

    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_invalid_token_is_rejected(client):
    client.cookies.set("georural_access", "token-invalido")

    assert client.get("/api/auth/me").status_code == 401


def test_inactive_user_cannot_login_or_use_session(client, db):
    response = _register(client)
    user = db.get(User, response.json()["id"])
    assert user is not None
    user.is_active = False
    db.commit()

    assert client.get("/api/auth/me").status_code == 401
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "user@example.com", "password": "senha1234"},
        ).status_code
        == 401
    )


def test_classify_requires_authentication(client):
    response = client.get("/api/classify/models")

    assert response.status_code == 401


def test_security_helpers():
    hashed = hash_password("senha1234")

    assert hashed != "senha1234"
    assert verify_password("senha1234", hashed) is True
    assert verify_password("outra", hashed) is False
    assert verify_password("senha1234", None) is False

    token = create_access_token(42)
    assert decode_access_token(token) == 42
    assert decode_access_token("invalido") is None

"""Autenticação com o Google Earth Engine — v2.0.

Suporta dois modos:
1. Local: Autenticação interativa via navegador (padrão)
2. Cloud: Service Account via variável de ambiente (Streamlit Cloud)
"""

import json
import os

import ee
from dotenv import load_dotenv

load_dotenv()


def _project_id() -> str:
    """Lê o ID do projeto do GEE sob demanda.

    O valor não é lido no import para que módulos possam ser importados em
    testes e no CI sem um ``.env`` configurado. O erro só aparece quando o
    Earth Engine é realmente inicializado.
    """
    project_id = os.environ.get("GEE_PROJECT_ID", "").strip()
    if not project_id:
        raise RuntimeError(
            "GEE_PROJECT_ID não configurado. Defina a variável no .env "
            "ou no ambiente antes de usar o Earth Engine."
        )
    return project_id


def initialize_gee():
    """Inicializa a conexão com o GEE.

    Tenta Service Account primeiro (deploy cloud), depois
    autenticação interativa (execução local).
    """
    project_id = _project_id()
    service_account_json = os.environ.get("GEE_SERVICE_ACCOUNT")
    service_account_email = os.environ.get("GEE_SERVICE_ACCOUNT_EMAIL")

    if service_account_json and service_account_email:
        key_data = json.loads(service_account_json)
        credentials = ee.ServiceAccountCredentials(
            email=service_account_email, key=key_data
        )
        ee.Initialize(credentials, project=project_id)
        print("GEE inicializado com Service Account")
    else:
        try:
            ee.Initialize(project=project_id)
            print("GEE inicializado (credenciais existentes)")
        except ee.EEException:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            print("GEE autenticado e inicializado")

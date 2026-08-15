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

GEE_PROJECT_ID = os.environ["GEE_PROJECT_ID"]


def initialize_gee():
    """Inicializa a conexão com o GEE.

    Tenta Service Account primeiro (deploy cloud), depois
    autenticação interativa (execução local).
    """
    service_account_json = os.environ.get("GEE_SERVICE_ACCOUNT")
    service_account_email = os.environ.get("GEE_SERVICE_ACCOUNT_EMAIL")

    if service_account_json and service_account_email:
        key_data = json.loads(service_account_json)
        credentials = ee.ServiceAccountCredentials(
            email=service_account_email, key=key_data
        )
        ee.Initialize(credentials, project=GEE_PROJECT_ID)
        print("GEE inicializado com Service Account")
    else:
        try:
            ee.Initialize(project=GEE_PROJECT_ID)
            print("GEE inicializado (credenciais existentes)")
        except ee.EEException:
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT_ID)
            print("GEE autenticado e inicializado")

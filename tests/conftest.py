"""Configuracao comum dos testes.

Coloca src/ e scripts/ no caminho de importacao para que os testes exercitem
exatamente os mesmos modulos que o notebook importa.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "notebook: executa o notebook integrado de ponta a ponta (lento; pule com -m 'not notebook')",
    )


@pytest.fixture
def nps():
    """Tres respostas de pesquisa, com as colunas que a integracao usa."""
    return pd.DataFrame({
        "RESPONDENT_ID": [1, 2, 3],
        "ID_GOLDENRECORD": [10, 20, 30],
        "DATA_STD": ["2024-01-15", "2024-02-20", "2024-03-25"],
        "NPS_PRINCIPAL": [100, -100, 0],
        "VOO_TIPO": ["Direto", "Conexão", "Direto"],
        "TEMPO_VOO": [120, 300, 90],
        "CANAL_COMPRA": ["Web", "Agency", "Mobile"],
        "BASE_AIRPORTLEG": ["VCP/CGH", "VCP/CNF/REC", "SDU/CGH"],
        "VOO_INTERNACIONAL": ["Domestic", "Domestic", "Domestic"],
    })


@pytest.fixture
def perfil():
    return pd.DataFrame({
        "RESPONDENT_ID": [1, 2, 3],
        "ID_GOLDENRECORD": [10, 20, 30],
        "QTDE_VIAGENS_12M": [2, 5, 1],
        "TIER_VIAGEM": ["Diamante", "Sem cadastro", "Safira"],
    })


@pytest.fixture
def viagem():
    return pd.DataFrame({
        "RESPONDENT_ID": [1, 2, 3],
        "ESTATISTICA_ATRASOSAIDA": [0, 45, 200],
        "ATRASO_CHEGADA": [0, 50, 210],
        "CANCELAMENTO_VOO": [False, False, True],
        "ANTECEDENCIA_CANCELAMENTO": [None, None, 5],
        "ASSENTOS": ["12A", "3B/7C", "20F"],
    })

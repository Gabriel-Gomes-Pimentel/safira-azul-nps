"""Testes da conferência da allowlist sobre a base analítica carregada (#278).

Na Sprint 3, a seção 1.3 de `notebooks/modelagem.ipynb` levantou `KeyError` em
`features_ausentes = [...]` por falta de `N_TRECHOS`: o parquet havia sido gravado
antes do fix(#177), quando a derivação ainda não acontecia em
`preparar_base_analitica`. Os cenários usam somente dados sintéticos.
"""

import pandas as pd
import pytest

from preprocessamento_nps import FEATURE_SET_V1, conferir_features_v1_na_base


def base_sem_n_trechos() -> pd.DataFrame:
    """Base como a de um parquet anterior ao #177: tem a origem, não a derivada."""
    return pd.DataFrame({
        "TIER_VIAGEM": ["SAFIRA", "DIAMANTE"],
        "VOO_TIPO": ["DIRETO", "CONEXAO"],
        "TIPO_ENTRETENIMENTO": ["TELA", "TELA"],
        "CANAL_COMPRA": ["WEB", "AGENCIA"],
        "SEGMENTO": ["LAZER", "CORPORATIVO"],
        "ESTATISTICA_ATRASOSAIDA": [0.0, 20.0],
        "ATRASO_CHEGADA": [0.0, 25.0],
        "CANCELAMENTO_VOO": [False, False],
        "ANTECEDENCIA_CANCELAMENTO": [float("nan"), float("nan")],
        "TEMPO_VOO": [90.0, 120.0],
        "BASE_AIRPORTLEG": ["VCP/CNF", "FOR/UDI/CNF/POA"],
    })


def test_base_sem_n_trechos_nao_levanta_keyerror_e_deriva_a_coluna():
    base = conferir_features_v1_na_base(base_sem_n_trechos())

    assert set(FEATURE_SET_V1) <= set(base.columns)
    assert base["N_TRECHOS"].tolist() == [1, 3]


def test_conferencia_nao_altera_a_base_recebida():
    original = base_sem_n_trechos()
    conferir_features_v1_na_base(original)

    assert "N_TRECHOS" not in original.columns


def test_base_com_n_trechos_passa_sem_rederivar():
    base = base_sem_n_trechos()
    base["N_TRECHOS"] = pd.array([7, 7], dtype="Int64")

    assert conferir_features_v1_na_base(base)["N_TRECHOS"].tolist() == [7, 7]


def test_base_sem_origem_de_n_trechos_falha_com_orientacao():
    base = base_sem_n_trechos().drop(columns=["BASE_AIRPORTLEG"])

    with pytest.raises(KeyError, match=r"N_TRECHOS.*BASE_AIRPORTLEG"):
        conferir_features_v1_na_base(base)


def test_feature_sem_derivacao_continua_obrigatoria():
    base = base_sem_n_trechos().drop(columns=["TEMPO_VOO"])

    with pytest.raises(KeyError, match="TEMPO_VOO"):
        conferir_features_v1_na_base(base)

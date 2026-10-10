"""Testes de verificar_metas_simultaneas e resumo_da_pendencia (card #247).

Tabela sintética, no formato de `modelo.tabela_comparativa`: uma linha por
modelo, com `cobertura` (Sensibilidade) e `precisao_no_topo` (Precisão) no
ponto operacional. Nada aqui lê `data/`.

Executar com:  pytest tests/test_pendencia_metas.py -v
"""
import numpy as np
import pandas as pd
import pytest

from pendencia_metas import (
    META_PRECISAO,
    META_SENSIBILIDADE,
    resumo_da_pendencia,
    verificar_metas_simultaneas,
)


@pytest.fixture
def tabela_pendencia_nao_resolvida():
    # Reproduz a leitura da Seção 4.3.2: nenhum modelo atinge as duas metas.
    return pd.DataFrame(
        {
            "cobertura": [0.4464, 0.50, 0.55],
            "precisao_no_topo": [0.5283, 0.45, 0.35],
            "fila_comparavel": [True, True, True],
        },
        index=pd.Index(["gradient_boosting", "random_forest", "arvore"], name="modelo"),
    )


@pytest.fixture
def tabela_pendencia_resolvida():
    return pd.DataFrame(
        {
            "cobertura": [0.4464, 0.75],
            "precisao_no_topo": [0.5283, 0.42],
            "fila_comparavel": [True, True],
        },
        index=pd.Index(["gradient_boosting", "logistica_tunada"], name="modelo"),
    )


def test_marca_true_so_quando_as_duas_metas_sao_atingidas(tabela_pendencia_resolvida):
    resultado = verificar_metas_simultaneas(tabela_pendencia_resolvida)
    assert resultado.loc["logistica_tunada", "atende_as_duas_metas"] == True  # noqa: E712
    assert resultado.loc["gradient_boosting", "atende_as_duas_metas"] == False  # noqa: E712


def test_nenhum_modelo_atende_quando_nenhum_bate_as_duas_metas(tabela_pendencia_nao_resolvida):
    resultado = verificar_metas_simultaneas(tabela_pendencia_nao_resolvida)
    assert not resultado["atende_as_duas_metas"].any()


def test_fila_nao_comparavel_nunca_atende_mesmo_com_metricas_boas():
    tabela = pd.DataFrame(
        {
            "cobertura": [0.90],
            "precisao_no_topo": [0.90],
            "fila_comparavel": [False],
        },
        index=pd.Index(["piso_degenerado"], name="modelo"),
    )
    resultado = verificar_metas_simultaneas(tabela)
    assert resultado.loc["piso_degenerado", "atende_as_duas_metas"] == False  # noqa: E712


def test_modelo_indisponivel_com_nan_nunca_atende():
    tabela = pd.DataFrame(
        {
            "cobertura": [np.nan],
            "precisao_no_topo": [np.nan],
            "fila_comparavel": [np.nan],
        },
        index=pd.Index(["arvore_pendente"], name="modelo"),
    )
    resultado = verificar_metas_simultaneas(tabela)
    assert resultado.loc["arvore_pendente", "atende_as_duas_metas"] == False  # noqa: E712


def test_funciona_sem_a_coluna_fila_comparavel(tabela_pendencia_resolvida):
    tabela = tabela_pendencia_resolvida.drop(columns=["fila_comparavel"])
    resultado = verificar_metas_simultaneas(tabela)
    assert resultado.loc["logistica_tunada", "atende_as_duas_metas"] == True  # noqa: E712


def test_levanta_erro_com_coluna_obrigatoria_ausente():
    tabela = pd.DataFrame({"cobertura": [0.8]}, index=pd.Index(["x"], name="modelo"))
    with pytest.raises(KeyError, match="precisao_no_topo"):
        verificar_metas_simultaneas(tabela)


def test_metas_sao_as_constantes_da_secao_4_3_2():
    assert META_SENSIBILIDADE == 0.70
    assert META_PRECISAO == 0.40


def test_resumo_nomeia_o_vencedor_quando_a_pendencia_e_resolvida(tabela_pendencia_resolvida):
    resultado = verificar_metas_simultaneas(tabela_pendencia_resolvida)
    texto = resumo_da_pendencia(resultado)
    assert "resolvida" in texto
    assert "logistica_tunada" in texto
    assert "gradient_boosting" not in texto.split("resolvida:")[1].split(".")[0]


def test_resumo_diz_que_a_pendencia_permanece_quando_ninguem_atende(tabela_pendencia_nao_resolvida):
    resultado = verificar_metas_simultaneas(tabela_pendencia_nao_resolvida)
    texto = resumo_da_pendencia(resultado)
    assert "permanece" in texto


def test_resumo_exige_coluna_de_veredito_ja_calculada(tabela_pendencia_nao_resolvida):
    with pytest.raises(KeyError, match="atende_as_duas_metas"):
        resumo_da_pendencia(tabela_pendencia_nao_resolvida)

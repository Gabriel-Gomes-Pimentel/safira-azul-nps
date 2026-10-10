"""Testes de sensibilidade_capacidade.py (card #248).

Score sintético com uma faixa de valores empatados, para forçar um salto de
cobertura previsível entre duas capacidades vizinhas. Nada aqui lê `data/`.

Executar com:  pytest tests/test_sensibilidade_capacidade.py -v
"""
import numpy as np
import pytest

from sensibilidade_capacidade import (
    ganho_por_contato_extra,
    sensibilidade_a_capacidade,
)


@pytest.fixture
def y_e_score():
    # 10 positivos, 10 negativos; score ordena perfeitamente os 10 primeiros.
    y = np.array([1] * 10 + [0] * 10)
    score = np.linspace(1.0, 0.05, 20)
    return y, score


def test_cobertura_sobe_com_a_capacidade(y_e_score):
    y, score = y_e_score
    tabela = sensibilidade_a_capacidade({"modelo_x": score}, y, capacidades=[5, 10])
    assert tabela.loc[("modelo_x", 5), "cobertura"] == pytest.approx(0.5)
    assert tabela.loc[("modelo_x", 10), "cobertura"] == pytest.approx(1.0)


def test_modelo_indisponivel_gera_nan_em_todas_as_capacidades(y_e_score):
    y, _ = y_e_score
    tabela = sensibilidade_a_capacidade({"arvore": None}, y, capacidades=[5, 10, 15])
    assert tabela["cobertura"].isna().all()
    assert len(tabela) == 3


def test_ganho_por_contato_bate_com_a_conta_manual(y_e_score):
    y, score = y_e_score
    tabela = sensibilidade_a_capacidade({"modelo_x": score}, y, capacidades=[5, 10])
    ganho = ganho_por_contato_extra(tabela)

    linha = ganho.loc[("modelo_x", 5, 10)]
    # cobertura vai de 0.5 (k=5) para 1.0 (k=10): ganho de 0.5 em 5 contatos.
    assert linha["ganho_de_cobertura"] == pytest.approx(0.5)
    assert linha["por_contato"] == pytest.approx(0.5 / 5)


def test_ganho_com_uma_so_capacidade_e_nan(y_e_score):
    y, score = y_e_score
    tabela = sensibilidade_a_capacidade({"modelo_x": score}, y, capacidades=[10])
    ganho = ganho_por_contato_extra(tabela)
    assert np.isnan(ganho.iloc[0]["ganho_de_cobertura"])


def test_ganho_por_contato_extra_agrupa_por_modelo_independentemente(y_e_score):
    y, score = y_e_score
    tabela = sensibilidade_a_capacidade(
        {"modelo_x": score, "modelo_y": None}, y, capacidades=[5, 10]
    )
    ganho = ganho_por_contato_extra(tabela)

    assert ("modelo_x", 5, 10) in ganho.index
    assert ("modelo_y", 5, 10) in ganho.index
    assert np.isnan(ganho.loc[("modelo_y", 5, 10), "ganho_de_cobertura"])

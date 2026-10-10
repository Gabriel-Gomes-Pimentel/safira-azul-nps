"""Testes de gerar_matrizes e falsos_negativos (card #250).

Scores e limiares sintéticos; nada aqui lê `data/` nem instancia um estimador
real. `modelo.matriz_de_confusao` já tem seus próprios testes — aqui a
cobertura é da orquestração entre os quatro candidatos, incluindo os
indisponíveis.

Executar com:  pytest tests/test_matrizes_confusao_candidatos.py -v
"""
import numpy as np
import pytest

from matrizes_confusao_candidatos import falsos_negativos, gerar_matrizes


@pytest.fixture
def y_verdadeiro():
    return np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0])


@pytest.fixture
def score_logistica():
    return np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.6, 0.4, 0.15, 0.05])


def test_gera_matriz_para_modelo_disponivel(y_verdadeiro, score_logistica):
    matrizes = gerar_matrizes(
        scores={"logistica": score_logistica, "arvore": None},
        y_verdadeiro=y_verdadeiro,
        limiares={"logistica": 0.5, "arvore": None},
    )
    assert matrizes["logistica"] is not None
    assert matrizes["logistica"].values.sum() == len(y_verdadeiro)


def test_modelo_sem_score_devolve_none(y_verdadeiro, score_logistica):
    matrizes = gerar_matrizes(
        scores={"logistica": score_logistica, "arvore": None},
        y_verdadeiro=y_verdadeiro,
        limiares={"logistica": 0.5, "arvore": 0.4},
    )
    assert matrizes["arvore"] is None


def test_modelo_sem_limiar_devolve_none_mesmo_com_score(y_verdadeiro, score_logistica):
    matrizes = gerar_matrizes(
        scores={"logistica": score_logistica, "gb": score_logistica},
        y_verdadeiro=y_verdadeiro,
        limiares={"logistica": 0.5, "gb": None},
    )
    assert matrizes["gb"] is None
    assert matrizes["logistica"] is not None


def test_chaves_divergentes_entre_scores_e_limiares_levantam_erro(y_verdadeiro, score_logistica):
    with pytest.raises(KeyError):
        gerar_matrizes(
            scores={"logistica": score_logistica},
            y_verdadeiro=y_verdadeiro,
            limiares={"arvore": 0.5},
        )


def test_falso_negativo_bate_com_a_contagem_manual(y_verdadeiro, score_logistica):
    # limiar 0.65: selecionados (score >= 0.65) -> índices 0,1,4 (0.9,0.8,0.7)
    # positivos reais (y=1) em 0,1,4,6 -> índice 6 (score 0.6) fica de fora: 1 FN
    matrizes = gerar_matrizes(
        scores={"logistica": score_logistica},
        y_verdadeiro=y_verdadeiro,
        limiares={"logistica": 0.65},
    )
    fn = falsos_negativos(matrizes)
    assert fn["logistica"] == 1


def test_falso_negativo_de_modelo_indisponivel_e_none():
    fn = falsos_negativos({"arvore": None})
    assert fn["arvore"] is None

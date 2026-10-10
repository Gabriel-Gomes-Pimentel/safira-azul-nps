"""Testes da função única de avaliação do protocolo (#241).

O que estes testes protegem é que as quatro duplas de modelagem leiam a mesma
conta: os nomes exatos das quatro métricas no dicionário devolvido, a fórmula do
F2 com beta=2 (peso 4x ao recall) e o caso de classe positiva ausente no lote,
que precisa devolver nan em vez de lançar exceção.

Nada aqui lê `data/`: os vetores são sintéticos, em linha com o Termo de
Abertura.

Executar com:  pytest tests/test_avaliacao.py -v
"""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, fbeta_score, recall_score, roc_auc_score

from avaliacao import NOMES_METRICAS, avaliar


@pytest.fixture
def lote():
    """Oito linhas com as três classes de acerto/erro que as métricas distinguem."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.2, 0.1, 0.05])
    return y_true, y_pred, y_proba


def test_devolve_exatamente_os_quatro_nomes_do_protocolo(lote):
    resultado = avaliar(*lote)
    assert set(resultado) == set(NOMES_METRICAS)


def test_valores_batem_com_as_funcoes_do_scikit_learn(lote):
    y_true, y_pred, y_proba = lote
    resultado = avaliar(*lote)
    assert resultado["F2"] == pytest.approx(fbeta_score(y_true, y_pred, beta=2))
    assert resultado["Sensibilidade"] == pytest.approx(recall_score(y_true, y_pred))
    assert resultado["Precisão Média"] == pytest.approx(average_precision_score(y_true, y_proba))
    assert resultado["ROC-AUC"] == pytest.approx(roc_auc_score(y_true, y_proba))


def test_f2_pondera_o_recall_quatro_vezes_mais_que_a_precisao(lote):
    """F2 = 5*P*R / (4*P+R): confere a fórmula do card 02A.1 à mão, não só via sklearn."""
    y_true, y_pred, _ = lote
    precisao = 2 / 3
    recall = 2 / 4
    f2_esperado = 5 * precisao * recall / (4 * precisao + recall)
    resultado = avaliar(*lote)
    assert resultado["F2"] == pytest.approx(f2_esperado)


def test_lote_sem_nenhum_positivo_nao_lanca_excecao_e_devolve_nan():
    y_true = np.zeros(6, dtype=int)
    y_pred = np.array([1, 0, 0, 1, 0, 0])
    y_proba = np.array([0.6, 0.4, 0.3, 0.55, 0.2, 0.1])

    with pytest.warns(UserWarning):
        resultado = avaliar(y_true, y_pred, y_proba)

    assert set(resultado) == set(NOMES_METRICAS)
    assert all(np.isnan(valor) for valor in resultado.values())


def test_nao_depende_de_nenhum_estimador_especifico():
    """Assinatura recebe só os três vetores: reutilizável por qualquer modelo."""
    import inspect

    parametros = list(inspect.signature(avaliar).parameters)
    assert parametros == ["y_true", "y_pred", "y_proba"]

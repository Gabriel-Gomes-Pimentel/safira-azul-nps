"""Testes unitários do scorer_f2 (card #243).

`avaliar()` (#241) já tem cobertura própria em `test_avaliacao.py`. O que faltava
era o `scorer_f2` (#242): que ele funcione dentro de um `GridSearchCV` de
verdade, que o corte de `zero_division` não emita aviso, e que o `BETA` dele
continue igual ao `BETA_F2` de `avaliacao.py` — as duas definições de F2 do
protocolo (#238) não podem divergir sem que um teste acuse.

Nada aqui lê `data/`: os dados são sintéticos, em linha com o Termo de Abertura.

Executar com:  pytest tests/test_scorer_f2.py -v
"""
import warnings

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from avaliacao import BETA_F2
from scorer_f2 import BETA, CLASSE_POSITIVA, scorer_f2


def test_beta_do_scorer_bate_com_o_beta_de_avaliar():
    """As duas implementações do F2 do protocolo (#238) usam o mesmo beta."""
    assert BETA == BETA_F2


def test_classe_positiva_e_detrator():
    assert CLASSE_POSITIVA == 1


@pytest.fixture
def dados_agrupados():
    aleatorio = np.random.RandomState(0)
    n = 600
    x = aleatorio.normal(size=(n, 4))
    y = (x[:, 0] + aleatorio.normal(scale=1.5, size=n) > 1).astype(int)
    grupos = aleatorio.randint(0, 120, size=n)
    return x, y, grupos


def test_scorer_funciona_dentro_de_gridsearchcv_agrupado(dados_agrupados):
    """O scorer precisa aceitar os folds de `validacao.criar_folds` sem erro."""
    x, y, grupos = dados_agrupados
    folds = [(np.asarray(a), np.asarray(b)) for a, b in GroupKFold(5).split(x, y, groups=grupos)]

    busca = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1.0]},
        scoring=scorer_f2,
        cv=folds,
        refit=True,
    )
    busca.fit(x, y)

    assert busca.best_score_ is not None
    assert 0.0 <= busca.best_score_ <= 1.0


def test_best_score_bate_com_o_f2_calculado_a_mao(dados_agrupados):
    """O número que o GridSearchCV escolhe precisa ser o mesmo F2 de fbeta_score."""
    x, y, grupos = dados_agrupados
    folds = [(np.asarray(a), np.asarray(b)) for a, b in GroupKFold(5).split(x, y, groups=grupos)]

    busca = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1.0]},
        scoring=scorer_f2,
        cv=folds,
        refit=True,
    )
    busca.fit(x, y)

    manual = []
    melhor = LogisticRegression(max_iter=1000, **busca.best_params_)
    for treino, validacao in folds:
        melhor.fit(x[treino], y[treino])
        manual.append(fbeta_score(y[validacao], melhor.predict(x[validacao]), beta=BETA))

    assert busca.best_score_ == pytest.approx(np.mean(manual), abs=1e-9)


def test_scorer_nao_emite_aviso_sem_positivo_previsto():
    """Um fold sem nenhum positivo previsto não pode emitir UndefinedMetricWarning."""
    x = np.zeros((10, 1))
    y = np.array([1, 0] * 5)
    modelo = DummyClassifier(strategy="constant", constant=0).fit(x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado = scorer_f2(modelo, x, y)

    assert resultado == 0.0


def test_scorer_e_um_objeto_scorer_do_sklearn():
    """`scorer_f2` precisa ser chamável como (estimador, X, y), não uma função solta."""
    assert callable(scorer_f2)

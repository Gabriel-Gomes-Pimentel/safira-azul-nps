"""Testes da grade de busca da Árvore de Decisão (#228).

Mesmo objetivo do `test_espaco_busca_logistica.py` (#207): proteger a
**validade da declaração**, não o resultado da busca. A grade é escrita antes
de o card #231 executá-la, e o teste de custo cobre a aritmética que decide se
a busca cabe na sessão do Colab.

Nada aqui lê `data/`: a fixture é sintética, em linha com o Termo de Abertura.

Executar com:  pytest tests/test_espaco_busca_arvore.py -v
"""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from espaco_busca_arvore import (
    GRADE_ARVORE,
    PESOS_DE_CLASSE,
    PROFUNDIDADE_MAXIMA,
    VALORES_CRITERION,
    VALORES_MIN_SAMPLES_LEAF,
    combinacoes,
    custo_estimado,
    grade_com_prefixo,
    medir_ajuste,
)


@pytest.fixture
def dados():
    """Matriz sintética binária, com uma interação para a árvore capturar.

    Duas colunas informativas combinadas por AND, o padrão que uma árvore
    encontra e um modelo linear só alcançaria com um termo de interação
    escrito à mão — o mesmo argumento que `modelo.py` usa para escolher
    árvores sobre um modelo aditivo.
    """
    aleatorio = np.random.RandomState(42)
    matriz = aleatorio.randn(400, 4)
    alvo = ((matriz[:, 0] > 0) & (matriz[:, 1] > 0)).astype(int)
    return matriz, alvo


def test_toda_combinacao_da_grade_ajusta_sem_erro(dados):
    """Nenhuma combinação declarada é inválida no scikit-learn."""
    matriz, alvo = dados

    for parametros in combinacoes():
        DecisionTreeClassifier(random_state=42, **parametros).fit(matriz, alvo)


def test_grade_nao_inclui_profundidade_ilimitada():
    """`max_depth=None` (sem limite) fica de fora por desenho: ver docstring do módulo."""
    assert None not in GRADE_ARVORE["max_depth"]
    assert max(GRADE_ARVORE["max_depth"]) == PROFUNDIDADE_MAXIMA


def test_grade_nao_declara_min_samples_split():
    """`min_samples_split` não é eixo separado — ver justificativa no docstring."""
    assert "min_samples_split" not in GRADE_ARVORE


def test_contagem_de_combinacoes_bate_com_o_produto_dos_eixos():
    """É este número que multiplica os folds na conta de custo."""
    esperado = (
        len(VALORES_CRITERION)
        * len(range(3, PROFUNDIDADE_MAXIMA + 1))
        * len(VALORES_MIN_SAMPLES_LEAF)
        * len(PESOS_DE_CLASSE)
    )

    assert len(combinacoes()) == esperado == 48
    assert len({tuple(sorted(c.items(), key=str)) for c in combinacoes()}) == esperado


def test_custo_recusa_tempo_nao_medido():
    """Sem medição não há conta: `None` é recusado, não tratado como zero."""
    with pytest.raises(ValueError, match="medido"):
        custo_estimado(None, n_folds=5)


def test_custo_recusa_numero_de_folds_sem_sentido():
    with pytest.raises(ValueError, match="n_folds"):
        custo_estimado(1.0, n_folds=1)


def test_custo_soma_todos_os_folds_e_o_refit():
    """combinações x folds x tempo do ajuste, mais um ajuste de refit."""
    custo = custo_estimado(2.0, n_folds=5)

    assert custo["segundos"] == 48 * 5 * 2.0 + 2.0
    assert custo["ajustes"] == 48 * 5 + 1
    assert custo["combinacoes"] == 48


def test_prefixo_do_pipeline_preserva_a_expansao():
    """O card #231 consome a mesma grade, só que com nome de passo do `Pipeline`."""
    prefixada = grade_com_prefixo("modelo")

    assert len(list(combinacoes(prefixada))) == len(combinacoes())
    for parametros in combinacoes(prefixada):
        assert all(chave.startswith("modelo__") for chave in parametros)


def test_prefixo_vazio_e_recusado():
    with pytest.raises(ValueError, match="passo"):
        grade_com_prefixo("")


def test_medicao_reporta_tempo_e_forma_da_arvore(dados):
    """A medição alimenta `custo_estimado` e também descreve a árvore obtida."""
    matriz, alvo = dados

    medicao = medir_ajuste(matriz, alvo, {"criterion": "gini", "max_depth": 3,
                                           "min_samples_leaf": 50, "class_weight": None})

    assert medicao["segundos"] > 0
    assert medicao["profundidade_obtida"] <= 3
    assert medicao["n_folhas"] >= 1

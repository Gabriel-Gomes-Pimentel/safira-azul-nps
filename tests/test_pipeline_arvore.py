"""Travas do pipeline da Árvore de Decisão (#230).

O card #229 entregou um `Pipeline` que encadeia o pré-processador do contrato e
a `DecisionTreeClassifier`. O que este arquivo protege é a razão de o
encadeamento existir — as mesmas três falhas do `test_pipeline_logistica.py`
(#209), nenhuma delas levanta exceção, e todas devolvem um número melhor do
que o real:

1. **Pré-processador ajustado fora do fold.** Se `criar_pipeline` parar de
   clonar, o objeto que o contrato já ajustou no treino inteiro entra no
   pipeline, e a mediana da imputação e os quartis do `RobustScaler` passam a
   carregar as linhas que cada fold usa como validação.
2. **Partição própria.** Uma partição própria aqui competiria com a partição
   temporal por Cliente do card 01 e com o `GroupKFold` de
   `validacao.criar_folds`.
3. **Coluna de pesquisa em `X`.** Qualquer `NPS_` ou `SUB_` só existe depois
   do instante da predição.

Os testes se dividem nos mesmos dois grupos de `test_pipeline_logistica.py`:
comportamento (sobre a fixture sintética) e origem (lendo a árvore sintática
do módulo, para a garantia valer a cada `pytest`, não a cada revisão de diff).

Tudo roda sobre fixture sintética. Nenhum teste lê `data/`, em linha com o
Termo de Abertura.

Executar com:  pytest tests/test_pipeline_arvore.py -v
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import pipeline_arvore
from espaco_busca_arvore import SEMENTE
from matriz import PREFIXOS_PROIBIDOS, preparar_matriz
from pipeline_arvore import (PASSO_MODELO, PASSO_PREPARO, ajustar_no_treino,
                              criar_pipeline, metrica_de_partida)

CORTES = {"corte_validacao": "2025-07-01", "corte_teste": "2026-01-01"}


@pytest.fixture
def base():
    """Mesma fixture sintética de `test_pipeline_logistica.py` (#209).

    Um Cliente por período, para que o desempate por recorrência não encolha
    as partições, e datas crescentes dentro de cada período. As duas últimas
    colunas são de pesquisa e entram de propósito, para a trava de vazamento
    ter o que recusar.
    """
    n = 60
    datas = (
        list(pd.date_range("2024-03-01", periods=30).astype(str))
        + list(pd.date_range("2025-09-01", periods=15).astype(str))
        + list(pd.date_range("2026-03-01", periods=15).astype(str))
    )
    return pd.DataFrame({
        "RESPONDENT_ID": range(1, n + 1),
        "ID_GOLDENRECORD": range(101, 101 + n),
        "DATA_STD": datas,
        "DETRATOR": ([1, 0] * (n // 2)),
        "TIER_VIAGEM": ["DIAMANTE", "SAFIRA"] * (n // 2),
        "VOO_TIPO": ["DIRETO", "CONEXAO"] * (n // 2),
        "TIPO_ENTRETENIMENTO": ["TELA"] * n,
        "CANAL_COMPRA": ["WEB", "AGENCIA"] * (n // 2),
        "SEGMENTO": ["CORPORATIVO", "LAZER"] * (n // 2),
        "ESTATISTICA_ATRASOSAIDA": np.arange(n, dtype=float),
        "ATRASO_CHEGADA": np.arange(n, dtype=float) * 2,
        "CANCELAMENTO_VOO": [False, True] * (n // 2),
        "ANTECEDENCIA_CANCELAMENTO": np.arange(n, dtype=float),
        "TEMPO_VOO": np.arange(n, dtype=float) + 60,
        "N_TRECHOS": (np.arange(n) % 3) + 1,
        "NPS_PRINCIPAL": [-100, 100] * (n // 2),
        "SUB_NOTA_TRIPULACAO": [1, 5] * (n // 2),
    })


@pytest.fixture
def contrato(base):
    """As saídas do contrato, exatamente como o notebook as consome."""
    return preparar_matriz(base, **CORTES)


@pytest.fixture
def treino(contrato):
    """`X` cru e `y` da partição de treino, que é o que o pipeline recebe."""
    return contrato["x"]["treino"], contrato["y"]["treino"]


FONTE = Path(pipeline_arvore.__file__).read_text(encoding="utf-8")
ARVORE = ast.parse(FONTE)


def _modulos_importados() -> set[str]:
    """Nomes de módulo que `pipeline_arvore.py` importa, em qualquer das formas."""
    modulos: set[str] = set()
    for no in ast.walk(ARVORE):
        if isinstance(no, ast.Import):
            modulos.update(alias.name for alias in no.names)
        elif isinstance(no, ast.ImportFrom) and no.module:
            modulos.add(no.module)
    return modulos


def _nomes_usados() -> set[str]:
    """Identificadores e atributos que o módulo de fato chama ou referencia."""
    nomes: set[str] = set()
    for no in ast.walk(ARVORE):
        if isinstance(no, ast.Name):
            nomes.add(no.id)
        elif isinstance(no, ast.Attribute):
            nomes.add(no.attr)
    return nomes


# ---------------------------------------------------- vazamento entre folds

def test_criar_pipeline_nao_ajusta_o_preprocessador_que_recebeu(contrato):
    """O objeto do contrato entra clonado, então ajustar o pipeline não o toca."""
    do_contrato = contrato["preprocessador"]
    pipeline = criar_pipeline(do_contrato)

    assert pipeline.named_steps[PASSO_PREPARO] is not do_contrato

    x, y = contrato["x"]["treino"], contrato["y"]["treino"]
    pipeline.fit(x, y)

    assert pipeline.named_steps[PASSO_PREPARO] is not contrato["preprocessador"]


def test_preprocessador_do_pipeline_nasce_nao_ajustado(contrato):
    """O `clone` copia a especificação e descarta o estado."""
    pipeline = criar_pipeline(contrato["preprocessador"])

    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps[PASSO_PREPARO])


def test_preprocessador_e_reajustado_com_as_linhas_de_cada_ajuste(contrato):
    """Cada `fit` do pipeline ajusta o pré-processamento só com as linhas daquele fit."""
    ajustes: list[list[int]] = []
    do_contrato = contrato["preprocessador"]

    class EspiaoDoContrato(BaseEstimator, TransformerMixin):
        def __init__(self, interno=None):
            self.interno = interno

        def fit(self, x, y=None):
            ajustes.append(list(x.index))
            self.interno.fit(x, y)
            return self

        def transform(self, x):
            return self.interno.transform(x)

    x, y = contrato["x"]["treino"], contrato["y"]["treino"]
    metade = len(x) // 2
    pipeline = criar_pipeline(EspiaoDoContrato(do_contrato))

    pipeline.fit(x.iloc[:metade], y.iloc[:metade])
    pipeline.fit(x.iloc[metade:], y.iloc[metade:])

    assert ajustes == [list(x.index[:metade]), list(x.index[metade:])]
    assert ajustes[0] != ajustes[1], "o segundo ajuste repetiu as linhas do primeiro"


def test_nenhuma_coluna_de_pesquisa_entra_no_x_do_pipeline(contrato):
    """CR03: nada derivado da resposta de NPS chega ao estimador."""
    x, y = contrato["x"]["treino"], contrato["y"]["treino"]

    assert "NPS_PRINCIPAL" in contrato["particoes"]["treino"].columns
    entrada_proibida = [c for c in x.columns if c.startswith(PREFIXOS_PROIBIDOS)]
    assert entrada_proibida == []

    pipeline = criar_pipeline(contrato["preprocessador"]).fit(x, y)
    saida = pipeline.named_steps[PASSO_PREPARO].get_feature_names_out()
    assert [c for c in saida if any(p in c for p in PREFIXOS_PROIBIDOS)] == []


def test_ajuste_descreve_a_matriz_que_o_contrato_produziu(treino, contrato):
    """A largura relatada é a do pré-processador, não um número escrito à parte."""
    x, y = treino
    relato = ajustar_no_treino(criar_pipeline(contrato["preprocessador"]), x, y)

    assert relato["n_linhas"] == len(x)
    assert relato["colunas_de_entrada"] == x.shape[1]
    assert relato["colunas_da_matriz"] == len(
        contrato["preprocessador"].get_feature_names_out()
    )


def test_ajuste_relata_profundidade_e_folhas_dentro_do_limite(treino, contrato):
    """CR02 (#230): a profundidade obtida nunca passa do `max_depth` pedido.

    É o comportamento que sustenta a Seção 4.4 dizer que a árvore final tem no
    máximo `PROFUNDIDADE_MAXIMA` níveis: quem lê o relato de `ajustar_no_treino`
    tem que poder confiar que ele reflete o limite pedido, não um número maior
    que passou despercebido.
    """
    x, y = treino
    pipeline = criar_pipeline(contrato["preprocessador"], max_depth=3)

    relato = ajustar_no_treino(pipeline, x, y)

    assert relato["profundidade_obtida"] <= 3
    assert relato["n_folhas"] >= 1


# ------------------------------------------------ nenhuma partição própria

def test_pipeline_nao_importa_selecao_de_modelo():
    """CR02: o módulo não pode tocar `sklearn.model_selection`."""
    assert not [m for m in _modulos_importados() if "model_selection" in m]


def test_pipeline_nao_instancia_particao_propria():
    """Nenhum nome de partição é sequer referenciado no código do módulo."""
    proibidos = {"KFold", "StratifiedKFold", "GroupKFold", "GroupShuffleSplit",
                 "TimeSeriesSplit", "train_test_split", "split"}
    assert not (proibidos & _nomes_usados())


def test_pipeline_nao_monta_pre_processamento_proprio():
    """O pré-processamento é o do contrato, clonado, e não um remontado aqui."""
    pacotes = ("sklearn.preprocessing", "sklearn.compose", "sklearn.impute")
    assert not [m for m in _modulos_importados() if m.startswith(pacotes)]

    usados = _nomes_usados()
    assert "fit_transform" not in usados
    assert not [n for n in usados if n.endswith(("Scaler", "Encoder", "Imputer"))]
    assert "ColumnTransformer" not in usados


def test_pipeline_nao_le_a_base_do_parceiro():
    """O módulo recebe matriz em memória; ele não abre arquivo nenhum."""
    usados = _nomes_usados()
    for proibido in ("read_parquet", "read_csv", "read_excel", "open"):
        assert proibido not in usados, f"{proibido} nao pode ser usado em pipeline_arvore.py"


def test_pipeline_nao_define_metrica_propria():
    """Métrica é do card #241, e entra em `metrica_de_partida` como argumento."""
    assert not [m for m in _modulos_importados() if m.startswith("sklearn.metrics")]

    usados = _nomes_usados()
    assert not [n for n in usados if n.endswith(("_score", "_curve", "_error"))]


# ------------------------------------------------- repasse para o `avaliar`

def test_metrica_de_partida_repassa_a_probabilidade_da_classe_positiva(treino, contrato):
    """`y_proba` tem que ser a coluna 1 de `predict_proba`, a de Detrator."""
    x, y = treino
    pipeline = criar_pipeline(contrato["preprocessador"]).fit(x, y)
    recebido = {}

    def avaliar_espiao(y_true, y_pred, y_proba):
        recebido.update(y_true=y_true, y_pred=y_pred, y_proba=y_proba)
        return {"F2": 0.5}

    devolvido = metrica_de_partida(pipeline, x, y, avaliar_espiao)

    assert np.array_equal(recebido["y_proba"], pipeline.predict_proba(x)[:, 1])
    assert np.array_equal(recebido["y_pred"], pipeline.predict(x))
    assert recebido["y_true"] is y
    assert devolvido == {"F2": 0.5}


def test_metrica_de_partida_recusa_avaliar_que_nao_e_funcao(treino, contrato):
    """Passar o dicionário de métricas no lugar da função é o engano provável."""
    x, y = treino
    pipeline = criar_pipeline(contrato["preprocessador"]).fit(x, y)

    with pytest.raises(TypeError, match="card #241"):
        metrica_de_partida(pipeline, x, y, {"F2": 0.5})


# ------------------------------------------------------- reprodutibilidade

def test_duas_construcoes_com_a_mesma_semente_dao_o_mesmo_resultado(treino, contrato):
    """CR03: mesma semente e mesma partição têm que devolver o mesmo número.

    A `DecisionTreeClassifier` usa aleatoriedade para escolher entre splits
    empatados (`splitter="best"` ainda sorteia a ordem de avaliação das
    features quando o ganho é idêntico), então sem semente fixa duas
    construções do mesmo pipeline poderiam devolver árvores diferentes — e as
    regras que o card #233 extrai deixariam de ser reproduzíveis.
    """
    x, y = treino

    primeiro = criar_pipeline(contrato["preprocessador"]).fit(x, y)
    segundo = criar_pipeline(contrato["preprocessador"]).fit(x, y)

    assert np.array_equal(primeiro.predict_proba(x), segundo.predict_proba(x))
    assert (
        primeiro.named_steps[PASSO_MODELO].tree_.node_count
        == segundo.named_steps[PASSO_MODELO].tree_.node_count
    )


def test_semente_chega_ao_estimador_e_nao_fica_implicita(contrato):
    """Semente ausente é o defeito silencioso: o resultado muda sem nada quebrar."""
    estimador = criar_pipeline(contrato["preprocessador"]).named_steps[PASSO_MODELO]
    assert estimador.random_state == SEMENTE

    outro = criar_pipeline(contrato["preprocessador"], semente=7)
    assert outro.named_steps[PASSO_MODELO].random_state == 7


def test_ajuste_repetido_relata_o_mesmo_numero_de_folhas(treino, contrato):
    """A reprodutibilidade tem que valer para o pipeline inteiro, não só para o estimador."""
    x, y = treino

    primeiro = ajustar_no_treino(criar_pipeline(contrato["preprocessador"]), x, y)
    segundo = ajustar_no_treino(criar_pipeline(contrato["preprocessador"]), x, y)

    assert primeiro["n_folhas"] == segundo["n_folhas"]
    assert primeiro["profundidade_obtida"] == segundo["profundidade_obtida"]
    assert primeiro["colunas_da_matriz"] == segundo["colunas_da_matriz"]

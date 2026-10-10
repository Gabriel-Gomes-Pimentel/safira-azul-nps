"""Testes da tabela comparativa final entre as quatro duplas (card 14A, #249).

O que estes testes protegem: que a tabela final reaproveita `modelo.tabela_comparativa`
em vez de recalcular o limiar por conta própria (CR03), que Sensibilidade,
Precisão Média e ROC-AUC vêm de `avaliar()` e não de uma fórmula própria (CR01),
que o corte usado é o de capacidade de contato e não o que maximizaria F2 (CR02),
que um modelo sem vencedor ainda (`None`) vira linha com `nan`, não exceção, e
que um `Pipeline` com `ColumnTransformer` próprio (a forma real dos quatro
candidatos, #187/#208) recebe o `DataFrame` cru que ele espera, e não a matriz
transformada de outro modelo (revisão da !120).

Nada aqui lê `data/`: os dados são sintéticos, no mesmo desenho de
`tests/test_modelo.py`.

Executar com:  pytest tests/test_tabela_final.py -v
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from avaliacao import avaliar
from tabela_final import COLUNAS_METRICAS, consolidar_tabela_final


class _ScoreOrdenado(BaseEstimator):
    """Score que cresce com a primeira coluna, entao ordena de verdade."""

    def fit(self, x, y=None):
        return self

    def predict_proba(self, x):
        bruto = np.asarray(x)[:, 0]
        p = (bruto - bruto.min()) / (bruto.max() - bruto.min() + 1e-9)
        p = np.clip(p * 0.98 + 0.01, 0, 1)
        return np.column_stack([1 - p, p])


class _ScoreFixo(BaseEstimator):
    """Emite sempre a mesma probabilidade, como um piso trivial."""

    def __init__(self, valor=0.2):
        self.valor = valor

    def fit(self, x, y=None):
        return self

    def predict_proba(self, x):
        p = np.full(len(x), self.valor)
        return np.column_stack([1 - p, p])


@pytest.fixture
def teste_sintetico():
    rng = np.random.default_rng(7)
    n = 400
    y = pd.Series(rng.binomial(1, 0.25, n))
    x = np.column_stack([y + rng.normal(0, 0.5, n), rng.normal(0, 1, n)])
    return x, y


def test_uma_linha_por_modelo_incluindo_ausente(teste_sintetico):
    x, y = teste_sintetico
    modelos = {"ordenado": _ScoreOrdenado().fit(x, y), "arvore": None}

    tabela = consolidar_tabela_final(modelos, {"ordenado": x}, y, k=100)

    assert list(tabela.index) == ["ordenado", "arvore"]
    assert tabela.loc["ordenado", "disponivel"] == True  # noqa: E712
    assert tabela.loc["arvore", "disponivel"] == False  # noqa: E712


def test_modelo_ausente_fica_em_nan_sem_lancar_excecao(teste_sintetico):
    """CR do card: um vencedor que ainda nao existe nao pode derrubar a tabela inteira."""
    x, y = teste_sintetico
    modelos = {"ordenado": _ScoreOrdenado().fit(x, y), "random_forest": None, "arvore": None}

    tabela = consolidar_tabela_final(modelos, {"ordenado": x}, y, k=100)

    for nome in ("random_forest", "arvore"):
        for coluna in COLUNAS_METRICAS:
            assert np.isnan(tabela.loc[nome, coluna])


def test_tabela_recusa_dicionario_vazio(teste_sintetico):
    x, y = teste_sintetico
    with pytest.raises(ValueError, match="nenhum modelo"):
        consolidar_tabela_final({}, {}, y, k=10)


def test_recusa_x_teste_sem_entrada_para_modelo_disponivel(teste_sintetico):
    """Contrato de entrada: cada modelo disponivel precisa da propria matriz em x_teste."""
    x, y = teste_sintetico
    with pytest.raises(KeyError, match="ordenado"):
        consolidar_tabela_final({"ordenado": _ScoreOrdenado().fit(x, y)}, {}, y, k=100)


def test_colunas_de_metrica_tem_os_nomes_exatos_de_avaliar(teste_sintetico):
    """CR01: a tabela do 18A.1 depende desses nomes, nao de sinonimo."""
    x, y = teste_sintetico
    tabela = consolidar_tabela_final({"ordenado": _ScoreOrdenado().fit(x, y)}, {"ordenado": x}, y, k=100)

    assert set(COLUNAS_METRICAS) <= set(tabela.columns)
    assert COLUNAS_METRICAS == ("F2", "Sensibilidade", "Precisão Média", "ROC-AUC")


def test_sensibilidade_precisao_media_roc_auc_vem_de_avaliar(teste_sintetico):
    """CR01: as tres metricas sao o que `avaliar()` devolveria no mesmo limiar, nao outra formula."""
    x, y = teste_sintetico
    estimador = _ScoreOrdenado().fit(x, y)

    tabela = consolidar_tabela_final({"ordenado": estimador}, {"ordenado": x}, y, k=100)

    score = estimador.predict_proba(x)[:, 1]
    limiar = tabela.loc["ordenado", "limiar"]
    y_pred = (score >= limiar).astype(int)
    esperado = avaliar(y, y_pred, score)

    assert tabela.loc["ordenado", "Sensibilidade"] == pytest.approx(esperado["Sensibilidade"])
    assert tabela.loc["ordenado", "Precisão Média"] == pytest.approx(esperado["Precisão Média"])
    assert tabela.loc["ordenado", "ROC-AUC"] == pytest.approx(esperado["ROC-AUC"])


def test_avaliar_recebe_y_pred_e_y_proba_certos(teste_sintetico):
    """Contraprova da funcao acima: um duble confere exatamente o que chega em `avaliar`."""
    x, y = teste_sintetico
    estimador = _ScoreOrdenado().fit(x, y)
    chamadas = []

    def avaliar_espiao(y_true, y_pred, y_proba):
        chamadas.append({"y_true": y_true, "y_pred": y_pred, "y_proba": y_proba})
        return {"F2": 0.0, "Sensibilidade": 0.0, "Precisão Média": 0.0, "ROC-AUC": 0.0}

    consolidar_tabela_final(
        {"ordenado": estimador}, {"ordenado": x}, y, k=100, avaliar=avaliar_espiao
    )

    assert len(chamadas) == 1
    score = estimador.predict_proba(x)[:, 1]
    np.testing.assert_array_equal(chamadas[0]["y_proba"], score)
    assert set(np.unique(chamadas[0]["y_pred"])) <= {0, 1}
    np.testing.assert_array_equal(np.asarray(chamadas[0]["y_true"]), np.asarray(y))


def test_limiar_e_o_mesmo_de_tabela_comparativa_para_a_mesma_capacidade(teste_sintetico):
    """CR02/CR03: o corte nao e recalculado aqui, e o de `modelo.tabela_comparativa`."""
    from modelo import tabela_comparativa

    x, y = teste_sintetico
    estimador = _ScoreOrdenado().fit(x, y)

    tabela = consolidar_tabela_final({"ordenado": estimador}, {"ordenado": x}, y, k=100)
    referencia = tabela_comparativa({"ordenado": estimador}, x, y, k=100)

    assert tabela.loc["ordenado", "limiar"] == pytest.approx(referencia.loc["ordenado", "limiar"])
    assert tabela.loc["ordenado", "n_na_fila"] == referencia.loc["ordenado", "n_na_fila"]


def test_mesma_capacidade_k_para_todos_os_modelos(teste_sintetico):
    """CR02: nao e o mesmo limiar numerico, e a mesma fila de k contatos."""
    x, y = teste_sintetico
    modelos = {
        "trivial": _ScoreFixo(valor=0.5).fit(x, y),
        "ordenado": _ScoreOrdenado().fit(x, y),
    }

    tabela = consolidar_tabela_final(modelos, {"trivial": x, "ordenado": x}, y, k=100)

    assert tabela.loc["trivial", "n_na_fila"] == len(y)  # score constante: fila degenerada
    assert tabela.loc["ordenado", "n_na_fila"] == 100
    assert tabela.loc["trivial", "fila_comparavel"] == False  # noqa: E712
    assert tabela.loc["ordenado", "fila_comparavel"] == True  # noqa: E712


def test_corte_nao_e_o_que_maximizaria_f2(teste_sintetico):
    """CR02: o limiar sai da capacidade de contato, nunca de uma busca por F2."""
    from sklearn.metrics import fbeta_score

    x, y = teste_sintetico
    estimador = _ScoreOrdenado().fit(x, y)
    score = estimador.predict_proba(x)[:, 1]

    tabela = consolidar_tabela_final({"ordenado": estimador}, {"ordenado": x}, y, k=50)
    limiar_capacidade = tabela.loc["ordenado", "limiar"]

    candidatos = np.unique(score)
    f2_por_limiar = {
        c: fbeta_score(y, (score >= c).astype(int), beta=2) for c in candidatos
    }
    limiar_que_maximiza_f2 = max(f2_por_limiar, key=f2_por_limiar.get)

    assert tabela.loc["ordenado", "n_na_fila"] == 50
    # O ponto de maxima F2 quase nunca cai exatamente na fila de 50 contatos.
    assert limiar_capacidade != pytest.approx(limiar_que_maximiza_f2)


def test_f2_e_apenas_registrado_nao_recalculado(teste_sintetico):
    """Descricao do card: F2 'so como registro do criterio de busca'."""
    x, y = teste_sintetico
    estimador = _ScoreOrdenado().fit(x, y)

    tabela = consolidar_tabela_final(
        {"ordenado": estimador}, {"ordenado": x}, y, k=100, f2_da_busca={"ordenado": 0.4242},
    )

    assert tabela.loc["ordenado", "F2"] == pytest.approx(0.4242)


def test_f2_sem_registro_fica_nan(teste_sintetico):
    x, y = teste_sintetico
    tabela = consolidar_tabela_final(
        {"ordenado": _ScoreOrdenado().fit(x, y)}, {"ordenado": x}, y, k=100, f2_da_busca={},
    )
    assert np.isnan(tabela.loc["ordenado", "F2"])


def test_reaproveita_tabela_comparativa_em_vez_de_duplicar(teste_sintetico, monkeypatch):
    """CR03: a montagem do limiar e da fila precisa vir de `modelo.tabela_comparativa`."""
    import tabela_final

    x, y = teste_sintetico
    chamadas = []
    original = tabela_final.modelo.tabela_comparativa

    def espiao(*args, **kwargs):
        chamadas.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(tabela_final.modelo, "tabela_comparativa", espiao)

    consolidar_tabela_final({"ordenado": _ScoreOrdenado().fit(x, y)}, {"ordenado": x}, y, k=100)

    assert len(chamadas) == 1


# --------------------------------------------- integracao com pipeline real (!120)
# Reproduz o defeito apontado na revisao: um Pipeline com ColumnTransformer
# proprio (a forma real dos quatro candidatos, #187/#208) espera o DataFrame
# cru, e nao a matriz ja transformada que outro modelo usa.


@pytest.fixture
def contrato_com_pipeline():
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({
        "ATRASO_CHEGADA": rng.normal(20, 10, n),
        "TEMPO_VOO": rng.normal(120, 30, n),
    })
    y = pd.Series((df["ATRASO_CHEGADA"] + rng.normal(0, 5, n) > 20).astype(int))

    preprocessador = ColumnTransformer([
        ("numericas", StandardScaler(), ["ATRASO_CHEGADA", "TEMPO_VOO"]),
    ])
    pipeline = Pipeline([
        ("preparo", preprocessador),
        ("modelo", LogisticRegression(random_state=42)),
    ])
    pipeline.fit(df, y)

    # O "primeiro candidato" da Secao 4.3.2: estimador desnudo, ajustado direto
    # sobre a matriz ja transformada, sem Pipeline por cima.
    x_transformado = preprocessador.transform(df)
    desnudo = LogisticRegression(random_state=42).fit(x_transformado, y)

    return {
        "df_cru": df,
        "y": y,
        "pipeline": pipeline,
        "x_transformado": x_transformado,
        "desnudo": desnudo,
    }


def test_pipeline_com_preprocessamento_recebe_o_dataframe_cru(contrato_com_pipeline):
    """Regressao da !120: passar a matriz transformada a um Pipeline com
    ColumnTransformer proprio levantava
    'Specifying the columns using strings is only supported for dataframes'."""
    c = contrato_com_pipeline

    tabela = consolidar_tabela_final(
        {"logistica": c["pipeline"]},
        {"logistica": c["df_cru"]},
        c["y"],
        k=50,
    )

    assert tabela.loc["logistica", "disponivel"] == True  # noqa: E712
    assert not np.isnan(tabela.loc["logistica", "Sensibilidade"])


def test_pipeline_e_estimador_desnudo_juntos_cada_um_com_sua_matriz(contrato_com_pipeline):
    """O caso real do #249: o vencedor da Regressao Logistica (Pipeline) ao lado
    do primeiro candidato da Secao 4.3.2 (estimador desnudo), cada um com a
    matriz que espera."""
    c = contrato_com_pipeline

    tabela = consolidar_tabela_final(
        {"logistica_211": c["pipeline"], "primeiro_candidato": c["desnudo"]},
        {"logistica_211": c["df_cru"], "primeiro_candidato": c["x_transformado"]},
        c["y"],
        k=50,
    )

    assert list(tabela.index) == ["logistica_211", "primeiro_candidato"]
    for nome in tabela.index:
        assert tabela.loc[nome, "disponivel"] == True  # noqa: E712
        assert not np.isnan(tabela.loc[nome, "Sensibilidade"])


def test_pipeline_com_matriz_errada_ainda_falha_de_forma_clara(contrato_com_pipeline):
    """Contraprova: a matriz transformada continua incompativel com o Pipeline
    que tem ColumnTransformer proprio — o contrato de entrada exige a matriz
    certa por modelo, e nao adivinha por conta propria."""
    c = contrato_com_pipeline

    with pytest.raises(ValueError):
        consolidar_tabela_final(
            {"logistica": c["pipeline"]},
            {"logistica": c["x_transformado"]},
            c["y"],
            k=50,
        )

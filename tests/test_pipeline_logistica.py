"""Travas do pipeline da Regressão Logística (#209).

O card #208 entregou um `Pipeline` que encadeia o pré-processador do contrato e a
`LogisticRegression`. O que este arquivo protege é a razão de o encadeamento
existir: nenhuma das três falhas abaixo levanta exceção, e todas as três
devolvem um número melhor do que o real.

1. **Pré-processador ajustado fora do fold.** Se `criar_pipeline` parar de
   clonar, o objeto que o contrato já ajustou no treino inteiro entra no
   pipeline, e a mediana da imputação e os quartis do `RobustScaler` passam a
   carregar as linhas que cada fold usa como validação.
2. **Partição própria.** Um `KFold` ou um `train_test_split` dentro do módulo
   seria uma terceira divisão competindo com a partição temporal por Cliente do
   card 01 e com o `GroupKFold` de `validacao.criar_folds`.
3. **Coluna de pesquisa em `X`.** Qualquer `NPS_` ou `SUB_` só existe depois do
   instante da predição; usá-la é prever a resposta com a própria resposta.

Os testes se dividem em dois grupos, como em
`tests/test_fumaca_regressao_logistica.py`. Os de comportamento exercitam o
pipeline sobre a fixture sintética. Os de origem leem a árvore sintática do
módulo e recusam o que ele não pode sequer importar, porque a garantia precisa
valer a cada `pytest` e não a cada revisão de diff.

Tudo roda sobre fixture sintética. Nenhum teste lê `data/`, em linha com o Termo
de Abertura, que veda versionar ou publicar base do parceiro.

Executar com:  pytest tests/test_pipeline_logistica.py -v
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import pipeline_logistica
from espaco_busca_logistica import MAX_ITER, SEMENTE
from matriz import PREFIXOS_PROIBIDOS, preparar_matriz
from pipeline_logistica import (PASSO_MODELO, PASSO_PREPARO, ajustar_no_treino,
                                criar_pipeline, metrica_de_partida)

CORTES = {"corte_validacao": "2025-07-01", "corte_teste": "2026-01-01"}


@pytest.fixture
def base():
    """Base sintética com a allowlist completa e os três períodos cobertos.

    Mesma forma da fixture de `tests/test_fumaca_regressao_logistica.py`: um
    Cliente por período, para que o desempate por recorrência não encolha as
    partições, e datas crescentes dentro de cada período, para que a verificação
    de anterioridade não recuse uma base que na prática está ordenada.

    As duas últimas colunas são de pesquisa e entram de propósito: é sobre elas
    que a trava de vazamento do card tem alguma coisa para recusar.
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


FONTE = Path(pipeline_logistica.__file__).read_text(encoding="utf-8")
ARVORE = ast.parse(FONTE)


def _modulos_importados() -> set[str]:
    """Nomes de módulo que `pipeline_logistica.py` importa, em qualquer das formas."""
    modulos: set[str] = set()
    for no in ast.walk(ARVORE):
        if isinstance(no, ast.Import):
            modulos.update(alias.name for alias in no.names)
        elif isinstance(no, ast.ImportFrom) and no.module:
            modulos.add(no.module)
    return modulos


def _nomes_usados() -> set[str]:
    """Identificadores e atributos que o módulo de fato chama ou referencia.

    Lê da árvore sintática, e não do texto, porque as docstrings do módulo citam
    `KFold`, `train_test_split`, `RobustScaler` e `ColumnTransformer` de
    propósito, para explicar o que ele não faz. Uma busca textual confundiria a
    explicação com o uso e proibiria justamente a documentação que deixa a
    fronteira clara.
    """
    nomes: set[str] = set()
    for no in ast.walk(ARVORE):
        if isinstance(no, ast.Name):
            nomes.add(no.id)
        elif isinstance(no, ast.Attribute):
            nomes.add(no.attr)
    return nomes


# ---------------------------------------------------- vazamento entre folds

def test_criar_pipeline_nao_ajusta_o_preprocessador_que_recebeu(contrato):
    """O objeto do contrato entra clonado, então ajustar o pipeline não o toca.

    Esta é a trava do CR01. Se o `clone` sair de `criar_pipeline`, o
    `ColumnTransformer` que `preparar_matriz` já ajustou sobre o treino inteiro
    passa a ser o mesmo objeto de dentro do pipeline: cada `fit` de fold o
    reajusta, e o objeto que o notebook segurava muda por baixo, sem aviso.
    """
    do_contrato = contrato["preprocessador"]
    pipeline = criar_pipeline(do_contrato)

    assert pipeline.named_steps[PASSO_PREPARO] is not do_contrato

    x, y = contrato["x"]["treino"], contrato["y"]["treino"]
    pipeline.fit(x, y)

    # O template segue ajustado como o contrato o deixou, e o passo do pipeline
    # e ele sao objetos diferentes: o `fit` acima nao voltou para o contrato.
    assert pipeline.named_steps[PASSO_PREPARO] is not contrato["preprocessador"]


def test_preprocessador_do_pipeline_nasce_nao_ajustado(contrato):
    """O `clone` copia a especificação e descarta o estado.

    Um pipeline que nascesse já ajustado pularia o primeiro `fit` do fold em
    silêncio: o `GridSearchCV` chamaria `fit`, o objeto reajustaria, e nada
    denunciaria que o ponto de partida vinha do treino inteiro.
    """
    pipeline = criar_pipeline(contrato["preprocessador"])

    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps[PASSO_PREPARO])


def test_preprocessador_e_reajustado_com_as_linhas_de_cada_ajuste(contrato):
    """Cada `fit` do pipeline ajusta o pré-processamento só com as linhas daquele fit.

    É o comportamento que sustenta a afirmação de ausência de vazamento na Seção
    4.4: dentro do `GroupKFold`, cada fold ajusta com os seus quatro quintos, e
    nunca com o quinto que vai medir.
    """
    ajustes: list[list[int]] = []
    do_contrato = contrato["preprocessador"]

    class EspiaoDoContrato(BaseEstimator, TransformerMixin):
        """Embrulha o pré-processador do contrato e registra o índice de cada `fit`.

        `ajustes` é variável de fechamento, e não parâmetro de `__init__`, porque
        `sklearn.base.clone` copia os parâmetros em profundidade: um registro
        passado como parâmetro viraria uma lista nova dentro do pipeline e o
        teste não veria nada.
        """

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
    """CR04: nada derivado da resposta de NPS chega ao estimador.

    A conferência é nas duas pontas, como em `matriz.conferir_contrato_da_matriz`:
    na entrada, para pegar uma allowlist alterada; na saída do pré-processador,
    para pegar uma transformação que reintroduza a coluna sob outro nome. A
    fixture traz `NPS_PRINCIPAL` e `SUB_NOTA_TRIPULACAO` justamente para que a
    asserção tenha o que recusar.
    """
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


def test_ajuste_que_para_no_limite_reporta_nao_convergencia(treino, contrato):
    """Truncar é achado a registrar, não exceção, e não pode passar por convergido.

    O card #212 lê os coeficientes deste modelo como odds ratio: coeficiente de
    ajuste truncado é provisório, e interpretá-lo como efeito da operação seria
    ler ruído de otimização.
    """
    x, y = treino
    pipeline = criar_pipeline(contrato["preprocessador"], max_iter=1)

    relato = ajustar_no_treino(pipeline, x, y)

    assert relato["convergiu"] is False
    assert relato["aviso"]
    assert relato["n_iter"] == 1


# ------------------------------------------------ nenhuma partição própria

def test_pipeline_nao_importa_selecao_de_modelo():
    """CR02: o módulo não pode tocar `sklearn.model_selection`.

    É de lá que sairia `KFold`, `StratifiedKFold`, `GroupKFold` ou
    `train_test_split`. A partição temporal por Cliente vem congelada do card 01
    e os folds são os de `validacao.criar_folds`; uma divisão criada aqui seria
    uma terceira, competindo com as duas que o grupo acordou. Barrar o pacote
    inteiro é mais firme do que barrar nome por nome, porque cobre também o que
    for adicionado lá depois.
    """
    assert not [m for m in _modulos_importados() if "model_selection" in m]


def test_pipeline_nao_instancia_particao_propria():
    """Nenhum nome de partição é sequer referenciado no código do módulo.

    Complementa o teste de import: pega o caso de alguém alcançar a classe por
    outro caminho, como `sklearn.model_selection.KFold` escrito por atributo a
    partir de um `import sklearn` solto.
    """
    proibidos = {"KFold", "StratifiedKFold", "GroupKFold", "GroupShuffleSplit",
                 "TimeSeriesSplit", "train_test_split", "split"}
    assert not (proibidos & _nomes_usados())


def test_pipeline_nao_monta_pre_processamento_proprio():
    """O pré-processamento é o do contrato, clonado, e não um remontado aqui.

    Remontar o `ColumnTransformer` produziria uma segunda definição do contrato,
    parecida o bastante para passar despercebida e diferente o bastante para
    invalidar a comparação da Seção 4.4.
    """
    pacotes = ("sklearn.preprocessing", "sklearn.compose", "sklearn.impute")
    assert not [m for m in _modulos_importados() if m.startswith(pacotes)]

    usados = _nomes_usados()
    assert "fit_transform" not in usados
    assert not [n for n in usados if n.endswith(("Scaler", "Encoder", "Imputer"))]
    assert "ColumnTransformer" not in usados


def test_pipeline_nao_le_a_base_do_parceiro():
    """O módulo recebe matriz em memória; ele não abre arquivo nenhum.

    `Path` é permitido aqui, ao contrário de `fumaca_logistica`, porque o módulo
    o usa para localizar `src/` e importar o vizinho `espaco_busca_logistica` no
    Colab. O que se proíbe é leitura de dado, não manipulação de caminho.
    """
    usados = _nomes_usados()
    for proibido in ("read_parquet", "read_csv", "read_excel", "open"):
        assert proibido not in usados, f"{proibido} nao pode ser usado em pipeline_logistica.py"


def test_pipeline_nao_define_metrica_propria():
    """Métrica é do card #241, e entra em `metrica_de_partida` como argumento.

    Uma métrica escrita aqui seria a versão divergente que o card 05 existe para
    impedir: a tabela comparativa do card 18A.1 depende de as quatro duplas
    lerem o mesmo número da mesma implementação.
    """
    assert not [m for m in _modulos_importados() if m.startswith("sklearn.metrics")]

    usados = _nomes_usados()
    assert not [n for n in usados if n.endswith(("_score", "_curve", "_error"))]


# ------------------------------------------------- repasse para o `avaliar`

def test_metrica_de_partida_repassa_a_probabilidade_da_classe_positiva(treino, contrato):
    """`y_proba` tem que ser a coluna 1 de `predict_proba`, a de Detrator.

    Precisão Média e ROC-AUC são calculadas sobre o score contínuo, não sobre o
    rótulo. A coluna 0 devolveria um número válido e errado, e errado na direção
    que não chama atenção: o ROC-AUC viraria o complemento do verdadeiro.
    """
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
    # O dicionario volta como `avaliar` devolveu, sem chave renomeada nem metrica
    # acrescentada: quem define o vocabulario da tabela do 18A.1 e o card #241.
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

    Os dois pipelines são construídos do zero, e não reaproveitados, porque é
    isso que o card #210 faz a cada ponto da grade e o que quem revisa faz ao
    tentar repetir o resultado do notebook. Se divergirem, a comparação da Seção
    4.4 deixa de ser entre modelos e passa a ser entre execuções.
    """
    x, y = treino

    primeiro = criar_pipeline(contrato["preprocessador"]).fit(x, y)
    segundo = criar_pipeline(contrato["preprocessador"]).fit(x, y)

    assert np.array_equal(primeiro.predict_proba(x), segundo.predict_proba(x))
    assert np.array_equal(
        primeiro.named_steps[PASSO_MODELO].coef_,
        segundo.named_steps[PASSO_MODELO].coef_,
    )


def test_reprodutibilidade_vale_no_solver_que_embaralha(treino, contrato):
    """O `liblinear` embaralha internamente, então é nele que a semente trabalha.

    Metade da grade do card #207 é `liblinear` com L1. No `lbfgs` a igualdade
    acima sairia de graça, porque ele é determinístico: testar só com o padrão
    deixaria a trava passar por vacuidade justamente na metade da busca em que
    ela importa.
    """
    x, y = treino
    ajuste = {"solver": "liblinear", "l1_ratio": 1.0}

    primeiro = criar_pipeline(contrato["preprocessador"], **ajuste).fit(x, y)
    segundo = criar_pipeline(contrato["preprocessador"], **ajuste).fit(x, y)

    assert np.array_equal(
        primeiro.named_steps[PASSO_MODELO].coef_,
        segundo.named_steps[PASSO_MODELO].coef_,
    )


def test_semente_chega_ao_estimador_e_nao_fica_implicita(contrato):
    """Semente ausente é o defeito silencioso: o resultado muda sem nada quebrar."""
    estimador = criar_pipeline(contrato["preprocessador"]).named_steps[PASSO_MODELO]
    assert estimador.random_state == SEMENTE

    outro = criar_pipeline(contrato["preprocessador"], semente=7)
    assert outro.named_steps[PASSO_MODELO].random_state == 7


def test_max_iter_do_pipeline_e_o_que_a_grade_do_207_fixa(contrato):
    """O limite de iterações é um número só, declarado no card #207.

    Se o pipeline nascesse com um `max_iter` diferente do que a grade fixa, a
    mesma combinação convergiria ou truncaria dependendo de quem construiu o
    objeto, e o resultado do card #210 deixaria de ser reproduzível a partir do
    que está escrito.
    """
    estimador = criar_pipeline(contrato["preprocessador"]).named_steps[PASSO_MODELO]
    assert estimador.max_iter == MAX_ITER


def test_ajuste_repetido_relata_o_mesmo_numero_de_iteracoes(treino, contrato):
    """A reprodutibilidade tem que valer para o pipeline inteiro, não só para o estimador.

    `ajustar_no_treino` ajusta o pré-processamento junto: se a imputação ou a
    escala variassem entre execuções, o número de iterações até convergir
    mudaria, mesmo com a semente do estimador fixa.
    """
    x, y = treino

    primeiro = ajustar_no_treino(criar_pipeline(contrato["preprocessador"]), x, y)
    segundo = ajustar_no_treino(criar_pipeline(contrato["preprocessador"]), x, y)

    assert primeiro["n_iter"] == segundo["n_iter"]
    assert primeiro["convergiu"] == segundo["convergiu"]
    assert primeiro["colunas_da_matriz"] == segundo["colunas_da_matriz"]

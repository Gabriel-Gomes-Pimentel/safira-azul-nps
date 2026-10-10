"""Testes do primeiro modelo candidato da secao 4.3.

O que estes testes protegem nao e a metrica do modelo, que depende da base e sai
no notebook: e o que faria essa metrica mentir sem quebrar nada. Sao tres riscos,
e cada um deles devolveria um numero melhor, nunca um erro.

1. Hiperparametro implicito. `early_stopping="auto"` liga a parada antecipada
   sozinha acima de dez mil linhas e separa uma fatia aleatoria do ajuste, que
   ignora o agrupamento por Cliente da secao 3.
2. Pre-processador ajustado fora do fold. A mediana da imputacao e a escala
   passariam a carregar informacao das linhas usadas como validacao naquele fold.
3. Modelo reaproveitado entre folds. O segundo fold comecaria do ajuste do
   primeiro e a media dos cinco mediria quem ja viu quase todo o treino.

Todos usam dados sinteticos, sem tocar nas bases da Azul.

Executar com:  pytest tests/test_modelo.py -v
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from modelo import (GRADE_HIPERPARAMETROS, HIPERPARAMETROS_CANDIDATO,
                    METRICA_PRINCIPAL, SEMENTE_PADRAO, avaliar_nos_folds,
                    buscar_hiperparametros, combinacoes_da_grade,
                    comparar_nos_folds, conferir_ganho_sobre_os_pisos,
                    criar_candidato, escolher_configuracao,
                    limiar_por_capacidade, matriz_de_confusao,
                    metricas_de_ordenacao, metricas_no_teste,
                    metricas_no_topo, tabela_de_capacidade)


@pytest.fixture
def treino():
    """60 respostas com sinal previsivel, indice fora do RangeIndex.

    O indice comeca em 500 de proposito: as particoes de `preparar_matriz`
    preservam o indice original da base, entao um `.loc` no lugar de `.iloc`
    quebraria aqui em vez de selecionar as linhas erradas em silencio.
    """
    n = 60
    x = pd.DataFrame(
        {"ATRASO": np.arange(n, dtype=float), "VIAGENS": np.arange(n, dtype=float) % 7},
        index=range(500, 500 + n),
    )
    y = pd.Series([0, 1] * (n // 2), index=x.index, name="DETRATOR")
    return x, y


@pytest.fixture
def folds():
    """Tres folds com as duas classes na validacao de cada um."""
    posicoes = np.arange(60)
    return [
        (posicoes[posicoes % 3 != k], posicoes[posicoes % 3 == k]) for k in range(3)
    ]


class PreprocessadorEspiao(BaseEstimator, TransformerMixin):
    """Escalonador que registra quantas vezes foi ajustado e com quantas linhas."""

    def __init__(self):
        self.ajustes = []

    def fit(self, x, y=None):
        self.ajustes.append(len(x))
        self.media_ = np.asarray(x).mean(axis=0)
        return self

    def transform(self, x):
        return np.asarray(x) - self.media_


def test_candidato_declara_todos_os_hiperparametros_da_configuracao():
    """DoD: nenhum hiperparametro fica no padrao implicito da biblioteca."""
    parametros = criar_candidato().get_params()
    for nome, valor in HIPERPARAMETROS_CANDIDATO.items():
        assert parametros[nome] == valor, f"{nome} nao chegou ao estimador"


def test_parada_antecipada_desligada():
    """No padrao "auto" a biblioteca separaria uma fatia aleatoria do ajuste.

    Essa fatia nao respeita `ID_GOLDENRECORD`, entao respostas da mesma pessoa
    cairiam no ajuste e na medicao interna: o vazamento que a secao 3 evita.
    """
    assert criar_candidato().get_params()["early_stopping"] is False


def test_semente_fixada_e_repassada_ao_estimador():
    assert criar_candidato().get_params()["random_state"] == SEMENTE_PADRAO
    assert criar_candidato(semente=7).get_params()["random_state"] == 7


def test_ajuste_fora_da_configuracao_declarada_e_recusado():
    """A busca do #104 varia o que esta declarado, nao inventa parametro solto."""
    with pytest.raises(ValueError, match="fora da configuracao declarada"):
        criar_candidato(profundidade_maxima=3)


def test_ajuste_declarado_sobrescreve_sem_alterar_a_configuracao(treino):
    original = dict(HIPERPARAMETROS_CANDIDATO)
    assert criar_candidato(max_iter=5).get_params()["max_iter"] == 5
    assert HIPERPARAMETROS_CANDIDATO == original


def test_mesma_semente_produz_o_mesmo_score(treino):
    """CR: rodar duas vezes seguidas tem que dar o mesmo numero."""
    x, y = treino
    ajuste = {"max_iter": 5, "min_samples_leaf": 5}
    primeiro = criar_candidato(**ajuste).fit(x, y).predict_proba(x)[:, 1]
    segundo = criar_candidato(**ajuste).fit(x, y).predict_proba(x)[:, 1]
    assert np.array_equal(primeiro, segundo)


def test_preprocessador_e_reajustado_dentro_de_cada_fold(treino, folds):
    """O template recebido nao pode sair ajustado, nem ser ajustado uma vez so."""
    x, y = treino
    espiao = PreprocessadorEspiao()

    avaliar_nos_folds(lambda: DummyClassifier(strategy="prior"), x, y, folds, espiao)

    assert espiao.ajustes == [], "o template foi ajustado em vez de clonado"
    assert not hasattr(espiao, "media_")


def test_cada_fold_ajusta_apenas_com_as_linhas_do_proprio_ajuste(treino, folds):
    x, y = treino
    vistos = []

    class Fabrica(BaseEstimator):
        """Classificador minimo que registra quantas linhas viu no ajuste."""

        def fit(self, x, y=None):
            vistos.append(len(x))
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, x):
            return np.tile([0.5, 0.5], (len(x), 1))

    avaliar_nos_folds(Fabrica, x, y, folds, StandardScaler())

    assert vistos == [len(ajuste) for ajuste, _ in folds]
    assert sum(vistos) < len(x) * len(folds), "algum fold ajustou com o treino inteiro"


def test_avaliacao_devolve_uma_linha_por_fold(treino, folds):
    x, y = treino
    tabela = avaliar_nos_folds(
        lambda: criar_candidato(max_iter=5, min_samples_leaf=5), x, y, folds,
        StandardScaler(),
    )
    assert list(tabela.index) == [1, 2, 3]
    assert {METRICA_PRINCIPAL, "roc_auc"} <= set(tabela.columns)
    assert tabela["n_validacao"].sum() == len(x)


def test_lista_de_folds_vazia_e_recusada(treino):
    x, y = treino
    with pytest.raises(ValueError, match="nenhum fold"):
        avaliar_nos_folds(DummyClassifier, x, y, [], StandardScaler())


def test_alvo_de_tamanho_diferente_e_recusado(treino, folds):
    x, y = treino
    with pytest.raises(ValueError, match="mesmo tamanho"):
        avaliar_nos_folds(DummyClassifier, x, y.iloc[:-1], folds, StandardScaler())


def test_comparacao_traz_media_e_desvio_de_cada_modelo(treino, folds):
    x, y = treino
    comparacao = comparar_nos_folds(
        {
            "trivial": lambda: DummyClassifier(strategy="prior"),
            "candidato": lambda: criar_candidato(max_iter=5, min_samples_leaf=5),
        },
        x, y, folds, StandardScaler(),
    )
    assert list(comparacao.index) == ["trivial", "candidato"]
    assert f"{METRICA_PRINCIPAL}_desvio" in comparacao.columns


def test_trava_do_cr02_falha_quando_o_candidato_nao_supera_o_piso():
    comparacao = pd.DataFrame(
        {METRICA_PRINCIPAL: [0.30, 0.42]},
        index=pd.Index(["candidato", "logistica"], name="modelo"),
    )
    with pytest.raises(AssertionError, match="nao supera"):
        conferir_ganho_sobre_os_pisos(comparacao, "candidato", ["logistica"])


def test_trava_do_cr02_passa_quando_o_candidato_supera_os_dois_pisos():
    comparacao = pd.DataFrame(
        {METRICA_PRINCIPAL: [0.55, 0.42, 0.20]},
        index=pd.Index(["candidato", "logistica", "trivial"], name="modelo"),
    )
    conferir_ganho_sobre_os_pisos(comparacao, "candidato", ["logistica", "trivial"])


def test_trava_do_cr02_recusa_modelo_ausente_da_comparacao():
    comparacao = pd.DataFrame(
        {METRICA_PRINCIPAL: [0.55]}, index=pd.Index(["candidato"], name="modelo"),
    )
    with pytest.raises(KeyError, match="ausentes"):
        conferir_ganho_sobre_os_pisos(comparacao, "candidato", ["logistica"])


# --- Busca de hiperparametros do #104 ---------------------------------------
#
# O risco aqui e diferente do das secoes acima. A busca nao mente sobre o modelo,
# mente sobre a **comparacao**: uma tabela com doze rotulos distintos e doze
# numeros produzidos pela mesma configuracao passaria por uma busca legitima, e a
# escolha final seria feita sobre ruido sem nada no caminho recusar.


def test_grade_gera_o_produto_cartesiano_em_ordem_estavel():
    grade = {"a": [1, 2], "b": [10, 20, 30]}
    combinacoes = combinacoes_da_grade(grade)

    assert len(combinacoes) == 6
    assert combinacoes[0] == {"a": 1, "b": 10}
    assert combinacoes[-1] == {"a": 2, "b": 30}
    assert combinacoes == combinacoes_da_grade(grade), "ordem mudou entre chamadas"


def test_configuracao_do_103_esta_dentro_da_grade():
    """Sem ela na mesma tabela, medida nos mesmos folds, o ganho nao e comparavel."""
    partida = {n: HIPERPARAMETROS_CANDIDATO[n] for n in GRADE_HIPERPARAMETROS}
    assert partida in combinacoes_da_grade(GRADE_HIPERPARAMETROS)


def test_busca_devolve_uma_linha_por_combinacao(treino, folds):
    """CR02: a tabela precisa reconstruir a decisao sem reexecutar a busca."""
    x, y = treino
    grade = {"max_iter": [5, 10], "learning_rate": [0.05, 0.5]}

    tabela = buscar_hiperparametros(grade, x, y, folds, StandardScaler())

    assert len(tabela) == 4
    assert set(tabela.columns) >= {"max_iter", "learning_rate", METRICA_PRINCIPAL,
                                   f"{METRICA_PRINCIPAL}_desvio", "roc_auc"}


def test_cada_combinacao_chega_ao_estimador_com_os_proprios_valores(
    treino, folds, monkeypatch
):
    """Trava contra o late binding da fabrica dentro do laco da busca.

    Se o lambda capturasse `ajustes` por referencia em vez de por valor, todas as
    combinacoes seriam avaliadas com os valores da ultima: a tabela sairia com um
    rotulo diferente por linha e o mesmo numero em todas, e a escolha final seria
    feita sobre uma comparacao que nunca aconteceu.
    """
    import modelo

    recebidos = []

    def espiao(semente=SEMENTE_PADRAO, **ajustes):
        recebidos.append(tuple(sorted(ajustes.items())))
        return DummyClassifier(strategy="prior")

    monkeypatch.setattr(modelo, "criar_candidato", espiao)
    x, y = treino
    grade = {"max_iter": [5, 10], "learning_rate": [0.05, 0.5]}

    modelo.buscar_hiperparametros(grade, x, y, folds, StandardScaler())

    assert len(recebidos) == 4 * len(folds), "uma fabrica por combinacao por fold"
    assert len(set(recebidos)) == 4, "as combinacoes nao chegaram distintas ao estimador"


def test_grade_vazia_e_recusada(treino, folds):
    x, y = treino
    with pytest.raises(ValueError, match="grade vazia"):
        buscar_hiperparametros({}, x, y, folds, StandardScaler())


def _tabela(partida, melhor, desvio):
    """Duas linhas: a configuracao de partida e uma alternativa melhor."""
    return pd.DataFrame([
        {"max_iter": 300, METRICA_PRINCIPAL: partida,
         f"{METRICA_PRINCIPAL}_desvio": desvio, "roc_auc": 0.70},
        {"max_iter": 600, METRICA_PRINCIPAL: melhor,
         f"{METRICA_PRINCIPAL}_desvio": desvio, "roc_auc": 0.71},
    ])


def test_ganho_menor_que_o_desvio_mantem_a_configuracao_de_partida():
    """O "Como revisar" do #104 pede exatamente que isso nao seja vendido como ganho."""
    escolha = escolher_configuracao(
        _tabela(partida=0.500, melhor=0.503, desvio=0.006), base={"max_iter": 300},
    )

    assert escolha["configuracao"] == {"max_iter": 300}
    assert escolha["e_a_de_partida"] is True
    assert escolha["ganho_supera_o_desvio"] is False


def test_ganho_maior_que_o_desvio_troca_a_configuracao():
    escolha = escolher_configuracao(
        _tabela(partida=0.500, melhor=0.520, desvio=0.006), base={"max_iter": 300},
    )

    assert escolha["configuracao"] == {"max_iter": 600}
    assert escolha["e_a_de_partida"] is False
    assert escolha["ganho_sobre_a_partida"] == pytest.approx(0.020)


def test_partida_ausente_da_tabela_e_recusada():
    with pytest.raises(ValueError, match="nao esta na grade"):
        escolher_configuracao(
            _tabela(partida=0.500, melhor=0.520, desvio=0.006), base={"max_iter": 999},
        )


# --- Metricas de ordenacao do #105 -------------------------------------------


def test_metricas_de_ordenacao_devolve_as_tres():
    y = pd.Series([0, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

    m = metricas_de_ordenacao(y, score)

    assert set(m) == {METRICA_PRINCIPAL, "roc_auc", "brier"}
    assert m["roc_auc"] == 1.0, "score perfeitamente ordenado"


def test_rotulo_binario_no_lugar_da_probabilidade_e_recusado():
    """O erro que o "Como revisar" do #105 aponta, e que degrada em silencio.

    Passar `predict()` no lugar de `predict_proba()[:, 1]` piora a precisao
    media e o Brier sem quebrar nada: o numero errado parece so um modelo pior.
    """
    y = pd.Series([0, 0, 1, 1])
    with pytest.raises(ValueError, match="predict_proba"):
        metricas_de_ordenacao(y, np.array([0.0, 1.0, 1.0, 0.0]))


def test_score_constante_do_piso_trivial_continua_aceito():
    """O piso da secao 2.2 emite uma constante, e ali ela e a probabilidade."""
    y = pd.Series([0, 0, 1, 1])
    m = metricas_de_ordenacao(y, np.zeros(4))

    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx(0.5), "erro quadratico contra a prevalencia"


def test_score_fora_de_zero_um_e_recusado():
    y = pd.Series([0, 1, 0, 1])
    with pytest.raises(ValueError, match=r"fora de \[0, 1\]"):
        metricas_de_ordenacao(y, np.array([-1.2, 0.4, 2.0, 0.9]))


def test_brier_enxerga_calibracao_que_as_outras_duas_nao_enxergam():
    """A razao de ser da terceira metrica, em um teste.

    Dividir todas as probabilidades por dois preserva a **ordem**, entao a
    precisao media e o ROC-AUC nao mudam em nada. O modelo passa a dizer "risco
    de 10%" onde o risco e de 20%, e so o Brier reclama. E por isso que o card
    pede as tres, e nao apenas as duas de ordenacao.
    """
    y = pd.Series([0, 0, 1, 1, 0, 1])
    calibrado = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    deslocado = calibrado / 2

    a = metricas_de_ordenacao(y, calibrado)
    b = metricas_de_ordenacao(y, deslocado)

    assert a[METRICA_PRINCIPAL] == pytest.approx(b[METRICA_PRINCIPAL])
    assert a["roc_auc"] == pytest.approx(b["roc_auc"])
    assert b["brier"] > a["brier"], "so o Brier percebe o deslocamento da escala"


def test_metricas_no_teste_usa_probabilidade_e_nunca_o_rotulo(treino):
    """A protecao e estrutural: nao ha por onde passar `predict()` aqui."""
    x, y = treino
    chamadas = []

    class ModeloEspiao:
        def predict_proba(self, x):
            chamadas.append("predict_proba")
            return np.column_stack([1 - np.linspace(0, 1, len(x)),
                                    np.linspace(0, 1, len(x))])

        def predict(self, x):
            chamadas.append("predict")
            return np.zeros(len(x))

    tabela = metricas_no_teste({"espiao": ModeloEspiao()}, x, y)

    assert chamadas == ["predict_proba"], "predict() nao pode ser chamado"
    assert list(tabela.index) == ["espiao"]
    assert set(tabela.columns) == {METRICA_PRINCIPAL, "roc_auc", "brier"}


def test_metricas_no_teste_devolve_uma_linha_por_modelo(treino):
    x, y = treino
    modelos = {
        "trivial": DummyClassifier(strategy="most_frequent").fit(x, y),
        "prior": DummyClassifier(strategy="prior").fit(x, y),
    }

    tabela = metricas_no_teste(modelos, x, y)

    assert len(tabela) == 2
    assert list(tabela.index) == ["trivial", "prior"]


# --- Limiar operacional e matriz de confusao do #106 -------------------------


@pytest.fixture
def fila():
    """Oito respostas com score conhecido, para contagem a mao."""
    y = pd.Series([0, 1, 0, 1, 1, 0, 0, 1], index=range(300, 308))
    score = np.array([0.10, 0.90, 0.20, 0.80, 0.70, 0.30, 0.05, 0.60])
    return y, score


def test_limiar_e_o_k_esimo_maior_score(fila):
    _, score = fila
    assert limiar_por_capacidade(score, 1) == 0.90
    assert limiar_por_capacidade(score, 3) == 0.70
    assert limiar_por_capacidade(score, 8) == 0.05


@pytest.mark.parametrize("k", [0, -1, 9])
def test_fila_vazia_ou_maior_que_a_particao_e_recusada(fila, k):
    """Capacidade acima do volume do periodo nao e fila, e contatar todo mundo."""
    _, score = fila
    with pytest.raises(ValueError, match="k precisa estar entre"):
        limiar_por_capacidade(score, k)


def test_precisao_e_cobertura_no_topo_conferem_com_a_contagem_a_mao(fila):
    y, score = fila
    m = metricas_no_topo(y, score, 3)

    # topo: 0,90 (Detrator), 0,80 (Detrator), 0,70 (Detrator)
    assert m["limiar"] == 0.70
    assert (m["vp"], m["fp"], m["fn"], m["vn"]) == (3, 0, 1, 4)
    assert m["precisao_no_topo"] == pytest.approx(1.0)
    assert m["cobertura"] == pytest.approx(0.75), "3 dos 4 Detratores"


def test_empate_no_limiar_seleciona_mais_que_k_e_isso_e_reportado():
    """Com empate, `score >= limiar` passa de k, e quem le precisa saber."""
    y = pd.Series([1, 0, 1, 0])
    score = np.array([0.5, 0.5, 0.5, 0.1])

    m = metricas_no_topo(y, score, 2)

    assert m["k_pedido"] == 2
    assert m["n_selecionados"] == 3, "os tres empatados em 0,5 entram"


def test_matriz_de_confusao_bate_com_as_metricas_no_mesmo_limiar(fila):
    """O "Como verificar" do #106: recontar os quadrantes a partir do score."""
    y, score = fila
    m = metricas_no_topo(y, score, 4)
    matriz = matriz_de_confusao(y, score, m["limiar"])

    assert matriz.loc["Detrator", "na fila de contato"] == m["vp"]
    assert matriz.loc["não Detrator", "na fila de contato"] == m["fp"]
    assert matriz.loc["Detrator", "fora da fila"] == m["fn"]
    assert matriz.loc["não Detrator", "fora da fila"] == m["vn"]
    assert int(matriz.values.sum()) == len(y)


def test_cobertura_cresce_e_precisao_cai_conforme_a_fila_aumenta(fila):
    """A troca que torna a escolha de k uma decisao de operacao, nao de modelo."""
    y, score = fila
    curta = metricas_no_topo(y, score, 2)
    longa = metricas_no_topo(y, score, 6)

    assert longa["cobertura"] > curta["cobertura"]
    assert longa["precisao_no_topo"] < curta["precisao_no_topo"]


def test_tabela_de_capacidade_traz_uma_linha_por_premissa(fila):
    y, score = fila
    tabela = tabela_de_capacidade(y, score, [1, 2], dias=2)

    assert list(tabela.index) == [1, 2]
    assert tabela.loc[1, "k_pedido"] == 2 and tabela.loc[2, "k_pedido"] == 4


# --- Tabela comparativa (#108) ----------------------------------------------
#
# O risco aqui nao e a tabela sair errada: e ela sair plausivel. Uma metrica
# dependente de corte comparada entre modelos medidos em filas de tamanhos
# diferentes parece uma comparacao e nao e, e nada no caminho recusa.


class _ScoreFixo(BaseEstimator):
    """Emite sempre a mesma probabilidade, como o piso trivial da secao 2.2."""

    def __init__(self, valor=0.2):
        self.valor = valor

    def fit(self, x, y=None):
        return self

    def predict_proba(self, x):
        p = np.full(len(x), self.valor)
        return np.column_stack([1 - p, p])


class _ScoreOrdenado(BaseEstimator):
    """Score que cresce com a primeira coluna, entao ordena de verdade."""

    def fit(self, x, y=None):
        return self

    def predict_proba(self, x):
        bruto = np.asarray(x)[:, 0]
        p = (bruto - bruto.min()) / (bruto.max() - bruto.min() + 1e-9)
        p = np.clip(p * 0.98 + 0.01, 0, 1)
        return np.column_stack([1 - p, p])


@pytest.fixture
def teste_sintetico():
    rng = np.random.default_rng(7)
    n = 400
    y = pd.Series(rng.binomial(1, 0.25, n))
    x = np.column_stack([y + rng.normal(0, 0.5, n), rng.normal(0, 1, n)])
    return x, y


def test_tabela_comparativa_usa_a_mesma_capacidade_para_todos(teste_sintetico):
    from modelo import tabela_comparativa

    x, y = teste_sintetico
    tabela = tabela_comparativa(
        {"ordenado": _ScoreOrdenado().fit(x, y)}, x, y, k=100,
    )
    assert tabela.loc["ordenado", "n_na_fila"] == 100


def test_score_degenerado_nao_finge_ter_fila(teste_sintetico):
    """O piso trivial seleciona a particao inteira, e a coluna precisa mostrar isso."""
    from modelo import tabela_comparativa

    x, y = teste_sintetico
    tabela = tabela_comparativa({"trivial": _ScoreFixo().fit(x, y)}, x, y, k=100)

    assert tabela.loc["trivial", "n_na_fila"] == len(y)
    assert tabela.loc["trivial", "precisao_no_topo"] == pytest.approx(y.mean())
    assert tabela.loc["trivial", "cobertura"] == pytest.approx(1.0)


def test_tabela_comparativa_traz_ordenacao_e_fila_na_mesma_linha(teste_sintetico):
    from modelo import METRICA_PRINCIPAL, tabela_comparativa

    x, y = teste_sintetico
    tabela = tabela_comparativa(
        {"trivial": _ScoreFixo().fit(x, y), "ordenado": _ScoreOrdenado().fit(x, y)},
        x, y, k=100,
    )
    esperadas = {METRICA_PRINCIPAL, "roc_auc", "brier",
                 "limiar", "n_na_fila", "precisao_no_topo", "cobertura", "f1"}
    assert esperadas <= set(tabela.columns)
    assert list(tabela.index) == ["trivial", "ordenado"]


def test_tabela_comparativa_recusa_dicionario_vazio(teste_sintetico):
    from modelo import tabela_comparativa

    x, y = teste_sintetico
    with pytest.raises(ValueError, match="nenhum modelo"):
        tabela_comparativa({}, x, y, k=10)


def test_ganhos_trazem_absoluto_e_razao(teste_sintetico):
    from modelo import ganhos_do_candidato, tabela_comparativa

    x, y = teste_sintetico
    tabela = tabela_comparativa(
        {"trivial": _ScoreFixo().fit(x, y), "ordenado": _ScoreOrdenado().fit(x, y)},
        x, y, k=100,
    )
    ganhos = ganhos_do_candidato(tabela, "ordenado", ["trivial"])

    piso = tabela.loc["trivial", "precisao_media"]
    alvo = tabela.loc["ordenado", "precisao_media"]
    assert ganhos.loc["trivial", "ganho_absoluto"] == pytest.approx(alvo - piso)
    assert ganhos.loc["trivial", "razao"] == pytest.approx(alvo / piso)


def test_ganhos_recusam_modelo_ausente(teste_sintetico):
    from modelo import ganhos_do_candidato, tabela_comparativa

    x, y = teste_sintetico
    tabela = tabela_comparativa({"ordenado": _ScoreOrdenado().fit(x, y)}, x, y, k=100)
    with pytest.raises(KeyError, match="modelos ausentes"):
        ganhos_do_candidato(tabela, "ordenado", ["nao existe"])


def test_ganhos_recusam_metrica_ausente(teste_sintetico):
    from modelo import ganhos_do_candidato, tabela_comparativa

    x, y = teste_sintetico
    tabela = tabela_comparativa({"ordenado": _ScoreOrdenado().fit(x, y)}, x, y, k=100)
    with pytest.raises(KeyError, match="metrica ausente"):
        ganhos_do_candidato(tabela, "ordenado", [], metrica="inexistente")


def test_fila_comparavel_marca_quem_respeitou_o_orcamento(teste_sintetico):
    """Pedir o mesmo k nao garante capacidade efetiva igual, e a coluna diz quando."""
    from modelo import tabela_comparativa

    x, y = teste_sintetico
    tabela = tabela_comparativa(
        {"trivial": _ScoreFixo().fit(x, y), "ordenado": _ScoreOrdenado().fit(x, y)},
        x, y, k=100,
    )
    assert bool(tabela.loc["ordenado", "fila_comparavel"]) is True
    assert bool(tabela.loc["trivial", "fila_comparavel"]) is False

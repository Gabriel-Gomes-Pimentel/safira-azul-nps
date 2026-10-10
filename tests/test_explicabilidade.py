"""Travas da permutation importance do melhor ensemble (card 14A, #191).

Este arquivo nao mede qual feature importa na base real: o ranking de verdade
sai do notebook, sobre a base da Azul. O que ele protege sao os erros que fariam
a funcao devolver um ranking com cara de valido e conteudo errado, sem levantar
erro nenhum:

1. Ranking que muda entre duas execucoes, por semente solta.
2. Importancia medida sobre a matriz transformada, com cada categorica
   espalhada em colunas one-hot.
3. Importancia medida no treino, onde uma feature memorizada parece importante.
4. Scoring diferente do da busca, ou feature da propria pesquisa de NPS no
   ranking.
5. Escolha do melhor ensemble que declara vencedor quem nao disputou.

O scorer daqui e um substituto do `scorer_f2` do #242, com a mesma
configuracao, porque aquele card ainda nao esta em `develop`. Todos os dados sao
sinteticos.

Executar com:  pytest tests/test_explicabilidade.py -v
"""
import inspect
import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.utils.validation import check_is_fitted

import ensembles
import explicabilidade as expl
from pipeline_logistica import criar_pipeline as criar_pipeline_logistica
from avaliacao import avaliar
from ensembles import criar_pipeline_random_forest
from matriz import preparar_matriz

# Substituto do `scorer_f2` do #242, com a mesma configuracao declarada la.
SCORER_TESTE = make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0)

# Poucas arvores rasas e poucas repeticoes: a suite precisa rodar em segundos, e
# nenhuma trava aqui depende do tamanho da floresta.
HIPERPARAMETROS_TESTE = {"n_estimators": 15, "max_depth": 4}
N_REPETICOES_TESTE = 3

# Os cortes da fixture `contrato`, que `preparar_matriz` registra nos metadados.
CORTES_TESTE = {"corte_validacao": "2025-06-01", "corte_teste": "2025-12-01"}

# Fila curta, porque a validacao sintetica tem poucas centenas de linhas; a regra
# dos 50 contatos por dia tem teste proprio, sobre o calendario real.
CAPACIDADE_TESTE = 40


@pytest.fixture(scope="module")
def contrato():
    """Base sintetica com Clientes repetidos e um sinal plantado em ATRASO_CHEGADA.

    O alvo depende de `ATRASO_CHEGADA` e so dela. Isso da ao teste uma resposta
    conhecida: se a funcao mede o que diz medir, essa feature tem de ficar em
    primeiro lugar.
    """
    gerador = np.random.default_rng(11)
    n = 1500
    datas = pd.to_datetime("2024-01-01") + pd.to_timedelta(gerador.integers(0, 900, size=n), unit="D")
    atraso_chegada = gerador.integers(0, 300, n)
    probabilidade = np.where(atraso_chegada > 150, 0.7, 0.08)
    base = pd.DataFrame({
        "RESPONDENT_ID": np.arange(n),
        "ID_GOLDENRECORD": gerador.integers(1, 500, size=n),
        "DATA_STD": datas.strftime("%Y-%m-%d"),
        "DETRATOR": (gerador.random(n) < probabilidade).astype(int),
        "TIER_VIAGEM": gerador.choice(["Diamante", "Safira", "Sem cadastro"], n),
        "VOO_TIPO": gerador.choice(["Direto", "Conexao"], n),
        "TIPO_ENTRETENIMENTO": gerador.choice(["Tela", "Streaming", "Sem"], n),
        "CANAL_COMPRA": gerador.choice(["Web", "Mobile", "Agency"], n),
        "SEGMENTO": gerador.choice(["Lazer", "Corporativo"], n),
        "ESTATISTICA_ATRASOSAIDA": gerador.integers(0, 300, n),
        "ATRASO_CHEGADA": atraso_chegada,
        "CANCELAMENTO_VOO": gerador.random(n) < 0.05,
        "ANTECEDENCIA_CANCELAMENTO": gerador.integers(0, 10, n),
        "TEMPO_VOO": gerador.integers(40, 600, n),
        "N_TRECHOS": gerador.integers(1, 4, n),
    })
    return preparar_matriz(base, "2025-06-01", "2025-12-01")


@pytest.fixture(scope="module")
def pipeline_ajustado(contrato):
    pipeline = criar_pipeline_random_forest(contrato["preprocessador"], **HIPERPARAMETROS_TESTE)
    return pipeline.fit(contrato["x"]["treino"], contrato["y"]["treino"])


@pytest.fixture(scope="module")
def tabela(pipeline_ajustado, contrato):
    return expl.importancia_no_contrato(
        pipeline_ajustado, contrato, SCORER_TESTE, n_repeats=N_REPETICOES_TESTE,
    )


# --- Reprodutibilidade --------------------------------------------------------


def test_duas_execucoes_devolvem_o_mesmo_ranking(tabela, pipeline_ajustado, contrato):
    """E o "Como verificar" do card: rodar duas vezes e conferir o ranking identico."""
    de_novo = expl.importancia_no_contrato(
        pipeline_ajustado, contrato, SCORER_TESTE, n_repeats=N_REPETICOES_TESTE,
    )
    pd.testing.assert_frame_equal(tabela, de_novo)


def test_paralelismo_nao_muda_o_ranking(tabela, pipeline_ajustado, contrato):
    """A semente de cada embaralhamento nao pode depender de quantos processos rodam."""
    paralela = expl.importancia_no_contrato(
        pipeline_ajustado, contrato, SCORER_TESTE, n_repeats=N_REPETICOES_TESTE, n_jobs=2,
    )
    pd.testing.assert_frame_equal(tabela, paralela)


def test_semente_padrao_e_a_do_projeto():
    padrao = inspect.signature(expl.calcular_importancia).parameters["random_state"].default
    assert padrao == expl.SEMENTE_PADRAO == 42
    assert expl.N_REPETICOES >= 5


# --- Formato do ranking -------------------------------------------------------


def test_uma_linha_por_feature_original_do_contrato(tabela, contrato):
    """Nenhum nome de coluna one-hot: o ranking e sobre o que o contrato entrega."""
    assert sorted(tabela.index) == sorted(contrato["x"]["treino"].columns)
    assert tabela.index.name == "feature"
    assert "TIER_VIAGEM" in tabela.index
    assert not any("__" in feature for feature in tabela.index)


def test_colunas_posicao_e_ordem(tabela):
    assert tuple(tabela.columns) == expl.COLUNAS_TABELA
    assert list(tabela["posicao"]) == list(range(1, len(tabela) + 1))
    assert tabela["queda_media"].is_monotonic_decreasing
    assert (tabela["queda_desvio"] >= 0).all()


def test_sinal_plantado_fica_em_primeiro(tabela):
    """Com o alvo dependendo so de ATRASO_CHEGADA, ela tem de liderar."""
    assert tabela.index[0] == "ATRASO_CHEGADA"
    assert expl.features_no_topo(tabela)[0] == "ATRASO_CHEGADA"
    assert len(expl.features_no_topo(tabela)) == expl.N_TOPO


def test_empate_e_desfeito_pelo_nome(pipeline_ajustado, contrato, monkeypatch):
    """Duas quedas iguais nao podem trocar de lugar conforme a ordem das colunas."""
    class ResultadoFalso:
        importances_mean = np.zeros(contrato["x"]["validacao"].shape[1])
        importances_std = np.zeros(contrato["x"]["validacao"].shape[1])

    monkeypatch.setattr(expl, "permutation_importance", lambda *a, **k: ResultadoFalso())
    tabela = expl.importancia_no_contrato(pipeline_ajustado, contrato, SCORER_TESTE)
    assert list(tabela.index) == sorted(contrato["x"]["validacao"].columns)


# --- Travas de entrada --------------------------------------------------------


def test_recusa_a_particao_de_treino(pipeline_ajustado, contrato):
    with pytest.raises(ValueError, match="treino"):
        expl.importancia_no_contrato(pipeline_ajustado, contrato, SCORER_TESTE, particao="treino")


def test_recusa_particao_fora_do_contrato(pipeline_ajustado, contrato):
    with pytest.raises(ValueError, match="fora do contrato"):
        expl.importancia_no_contrato(pipeline_ajustado, contrato, SCORER_TESTE, particao="avaliacao")


def test_particao_padrao_e_a_validacao(pipeline_ajustado, contrato, monkeypatch):
    recebido = {}

    def espiao(pipeline, x, y, scoring, **kwargs):
        recebido["x"], recebido["y"] = x, y
        return "ok"

    monkeypatch.setattr(expl, "calcular_importancia", espiao)
    expl.importancia_no_contrato(pipeline_ajustado, contrato, SCORER_TESTE)
    assert recebido["x"] is contrato["x"]["validacao"]
    assert recebido["y"] is contrato["y"]["validacao"]


def test_recusa_scoring_ausente(pipeline_ajustado, contrato):
    with pytest.raises(ValueError, match="scoring"):
        expl.importancia_no_contrato(pipeline_ajustado, contrato, None)


def test_recusa_uma_repeticao_so(pipeline_ajustado, contrato):
    with pytest.raises(ValueError, match="n_repeats"):
        expl.importancia_no_contrato(pipeline_ajustado, contrato, SCORER_TESTE, n_repeats=1)


def test_recusa_matriz_transformada(pipeline_ajustado, contrato):
    """A matriz que saiu do ColumnTransformer espalharia as categoricas no ranking."""
    with pytest.raises(TypeError, match="cru"):
        expl.calcular_importancia(
            pipeline_ajustado, contrato["matrizes"]["validacao"],
            contrato["y"]["validacao"], SCORER_TESTE,
        )


def test_recusa_colunas_diferentes_das_do_ajuste(pipeline_ajustado, contrato):
    x = contrato["x"]["validacao"].drop(columns="N_TRECHOS")
    with pytest.raises(ValueError, match="features com que o pipeline foi ajustado"):
        expl.calcular_importancia(pipeline_ajustado, x, contrato["y"]["validacao"], SCORER_TESTE)


def test_recusa_pipeline_sem_ajuste(contrato):
    virgem = criar_pipeline_random_forest(contrato["preprocessador"], **HIPERPARAMETROS_TESTE)
    with pytest.raises(NotFittedError, match="ajustado no treino"):
        expl.importancia_no_contrato(virgem, contrato, SCORER_TESTE)


def test_recusa_feature_da_pesquisa_de_nps(contrato):
    """O "Como revisar" do card: nenhuma feature da propria pesquisa no ranking."""
    x_treino = contrato["x"]["treino"].assign(NPS_PRINCIPAL=0)
    pipeline = criar_pipeline_random_forest(contrato["preprocessador"], **HIPERPARAMETROS_TESTE)
    pipeline.fit(x_treino, contrato["y"]["treino"])
    x_avaliacao = contrato["x"]["validacao"].assign(NPS_PRINCIPAL=0)
    with pytest.raises(ValueError, match="NPS"):
        expl.calcular_importancia(pipeline, x_avaliacao, contrato["y"]["validacao"], SCORER_TESTE)


# --- Escolha do melhor ensemble ----------------------------------------------


def test_escolhe_o_de_maior_f2():
    escolha = expl.escolher_melhor_ensemble({
        "Random Forest": {"F2": 0.41, "ROC-AUC": 0.70},
        "Gradient Boosting": {"F2": 0.44, "ROC-AUC": 0.68},
    })
    assert escolha["vencedor"] == "Gradient Boosting"
    assert escolha["metrica"] == "F2"
    assert escolha["comparacao_completa"] is True
    assert escolha["empate"] is False


def test_ensemble_sem_numero_fica_fora_e_e_sinalizado():
    """Com o #190 pendente, o Random Forest nao pode ser anunciado como vencedor da disputa."""
    escolha = expl.escolher_melhor_ensemble({
        "Random Forest": {"F2": 0.41},
        "Gradient Boosting": None,
    })
    assert escolha["vencedor"] == "Random Forest"
    assert escolha["ausentes"] == ["Gradient Boosting"]
    assert escolha["comparacao_completa"] is False


def test_nan_de_avaliar_nao_disputa():
    escolha = expl.escolher_melhor_ensemble({
        "Random Forest": {"F2": 0.30},
        "Gradient Boosting": {"F2": float("nan")},
    })
    assert escolha["vencedor"] == "Random Forest"
    assert escolha["ausentes"] == ["Gradient Boosting"]


def test_empate_e_sinalizado():
    escolha = expl.escolher_melhor_ensemble({"A": {"F2": 0.4}, "B": {"F2": 0.4}})
    assert escolha["vencedor"] == "A"
    assert escolha["empate"] is True


def test_sem_nenhum_numero_levanta_erro():
    with pytest.raises(ValueError, match="Nenhum ensemble"):
        expl.escolher_melhor_ensemble({"Random Forest": None, "Gradient Boosting": None})


# --- Medicao dos ensembles sobre a mesma matriz --------------------------------


def _pipeline_sem_ajuste(contrato):
    return criar_pipeline_random_forest(contrato["preprocessador"], **HIPERPARAMETROS_TESTE)


def _medir(contrato, pipelines, avaliar_fn=avaliar, **kwargs):
    return expl.medir_ensembles(pipelines, contrato, avaliar_fn, capacidade=CAPACIDADE_TESTE, **kwargs)


def test_mede_todos_os_ensembles_na_mesma_particao(contrato):
    """Os dois numeros saem do mesmo treino e da mesma validacao, a do `preparo` recebido."""
    chamadas = []

    def avaliar_registrando(y_true, y_pred, y_proba):
        chamadas.append(y_true)
        return avaliar(y_true, y_pred, y_proba)

    medicao = _medir(
        contrato,
        {"Random Forest": _pipeline_sem_ajuste(contrato), "Gradient Boosting": _pipeline_sem_ajuste(contrato)},
        avaliar_registrando,
    )
    assert medicao["particao"] == "validacao"
    assert medicao["cortes"] == CORTES_TESTE
    assert medicao["capacidade"] == CAPACIDADE_TESTE
    # Duas leituras por ensemble (limiar 0,5 e fila), todas na mesma validacao.
    assert len(chamadas) == 4
    assert all(y.index.equals(contrato["y"]["validacao"].index) for y in chamadas)
    for nome in ("Random Forest", "Gradient Boosting"):
        assert "F2" in medicao["metricas"][nome]
        assert "F2" in medicao["metricas_fila"][nome]
        check_is_fitted(medicao["ajustados"][nome])
        assert list(medicao["ajustados"][nome].feature_names_in_) == list(contrato["x"]["treino"].columns)


def test_medicao_devolve_as_metricas_de_avaliar_sem_alterar(contrato):
    medicao = _medir(contrato, {"Random Forest": _pipeline_sem_ajuste(contrato)})
    ajustado = medicao["ajustados"]["Random Forest"]
    x_val, y_val = contrato["x"]["validacao"], contrato["y"]["validacao"]
    proba = ajustado.predict_proba(x_val)[:, 1]
    assert medicao["metricas"]["Random Forest"] == avaliar(y_val, ajustado.predict(x_val), proba)
    assert medicao["metricas_fila"]["Random Forest"] == avaliar(
        y_val, expl.rotulos_na_fila(proba, CAPACIDADE_TESTE), proba
    )


def test_medicao_nao_altera_a_entrada_e_pode_rodar_de_novo(contrato):
    """O pipeline recebido e clonado: continua sem ajuste, e a segunda chamada nao trava."""
    pipelines = {"Random Forest": _pipeline_sem_ajuste(contrato)}
    primeira = _medir(contrato, pipelines)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipelines["Random Forest"])
    assert primeira["ajustados"]["Random Forest"] is not pipelines["Random Forest"]

    segunda = _medir(contrato, pipelines)
    assert segunda["metricas"] == primeira["metricas"]
    assert segunda["metricas_fila"] == primeira["metricas_fila"]


def test_pipeline_ja_ajustado_e_reajustado_no_treino_daqui(pipeline_ajustado, contrato):
    """O clone descarta o ajuste de fora, entao o que vale e sempre o treino deste `preparo`."""
    de_fora = _medir(contrato, {"Random Forest": pipeline_ajustado})
    do_zero = _medir(contrato, {"Random Forest": _pipeline_sem_ajuste(contrato)})
    assert de_fora["metricas_fila"] == do_zero["metricas_fila"]


def test_ensemble_sem_busca_entra_sem_numero(contrato):
    medicao = _medir(contrato, {"Random Forest": None, "Gradient Boosting": _pipeline_sem_ajuste(contrato)})
    assert medicao["metricas"]["Random Forest"] is None
    assert medicao["metricas_fila"]["Random Forest"] is None
    assert medicao["ajustados"]["Random Forest"] is None
    escolha = expl.escolher_melhor_ensemble(medicao["metricas_fila"])
    assert escolha["vencedor"] == "Gradient Boosting"
    assert escolha["comparacao_completa"] is False


def test_fila_maior_que_a_particao_falha_antes_de_ajustar(contrato):
    """Na base sintetica a validacao tem poucas centenas de linhas; 9.200 nao cabe."""
    n_validacao = len(contrato["y"]["validacao"])
    with pytest.raises(ValueError, match="nao cabe"):
        expl.medir_ensembles(
            {"Random Forest": _pipeline_sem_ajuste(contrato)}, contrato, avaliar, capacidade=n_validacao + 1,
        )


def test_escolha_pela_fila_nao_depende_do_class_weight(contrato):
    """Dois pipelines reais que so diferem no `class_weight`.

    Na Regressao Logistica, `class_weight="balanced"` desloca o intercepto e quase
    nao mexe na ordem. O F2 no limiar 0,5 muda entre os dois; o F2 na fila nao.
    E o caso do CR01: um criterio que premia a calibracao para 0,5 e outro que
    le os dois modelos no mesmo numero de contatos.
    """
    medicao = _medir(contrato, {
        "sem reponderar": criar_pipeline_logistica(contrato["preprocessador"], class_weight=None),
        "balanced": criar_pipeline_logistica(contrato["preprocessador"], class_weight="balanced"),
    })
    limiar = {nome: m["F2"] for nome, m in medicao["metricas"].items()}
    fila = {nome: m["F2"] for nome, m in medicao["metricas_fila"].items()}
    # Observado na base sintetica: 0,046 de diferenca no limiar e 0,0 na fila.
    assert abs(limiar["balanced"] - limiar["sem reponderar"]) > 0.02
    assert abs(fila["balanced"] - fila["sem reponderar"]) < 0.005


def test_medicao_recusa_o_treino(contrato):
    with pytest.raises(ValueError, match="treino"):
        _medir(contrato, {"Random Forest": _pipeline_sem_ajuste(contrato)}, particao="treino")


def test_cortes_vem_dos_metadados_do_preparo(contrato):
    assert expl.cortes_do_preparo(contrato) == CORTES_TESTE


# --- Fila de capacidade --------------------------------------------------------


def test_fila_marca_exatamente_os_k_maiores():
    proba = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
    np.testing.assert_array_equal(expl.rotulos_na_fila(proba, 3), [1, 0, 1, 0, 1])


def test_fila_desfaz_empate_pela_ordem_e_tem_sempre_k():
    """Com empate no limite, `proba >= limiar` passaria de k; a fila nao."""
    proba = np.array([0.5, 0.9, 0.5, 0.5])
    rotulos = expl.rotulos_na_fila(proba, 2)
    assert rotulos.sum() == 2
    np.testing.assert_array_equal(rotulos, [1, 1, 0, 0])


def test_fila_recusa_tamanho_impossivel():
    with pytest.raises(ValueError):
        expl.rotulos_na_fila(np.array([0.2, 0.4]), 3)
    with pytest.raises(ValueError):
        expl.rotulos_na_fila(np.array([0.2, 0.4]), 0)


def test_fila_nao_depende_da_calibracao():
    """Multiplicar a probabilidade (outro `class_weight`) muda o limiar 0,5, nao a fila."""
    proba = np.array([0.30, 0.20, 0.45, 0.10, 0.40])
    np.testing.assert_array_equal(expl.rotulos_na_fila(proba, 2), expl.rotulos_na_fila(proba * 2, 2))
    assert (proba >= 0.5).sum() != (proba * 2 >= 0.5).sum()


def test_capacidade_da_validacao_usa_os_dias_da_particao():
    """Mesma regra dos 9.050 do teste (50 x 181 dias), aplicada aos 184 dias da validacao."""
    preparo = {"metadados": {"corte_validacao": "2025-07-01", "corte_teste": "2026-01-01"}}
    assert expl.capacidade_da_particao(preparo) == 9200
    with pytest.raises(ValueError):
        expl.capacidade_da_particao(preparo, "teste")


# --- Cortes das buscas ----------------------------------------------------------


def test_busca_com_os_mesmos_cortes_passa(contrato):
    expl.conferir_cortes_da_busca({"hiperparametros": {}, **CORTES_TESTE}, contrato, "Random Forest")


def test_busca_sem_cortes_e_recusada(contrato):
    with pytest.raises(ValueError, match="nao registra os cortes"):
        expl.conferir_cortes_da_busca({"hiperparametros": {}}, contrato, "Gradient Boosting")


def test_busca_em_outra_matriz_e_recusada(contrato):
    """O caso que o alinhamento de datas do 3ef81e5 criou: busca antiga, medicao nova."""
    registro = {"hiperparametros": {}, "corte_validacao": "2025-07-01", "corte_teste": "2026-01-01"}
    with pytest.raises(ValueError, match="outra matriz"):
        expl.conferir_cortes_da_busca(registro, contrato, "Random Forest")


def test_json_versionado_do_gradient_boosting_registra_os_cortes_do_contrato():
    registro = json.loads(ensembles.ARQUIVO_HIPERPARAMETROS_GRADIENT_BOOSTING.read_text(encoding="utf-8"))
    assert registro["corte_validacao"] == "2025-07-01"
    assert registro["corte_teste"] == "2026-01-01"


# --- Comparacao com a EDA, grafico e artefato ---------------------------------


def test_comparacao_com_a_eda_usa_o_nome_do_contrato(tabela):
    comparacao = expl.comparar_com_eda(tabela)
    assert list(comparacao.index) == ["ATRASO_CHEGADA", "FAIXA_ATRASO", "N_TRECHOS"]
    assert comparacao.loc["FAIXA_ATRASO", "feature_no_contrato"] == "ESTATISTICA_ATRASOSAIDA"
    assert comparacao.loc["ATRASO_CHEGADA", "posicao"] == 1
    assert bool(comparacao.loc["ATRASO_CHEGADA", "no_topo"]) is True


def test_feature_da_eda_ausente_nao_some_da_comparacao(tabela):
    comparacao = expl.comparar_com_eda(tabela, {"FAIXA_ATRASO": "FAIXA_ATRASO"})
    assert comparacao.loc["FAIXA_ATRASO", "posicao"] is None
    assert bool(comparacao.loc["FAIXA_ATRASO", "no_topo"]) is False


def test_grafico_mostra_media_e_desvio_de_cada_feature(tabela):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.container import BarContainer
    figura = expl.plotar_importancia(tabela)
    ax = figura.axes[0]
    assert len(ax.patches) == len(tabela)
    # As barras de erro sao o desvio padrao: sem elas o CR04 nao se cumpre.
    barras = next(c for c in ax.containers if isinstance(c, BarContainer))
    assert barras.errorbar is not None
    assert len(barras.errorbar.lines[2][0].get_segments()) == len(tabela)
    assert ax.get_yticklabels()[-1].get_text() == tabela.index[0]
    matplotlib.pyplot.close(figura)


def _medicao_ficticia():
    return {
        "metricas": {"Random Forest": {"F2": 0.45, "Precisão Média": 0.52}, "Gradient Boosting": None},
        "metricas_fila": {"Random Forest": {"F2": 0.40}, "Gradient Boosting": None},
        "capacidade": 9200,
        "particao": "validacao",
        "cortes": dict(CORTES_TESTE),
    }


def test_artefato_exige_a_medicao(tabela, tmp_path):
    """Sem a medicao, o JSON nao diria de que matriz nem por qual criterio saiu."""
    escolha = expl.escolher_melhor_ensemble({"Random Forest": {"F2": 0.4}})
    with pytest.raises(TypeError):
        expl.salvar_ranking(tabela, escolha, scoring_nome="scorer_f2", caminho=tmp_path / "a.json")
    sem_corte = {**_medicao_ficticia(), "cortes": {"corte_validacao": "2025-06-01"}}
    with pytest.raises(ValueError, match="corte_teste"):
        expl.salvar_ranking(tabela, escolha, scoring_nome="scorer_f2", caminho=tmp_path / "b.json", medicao=sem_corte)


def test_artefato_guarda_configuracao_e_ranking(tabela, tmp_path):
    medicao = _medicao_ficticia()
    escolha = expl.escolher_melhor_ensemble(medicao["metricas_fila"])
    caminho = expl.salvar_ranking(
        tabela, escolha, scoring_nome="scorer_f2", n_repeats=N_REPETICOES_TESTE,
        caminho=tmp_path / "importancia.json", medicao=medicao,
    )
    registro = expl.carregar_ranking(caminho)
    assert registro["ensemble"] == "Random Forest"
    assert registro["comparacao_completa"] is False
    assert registro["particao"] == "validacao"
    assert registro["corte_validacao"] == "2025-06-01"
    assert registro["corte_teste"] == "2025-12-01"
    # As duas leituras ficam lado a lado, e o JSON diz qual decidiu.
    assert registro["criterio_escolha"] == expl.CRITERIO_ESCOLHA
    assert registro["capacidade_fila"] == 9200
    # O formato antigo continua: um numero por ensemble, o que decidiu.
    assert registro["metricas_por_ensemble"] == {"Random Forest": 0.40}
    assert registro["leituras_por_ensemble"]["Random Forest"] == {
        "F2 na fila": 0.40, "F2 no limiar 0,5": 0.45, "Precisão Média": 0.52,
    }
    assert registro["leituras_por_ensemble"]["Gradient Boosting"] is None
    assert registro["n_repeats"] == N_REPETICOES_TESTE
    assert registro["random_state"] == 42
    assert registro["scoring"] == "scorer_f2"
    assert registro["topo"] == expl.features_no_topo(tabela)
    assert [linha["feature"] for linha in registro["ranking"]] == list(tabela.index)
    assert all(isinstance(linha["posicao"], int) for linha in registro["ranking"])
    assert not math.isnan(registro["ranking"][0]["queda_media"])
    # So agregados: nenhum identificador de Cliente ou de resposta no arquivo.
    texto = json.dumps(registro)
    assert "ID_GOLDENRECORD" not in texto and "RESPONDENT_ID" not in texto

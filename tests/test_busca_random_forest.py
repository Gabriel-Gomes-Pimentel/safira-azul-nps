"""Travas da busca aleatoria do Random Forest (card 08B, #189).

Este arquivo nao mede qualidade de modelo: o vencedor real sai do notebook,
sobre a base real. O que ele protege sao os erros que fariam a busca devolver
hiperparametros inflados ou impossiveis de reconstruir, sem levantar erro
nenhum:

1. Validacao que ignora o Cliente. Um `cv` inteiro vira particionador
   estratificado, e o mesmo Cliente cai no ajuste e na validacao.
2. Busca curta ou sem semente, que sortearia combinacoes diferentes a cada
   execucao.
3. JSON que nao reconstroi o vencedor, porque perdeu a semente ou gravou um
   inteiro como texto.
4. Resumo do `cv_results_` carregando mais do que parametros e agregados.

O espaco e o scorer daqui sao substitutos pequenos: o espaco real e o
`ESPACO_RANDOM_FOREST` do #186 e o scorer real e o `scorer_f2` do #242, que
ainda nao estao em `develop`. Todos os dados sao sinteticos.

Executar com:  pytest tests/test_busca_random_forest.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV

import busca_random_forest as busca_rf
from ensembles import PASSO_MODELO
from matriz import preparar_matriz
from validacao import N_FOLDS, criar_folds

RAIZ = Path(__file__).resolve().parents[1]

# Mesmos eixos do #186, com intervalos pequenos para a suite rodar em segundos.
ESPACO_TESTE = {
    "n_estimators": stats.randint(3, 7),
    "max_depth": stats.randint(2, 5),
    "min_samples_leaf": stats.randint(1, 5),
    "max_features": stats.uniform(0.3, 0.7),
    "class_weight": ["balanced", "balanced_subsample", None],
}

# Substituto do `scorer_f2` do #242, com a mesma configuracao declarada la.
SCORER_TESTE = make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0)

# Montados por partes pelo mesmo motivo de `tests/test_ensembles.py`: o card pede
# que a busca por eles no diff volte vazia, e este arquivo nao pode ser a
# excecao.
TRECHOS_PROIBIDOS = (
    "K" + "Fold(",
    "Stratified" + "K" + "Fold(",
    "cv" + "=5",
    "cv" + "=" + "N_FOLDS",
    "cross" + "_val_" + "score",
    "train" + "_test_" + "split",
)
ARQUIVOS_DO_CARD = (RAIZ / "src" / "busca_random_forest.py", Path(__file__))


@pytest.fixture(scope="module")
def contrato():
    """Base sintetica com Clientes repetidos, pelo mesmo `preparar_matriz` do projeto."""
    gerador = np.random.default_rng(7)
    n = 1200
    datas = pd.to_datetime("2024-01-01") + pd.to_timedelta(gerador.integers(0, 900, size=n), unit="D")
    base = pd.DataFrame({
        "RESPONDENT_ID": np.arange(n),
        "ID_GOLDENRECORD": gerador.integers(1, 400, size=n),
        "DATA_STD": datas.strftime("%Y-%m-%d"),
        "DETRATOR": (gerador.random(n) < 0.2).astype(int),
        "TIER_VIAGEM": gerador.choice(["Diamante", "Safira", "Sem cadastro"], n),
        "VOO_TIPO": gerador.choice(["Direto", "Conexao"], n),
        "TIPO_ENTRETENIMENTO": gerador.choice(["Tela", "Streaming", "Sem"], n),
        "CANAL_COMPRA": gerador.choice(["Web", "Mobile", "Agency"], n),
        "SEGMENTO": gerador.choice(["Lazer", "Corporativo"], n),
        "ESTATISTICA_ATRASOSAIDA": gerador.integers(0, 300, n),
        "ATRASO_CHEGADA": gerador.integers(0, 300, n),
        "CANCELAMENTO_VOO": gerador.random(n) < 0.05,
        "ANTECEDENCIA_CANCELAMENTO": gerador.integers(0, 10, n),
        "TEMPO_VOO": gerador.integers(40, 600, n),
        "N_TRECHOS": gerador.integers(1, 4, n),
    })
    return preparar_matriz(base, "2025-06-01", "2025-12-01")


@pytest.fixture(scope="module")
def folds(contrato):
    return criar_folds(contrato["x"]["treino"], contrato["grupos"]["treino"])


def _nova_busca(contrato, folds, random_state=11):
    return busca_rf.criar_busca_random_forest(
        contrato["preprocessador"], ESPACO_TESTE, folds, SCORER_TESTE, random_state=random_state,
    )


def _executar(contrato, folds, random_state=11):
    busca = _nova_busca(contrato, folds, random_state)
    relato = busca_rf.executar_busca(
        busca, contrato["x"]["treino"], contrato["y"]["treino"], contrato["grupos"]["treino"],
    )
    return busca, relato


@pytest.fixture(scope="module")
def busca_executada(contrato, folds):
    return _executar(contrato, folds)


# ------------------------------------------------------------ montagem (CR01, CR02)
def test_cv_inteiro_e_recusado(contrato):
    """CR02: o atalho que viraria um particionador estratificado sem Cliente."""
    with pytest.raises(TypeError, match="Cliente"):
        busca_rf.criar_busca_random_forest(contrato["preprocessador"], ESPACO_TESTE, 5, SCORER_TESTE)


def test_busca_com_menos_de_40_iteracoes_e_recusada(contrato, folds):
    """CR01: n_iter minimo de 40."""
    with pytest.raises(ValueError, match="40"):
        busca_rf.criar_busca_random_forest(
            contrato["preprocessador"], ESPACO_TESTE, folds, SCORER_TESTE, n_iter=39,
        )


def test_scoring_e_obrigatorio(contrato, folds):
    with pytest.raises(ValueError, match="#242"):
        busca_rf.criar_busca_random_forest(contrato["preprocessador"], ESPACO_TESTE, folds, None)


def test_busca_usa_os_folds_do_contrato_e_semente_fixa(contrato, folds):
    """CR01 e CR02: os folds recebidos, 40 iteracoes e a mesma semente no sorteio e na floresta."""
    busca = _nova_busca(contrato, folds, random_state=11)

    assert isinstance(busca, RandomizedSearchCV)
    assert busca.n_iter == busca_rf.N_ITER_MINIMO
    assert busca.random_state == 11
    assert busca.estimator.named_steps[PASSO_MODELO].random_state == 11
    assert len(busca.cv) == N_FOLDS
    for (ajuste, validacao), (ajuste_c, validacao_c) in zip(busca.cv, folds):
        np.testing.assert_array_equal(ajuste, ajuste_c)
        np.testing.assert_array_equal(validacao, validacao_c)
    # O espaco entra prefixado pelo passo do pipeline do #187.
    assert {chave for sub in busca.param_distributions for chave in sub} == {
        f"{PASSO_MODELO}__{nome}" for nome in ESPACO_TESTE
    }


def test_fit_recebe_groups(contrato, folds, monkeypatch):
    """CR02: `groups=groups` chega ao `fit` da busca."""
    recebidos = {}
    fit_original = RandomizedSearchCV.fit

    def espiao(self, x, y=None, **kwargs):
        recebidos.update(kwargs)
        return fit_original(self, x, y, **kwargs)

    monkeypatch.setattr(RandomizedSearchCV, "fit", espiao)
    busca = _nova_busca(contrato, folds)
    busca_rf.executar_busca(busca, contrato["x"]["treino"], contrato["y"]["treino"], contrato["grupos"]["treino"])

    assert recebidos["groups"] is contrato["grupos"]["treino"]


def test_folds_que_misturam_cliente_sao_recusados_antes_do_fit(contrato, monkeypatch):
    """A conferencia do contrato roda antes de a busca gastar horas."""
    n = len(contrato["y"]["treino"])
    posicoes = np.arange(n)
    # Particao por posicao, sem olhar o Cliente: o mesmo Cliente cai nos dois lados.
    folds_vazados = [
        (np.setdiff1d(posicoes, bloco), bloco) for bloco in np.array_split(posicoes, N_FOLDS)
    ]
    busca = _nova_busca(contrato, folds_vazados)
    chamou_fit = []
    monkeypatch.setattr(RandomizedSearchCV, "fit", lambda self, *a, **k: chamou_fit.append(True))

    with pytest.raises(AssertionError):
        busca_rf.executar_busca(busca, contrato["x"]["treino"], contrato["y"]["treino"], contrato["grupos"]["treino"])
    assert not chamou_fit


# ------------------------------------------------------------------ execucao
def test_relato_conta_todos_os_ajustes(busca_executada):
    """40 combinacoes vezes 5 folds, mais o refit: nenhum fold pulado."""
    busca, relato = busca_executada
    assert relato["n_combinacoes"] == busca_rf.N_ITER_MINIMO
    assert relato["n_folds"] == N_FOLDS
    assert relato["n_ajustes"] == busca_rf.N_ITER_MINIMO * N_FOLDS + 1
    assert relato["tempo_total_s"] > 0
    assert relato["melhor_score_medio"] == pytest.approx(busca.best_score_)


def test_mesma_semente_repete_a_busca(contrato, folds, busca_executada):
    """CR01: a mesma semente sorteia as mesmas combinacoes e elege o mesmo vencedor."""
    primeira, _ = busca_executada
    segunda, _ = _executar(contrato, folds)

    assert primeira.best_params_ == segunda.best_params_
    np.testing.assert_array_equal(
        primeira.cv_results_["mean_test_score"], segunda.cv_results_["mean_test_score"]
    )


# ------------------------------------------------------ versionamento (CR04)
def test_resumo_so_tem_parametros_e_agregados(busca_executada):
    """Nada por fold e nada de Cliente no arquivo versionado."""
    busca, _ = busca_executada
    resumo = busca_rf.resumir_cv_results(busca)

    assert list(resumo.columns) == sorted(ESPACO_TESTE, key=list(resumo.columns).index) + list(busca_rf.COLUNAS_RESUMO)
    assert set(resumo.columns) == set(ESPACO_TESTE) | set(busca_rf.COLUNAS_RESUMO)
    assert len(resumo) == busca_rf.N_ITER_MINIMO
    assert resumo["rank_test_score"].is_monotonic_increasing
    assert not any(c.startswith("split") for c in resumo.columns)


def test_json_reconstroi_o_vencedor(contrato, busca_executada, tmp_path):
    """CR04: o pipeline remontado pelo JSON preve igual ao `best_estimator_` da busca."""
    busca, relato = busca_executada
    caminho_hp, caminho_resumo = busca_rf.salvar_resultados(
        busca, relato, random_state=11,
        caminho_hiperparametros=tmp_path / "hp.json",
        caminho_resumo=tmp_path / "resumo.json",
        cortes={"corte_validacao": "2025-07-01", "corte_teste": "2026-01-01"},
    )

    registro = busca_rf.carregar_hiperparametros(caminho_hp)
    assert registro["random_state"] == 11
    assert registro["corte_validacao"] == "2025-07-01"
    assert registro["corte_teste"] == "2026-01-01"
    assert registro["n_iter"] == busca_rf.N_ITER_MINIMO
    assert registro["n_folds"] == N_FOLDS
    # Inteiro continua inteiro depois do JSON; texto aqui quebraria o estimador.
    assert isinstance(registro["hiperparametros"]["n_estimators"], int)
    assert caminho_resumo.exists()

    reconstruido = busca_rf.reconstruir_pipeline(contrato["preprocessador"], registro)
    reconstruido.fit(contrato["x"]["treino"], contrato["y"]["treino"])

    x_validacao = contrato["x"]["validacao"]
    np.testing.assert_array_equal(
        reconstruido.predict(x_validacao), busca.best_estimator_.predict(x_validacao)
    )
    # Igualdade bit a bit, e nao aproximada: com o `n_jobs=1` do #187 a ordem da
    # soma dos votos e fixa. Com threads, este teste falhava por 2e-16.
    np.testing.assert_array_equal(
        reconstruido.predict_proba(x_validacao),
        busca.best_estimator_.predict_proba(x_validacao),
    )


# ------------------------------------------------------ vencedor no limite
def test_limite_discreto_so_conta_o_piso_e_o_teto_exatos():
    espaco = {"max_depth": stats.randint(3, 21)}
    assert busca_rf.parametros_no_limite({"max_depth": 20}, espaco) == ["max_depth"]
    assert busca_rf.parametros_no_limite({"max_depth": 3}, espaco) == ["max_depth"]
    assert busca_rf.parametros_no_limite({"max_depth": 19}, espaco) == []


def test_limite_continuo_usa_a_margem_da_amplitude():
    espaco = {"max_features": stats.uniform(0.3, 0.7)}
    assert busca_rf.parametros_no_limite({"max_features": 0.98}, espaco) == ["max_features"]
    assert busca_rf.parametros_no_limite({"max_features": 0.31}, espaco) == ["max_features"]
    assert busca_rf.parametros_no_limite({"max_features": 0.65}, espaco) == []


def test_categoria_nao_tem_limite():
    espaco = {"class_weight": ["balanced", "balanced_subsample", None]}
    assert busca_rf.parametros_no_limite({"class_weight": "balanced"}, espaco) == []
    assert busca_rf.parametros_no_limite({"class_weight": None}, espaco) == []


# --------------------------------------------------------------------- travas
@pytest.mark.parametrize("arquivo", ARQUIVOS_DO_CARD, ids=lambda p: p.name)
def test_card_nao_cria_particao_nem_cv_inteiro(arquivo):
    """CR02 e "Como verificar": nenhum particionador proprio nem `cv` inteiro no card."""
    fonte = arquivo.read_text(encoding="utf-8").replace(" ", "")
    encontrados = [trecho for trecho in TRECHOS_PROIBIDOS if trecho in fonte]
    assert not encontrados, f"{arquivo.name} contem {encontrados}"


def test_salvar_resultados_exige_os_cortes(busca_executada, tmp_path):
    """Sem os cortes, o JSON nao diz em que matriz a busca escolheu os vencedores."""
    busca, relato = busca_executada
    with pytest.raises(TypeError):
        busca_rf.salvar_resultados(
            busca, relato,
            caminho_hiperparametros=tmp_path / "hp.json",
            caminho_resumo=tmp_path / "resumo.json",
        )
    with pytest.raises(ValueError, match="corte_teste"):
        busca_rf.salvar_resultados(
            busca, relato,
            caminho_hiperparametros=tmp_path / "hp.json",
            caminho_resumo=tmp_path / "resumo.json",
            cortes={"corte_validacao": "2025-07-01"},
        )

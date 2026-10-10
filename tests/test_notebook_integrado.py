"""Testes de ponta a ponta do notebook integrado (notebooks/comparacao_modelos.ipynb).

O notebook e executado de verdade, com `nbclient`. Uma celula acrescentada so na
copia em memoria exporta para JSON as variaveis que o proprio notebook calculou;
o arquivo `.ipynb` nao e alterado e nenhuma metrica e recalculada aqui. A partir
dessa exportacao, os testes:

1. validam os resultados: seis modelos registrados, metricas em [0, 1], tabela
   ordenada pela metrica principal, valores analiticos do piso e da referencia
   constante, e calibracao que preserva a fila e melhora o Brier;
2. comparam as metricas com os benchmarks: metas de negocio, piso de referencia,
   referencia constante e, na base real, os valores registrados em
   `tests/benchmarks/comparacao_modelos_base_real.json`;
3. gravam o relatorio em `out/relatorio_metricas_base_sintetica.md` (e `.json`)
   e, na base real, em `out/relatorio_metricas_base_real.md` (e `.json`).

As metas de negocio entram no relatorio, mas nao reprovam o teste: um candidato
abaixo da meta e um resultado a registrar, nao um defeito do notebook.

Base sintetica: roda sempre e gera `data/dummy/` se faltar (5 s). O tempo varia
muito com a maquina: 47 s em 12 nucleos, 6m47s na maquina da revisao do #205.
Base real: opt-in, porque leva de 3 a 7 minutos e exige o dado do parceiro:

    NOTEBOOK_BASE_REAL=1 pytest tests/test_notebook_integrado.py -v

Para pular a execucao do notebook numa rodada rapida: pytest -m "not notebook".
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

from execucao_notebook import (  # noqa: E402
    PASTA_SAIDA,
    executar_notebook,
    exigir_base_real,
    garantir_base_sintetica,
)
from relatorio_metricas import (  # noqa: E402
    NOME_GB,
    NOME_GB_CALIBRADO,
    NOME_REFERENCIA,
    comparar_com_piso,
    comparar_com_registro,
    escrever_relatorio,
    montar_relatorio,
)

pytestmark = pytest.mark.notebook

BENCHMARK_REAL = Path(__file__).parent / "benchmarks" / "comparacao_modelos_base_real.json"
PASTA_RELATORIO = PASTA_SAIDA
MODELOS = {"Extra Trees", "Regressão Logística", "Árvore de Decisão", "Random Forest",
           "Gradient Boosting", "Classe Majoritária"}


@pytest.fixture(scope="module")
def sintetico(tmp_path_factory):
    garantir_base_sintetica()
    exportado = executar_notebook(True, tmp_path_factory.mktemp("sintetico") / "exportado.json")
    escrever_relatorio(montar_relatorio(exportado), PASTA_RELATORIO,
                       "relatorio_metricas_base_sintetica")
    return exportado


@pytest.fixture(scope="module")
def registro():
    return json.loads(BENCHMARK_REAL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real(tmp_path_factory, registro):
    exigir_base_real()
    exportado = executar_notebook(False, tmp_path_factory.mktemp("real") / "exportado.json")
    escrever_relatorio(montar_relatorio(exportado, registro), PASTA_RELATORIO,
                       "relatorio_metricas_base_real")
    return exportado


@pytest.fixture(params=["sintetico", "real"])
def exportado(request):
    """As validacoes estruturais valem para as duas bases."""
    return request.getfixturevalue(request.param)


# --- 1. Validacao dos resultados --------------------------------------------------

def test_registra_os_seis_modelos_com_as_tres_metricas(exportado):
    assert set(exportado["resultados"]) == MODELOS
    for valores in exportado["resultados"].values():
        assert set(valores) == set(exportado["metricas"])


def test_metricas_ficam_entre_zero_e_um(exportado):
    for tabela in (exportado["resultados"], exportado["calibracao"]):
        for modelo, valores in tabela.items():
            assert all(0.0 <= valor <= 1.0 for valor in valores.values()), modelo


def test_tabela_segue_a_ordem_da_metrica_principal(exportado):
    metrica = exportado["metrica_principal"]
    valores = [exportado["resultados"][modelo][metrica] for modelo in exportado["ordem"]]
    assert valores == sorted(valores, reverse=True)


def test_piso_reproduz_os_valores_analiticos(exportado):
    """Score constante igual a 0: Precisao Media e Brier iguais a prevalencia, ROC-AUC 0,5."""
    piso = exportado["resultados"][exportado["nome_piso"]]
    prevalencia = exportado["prevalencia"]["validacao"]

    assert piso["precisao_media"] == pytest.approx(prevalencia, abs=1e-9)
    assert piso["roc_auc"] == pytest.approx(0.5, abs=1e-12)
    assert piso["brier"] == pytest.approx(prevalencia, abs=1e-9)


def test_referencia_constante_reproduz_o_brier_analitico(exportado):
    """Probabilidade constante p (treino) numa particao com prevalencia q: q(1-p)² + (1-q)p²."""
    referencia = exportado["calibracao"][NOME_REFERENCIA]
    p, q = exportado["prevalencia"]["treino"], exportado["prevalencia"]["validacao"]

    assert referencia["brier"] == pytest.approx(q * (1 - p) ** 2 + (1 - q) * p ** 2, abs=1e-9)
    assert referencia["roc_auc"] == pytest.approx(0.5, abs=1e-12)


def test_calibracao_preserva_a_fila_e_melhora_o_brier(exportado):
    antes, depois = exportado["calibracao"][NOME_GB], exportado["calibracao"][NOME_GB_CALIBRADO]

    assert antes == pytest.approx(exportado["resultados"][NOME_GB], abs=1e-12)
    assert exportado["ordem_preservada"]
    assert depois["precisao_media"] == pytest.approx(antes["precisao_media"], abs=1e-12)
    assert depois["roc_auc"] == pytest.approx(antes["roc_auc"], abs=1e-12)
    assert exportado["brier_melhorou"] and depois["brier"] < antes["brier"]


# --- 2. Comparacao com benchmarks (base real) ---------------------------------------

def test_base_real_reproduz_o_benchmark_registrado(real, registro):
    medido = {**real["resultados"], **real["calibracao"]}
    tabela = comparar_com_registro(medido, registro["valores"], registro["tolerancia"])
    desvios = tabela.loc[~tabela["dentro_da_tolerancia"]]

    assert real["linhas"]["validacao"] == registro["linhas_validacao"]
    assert desvios.empty, f"métricas fora da tolerância:\n{desvios.to_string()}"


def test_base_real_candidatos_superam_o_piso(real):
    tabela = comparar_com_piso(real["resultados"], real["metricas"], real["nome_piso"])
    abaixo = tabela.loc[~tabela["supera_piso"]]
    assert abaixo.empty, f"candidatos que não superam o piso:\n{abaixo.to_string()}"


def test_base_real_calibracao_supera_a_referencia_constante(real):
    calibracao = real["calibracao"]
    assert calibracao[NOME_GB_CALIBRADO]["brier"] < calibracao[NOME_REFERENCIA]["brier"]


# --- 3. Relatorio ---------------------------------------------------------------------

@pytest.mark.parametrize("base", ["sintetica", "real"])
def test_relatorio_e_gravado_em_markdown_e_json(base, request):
    exportado = request.getfixturevalue("sintetico" if base == "sintetica" else "real")
    caminho_md = PASTA_RELATORIO / f"relatorio_metricas_base_{base}.md"
    caminho_json = caminho_md.with_suffix(".json")

    dados = json.loads(caminho_json.read_text(encoding="utf-8"))
    assert dados["resultados"] == exportado["resultados"]
    assert dados["base"] == ("sintética" if base == "sintetica" else "real")
    assert "## Comparação com as metas de negócio" in caminho_md.read_text(encoding="utf-8")
    if base == "real":
        assert dados["registro"]["todos_dentro_da_tolerancia"] is True

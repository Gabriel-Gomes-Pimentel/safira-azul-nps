"""Reprodutibilidade do notebook integrado (card "Testar reprodutibilidade dos modelos candidatos").

O notebook e executado duas vezes consecutivas, cada uma num kernel novo e com a
mesma semente (`SEMENTE = 42`). O log de metricas de cada execucao vai para um
arquivo separado, e os testes verificam:

- CR01: metricas identicas nas duas execucoes;
- CR02: logs de metricas iguais linha a linha;
- CR03: nenhuma variacao superior a 0,01% em nenhuma metrica;
- DoR: semente fixada em todo modelo com componente aleatorio.

Evidencias em `out/reprodutibilidade/base_<sintetica|real>/`: `execucao_1.log`,
`execucao_2.log`, `comparacao.json` e `relatorio_reprodutibilidade.md`, o
relatorio de comparacao a anexar ao card.

Base sintetica: roda sempre, em cerca de 40 segundos.
Base real: opt-in, porque leva de 8 a 14 minutos e exige o dado do parceiro:

    NOTEBOOK_BASE_REAL=1 pytest tests/test_reprodutibilidade_notebook.py -v
"""
import pytest

pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

from execucao_notebook import (  # noqa: E402
    PASTA_SAIDA,
    executar_notebook,
    exigir_base_real,
    garantir_base_sintetica,
)
from reprodutibilidade import (  # noqa: E402
    avaliar_criterios,
    comparar_execucoes,
    diferencas_entre_logs,
    escrever_evidencias,
    linhas_de_log,
)

pytestmark = pytest.mark.notebook

N_EXECUCOES = 2


@pytest.fixture(scope="module", params=["sintetica", "real"])
def reprodutibilidade(request, tmp_path_factory):
    base_sintetica = request.param == "sintetica"
    if base_sintetica:
        garantir_base_sintetica()
    else:
        exigir_base_real()
    pasta_temporaria = tmp_path_factory.mktemp(f"reprodutibilidade_{request.param}")
    execucoes = [executar_notebook(base_sintetica, pasta_temporaria / f"exportado_{i}.json")
                 for i in range(1, N_EXECUCOES + 1)]
    logs = [linhas_de_log(execucao) for execucao in execucoes]
    tabela = comparar_execucoes(execucoes)
    criterios = avaliar_criterios(tabela, logs)
    caminhos = escrever_evidencias(PASTA_SAIDA / "reprodutibilidade" / f"base_{request.param}",
                                   execucoes, logs, tabela, criterios)
    return {"execucoes": execucoes, "logs": logs, "tabela": tabela, "criterios": criterios,
            "caminhos": caminhos}


def test_cr01_metricas_identicas_com_a_mesma_semente(reprodutibilidade):
    tabela = reprodutibilidade["tabela"]
    divergentes = tabela.loc[~tabela["identico"]]
    assert divergentes.empty, f"métricas que mudaram entre execuções:\n{divergentes.to_string()}"


def test_cr02_logs_iguais_linha_a_linha(reprodutibilidade):
    caminhos = reprodutibilidade["caminhos"]
    arquivos = [caminhos[f"execucao_{i}"].read_text(encoding="utf-8") for i in range(1, N_EXECUCOES + 1)]

    assert all(arquivo == arquivos[0] for arquivo in arquivos[1:]), "\n".join(
        diferencas_entre_logs(reprodutibilidade["logs"]))


def test_cr03_variacao_de_no_maximo_001_por_cento(reprodutibilidade):
    criterios = reprodutibilidade["criterios"]
    assert criterios["CR03"], (
        f"maior variação: {criterios['maior_diferenca_percentual']}% "
        f"(limite {criterios['limite_percentual']}%)")


def test_semente_fixada_em_todo_modelo_aleatorio(reprodutibilidade):
    """DoR do card: todo modelo com `random_state` usa a semente do notebook."""
    for execucao in reprodutibilidade["execucoes"]:
        piso = execucao["nome_piso"]
        for modelo, semente in execucao["random_state"].items():
            if modelo == piso:
                # O piso nao sorteia nada: a estrategia most_frequent e deterministica.
                assert execucao["estrategia_piso"] == "most_frequent"
            else:
                assert semente == execucao["semente"] == 42, modelo


def test_relatorio_de_comparacao_gravado(reprodutibilidade):
    texto = reprodutibilidade["caminhos"]["relatorio"].read_text(encoding="utf-8")

    for codigo in ("CR01", "CR02", "CR03"):
        assert f"| {codigo} |" in texto
    assert "## Diferenças entre os logs" in texto
    assert reprodutibilidade["caminhos"]["comparacao"].exists()

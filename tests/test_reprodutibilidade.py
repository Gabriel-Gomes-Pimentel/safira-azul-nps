"""Testes do log de metricas e da comparacao entre execucoes (src/reprodutibilidade.py).

Usam execucoes sinteticas: identicas, com diferenca no ultimo bit e com diferenca
acima de 0,01%, para exercitar cada criterio do card de reprodutibilidade sem
executar o notebook. A execucao real fica em
`tests/test_reprodutibilidade_notebook.py`.
"""
import copy
import json

import pytest

from reprodutibilidade import (
    avaliar_criterios,
    comparar_execucoes,
    diferencas_entre_logs,
    escrever_evidencias,
    linhas_de_log,
    relatorio_em_markdown,
)

PISO = "Classe Majoritária"


def execucao():
    return {
        "base_sintetica": False,
        "caminho_base": "data/processed/base_analitica.parquet",
        "semente": 42,
        "random_state": {"Gradient Boosting": 42, PISO: None},
        "estrategia_piso": "most_frequent",
        "nome_piso": PISO,
        "linhas": {"treino": 1000, "validacao": 200, "teste": 100},
        "prevalencia": {"treino": 0.2043, "validacao": 0.216, "teste": 0.2041},
        "ordem": ["Gradient Boosting", PISO],
        "resultados": {
            "Gradient Boosting": {"precisao_media": 0.5173974512, "roc_auc": 0.7330101, "brier": 0.183273},
            PISO: {"precisao_media": 0.216, "roc_auc": 0.5, "brier": 0.216},
        },
        "calibracao": {
            "Gradient Boosting calibrado": {"precisao_media": 0.5173974512, "roc_auc": 0.7330101,
                                            "brier": 0.140475},
        },
        "score_medio": {"sem_calibracao": 0.419, "calibrado": 0.189},
    }


def com_valor(base, valor):
    alterada = copy.deepcopy(base)
    alterada["resultados"]["Gradient Boosting"]["brier"] = valor
    return alterada


def test_log_tem_uma_linha_por_valor_com_todas_as_casas():
    log = linhas_de_log(execucao())

    assert "semente | 42" in log
    assert "resultados | Gradient Boosting | precisao_media | 0.5173974512" in log
    assert "ordem | Gradient Boosting > Classe Majoritária" in log
    assert log == linhas_de_log(execucao())


def test_execucoes_identicas_atendem_aos_tres_criterios():
    execucoes = [execucao(), execucao()]
    logs = [linhas_de_log(e) for e in execucoes]
    criterios = avaliar_criterios(comparar_execucoes(execucoes), logs)

    assert criterios["CR01"] and criterios["CR02"] and criterios["CR03"]
    assert criterios["maior_diferenca_percentual"] == 0.0
    assert diferencas_entre_logs(logs) == []


def test_diferenca_no_ultimo_bit_reprova_cr01_e_cr02_mas_nao_cr03():
    base = execucao()
    execucoes = [base, com_valor(base, 0.183273 + 2.8e-17)]
    logs = [linhas_de_log(e) for e in execucoes]
    tabela = comparar_execucoes(execucoes)
    criterios = avaliar_criterios(tabela, logs)

    assert not criterios["CR01"] and not criterios["CR02"] and criterios["CR03"]
    divergente = tabela.loc[~tabela["identico"]]
    assert list(divergente["metrica"]) == ["brier"]
    assert any(linha.startswith("+resultados | Gradient Boosting | brier")
               for linha in diferencas_entre_logs(logs))


def test_variacao_acima_de_001_por_cento_reprova_cr03():
    base = execucao()
    variado = com_valor(base, 0.183273 * 1.0002)  # 0,02% acima
    criterios = avaliar_criterios(comparar_execucoes([base, variado]),
                                  [linhas_de_log(base), linhas_de_log(variado)])

    assert not criterios["CR03"]
    assert criterios["maior_diferenca_percentual"] == pytest.approx(0.02, rel=1e-6)


def test_metrica_ausente_numa_execucao_conta_como_nao_reproduzida():
    base = execucao()
    sem_modelo = copy.deepcopy(base)
    del sem_modelo["calibracao"]["Gradient Boosting calibrado"]
    tabela = comparar_execucoes([base, sem_modelo])

    ausentes = tabela.loc[tabela["modelo"] == "Gradient Boosting calibrado"]
    assert not ausentes["identico"].any()
    assert (ausentes["diferenca_percentual"] == float("inf")).all()


def test_comparacao_exige_duas_execucoes():
    with pytest.raises(ValueError, match="pelo menos duas"):
        comparar_execucoes([execucao()])


def test_evidencias_gravam_logs_comparacao_e_relatorio(tmp_path):
    execucoes = [execucao(), execucao()]
    logs = [linhas_de_log(e) for e in execucoes]
    tabela = comparar_execucoes(execucoes)
    caminhos = escrever_evidencias(tmp_path, execucoes, logs, tabela, avaliar_criterios(tabela, logs))

    assert caminhos["execucao_1"].read_text(encoding="utf-8") == caminhos["execucao_2"].read_text(encoding="utf-8")
    assert json.loads(caminhos["comparacao"].read_text(encoding="utf-8"))["criterios"]["CR01"] is True
    texto = caminhos["relatorio"].read_text(encoding="utf-8")
    assert "| CR01 | Métricas idênticas em execuções consecutivas com a mesma semente | atende |" in texto
    assert "| CR03 | Nenhuma variação superior a 0,01% nas métricas | atende (maior variação: 0,000000%) |" in texto
    assert "| Classe Majoritária | não se aplica (estratégia `most_frequent`, determinística) |" in texto
    assert "os 2 logs são idênticos linha a linha" in texto


def test_relatorio_mostra_o_diff_quando_os_logs_divergem():
    base = execucao()
    execucoes = [base, com_valor(base, 0.2)]
    logs = [linhas_de_log(e) for e in execucoes]
    tabela = comparar_execucoes(execucoes)
    texto = relatorio_em_markdown(execucoes, logs, tabela, avaliar_criterios(tabela, logs))

    assert "| CR01 | Métricas idênticas em execuções consecutivas com a mesma semente | não atende |" in texto
    assert "```diff" in texto and "+resultados | Gradient Boosting | brier | 0.2" in texto

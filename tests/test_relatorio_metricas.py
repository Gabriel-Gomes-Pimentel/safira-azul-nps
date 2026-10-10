"""Testes das comparacoes com benchmark e do relatorio de metricas (src/relatorio_metricas.py).

Usam numeros sinteticos escolhidos para cair dos dois lados de cada meta e do
piso, sem executar o notebook. A execucao real do notebook fica em
`tests/test_notebook_integrado.py`.
"""
import json
from pathlib import Path

import pytest

from relatorio_metricas import (
    NOME_GB_CALIBRADO,
    NOME_REFERENCIA,
    comparar_com_metas,
    comparar_com_piso,
    comparar_com_registro,
    escrever_relatorio,
    maior_e_melhor,
    montar_relatorio,
    relatorio_em_markdown,
)

BENCHMARK_REAL = Path(__file__).parent / "benchmarks" / "comparacao_modelos_base_real.json"

# Mesmo formato do `METRICAS` do notebook depois de passar por JSON (tupla vira lista).
METRICAS = {
    "precisao_media": ["Precisão Média", "maior é melhor", 0.40],
    "roc_auc": ["ROC-AUC", "maior é melhor", 0.75],
    "brier": ["Brier", "menor é melhor", None],
}
PISO = "Classe Majoritária"


def exportado(**alteracoes):
    """O que o notebook exporta, com um candidato que atinge uma meta e outro que atinge a outra."""
    dados = {
        "base_sintetica": False,
        "caminho_base": "data/processed/base_analitica.parquet",
        "linhas": {"treino": 1000, "validacao": 48301, "teste": 500},
        "prevalencia": {"treino": 0.2, "validacao": 0.216, "teste": 0.2},
        "metrica_principal": "precisao_media",
        "metricas": METRICAS,
        "nome_piso": PISO,
        "ordem": ["Gradient Boosting", "Árvore de Decisão", PISO],
        "resultados": {
            "Gradient Boosting": {"precisao_media": 0.52, "roc_auc": 0.73, "brier": 0.18},
            "Árvore de Decisão": {"precisao_media": 0.39, "roc_auc": 0.76, "brier": 0.25},
            PISO: {"precisao_media": 0.216, "roc_auc": 0.5, "brier": 0.216},
        },
        "calibracao": {
            "Gradient Boosting": {"precisao_media": 0.52, "roc_auc": 0.73, "brier": 0.18},
            NOME_GB_CALIBRADO: {"precisao_media": 0.52, "roc_auc": 0.73, "brier": 0.14},
            NOME_REFERENCIA: {"precisao_media": 0.216, "roc_auc": 0.5, "brier": 0.17},
        },
        "brier_melhorou": True,
        "ordem_preservada": True,
        "score_medio": {"sem_calibracao": 0.42, "calibrado": 0.19},
    }
    dados.update(alteracoes)
    return dados


def test_sentido_de_leitura_desconhecido_e_recusado():
    assert maior_e_melhor("maior é melhor") and not maior_e_melhor("menor é melhor")
    with pytest.raises(ValueError, match="sentido de leitura desconhecido"):
        maior_e_melhor("depende")


def test_metas_marcam_quem_atinge_e_a_distancia():
    tabela = comparar_com_metas(exportado()["resultados"], METRICAS, PISO).set_index(["modelo", "metrica"])

    assert PISO not in tabela.index.get_level_values("modelo")
    assert "Brier" not in tabela.index.get_level_values("metrica")
    assert tabela.loc[("Gradient Boosting", "Precisão Média"), "atinge_meta"]
    assert not tabela.loc[("Gradient Boosting", "ROC-AUC"), "atinge_meta"]
    assert tabela.loc[("Gradient Boosting", "ROC-AUC"), "distancia"] == pytest.approx(-0.02)
    assert not tabela.loc[("Árvore de Decisão", "Precisão Média"), "atinge_meta"]
    assert tabela.loc[("Árvore de Decisão", "ROC-AUC"), "atinge_meta"]


def test_piso_e_comparado_no_sentido_de_cada_metrica():
    tabela = comparar_com_piso(exportado()["resultados"], METRICAS, PISO).set_index(["modelo", "metrica"])

    # Brier menor que o do piso e melhor; Brier maior e pior.
    assert tabela.loc[("Gradient Boosting", "Brier"), "supera_piso"]
    assert not tabela.loc[("Árvore de Decisão", "Brier"), "supera_piso"]
    assert tabela.loc[("Árvore de Decisão", "ROC-AUC"), "supera_piso"]
    assert len(tabela) == 2 * len(METRICAS)


def test_registro_aponta_desvio_e_modelo_ausente():
    medido = {"Gradient Boosting": {"precisao_media": 0.5174, "brier": 0.1843}}
    registrado = {
        "Gradient Boosting": {"precisao_media": 0.517397, "brier": 0.183273},
        "Random Forest": {"precisao_media": 0.427083},
    }
    tabela = comparar_com_registro(medido, registrado, tolerancia=5e-4).set_index(["modelo", "metrica"])

    assert tabela.loc[("Gradient Boosting", "precisao_media"), "dentro_da_tolerancia"]
    assert not tabela.loc[("Gradient Boosting", "brier"), "dentro_da_tolerancia"]
    assert not tabela.loc[("Random Forest", "precisao_media"), "dentro_da_tolerancia"]
    assert tabela.loc[("Random Forest", "precisao_media"), "medido"] is None


def test_relatorio_da_base_real_traz_as_quatro_comparacoes(tmp_path):
    registro = {"registrado_em": "2026-09-25", "tolerancia": 5e-4,
                "valores": {"Gradient Boosting": {"precisao_media": 0.52},
                            NOME_GB_CALIBRADO: {"brier": 0.14}}}
    relatorio = montar_relatorio(exportado(), registro)
    caminho_md, caminho_json = escrever_relatorio(relatorio, tmp_path, "relatorio")

    texto = caminho_md.read_text(encoding="utf-8")
    for titulo in ("## Métricas por modelo", "## Comparação com as metas de negócio",
                   "## Comparação com o piso de referência (Classe Majoritária)",
                   "## Calibração do Gradient Boosting", "## Regressão contra o benchmark registrado"):
        assert titulo in texto
    assert "48.301 respostas" in texto and "0,2160" in texto
    assert "| Gradient Boosting | candidato | 0,5200 | 0,7300 | 0,1800 |" in texto
    assert "| Classe Majoritária | piso de referência |" in texto
    assert "Todos dentro da tolerância: sim." in texto
    assert "Base sintética" not in texto

    dados = json.loads(caminho_json.read_text(encoding="utf-8"))
    assert dados["base"] == "real"
    assert dados["calibracao"]["supera_referencia_constante"] is True
    assert dados["registro"]["todos_dentro_da_tolerancia"] is True


def test_relatorio_da_base_sintetica_avisa_e_dispensa_o_registro():
    texto = relatorio_em_markdown(montar_relatorio(exportado(base_sintetica=True)))

    assert "**Base sintética:**" in texto
    assert "Não se aplica: o benchmark registrado vale só para a base real." in texto


def test_benchmark_registrado_cobre_os_modelos_do_notebook():
    benchmark = json.loads(BENCHMARK_REAL.read_text(encoding="utf-8"))
    modelos = {"Gradient Boosting", "Regressão Logística", "Extra Trees", "Árvore de Decisão",
               "Random Forest", PISO, NOME_GB_CALIBRADO, NOME_REFERENCIA}

    assert set(benchmark["valores"]) == modelos
    for valores in benchmark["valores"].values():
        assert set(valores) == set(METRICAS)
        assert all(0.0 <= valor <= 1.0 for valor in valores.values())
    assert 0 < benchmark["tolerancia"] < 0.01

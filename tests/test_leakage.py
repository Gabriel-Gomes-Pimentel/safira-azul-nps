"""Testes do corte de leakage temporal sobre a base dummy (#198).

`test_split.py` cobre as regras de `dividir` e `conferir` em poucas linhas
sintéticas; aqui a divisão roda ponta a ponta sobre a base dummy, com os cortes
do projeto, e as verificações de `validar_leakage` precisam passar nela e
falhar quando o vazamento é injetado de propósito.

Executar com:  pytest tests/test_leakage.py -v
"""
import pandas as pd
import pytest

import split
from gerar_dummy import BASE_ANALITICA_DUMMY, gerar
from validar_leakage import CORTE_TESTE, CORTE_VALIDACAO, validar, verificar

# Fração pequena: o que se verifica é o corte, não o volume.
FRACAO_TESTE = 0.01


@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    destino = tmp_path_factory.mktemp("dummy")
    gerar(destino, fracao=FRACAO_TESTE)
    return destino / BASE_ANALITICA_DUMMY


@pytest.fixture
def particoes(parquet):
    return split.dividir(pd.read_parquet(parquet), CORTE_VALIDACAO, CORTE_TESTE)[0]


def _falhas(particoes, criterio):
    return [r for r in verificar(particoes) if r["criterio"] == criterio and not r["ok"]]


def test_cr01_teste_e_validacao_sao_posteriores_ao_treino(particoes):
    assert not _falhas(particoes, "CR01")
    assert particoes["teste"]["DATA_STD"].min() > particoes["treino"]["DATA_STD"].max()


def test_cr02_nenhum_id_em_dois_conjuntos(particoes):
    assert not _falhas(particoes, "CR02")


def test_cr01_acusa_linha_futura_no_treino(particoes):
    """Uma linha do teste copiada para o treino é exatamente o vazamento temporal."""
    futura = particoes["teste"].iloc[[0]].copy()
    futura["RESPONDENT_ID"] = -1
    futura["ID_GOLDENRECORD"] = -1
    particoes["treino"] = pd.concat([particoes["treino"], futura])
    falhas = _falhas(particoes, "CR01")
    assert {f["verificacao"] for f in falhas} == {"treino antes de validacao",
                                                  "treino antes de teste"}
    assert not _falhas(particoes, "CR02")


def test_cr02_acusa_cliente_compartilhado(particoes):
    """Cliente do teste reaparecendo no treino, com data do treino: só CR02 acusa."""
    intrusa = particoes["treino"].iloc[[0]].copy()
    intrusa["ID_GOLDENRECORD"] = particoes["teste"]["ID_GOLDENRECORD"].iloc[0]
    particoes["treino"] = pd.concat([particoes["treino"].iloc[1:], intrusa])
    falhas = _falhas(particoes, "CR02")
    assert [f["verificacao"] for f in falhas] == ["ID_GOLDENRECORD: treino x teste"]
    assert not _falhas(particoes, "CR01")


def test_cr03_relatorio_gerado(parquet, tmp_path):
    saida = tmp_path / "relatorio.md"
    assert validar(parquet, saida)
    texto = saida.read_text(encoding="utf-8")
    assert "APROVADO" in texto and "FALHOU" not in texto
    assert texto.count("| CR01 |") == 3 and texto.count("| CR02 |") == 6

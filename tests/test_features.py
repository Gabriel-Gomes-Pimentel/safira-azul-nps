"""Testes das features de Cliente e histórico.

O que mais importa aqui não é o cálculo em si, e sim a garantia de anterioridade:
uma linha nunca pode enxergar uma resposta futura do mesmo Cliente. É o mesmo
tipo de vazamento que o particionamento em `split.py` evita, só que dentro da
própria feature, e por isso merece o mesmo nível de suspeita.

Executar com:  pytest tests/test_features.py -v
"""
import numpy as np
import pandas as pd
import pytest

from features import FEATURES_HISTORICO, adicionar_historico, cobertura_do_historico


@pytest.fixture
def base():
    """Cliente 10 responde três vezes; Cliente 20 responde uma vez.

    As linhas do Cliente 10 são embaralhadas de propósito na ordem de
    armazenamento, para que o teste prove que a função ordena por
    RESPONDENT_ID e não confia na ordem de chegada das linhas.
    """
    return pd.DataFrame({
        "RESPONDENT_ID": [300, 100, 200, 400],
        "ID_GOLDENRECORD": [10, 10, 10, 20],
        "DETRATOR": [1, 0, 1, 0],
    })


def test_primeira_resposta_do_cliente_nao_tem_historico(base):
    r = adicionar_historico(base)
    primeira = r[r["RESPONDENT_ID"] == 100].iloc[0]
    assert primeira["HIST_RESPOSTAS_ANTERIORES"] == 0
    assert pd.isna(primeira["HIST_DETRATOU_ANTES"])
    assert pd.isna(primeira["HIST_TAXA_DETRACAO_ANTERIOR"])


def test_historico_so_enxerga_respostas_anteriores_na_ordem_correta(base):
    """RESPONDENT_ID 200 vem depois de 100 (que detratou=0) e antes de 300."""
    r = adicionar_historico(base)
    linha_200 = r[r["RESPONDENT_ID"] == 200].iloc[0]
    assert linha_200["HIST_RESPOSTAS_ANTERIORES"] == 1
    assert linha_200["HIST_DETRATOU_ANTES"] == 0.0
    assert linha_200["HIST_TAXA_DETRACAO_ANTERIOR"] == 0.0


def test_terceira_resposta_contabiliza_as_duas_anteriores(base):
    """RESPONDENT_ID 300 é a mais recente: vê 100 (não detratou) e 200 (detratou)."""
    r = adicionar_historico(base)
    linha_300 = r[r["RESPONDENT_ID"] == 300].iloc[0]
    assert linha_300["HIST_RESPOSTAS_ANTERIORES"] == 2
    assert linha_300["HIST_DETRATOU_ANTES"] == 1.0
    assert linha_300["HIST_TAXA_DETRACAO_ANTERIOR"] == pytest.approx(0.5)


def test_historico_nao_vaza_entre_clientes_diferentes(base):
    """Cliente 20 nunca respondeu antes, mesmo com outras linhas na base."""
    r = adicionar_historico(base)
    linha_20 = r[r["ID_GOLDENRECORD"] == 20].iloc[0]
    assert linha_20["HIST_RESPOSTAS_ANTERIORES"] == 0
    assert pd.isna(linha_20["HIST_DETRATOU_ANTES"])


def test_linha_sem_cliente_nao_recebe_historico():
    df = pd.DataFrame({
        "RESPONDENT_ID": [1, 2],
        "ID_GOLDENRECORD": [np.nan, 10],
        "DETRATOR": [1, 0],
    })
    r = adicionar_historico(df)
    sem_cliente = r[r["RESPONDENT_ID"] == 1].iloc[0]
    assert sem_cliente["HIST_RESPOSTAS_ANTERIORES"] == 0
    assert pd.isna(sem_cliente["HIST_DETRATOU_ANTES"])


def test_nao_altera_o_indice_nem_a_quantidade_de_linhas(base):
    r = adicionar_historico(base)
    assert len(r) == len(base)
    assert list(r.index) == list(base.index)


def test_recusa_base_sem_coluna_obrigatoria(base):
    with pytest.raises(KeyError):
        adicionar_historico(base.drop(columns=["ID_GOLDENRECORD"]))


def test_cobertura_do_historico_exige_a_feature_ja_calculada(base):
    with pytest.raises(KeyError):
        cobertura_do_historico(base)


def test_cobertura_do_historico_reporta_numeros_consistentes(base):
    r = adicionar_historico(base)
    cobertura = cobertura_do_historico(r)
    assert cobertura["linhas"] == 4
    assert cobertura["linhas_com_historico"] == 2  # RESPONDENT_ID 200 e 300
    assert cobertura["clientes"] == 2
    assert cobertura["clientes_com_mais_de_uma_resposta"] == 1

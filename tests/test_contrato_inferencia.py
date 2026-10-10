"""Contrato batch exercitado exclusivamente com valores artificiais."""

import csv

import numpy as np
import pandas as pd
import pytest

from contrato_inferencia import (
    COLUNAS_MODELO, COLUNAS_PASSAGEM, ErroContrato, ler_entrada, validar_entrada,
)
from features import FEATURES_HISTORICO
from preprocessamento_nps import FEATURE_SET_V1, FEATURES_POS_ENCERRAMENTO_JORNADA

CORTE = "2026-01-02T12:00:00-03:00"
FUSO = "America/Sao_Paulo"


@pytest.fixture
def entrada():
    return pd.DataFrame([{
        "ID_CLIENTE": " 0007 ", "ID_JORNADA": "00009",
        "T_EVENTO_ELEGIBILIDADE": "2026-01-02T01:00:00Z",
        "T_DISPONIBILIDADE_FEATURES": "2026-01-02T02:00:00+00:00",
        "TIER_VIAGEM": "  nova\t categoria  ", "VOO_TIPO": "NA",
        "TIPO_ENTRETENIMENTO": "NULL", "CANAL_COMPRA": "web", "SEGMENTO": " ",
        "ESTATISTICA_ATRASOSAIDA": "0", "ATRASO_CHEGADA": "-2.5",
        "CANCELAMENTO_VOO": "false", "ANTECEDENCIA_CANCELAMENTO": "",
        "TEMPO_VOO": "100", "N_TRECHOS": "1",
        "HIST_RESPOSTAS_ANTERIORES": "0", "HIST_DETRATOU_ANTES": "",
        "HIST_TAXA_DETRACAO_ANTERIOR": "",
    }], dtype=object)


def validar(df, **kwargs):
    return validar_entrada(df, t_score=kwargs.get("t_score", CORTE),
                           fuso_operacional=kwargs.get("fuso_operacional", FUSO))


def test_normalizacao_copia_ids_dia_e_ordem(entrada):
    original = entrada.copy(deep=True)
    resultado, resumo = validar(entrada[entrada.columns[::-1]])
    pd.testing.assert_frame_equal(entrada, original)
    assert COLUNAS_MODELO == tuple(FEATURE_SET_V1) + FEATURES_HISTORICO
    assert list(resultado) == COLUNAS_PASSAGEM + list(COLUNAS_MODELO) + ["DIA_OPERACIONAL"]
    assert resultado.loc[0, "ID_CLIENTE"] == " 0007 "
    assert resultado.loc[0, "ID_JORNADA"] == "00009"
    assert resultado.loc[0, "T_EVENTO_ELEGIBILIDADE"] == original.loc[0, "T_EVENTO_ELEGIBILIDADE"]
    assert resultado.loc[0, "DIA_OPERACIONAL"] == "2026-01-01"
    assert resultado.loc[0, "TIER_VIAGEM"] == "NOVA CATEGORIA"
    assert resultado.loc[0, "VOO_TIPO"] == "NA"
    assert resultado.loc[0, "TIPO_ENTRETENIMENTO"] == "NULL"
    assert pd.isna(resultado.loc[0, "SEGMENTO"])
    assert pd.isna(resultado.loc[0, "ANTECEDENCIA_CANCELAMENTO"])
    assert resumo == {"categorias_normalizadas": 2, "tempo_voo_nao_positivo": 0,
                      "campos_mascarados_cancelamento": 0}


@pytest.mark.parametrize("booleano", [True, False, np.bool_(True), "true", "false"])
@pytest.mark.parametrize("historico", [(0, None, None), (2, 1, .5), (3, 0, 0), (4, 1, 1)])
def test_positivos(entrada, booleano, historico):
    entrada["CANCELAMENTO_VOO"] = booleano
    for coluna, valor in zip(FEATURES_HISTORICO, historico):
        entrada[coluna] = valor
    resultado, _ = validar(entrada)
    assert len(resultado) == 1
    assert resultado["CANCELAMENTO_VOO"].dtype == bool


@pytest.mark.parametrize("dtype", ["string", "category", object])
def test_dtype_categorico_e_indice_duplicado(entrada, dtype):
    entrada["TIER_VIAGEM"] = entrada["TIER_VIAGEM"].astype(dtype)
    segunda = entrada.copy()
    segunda["ID_JORNADA"] = "00010"
    lote = pd.concat([entrada, segunda])
    resultado, _ = validar(lote)
    assert list(resultado.index) == [0, 0]
    assert resultado["TIER_VIAGEM"].tolist() == ["NOVA CATEGORIA"] * 2


def test_vazio(entrada, tmp_path):
    vazio = entrada.iloc[:0]
    caminho = tmp_path / "sintetico.csv"
    vazio.to_csv(caminho, index=False)
    resultado, resumo = validar(ler_entrada(caminho))
    assert resultado.empty
    assert list(resultado) == COLUNAS_PASSAGEM + list(COLUNAS_MODELO) + ["DIA_OPERACIONAL"]
    assert set(resumo.values()) == {0}


@pytest.mark.parametrize("coluna", [c for c in COLUNAS_MODELO
                                    if c not in ("CANCELAMENTO_VOO", FEATURES_HISTORICO[0])])
@pytest.mark.parametrize("ausente", [None, pd.NA, np.nan, "", " \t "])
def test_ausencias_permitidas(entrada, coluna, ausente):
    entrada.at[0, coluna] = ausente
    resultado, _ = validar(entrada)
    assert pd.isna(resultado.loc[0, coluna])


def test_numericos_nullable(entrada):
    entrada["N_TRECHOS"] = pd.Series([2], dtype="Int64")
    entrada["TEMPO_VOO"] = pd.Series([120.5], dtype="Float64")
    entrada["CANCELAMENTO_VOO"] = pd.Series([False], dtype="boolean")
    resultado, _ = validar(entrada)
    assert resultado.loc[0, "N_TRECHOS"] == 2
    assert resultado.loc[0, "TEMPO_VOO"] == 120.5


@pytest.mark.parametrize("coluna", COLUNAS_PASSAGEM + list(COLUNAS_MODELO))
def test_cada_coluna_obrigatoria(entrada, coluna):
    with pytest.raises(ErroContrato, match="obrigatoria ausente"):
        validar(entrada.drop(columns=coluna))


def test_todas_14_ausentes(entrada):
    with pytest.raises(ErroContrato, match="obrigatoria ausente"):
        validar(entrada[COLUNAS_PASSAGEM])


@pytest.mark.parametrize("coluna", ["NPS_NOVO", "SUB_NOVO", "DETRATOR", "CATEGORIA_NPS",
                                    "CLASSE_NPS", "target", "DATA_STD", "SEGREDO\n0007"])
def test_extras_proibidos_sem_vazamento(entrada, coluna):
    entrada[coluna] = "VALOR_CONFIDENCIAL"
    with pytest.raises(ErroContrato) as erro:
        validar(entrada)
    assert "extras/proibidas" in str(erro.value)
    assert coluna not in str(erro.value)
    assert "VALOR_CONFIDENCIAL" not in str(erro.value)


@pytest.mark.parametrize("coluna,valor", [
    ("ID_CLIENTE", " "), ("ID_JORNADA", None), ("ID_CLIENTE", 7),
    ("TIER_VIAGEM", 123), ("VOO_TIPO", True), ("SEGMENTO", ["x"]),
    ("CANCELAMENTO_VOO", 0), ("CANCELAMENTO_VOO", 1),
    ("CANCELAMENTO_VOO", 1.0), ("CANCELAMENTO_VOO", "True"),
    ("CANCELAMENTO_VOO", " false "), ("CANCELAMENTO_VOO", ""),
    ("CANCELAMENTO_VOO", None),
    ("N_TRECHOS", 0), ("N_TRECHOS", -1), ("N_TRECHOS", 1.5),
    ("HIST_RESPOSTAS_ANTERIORES", ""), ("HIST_RESPOSTAS_ANTERIORES", -1),
    ("HIST_RESPOSTAS_ANTERIORES", .5),
])
def test_valores_invalidos(entrada, coluna, valor):
    entrada.at[0, coluna] = valor
    with pytest.raises(ErroContrato):
        validar(entrada)


@pytest.mark.parametrize("coluna", [c for c in COLUNAS_MODELO if c not in FEATURE_SET_V1[:5]
                                    and c != "CANCELAMENTO_VOO"])
@pytest.mark.parametrize("valor", ["NA", "NULL", "NaN", "1,2", " 2 ", "1_000", True,
                                   np.bool_(False), float("inf"), -float("inf"), "1e999",
                                   np.complex128(1 + 2j)])
def test_numericos_estritos(entrada, coluna, valor):
    entrada.at[0, coluna] = valor
    with pytest.raises(ErroContrato, match="numero finito nao booleano"):
        validar(entrada)


@pytest.mark.parametrize("historico", [(0, 0, None), (0, None, 0), (1, None, 0),
                                      (1, 2, .5), (1, 1, None), (1, 0, -.1),
                                      (1, 1, 1.1), (2, 0, .5), (2, 1, 0)])
def test_historico_incoerente(entrada, historico):
    for coluna, valor in zip(FEATURES_HISTORICO, historico):
        entrada[coluna] = valor
    with pytest.raises(ErroContrato):
        validar(entrada)


@pytest.mark.parametrize("duracao", ["0", "-2", 0, -10])
def test_duracao_nao_positiva(entrada, duracao):
    entrada["TEMPO_VOO"] = duracao
    resultado, resumo = validar(entrada)
    assert resultado["TEMPO_VOO"].isna().all()
    assert resumo["tempo_voo_nao_positivo"] == 1


def test_mascara_cancelamento_e_validacao_posterior_trechos(entrada):
    entrada["CANCELAMENTO_VOO"] = "true"
    entrada["N_TRECHOS"] = .5
    entrada["ANTECEDENCIA_CANCELAMENTO"] = 7
    resultado, resumo = validar(entrada)
    assert resultado[list(FEATURES_POS_ENCERRAMENTO_JORNADA)].isna().all().all()
    assert resultado.loc[0, "ANTECEDENCIA_CANCELAMENTO"] == 7
    assert resumo["campos_mascarados_cancelamento"] == 4
    entrada["N_TRECHOS"] = "texto-invalido"
    with pytest.raises(ErroContrato, match="numero finito"):
        validar(entrada)


def test_duplicidade_jornada_e_colunas(entrada):
    with pytest.raises(ErroContrato, match="chave duplicada; quantidade=1"):
        validar(pd.concat([entrada, entrada]))
    with pytest.raises(ErroContrato, match="cabecalho duplicado"):
        validar(pd.concat([entrada, entrada[["ID_CLIENTE"]]], axis=1))


@pytest.mark.parametrize("campo", ["t_score", "T_EVENTO_ELEGIBILIDADE", "T_DISPONIBILIDADE_FEATURES"])
@pytest.mark.parametrize("valor", ["today", "2026-01-01", "2026-01-01T12:00:00",
                                   "2026-02-30T12:00:00Z", "2026-01-01 12:00:00Z",
                                   "2026-01-01T12:00:00+03:99", "2026-01-01T12:00:00+24:00",
                                   "", None, 123])
def test_timestamp_invalido(entrada, campo, valor):
    kwargs = {}
    if campo == "t_score":
        kwargs[campo] = valor
    else:
        entrada[campo] = valor
    with pytest.raises(ErroContrato, match="ISO 8601"):
        validar(entrada, **kwargs)


@pytest.mark.parametrize("campo", COLUNAS_PASSAGEM[2:])
def test_futuro_e_igualdade_offset(entrada, campo):
    entrada[campo] = "2026-01-02T15:00:01Z"
    with pytest.raises(ErroContrato, match="posterior a t_score"):
        validar(entrada)
    entrada[campo] = "2026-01-02T17:00:00+02:00"
    validar(entrada)


def test_corte_preserva_precisao_nanosegundo(entrada):
    entrada["T_EVENTO_ELEGIBILIDADE"] = "2026-01-02T15:00:00.000000001Z"
    with pytest.raises(ErroContrato, match="posterior a t_score"):
        validar(entrada)
    validar(entrada, t_score="2026-01-02T15:00:00.000000001Z")


@pytest.mark.parametrize("fuso", ["", None, 3, "GMT-3", "America/Inexistente", "../etc/passwd"])
def test_fuso_invalido(entrada, fuso):
    with pytest.raises(ErroContrato, match="fuso IANA"):
        validar(entrada, fuso_operacional=fuso)


def test_csv_sem_inferencia(entrada, tmp_path):
    caminho = tmp_path / "sintetico.csv"
    entrada.to_csv(caminho, index=False)
    lido = ler_entrada(caminho)
    pd.testing.assert_frame_equal(lido, entrada)
    assert lido.loc[0, "ID_JORNADA"] == "00009"
    validar(lido)


@pytest.mark.parametrize("encoding,sep,mensagem", [
    ("utf-8-sig", ",", "codificacao fora do contrato.*sem BOM"),
    ("utf-8", ";", "delimitador fora do contrato.*virgula"),
])
def test_csv_formato_fora_do_contrato(entrada, tmp_path, encoding, sep, mensagem):
    caminho = tmp_path / "sintetico.csv"
    entrada.to_csv(caminho, index=False, encoding=encoding, sep=sep)
    with pytest.raises(ErroContrato, match=mensagem):
        ler_entrada(caminho)


@pytest.mark.parametrize("modo", ["duplicado", "curto", "longo", "aspas", "utf8", "vazio"])
def test_csv_malformado(entrada, tmp_path, modo):
    caminho = tmp_path / "sintetico.csv"
    if modo == "utf8":
        caminho.write_bytes(b"\xff")
    elif modo == "vazio":
        caminho.write_text("", encoding="utf-8")
    else:
        with caminho.open("w", encoding="utf-8", newline="") as arquivo:
            escritor = csv.writer(arquivo)
            cabecalho = list(entrada)
            if modo == "duplicado":
                cabecalho[-1] = cabecalho[0]
            escritor.writerow(cabecalho)
            linha = entrada.iloc[0].tolist()
            if modo == "curto":
                escritor.writerow(linha[:-1])
            elif modo == "longo":
                escritor.writerow(linha + ["SEGREDO"])
            elif modo == "aspas":
                arquivo.write('"SEM_FECHAMENTO')
    with pytest.raises(ErroContrato) as erro:
        ler_entrada(caminho)
    assert "SEM_FECHAMENTO" not in str(erro.value)
    assert "SEGREDO" not in str(erro.value)


def test_erros_nao_expoem_valores(entrada):
    entrada["TEMPO_VOO"] = "VALOR_CONFIDENCIAL"
    with pytest.raises(ErroContrato) as erro:
        validar(entrada)
    mensagem = str(erro.value)
    assert "TEMPO_VOO" in mensagem and "quantidade=1" in mensagem
    assert "VALOR_CONFIDENCIAL" not in mensagem and "00009" not in mensagem

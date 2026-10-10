"""Entrada estrita safira-batch-v1, sem ajuste ou reconstrucao de historico."""

from __future__ import annotations

import csv
from numbers import Number
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from features import FEATURES_HISTORICO
from preprocessamento_nps import (
    FEATURE_SET_V1,
    FEATURES_POS_ENCERRAMENTO_JORNADA,
    aplicar_contrato_temporal_score_pos_viagem,
)

COLUNAS_MODELO = tuple(FEATURE_SET_V1) + FEATURES_HISTORICO
COLUNAS_PASSAGEM = [
    "ID_CLIENTE", "ID_JORNADA", "T_EVENTO_ELEGIBILIDADE", "T_DISPONIBILIDADE_FEATURES",
]
_TEXTO = tuple(FEATURE_SET_V1[:5])
_NUMERO = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INSTANTE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}"
    r"(?::[0-9]{2}(?:\.[0-9]{1,9})?)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)


class ErroContrato(ValueError):
    """Violacao do lote; mensagens nao incluem valores ou identificadores."""


def _exigir(coluna: str, regra: str, quantidade: int) -> None:
    if quantidade:
        raise ErroContrato(f"{coluna}: {regra}; quantidade={int(quantidade)}.")


def _schema(colunas) -> None:
    colunas = pd.Index(colunas)
    _exigir("schema", "cabecalho duplicado", colunas.duplicated().sum())
    esperadas = set(COLUNAS_PASSAGEM) | set(COLUNAS_MODELO)
    for coluna in COLUNAS_PASSAGEM + list(COLUNAS_MODELO):
        _exigir(coluna, "coluna obrigatoria ausente", int(coluna not in colunas))
    # Nomes arbitrarios podem conter dados pessoais; nao os ecoar.
    _exigir("schema", "colunas extras/proibidas", sum(c not in esperadas for c in colunas))


def ler_entrada(caminho: str | Path) -> pd.DataFrame:
    """Le CSV UTF-8/comma sem inferencia, rejeitando schema e registros malformados."""
    try:
        with Path(caminho).open(encoding="utf-8", newline="") as arquivo:
            leitor = csv.reader(arquivo, delimiter=",", strict=True)
            cabecalho = next(leitor, None)
            _exigir("schema", "cabecalho obrigatorio", int(cabecalho is None))
            _exigir("CSV", "codificacao fora do contrato: UTF-8 sem BOM obrigatorio",
                    int(bool(cabecalho) and cabecalho[0].startswith("\ufeff")))
            _exigir("CSV", "delimitador fora do contrato: usar virgula, nao ponto e virgula",
                    int(len(cabecalho) == 1 and ";" in cabecalho[0]))
            _schema(cabecalho)
            linhas = []
            invalidas = 0
            for linha in leitor:
                if len(linha) != len(cabecalho):
                    invalidas += 1
                else:
                    linhas.append(linha)
            _exigir("CSV", "quantidade de campos diferente do cabecalho", invalidas)
    except (OSError, UnicodeError, csv.Error):
        raise ErroContrato("CSV: leitura UTF-8 ou sintaxe invalida; quantidade=1.") from None
    return pd.DataFrame(linhas, columns=cabecalho, dtype=object)


def _ausente(valor) -> bool:
    if isinstance(valor, str):
        return not valor.strip()
    return pd.api.types.is_scalar(valor) and bool(pd.isna(valor))


def _timestamp(valor: str) -> pd.Timestamp:
    if not isinstance(valor, str) or not _INSTANTE.fullmatch(valor):
        raise ValueError
    return pd.Timestamp(valor)


def validar_entrada(
    df: pd.DataFrame, *, t_score: str, fuso_operacional: str,
) -> tuple[pd.DataFrame, dict]:
    """Devolve copia ordenada (passagem, 14 features, dia) e contagens de correcoes.

    Valida anterioridade declarada, nao a proveniencia as-of das fontes.
    Parsing precede a mascara; o dominio de N_TRECHOS e validado apos a mascara.
    Timestamps e IDs permanecem exatamente como recebidos.
    """
    try:
        corte = _timestamp(t_score)
    except (ValueError, OverflowError):
        raise ErroContrato("t_score: ISO 8601 com data, hora e offset obrigatorios; quantidade=1.") from None
    try:
        if not isinstance(fuso_operacional, str) or not fuso_operacional:
            raise ValueError
        fuso = ZoneInfo(fuso_operacional)
    except (ValueError, ZoneInfoNotFoundError):
        raise ErroContrato("fuso_operacional: fuso IANA invalido; quantidade=1.") from None
    if not isinstance(df, pd.DataFrame):
        raise ErroContrato("schema: DataFrame obrigatorio; quantidade=1.")
    _schema(df.columns)
    resultado = df.loc[:, COLUNAS_PASSAGEM + list(COLUNAS_MODELO)].copy(deep=True)
    resumo = {"categorias_normalizadas": 0, "tempo_voo_nao_positivo": 0,
              "campos_mascarados_cancelamento": 0}

    for coluna in ("ID_CLIENTE", "ID_JORNADA"):
        invalidos = sum(not isinstance(v, str) or not v.strip() for v in resultado[coluna])
        _exigir(coluna, "texto nao vazio obrigatorio", invalidos)
    _exigir("ID_JORNADA", "chave duplicada", resultado["ID_JORNADA"].duplicated().sum())

    for coluna in COLUNAS_PASSAGEM[2:]:
        instantes, invalidos = [], 0
        for valor in resultado[coluna]:
            try:
                instantes.append(_timestamp(valor))
            except (ValueError, OverflowError):
                invalidos += 1
        _exigir(coluna, "ISO 8601 com data, hora e offset obrigatorios", invalidos)
        _exigir(coluna, "instante posterior a t_score", sum(t > corte for t in instantes))
        if coluna == "T_EVENTO_ELEGIBILIDADE":
            resultado["DIA_OPERACIONAL"] = [t.astimezone(fuso).date().isoformat() for t in instantes]

    for coluna in _TEXTO:
        valores, invalidos = [], 0
        for valor in resultado[coluna]:
            if _ausente(valor):
                valores.append(np.nan)
            elif isinstance(valor, str):
                normalizado = re.sub(r"\s+", " ", valor.strip().upper())
                resumo["categorias_normalizadas"] += int(normalizado != valor)
                valores.append(normalizado)
            else:
                invalidos += 1
        _exigir(coluna, "categoria deve ser texto ou nulo", invalidos)
        resultado[coluna] = pd.Series(valores, index=resultado.index, dtype=object)

    booleanos, invalidos = [], 0
    for valor in resultado["CANCELAMENTO_VOO"]:
        if isinstance(valor, (bool, np.bool_)):
            booleanos.append(bool(valor))
        elif isinstance(valor, str) and valor in ("true", "false"):
            booleanos.append(valor == "true")
        else:
            invalidos += 1
    _exigir("CANCELAMENTO_VOO", "booleano nao nulo: true/false", invalidos)
    resultado["CANCELAMENTO_VOO"] = pd.Series(booleanos, index=resultado.index, dtype=bool)

    for coluna in COLUNAS_MODELO:
        if coluna in _TEXTO or coluna == "CANCELAMENTO_VOO":
            continue
        valores, invalidos = [], 0
        for valor in resultado[coluna]:
            if _ausente(valor):
                valores.append(np.nan)
                continue
            try:
                if isinstance(valor, (bool, np.bool_, complex, np.complexfloating)) or not (
                    isinstance(valor, str) and _NUMERO.fullmatch(valor)
                    or isinstance(valor, Number)
                ):
                    raise ValueError
                numero = float(valor)
                if not np.isfinite(numero):
                    raise ValueError
                valores.append(numero)
            except (ValueError, TypeError, OverflowError):
                invalidos += 1
        _exigir(coluna, "numero finito nao booleano ou nulo", invalidos)
        resultado[coluna] = pd.Series(valores, index=resultado.index, dtype="float64")

    duracao_invalida = resultado["TEMPO_VOO"] <= 0
    resumo["tempo_voo_nao_positivo"] = int(duracao_invalida.sum())
    resultado.loc[duracao_invalida, "TEMPO_VOO"] = np.nan
    canceladas = resultado["CANCELAMENTO_VOO"]
    resumo["campos_mascarados_cancelamento"] = int(
        resultado.loc[canceladas, list(FEATURES_POS_ENCERRAMENTO_JORNADA)].notna().sum().sum()
    )
    resultado = aplicar_contrato_temporal_score_pos_viagem(resultado)
    trechos = resultado["N_TRECHOS"]
    _exigir("N_TRECHOS", "inteiro >= 1 quando informado",
            (trechos.notna() & ((trechos < 1) | (trechos % 1 != 0))).sum())

    contagem, indicador, taxa = (resultado[c] for c in FEATURES_HISTORICO)
    _exigir(FEATURES_HISTORICO[0], "inteiro nao nulo >= 0",
            (contagem.isna() | (contagem < 0) | (contagem % 1 != 0)).sum())
    zero = contagem == 0
    _exigir("historico", "contagem zero exige indicador e taxa nulos",
            (zero & (indicador.notna() | taxa.notna())).sum())
    _exigir(FEATURES_HISTORICO[1], "contagem positiva exige indicador 0/1",
            (~zero & ~indicador.isin([0, 1])).sum())
    _exigir(FEATURES_HISTORICO[2], "contagem positiva exige taxa em [0,1]",
            (~zero & (taxa.isna() | (taxa < 0) | (taxa > 1))).sum())
    _exigir("historico", "indicador deve ser 1 se e somente se taxa > 0",
            (~zero & (indicador != (taxa > 0).astype(float))).sum())
    return resultado, resumo

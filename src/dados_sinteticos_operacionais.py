"""Dados artificiais para demonstracao, nunca derivados de passageiros reais."""

import numpy as np
import pandas as pd


def criar_base_sintetica(n=900):
    rng = np.random.default_rng(42)
    i = np.arange(n)
    atraso = rng.integers(0, 120, n)
    return pd.DataFrame({
        "RESPONDENT_ID": i + 1,
        "ID_GOLDENRECORD": i // 2,
        "DATA_STD": pd.to_datetime(np.select(
            [i < n * .6, i < n * .8], ["2024-06-01", "2025-08-01"], "2026-03-01")),
        "DETRATOR": ((atraso > 60) ^ (rng.random(n) < .2)).astype(int),
        "TIER_VIAGEM": np.where(i % 2, "PERFIL SINTETICO A", "PERFIL SINTETICO B"),
        "VOO_TIPO": "TIPO SINTETICO",
        "TIPO_ENTRETENIMENTO": "ENTRETENIMENTO SINTETICO",
        "CANAL_COMPRA": "CANAL SINTETICO",
        "SEGMENTO": "SEGMENTO SINTETICO",
        "ESTATISTICA_ATRASOSAIDA": atraso.astype(float),
        "ATRASO_CHEGADA": atraso.astype(float),
        "CANCELAMENTO_VOO": i % 10 == 0,
        "ANTECEDENCIA_CANCELAMENTO": np.where(i % 10 == 0, 10., np.nan),
        "TEMPO_VOO": np.full(n, 90.),
        "N_TRECHOS": np.full(n, 1.),
    })


def criar_entrada_sintetica():
    df = criar_base_sintetica(12).drop(columns=["RESPONDENT_ID", "ID_GOLDENRECORD", "DATA_STD", "DETRATOR"])
    df["ID_CLIENTE"] = [f"CLIENTE-ARTIFICIAL-{i // 2:03}" for i in range(len(df))]
    df["ID_JORNADA"] = [f"JORNADA-ARTIFICIAL-{i:03}" for i in range(len(df))]
    df["T_EVENTO_ELEGIBILIDADE"] = ["2026-10-05T10:00:00Z"] * 6 + ["2026-10-06T10:00:00Z"] * 6
    df["T_DISPONIBILIDADE_FEATURES"] = "2026-10-06T11:00:00Z"
    df["HIST_RESPOSTAS_ANTERIORES"] = 0
    df["HIST_DETRATOU_ANTES"] = np.nan
    df["HIST_TAXA_DETRACAO_ANTERIOR"] = np.nan
    return df

"""Testes do contrato de dados do score pós-viagem.

Os cenários usam somente dados sintéticos: o contrato deve proteger a estrutura
da base sem ler nem registrar dados confidenciais da Azul.
"""

import pandas as pd
import pytest

from preprocessamento_nps import validar_contrato_dados_score_pos_viagem


def base_contratada() -> pd.DataFrame:
    return pd.DataFrame({
        "RESPONDENT_ID": pd.array([1, 2, 3], dtype="Int64"),
        "ID_GOLDENRECORD": pd.array([10, 20, 30], dtype="Int64"),
        "DETRATOR": pd.array([0, 1, 0], dtype="Int64"),
        "TIER_VIAGEM": ["SAFIRA", "DIAMANTE", "SAFIRA"],
        "VOO_TIPO": ["DIRETO", "CONEXAO", "DIRETO"],
        "TIPO_ENTRETENIMENTO": ["TELA", "TELA", "TELA"],
        "CANAL_COMPRA": ["WEB", "AGENCIA", "WEB"],
        "SEGMENTO": ["LAZER", "CORPORATIVO", "LAZER"],
        "ESTATISTICA_ATRASOSAIDA": [0.0, 20.0, 0.0],
        "ATRASO_CHEGADA": [0.0, 25.0, 0.0],
        "CANCELAMENTO_VOO": [False, False, False],
        "ANTECEDENCIA_CANCELAMENTO": [float("nan"), float("nan"), float("nan")],
        "TEMPO_VOO": [90.0, 120.0, 80.0],
        "N_TRECHOS": [1, 2, 1],
    })


def test_contrato_aceita_base_de_treino_e_registra_cliente_ausente():
    base = base_contratada()
    base.loc[2, "ID_GOLDENRECORD"] = pd.NA

    resultado = validar_contrato_dados_score_pos_viagem(base, exigir_alvo=True)

    assert resultado == {"linhas_sem_cliente": 1}


def test_contrato_de_score_nao_exige_target():
    resultado = validar_contrato_dados_score_pos_viagem(
        base_contratada().drop(columns=["DETRATOR", "ID_GOLDENRECORD"])
    )

    assert resultado == {"linhas_sem_cliente": 0}


def test_contrato_de_treino_aceita_base_completa():
    resultado = validar_contrato_dados_score_pos_viagem(base_contratada(), exigir_alvo=True)

    assert resultado == {"linhas_sem_cliente": 0}


def test_contrato_de_treino_exige_target():
    with pytest.raises(KeyError, match="DETRATOR"):
        validar_contrato_dados_score_pos_viagem(
            base_contratada().drop(columns="DETRATOR"), exigir_alvo=True
        )


def test_contrato_de_treino_exige_identificador_cliente():
    with pytest.raises(KeyError, match="ID_GOLDENRECORD"):
        validar_contrato_dados_score_pos_viagem(
            base_contratada().drop(columns="ID_GOLDENRECORD"), exigir_alvo=True
        )


def test_contrato_recusa_respondent_id_nulo():
    base = base_contratada()
    base.loc[0, "RESPONDENT_ID"] = pd.NA

    with pytest.raises(ValueError, match="RESPONDENT_ID.*nulo"):
        validar_contrato_dados_score_pos_viagem(base)


def test_contrato_recusa_respondent_id_duplicado():
    base = base_contratada()
    base.loc[1, "RESPONDENT_ID"] = 1

    with pytest.raises(ValueError, match="RESPONDENT_ID.*duplicado"):
        validar_contrato_dados_score_pos_viagem(base)


def test_contrato_recusa_target_fora_do_dominio_binario():
    base = base_contratada()
    base.loc[1, "DETRATOR"] = 2

    with pytest.raises(ValueError, match="DETRATOR.*domínio binário"):
        validar_contrato_dados_score_pos_viagem(base, exigir_alvo=True)


def test_contrato_recusa_target_nulo():
    base = base_contratada()
    base.loc[1, "DETRATOR"] = pd.NA

    with pytest.raises(ValueError, match="DETRATOR.*domínio binário"):
        validar_contrato_dados_score_pos_viagem(base, exigir_alvo=True)


@pytest.mark.parametrize("valores", [
    pd.Series([False, True, False], dtype="bool"),
    pd.Series([0.0, 1.0, 0.0], dtype="float64"),
    pd.Series([0, True, 0], dtype="object"),
    pd.Series([0, 1.0, 0], dtype="object"),
])
def test_contrato_recusa_target_nao_inteiro_mesmo_com_valores_equivalentes(valores):
    base = base_contratada()
    base["DETRATOR"] = valores

    with pytest.raises(TypeError, match="DETRATOR.*dtype inteiro"):
        validar_contrato_dados_score_pos_viagem(base, exigir_alvo=True)


def test_contrato_aceita_target_coerente_com_nps_principal():
    base = base_contratada()
    base["NPS_PRINCIPAL"] = [-100 if alvo == 1 else 0 for alvo in base["DETRATOR"]]
    base.loc[2, "NPS_PRINCIPAL"] = 100

    assert validar_contrato_dados_score_pos_viagem(base, exigir_alvo=True) == {
        "linhas_sem_cliente": 0
    }


def test_contrato_recusa_target_divergente_do_nps_principal():
    base = base_contratada()
    base["NPS_PRINCIPAL"] = [0, 100, 0]

    with pytest.raises(ValueError, match="DETRATOR diverge de NPS_PRINCIPAL em 1 linha"):
        validar_contrato_dados_score_pos_viagem(base, exigir_alvo=True)


def test_contrato_recusa_nps_invalido_quando_presente_no_treino():
    base = base_contratada()
    base["NPS_PRINCIPAL"] = [0, -100, 42]

    with pytest.raises(ValueError, match="NPS_PRINCIPAL.*fora da escala"):
        validar_contrato_dados_score_pos_viagem(base, exigir_alvo=True)


def test_contrato_recusa_feature_obrigatoria_com_tipo_incompativel():
    base = base_contratada()
    base["ATRASO_CHEGADA"] = base["ATRASO_CHEGADA"].astype("string")

    with pytest.raises(TypeError, match="ATRASO_CHEGADA"):
        validar_contrato_dados_score_pos_viagem(base)

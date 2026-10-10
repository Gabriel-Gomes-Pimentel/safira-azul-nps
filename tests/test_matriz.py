"""Testes da matriz de modelagem da seção 4.3.

O que estes testes protegem não é o formato da matriz, e sim duas garantias que,
se quebrarem, produzem uma métrica de teste boa e mentirosa: nenhuma coluna de
pesquisa entra nas features, e o pré-processador não vê validação nem teste
durante o ajuste.

A segunda é a mais silenciosa das duas. Um pré-processador ajustado na base
inteira aprende a mediana e as categorias de 2026 e as devolve ao treino, e nada
no resultado denuncia isso — o número final apenas fica melhor do que deveria.

Todos usam dados sintéticos, sem tocar nas bases da Azul.

Executar com:  pytest tests/test_matriz.py -v
"""
import numpy as np
import pandas as pd
import pytest

import matriz
from matriz import conferir_contrato_da_matriz, preparar_matriz, resumo_da_matriz

CORTES = {"corte_validacao": "2025-07-01", "corte_teste": "2026-01-01"}


@pytest.fixture
def base():
    """Base sintética com as colunas da allowlist e os três períodos cobertos.

    Cada Cliente aparece em um único período, para que a regra de desempate por
    recorrência não reduza as partições e mascare o que se quer medir aqui.
    """
    n = 60
    # As datas crescem com o RESPONDENT_ID, um dia por linha dentro de cada
    # período. Datas repetidas em bloco produziriam empates que derrubam a
    # correlação de Spearman e fariam a verificação de anterioridade recusar uma
    # base sintética que, na prática, está ordenada.
    datas = (
        list(pd.date_range("2024-03-01", periods=30).astype(str))
        + list(pd.date_range("2025-09-01", periods=15).astype(str))
        + list(pd.date_range("2026-03-01", periods=15).astype(str))
    )
    return pd.DataFrame({
        "RESPONDENT_ID": range(1, n + 1),
        "ID_GOLDENRECORD": pd.array(range(101, 101 + n), dtype="Int64"),
        "DATA_STD": datas,
        "DETRATOR": ([1, 0] * (n // 2)),
        "TIER_VIAGEM": ["DIAMANTE", "SAFIRA"] * (n // 2),
        "VOO_TIPO": ["DIRETO", "CONEXAO"] * (n // 2),
        "TIPO_ENTRETENIMENTO": ["TELA"] * n,
        "CANAL_COMPRA": ["WEB", "AGENCIA"] * (n // 2),
        "SEGMENTO": ["CORPORATIVO", "LAZER"] * (n // 2),
        "ESTATISTICA_ATRASOSAIDA": np.arange(n, dtype=float),
        "ATRASO_CHEGADA": np.arange(n, dtype=float) * 2,
        "CANCELAMENTO_VOO": [False, True] * (n // 2),
        "ANTECEDENCIA_CANCELAMENTO": np.arange(n, dtype=float),
        "TEMPO_VOO": np.arange(n, dtype=float) + 60,
        "N_TRECHOS": (np.arange(n) % 3) + 1,
        # Colunas de pesquisa: existem na base e não podem chegar à matriz.
        "NPS_PRINCIPAL": [-100, 100] * (n // 2),
        "SUB_NOTA_TRIPULACAO": [1, 5] * (n // 2),
    })


# ------------------------------------------------- vazamento de coluna de pesquisa

def test_matriz_nao_recebe_coluna_de_pesquisa(base):
    preparo = preparar_matriz(base, **CORTES)
    for nome in ("treino", "validacao", "teste"):
        colunas = preparo["x"][nome].columns
        assert not [c for c in colunas if c.startswith(("NPS_", "SUB_"))]


def test_contrato_dispara_com_coluna_de_pesquisa_na_entrada(base):
    x = base[["TEMPO_VOO", "NPS_PRINCIPAL"]]
    with pytest.raises(AssertionError, match="allowlist|pesquisa"):
        conferir_contrato_da_matriz(x)


def test_contrato_dispara_com_coluna_fora_da_allowlist(base):
    x = base[["TEMPO_VOO"]].assign(COLUNA_INVENTADA=1)
    with pytest.raises(AssertionError, match="allowlist"):
        conferir_contrato_da_matriz(x)


def test_contrato_dispara_quando_falta_feature_da_allowlist(base):
    """Feature ausente na fonte tem que interromper, nao treinar em silencio.

    E o mecanismo que deixou PERFIL_TUDOAZUL passar despercebido por duas
    semanas: selecionar_features_score_pos_viagem so registra a ausencia e
    segue. A recusa precisa vir daqui.
    """
    x = base[["TEMPO_VOO", "VOO_TIPO"]]  # faltam as outras 9 da allowlist
    with pytest.raises(AssertionError, match="allowlist"):
        conferir_contrato_da_matriz(x)


def test_preparar_matriz_dispara_quando_feature_da_allowlist_falta_na_fonte(base):
    sem_tier = base.drop(columns=["TIER_VIAGEM"])
    with pytest.raises(KeyError, match="TIER_VIAGEM"):
        preparar_matriz(sem_tier, **CORTES)


# ------------------------------------------------------- features de historico

def test_matriz_inclui_as_features_de_historico_por_padrao(base):
    preparo = preparar_matriz(base, **CORTES)
    for nome in ("treino", "validacao", "teste"):
        colunas = set(preparo["x"][nome].columns)
        assert {"HIST_RESPOSTAS_ANTERIORES", "HIST_DETRATOU_ANTES",
                "HIST_TAXA_DETRACAO_ANTERIOR"} <= colunas


def test_incluir_historico_falso_nao_adiciona_as_colunas(base):
    preparo = preparar_matriz(base, **CORTES, incluir_historico=False)
    colunas = set(preparo["x"]["treino"].columns)
    assert "HIST_RESPOSTAS_ANTERIORES" not in colunas
    assert preparo["metadados"]["cobertura_historico"] is None


def test_metadados_registram_a_cobertura_do_historico(base):
    preparo = preparar_matriz(base, **CORTES)
    cobertura = preparo["metadados"]["cobertura_historico"]
    assert cobertura is not None
    assert cobertura["linhas"] == len(base)


def test_contrato_ainda_recusa_coluna_fora_da_allowlist_com_historico_ligado(base):
    """Historico amplia a allowlist só para si, e não para qualquer coluna."""
    x = base[["TEMPO_VOO"]].assign(HIST_RESPOSTAS_ANTERIORES=0, COLUNA_INVENTADA=1)
    with pytest.raises(AssertionError, match="allowlist"):
        conferir_contrato_da_matriz(
            x, colunas_extras_permitidas=frozenset({"HIST_RESPOSTAS_ANTERIORES"})
        )


# --------------------------------------- ajuste do pre-processador so no treino

def test_preprocessador_e_ajustado_apenas_no_treino(base):
    """A mediana aprendida tem de ser a do treino, não a da base inteira.

    `ESTATISTICA_ATRASOSAIDA` cresce monotonicamente com a linha, então a mediana
    do treino e a da base completa são valores distintos e verificáveis.
    """
    preparo = preparar_matriz(base, **CORTES)
    imputador = (preparo["preprocessador"]
                 .named_transformers_["numericas"].named_steps["imputar"])
    coluna = preparo["metadados"]["numericas"].index("ESTATISTICA_ATRASOSAIDA")
    mediana_aprendida = imputador.statistics_[coluna]

    esperada_do_treino = preparo["x"]["treino"]["ESTATISTICA_ATRASOSAIDA"].median()
    da_base_inteira = base["ESTATISTICA_ATRASOSAIDA"].median()

    assert mediana_aprendida == pytest.approx(esperada_do_treino)
    assert mediana_aprendida != pytest.approx(da_base_inteira)


def test_as_tres_matrizes_tem_o_mesmo_numero_de_colunas(base):
    """Validação e teste passam pelo transform do treino, não por um ajuste próprio."""
    preparo = preparar_matriz(base, **CORTES)
    larguras = {m.shape[1] for m in preparo["matrizes"].values()}
    assert len(larguras) == 1


# ------------------------------------------------------------ particoes e grupos

def test_devolve_as_tres_particoes_nao_vazias(base):
    preparo = preparar_matriz(base, **CORTES)
    for nome in ("treino", "validacao", "teste"):
        assert len(preparo["y"][nome]) > 0
        assert len(preparo["x"][nome]) == len(preparo["y"][nome])


def test_grupos_acompanham_as_linhas_de_cada_particao(base):
    """Os grupos são o que o GroupKFold usa; desalinhá-los quebra a validação."""
    preparo = preparar_matriz(base, **CORTES)
    for nome in ("treino", "validacao", "teste"):
        assert list(preparo["grupos"][nome].index) == list(preparo["x"][nome].index)


def test_recusa_base_sem_a_coluna_alvo(base):
    with pytest.raises(KeyError, match="DETRATOR"):
        preparar_matriz(base.drop(columns=["DETRATOR"]), **CORTES)


def test_metadados_da_matriz_registram_linhas_sem_cliente(base):
    base.loc[0, "ID_GOLDENRECORD"] = pd.NA

    preparo = preparar_matriz(base, **CORTES)

    assert preparo["metadados"]["linhas_sem_cliente_excluidas"] == 1
    assert preparo["metadados"]["contrato_dados"] == {"linhas_sem_cliente": 1}


def test_matriz_recusa_divergencia_na_contagem_sem_cliente(base, monkeypatch):
    dividir_original = matriz.split.dividir

    def dividir_com_contagem_divergente(*args, **kwargs):
        particoes, metadados = dividir_original(*args, **kwargs)
        metadados["linhas_sem_cliente_excluidas"] += 1
        return particoes, metadados

    monkeypatch.setattr(matriz.split, "dividir", dividir_com_contagem_divergente)
    monkeypatch.setattr(matriz.split, "conferir", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="linhas sem ID_GOLDENRECORD divergiu"):
        preparar_matriz(base, **CORTES)


# ------------------------------------------------- evidencia das linhas sem data

def test_politica_treino_exige_a_verificacao_de_anterioridade(base):
    """Enviar linha sem data ao treino sem provar anterioridade é o que se evita."""
    sem_data = base.head(2).copy()
    sem_data["RESPONDENT_ID"] = [9001, 9002]      # acima das datadas, quebra a ordem
    sem_data["ID_GOLDENRECORD"] = [9001, 9002]
    sem_data["DATA_STD"] = None
    df = pd.concat([base, sem_data], ignore_index=True)
    with pytest.raises(AssertionError, match="sobrepoem"):
        preparar_matriz(df, **CORTES, sem_data="treino")


def test_metadados_registram_a_verificacao_quando_ela_se_aplica(base):
    sem_data = base.head(2).copy()
    sem_data["RESPONDENT_ID"] = [1, 2]            # abaixo das datadas, ordem preservada
    sem_data["ID_GOLDENRECORD"] = [9001, 9002]
    sem_data["DATA_STD"] = None
    df = pd.concat([sem_data, base.assign(RESPONDENT_ID=base["RESPONDENT_ID"] + 1000)],
                   ignore_index=True)
    preparo = preparar_matriz(df, **CORTES, sem_data="treino")
    assert preparo["metadados"]["anterioridade_sem_data"]["linhas_sem_data"] == 2


def test_resumo_traz_uma_linha_por_particao(base):
    tabela = resumo_da_matriz(preparar_matriz(base, **CORTES))
    assert list(tabela.index) == ["treino", "validacao", "teste"]
    for coluna in ("n", "clientes", "prevalencia_pct", "colunas_matriz"):
        assert coluna in tabela.columns

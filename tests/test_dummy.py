"""Testes do baseline dummy da base analítica.

O que se testa aqui não é a semelhança estatística com o dado real, e sim que o
dummy cumpre os contratos que o projeto já impõe à base analítica de verdade, e
que atravessa o que consome essa base até a matriz de modelagem.

As validações usadas são as do próprio projeto — `validar_schema_features_v1`,
`validar_parquet`, `preparar_matriz` — e não cópias delas escritas para o teste.
Uma cópia passaria a concordar consigo mesma no dia em que o contrato mudasse;
estas param junto com o pipeline.

A conferência de colunas e tipos contra `data/processed/base_analitica.parquet`
fica fora da suíte, em `gerar_dummy.conferir`, porque exige o dado do parceiro.

Executar com:  pytest tests/test_dummy.py -v
"""
import pandas as pd
import pytest

import features
import matriz
from gerar_dummy import (BASE_ANALITICA_DUMMY, COLUNAS_INTEGRADA, PRIMEIRO_CLIENTE,
                         PRIMEIRO_ID, PRIMEIRO_TIER_NOVO, TIERS_ANTIGOS, gerar)
from preprocessamento_nps import validar_parquet, validar_schema_features_v1

# Fração pequena: o que se verifica é estrutura, não volume.
FRACAO_TESTE = 0.01
# Os mesmos cortes da política de particionamento temporal do projeto.
CORTE_VALIDACAO, CORTE_TESTE = "2025-07-01", "2026-01-01"


@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    destino = tmp_path_factory.mktemp("dummy")
    gerar(destino, fracao=FRACAO_TESTE)
    return destino / BASE_ANALITICA_DUMMY


@pytest.fixture(scope="module")
def base(parquet):
    return pd.read_parquet(parquet)


def test_parquet_e_legivel(parquet):
    validar_parquet(parquet)


def test_base_cumpre_o_contrato_de_features_v1(base):
    """A trava que o projeto aplica à base real aceita a dummy sem ressalva."""
    validar_schema_features_v1(base)


def test_colunas_da_integracao_vem_primeiro_e_na_ordem(base):
    """As 46 colunas do gerador, e depois as derivadas que o preparo acrescenta."""
    assert list(base.columns)[:len(COLUNAS_INTEGRADA)] == list(COLUNAS_INTEGRADA)
    assert len(base.columns) > len(COLUNAS_INTEGRADA)


def test_derivadas_vem_do_preparo(base):
    """As derivadas saem de `preparar_base_analitica`, não de um esquema copiado."""
    assert base["DETRATOR"].isin((0, 1)).all()
    assert (base["DETRATOR"] == (base["NPS_PRINCIPAL"] == -100)).all()
    assert base["CATEGORIA_NPS"].isin(("DETRATOR", "NEUTRO", "PROMOTOR")).all()
    assert (base["DATA_STD_CONVERTIDA"] == base["DATA_STD"]).all()
    assert (base["MES_ANO"] == base["DATA_STD"].dt.to_period("M").astype("string")).all()
    # Caixa alta vinda de `padronizar_categoricas`, como na base real.
    assert base["VOO_TIPO"].isin(("DIRETO", "CONEXÃO", "ESCALA")).all()
    # N_TRECHOS é derivada de BASE_AIRPORTLEG e cobre a mesma faixa do real,
    # incluindo os itinerários longos raros.
    assert base["N_TRECHOS"].between(1, 6).all()
    assert base["N_TRECHOS"].max() > 2
    assert (base["N_TRECHOS"] == base["BASE_AIRPORTLEG"].str.count("/")).all()


def test_chave_e_unica_e_acompanha_a_cronologia(base):
    """A ordem por chave tem que reproduzir a cronologia, como no dado real."""
    assert not base["RESPONDENT_ID"].duplicated().any()
    ordem = base["RESPONDENT_ID"].corr(base["DATA_STD"].astype("int64"),
                                       method="spearman")
    assert ordem > 0.999


def test_itinerario_e_coerente_entre_as_colunas(base):
    """Um aeroporto a mais que trechos, e um assento por trecho."""
    trechos = base["N_TRECHOS"]
    assert (base["EQUIPAMENTO_TIPO"].str.count("/") + 1 == trechos).all()
    assert (base["VOO_NUMERO"].str.count("/") + 1 == trechos).all()
    # ASSENTOS vem ausente em poucas linhas, como no dado real.
    assinalado = base["ASSENTOS"].notna()
    assert (base.loc[assinalado, "ASSENTOS"].str.count("/") + 1
            == trechos[assinalado]).all()


def test_historico_por_cliente_tem_o_que_enxergar(base):
    """Sem reincidência de ID_GOLDENRECORD as features de histórico nasceriam vazias."""
    df = features.adicionar_historico(base)
    assert df["HIST_RESPOSTAS_ANTERIORES"].gt(0).any()
    assert df["HIST_DETRATOU_ANTES"].notna().any()


def test_base_atravessa_o_particionamento_e_a_matriz(base):
    """Ponta a ponta até a matriz, com os cortes temporais do projeto."""
    preparo = matriz.preparar_matriz(base, corte_validacao=CORTE_VALIDACAO,
                                     corte_teste=CORTE_TESTE)
    for nome, particao in preparo["particoes"].items():
        assert len(particao) > 0, f"partição {nome} vazia"


def test_nenhum_identificador_real_no_dummy(base):
    """CR04: as chaves são de uma faixa própria, disjunta da do dado real."""
    assert base["RESPONDENT_ID"].min() >= PRIMEIRO_ID
    assert base["ID_GOLDENRECORD"].dropna().min() >= PRIMEIRO_CLIENTE


TIERS_NOVOS = ("AZUL ONE", "DIAMANTE UNIQUE")


def test_tiers_novos_ficam_fora_do_treino(base):
    """Os dois tiers novos são posteriores ao corte, como na base real.

    Na base real `AZUL ONE` e `DIAMANTE UNIQUE` só existem a partir de
    `PRIMEIRO_TIER_NOVO`, depois do corte de validação. O `OneHotEncoder` do
    contrato é ajustado só no treino, então aprende cinco níveis e a matriz sai
    com 38 colunas; as linhas desses dois tiers atravessam a pontuação com o
    bloco de `TIER_VIAGEM` zerado, por `handle_unknown="ignore"`.

    Um gerador que espalha os dois pela série inteira coloca ambos no treino, a
    matriz vai a 40 colunas, e o caminho de categoria não vista deixa de ser
    exercitado justamente pela base que existe para exercitá-lo. Foi o que
    aconteceu até o card #254.
    """
    novos = base["TIER_VIAGEM"].isin(TIERS_NOVOS)
    assert novos.any(), "nenhuma linha de tier novo: a janela pós-corte ficou vazia"
    assert base.loc[novos, "DATA_STD"].min() >= pd.Timestamp(PRIMEIRO_TIER_NOVO)

    preparo = matriz.preparar_matriz(base, corte_validacao=CORTE_VALIDACAO,
                                     corte_teste=CORTE_TESTE)
    treino = preparo["x"]["treino"]["TIER_VIAGEM"]
    assert not treino.isin(TIERS_NOVOS).any()
    assert treino.nunique() == len(TIERS_ANTIGOS)

"""Testes das travas de integridade da seção 4.2.1.

Cada trava foi criada em resposta a um defeito real: descarte silencioso de
duplicata, junção sem validação de cardinalidade, cobertura incompleta virando
peso zero. Um teste que confirma o caminho feliz não prova nada sobre uma trava;
o que interessa é que ela **dispare** quando deve. É isso que este arquivo
verifica, com dados sintéticos e sem tocar nas bases da Azul.

Executar com:  pytest -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clean import (conferir_cobertura, deduplicar, duplicatas_divergentes,
                   faixa_atraso, integrar, pesos_pos_estratificacao)
from preprocessamento_nps import (
    FEATURE_SET_V1,
    FEATURES_SCORE_POS_VIAGEM,
    QUANTIDADE_COLUNAS_INTEGRADAS_ESPERADA,
    consolidar_duplicidades_tempo_voo,
    criar_target_detrator,
    criar_preprocessador_modelagem,
    derivar_n_trechos,
    dividir_treino_teste_temporal_por_cliente,
    normalizar_data_std,
    preparar_base_analitica,
    selecionar_features_score_pos_viagem,
    validar_base_integrada,
    validar_parquet,
)
from stats import (classificar_colunas, cramers_v, diagnostico_pesos, ic_wilson,
                   media_ponderada_ic)


# --------------------------------------------------------------- deduplicacao
def test_remove_apenas_duplicata_integralmente_identica(nps):
    """Cópia idêntica sai; a contagem de linhas cai exatamente uma."""
    com_copia = pd.concat([nps, nps.iloc[[0]]], ignore_index=True)
    resultado = deduplicar(com_copia, "teste")
    assert len(resultado) == len(nps)


def test_bloqueia_chave_repetida_com_conteudo_divergente(nps):
    """O caso que motivou a trava: mesma chave, conteúdo diferente."""
    divergente = nps.copy()
    divergente.loc[1, "RESPONDENT_ID"] = 1          # duplica a chave
    divergente.loc[1, "VOO_TIPO"] = "OUTRO_VALOR"   # com conteúdo diferente

    with pytest.raises(ValueError, match="divergencia"):
        deduplicar(divergente, "teste")


def test_aceita_divergencia_previamente_aprovada(nps):
    """Divergência só na coluna aprovada passa, mantendo a primeira ocorrência."""
    divergente = nps.copy()
    divergente.loc[1, "RESPONDENT_ID"] = 1
    divergente.loc[1, [c for c in nps.columns if c != "RESPONDENT_ID"]] = \
        nps.loc[0, [c for c in nps.columns if c != "RESPONDENT_ID"]].to_numpy()
    divergente.loc[1, "TEMPO_VOO"] = 999

    resultado = deduplicar(divergente, "teste", divergencia_aprovada=("TEMPO_VOO",))
    assert len(resultado) == 2
    assert resultado.loc[resultado["RESPONDENT_ID"] == 1, "TEMPO_VOO"].iloc[0] == 120


def test_duplicatas_divergentes_nomeia_as_colunas(nps):
    """O diagnóstico precisa dizer qual coluna diverge, não apenas que diverge."""
    divergente = nps.copy()
    divergente.loc[1, "RESPONDENT_ID"] = 1
    divergente.loc[1, "TEMPO_VOO"] = 999

    achados = duplicatas_divergentes(divergente)
    assert 1 in achados
    assert "TEMPO_VOO" in achados[1]


def test_consolida_tempo_voo_pela_media_e_registra_a_intervencao(nps):
    """A regra aprovada não pode voltar a manter arbitrariamente a primeira linha."""
    duplicada = nps.iloc[[0]].copy()
    duplicada.loc[:, "TEMPO_VOO"] = 300
    fonte = pd.concat([nps, duplicada], ignore_index=True)

    resultado, quantidade = consolidar_duplicidades_tempo_voo(fonte)

    linha = resultado.loc[resultado["RESPONDENT_ID"] == 1].iloc[0]
    assert quantidade == 1
    assert linha["TEMPO_VOO"] == 210
    assert linha["TEMPO_VOO_CONSOLIDADO"] == 1


def test_bloqueia_consolidacao_quando_duplicata_diverge_em_outra_coluna(nps):
    duplicada = nps.iloc[[0]].copy()
    duplicada.loc[:, "TEMPO_VOO"] = 300
    duplicada.loc[:, "VOO_TIPO"] = "Escala"

    with pytest.raises(ValueError, match="além de TEMPO_VOO"):
        consolidar_duplicidades_tempo_voo(pd.concat([nps, duplicada], ignore_index=True))


# ------------------------------------------------------------------ cobertura
def test_cobertura_completa_nao_levanta(nps, perfil, viagem):
    conferir_cobertura(nps, perfil, viagem)   # não deve levantar


def test_bloqueia_cobertura_incompleta(nps, perfil, viagem):
    """Sem a trava, o inner join descartaria a resposta órfã em silêncio."""
    perfil_incompleto = perfil.iloc[:2]

    with pytest.raises(ValueError, match="Cobertura incompleta"):
        conferir_cobertura(nps, perfil_incompleto, viagem)


def test_integracao_preserva_a_contagem(nps, perfil, viagem):
    df = integrar(nps, perfil, viagem)
    assert len(df) == len(nps)
    assert "VOO_INTERNACIONAL" not in df.columns   # constante, removida por F4


def test_bloqueia_chave_duplicada_na_auxiliar(nps, perfil, viagem):
    """Cardinalidade violada tem que parar, não multiplicar linhas."""
    perfil_duplicado = pd.concat([perfil, perfil.iloc[[0]]], ignore_index=True)
    perfil_duplicado.loc[3, "QTDE_VIAGENS_12M"] = 99   # divergente, logo não é cópia

    with pytest.raises(ValueError):
        integrar(nps, perfil_duplicado, viagem)


def test_bloqueia_id_goldenrecord_divergente(nps, perfil, viagem):
    """A cópia redundante só pode ser descartada se de fato coincidir."""
    perfil_alterado = perfil.copy()
    perfil_alterado.loc[0, "ID_GOLDENRECORD"] = 999

    with pytest.raises(ValueError, match="ID_GOLDENRECORD"):
        integrar(nps, perfil_alterado, viagem)


def test_id_goldenrecord_nulo_dos_dois_lados_e_equivalente(nps, perfil, viagem):
    """NaN != NaN em comparação direta; nulo nos dois lados não é divergência."""
    nps_nulo, perfil_nulo = nps.copy(), perfil.copy()
    nps_nulo.loc[0, "ID_GOLDENRECORD"] = np.nan
    perfil_nulo.loc[0, "ID_GOLDENRECORD"] = np.nan

    df = integrar(nps_nulo, perfil_nulo, viagem)
    assert len(df) == len(nps)


# ------------------------------------------------------- pos-estratificacao
def _base_com_estratos():
    """Base mínima com as três colunas da chave de pós-estratificação."""
    return pd.DataFrame({
        "MES_ANO": pd.to_datetime(["2024-01-01"] * 4),
        "FAIXA_ATRASO": faixa_atraso(pd.Series([0, 0, 200, 200])),
        "CANAL_COMPRA": ["Web", "Agency", "Web", "Agency"],
        "DETRATOR": [0, 0, 1, 1],
    })


def _populacao(faixas=("a. Sem Atraso", "d. >120m"), canais=("Web", "Agency")):
    linhas = [{"MES_ANO": pd.Timestamp("2024-01-01"),
               "DELAY_DEPARTURE_RANGE": f, "CANAL_COMPRA": c, "PERC_PAX": 0.25}
              for f in faixas for c in canais]
    return pd.DataFrame(linhas)


def test_peso_calculado_para_todos_os_estratos():
    df = pesos_pos_estratificacao(_base_com_estratos(), _populacao())
    assert (df["PESO_POP"] > 0).all()


def test_bloqueia_estrato_sem_contrapartida_populacional():
    """O fillna(0) removia esses respondentes do estimador sem avisar."""
    populacao_incompleta = _populacao(canais=("Web",))   # falta Agency

    with pytest.raises(ValueError, match="Pos-estratificacao incompleta|sem PESO_POP"):
        pesos_pos_estratificacao(_base_com_estratos(), populacao_incompleta)


def test_modo_permissivo_preenche_com_zero():
    """O escape existe, mas exige ser pedido explicitamente."""
    df = pesos_pos_estratificacao(_base_com_estratos(), _populacao(canais=("Web",)),
                                  exigir_cobertura_integral=False)
    assert (df["PESO_POP"] == 0).any()


# ------------------------------------------------------------- estatistica
def test_n_efetivo_igual_a_n_com_pesos_iguais():
    """Pesos uniformes não perdem eficiência: n efetivo tem que bater com n."""
    df = pd.DataFrame({"PESO_POP": np.ones(100)})
    d = diagnostico_pesos(df)
    assert d["n efetivo (Kish)"] == pytest.approx(100)
    assert d["Perda de eficiência (%)"] == pytest.approx(0)


def test_n_efetivo_cai_com_pesos_desiguais():
    df = pd.DataFrame({"PESO_POP": np.array([1.0] * 99 + [100.0])})
    assert diagnostico_pesos(df)["n efetivo (Kish)"] < 100


def test_media_ponderada_com_pesos_iguais_e_a_media_simples():
    y = np.array([0, 1, 1, 0, 1])
    media, inf, sup, _ = media_ponderada_ic(y, np.ones(5))
    assert media == pytest.approx(y.mean())
    assert inf < media < sup


def test_wilson_dentro_do_intervalo_valido():
    inf, sup = ic_wilson(50, 100)
    assert 0 <= inf < 50 < sup <= 100


def test_nulo_como_categoria_muda_o_v_de_cramer():
    """Descartar nulo estrutural esconde sinal: é o caso de TIPO_ENTRETENIMENTO.

    O grupo nulo detrata muito acima dos demais, enquanto A e B quase não se
    distinguem entre si. Descartando o nulo, sobra apenas a diferença fraca.
    """
    x = pd.Series([None] * 50 + ["A"] * 50 + ["B"] * 50)
    y = pd.Series([1] * 45 + [0] * 5        # nulo: 90% de detração
                  + [1] * 10 + [0] * 40     # A: 20%
                  + [1] * 12 + [0] * 38)    # B: 24%, quase igual a A

    com_nulo = cramers_v(x, y, nulo_como_categoria=True)
    sem_nulo = cramers_v(x, y, nulo_como_categoria=False)

    assert com_nulo > sem_nulo
    assert com_nulo > 0.5      # o sinal do grupo nulo é forte
    assert sem_nulo < 0.1      # sem ele, sobra quase nada


def test_classificar_colunas_devolve_tres_listas():
    """A assinatura de 2 valores quebrava o notebook; temporais têm lista própria."""
    df = pd.DataFrame({
        "numerica": [1.5, 2.5, 3.5, 4.5],
        "flag": [0, 1, 0, 1],
        "texto": ["a", "b", "c", "d"],
        "data": pd.to_datetime(["2024-01-01"] * 4),
    })
    num, cat, temporais = classificar_colunas(df)
    assert num == ["numerica"]
    assert set(cat) == {"flag", "texto"}
    assert temporais == ["data"]


def test_faixa_atraso_respeita_as_bordas_da_taxonomia():
    faixas = faixa_atraso(pd.Series([0, 14, 15, 60, 61, 120, 121]))
    assert list(faixas) == ["a. Sem Atraso", "a. Sem Atraso", "b. 15m - 60m",
                            "b. 15m - 60m", "c. 61m - 120m", "c. 61m - 120m",
                            "d. >120m"]
    assert faixas.ordered


def test_normaliza_data_std_mista_antes_da_concatenacao():
    """Timestamp do Excel e texto de CSV precisam chegar ao mesmo dtype."""
    fonte_mista = pd.DataFrame({
        "DATA_STD": [pd.Timestamp("2023-07-01"), "2024-01-06"],
    })

    resultado = normalizar_data_std(fonte_mista, Path("NPS_teste.xlsx"))

    assert pd.api.types.is_datetime64_any_dtype(resultado["DATA_STD"])
    assert resultado["DATA_STD"].notna().all()


def test_bloqueia_data_std_ausente_ou_invalida_na_leitura():
    """Datas inválidas não podem alcançar a concatenação nem o split."""
    fonte_invalida = pd.DataFrame({"DATA_STD": ["2024-01-06", "data-invalida", None]})

    with pytest.raises(ValueError, match="inválido"):
        normalizar_data_std(fonte_invalida, Path("NPS_teste.csv"))


def test_bloqueia_base_integrada_com_quantidade_de_colunas_inesperada():
    fonte = pd.DataFrame({f"COLUNA_{i}": [i] for i in range(QUANTIDADE_COLUNAS_INTEGRADAS_ESPERADA - 1)})

    with pytest.raises(ValueError, match="eram esperadas"):
        validar_base_integrada(fonte)


def test_aceita_base_integrada_com_schema_de_quantidade_esperada():
    fonte = pd.DataFrame({f"COLUNA_{i}": [i] for i in range(QUANTIDADE_COLUNAS_INTEGRADAS_ESPERADA)})

    validar_base_integrada(fonte)


def test_bloqueia_parquet_invalido_antes_da_leitura(tmp_path):
    arquivo = tmp_path / "invalido.parquet"
    arquivo.write_bytes(b"nao e um parquet")

    with pytest.raises(ValueError, match="Parquet inválido"):
        validar_parquet(arquivo)


# --------------------------------------------------------------- target NPS
def test_cria_target_binario_para_codificacao_nps_da_azul():
    fonte = pd.DataFrame({"NPS_PRINCIPAL": [-100, 0, 100]})

    target = criar_target_detrator(fonte)

    assert target.name == "DETRATOR"
    assert target.dtype == "int8"
    assert target.tolist() == [1, 0, 0]


@pytest.mark.parametrize("valor", [None, -1, 6, 7, 10])
def test_bloqueia_target_quando_nps_principal_nao_segue_codificacao_contratada(valor):
    fonte = pd.DataFrame({"NPS_PRINCIPAL": [valor]})

    with pytest.raises(ValueError):
        criar_target_detrator(fonte)


# ---------------------------------------------------------- features do score
def base_feature_set_v1(n_linhas=2):
    """Contrato completo da V1 com valores exclusivamente sintéticos."""
    return pd.DataFrame({
        "TIER_VIAGEM": ["TIER_A" if i % 2 else "TIER_B" for i in range(n_linhas)],
        "VOO_TIPO": ["DIRETO" if i % 2 else "CONEXAO" for i in range(n_linhas)],
        "TIPO_ENTRETENIMENTO": ["SISTEMA_A" if i % 2 else pd.NA for i in range(n_linhas)],
        "CANAL_COMPRA": ["CANAL_A" if i % 2 else "CANAL_B" for i in range(n_linhas)],
        "SEGMENTO": ["SEGMENTO_A" if i % 2 else "SEGMENTO_B" for i in range(n_linhas)],
        "ESTATISTICA_ATRASOSAIDA": list(range(n_linhas)),
        "ATRASO_CHEGADA": list(range(10, 10 + n_linhas)),
        "CANCELAMENTO_VOO": [False] * n_linhas,
        "ANTECEDENCIA_CANCELAMENTO": [np.nan] * n_linhas,
        "TEMPO_VOO": list(range(100, 100 + n_linhas)),
        "N_TRECHOS": [1 + i % 2 for i in range(n_linhas)],
    })


def test_allowlist_do_score_exclui_alvo_pesquisa_e_campos_tecnicos():
    fonte = base_feature_set_v1(1).assign(
        NPS_COMISSARIOS=-100,
        DETRATOR=1,
        RESPONDENT_ID=1,
        TEMPO_VOO_CONSOLIDADO=1,
        CAMPO_TECNICO_NOVO=999,
    )

    matriz, selecionadas, ausentes = selecionar_features_score_pos_viagem(fonte)

    assert selecionadas == FEATURE_SET_V1
    assert list(matriz.columns) == selecionadas
    assert "NPS_COMISSARIOS" not in matriz
    assert "TEMPO_VOO_CONSOLIDADO" not in matriz
    assert ausentes == []


def test_feature_set_v1_tem_composicao_explicita_e_sem_redundancias_conhecidas():
    assert FEATURE_SET_V1 == [
        "TIER_VIAGEM",
        "VOO_TIPO",
        "TIPO_ENTRETENIMENTO",
        "CANAL_COMPRA",
        "SEGMENTO",
        "ESTATISTICA_ATRASOSAIDA",
        "ATRASO_CHEGADA",
        "CANCELAMENTO_VOO",
        "ANTECEDENCIA_CANCELAMENTO",
        "TEMPO_VOO",
        "N_TRECHOS",
    ]
    assert "PERFIL_TUDOAZUL" not in FEATURE_SET_V1
    assert "QTDE_VIAGENS_12M" not in FEATURE_SET_V1
    assert "QTDE_VIAGENS_24M" not in FEATURE_SET_V1
    assert "QTDE_VIAGENS_36M" not in FEATURE_SET_V1


def test_feature_set_v1_bloqueia_todo_campo_da_pesquisa_e_preserva_id_estrutural():
    fonte = base_feature_set_v1(1).assign(
        NPS_CAMPO_NOVO=-100,
        SUB_ENTRETENIMENTO1="SIM",
        SUB_ENTRETENIMENTO2="FALHA_SINTETICA",
        SUB_FIL_MOTIVOVIAGEM="MOTIVO_SINTETICO",
        SUB_FIL_FREQUENCIAAZUL="FREQUENCIA_SINTETICA",
        ID_GOLDENRECORD=10,
    )

    matriz, selecionadas, _ = selecionar_features_score_pos_viagem(fonte)

    assert selecionadas == FEATURE_SET_V1
    assert list(matriz.columns) == FEATURE_SET_V1
    assert fonte["ID_GOLDENRECORD"].tolist() == [10]


def test_deriva_n_trechos_sem_usar_assentos_ou_expor_rota_bruta():
    fonte = pd.DataFrame({
        "BASE_AIRPORTLEG": ["AAA/BBB", "AAA/CCC/BBB", pd.NA],
        "ASSENTOS": ["1A", "1A/2B", "3C"],
    })

    resultado = derivar_n_trechos(fonte)

    assert resultado.tolist() == [1, 2, pd.NA]


def test_selecao_materializa_n_trechos_quando_a_rota_bruta_esta_disponivel():
    fonte = base_feature_set_v1().drop(columns="N_TRECHOS")
    fonte["BASE_AIRPORTLEG"] = ["AAA/BBB", "AAA/CCC/BBB"]

    matriz, selecionadas, _ = selecionar_features_score_pos_viagem(fonte)

    assert selecionadas == FEATURE_SET_V1
    assert list(matriz.columns) == selecionadas
    assert matriz["N_TRECHOS"].tolist() == [1, 2]
    assert "BASE_AIRPORTLEG" not in matriz


def test_preparar_base_analitica_materializa_n_trechos():
    """#177: a base salva precisa trazer N_TRECHOS pronta, e nao so sob demanda.

    `selecionar_features_score_pos_viagem` ja materializa N_TRECHOS na hora de
    montar a matriz, mas quem so carrega o parquet salvo (como a celula de
    diagnostico do notebook de modelagem) nunca passa por ali. Sem essa coluna
    no parquet, o diagnostico recusa uma base que na pratica esta correta.
    """
    fonte = pd.DataFrame({
        "DATA_STD": ["2024-01-01", "2024-01-02"],
        "NPS_PRINCIPAL": [-100, 100],
        "TEMPO_VOO": [90.0, 120.0],
        "BASE_AIRPORTLEG": ["AAA/BBB", "AAA/CCC/BBB"],
    })

    resultado = preparar_base_analitica(fonte)

    assert "N_TRECHOS" in resultado.columns
    assert resultado["N_TRECHOS"].tolist() == [1, 2]


def test_selecao_falha_quando_feature_obrigatoria_esta_ausente():
    fonte = base_feature_set_v1().drop(columns="TIER_VIAGEM")

    with pytest.raises(KeyError, match="TIER_VIAGEM"):
        selecionar_features_score_pos_viagem(fonte)


def test_selecao_falha_quando_dtype_diverge_do_contrato():
    fonte = base_feature_set_v1()
    fonte["ATRASO_CHEGADA"] = fonte["ATRASO_CHEGADA"].astype("string")

    with pytest.raises(TypeError, match="ATRASO_CHEGADA"):
        selecionar_features_score_pos_viagem(fonte)


def test_contagem_sem_corte_em_t_score_nao_entra_na_v1():
    fonte = base_feature_set_v1().assign(QTDE_VIAGENS_12M=[999, 999])

    matriz, _, _ = selecionar_features_score_pos_viagem(fonte)

    assert "QTDE_VIAGENS_12M" not in matriz


def test_score_de_cancelamento_mascara_features_posteriores_ao_evento():
    fonte = base_feature_set_v1()
    fonte.loc[1, "CANCELAMENTO_VOO"] = True
    fonte.loc[1, "ANTECEDENCIA_CANCELAMENTO"] = 2

    matriz, _, _ = selecionar_features_score_pos_viagem(fonte)

    posteriores = [
        "ESTATISTICA_ATRASOSAIDA", "ATRASO_CHEGADA", "TEMPO_VOO", "N_TRECHOS"
    ]
    assert matriz.loc[1, posteriores].isna().all()
    assert matriz.loc[0, posteriores].notna().all()
    assert matriz.loc[1, "ANTECEDENCIA_CANCELAMENTO"] == 2


def test_allowlist_do_score_inclui_tier_viagem_quando_presente_na_fonte():
    """TIER_VIAGEM entra na matriz e o alias legado permanece sincronizado."""
    fonte = base_feature_set_v1(1)

    matriz, selecionadas, ausentes = selecionar_features_score_pos_viagem(fonte)

    assert ausentes == []
    assert "TIER_VIAGEM" in selecionadas
    assert "TIER_VIAGEM" in matriz.columns
    assert FEATURES_SCORE_POS_VIAGEM == tuple(FEATURE_SET_V1)


def test_preprocessador_usa_allowlist_em_vez_de_cardinalidade():
    fonte = base_temporal().join(base_feature_set_v1(10)).assign(
        DETRATOR=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        NPS_COMISSARIOS=[100, -100] * 5,
        TEMPO_VOO_CONSOLIDADO=[0] * 10,
        CAMPO_NUMERICO_NOVO=list(range(10)),
    )

    preprocessador, (x_treino, x_teste, _, _), matrizes, (numericas, categoricas), _ = \
        criar_preprocessador_modelagem(fonte)

    assert list(x_treino.columns) == FEATURE_SET_V1
    assert list(x_teste.columns) == FEATURE_SET_V1
    assert numericas == [
        "ESTATISTICA_ATRASOSAIDA", "ATRASO_CHEGADA",
        "ANTECEDENCIA_CANCELAMENTO", "TEMPO_VOO", "N_TRECHOS",
    ]
    assert categoricas == [
        "TIER_VIAGEM", "VOO_TIPO", "TIPO_ENTRETENIMENTO",
        "CANAL_COMPRA", "SEGMENTO", "CANCELAMENTO_VOO",
    ]
    codificador = preprocessador.named_transformers_["categoricas"].named_steps["codificar"]
    indice_entretenimento = categoricas.index("TIPO_ENTRETENIMENTO")
    assert "CATEGORIA_AUSENTE" in codificador.categories_[indice_entretenimento]

    categoria_nova = x_teste.iloc[[0]].copy()
    categoria_nova["TIER_VIAGEM"] = "TIER_INEDITO_SINTETICO"
    transformada = preprocessador.transform(categoria_nova)
    assert transformada.shape[1] == matrizes[1].shape[1]


def test_schema_bloqueia_cancelamento_nulo():
    fonte = base_feature_set_v1()
    fonte["CANCELAMENTO_VOO"] = pd.Series([False, pd.NA], dtype="boolean")

    with pytest.raises(ValueError, match="CANCELAMENTO_VOO"):
        selecionar_features_score_pos_viagem(fonte)


# ------------------------------------------------ divisao temporal por cliente
def base_temporal():
    """Seis meses de respostas, com Cliente recorrente e Cliente ausente.

    O Cliente 10 responde em janeiro e em maio; maio cai no teste, logo a
    resposta de janeiro precisa sair do treino. Duas linhas ficam sem
    ID_GOLDENRECORD, uma de cada lado do corte temporal.
    """
    return pd.DataFrame({
        "DATA_STD_CONVERTIDA": pd.to_datetime([
            "2024-01-10", "2024-01-20", "2024-02-05", "2024-03-05", "2024-03-15",
            "2024-04-10", "2024-05-10", "2024-05-20", "2024-06-10", "2024-06-20",
        ]),
        "ID_GOLDENRECORD": [10, 20, 30, np.nan, 40, 50, 10, 60, np.nan, 70],
    })


def test_divisao_exclui_da_validacao_os_registros_sem_cliente():
    """Os 103 registros sem ID_GOLDENRECORD da base real nao podem travar o split.

    Como o nulo ocorre ao mesmo tempo na pesquisa e no perfil, nao ha como
    saber se duas dessas linhas sao do mesmo Cliente: trata-las como grupos
    unitarios reabriria o vazamento. Elas ficam fora do treino e do teste.
    """
    df = base_temporal()
    treino, teste, metadados = dividir_treino_teste_temporal_por_cliente(df)

    sem_cliente = df.index[df["ID_GOLDENRECORD"].isna()]
    assert not set(sem_cliente) & set(treino)
    assert not set(sem_cliente) & set(teste)
    assert metadados["registros_sem_cliente_excluidos"] == 2


def test_divisao_nao_deixa_o_mesmo_cliente_nos_dois_conjuntos():
    """A trava contra vazamento continua valendo depois da exclusao dos nulos."""
    df = base_temporal()
    treino, teste, metadados = dividir_treino_teste_temporal_por_cliente(df)

    clientes_treino = set(df.loc[treino, "ID_GOLDENRECORD"])
    clientes_teste = set(df.loc[teste, "ID_GOLDENRECORD"])
    assert not clientes_treino & clientes_teste
    assert metadados["linhas_removidas_por_recorrencia"] == 1   # o Cliente 10 em janeiro
    assert df.loc[teste, "DATA_STD_CONVERTIDA"].min() > df.loc[treino, "DATA_STD_CONVERTIDA"].max()


def test_divisao_registra_em_log_a_exclusao_dos_sem_cliente(capsys):
    """Descarte silencioso foi o defeito de origem das travas: aqui ele e anunciado."""
    dividir_treino_teste_temporal_por_cliente(base_temporal())

    saida = capsys.readouterr().out
    assert "sem ID_GOLDENRECORD" in saida
    assert "2 registro" in saida


def test_divisao_falha_quando_nenhum_registro_tem_cliente():
    """Excluir os nulos nao pode virar excluir a base inteira sem avisar."""
    df = base_temporal()
    df["ID_GOLDENRECORD"] = np.nan

    with pytest.raises(ValueError, match="ID_GOLDENRECORD"):
        dividir_treino_teste_temporal_por_cliente(df)

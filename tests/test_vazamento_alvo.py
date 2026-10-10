"""Trava automatizada contra vazamento de alvo na matriz de features.

A seção 1.4 do `notebooks/modelagem.ipynb` já cobra o contrato da matriz, mas a
cobrança vive dentro de uma célula: só dispara quando alguém abre o notebook e
executa até lá. O risco R01 da seção 4.1.5 não pode depender disso. Este arquivo
move a mesma verificação para a suíte, onde ela roda a cada `pytest` e barra a
contaminação antes de o modelo ser treinado.

O que se verifica não é a lista de colunas declarada em algum lugar, e sim a
matriz que o pré-processador realmente entrega ao treino, nas duas pontas: as
features de entrada (`x_treino.columns`) e os nomes gerados na saída
(`get_feature_names_out`). Uma lista atualizada com uma matriz desatualizada
passaria despercebida se a checagem parasse na declaração.

A proteção efetiva é a allowlist `FEATURE_SET_V1`: nada entra por
inferência de tipo ou cardinalidade. Por isso injetar uma coluna proibida na
*fixture* não faz a trava disparar — a allowlist a descarta antes. O que faz a
trava disparar é a allowlist ser contaminada, que é o defeito real contra o qual
este arquivo protege, e é assim que os testes de disparo abaixo o reproduzem.

Tudo roda sobre fixture sintética. Nenhum teste lê `data/`, em linha com o Termo
de Abertura, que veda versionar ou publicar base do parceiro.

Executar com:  pytest tests/test_vazamento_alvo.py -v
"""
import pandas as pd
import pytest

import preprocessamento_nps
from preprocessamento_nps import (
    FEATURE_SET_V1,
    criar_preprocessador_modelagem,
    selecionar_features_score_pos_viagem,
)


# --------------------------------------------------------- lista de proibidos
# Cada entrada declara o motivo da exclusão. O motivo importa tanto quanto o
# nome: sem ele, a próxima pessoa que precisar de uma feature nova não tem como
# saber se está diante de uma regra ou de um esquecimento.
COLUNAS_PROIBIDAS = {
    "DETRATOR":
        "É o próprio alvo. Entra como rótulo y, nunca como coluna de X.",
    "NPS_PRINCIPAL":
        "Resposta original da pesquisa, de -100 a 100, da qual DETRATOR é "
        "derivado dentro de preparar_base_analitica. É o alvo em escala bruta.",
    "NPS_CHECKIN":
        "Nota de etapa da jornada, coletada no mesmo instrumento que origina o "
        "alvo e no mesmo momento: só existe depois da predição.",
    "NPS_EMBARQUE":
        "Nota de etapa da jornada; mesmo instrumento e mesmo momento do alvo.",
    "NPS_COMISSARIOS":
        "Nota de etapa da jornada; mesmo instrumento e mesmo momento do alvo.",
    "SUB_LIMPEZA":
        "Subquestão da mesma pesquisa de NPS; posterior ao momento da predição.",
    "ID_GOLDENRECORD":
        "Identificador do Cliente. É a chave de agrupamento da divisão temporal, "
        "não um atributo da viagem: sem poder preditivo e permite ao modelo "
        "memorizar o Cliente em vez de aprender o comportamento.",
    "RESPONDENT_ID":
        "Identificador da resposta. Permite memorização linha a linha e não "
        "existe fora da base de pesquisa.",
    "DATA_STD":
        "Data da resposta na forma textual lida da fonte. Não é atributo da "
        "viagem e, como chave do corte temporal, informaria ao modelo de que "
        "lado da divisão a linha caiu.",
    "DATA_STD_CONVERTIDA":
        "Mesma data já normalizada, usada por dividir_treino_teste_temporal_por_"
        "cliente para definir treino e teste. Vale a mesma exclusão.",
    "TEMPO_VOO_CONSOLIDADO":
        "Marcador técnico de intervenção da limpeza, não atributo da viagem. "
        "Registra o que o pipeline fez com a linha, não o que o Cliente viveu.",
    "PESO_POP":
        "Peso de pós-estratificação, construído a partir da distribuição do "
        "alvo na amostra. Serve à estimativa populacional, não ao treino.",
}

# Prefixos capturam também as derivadas: uma coluna nova nascida de uma resposta
# de pesquisa continua sendo vazamento, ainda que receba outro nome. A allowlist
# já barra o caso por construção; o prefixo garante a mensagem de erro correta
# caso a allowlist seja alterada.
PREFIXOS_PROIBIDOS = {
    "NPS_":
        "Toda resposta da pesquisa de NPS, e qualquer derivada dela, carrega o "
        "alvo e é posterior ao momento da predição.",
    "SUB_":
        "Subquestões da pesquisa de NPS; mesma origem e mesmo momento.",
}


def conferir_ausencia_de_vazamento(colunas, origem: str) -> None:
    """Levanta se alguma coluna proibida alcançar a matriz do modelo.

    O casamento é por *substring* porque a saída do ``ColumnTransformer`` vem
    prefixada pelo nome do bloco e sufixada pela categoria: ``ATRASO_CHEGADA``
    chega como ``numericas__ATRASO_CHEGADA`` e ``VOO_TIPO`` como
    ``categoricas__VOO_TIPO_Direto``. Comparar por igualdade deixaria passar a
    coluna proibida justamente na ponta em que ela é mais difícil de enxergar.
    """
    achados = []
    for coluna in colunas:
        for prefixo, motivo in PREFIXOS_PROIBIDOS.items():
            if prefixo in coluna:
                achados.append(f"{coluna} ({motivo})")
                break
        else:
            for proibida, motivo in COLUNAS_PROIBIDAS.items():
                if proibida in coluna:
                    achados.append(f"{coluna} ({motivo})")
                    break
    if achados:
        raise AssertionError(
            f"Vazamento de alvo na matriz do modelo, em {origem}: "
            + "; ".join(achados)
        )


# ---------------------------------------------------------- fixture sintética
def _ciclo(valores, n):
    return [valores[i % len(valores)] for i in range(n)]


def base_sintetica() -> pd.DataFrame:
    """Seis meses de jornadas encerradas, com as 11 features e o que é proibido.

    A base carrega as colunas de pesquisa de propósito: é assim que a base
    analítica real chega ao notebook, já que ``DETRATOR`` é derivado de
    ``NPS_PRINCIPAL`` e a persistência em Parquet não descarta coluna alguma.
    Exigir uma base limpa aqui testaria uma entrada que não existe.

    Dois Clientes reaparecem nos meses de teste, para que a remoção por
    recorrência da divisão temporal também seja exercitada.
    """
    datas = [f"2024-{mes:02d}-{5 + 3 * i:02d}" for mes in range(1, 7) for i in range(5)]
    n = len(datas)

    clientes = list(range(101, 101 + n))
    clientes[0] = clientes[26]   # recorrente: aparece em janeiro e em junho
    clientes[3] = clientes[27]   # recorrente: aparece em janeiro e em junho

    return pd.DataFrame({
        # chaves de partição
        "DATA_STD_CONVERTIDA": pd.to_datetime(datas),
        "ID_GOLDENRECORD": clientes,
        # alvo
        "DETRATOR": _ciclo([0, 1, 0, 0, 1], n),
        # as 11 features da allowlist
        "TIER_VIAGEM": _ciclo(["Diamante", "Safira", "Topazio", "Sem cadastro"], n),
        "VOO_TIPO": _ciclo(["Direto", "Conexão"], n),
        "TIPO_ENTRETENIMENTO": _ciclo(["Wi-Fi", "Tela individual", None], n),
        "CANAL_COMPRA": _ciclo(["Web", "Agency", "Mobile"], n),
        "SEGMENTO": _ciclo(["Lazer", "Corporativo"], n),
        "ESTATISTICA_ATRASOSAIDA": _ciclo([0, 20, 75, 130], n),
        "ATRASO_CHEGADA": _ciclo([0, 25, 80, 140], n),
        "CANCELAMENTO_VOO": _ciclo([False, False, False, True], n),
        "ANTECEDENCIA_CANCELAMENTO": _ciclo([None, None, None, 5.0], n),
        "TEMPO_VOO": _ciclo([90, 120, 240, 300], n),
        "N_TRECHOS": _ciclo([1, 2, 3], n),
        # o que não pode alcançar a matriz
        "NPS_PRINCIPAL": _ciclo([100, -100, 0, 50, -50], n),
        "NPS_CHECKIN": _ciclo([100, -100, 0], n),
        "NPS_EMBARQUE": _ciclo([0, 100, -100], n),
        "NPS_COMISSARIOS": _ciclo([-100, 100], n),
        "SUB_LIMPEZA": _ciclo([1, 0], n),
        "RESPONDENT_ID": list(range(1, n + 1)),
        "DATA_STD": datas,
        "TEMPO_VOO_CONSOLIDADO": _ciclo([0, 1], n),
        "PESO_POP": _ciclo([0.8, 1.2], n),
        # derivada de pesquisa batizada com outro nome: não tem prefixo proibido
        # e ainda assim não pode entrar, porque não está na allowlist
        "SATISFACAO_MEDIA_ETAPAS": _ciclo([33.3, -66.6, 0.0], n),
    })


@pytest.fixture
def base():
    return base_sintetica()


# ------------------------------------------------ coerência da lista declarada
def test_lista_de_proibidos_nao_colide_com_a_allowlist():
    """Uma feature aprovada na lista de proibidos travaria a matriz correta."""
    colisao = set(COLUNAS_PROIBIDAS) & set(FEATURE_SET_V1)
    assert not colisao, f"Colunas em ambas as listas: {sorted(colisao)}"

    for aprovada in FEATURE_SET_V1:
        conferir_ausencia_de_vazamento([aprovada], "allowlist")


def test_todo_proibido_tem_motivo_registrado():
    """CR04: a lista sem justificativa vira ruído e é removida na primeira dúvida."""
    for coluna, motivo in {**COLUNAS_PROIBIDAS, **PREFIXOS_PROIBIDOS}.items():
        assert motivo.strip(), f"{coluna} está na lista sem motivo declarado."


# --------------------------------------------- a matriz atual está limpa (CR02)
def test_matriz_de_treino_usa_exatamente_a_allowlist(base):
    _, (x_treino, x_teste, _, _), _, _, _ = criar_preprocessador_modelagem(base)

    assert set(x_treino.columns) == set(FEATURE_SET_V1)
    assert list(x_teste.columns) == list(x_treino.columns)


def test_nenhuma_coluna_proibida_alcanca_a_entrada_da_matriz(base):
    _, (x_treino, x_teste, _, _), _, _, _ = criar_preprocessador_modelagem(base)

    conferir_ausencia_de_vazamento(x_treino.columns, "entrada do pré-processador (treino)")
    conferir_ausencia_de_vazamento(x_teste.columns, "entrada do pré-processador (teste)")


def test_nenhuma_coluna_proibida_alcanca_a_saida_do_preprocessador(base):
    """A ponta que o modelo enxerga: nomes gerados após imputação e codificação."""
    preprocessador, _, _, _, _ = criar_preprocessador_modelagem(base)

    conferir_ausencia_de_vazamento(
        preprocessador.get_feature_names_out(), "saída do pré-processador"
    )


def test_derivada_de_pesquisa_com_outro_nome_nao_entra(base):
    """SATISFACAO_MEDIA_ETAPAS não tem prefixo proibido; a allowlist a barra."""
    _, (x_treino, _, _, _), _, _, _ = criar_preprocessador_modelagem(base)

    assert "SATISFACAO_MEDIA_ETAPAS" not in x_treino.columns


def test_matriz_nao_depende_de_nenhuma_base_do_parceiro(base, monkeypatch):
    """CR03: qualquer leitura de arquivo aqui seria um caminho para data/."""
    def recusar(*args, **kwargs):
        raise AssertionError("O teste tentou ler um arquivo; ele deve rodar só com a fixture.")

    monkeypatch.setattr(pd, "read_parquet", recusar)
    monkeypatch.setattr(pd, "read_csv", recusar)
    monkeypatch.setattr(pd, "read_excel", recusar)

    criar_preprocessador_modelagem(base)


# ----------------------------------------------- a trava de fato dispara (CR01)
@pytest.mark.parametrize("proibida", sorted(COLUNAS_PROIBIDAS))
def test_trava_dispara_quando_a_allowlist_e_contaminada(base, monkeypatch, proibida):
    """O defeito real: alguém acrescenta a coluna à allowlist do módulo.

    O contrato V1 valida a própria allowlist antes de selecionar a matriz, então
    a contaminação precisa ser recusada já nessa primeira barreira.
    """
    monkeypatch.setattr(
        preprocessamento_nps,
        "FEATURE_SET_V1",
        FEATURE_SET_V1 + [proibida],
    )
    with pytest.raises(RuntimeError, match="FEATURE_SET_V1"):
        selecionar_features_score_pos_viagem(base)


def test_trava_dispara_na_matriz_real_com_allowlist_contaminada(base, monkeypatch):
    """O percurso completo é interrompido antes do ajuste do pré-processador."""
    monkeypatch.setattr(
        preprocessamento_nps,
        "FEATURE_SET_V1",
        FEATURE_SET_V1 + ["NPS_PRINCIPAL"],
    )
    with pytest.raises(RuntimeError, match="NPS_PRINCIPAL"):
        criar_preprocessador_modelagem(base)


def test_trava_enxerga_o_nome_prefixado_pela_saida_do_transformer():
    """A comparação por igualdade falharia aqui; é o caso que exige substring."""
    with pytest.raises(AssertionError, match="ID_GOLDENRECORD"):
        conferir_ausencia_de_vazamento(
            ["numericas__ID_GOLDENRECORD"], "saída do pré-processador"
        )

    with pytest.raises(AssertionError, match="NPS_"):
        conferir_ausencia_de_vazamento(
            ["categoricas__NPS_PRINCIPAL_100"], "saída do pré-processador"
        )


def test_trava_nao_dispara_sobre_nome_prefixado_de_feature_aprovada():
    """Falso positivo custa caro: bloquearia a matriz correta e seria desligado."""
    conferir_ausencia_de_vazamento(
        [
            "numericas__ATRASO_CHEGADA",
            "numericas__TEMPO_VOO",
            "numericas__N_TRECHOS",
            "categoricas__VOO_TIPO_Direto",
            "categoricas__CANCELAMENTO_VOO_True",
            "categoricas__TIER_VIAGEM_Sem cadastro",
        ],
        "saída do pré-processador",
    )

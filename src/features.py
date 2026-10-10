"""Features de Cliente e historico, com a anterioridade garantida por construcao.

Este e o bloco de maior risco de vazamento temporal da matriz. A hipotese 4 da
secao 4.2.4 mostrou que quem detratou uma vez volta a detratar com chance 4,61
vezes maior, e que o efeito sobrevive ao controle por atraso e cancelamento. Isso
faz do historico um preditor forte e, exatamente por isso, perigoso: qualquer
agregado calculado sobre a janela inteira da base carregaria respostas
posteriores ao voo que esta sendo pontuado, e o modelo passaria a prever a
resposta usando a propria resposta.

A garantia aqui e estrutural, nao um cuidado de quem escreve a celula. Cada linha
enxerga apenas as respostas do mesmo Cliente que a precedem, e a ordem vem de
`RESPONDENT_ID`, que acompanha a cronologia com correlacao de Spearman de 0,9999
(ver `split.verificar_anterioridade_sem_data`). Usar a data em vez do
identificador teria excluido, ate o fix da concatenacao das fontes (#92), as
120 mil linhas de `NPS_01` que chegavam sem data — justamente as mais antigas,
e por isso as que mais apareceriam como historico das demais. Hoje essas linhas
tem data normal e o bloco sem data esta vazio, mas o identificador segue sendo
a escolha certa: nao depende de a fonte trazer data ou nao.

Ausencia de historico e marcada, nunca preenchida com zero. Cliente sem resposta
anterior e Cliente que respondeu antes e nao detratou sao situacoes diferentes, e
zerar as duas faria o modelo ler ausencia de informacao como ausencia de
detracao. O indicador do `SimpleImputer` no pre-processador preserva essa
distincao.
"""

from __future__ import annotations

import pandas as pd

ALVO = "DETRATOR"
COLUNA_CLIENTE = "ID_GOLDENRECORD"
COLUNA_ORDEM = "RESPONDENT_ID"

FEATURES_HISTORICO = (
    "HIST_RESPOSTAS_ANTERIORES",
    "HIST_DETRATOU_ANTES",
    "HIST_TAXA_DETRACAO_ANTERIOR",
)


def adicionar_historico(
    df: pd.DataFrame,
    coluna_cliente: str = COLUNA_CLIENTE,
    coluna_ordem: str = COLUNA_ORDEM,
    alvo: str = ALVO,
) -> pd.DataFrame:
    """Acrescenta as tres features de historico, sem alterar as linhas.

    Devolve uma copia com as colunas de `FEATURES_HISTORICO`, no indice original:

    - `HIST_RESPOSTAS_ANTERIORES`: quantas respostas o Cliente deu antes desta.
      Zero e informacao legitima, e significa primeira resposta.
    - `HIST_DETRATOU_ANTES`: 1 se detratou em alguma resposta anterior, 0 se
      respondeu antes e nunca detratou, ausente se nao ha resposta anterior.
    - `HIST_TAXA_DETRACAO_ANTERIOR`: proporcao de detracao nas anteriores,
      ausente quando nao ha anteriores.

    As duas ultimas ficam ausentes, e nao zeradas, quando nao ha historico. E a
    diferenca entre "nao sei" e "sei que nao", e o modelo precisa distingui-las.

    Linhas sem Cliente identificado nao recebem historico: sem o identificador
    nao ha como saber se duas delas pertencem a mesma pessoa, e trata-las como
    Clientes distintos inventaria historico zerado para todas.
    """
    for coluna in (coluna_cliente, coluna_ordem, alvo):
        if coluna not in df.columns:
            raise KeyError(f"a construcao do historico exige a coluna {coluna!r}")

    resultado = df.copy()
    # A ordem define o que e passado; sem ordenar, o cumsum somaria respostas em
    # ordem de armazenamento e o "anterior" deixaria de ser anterior.
    ordenado = resultado.sort_values(coluna_ordem, kind="mergesort")
    tem_cliente = ordenado[coluna_cliente].notna()
    base = ordenado[tem_cliente]

    grupo = base.groupby(coluna_cliente, sort=False)
    detrator = base[alvo].astype("float64")

    anteriores = grupo.cumcount()
    # cumsum inclui a linha corrente; subtrai-la deixa apenas as anteriores.
    detracoes_anteriores = grupo[alvo].cumsum().astype("float64") - detrator

    com_historico = anteriores > 0
    detratou_antes = (detracoes_anteriores > 0).astype("float64").where(com_historico)
    taxa = (detracoes_anteriores / anteriores.where(com_historico)).where(com_historico)

    resultado[FEATURES_HISTORICO[0]] = anteriores.reindex(resultado.index).fillna(0).astype("int64")
    resultado[FEATURES_HISTORICO[1]] = detratou_antes.reindex(resultado.index)
    resultado[FEATURES_HISTORICO[2]] = taxa.reindex(resultado.index)
    return resultado


def cobertura_do_historico(df: pd.DataFrame) -> dict[str, object]:
    """Mede quanto da base tem historico, para o texto da Secao 4.3.

    A cobertura e o que delimita a leitura da feature: com historico em uma
    fracao pequena da base, ela e preditor complementar e nunca principal, como
    a propria secao 4.2.4 registra.
    """
    if FEATURES_HISTORICO[0] not in df.columns:
        raise KeyError("chame adicionar_historico antes de medir a cobertura")

    com_historico = df[FEATURES_HISTORICO[0]] > 0
    detratou = df.loc[com_historico, FEATURES_HISTORICO[1]]
    return {
        "linhas": int(len(df)),
        "linhas_com_historico": int(com_historico.sum()),
        "pct_com_historico": round(float(com_historico.mean()) * 100, 2),
        "clientes": int(df[COLUNA_CLIENTE].nunique()),
        "clientes_com_mais_de_uma_resposta": int(
            (df.groupby(COLUNA_CLIENTE).size() > 1).sum()
        ),
        "detratou_antes_pct": (
            round(float(detratou.mean()) * 100, 2) if len(detratou) else None
        ),
    }

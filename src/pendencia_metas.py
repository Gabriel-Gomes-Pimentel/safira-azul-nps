"""Verifica a pendencia de metas simultaneas sobre a tabela comparativa (card #247).

A Secao 4.3.2 mediu que o primeiro candidato nao atinge Sensibilidade >= 0,70 e
Precisao Media >= 0,40 ao mesmo tempo, em nenhum tamanho de fila, e deixou essa
pendencia explicitamente para a Secao 4.4. Este modulo nao recalcula limiar nem
matriz de confusao: essas contas ja existem em `modelo.limiar_por_capacidade`,
`modelo.metricas_no_topo` e `modelo.tabela_comparativa` (#108), reaproveitadas
pelo card #249 (`tabela_final.consolidar_tabela_final`). O que falta e a
pergunta que a Secao 4.3.2 deixou em aberto: **sobre os quatro candidatos
tunados, algum resolve a pendencia?**

`verificar_metas_simultaneas` responde essa pergunta sobre a tabela que #249 ja
produz, no mesmo orcamento de capacidade (a coluna `cobertura` de
`metricas_no_topo` e a Sensibilidade no ponto operacional; `precisao_no_topo` e
a Precisao no mesmo ponto -- ver `modelo.tabela_comparativa`). Nao le `data/`
nem depende de estimador algum: opera sobre a tabela ja calculada, entao e
testavel com uma tabela sintetica, sem os quatro modelos reais.
"""

from __future__ import annotations

import pandas as pd

META_SENSIBILIDADE = 0.70
META_PRECISAO = 0.40

COLUNA_SENSIBILIDADE = "cobertura"
COLUNA_PRECISAO = "precisao_no_topo"
COLUNA_FILA_COMPARAVEL = "fila_comparavel"


def verificar_metas_simultaneas(
    tabela: pd.DataFrame,
    meta_sensibilidade: float = META_SENSIBILIDADE,
    meta_precisao: float = META_PRECISAO,
) -> pd.DataFrame:
    """Adiciona a coluna `atende_as_duas_metas` a uma tabela comparativa.

    Espera uma tabela no formato de `modelo.tabela_comparativa`/
    `tabela_final.consolidar_tabela_final`: uma linha por modelo, com as
    colunas `cobertura` (Sensibilidade no ponto operacional) e
    `precisao_no_topo` (Precisao no mesmo ponto). Quando a tabela trouxer
    `fila_comparavel`, uma linha `False` nessa coluna nunca atende as metas,
    porque descreve uma fila de outro tamanho (score degenerado, seção 4.3.2) e
    nao e comparavel ao orcamento pedido -- ela entra como `False` direto, sem
    reavaliar as duas colunas de metrica.

    Linhas sem estimador ainda (`nan` em `cobertura` ou `precisao_no_topo`, o
    padrao que #249 usa para modelo indisponivel) tambem retornam `False`, nunca
    `True` por comparacao com `nan`.
    """
    faltando = [c for c in (COLUNA_SENSIBILIDADE, COLUNA_PRECISAO) if c not in tabela.columns]
    if faltando:
        raise KeyError(
            f"tabela sem a(s) coluna(s) {faltando}: esperado o formato de "
            "modelo.tabela_comparativa (cobertura, precisao_no_topo)"
        )

    atende = (
        (tabela[COLUNA_SENSIBILIDADE] >= meta_sensibilidade)
        & (tabela[COLUNA_PRECISAO] >= meta_precisao)
    )
    # comparação com NaN já resolve para False em ambos os lados do "&", mas
    # fillna(False) explícito evita herdar um NaN caso a coluna venha com
    # dtype "object" em vez de float (ex.: tabela lida de um CSV/JSON externo).
    atende = atende.fillna(False)

    if COLUNA_FILA_COMPARAVEL in tabela.columns:
        # `.astype("boolean")` antes do `fillna` evita o FutureWarning de
        # downcast do pandas: uma coluna que chega com dtype "object" (comum
        # numa coluna que mistura True/False/NaN vinda de JSON externo) dispara
        # o aviso já dentro do próprio `fillna`, e converter depois não evita.
        fila_comparavel = tabela[COLUNA_FILA_COMPARAVEL].astype("boolean").fillna(False).astype(bool)
        atende = atende & fila_comparavel

    resultado = tabela.copy()
    resultado["atende_as_duas_metas"] = atende
    return resultado


def resumo_da_pendencia(tabela_com_veredito: pd.DataFrame) -> str:
    """Uma frase objetiva sobre se a pendencia da Secao 4.3.2 foi resolvida.

    Existe para o card 18A.3 nao precisar reler a tabela inteira so para narrar
    a conclusao mais importante da secao: se algum modelo atende as duas metas
    ao mesmo tempo, e qual.
    """
    if "atende_as_duas_metas" not in tabela_com_veredito.columns:
        raise KeyError(
            "tabela sem a coluna 'atende_as_duas_metas': rode "
            "verificar_metas_simultaneas antes de pedir o resumo"
        )

    vencedores = tabela_com_veredito.index[tabela_com_veredito["atende_as_duas_metas"]].tolist()
    if vencedores:
        nomes = ", ".join(vencedores)
        return (
            f"A pendência da Seção 4.3.2 foi resolvida: {nomes} atinge(m) "
            f"Sensibilidade >= {META_SENSIBILIDADE:.0%} e Precisão >= "
            f"{META_PRECISAO:.0%} simultaneamente, na mesma capacidade de "
            "contato do primeiro candidato."
        )
    return (
        "A pendência da Seção 4.3.2 permanece: nenhum dos candidatos tunados "
        f"atinge Sensibilidade >= {META_SENSIBILIDADE:.0%} e Precisão >= "
        f"{META_PRECISAO:.0%} ao mesmo tempo, na mesma capacidade de contato. "
        "A decisão de renegociar a meta ou a capacidade com a Azul (card "
        "15B.2) continua em aberto."
    )

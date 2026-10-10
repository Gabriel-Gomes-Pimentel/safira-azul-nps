"""Matrizes de confusao dos quatro candidatos no limiar operacional (card #250).

Nao reimplementa a matriz de confusao: `modelo.matriz_de_confusao` (#106) ja
recebe `(y_verdadeiro, score, limiar)` e devolve a matriz 2x2 rotulada pela acao
da operacao. O que faltava era aplicar essa mesma funcao aos quatro candidatos
do Artefato 7, no **mesmo limiar por modelo que a tabela comparativa do card
18A.1 ja calculou** (#249, `tabela_final.consolidar_tabela_final`) -- e nao um
limiar recalculado aqui, que poderia divergir do da tabela por um arredondamento
ou por outra particao usada sem querer.

Por isso `gerar_matrizes` recebe o limiar de cada modelo pronto (a coluna
`limiar` da tabela do #249), e nao um estimador nem `k`: reconstruir o limiar
aqui duplicaria a conta de `modelo.tabela_comparativa` e arriscaria as duas
contas discordarem sobre o mesmo modelo. Um modelo ainda sem estimador (a
Arvore ou o Gradient Boosting, se a busca de hiperparametro deles nao tiver
terminado) recebe `score=None` e a funcao devolve `None` no lugar da matriz, em
vez de lancar excecao -- o mesmo padrao de "indisponivel" que #249 usa para o
modelo sem vencedor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
_CAMINHO_SRC = str(_RAIZ / "src")
if _CAMINHO_SRC not in sys.path:
    sys.path.insert(0, _CAMINHO_SRC)

from modelo import matriz_de_confusao  # noqa: E402


def gerar_matrizes(
    scores: dict[str, np.ndarray | None],
    y_verdadeiro,
    limiares: dict[str, float | None],
) -> dict[str, pd.DataFrame | None]:
    """Uma matriz de confusão por modelo, no limiar já calculado para ele.

    `scores` e `limiares` precisam ter as mesmas chaves (os nomes dos quatro
    candidatos). Um modelo com `scores[nome] is None` ou `limiares[nome] is
    None` — indisponível em qualquer uma das duas entradas — devolve `None`,
    nunca uma matriz calculada com metade da informação.
    """
    faltando_score = set(limiares) - set(scores)
    faltando_limiar = set(scores) - set(limiares)
    if faltando_score or faltando_limiar:
        raise KeyError(
            "scores e limiares precisam ter as mesmas chaves; "
            f"só em limiares: {faltando_score or None}, só em scores: {faltando_limiar or None}"
        )

    resultado: dict[str, pd.DataFrame | None] = {}
    for nome in scores:
        score = scores[nome]
        limiar = limiares[nome]
        if score is None or limiar is None:
            resultado[nome] = None
            continue
        resultado[nome] = matriz_de_confusao(y_verdadeiro, score, limiar)
    return resultado


def falsos_negativos(matrizes: dict[str, pd.DataFrame | None]) -> dict[str, int | None]:
    """O número de falsos negativos de cada matriz, para anotar no gráfico.

    Isolado de `gerar_matrizes` porque o card #250 exige essa contagem anotada
    em cada matriz — é a categoria mais custosa segundo a Seção 4.3.2/card
    #238 —, e ler direto da célula `["Detrator", "fora da fila"]` em quatro
    lugares diferentes (aqui e no notebook) arrisca trocar linha por coluna.
    """
    saida: dict[str, int | None] = {}
    for nome, matriz in matrizes.items():
        if matriz is None:
            saida[nome] = None
            continue
        saida[nome] = int(matriz.loc["Detrator", "fora da fila"])
    return saida

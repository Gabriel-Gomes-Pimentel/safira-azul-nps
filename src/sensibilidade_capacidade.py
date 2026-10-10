"""Sensibilidade do corte operacional a variacoes de capacidade (card #248).

O card 15B.1/#247 fixa o corte de cada candidato numa capacidade de contato
(50/dia na Secao 4.3.2). Este modulo responde a pergunta seguinte: quanto
Sensibilidade se ganha ou perde por variar essa capacidade -- e nao reimplementa
nenhuma conta, so aplica `modelo.metricas_no_topo` (#106) a uma lista de
capacidades vizinhas em vez de uma so.

Um ganho desproporcional de Sensibilidade por um pequeno aumento de capacidade e
o argumento concreto que sustentaria uma renegociacao da meta com a Azul (a
pendencia da Secao 4.3.2, que os cards #247/18A.3 tratam) -- e e por isso que
`ganho_por_contato_extra` isola esse numero, em vez de deixa-lo implicito numa
tabela de varias linhas.
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

from modelo import metricas_no_topo  # noqa: E402


def sensibilidade_a_capacidade(
    scores: dict[str, np.ndarray | None],
    y_verdadeiro,
    capacidades: list[int],
) -> pd.DataFrame:
    """Uma linha por (modelo, capacidade), com `cobertura` e `precisao_no_topo`.

    `capacidades` é a lista de `k` (número de contatos) a testar — normalmente
    a capacidade adotada e algumas vizinhas (ex.: ±10 contatos/dia × dias do
    período), na mesma unidade que `modelo.limiar_por_capacidade` já usa.

    Um modelo com `scores[nome] is None` (candidato sem estimador tunado
    ainda) entra com `nan` em todas as capacidades, em vez de lançar exceção —
    mesmo padrão de indisponibilidade dos cards #247/#249/#250.
    """
    linhas = []
    for nome, score in scores.items():
        for k in capacidades:
            if score is None:
                linhas.append({
                    "modelo": nome, "capacidade": k,
                    "cobertura": float("nan"), "precisao_no_topo": float("nan"),
                })
                continue
            m = metricas_no_topo(y_verdadeiro, score, k)
            linhas.append({
                "modelo": nome, "capacidade": k,
                "cobertura": m["cobertura"], "precisao_no_topo": m["precisao_no_topo"],
            })
    return pd.DataFrame(linhas).set_index(["modelo", "capacidade"])


def ganho_por_contato_extra(tabela: pd.DataFrame) -> pd.DataFrame:
    """Variação de Sensibilidade por contato adicional, entre capacidades vizinhas.

    Para cada modelo, ordena as capacidades já presentes na tabela e calcula a
    diferença de `cobertura` dividida pela diferença de `capacidade` entre
    pontos consecutivos — a inclinação local da curva de Sensibilidade. Um
    valor alto aqui é o sinal de que um pequeno reforço de capacidade compraria
    Sensibilidade desproporcional, o argumento que #248 existe para tornar
    explícito.

    Exige ao menos duas capacidades por modelo; um modelo com uma só entra
    inteiro como `nan` (nada para comparar).
    """
    linhas = []
    for nome, bloco in tabela.groupby(level="modelo"):
        bloco = bloco.reset_index().sort_values("capacidade")
        capacidades = bloco["capacidade"].to_numpy()
        coberturas = bloco["cobertura"].to_numpy()

        if len(capacidades) < 2:
            linhas.append({"modelo": nome, "de_capacidade": None, "para_capacidade": None,
                            "ganho_de_cobertura": float("nan"), "por_contato": float("nan")})
            continue

        for i in range(len(capacidades) - 1):
            delta_capacidade = capacidades[i + 1] - capacidades[i]
            delta_cobertura = coberturas[i + 1] - coberturas[i]
            por_contato = (
                float(delta_cobertura / delta_capacidade) if delta_capacidade else float("nan")
            )
            linhas.append({
                "modelo": nome,
                "de_capacidade": int(capacidades[i]),
                "para_capacidade": int(capacidades[i + 1]),
                "ganho_de_cobertura": float(delta_cobertura),
                "por_contato": por_contato,
            })
    return pd.DataFrame(linhas).set_index(["modelo", "de_capacidade", "para_capacidade"])

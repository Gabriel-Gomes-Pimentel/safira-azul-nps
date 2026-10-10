"""Ajuste unico da Regressao Logistica sobre o contrato, so para medir (card #206).

Este modulo existe para responder duas perguntas antes de qualquer busca de
hiperparametros comecar: **quanto custa um ajuste** e **o modelo converge**. As
duas respostas alimentam o card #207, que dimensiona a grade de `GridSearchCV`
contra o tempo medido aqui, e sem elas a grade seria escolhida no chute e
estouraria a sessao do Colab no meio da execucao.

O que este modulo deliberadamente **nao** faz:

- nao monta `Pipeline` e nao encadeia pre-processador com estimador: isso e o
  card #208, e antecipa-lo aqui criaria duas definicoes do mesmo objeto;
- nao particiona e nao roda validacao cruzada propria. A matriz que entra aqui
  ja vem particionada e transformada por `matriz.preparar_matriz`, que ajusta o
  pre-processador apenas no treino. Reparticionar seria descartar o congelamento
  de `congelamento.obter_particoes` e medir uma divisao que ninguem acordou;
- nao avalia qualidade. Nenhuma metrica de ordenacao ou de limiar sai daqui: o
  numero que importa neste card e o relogio, e reportar uma metrica obtida sem
  validacao convidaria a ler como desempenho o que e so fumaca.

A escala das features importa para a convergencia e ja esta resolvida no
contrato: `matriz._montar_preprocessador` aplica `RobustScaler` nas numericas,
entao o `lbfgs` recebe colunas em ordem de grandeza comparavel. Se mesmo assim o
ajuste nao convergir, o achado a registrar e o `max_iter` necessario, nao um
escalonamento novo montado por fora do contrato.
"""

from __future__ import annotations

import time
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

# Mesma semente de `congelamento.SEMENTE`. Nao e importada de la porque aquele
# modulo carrega pandas e toca o artefato de indices, e este aqui precisa rodar
# sobre uma matriz que ja esta na memoria.
SEMENTE = 42

# Padrao do scikit-learn. Fica explicito como constante porque o card #206 pede
# que o valor usado seja registrado junto do tempo, e um default implicito nao
# aparece no relatorio.
MAX_ITER_PADRAO = 100

# Escala de tentativas quando o padrao nao converge. Dobra a cada passo para que
# o custo total de procurar fique da ordem do ultimo ajuste, e nao da soma de
# uma varredura fina.
ESCALA_MAX_ITER = (100, 200, 400, 800, 1600)


def medir_ajuste_unico(
    matriz,
    alvo,
    max_iter: int = MAX_ITER_PADRAO,
    solver: str = "lbfgs",
    semente: int = SEMENTE,
) -> dict[str, object]:
    """Ajusta `LogisticRegression` uma vez e devolve tempo, iteracoes e convergencia.

    `matriz` e o bloco ja transformado pelo pre-processador do contrato, isto e,
    `preparo["matrizes"]["treino"]`; `alvo` e `preparo["y"]["treino"]`. Passar a
    matriz crua faria o ajuste medir tambem a codificacao das categoricas, que
    nao pertence a este numero.

    O aviso de convergencia e capturado em vez de silenciado: `convergiu` vira
    `False` e o texto do aviso fica em `aviso`, porque um ajuste que parou no
    limite de iteracoes entrega coeficiente provisorio, e o card #212 le esses
    coeficientes como odds ratio.
    """
    modelo = LogisticRegression(max_iter=max_iter, solver=solver, random_state=semente)

    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always", ConvergenceWarning)
        inicio = time.perf_counter()
        modelo.fit(matriz, alvo)
        segundos = time.perf_counter() - inicio

    avisos_de_convergencia = [
        str(a.message) for a in capturados if issubclass(a.category, ConvergenceWarning)
    ]

    # n_iter_ vem por classe; na classificacao binaria e um vetor de um elemento.
    iteracoes = int(max(modelo.n_iter_))

    return {
        "segundos": segundos,
        "max_iter": max_iter,
        "solver": solver,
        "n_iter": iteracoes,
        "convergiu": not avisos_de_convergencia,
        "aviso": avisos_de_convergencia[0] if avisos_de_convergencia else None,
        "n_linhas": int(matriz.shape[0]),
        "n_colunas": int(matriz.shape[1]),
    }


def menor_max_iter_que_converge(
    matriz,
    alvo,
    escala: tuple[int, ...] = ESCALA_MAX_ITER,
    solver: str = "lbfgs",
    semente: int = SEMENTE,
) -> dict[str, object]:
    """Sobe `max_iter` pela `escala` ate convergir e devolve a tentativa vencedora.

    Serve ao achado que o card #206 pede quando o padrao nao basta: qual valor de
    `max_iter` a grade do card #207 precisa fixar. Devolve tambem a trilha de
    tentativas, porque o tempo do ajuste que converge e o unico que deve entrar na
    conta de custo da busca — os ajustes truncados custam menos e subestimariam
    a grade.

    Se nenhuma tentativa convergir, `convergiu` volta `False` com a ultima
    medicao. Nao levanta excecao: nao convergir em 1600 iteracoes e um resultado
    a registrar no notebook, nao um erro de execucao.
    """
    tentativas: list[dict[str, object]] = []
    for max_iter in escala:
        medicao = medir_ajuste_unico(
            matriz, alvo, max_iter=max_iter, solver=solver, semente=semente
        )
        tentativas.append(medicao)
        if medicao["convergiu"]:
            return {"escolhido": medicao, "tentativas": tentativas, "convergiu": True}

    return {"escolhido": tentativas[-1], "tentativas": tentativas, "convergiu": False}

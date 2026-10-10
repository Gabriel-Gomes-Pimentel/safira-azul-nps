"""Espaco de busca da Regressao Logistica, declarado antes de qualquer execucao (card #207).

Este modulo existe para que a grade do `GridSearchCV` do card #210 seja uma
decisao escrita e justificada, e nao o que sobrou depois de ver resultado. Por
isso ela mora aqui, em codigo versionado, e nao numa celula do notebook: o
notebook mostra a grade e a conta de custo, o `pytest` confere que toda
combinacao dela e valida, e o card #210 importa exatamente o mesmo objeto.

O que a grade **nao** busca, e por que:

- **`solver` nao e eixo de busca.** A verossimilhanca da Regressao Logistica com
  penalidade L1 ou L2 e convexa, entao dois solvers que convergem chegam ao
  mesmo otimo: o que muda entre eles e o tempo, nao o modelo. Buscar sobre o
  solver gastaria folds para escolher entre respostas iguais. Ele fica fixo em
  cada subgrade, escolhido pela penalidade que aquela subgrade usa.
- **`max_iter` nao e eixo de busca.** O card #206 mediu que o ajuste converge em
  534 iteracoes e trunca com o padrao de 100. Buscar sobre esse limite mediria
  quem parou antes contra quem terminou, e o card #212 le esses coeficientes
  como odds ratio.
- **`penalty` nao aparece.** Ele esta depreciado desde o scikit-learn 1.8 e sai
  na 1.10; quem expressa a penalidade agora e `l1_ratio`, com `0.0` para L2 e
  `1.0` para L1. Usar o parametro antigo faria a busca inteira emitir
  `FutureWarning` e quebraria na proxima minor.

A grade e uma **lista de subgrades**, que e como o `GridSearchCV` recebe
espacos em que nem toda combinacao existe. Sem isso, um produto cartesiano unico
de `l1_ratio` por `solver` geraria `lbfgs` com L1, que o scikit-learn recusa com
`ValueError`, e a busca morreria no meio depois de ja ter gasto folds.
"""

from __future__ import annotations

import time
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

# `ParameterGrid` so expande a declaracao, que e a mesma expansao que o
# `GridSearchCV` faz internamente. Contar as combinacoes com ela, em vez de com
# um produto cartesiano escrito a mao, garante que a conta de custo do notebook
# descreve a busca que vai rodar de verdade, inclusive na forma de lista de
# subgrades. Nada aqui particiona a base: as particoes vem congeladas do
# contrato e os folds sao os de `validacao.criar_folds`.
from sklearn.model_selection import ParameterGrid

# O card #206 mediu 534 iteracoes ate convergir e concluiu que 800 bastava, mas
# mediu apenas o ponto de partida da biblioteca: `C=1.0` sem reponderacao. A
# medicao deste card cobriu os extremos da grade e achou um caso pior: `C=10.0`
# com `class_weight="balanced"` precisa de **838 iteracoes**, isto e, trunca em
# 800. O limite sobe para 1600, o degrau seguinte da mesma escala do #206.
#
# Subir o teto nao encarece a busca: `max_iter` e limite, nao trabalho fixo, e o
# `lbfgs` para quando converge. As combinacoes que ja convergiam em 469 ou 508
# iteracoes continuam custando o mesmo; o que muda e que a unica que truncava
# passa a entregar coeficiente convergido, que e o que o card #212 le como odds
# ratio.
MAX_ITER = 1600

# Semente de `congelamento.SEMENTE`, repetida pelo mesmo motivo do
# `fumaca_logistica`: importar de la carregaria pandas e o artefato de indices
# so para ler um inteiro. O `liblinear` embaralha internamente, entao sem semente
# fixa duas execucoes da mesma combinacao dariam coeficientes diferentes.
SEMENTE = 42

# Escala logaritmica de um decimo em um decimo, de regularizacao forte a fraca.
# Quatro pontos bastam nesta base: sao 38 colunas para 341.962 linhas de treino,
# regime em que a verossimilhanca domina a penalidade e o efeito de `C` tende a
# ser pequeno. A grade existe para medir esse efeito, nao para presumi-lo, e
# quatro pontos separados por uma ordem de grandeza mostram se ha efeito sem
# gastar folds na terceira casa decimal.
VALORES_C = [0.01, 0.1, 1.0, 10.0]

# `0.0` e penalidade L2 pura e `1.0` e L1 pura, na notacao que substituiu o
# `penalty`. A L1 entra porque zera coeficiente, e um coeficiente zerado muda a
# leitura do card #212: a feature deixa de ter odds ratio para reportar. Valores
# intermediarios (elastic net) ficam de fora de proposito, porque so o `saga` os
# aceita e ele e o solver mais lento dos tres nesta base, o que triplicaria o
# custo da busca para responder uma pergunta que o par L1/L2 ja responde.
VALORES_L1_RATIO = [0.0, 1.0]

# A prevalencia de Detrator no treino e 0,2043, medida no card #206: quatro nao
# Detratores para cada Detrator. `None` mantem o peso igual e tende a favorecer a
# classe majoritaria; `"balanced"` reponderar na proporcao inversa da frequencia
# e costuma achar mais Detratores ao custo de mais falso positivo. O custo dos
# dois erros nao e o mesmo para a operacao da Azul, entao a escolha entre eles e
# uma decisao do projeto e precisa ser medida, nao herdada do padrao da
# biblioteca.
PESOS_DE_CLASSE = [None, "balanced"]

# Pior tempo medido para cada solver, em segundos, sobre a matriz de treino do
# contrato (341.962 linhas x 38 colunas), em WSL2 local com CPU, scikit-learn
# 1.9.1 e scipy 1.18.1. Sao os valores que `custo_estimado` multiplica.
#
# Dois motivos para eles serem teto e nao media. Primeiro, cada ajuste da busca
# roda sobre quatro quintos do treino, porque o quinto restante e o fold de
# validacao, entao o ajuste real e mais barato do que o medido aqui. Segundo, a
# grade tem combinacoes muito mais baratas do que o pior caso, e o `liblinear`
# inteiro custa menos do que um unico ajuste do `lbfgs`. Uma busca que cabe pelo
# teto cabe de fato.
SEGUNDOS_POR_AJUSTE = {"lbfgs": 76.9, "liblinear": 7.8}

GRADE_LOGISTICA = [
    {
        # L2 com `lbfgs`: e o par medido no card #206, e o unico com tempo de
        # ajuste ja conhecido nesta base.
        "C": VALORES_C,
        "l1_ratio": [0.0],
        "class_weight": PESOS_DE_CLASSE,
        "solver": ["lbfgs"],
        "max_iter": [MAX_ITER],
    },
    {
        # L1 com `liblinear`: dos dois solvers que aceitam L1, `liblinear` e
        # `saga`, o primeiro e o indicado para problema binario e converge por
        # descida coordenada, enquanto o `saga` depende de escala e de muito mais
        # epocas para a mesma solucao.
        "C": VALORES_C,
        "l1_ratio": [1.0],
        "class_weight": PESOS_DE_CLASSE,
        "solver": ["liblinear"],
        "max_iter": [MAX_ITER],
    },
]


def combinacoes(grade: list[dict] | dict = GRADE_LOGISTICA) -> list[dict]:
    """Expande a grade na lista de combinacoes que o `GridSearchCV` vai ajustar.

    A ordem e deterministica, para que a tabela do card #211 saia na mesma
    sequencia em duas execucoes e o diff do notebook mostre so o que mudou.
    """
    return list(ParameterGrid(grade))


def custo_estimado(
    segundos_por_ajuste: dict[str, float],
    n_folds: int,
    grade: list[dict] | dict = GRADE_LOGISTICA,
) -> dict[str, object]:
    """Conta quanto a busca custa: combinacoes x folds x tempo de um ajuste.

    `segundos_por_ajuste` e medido por solver, e nao um numero unico, porque
    `lbfgs` e `liblinear` nao custam o mesmo nesta base: o primeiro faz uma
    fatoracao aproximada do Hessiano a cada passo sobre as 38 colunas, o segundo
    percorre coordenada por coordenada. Usar a media dos dois esconderia qual
    metade da grade domina o custo, que e justamente o que decide se a busca cabe
    na sessao do Colab.

    O `refit` do `GridSearchCV` entra como um ajuste extra, sobre o treino
    inteiro, com o tempo do solver mais caro: e um ajuste que realmente acontece
    e ignora-lo subestimaria a conta.

    Devolve os segundos por solver, o total e o numero de ajustes, para o
    notebook mostrar a conta e nao so o resultado dela.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds precisa ser pelo menos 2, recebido {n_folds}")

    faltando = {c["solver"] for c in combinacoes(grade)} - set(segundos_por_ajuste)
    if faltando:
        raise ValueError(
            f"sem tempo medido para o(s) solver(es) {sorted(faltando)}: "
            "a conta de custo exige medicao, nao estimativa de catalogo"
        )

    por_solver: dict[str, float] = {}
    for combinacao in combinacoes(grade):
        solver = combinacao["solver"]
        por_solver[solver] = por_solver.get(solver, 0.0) + n_folds * segundos_por_ajuste[solver]

    refit = max(segundos_por_ajuste[s] for s in por_solver)
    ajustes = len(combinacoes(grade)) * n_folds + 1

    return {
        "por_solver": por_solver,
        "refit": refit,
        "segundos": sum(por_solver.values()) + refit,
        "ajustes": ajustes,
        "combinacoes": len(combinacoes(grade)),
        "n_folds": n_folds,
    }


def grade_com_prefixo(passo: str, grade: list[dict] | dict = GRADE_LOGISTICA) -> list[dict]:
    """Reescreve as chaves como `passo__parametro`, que e o que um `Pipeline` exige.

    O card #208 monta o `Pipeline` que encadeia o pre-processador do contrato com
    o estimador, e a partir dali o `GridSearchCV` so reconhece nome de parametro
    prefixado pelo passo. Sem esta funcao o card #210 reescreveria a grade a mao
    para prefixar, e passariam a existir duas declaracoes do mesmo espaco de
    busca, que e exatamente o que este card evita.
    """
    if not passo:
        raise ValueError("passo vazio: o prefixo precisa nomear um passo do Pipeline")

    subgrades = grade if isinstance(grade, list) else [grade]
    return [{f"{passo}__{chave}": valores for chave, valores in sub.items()} for sub in subgrades]


def medir_ajuste(
    matriz,
    alvo,
    parametros: dict,
    semente: int = SEMENTE,
) -> dict[str, object]:
    """Ajusta uma combinacao da grade **uma vez** e devolve tempo e convergencia.

    Serve a conta de custo: `custo_estimado` exige tempo medido por solver, e
    quem mede e o notebook, com esta funcao, sobre a matriz do contrato. O card
    #206 mediu apenas o `lbfgs` no ponto de partida da biblioteca, entao o
    `liblinear` e o efeito de `C` e de `class_weight` sobre o tempo continuam
    desconhecidos ate serem medidos aqui.

    Meca o **pior caso** de cada solver medindo os dois extremos de `C` com e sem
    reponderacao, e nao um ponto so. O tempo nesta base nao cresce com `C`: sem
    reponderacao o extremo caro e `C=0.01` (72,2 s contra 42,1 s em `C=10.0`),
    e com `class_weight="balanced"` a ordem se inverte (36,9 s contra 76,9 s).
    Escolher um extremo por intuicao daria uma conta de custo que erra por quase
    o dobro, e sempre e possivel errar para menos, que e o lado que promete uma
    busca que nao cabe na sessao.

    O aviso de convergencia e capturado, como no `fumaca_logistica`: uma
    combinacao que trunca em `max_iter=800` e um achado que muda a grade, nao um
    detalhe de execucao.
    """
    modelo = LogisticRegression(random_state=semente, **parametros)

    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always", ConvergenceWarning)
        inicio = time.perf_counter()
        modelo.fit(matriz, alvo)
        segundos = time.perf_counter() - inicio

    avisos = [str(a.message) for a in capturados if issubclass(a.category, ConvergenceWarning)]

    return {
        "parametros": parametros,
        "segundos": segundos,
        "n_iter": int(max(modelo.n_iter_)),
        "convergiu": not avisos,
        "aviso": avisos[0] if avisos else None,
        "coeficientes_nulos": int((modelo.coef_ == 0).sum()),
    }

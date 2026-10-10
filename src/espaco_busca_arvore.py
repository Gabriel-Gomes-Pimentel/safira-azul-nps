"""Espaco de busca da Arvore de Decisao, declarado antes de qualquer execucao (card #228).

Este modulo existe pelo mesmo motivo que `espaco_busca_logistica.py` (#207): a
grade do `GridSearchCV` do card #231 precisa ser uma decisao escrita e
justificada antes de ver resultado, e nao o que sobrou depois. A grade mora
aqui, em codigo versionado, o `pytest` confere que toda combinacao e valida, e
o card #231 importa exatamente este objeto.

**A Arvore de Decisao e o modelo interpretavel do grupo por construcao, e isso
limita a propria busca.** O card #233 extrai a arvore final como regras
legiveis, e o card #234 traduz essas regras para a operacao da Azul. Uma
arvore profunda continua ajustando bem, mas deixa de caber num numero de
regras que alguem consegue ler e usar — e nesse caso a Arvore perderia a
unica vantagem que a Regressao Logistica (#212, odds ratio) nao tem. Por isso
a profundidade e o tamanho minimo de folha **nao** sao apenas hiperparametros
de generalizacao aqui: sao parte do contrato de explicabilidade do ART.7.

O que a grade **nao** busca, e por que:

- **`max_depth` nao inclui `None` (sem limite).** Uma arvore sem teto de
  profundidade se ajusta ate separar quase toda folha, e o `export_text` do
  card #233 devolveria uma lista de regras longa demais para o card #234
  interpretar para a operacao. O teto de 6 já produz ate 64 folhas, que e o
  bastante para comparar profundidade sem sair do que uma leitura humana
  aguenta.
- **`min_samples_split` nao e eixo separado.** Ele e quase redundante com
  `min_samples_leaf` nesta arvore binaria: forcar uma folha minima ja limita
  quantas vezes um no pode se dividir. Teria o mesmo efeito prático de
  `min_samples_leaf` com o dobro de eixos na grade, sem isolar nada que o
  outro parametro nao isole.
- **`min_samples_leaf` comeca em 50, nao em 1 ou no padrao (1) da
  biblioteca.** A folha e a unidade que o card #234 le como regra com uma
  confianca associada (proporcao de Detrator na folha). Uma folha de poucas
  observacoes descreve ruido de um punhado de Clientes, nao um padrao
  operacional — o mesmo argumento que `modelo.py` usa para o
  `min_samples_leaf=100` do gradient boosting, adaptado ao fato de que aqui a
  folha e a propria unidade de leitura, e nao um detalhe interno do ensemble.

A grade e um dicionario simples, nao uma lista de subgrades como na
Regressao Logistica: a Arvore de Decisao nao tem parametro que so aceita
certos valores em combinacao com outro (o equivalente ao par
`solver`/`l1_ratio` de la), entao o produto cartesiano unico basta.
"""

from __future__ import annotations

import time
import warnings

from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier

# Semente do projeto (mesma de `congelamento.SEMENTE` e de
# `espaco_busca_logistica.SEMENTE`). Fixa aqui, e nao importada, pelo mesmo
# motivo do `fumaca_logistica`: evita carregar pandas e o artefato de indices
# so para ler um inteiro.
SEMENTE = 42

# Profundidade maxima da grade. Acima disso a arvore ainda treina, mas deixa
# de caber no numero de regras que o card #234 consegue traduzir para a
# operacao — ver a justificativa no docstring do modulo.
PROFUNDIDADE_MAXIMA = 6

# Duas medidas de impureza padrao do scikit-learn para arvore de classificacao.
# `log_loss` fica de fora: e equivalente a `entropy` para classificacao binaria
# nesta versao do scikit-learn, e incluir os dois gastaria folds comparando
# respostas praticamente identicas, o mesmo motivo que tira `solver` da busca
# da Regressao Logistica.
VALORES_CRITERION = ["gini", "entropy"]

# Tamanho minimo de folha. O piso de 50 esta justificado no docstring do
# modulo; 100 e 200 testam se folhas ainda maiores, portanto regras ainda mais
# estaveis, custam preditividade — e essa e a troca que o card #232 precisa
# registrar ao escolher o vencedor.
VALORES_MIN_SAMPLES_LEAF = [50, 100, 200]

# Mesma prevalencia de Detrator medida no card #206 do Gabriel (0,2043 no
# treino): quatro nao Detratores para cada Detrator. `None` mantem o peso
# igual; `"balanced"` reponderar pela frequencia inversa. O custo dos dois
# erros nao e o mesmo para a operacao, entao a escolha e medida, nao herdada
# do padrao da biblioteca — mesmo argumento do eixo `class_weight` do #207.
PESOS_DE_CLASSE = [None, "balanced"]

GRADE_ARVORE = {
    "criterion": VALORES_CRITERION,
    "max_depth": list(range(3, PROFUNDIDADE_MAXIMA + 1)),
    "min_samples_leaf": VALORES_MIN_SAMPLES_LEAF,
    "class_weight": PESOS_DE_CLASSE,
}

# Tempo de um ajuste da Arvore sobre a matriz de treino do contrato (341.962
# linhas x 38 colunas), medido com `medir_ajuste()` no pior caso da grade
# (profundidade maxima, min_samples_leaf minimo). Medido sobre a base real no
# card #231, em 24/09/2026.
SEGUNDOS_POR_AJUSTE: float = 3.0979


def combinacoes(grade: dict = GRADE_ARVORE) -> list[dict]:
    """Expande a grade na lista de combinacoes que o `GridSearchCV` vai ajustar.

    A ordem e deterministica, para que a tabela do card #232 saia na mesma
    sequencia em duas execucoes e o diff do notebook mostre so o que mudou.
    """
    return list(ParameterGrid(grade))


def custo_estimado(
    segundos_por_ajuste: float,
    n_folds: int,
    grade: dict = GRADE_ARVORE,
) -> dict[str, object]:
    """Conta quanto a busca custa: combinacoes x folds x tempo de um ajuste.

    Ao contrario de `espaco_busca_logistica.custo_estimado`, aqui so existe um
    tempo de ajuste, porque a Arvore de Decisao nao tem eixo equivalente a
    `solver` que mude o algoritmo de ajuste. `segundos_por_ajuste` e
    obrigatorio e nao tem valor padrao neste modulo — ver `SEGUNDOS_POR_AJUSTE`
    acima. O `refit` entra como um ajuste extra sobre o treino inteiro, mesmo
    criterio do card #210 da Regressao Logistica.
    """
    if segundos_por_ajuste is None:
        raise ValueError(
            "segundos_por_ajuste nao foi medido. Rode medir_ajuste() no Colab, "
            "sobre a matriz de treino real, e passe o resultado aqui — nao "
            "existe estimativa de catalogo para este calculo."
        )
    if n_folds < 2:
        raise ValueError(f"n_folds precisa ser pelo menos 2, recebido {n_folds}")

    n_combinacoes = len(combinacoes(grade))
    segundos_busca = n_combinacoes * n_folds * segundos_por_ajuste
    return {
        "segundos_por_ajuste": segundos_por_ajuste,
        "refit": segundos_por_ajuste,
        "segundos": segundos_busca + segundos_por_ajuste,
        "ajustes": n_combinacoes * n_folds + 1,
        "combinacoes": n_combinacoes,
        "n_folds": n_folds,
    }


def grade_com_prefixo(passo: str, grade: dict = GRADE_ARVORE) -> dict:
    """Reescreve as chaves como `passo__parametro`, que e o que um `Pipeline` exige.

    O card #229 monta o `Pipeline` que encadeia o pre-processador do contrato
    com o `DecisionTreeClassifier`, e a partir dali o `GridSearchCV` do card
    #231 so reconhece nome de parametro prefixado pelo passo.
    """
    if not passo:
        raise ValueError("passo vazio: o prefixo precisa nomear um passo do Pipeline")
    return {f"{passo}__{chave}": valores for chave, valores in grade.items()}


def medir_ajuste(
    matriz,
    alvo,
    parametros: dict,
    semente: int = SEMENTE,
) -> dict[str, object]:
    """Ajusta uma combinacao da grade **uma vez** e devolve o tempo gasto.

    Serve a conta de custo: `custo_estimado` exige `segundos_por_ajuste`
    medido, e quem mede e o notebook, com esta funcao, sobre a matriz real do
    contrato. Diferente da Regressao Logistica, a Arvore de Decisao nao tem
    aviso de convergencia para capturar — ela sempre para, por construcao,
    quando a profundidade ou o tamanho de folha limitam a divisao.
    """
    modelo = DecisionTreeClassifier(random_state=semente, **parametros)

    inicio = time.perf_counter()
    modelo.fit(matriz, alvo)
    segundos = time.perf_counter() - inicio

    return {
        "parametros": parametros,
        "segundos": segundos,
        "profundidade_obtida": int(modelo.get_depth()),
        "n_folhas": int(modelo.get_n_leaves()),
    }

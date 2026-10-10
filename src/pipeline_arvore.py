"""Pipeline da Arvore de Decisao sobre o contrato de dados (card #229).

Mesma razao de existir do `pipeline_logistica.py` (#208): entregar **um unico
objeto** que encadeia o pre-processador do contrato e o estimador, para que o
`GridSearchCV` do card #231 reajuste o pre-processamento dentro de cada fold em
vez de uma vez so sobre o treino inteiro. Ajustar fora do fold vazaria a
mediana da imputacao e os quartis do `RobustScaler` para dentro da validacao, e
a metrica subiria sem que o modelo tivesse melhorado — o card #230 trava isso
por teste, do mesmo jeito que o #209 trava para a Regressao Logistica.

O pre-processador nao e remontado aqui, pelo mesmo motivo do #208:
`criar_pipeline` recebe `preparo["preprocessador"]` de `matriz.preparar_matriz`
e faz `clone` dele, preservando a especificacao (imputacao, `RobustScaler`,
`OneHotEncoder`, mesmas colunas) e descartando o ajuste. Remontar o
`ColumnTransformer` aqui criaria uma segunda definicao do contrato — e como as
duas duplas de modelos interpretaveis comparam os mesmos dados na Secao 4.4,
uma segunda definicao poderia divergir da primeira sem que ninguem notasse.

O que este modulo deliberadamente **nao** faz, pelos mesmos tres motivos do
#208:

- **nao particiona.** A particao temporal por Cliente vem congelada de
  `congelamento` e os folds sao os de `validacao.criar_folds`;
- **nao escolhe hiperparametro.** Os valores ficam no padrao da biblioteca; a
  escolha e a busca do card #231 sobre a grade do card #228;
- **nao define metrica.** `metrica_de_partida` recebe a funcao `avaliar` do
  card #241 e so repassa os vetores.

Diferenca em relacao ao #208: a `LogisticRegression` tem `max_iter` porque o
solver pode truncar antes de convergir, e a grade do #207 fixa esse limite. A
`DecisionTreeClassifier` nao tem esse eixo — ela para por construcao quando
`max_depth` ou `min_samples_leaf` impedem nova divisao —, entao nao ha
parametro equivalente para fixar aqui.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# O modulo localiza os proprios vizinhos em vez de depender de quem o importa,
# mesmo motivo de `pipeline_logistica.py`: sob pytest o conftest prepara o
# caminho, no Colab e na execucao direta nao ha ninguem para preparar.
_RAIZ = Path(__file__).resolve().parents[1]
_CAMINHO_SRC = str(_RAIZ / "src")
if _CAMINHO_SRC not in sys.path:
    sys.path.insert(0, _CAMINHO_SRC)

# `SEMENTE` vem do card #228 em vez de redeclarada aqui, mesmo motivo do #208
# com `espaco_busca_logistica`: duas constantes com o mesmo valor em dois
# lugares sao uma so ate alguem editar apenas uma delas.
from espaco_busca_arvore import SEMENTE  # noqa: E402

# Nomes dos passos. Sao publicos porque o `GridSearchCV` do card #231 enderaca
# hiperparametro por `passo__parametro`, e e este nome que
# `espaco_busca_arvore.grade_com_prefixo` recebe.
PASSO_PREPARO = "preparo"
PASSO_MODELO = "modelo"


def criar_pipeline(
    preprocessador,
    semente: int = SEMENTE,
    **hiperparametros,
) -> Pipeline:
    """Encadeia o pre-processador do contrato e a `DecisionTreeClassifier` num objeto so.

    `preprocessador` e `preparo["preprocessador"]`, o `ColumnTransformer` que
    `matriz.preparar_matriz` montou a partir das colunas da allowlist. Ele entra
    clonado: a especificacao e a mesma, o estado ajustado nao vem junto.

    `**hiperparametros` existe para o card #231 e para os testes construirem uma
    combinacao especifica sem montar o `Pipeline` a mao. Sem eles, o estimador
    nasce no padrao da biblioteca, que e o que este card pede: quem escolhe
    `max_depth`, `min_samples_leaf`, `criterion` e `class_weight` e a busca do
    card #231, nao esta funcao.

    Devolve um pipeline **nao ajustado**. Chamar `fit` nele ajusta os dois
    passos na ordem, e e isso que faz o pre-processamento acontecer dentro do
    fold.
    """
    return Pipeline([
        (PASSO_PREPARO, clone(preprocessador)),
        (PASSO_MODELO, DecisionTreeClassifier(random_state=semente, **hiperparametros)),
    ])


def ajustar_no_treino(pipeline: Pipeline, x_treino, y_treino) -> dict[str, object]:
    """Ajusta o pipeline uma vez sobre a particao de treino e relata o ajuste.

    `x_treino` e `preparo["x"]["treino"]`, a matriz **crua**, com as colunas da
    allowlist ainda por transformar — nao `preparo["matrizes"]["treino"]`, que
    ja passou pelo pre-processador. Passar a matriz transformada ajustaria o
    `ColumnTransformer` do pipeline sobre a saida de outro `ColumnTransformer`,
    erro que nao levanta excecao e so aparece depois, quando o pipeline espera
    colunas que a base nao tem.

    O relato cobre profundidade e numero de folhas obtidos, alem do tamanho da
    matriz: sao os dois numeros que o card #232 compara entre as combinacoes
    da grade, e que o card #233 usa para decidir se a arvore final ainda cabe
    numa leitura de regras.
    """
    inicio = time.perf_counter()
    pipeline.fit(x_treino, y_treino)
    segundos = time.perf_counter() - inicio

    estimador = pipeline.named_steps[PASSO_MODELO]
    preparo = pipeline.named_steps[PASSO_PREPARO]

    return {
        "segundos": segundos,
        "profundidade_obtida": int(estimador.get_depth()),
        "n_folhas": int(estimador.get_n_leaves()),
        "n_linhas": int(x_treino.shape[0]),
        "colunas_de_entrada": int(x_treino.shape[1]),
        "colunas_da_matriz": len(preparo.get_feature_names_out()),
    }


def metrica_de_partida(
    pipeline: Pipeline,
    x,
    y,
    avaliar: Callable,
) -> dict[str, object]:
    """Pontua o pipeline ja ajustado e devolve o que `avaliar` calcular.

    Ponto de partida do card, antes de qualquer busca de hiperparametro — sem
    ele o ganho que o card #231 reportar nao tem contra o que ser lido.

    `avaliar` e a funcao do card #241 (`avaliar(y_true, y_pred, y_proba)`), e
    entra como argumento pelo mesmo motivo do #208: o modulo continua
    importavel enquanto aquele card nao esta em `develop`, e nenhuma metrica e
    definida aqui — quem define o vocabulario da tabela comparativa e o card
    #241, nao esta funcao.

    `y_proba` e a probabilidade da classe positiva (Detrator), a coluna 1 de
    `predict_proba`. As metricas de ordenacao do protocolo (Precisao Media,
    ROC-AUC) sao calculadas sobre o score continuo, entao passar a coluna
    errada devolveria um numero valido e errado.
    """
    if not callable(avaliar):
        raise TypeError(
            "avaliar precisa ser a funcao do card #241, com assinatura "
            "avaliar(y_true, y_pred, y_proba)"
        )

    y_pred = pipeline.predict(x)
    y_proba = pipeline.predict_proba(x)[:, 1]
    return avaliar(y, y_pred, y_proba)

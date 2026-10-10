"""Pipeline da Regressao Logistica sobre o contrato de dados (card #208).

Este modulo entrega **um unico objeto** que encadeia o pre-processador do
contrato e o estimador. O encadeamento e a razao de existir do card: enquanto o
pre-processador vive solto, quem ajusta decide onde ajustar, e a escolha mais
comoda, ajustar uma vez sobre o treino inteiro e so transformar dentro dos
folds, e justamente a que vaza. A mediana da imputacao e os quartis do
`RobustScaler` passariam a carregar as linhas que aquele fold usa como
validacao, e a metrica de validacao subiria sem que nada tivesse melhorado.

Dentro de um `Pipeline`, essa escolha deixa de existir: `GridSearchCV` chama
`fit` no objeto inteiro a cada fold, e o `ColumnTransformer` e reajustado ali,
sobre os quatro quintos daquele fold. Nao e uma disciplina que alguem precisa
lembrar de seguir, e o que o card #209 trava por teste.

O pre-processador nao e remontado aqui. `criar_pipeline` recebe o objeto que
`matriz.preparar_matriz` devolve em `preparo["preprocessador"]` e faz `clone`
dele: mesma especificacao (imputacao, `RobustScaler`, `OneHotEncoder`, mesmas
listas de colunas numericas e categoricas), sem o estado ajustado. Remontar o
`ColumnTransformer` aqui criaria uma segunda definicao do contrato, e duas
definicoes divergem sem ninguem perceber. O `clone` tambem descarta de proposito
o ajuste que `preparar_matriz` ja fez no treino: o pipeline precisa nascer
virgem para que o primeiro `fit` seja o do fold.

O que este modulo deliberadamente **nao** faz:

- **nao particiona.** Nada de `KFold`, `StratifiedKFold` ou `train_test_split`
  aqui: a particao temporal por Cliente vem congelada de `congelamento` e os
  folds sao os de `validacao.criar_folds`. Uma particao propria neste arquivo
  seria uma terceira divisao competindo com as duas que o grupo acordou;
- **nao escolhe hiperparametro.** Os valores ficam no padrao da biblioteca, e a
  escolha e a busca do card #210 sobre a grade do card #207. O unico valor
  fixado e `max_iter`, que nao e eixo de busca e sim o limite de iteracoes que o
  card #207 mediu como suficiente para o pior caso da grade;
- **nao define metrica.** `metrica_de_partida` recebe a funcao `avaliar` do card
  #241 e repassa os vetores; nenhuma metrica e calculada aqui. Escrever uma
  metrica propria neste arquivo criaria a versao divergente que o card 05 existe
  para impedir.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Callable

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# O modulo localiza os proprios vizinhos em vez de depender de quem o importa,
# pelo mesmo motivo de `matriz.py`: sob pytest o conftest prepara o caminho, no
# Colab e na execucao direta nao ha ninguem para preparar.
_RAIZ = Path(__file__).resolve().parents[1]
_CAMINHO_SRC = str(_RAIZ / "src")
if _CAMINHO_SRC not in sys.path:
    sys.path.insert(0, _CAMINHO_SRC)

# `MAX_ITER` e `SEMENTE` vem do card #207 em vez de serem redeclarados aqui.
# Sao o mesmo numero em dois lugares no momento em que alguem edita so um deles,
# e o card #210 monta a busca com a grade de la: se o pipeline nascesse com um
# `max_iter` diferente do que a grade fixa, a combinacao venceria ou truncaria
# dependendo de quem construiu o objeto.
from espaco_busca_logistica import MAX_ITER, SEMENTE  # noqa: E402

# Nomes dos passos. Sao publicos porque o `GridSearchCV` do card #210 enderecada
# hiperparametro por `passo__parametro`, e e este nome que
# `espaco_busca_logistica.grade_com_prefixo` recebe. Trocar a string aqui sem
# trocar la faria a busca varrer um parametro que nao existe, e o
# scikit-learn levanta `ValueError` so no primeiro ajuste.
PASSO_PREPARO = "preparo"
PASSO_MODELO = "modelo"


def criar_pipeline(
    preprocessador,
    max_iter: int = MAX_ITER,
    semente: int = SEMENTE,
    **hiperparametros,
) -> Pipeline:
    """Encadeia o pre-processador do contrato e a `LogisticRegression` num objeto so.

    `preprocessador` e `preparo["preprocessador"]`, o `ColumnTransformer` que
    `matriz.preparar_matriz` montou a partir das colunas da allowlist. Ele entra
    clonado: a especificacao e a mesma, o estado ajustado nao vem junto.

    `max_iter` fica em `espaco_busca_logistica.MAX_ITER` (1600) e nao no padrao
    da biblioteca (100) porque o card #206 mediu que o ajuste desta base trunca
    em 100 iteracoes, e um ajuste truncado entrega coeficiente provisorio, que e
    exatamente o que o card #212 le como odds ratio. Nao e escolha de
    hiperparametro: `max_iter` e limite, nao trabalho fixo, e o solver para
    quando converge.

    `**hiperparametros` existe para o card #210 e para os testes construirem uma
    combinacao especifica sem montar o `Pipeline` a mao. Sem eles, o estimador
    nasce no padrao da biblioteca, que e o que este card pede: quem escolhe `C`,
    `l1_ratio` e `class_weight` e a busca, nao esta funcao.

    Devolve um pipeline **nao ajustado**. Chamar `fit` nele ajusta os dois passos
    na ordem, e e isso que faz o pre-processamento acontecer dentro do fold.
    """
    return Pipeline([
        (PASSO_PREPARO, clone(preprocessador)),
        (
            PASSO_MODELO,
            LogisticRegression(max_iter=max_iter, random_state=semente, **hiperparametros),
        ),
    ])


def ajustar_no_treino(pipeline: Pipeline, x_treino, y_treino) -> dict[str, object]:
    """Ajusta o pipeline uma vez sobre a particao de treino e relata o ajuste.

    `x_treino` e `preparo["x"]["treino"]`, a matriz **crua**, com as colunas da
    allowlist ainda por transformar, e nao `preparo["matrizes"]["treino"]`, que ja
    passou pelo pre-processador. Passar a matriz transformada faria o
    `ColumnTransformer` do pipeline ser ajustado sobre a saida de outro
    `ColumnTransformer`, que e um erro silencioso: o objeto ajusta, o `fit` nao
    reclama, e a partir dali o pipeline espera colunas que a base nao tem.

    O aviso de convergencia e capturado em vez de silenciado, como em
    `fumaca_logistica.medir_ajuste_unico`: `convergiu` vira `False` e o texto fica
    em `aviso`. O relato cobre o que muda a leitura dos cards seguintes: tempo,
    iteracoes, convergencia e a largura da matriz que o pre-processador produziu.
    """
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always", ConvergenceWarning)
        inicio = time.perf_counter()
        pipeline.fit(x_treino, y_treino)
        segundos = time.perf_counter() - inicio

    avisos = [str(a.message) for a in capturados if issubclass(a.category, ConvergenceWarning)]

    estimador = pipeline.named_steps[PASSO_MODELO]
    preparo = pipeline.named_steps[PASSO_PREPARO]

    return {
        "segundos": segundos,
        # n_iter_ vem por classe; na classificacao binaria e um vetor de um elemento.
        "n_iter": int(max(estimador.n_iter_)),
        "max_iter": estimador.max_iter,
        "convergiu": not avisos,
        "aviso": avisos[0] if avisos else None,
        "n_linhas": int(x_treino.shape[0]),
        "colunas_de_entrada": int(x_treino.shape[1]),
        "colunas_da_matriz": len(preparo.get_feature_names_out()),
        "coeficientes_nulos": int((estimador.coef_ == 0).sum()),
    }


def metrica_de_partida(
    pipeline: Pipeline,
    x,
    y,
    avaliar: Callable,
) -> dict[str, object]:
    """Pontua o pipeline ja ajustado e devolve o que `avaliar` calcular.

    Este e o ponto de partida do card: o desempenho do modelo **antes** de
    qualquer busca de hiperparametro. Sem ele, o ganho que o card #210 vai
    reportar nao tem contra o que ser lido, e qualquer numero que a busca
    produzir parece bom.

    `avaliar` e a funcao do card #241 (`avaliar(y_true, y_pred, y_proba)`), e
    entra como argumento de proposito. Duas razoes: o modulo continua importavel
    enquanto aquele card nao esta em `develop`, e nenhuma metrica e definida
    aqui. O dicionario devolvido e o que `avaliar` devolver, sem renomear chave
    nem acrescentar metrica, porque a tabela comparativa do card 18A.1 depende de as
    quatro duplas falarem exatamente a mesma lingua.

    `y_proba` e a probabilidade da classe positiva (Detrator), a coluna 1 de
    `predict_proba`. As metricas de ordenacao do protocolo, Precisao Media e
    ROC-AUC, sao calculadas sobre o score continuo e nao sobre o rotulo, entao
    passar a coluna errada devolveria um numero valido e errado.
    """
    if not callable(avaliar):
        raise TypeError(
            "avaliar precisa ser a funcao do card #241, com assinatura "
            "avaliar(y_true, y_pred, y_proba)"
        )

    y_pred = pipeline.predict(x)
    y_proba = pipeline.predict_proba(x)[:, 1]
    return avaliar(y, y_pred, y_proba)

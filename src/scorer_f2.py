"""Scorer F2 para GridSearchCV e RandomizedSearchCV (card #242).

`GridSearchCV` e `RandomizedSearchCV` escolhem a configuracao que maximiza um
unico escalar, passado em `scoring`. Este modulo entrega esse escalar para as
quatro duplas de modelagem, para que Regressao Logistica, Arvore de Decisao,
Random Forest e Gradient Boosting sejam tunados pelo mesmo criterio. A escolha do
F2 e a sua justificativa estao em `documents/extras/protocolo-avaliacao-artefato7.md`
(card #238): o F2 orienta apenas a busca de hiperparametro, e nao substitui
Sensibilidade, Precisao Media e ROC-AUC da secao 4.3.2 na leitura dos resultados.

**O scorer mede o corte do proprio estimador.** `make_scorer` sobre `fbeta_score`
avalia `predict()`, isto e, o rotulo que o modelo emite com o corte padrao dele,
e nao um limiar operacional. Isso e deliberado: o limiar do projeto e definido
pela capacidade de contato da equipe (secao 4.3.2, card #247), depois que os
modelos estao tunados, e nao pode entrar na busca de hiperparametro.

**`zero_division=0`.** Um fold em que o modelo nao aponta nenhum Detrator deixa a
precisao indefinida. Com o padrao da biblioteca isso emite `UndefinedMetricWarning`
a cada fold; aqui o fold pontua 0, que e o valor correto para quem nao encontrou
nenhum caso, e a busca segue sem ruido.

Uso, com os folds por Cliente de `validacao.criar_folds`:

    GridSearchCV(pipeline, grade, scoring=scorer_f2, cv=folds)
"""

from __future__ import annotations

from sklearn.metrics import fbeta_score, make_scorer

# Exportado para que `avaliar()` use o mesmo beta: duas definicoes de F2 no
# projeto poderiam divergir sem ninguem perceber, e a busca otimizaria uma
# metrica diferente da que a tabela comparativa reporta.
BETA = 2

# A classe positiva e Detrator, codificada como 1 em `DETRATOR` (secao 4.1.3).
CLASSE_POSITIVA = 1

scorer_f2 = make_scorer(
    fbeta_score,
    beta=BETA,
    pos_label=CLASSE_POSITIVA,
    zero_division=0,
)

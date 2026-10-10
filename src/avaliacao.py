"""Metricas unicas do protocolo de avaliacao (card 02A.3), para as quatro duplas de modelagem (card #241).

O card 02A.3 registrou o protocolo de avaliacao e o card 02A.1 definiu o F2 como
criterio de busca de hiperparametros. Sem uma funcao unica, cada dupla (Regressao
Logistica, Arvore, Random Forest, Gradient Boosting) calcularia essas metricas a
propria maneira, e a tabela comparativa do card 18A.1 compararia numeros que nao
foram produzidos pela mesma conta. `avaliar` existe para que as quatro leiam a
mesma formula.

Por isso a funcao nao recebe modelo nem pipeline, so os tres vetores que
qualquer estimador binario produz (`y_true`, `y_pred`, `y_proba`): acoplar a um
estimador especifico impediria a reutilizacao entre as quatro duplas, que e o
proposito deste card.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, fbeta_score, recall_score, roc_auc_score

# Beta=2 pondera o recall 4x mais que a precisao na media harmonica (peso beta^2
# no denominador), porque a Secao 4.1.3 classificou o falso negativo -- o
# Detrator que passa despercebido -- como o erro mais caro do protocolo. O valor
# vem do card 02A.1, que o adotou como criterio de busca de hiperparametros; esta
# funcao nao o redefine, so o reaplica onde o protocolo pede a mesma conta.
BETA_F2 = 2

NOMES_METRICAS = ("F2", "Sensibilidade", "Precisão Média", "ROC-AUC")


def avaliar(y_true, y_pred, y_proba) -> dict[str, float]:
    """Calcula num dicionario unico as quatro metricas do protocolo (card 02A.3).

    - **F2**: media harmonica ponderada de precisao e recall, com peso maior
      para o recall. Para beta=2:

          F2 = (1 + beta^2) * precisao * recall / (beta^2 * precisao + recall)
             = 5 * precisao * recall / (4 * precisao + recall)

      Beta=2 foi o criterio de busca fixado pelo card 02A.1.
    - **Sensibilidade** (Recall), **Precisão Média** (Average Precision) e
      **ROC-AUC**: as tres metricas de negocio da Secao 4.3.2, com esses nomes
      exatos, para que a tabela do card 18A.1 fale a mesma lingua entre modelos.

    `y_pred` e o rotulo binarizado no limiar do candidato (usado por F2 e
    Sensibilidade); `y_proba` e a probabilidade continua da classe positiva
    (usada por Precisão Média e ROC-AUC, que variam o limiar internamente).

    Quando o lote avaliado nao tem nenhum exemplo da classe positiva (ex.: um
    fold sem nenhum Detrator), as quatro metricas dependem de uma classe que nao
    esta presente e nao tem valor definido: nao ha coluna do positivo pra
    ordenar (Precisão Média, ROC-AUC) nem positivo real pra recuperar (Recall,
    F2). O `scikit-learn` 1.9.1 nao lanca excecao nesse caso: devolve `0.0` para
    F2, Sensibilidade e Precisão Média e `nan` so para o ROC-AUC, com avisos
    genericos de biblioteca. Essa mistura seria lida na tabela comparativa como
    um fold ruim, e nao como um fold sem o que medir. A funcao intercepta o caso
    antes do `scikit-learn`, devolve `nan` nas quatro metricas e registra um
    unico aviso, para que o candidato atue sobre o restante da busca sem que um
    fold degenerado pese como desempenho zero.
    """
    y_true = np.asarray(y_true)

    if not np.any(y_true == 1):
        warnings.warn(
            "avaliar(): nenhum exemplo da classe positiva no lote avaliado "
            f"({len(y_true)} linha(s)); F2, Sensibilidade, Precisão Média e "
            "ROC-AUC retornam nan.",
            UserWarning,
            stacklevel=2,
        )
        return {nome: float("nan") for nome in NOMES_METRICAS}

    return {
        "F2": float(fbeta_score(y_true, y_pred, beta=BETA_F2, zero_division=0)),
        "Sensibilidade": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precisão Média": float(average_precision_score(y_true, y_proba)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
    }

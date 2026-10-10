"""Diagnostico e correcao de calibracao dos quatro candidatos (cards #245/#246).

O threshold operacional (card 15B.1/#247) so tem sentido em termos de negocio
se a probabilidade que ele corta significar risco de fato -- e nao apenas
ordenar bem os casos. `metricas_de_ordenacao` (`modelo.py`, #105) ja calcula o
Brier de um modelo isolado; este modulo aplica a mesma conta aos quatro
candidatos de uma vez (`diagnosticar_calibracao`, #245) e decide, so entao,
quais precisam de correcao (`calibrar_se_necessario`, #246).

Nao reimplementa nada do scikit-learn: usa `sklearn.calibration.
CalibratedClassifierCV` para corrigir e `sklearn.metrics.brier_score_loss`
(a mesma metrica de `modelo.metricas_de_ordenacao`) para medir antes e
depois. Nao le `data/`: testavel com um classificador sintetico
deliberadamente mal calibrado (`class_weight` extremo), sem os quatro
estimadores reais.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

_RAIZ = Path(__file__).resolve().parents[1]
_CAMINHO_SRC = str(_RAIZ / "src")
if _CAMINHO_SRC not in sys.path:
    sys.path.insert(0, _CAMINHO_SRC)

METODOS_VALIDOS = ("sigmoid", "isotonic")


def diagnosticar_calibracao(
    scores: dict[str, np.ndarray | None],
    y_verdadeiro,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Brier score e curva de calibração de cada candidato, numa linha por modelo.

    Não decide sozinho quem precisa de correção: devolve o número (Brier) e os
    pontos da curva (`fracao_positivos_por_bin`, `media_predita_por_bin`) para
    quem lê a Seção 4.4 decidir com evidência, o mesmo padrão de "registrar o
    número, não a regra automática" que `busca_arvore_decisao.md` (#232) já
    usa para a escolha de profundidade.

    Um modelo sem estimador ainda (`scores[nome] is None`, o candidato que a
    busca de hiperparâmetro dele não terminou) entra com `nan` em vez de
    lançar exceção — mesmo padrão de indisponibilidade de `tabela_final`
    (#249) e `matrizes_confusao_candidatos` (#250).
    """
    linhas = []
    for nome, score in scores.items():
        if score is None:
            linhas.append({
                "modelo": nome,
                "brier": float("nan"),
                "fracao_positivos_por_bin": None,
                "media_predita_por_bin": None,
            })
            continue

        valores = np.asarray(score)
        if valores.min() < 0.0 or valores.max() > 1.0:
            raise ValueError(
                f"score de '{nome}' fora de [0, 1] (min {valores.min()}, max "
                f"{valores.max()}): precisa ser predict_proba, não predict()."
            )

        fracao_positivos, media_predita = calibration_curve(
            y_verdadeiro, valores, n_bins=n_bins, strategy="quantile"
        )
        linhas.append({
            "modelo": nome,
            "brier": float(brier_score_loss(y_verdadeiro, valores)),
            "fracao_positivos_por_bin": fracao_positivos,
            "media_predita_por_bin": media_predita,
        })
    return pd.DataFrame(linhas).set_index("modelo")


def calibrar_se_necessario(
    estimador_ajustado,
    x_treino,
    y_treino,
    calibrar: bool,
    metodo: str = "sigmoid",
) -> object:
    """Aplica `CalibratedClassifierCV` só quando `calibrar=True`; senão devolve o mesmo objeto.

    `calibrar` vem de uma decisão já tomada sobre o diagnóstico de
    `diagnosticar_calibracao` (card #245/#246) — esta função não reavalia o
    Brier, só executa a decisão. Isso evita a função decidir por conta própria
    um limiar de "mal calibrado o bastante" que ninguém escreveu e justificou.

    `estimador_ajustado` precisa estar ajustado: a calibração é sobre o modelo
    já tunado do card correspondente (#211/#189/#190/#232), não um novo ajuste
    com hiperparâmetro diferente. Por isso ele entra envolto em
    `FrozenEstimator` — o substituto de `cv="prefit"`, removido no
    scikit-learn 1.9.1 (o pinado em `requirements.txt`; `cv="prefit"` ainda
    roda com aviso de depreciação na 1.6, usada neste ambiente de revisão, mas
    quebra direto na versão real do projeto). `FrozenEstimator` impede
    qualquer novo `fit` do modelo base: só o mapa de calibração é ajustado
    sobre `x_treino`/`y_treino`, nunca o estimador em si — o mesmo tipo de
    vazamento que os cards #209/#230 já travam para o pré-processador, agora
    para o próprio modelo.

    O que o `FrozenEstimator` **não** resolve: se `x_treino` for a mesma
    partição em que o estimador foi ajustado, o mapa sigmoide é estimado sobre
    scores in-sample, mais separados do que os de dado novo, e o ganho de Brier
    medido depois fica abaixo do que uma fatia disjunta daria. A trava do
    modelo e a exigência do mapa são cuidados diferentes, e esta função só
    garante a primeira: quem chama responde por `x_treino` ser disjunto do
    ajuste, ou por registrar que não é. Numa fatia separada, ela precisa ser
    agrupada por Cliente, como o corte do card #209 exige, senão a disjunção é
    aparente.
    """
    if not calibrar:
        return estimador_ajustado
    if metodo not in METODOS_VALIDOS:
        raise ValueError(f"metodo precisa ser um de {METODOS_VALIDOS}, recebido {metodo!r}")

    calibrado = CalibratedClassifierCV(FrozenEstimator(estimador_ajustado), method=metodo)
    calibrado.fit(x_treino, y_treino)
    return calibrado


def brier_antes_e_depois(
    y_verdadeiro,
    score_antes: np.ndarray,
    score_depois: np.ndarray,
) -> dict[str, float]:
    """Compara o Brier antes/depois da calibração, para o card #246 registrar o ganho.

    Existe porque o CR do card #246 exige que a calibração só seja aceita se
    o Brier melhorar — sem essa comparação explícita, uma calibração que piora
    o modelo entraria na tabela comparativa sem ninguém perceber.
    """
    antes = float(brier_score_loss(y_verdadeiro, score_antes))
    depois = float(brier_score_loss(y_verdadeiro, score_depois))
    return {"brier_antes": antes, "brier_depois": depois, "melhorou": depois < antes}

"""Reproduz localmente cinco candidatos e um baseline, sem versionar outputs.

Execute da raiz com `.venv/bin/python scripts/comparar_seis_modelos.py`.
O parquet deve existir em data/processed/base_analitica.parquet; a saída agregada
fica em out/comparacao_seis_modelos.json, diretório ignorado pelo Git.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "documents" / "extras" / "resultados"
sys.path.insert(0, str(RAIZ / "src"))

from avaliacao import avaliar  # noqa: E402
from ensembles import criar_pipeline_random_forest, melhor_gradient_boosting  # noqa: E402
from matriz import preparar_matriz  # noqa: E402
from modelo import limiar_por_capacidade  # noqa: E402
from pipeline_arvore import criar_pipeline as criar_arvore  # noqa: E402
from pipeline_logistica import criar_pipeline as criar_logistica  # noqa: E402
from scorer_f2 import scorer_f2  # noqa: E402
from validacao import criar_folds  # noqa: E402

PARQUET = RAIZ / "data/processed/base_analitica.parquet"
SAIDA = RAIZ / "out/comparacao_seis_modelos.json"
CAPACIDADE = 9050
CORTE_VALIDACAO = "2025-07-01"
CORTE_TESTE = "2026-01-01"


def selecionar_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Marca exatamente k maiores scores; empates preservam a ordem de entrada.

    O desempate não contém sinal preditivo. Para o baseline constante, a fila
    top-k é uma amostra arbitrária da ordem original e não mede sua qualidade.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("scores deve ser um vetor finito")
    if not 0 <= k <= len(scores):
        raise ValueError("k deve estar entre zero e o tamanho do lote")
    escolhidos = np.argsort(-scores, kind="stable")[:k]
    rotulos = np.zeros(len(scores), dtype=int)
    rotulos[escolhidos] = 1
    return rotulos


def executar(parquet: Path = PARQUET, saida: Path = SAIDA) -> dict:
    """Ajusta no treino, mede validação/teste e calcula CV dos dois não tunados.

    O F2 de CV usa predict() nos folds agrupados; o F2 operacional usa o
    limiar da fila. O teste nunca participa da escolha de hiperparâmetros.
    """
    preparo = preparar_matriz(pd.read_parquet(parquet), CORTE_VALIDACAO, CORTE_TESTE)
    prep = preparo["preprocessador"]
    log = json.loads((RESULTADOS / "hiperparametros_logistica.json").read_text())[0]
    arvore = json.loads((RAIZ / "src/hiperparametros_arvore.json").read_text())
    boosting = json.loads((RESULTADOS / "hiperparametros_gradient_boosting.json").read_text())

    extra = ExtraTreesClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=1,
    )
    extra_pipeline = Pipeline([("preparo", clone(prep)), ("modelo", extra)])
    # (estimador, representação de entrada, F2 já registrado, calcular CV aqui)
    modelos = {
        "classe majoritária (baseline)": (DummyClassifier(strategy="most_frequent"), "matrizes", None, False),
        "regressão logística tunada": (
            criar_logistica(prep, max_iter=log["max_iter"], C=log["C"],
                            class_weight=log["class_weight"], solver=log["solver"],
                            l1_ratio=log["l1_ratio"]), "x", log["f2"], False,
        ),
        "árvore de decisão tunada": (
            criar_arvore(prep, **arvore["hiperparametros"]), "x",
            arvore["f2_validacao_cruzada"], False,
        ),
        "gradient boosting tunado": (
            melhor_gradient_boosting(prep), "x", boosting["melhor_score_medio"], False,
        ),
        "random forest padrão (sem busca)": (
            criar_pipeline_random_forest(prep), "x", None, True,
        ),
        "extra trees inicial (sem busca)": (extra_pipeline, "x", None, True),
    }

    resultado = {
        "split": {p: len(preparo["y"][p]) for p in ("treino", "validacao", "teste")},
        "capacidade": CAPACIDADE,
        "modelos": {},
    }
    # A preparação de cada Pipeline é clonada, para que os folds ajustem
    # imputer/scaler/encoder apenas nas linhas de treino de cada fold.
    folds = criar_folds(preparo["x"]["treino"], preparo["grupos"]["treino"])
    for nome, (estimador, matriz, f2_cv, calcular_cv) in modelos.items():
        inicio = time.monotonic()
        if calcular_cv:
            f2_cv = float(np.mean(cross_val_score(
                estimador, preparo["x"]["treino"], preparo["y"]["treino"],
                scoring=scorer_f2, cv=folds, n_jobs=4,
            )))

        estimador.fit(preparo[matriz]["treino"], preparo["y"]["treino"])
        registro = {"f2_cv_predict": f2_cv, "particoes": {}}
        for particao in ("validacao", "teste"):
            x = preparo[matriz][particao]
            y = preparo["y"][particao]
            scores = estimador.predict_proba(x)[:, 1]
            limiar = limiar_por_capacidade(scores, CAPACIDADE)
            rotulos = (scores >= limiar).astype(int)
            top_k = selecionar_top_k(scores, min(CAPACIDADE, len(scores)))
            registro["particoes"][particao] = {
                "metricas_limiar_capacidade": avaliar(y, rotulos, scores),
                "metricas_top_k_exato": avaliar(y, top_k, scores),
                "f2_predict_padrao": avaliar(y, estimador.predict(x), scores)["F2"],
                "limiar": float(limiar),
                "fila": int(rotulos.sum()),
                "fila_top_k": int(top_k.sum()),
            }
        registro["tempo_total_s"] = round(time.monotonic() - inicio, 1)
        resultado["modelos"][nome] = registro
        # Checkpoint agregado local, inclusive se uma execução longa for interrompida.
        saida.parent.mkdir(parents=True, exist_ok=True)
        saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n")
    # A recomendação é feita sobre a validação temporal. Não usar o teste
    # para ordenar modelos, escolher parâmetros ou decidir o vencedor.
    candidatos = {nome: reg for nome, reg in resultado["modelos"].items()
                  if "baseline" not in nome}
    resultado["ranking_validacao_ap"] = sorted(
        candidatos,
        key=lambda nome: candidatos[nome]["particoes"]["validacao"]
        ["metricas_top_k_exato"]["Precisão Média"],
        reverse=True,
    )
    saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n")
    return resultado


if __name__ == "__main__":
    executar()

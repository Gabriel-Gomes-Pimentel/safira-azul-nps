"""Contrato executável do treinamento do Extra Trees no notebook integrado."""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "comparacao_modelos.ipynb"


def _codigo_de_treino() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celulas = [
        "".join(celula["source"])
        for celula in notebook["cells"]
        if celula["cell_type"] == "code" and "modelo_extra_trees.fit(" in "".join(celula["source"])
    ]
    assert len(celulas) == 1
    return celulas[0]


def test_ajuste_recebe_somente_treino_e_registra_estimador():
    x_treino, y_treino = object(), object()
    x_validacao, y_validacao, x_teste = object(), object(), object()
    chamadas = []

    class EstimadorEspiao:
        def __init__(self, **parametros):
            self.parametros = parametros

        def fit(self, x, y):
            chamadas.append((x, y))
            return self

    contexto = {
        "ExtraTreesClassifier": EstimadorEspiao,
        "SEMENTE": 42,
        "X_treino": x_treino,
        "y_treino": y_treino,
        "X_validacao": x_validacao,
        "y_validacao": y_validacao,
        "X_teste": x_teste,
        "modelos": {},
    }
    exec(_codigo_de_treino(), contexto)

    assert len(chamadas) == 1
    assert chamadas[0][0] is x_treino
    assert chamadas[0][1] is y_treino
    assert contexto["modelos"]["Extra Trees"] is contexto["modelo_extra_trees"]
    assert contexto["modelo_extra_trees"].parametros == {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": 42,
        # Uma thread, como `scripts/comparar_seis_modelos.py`: `predict_proba` acumula
        # os votos na ordem em que as threads terminam e move o ultimo bit (Secao 5.4).
        "n_jobs": 1,
    }


def test_estimador_real_ajusta_amostra_sintetica():
    x_treino = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [2, 0], [0, 2]])
    y_treino = np.array([0, 1, 0, 1, 1, 0])
    contexto = {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "SEMENTE": 42,
        "X_treino": x_treino,
        "y_treino": y_treino,
        "modelos": {},
    }
    exec(_codigo_de_treino(), contexto)

    modelo = contexto["modelos"]["Extra Trees"]
    assert modelo.n_features_in_ == x_treino.shape[1]
    assert modelo.classes_.tolist() == [0, 1]

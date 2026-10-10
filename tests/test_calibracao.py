"""Testes de calibracao.py (cards #245/#246).

Usa um classificador sintetico e deliberadamente mal calibrado (probabilidade
sempre proxima de 0 ou 1, por um `RandomForestClassifier` raso sobre poucos
dados) para provar que `CalibratedClassifierCV` de fato melhora o Brier, sem
depender de nenhum dos quatro estimadores reais do Artefato 7. Nada aqui le
`data/`.

Executar com:  pytest tests/test_calibracao.py -v
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

from calibracao import (
    brier_antes_e_depois,
    calibrar_se_necessario,
    diagnosticar_calibracao,
)


@pytest.fixture
def dados_treino_calibracao():
    """Split em dois blocos: um para ajustar o estimador base, outro para calibrar."""
    aleatorio = np.random.RandomState(0)
    x = aleatorio.randn(400, 3)
    y = (x[:, 0] + 0.3 * aleatorio.randn(400) > 0).astype(int)
    return x[:200], y[:200], x[200:], y[200:]


@pytest.fixture
def estimador_mal_calibrado(dados_treino_calibracao):
    """Árvores rasas e poucas: tendem a emitir 0.0/1.0 puro, o padrão clássico
    de score mal calibrado que uma curva de calibração revela."""
    x_ajuste, y_ajuste, _, _ = dados_treino_calibracao
    modelo = RandomForestClassifier(n_estimators=3, max_depth=1, random_state=0)
    modelo.fit(x_ajuste, y_ajuste)
    return modelo


def test_diagnostico_reporta_brier_por_modelo(estimador_mal_calibrado, dados_treino_calibracao):
    _, _, x_calib, y_calib = dados_treino_calibracao
    score = estimador_mal_calibrado.predict_proba(x_calib)[:, 1]

    diagnostico = diagnosticar_calibracao({"modelo_x": score}, y_calib)

    assert "brier" in diagnostico.columns
    assert diagnostico.loc["modelo_x", "brier"] == pytest.approx(
        brier_score_loss(y_calib, score)
    )


def test_diagnostico_devolve_nan_para_modelo_indisponivel(dados_treino_calibracao):
    _, _, _, y_calib = dados_treino_calibracao
    diagnostico = diagnosticar_calibracao({"gradient_boosting": None}, y_calib)
    assert np.isnan(diagnostico.loc["gradient_boosting", "brier"])
    assert diagnostico.loc["gradient_boosting", "fracao_positivos_por_bin"] is None


def test_diagnostico_recusa_score_que_parece_predict(dados_treino_calibracao):
    _, _, _, y_calib = dados_treino_calibracao
    rotulo_binario = (y_calib > 0).astype(float)  # só 0.0/1.0: não é probabilidade real
    with pytest.raises(ValueError, match="predict_proba"):
        diagnosticar_calibracao({"modelo_x": rotulo_binario * 2}, y_calib)  # força >1 também


def test_calibrar_se_necessario_false_devolve_o_mesmo_objeto(estimador_mal_calibrado, dados_treino_calibracao):
    _, _, x_calib, y_calib = dados_treino_calibracao
    resultado = calibrar_se_necessario(estimador_mal_calibrado, x_calib, y_calib, calibrar=False)
    assert resultado is estimador_mal_calibrado


def test_calibrar_se_necessario_true_melhora_o_brier(estimador_mal_calibrado, dados_treino_calibracao):
    _, _, x_calib, y_calib = dados_treino_calibracao
    score_antes = estimador_mal_calibrado.predict_proba(x_calib)[:, 1]

    calibrado = calibrar_se_necessario(
        estimador_mal_calibrado, x_calib, y_calib, calibrar=True, metodo="sigmoid"
    )
    score_depois = calibrado.predict_proba(x_calib)[:, 1]

    comparacao = brier_antes_e_depois(y_calib, score_antes, score_depois)
    assert comparacao["melhorou"], (
        f"Brier não melhorou: antes={comparacao['brier_antes']:.4f}, "
        f"depois={comparacao['brier_depois']:.4f}"
    )


def test_calibrar_se_necessario_rejeita_metodo_invalido(estimador_mal_calibrado, dados_treino_calibracao):
    _, _, x_calib, y_calib = dados_treino_calibracao
    with pytest.raises(ValueError, match="metodo"):
        calibrar_se_necessario(estimador_mal_calibrado, x_calib, y_calib, calibrar=True, metodo="linear")


def test_brier_antes_e_depois_marca_piora_corretamente():
    y = np.array([1, 0, 1, 0])
    score_bom = np.array([0.9, 0.1, 0.8, 0.2])
    score_ruim = np.array([0.5, 0.5, 0.5, 0.5])
    comparacao = brier_antes_e_depois(y, score_bom, score_ruim)
    assert comparacao["melhorou"] is False

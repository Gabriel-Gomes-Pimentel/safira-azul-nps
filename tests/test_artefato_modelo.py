import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import artefato_modelo as art
import gerar_artefato_modelo as cli
from dados_sinteticos_operacionais import criar_base_sintetica
from matriz import preparar_matriz


@pytest.fixture(scope="module")
def ajustado():
    base = criar_base_sintetica()
    pipeline, proveniencia = art.treinar_modelo(base)
    x = preparar_matriz(base, "2025-07-01", "2026-01-01")["x"]["treino"].iloc[:20]
    return pipeline, proveniencia, x


def salvar(tmp_path, ajustado):
    pipeline, proveniencia, _ = ajustado
    modelo, manifesto = tmp_path / "modelo.joblib", tmp_path / "manifesto.json"
    registro = art.salvar_artefato(pipeline, proveniencia, modelo=modelo, manifesto=manifesto,
                                  versao_modelo="sintetico-v1", versao_fontes="artificial",
                                  fingerprint_base="hash-artificial", sintetico=True)
    return modelo, manifesto, registro


def test_round_trip_e_proveniencia(tmp_path, ajustado):
    modelo, manifesto, registro = salvar(tmp_path, ajustado)
    recuperado, _ = art.carregar_artefato(modelo, manifesto)
    with threadpool_limits(limits=1):
        np.testing.assert_array_equal(recuperado.predict_proba(ajustado[2]), ajustado[0].predict_proba(ajustado[2]))
    assert registro["sha256"] == art.sha256(modelo)
    assert registro["features"] == list(art.COLUNAS_MODELO)
    assert registro["calibracao"] == {"metodo": "sigmoid", "particao": "treino", "in_sample": True}
    assert registro["aprovado_producao"] is False
    assert registro["dados_sinteticos"] is True
    assert registro["periodo_treino"]["fim"].startswith("2024")


def test_determinismo_e_independencia_do_teste(ajustado):
    base = criar_base_sintetica()
    base.loc[base.DATA_STD >= "2025-07-01", "DETRATOR"] = 1 - base.loc[base.DATA_STD >= "2025-07-01", "DETRATOR"]
    segundo, _ = art.treinar_modelo(base)
    with threadpool_limits(limits=1):
        np.testing.assert_array_equal(segundo.predict_proba(ajustado[2]), ajustado[0].predict_proba(ajustado[2]))


@pytest.mark.parametrize("campo,valor", [("sha256", "adulterado"), ("features", []),
    ("versoes", {}), ("versao_contrato", "outra"), ("classes", [1, 0])])
def test_recusa_manifesto_antes_de_carregar(tmp_path, ajustado, monkeypatch, campo, valor):
    modelo, manifesto, registro = salvar(tmp_path, ajustado)
    registro[campo] = valor
    manifesto.write_text(json.dumps(registro))
    def proibido(*a, **k):
        pytest.fail("Nao deveria desserializar")
    monkeypatch.setattr(art.joblib, "load", proibido)
    with pytest.raises(art.ErroArtefato):
        art.carregar_artefato(modelo, manifesto)


def test_recusa_binario_adulterado(tmp_path, ajustado):
    modelo, manifesto, _ = salvar(tmp_path, ajustado)
    modelo.write_bytes(modelo.read_bytes() + b"alterado")
    with pytest.raises(art.ErroArtefato):
        art.carregar_artefato(modelo, manifesto)


def test_nao_sobrescreve_versao(tmp_path, ajustado):
    salvar(tmp_path, ajustado)
    with pytest.raises(art.ErroArtefato):
        salvar(tmp_path, ajustado)


def test_configuracao_ausente(tmp_path):
    with pytest.raises(art.ErroArtefato, match="Configuracao"):
        art.treinar_modelo(criar_base_sintetica(), parametros=tmp_path / "ausente.json")


def test_destino_git_protegido():
    art._proteger_destino(art.RAIZ / "artifacts/modelo.joblib")
    with pytest.raises(art.ErroArtefato):
        art._proteger_destino(art.RAIZ / "manifesto-publicavel.json")


def test_construtor_notebook_canonico():
    registro = json.loads(art.PARAMETROS.read_text(encoding="utf-8"))
    art.conferir_parametros_notebook(registro["hiperparametros"], registro["random_state"])


@pytest.mark.parametrize("campo,valor", [("learning_rate", .05), ("max_iter", 201)])
def test_divergencia_json_bloqueia_antes_do_preparo(tmp_path, monkeypatch, campo, valor):
    registro = json.loads(art.PARAMETROS.read_text(encoding="utf-8"))
    registro["hiperparametros"][campo] = valor
    caminho = tmp_path / "parametros.json"
    caminho.write_text(json.dumps(registro), encoding="utf-8")
    def proibido(*args, **kwargs):
        pytest.fail("Nao deve preparar ou ajustar um modelo divergente")
    monkeypatch.setattr(art, "preparar_matriz", proibido)
    with pytest.raises(art.ErroArtefato, match="divergente do JSON"):
        art.treinar_modelo(None, parametros=caminho)


@pytest.mark.parametrize("codigo", [
    "modelo_gradient_boosting = HistGradientBoostingClassifier(random_state=7)",
    "modelo_gradient_boosting = HistGradientBoostingClassifier(**parametros)",
    "modelo_gradient_boosting = outro_construtor()",
    "pass",
])
def test_construtor_notebook_invalido(tmp_path, codigo):
    caminho = tmp_path / "notebook.ipynb"
    caminho.write_text(json.dumps({"cells": [{"cell_type": "code", "source": [codigo]}]}),
                       encoding="utf-8")
    registro = json.loads(art.PARAMETROS.read_text(encoding="utf-8"))
    with pytest.raises(art.ErroArtefato, match="Construtor do notebook"):
        art.conferir_parametros_notebook(registro["hiperparametros"], 42, notebook=caminho)


@pytest.mark.parametrize("erro", [
    art.ErroArtefato("Destinos devem ser distintos e nao podem sobrescrever uma versao."),
    art.ErroArtefato("Destino de output deve estar protegido pelo gitignore."),
    art.ErroArtefato("Features de treino divergem do contrato operacional."),
    ValueError("VALOR_CONFIDENCIAL"),
])
def test_script_preserva_erros_seguros(monkeypatch, capsys, tmp_path, erro):
    monkeypatch.setattr(cli.sys, "argv", ["gerar_artefato_modelo.py", "--sintetico",
        "--modelo", str(tmp_path / "modelo.joblib"), "--manifesto", str(tmp_path / "manifesto.json"),
        "--versao-modelo", "artificial-v1", "--versao-fontes", "artificial"])
    def falhar(*args, **kwargs):
        raise erro
    monkeypatch.setattr(cli, "treinar_modelo", falhar)
    assert cli.main() == 3
    mensagem = capsys.readouterr().err
    assert "Geracao interrompida:" in mensagem
    if isinstance(erro, art.ErroArtefato):
        assert str(erro) in mensagem
    assert "VALOR_CONFIDENCIAL" not in mensagem

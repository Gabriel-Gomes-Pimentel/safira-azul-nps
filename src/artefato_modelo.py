"""Geracao e carga confiavel do modelo avaliado, sem uso do teste no fit."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from matriz import preparar_matriz
from features import FEATURES_HISTORICO
from preprocessamento_nps import FEATURE_SET_V1

RAIZ = Path(__file__).resolve().parents[1]
PARAMETROS = RAIZ / "documents/extras/resultados/hiperparametros_gradient_boosting.json"
NOTEBOOK = RAIZ / "notebooks/comparacao_modelos.ipynb"
COLUNAS_MODELO = tuple(FEATURE_SET_V1) + FEATURES_HISTORICO
VERSAO_CONTRATO = "safira-batch-v1"
VERSAO_FILA = "contato-unico-topk-v1"
PACOTES = ("scikit-learn", "pandas", "numpy", "scipy", "joblib", "threadpoolctl")


class ErroArtefato(ValueError):
    """Artefato/configuracao invalido; nao incluir dados na mensagem."""


def sha256(caminho):
    digest = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def versoes_ambiente():
    return {"python": platform.python_version(), **{
        nome: importlib.metadata.version(nome) for nome in PACOTES}}


def _json_seguro(valor):
    if isinstance(valor, dict):
        return {str(k): _json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json_seguro(v) for v in valor]
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


def conferir_parametros_notebook(hp, semente, *, notebook=NOTEBOOK):
    """Confere o construtor versionado por AST, sem executar celulas."""
    try:
        registro = json.loads(Path(notebook).read_text(encoding="utf-8"))
        atribuicoes = {}
        for celula in registro["cells"]:
            if celula["cell_type"] != "code":
                continue
            for no in ast.parse("".join(celula["source"])).body:
                if isinstance(no, ast.Assign):
                    for alvo in no.targets:
                        if isinstance(alvo, ast.Name) and alvo.id in (
                            "SEMENTE", "modelo_gradient_boosting",
                        ):
                            if alvo.id in atribuicoes:
                                raise ValueError
                            atribuicoes[alvo.id] = no.value
        chamada = atribuicoes["modelo_gradient_boosting"]
        if (not isinstance(chamada, ast.Call) or chamada.args
                or not isinstance(chamada.func, ast.Name)
                or chamada.func.id != "HistGradientBoostingClassifier"):
            raise ValueError
        encontrados = {}
        for argumento in chamada.keywords:
            if argumento.arg is None or argumento.arg in encontrados:
                raise ValueError
            valor = argumento.value
            if isinstance(valor, ast.Name) and valor.id == "SEMENTE":
                valor = atribuicoes["SEMENTE"]
            encontrados[argumento.arg] = ast.literal_eval(valor)
        if encontrados != {**hp, "random_state": semente, "early_stopping": False}:
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError, SyntaxError):
        raise ErroArtefato("Construtor do notebook ausente, invalido ou divergente do JSON.") from None


def treinar_modelo(base, *, parametros=PARAMETROS):
    """Reproduz preparo/ajuste existentes; retorna pipeline fitted e proveniencia."""
    try:
        registro = json.loads(Path(parametros).read_text(encoding="utf-8"))
        hp = registro["hiperparametros"]
        esperados = {"class_weight", "l2_regularization", "learning_rate", "max_iter",
                     "max_leaf_nodes", "min_samples_leaf"}
        if set(hp) != esperados or registro["random_state"] != 42:
            raise ValueError
        if (registro["corte_validacao"], registro["corte_teste"]) != ("2025-07-01", "2026-01-01"):
            raise ValueError
        estimador = HistGradientBoostingClassifier(**hp, random_state=42, early_stopping=False)
    except (OSError, ValueError, KeyError, TypeError):
        raise ErroArtefato("Configuracao canonica ausente ou invalida.") from None
    conferir_parametros_notebook(hp, registro["random_state"])
    preparo = preparar_matriz(base, corte_validacao="2025-07-01", corte_teste="2026-01-01")
    if tuple(preparo["x"]["treino"].columns) != COLUNAS_MODELO:
        raise ErroArtefato("Features de treino divergem do contrato operacional.")
    y = preparo["y"]["treino"]
    if set(y.unique()) != {0, 1}:
        raise ErroArtefato("Treino deve conter as duas classes.")
    with threadpool_limits(limits=1):
        estimador.fit(preparo["matrizes"]["treino"], y)
        calibrado = CalibratedClassifierCV(FrozenEstimator(estimador), method="sigmoid")
        calibrado.fit(preparo["matrizes"]["treino"], y)
    pipeline = Pipeline([("preparo", preparo["preprocessador"]), ("modelo", calibrado)])
    datas = preparo["particoes"]["treino"]["DATA_STD"].dropna()
    proveniencia = {
        "parametros": estimador.get_params(), "semente": 42,
        "sha256_configuracao": sha256(parametros),
        "cortes": {"validacao": "2025-07-01", "teste": "2026-01-01"},
        "split": "temporal-por-cliente; sem_data=treino; cliente-no-periodo-mais-recente",
        "periodo_treino": {"inicio": str(datas.min()), "fim": str(datas.max())},
        "calibracao": {"metodo": "sigmoid", "particao": "treino", "in_sample": True},
        "avaliacao_referencia": "comparacao_modelos.ipynb, secoes 6.1 e 9; nao reexecutada na geracao",
        "limitacoes": ["historico de treino por ordem de resposta, nao as-of comprovado",
                       "calibracao in-sample", "politica da fila V1 distinta da avaliacao historica"],
    }
    return pipeline, proveniencia


def _proteger_destino(caminho):
    caminho = Path(caminho).resolve()
    if caminho.is_relative_to(RAIZ):
        check = subprocess.run(["git", "check-ignore", "--no-index", "-q", str(caminho)],
                               cwd=RAIZ, capture_output=True)
        if check.returncode != 0:
            raise ErroArtefato("Destino de output deve estar protegido pelo gitignore.")


def salvar_artefato(pipeline, proveniencia, *, modelo, manifesto, versao_modelo,
                    versao_fontes, fingerprint_base, sintetico=False):
    """Nao sobrescreve versoes; confirma manifestos/outputs privados antes de gerar."""
    modelo, manifesto = Path(modelo), Path(manifesto)
    for valor in (versao_modelo, versao_fontes, fingerprint_base):
        if not isinstance(valor, str) or not valor.strip():
            raise ErroArtefato("Versao e proveniencia da base sao obrigatorias.")
    if modelo.resolve() == manifesto.resolve() or modelo.exists() or manifesto.exists():
        raise ErroArtefato("Destinos devem ser distintos e nao podem sobrescrever uma versao.")
    for destino in (modelo, manifesto):
        _proteger_destino(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "nao-disponivel"
    temporarios = []
    publicado = False
    try:
        for destino in (modelo, manifesto):
            fd, nome = tempfile.mkstemp(dir=destino.parent, prefix=".safira-", suffix=".tmp")
            os.close(fd)
            temporarios.append(Path(nome))
        joblib.dump(pipeline, temporarios[0], compress=3)
        registro = {
            "versao_modelo": versao_modelo, "versao_contrato": VERSAO_CONTRATO,
            "versao_politica_fila": VERSAO_FILA, "commit_fonte": commit,
            "sha256": sha256(temporarios[0]), "versoes": versoes_ambiente(),
            "features": list(COLUNAS_MODELO),
            "tipos": ["texto"] * 5 + ["numero", "numero", "booleano", "numero", "numero", "inteiro",
                      "inteiro", "numero", "numero"],
            "features_transformadas": list(pipeline["preparo"].get_feature_names_out()),
            "classes": pipeline.classes_.tolist(), "classe_positiva": 1,
            "versao_fontes": versao_fontes, "fingerprint_base": fingerprint_base,
            "gerado_em": datetime.now(timezone.utc).isoformat(),
            "dados_sinteticos": bool(sintetico), "aprovado_producao": False,
            **proveniencia,
        }
        temporarios[1].write_text(json.dumps(_json_seguro(registro), indent=2, ensure_ascii=True,
                                            allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporarios[0], modelo)
        publicado = True
        os.replace(temporarios[1], manifesto)
        return registro
    except Exception:
        if publicado:
            modelo.unlink(missing_ok=True)
        raise
    finally:
        for temporario in temporarios:
            temporario.unlink(missing_ok=True)


def carregar_artefato(modelo, manifesto):
    """Hash e ambiente antes de joblib.load; origem confiavel continua obrigatoria."""
    try:
        registro = json.loads(Path(manifesto).read_text(encoding="utf-8"))
        if (registro["features"] != list(COLUNAS_MODELO)
                or registro["versao_contrato"] != VERSAO_CONTRATO
                or registro["versao_politica_fila"] != VERSAO_FILA
                or registro["classes"] != [0, 1] or registro["classe_positiva"] != 1
                or registro["versoes"] != versoes_ambiente()
                or registro["sha256"] != sha256(modelo)
                or not isinstance(registro["versao_modelo"], str) or not registro["versao_modelo"].strip()
                or type(registro["dados_sinteticos"]) is not bool):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise ErroArtefato("Manifesto, hash, features ou ambiente incompativeis.") from None
    try:
        pipeline = joblib.load(modelo)
        if (not isinstance(pipeline, Pipeline) or tuple(pipeline.feature_names_in_) != COLUNAS_MODELO
                or pipeline.classes_.tolist() != [0, 1]
                or not isinstance(pipeline["modelo"], CalibratedClassifierCV)
                or not isinstance(pipeline["modelo"].estimator, FrozenEstimator)
                or not isinstance(pipeline["modelo"].estimator.estimator, HistGradientBoostingClassifier)):
            raise ValueError
    except Exception:
        raise ErroArtefato("Binario nao contem o pipeline calibrado esperado.") from None
    return pipeline, registro

"""Execucao do notebook integrado para os testes (compartilhada entre os arquivos de teste).

O notebook e executado de verdade, com `nbclient`, num kernel novo a cada chamada.
Uma celula acrescentada so na copia em memoria exporta para JSON as variaveis que
o proprio notebook calculou; o arquivo `.ipynb` nao e alterado e nenhuma metrica
e recalculada fora dele.
"""
import json
import os
from pathlib import Path

import pytest

from gerar_dummy import BASE_ANALITICA_DUMMY, gerar

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks" / "comparacao_modelos.ipynb"
BASE_SINTETICA = RAIZ / "data" / "dummy" / BASE_ANALITICA_DUMMY
BASE_REAL = RAIZ / "data" / "processed" / "base_analitica.parquet"
PASTA_SAIDA = RAIZ / "out"

# Acrescentada so na copia em memoria. Le as variaveis do notebook; nao recalcula metrica.
CELULA_EXPORTACAO = '''
import json as _json
import numpy as _np
from pathlib import Path as _Path

_exportado = {
    "base_sintetica": bool(USAR_BASE_DUMMY),
    "caminho_base": str(caminho_base),
    "semente": SEMENTE,
    "random_state": {nome: modelo.get_params().get("random_state") for nome, modelo in modelos.items()},
    "estrategia_piso": modelos[NOME_PISO].get_params().get("strategy"),
    "linhas": {p: int(len(preparo["y"][p])) for p in ("treino", "validacao", "teste")},
    "prevalencia": {p: float(preparo["y"][p].mean()) for p in ("treino", "validacao", "teste")},
    "metrica_principal": METRICA_PRINCIPAL,
    "metricas": {coluna: list(definicao) for coluna, definicao in METRICAS.items()},
    "nome_piso": NOME_PISO,
    "ordem": list(resultados.index),
    "resultados": resultados.to_dict(orient="index"),
    "calibracao": calibracao.to_dict(orient="index"),
    "brier_melhorou": bool(comparacao_brier["melhorou"]),
    "score_medio": {"sem_calibracao": float(score_sem_calibracao.mean()),
                    "calibrado": float(score_calibrado.mean())},
    "ordem_preservada": bool(_np.array_equal(_np.argsort(score_sem_calibracao, kind="stable"),
                                             _np.argsort(score_calibrado, kind="stable"))),
}
_Path(_DESTINO).write_text(_json.dumps(_exportado, ensure_ascii=False), encoding="utf-8")
'''


def garantir_base_sintetica() -> None:
    """Gera a base sintetica no caminho que o notebook procura, se ela ainda nao existir."""
    if not BASE_SINTETICA.exists():
        gerar(BASE_SINTETICA.parent)


def exigir_base_real() -> None:
    """Pula o teste quando a execucao na base real nao foi pedida ou a base nao existe."""
    if os.environ.get("NOTEBOOK_BASE_REAL") != "1":
        pytest.skip("a execução na base real é opt-in: defina NOTEBOOK_BASE_REAL=1")
    if not BASE_REAL.exists():
        pytest.skip(f"base real ausente: {BASE_REAL}")


def executar_notebook(base_sintetica: bool, destino: Path) -> dict:
    """Executa o notebook inteiro num kernel novo e devolve o que a celula final exportou."""
    import nbclient
    import nbformat

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell(f"_DESTINO = {str(destino)!r}\n" + CELULA_EXPORTACAO))
    cliente = nbclient.NotebookClient(
        notebook, timeout=1800, kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    cliente.execute(env={**os.environ, "USAR_BASE_DUMMY": "1" if base_sintetica else "0"})
    exportado = json.loads(destino.read_text(encoding="utf-8"))
    # O kernel roda em notebooks/; nos relatorios, o caminho da base fica relativo a raiz.
    caminho = (NOTEBOOK.parent / exportado["caminho_base"]).resolve()
    if caminho.is_relative_to(RAIZ):
        exportado["caminho_base"] = caminho.relative_to(RAIZ).as_posix()
    return exportado

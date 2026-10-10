"""Notebook de demonstracao executado com artefato e dados artificiais."""

import ast
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

from artefato_modelo import salvar_artefato, treinar_modelo
from dados_sinteticos_operacionais import criar_base_sintetica


RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks/pipeline_modelo_final.ipynb"


def test_notebook_sem_outputs_ou_fit():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for celula in notebook.cells:
        if celula.cell_type != "code":
            continue
        assert celula.execution_count is None and not celula.outputs
        for no in ast.walk(ast.parse(celula.source)):
            if isinstance(no, ast.Call) and isinstance(no.func, ast.Attribute):
                assert no.func.attr not in {"fit", "fit_transform", "fit_predict"}


def test_pipeline_joblib_notebook(tmp_path):
    modelo, proveniencia = treinar_modelo(criar_base_sintetica())
    binario = tmp_path / "modelo.joblib"
    manifesto = tmp_path / "manifesto.json"
    salvar_artefato(modelo, proveniencia, modelo=binario, manifesto=manifesto,
                    versao_modelo="teste-notebook-v1", versao_fontes="artificial",
                    fingerprint_base="artificial", sintetico=True)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell("""
assert MODO_VIDEO and USAR_BASE_SINTETICA
assert len(y_teste) == 180
assert len(score_teste) == len(y_teste)
assert len(fatores) == 14
assert len(fila) == 120 and len(contatos) == 50
for metrica in ('Precisão Média', 'ROC-AUC', 'Brier'):
    assert 0 <= metricas[metrica] <= 1
entrada_dois_dias = entrada_demo.copy()
entrada_dois_dias.loc[:59, 'T_EVENTO_ELEGIBILIDADE'] = '2026-10-05T10:00:00Z'
dois_dias, _ = pontuar_fila(entrada_dois_dias, capacidade=5, t_score=T_SCORE, fuso=FUSO)
assert dois_dias.groupby('DIA_OPERACIONAL')['PRIORIZADO'].sum().tolist() == [5, 5]
vazia, _ = pontuar_fila(entrada_demo.iloc[:0], capacidade=50, t_score=T_SCORE, fuso=FUSO)
assert vazia.empty and 'PROBABILIDADE_DETRACAO' in vazia
"""))
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv[0] = sys.executable
    cliente = NotebookClient(notebook, km=km, timeout=180,
                             resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    cliente.execute(env={**os.environ, "SAFIRA_MODELO": str(binario),
                         "SAFIRA_MANIFESTO": str(manifesto), "SAFIRA_MODO_VIDEO": "1",
                         "SAFIRA_BASE_SINTETICA": "1", "MPLBACKEND": "module://matplotlib_inline.backend_inline"})
    assert any(output.output_type in {"display_data", "execute_result"}
               and "image/png" in output.get("data", {})
               for cell in notebook.cells if cell.cell_type == "code"
               for output in cell.outputs)

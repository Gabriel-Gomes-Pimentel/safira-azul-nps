"""Testes das curvas do modelo candidato (#107), secao 4.3.

O que estes testes protegem nao e a aparencia da figura, que se julga olhando:
e o que faria a figura afirmar algo falso sem quebrar nada.

1. Ponto de operacao fora do lugar. Se a marcacao nao cair sobre a curva, a
   figura mostra um desempenho que o limiar escolhido nao entrega, e ninguem
   percebe olhando.
2. Ponto diferente entre as duas figuras. As duas recebem o limiar e derivam as
   coordenadas, e nao o contrario, justamente para nao poderem divergir.
3. Piso errado na precisao contra cobertura. A linha de referencia e a
   prevalencia da particao; qualquer outro valor infla ou desinfla o ganho.

Todos usam dados sinteticos, sem tocar nas bases da Azul.

Executar com:  pytest tests/test_graficos_curvas.py -v
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graficos as gr  # noqa: E402


@pytest.fixture
def dados():
    """Alvo desbalanceado e score com poder preditivo, como a particao de teste."""
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.2, 4000)
    score = np.clip(rng.beta(2, 6, 4000) + 0.25 * y, 0, 1)
    limiar = float(np.sort(score)[::-1][800 - 1])
    return y, score, limiar


def test_ponto_de_operacao_bate_com_a_contagem_manual(dados):
    y, score, limiar = dados
    p = gr._ponto_de_operacao(y, score, limiar)

    selecionado = score >= limiar
    vp = int((selecionado & (y == 1)).sum())
    fp = int((selecionado & (y == 0)).sum())

    assert p["n_selecionados"] == int(selecionado.sum())
    assert p["precisao"] == pytest.approx(vp / (vp + fp))
    assert p["cobertura"] == pytest.approx(vp / int((y == 1).sum()))
    assert p["tfp"] == pytest.approx(fp / int((y == 0).sum()))


def test_as_duas_figuras_marcam_o_mesmo_ponto(dados):
    """A cobertura do marcador tem que ser identica nas duas, ou elas se contradizem."""
    y, score, limiar = dados
    roc = gr.g10_curva_roc(y, score, limiar)
    pr = gr.g11_precisao_cobertura(y, score, limiar)

    cobertura_roc = roc.axes[0].collections[0].get_offsets()[0][1]
    cobertura_pr = pr.axes[0].collections[0].get_offsets()[0][0]
    assert cobertura_roc == pytest.approx(cobertura_pr)


def test_marcador_cai_sobre_a_curva_de_precisao_cobertura(dados):
    y, score, limiar = dados
    fig = gr.g11_precisao_cobertura(y, score, limiar)
    ax = fig.axes[0]

    x_marcador, y_marcador = ax.collections[0].get_offsets()[0]
    curva = next(l for l in ax.get_lines() if l.get_linestyle() == "-")
    cobertura, precisao = curva.get_xdata(), curva.get_ydata()

    i = int(np.argmin(np.abs(cobertura - x_marcador)))
    assert precisao[i] == pytest.approx(y_marcador, abs=0.01)


def test_piso_da_precisao_e_a_prevalencia(dados):
    y, score, limiar = dados
    ax = gr.g11_precisao_cobertura(y, score, limiar).axes[0]

    tracejada = next(l for l in ax.get_lines() if l.get_linestyle() == "--")
    assert tracejada.get_ydata()[0] == pytest.approx(y.mean())


def test_roc_traz_a_diagonal_do_aleatorio(dados):
    y, score, limiar = dados
    ax = gr.g10_curva_roc(y, score, limiar).axes[0]

    tracejada = next(l for l in ax.get_lines() if l.get_linestyle() == "--")
    assert list(tracejada.get_xdata()) == [0, 1]
    assert list(tracejada.get_ydata()) == [0, 1]


def test_as_duas_curvas_se_distinguem_sem_cor(dados):
    """CR01 pede figura legivel impressa: estilo de linha, e nao so cor."""
    y, score, limiar = dados
    for fig in (gr.g10_curva_roc(y, score, limiar),
                gr.g11_precisao_cobertura(y, score, limiar)):
        estilos = {l.get_linestyle() for l in fig.axes[0].get_lines()}
        assert {"-", "--"} <= estilos


def test_eixos_rotulados_e_linha_de_fonte(dados):
    """CR01: eixo rotulado e linha de fonte, no padrao das figuras da secao 4.2."""
    y, score, limiar = dados
    for fig in (gr.g10_curva_roc(y, score, limiar),
                gr.g11_precisao_cobertura(y, score, limiar)):
        ax = fig.axes[0]
        assert ax.get_xlabel() and ax.get_ylabel()
        assert any("Fonte: Autoria própria." in t.get_text() for t in ax.texts)


def test_limiar_opcional_nao_marca_ponto(dados):
    """Sem limiar a figura sai sem marcador, em vez de inventar um ponto."""
    y, score, _ = dados
    assert len(gr.g10_curva_roc(y, score).axes[0].collections) == 0
    assert len(gr.g11_precisao_cobertura(y, score).axes[0].collections) == 0


def test_score_de_rotulo_binario_nao_passa_por_curva_continua(dados):
    """Um score 0/1 so tem um ponto util; a curva viraria duas retas sem aviso."""
    y, _, _ = dados
    binario = y.astype(float)
    ax = gr.g11_precisao_cobertura(y, binario).axes[0]
    curva = next(l for l in ax.get_lines() if l.get_linestyle() == "-")
    assert len(np.unique(curva.get_xdata())) <= 3

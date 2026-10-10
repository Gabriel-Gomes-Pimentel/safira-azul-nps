"""SAFIRA | Secao 3 - Figura do ciclo CRISP-DM.

Gera ``assets/ciclo_crisp_dm.png``, a figura de autoria propria que acompanha a
descricao das seis fases do CRISP-DM na Secao 3. O desenho e produzido por
codigo, e nao colado de terceiro, para que a autoria seja verificavel e para que
a figura possa ser regerada se a nomenclatura das fases mudar no documento.

A paleta e a familia tipografica sao as mesmas de ``src/graficos.py``, para que
esta figura nao destoe das demais do documento.

Uso:
    python scripts/figura_crispdm.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Paleta institucional, identica a de src/graficos.py.
AZ_ESC, AZ_CLA, CINZA = "#0A2A6B", "#00A0DF", "#9AA5B1"
TEXTO_CLARO = "#D6E4F7"

SAIDA = Path("assets/ciclo_crisp_dm.png")

# Geometria do anel. As seis fases ocupam os vertices de um hexagono inscrito
# numa elipse: a elipse e mais larga que alta porque as caixas sao horizontais.
RX, RY = 4.6, 3.0
LARG, ALT = 3.0, 1.15

# Ordem canonica do processo, em sentido horario a partir do topo esquerdo.
# O angulo posiciona a caixa na elipse; o par (pt, en) fixa a nomenclatura.
FASES = [
    (120, "1. Entendimento\ndo Negócio", "Business Understanding"),
    (60, "2. Entendimento\ndos Dados", "Data Understanding"),
    (0, "3. Preparação\ndos Dados", "Data Preparation"),
    (-60, "4. Modelagem", "Modeling"),
    (-120, "5. Avaliação", "Evaluation"),
    (180, "6. Implantação", "Deployment"),
]

# Pares em que o CRISP-DM preve ida e volta dentro do proprio ciclo.
BIDIRECIONAIS = {(0, 1), (2, 3)}


def _centro(angulo):
    """Centro da caixa da fase, em coordenadas de dados."""
    from math import cos, radians, sin
    return RX * cos(radians(angulo)), RY * sin(radians(angulo))


def _borda(centro, destino, folga=0.18):
    """Ponto em que o segmento centro->destino cruza a borda da caixa.

    Sem isso a seta nasceria no centro da caixa e passaria por cima do texto.
    """
    cx, cy = centro
    dx, dy = destino[0] - cx, destino[1] - cy
    if dx == 0 and dy == 0:
        return centro
    meia_l, meia_a = LARG / 2 + folga, ALT / 2 + folga
    escalas = []
    if dx:
        escalas.append(meia_l / abs(dx))
    if dy:
        escalas.append(meia_a / abs(dy))
    t = min(escalas)
    return cx + dx * t, cy + dy * t


def _caixa(ax, centro, titulo, original):
    cx, cy = centro
    ax.add_patch(FancyBboxPatch(
        (cx - LARG / 2, cy - ALT / 2), LARG, ALT,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=AZ_ESC, edgecolor=AZ_CLA, linewidth=1.6, zorder=3,
    ))
    ax.text(cx, cy + 0.12, titulo, ha="center", va="center", zorder=4,
            color="white", fontsize=12.5, fontweight="bold", linespacing=1.25)
    ax.text(cx, cy - 0.36, original, ha="center", va="center", zorder=4,
            color=TEXTO_CLARO, fontsize=8.5, style="italic")


def _seta(ax, origem, destino, dupla=False, tracejada=False, curva=0.12):
    estilo = "<|-|>" if dupla else "-|>"
    ax.add_patch(FancyArrowPatch(
        origem, destino,
        arrowstyle=estilo, mutation_scale=17,
        connectionstyle=f"arc3,rad={curva}",
        linewidth=1.8 if not tracejada else 1.4,
        linestyle="--" if tracejada else "-",
        color=CINZA if tracejada else AZ_ESC, zorder=2,
        shrinkA=0, shrinkB=0,
    ))


def figura_ciclo():
    """Devolve a Figure do ciclo CRISP-DM."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=170)

    centros = [_centro(ang) for ang, _, _ in FASES]

    # Anel principal: cada fase aponta para a seguinte, e a sexta fecha o ciclo
    # de volta na primeira, que e o que faz do CRISP-DM um processo ciclico.
    for i in range(len(FASES)):
        j = (i + 1) % len(FASES)
        origem = _borda(centros[i], centros[j])
        destino = _borda(centros[j], centros[i])
        _seta(ax, origem, destino, dupla=(i, j) in BIDIRECIONAIS)

    # Retorno da Avaliacao ao Entendimento do Negocio: quando a avaliacao mostra
    # que o objetivo de negocio nao foi atendido, o ciclo volta ao inicio em vez
    # de seguir para a implantacao.
    origem = _borda(centros[4], centros[0], folga=0.05)
    destino = _borda(centros[0], centros[4], folga=0.05)
    # A curvatura afasta a tracejada da reta que liga as duas fases: rente a ela
    # a tracejada se confundiria com a seta solida que vai da Implantacao ao
    # Entendimento. O valor e limitado pelos dois lados: a barriga da curva
    # cresce para a esquerda, na direcao da caixa da Implantacao, e o rotulo
    # ocupa a faixa a direita. Em -0,42 a curva passava por tras da caixa, que
    # tem zorder maior, e o retorno aparecia interrompido; -0,25 deixa folga de
    # cerca de 0,3 em unidades de dados para a caixa e para o rotulo.
    _seta(ax, origem, destino, tracejada=True, curva=-0.25)
    ax.text(-1.55, 0.0, "revisão do\nobjetivo de negócio",
            ha="center", va="center", fontsize=9, style="italic",
            color=CINZA, linespacing=1.35, zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="none"))

    for centro, (_, titulo, original) in zip(centros, FASES):
        _caixa(ax, centro, titulo, original)

    ax.text(1.35, 0.62, "CRISP-DM", ha="center", va="center",
            fontsize=16, fontweight="bold", color=AZ_ESC, zorder=5)
    ax.text(1.35, -0.28, "processo cíclico:\na sexta fase\nrealimenta a primeira",
            ha="center", va="center", fontsize=9, color=CINZA,
            linespacing=1.35, zorder=5)

    ax.set_xlim(-RX - LARG / 2 - 0.4, RX + LARG / 2 + 0.4)
    ax.set_ylim(-RY - ALT / 2 - 0.5, RY + ALT / 2 + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


if __name__ == "__main__":
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    fig = figura_ciclo()
    fig.savefig(SAIDA, facecolor="white")
    plt.close(fig)
    print(f"{SAIDA} ok")

"""
SAFIRA | Secao 4.2.1 - Geracao das figuras da exploracao de dados.

Divisao de responsabilidades entre as bibliotecas, deliberada e documentada na
secao 4.2.1: seaborn responde pela gramatica estatistica, pela camada de dados
e pelo tema (barplot, lineplot, heatmap, relplot, set_theme); matplotlib
responde pelo que o seaborn nao abstrai, isto e, eixos secundarios, anotacoes
posicionais, formatacao dos ticks e composicao de subplots com proporcoes
assimetricas.

Cada funcao recebe o DataFrame limpo e devolve a Figure, para que o grafico
apareca como saida de celula no notebook. A escrita em disco acontece apenas
quando o modulo e executado como script.
"""
from __future__ import annotations

import os
import textwrap

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from clean import faixa_antecedencia

# ------------------------------------------------------------------- constantes
PALETA_AZUL = ["#00A0DF", "#2E5FA3", "#E8871E", "#C0392B"]
AZ_ESC, AZ_CLA, CINZA, VERM, LARANJA = "#0A2A6B", "#00A0DF", "#9AA5B1", "#C0392B", "#E8871E"
# Familia extra, exclusiva do G1: separa visualmente o painel de vies do
# painel de detracao, sem reaproveitar cores ja associadas a outro
# significado no restante do documento.
ROXO_CLA, ROXO_ESC = "#9482B0", "#3D2C52"
# Extra, exclusiva do G3: sequencia de severidade (azul -> ouro -> laranja ->
# vermelho) pra diferenciar quatro faixas de atraso na mesma figura, coisa
# que a paleta institucional de duas cores nao cobre.
OURO = "#C99A3B"
PALETA_SEVERIDADE = [AZ_CLA, OURO, LARANJA, VERM]
# Extra, exclusiva do G6: vermelho pleno para a variavel mais associada ao
# alvo, salmao para a segunda, cinza quente para o resto (associacao nula).
SALMAO = "#CD6B4E"
CINZA_BARRA = "#C9C4BB"

NOMES_VAR = {
    "ESTATISTICA_ATRASOSAIDA": "Atraso na saída",
    "ATRASO_CHEGADA": "Atraso na chegada",
    "TEMPO_VOO": "Tempo de voo",
    "N_TRECHOS": "Nº de trechos",
    "QTDE_VIAGENS_12M": "Viagens em 12 meses",
    "QTDE_VIAGENS_24M": "Viagens em 24 meses",
    "QTDE_VIAGENS_36M": "Viagens em 36 meses",
}
ORDEM_VAR = list(NOMES_VAR.keys())

ORD_ATRASO = ["a. Sem Atraso", "b. 15m - 60m", "c. 61m - 120m", "d. >120m"]
LAB_ATRASO = ["Sem atraso\n(<15 min)", "15 a 60 min", "61 a 120 min", "Acima de 120 min"]
LAB_ATRASO_LEGENDA = ["Sem atraso (até 15 min)", "15 a 60 min", "61 a 120 min", "Acima de 120 min"]

NOME_TRI = {"Q1": "jan–mar", "Q2": "abr–jun", "Q3": "jul–set", "Q4": "out–dez"}

TIERS = ["Sem cadastro", "Azul Fidelidade", "Topazio", "Safira", "Diamante"]
TIERS_LAB = ["Sem cadastro", "Azul Fidelidade", "Topázio", "Safira", "Diamante"]

BINS_LIMIAR = [-1, 0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 10_000]
LAB_LIMIAR = ["0", "1-5", "6-10", "11-15", "16-20", "21-30", "31-45", "46-60",
              "61-90", "91-120", "121-180", "181-240", ">240"]

# Acima deste incremento em pontos percentuais a curva de risco muda de regime.
# Usado para colorir as barras, tracar a linha de referencia e localizar a
# faixa de inflexao: um valor so, para os tres nao poderem divergir.
LIMIAR_MARGINAL = 4

OUT_DIR = os.environ.get("SAFIRA_OUT_DIR", "out/figuras")


def aplicar_tema() -> None:
    """Tema visual unico para todas as figuras, com a paleta institucional."""
    sns.set_theme(
        style="whitegrid", context="notebook", palette=PALETA_AZUL,
        rc={"grid.linestyle": "--", "grid.alpha": 0.25, "axes.edgecolor": "#333"},
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    })


def virgula(v: float, casas: int = 1, sufixo: str = "") -> str:
    """Formata numero com separador decimal em virgula, conforme a norma."""
    return f"{v:.{casas}f}".replace(".", ",") + sufixo


def _fmt(casas: int = 0, sufixo: str = "") -> FuncFormatter:
    """Formatador de eixo com virgula decimal, independente do locale."""
    return FuncFormatter(lambda v, _pos: virgula(v, casas, sufixo))


def rotulos(matriz, casas: int = 1):
    """Anotacoes de heatmap com virgula decimal, ja que o fmt do seaborn usa ponto."""
    return np.array([["" if pd.isna(v) else virgula(v, casas) for v in linha]
                     for linha in np.asarray(matriz)])


def _rodape(ax, texto: str, y: float = -0.20) -> None:
    ax.text(0, y, texto, transform=ax.transAxes, fontsize=8.5, color="#555")


def _cabecalho_kicker(fig, x: float, kicker: str, titulo: str, subtitulo: str = "") -> None:
    """Antetitulo, titulo e subtitulo (opcional) em coordenadas de figura.

    Usa `fig.text` em vez de `ax.text(transform=ax.transAxes)` porque a
    posicao do cabecalho, aqui, nao deve depender da altura interna do eixo
    (que varia com o numero de anotacoes do grafico); a margem superior da
    figura e reservada para ele via `subplots_adjust(top=...)`. Sem
    subtitulo, usado quando cada painel abaixo tem sua propria legenda
    (G6), o titulo ganha uma linha a mais de respiro.
    """
    fig.text(x, 0.94, kicker, fontsize=8.5, color=CINZA, fontweight="bold", va="top")
    fig.text(x, 0.885, titulo, fontsize=15.5, color="#111", fontweight="bold", va="top")
    if subtitulo:
        fig.text(x, 0.795, subtitulo, fontsize=9.3, color="#666", va="top", linespacing=1.6)


def _cabecalho_subpainel(ax, letra: str, titulo: str, subtitulo: str) -> None:
    """Cabecalho leve de sub-painel: "(a) Titulo" e subtitulo cinza abaixo.

    Variante mais simples do `_cabecalho_kicker`, sem regua nem numero em
    caixa: usada quando os dois paineis ja estao sob um cabecalho geral
    (G6), e cada um so precisa de uma legenda curta pra se diferenciar.
    """
    ax.text(0, 1.155, f"({letra})", transform=ax.transAxes, fontsize=11,
            color="#111", fontweight="bold", va="bottom")
    ax.text(0.075, 1.155, titulo, transform=ax.transAxes, fontsize=12.5,
            color="#111", fontweight="bold", va="bottom")
    ax.text(0, 1.04, subtitulo, transform=ax.transAxes, fontsize=9.3,
            color="#666", va="bottom", linespacing=1.5)


aplicar_tema()


# ------------------------------------------- G1: atraso, detracao e vies de resposta
def _cabecalho_painel(ax, indice: str, titulo: str, subtitulo: str):
    """Regua preta e cabecalho numerado que abrem um painel do Grafico 1.

    Identifica cada painel como uma leitura distinta do mesmo fenomeno (a
    associacao com a detracao e o vies amostral) antes que o leitor chegue
    as barras. Devolve o texto do subtitulo para quem precisar medir sua
    largura e emendar um trecho adicional na mesma linha.
    """
    ax.plot([0, 1], [1.32, 1.32], transform=ax.transAxes, color="#111",
            lw=1.6, clip_on=False, solid_capstyle="butt")
    ax.text(0, 1.20, indice, transform=ax.transAxes, fontsize=9,
            color=CINZA, fontweight="bold", va="bottom")
    ax.text(0.055, 1.185, titulo, transform=ax.transAxes, fontsize=14,
            color="#111", fontweight="bold", va="bottom")
    return ax.text(0, 1.075, subtitulo, transform=ax.transAxes, fontsize=9.5,
                    color="#666", va="bottom")


def _barras_painel(ax, serie: pd.Series, rotulos_x: list[str], cor_base: str,
                    cor_destaque: str, casas_eixo: int, casas_rotulo: int,
                    sufixo: str) -> None:
    """Barras de uma faixa de atraso com a ultima faixa destacada.

    Usada pelos dois paineis do Grafico 1: mesma grade de categorias no eixo
    x, cores e formato de rotulo diferentes por painel, para a leitura lado
    a lado funcionar sem forcar as duas escalas no mesmo eixo.
    """
    dados = serie.reindex(ORD_ATRASO).reset_index()
    dados.columns = ["FAIXA_ATRASO", "VALOR"]
    cores = [cor_base] * (len(dados) - 1) + [cor_destaque]
    sns.barplot(data=dados, x="FAIXA_ATRASO", y="VALOR", order=ORD_ATRASO,
                hue="FAIXA_ATRASO", palette=cores, legend=False, width=0.6, ax=ax)

    ax.set_facecolor("#FAF9F6")
    ax.set(xlabel="", ylabel="")
    ax.margins(x=0.09)
    ax.grid(axis="y", color="#D8D5CF", lw=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(range(len(dados)))
    ax.set_xticklabels(rotulos_x, fontsize=10, color="#333")
    ax.tick_params(axis="y", labelsize=10, colors="#555", length=0)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.get_xticklabels()[-1].set_color("#111")
    ax.yaxis.set_major_formatter(_fmt(casas_eixo, sufixo))
    ax.set_ylim(0, serie.max() * 1.28)
    for i, v in enumerate(dados["VALOR"].to_numpy()):
        cor_rotulo = cor_destaque if i == len(dados) - 1 else "#222"
        ax.text(i, v + serie.max() * 0.035, virgula(v, casas_rotulo, sufixo),
                ha="center", va="bottom", fontsize=12.5, fontweight="bold",
                color=cor_rotulo)


def g1_atraso_dose_resposta(df: pd.DataFrame, dist: pd.DataFrame):
    """Dois paineis de barras, lado a lado, para a detracao e o vies do item (e).

    A versao anterior sobrepunha as duas leituras num so eixo com escala
    secundaria (linha para a razao, barras para a taxa); a comparacao ficava
    dificil porque as duas grandezas nao compartilham unidade nem intuicao
    de eixo. Separar em dois paineis com a mesma ordem de categorias no eixo
    x preserva a leitura conjunta (os mesmos rotulos, lado a lado) sem forcar
    percentual e razao na mesma reta.
    """
    base = (df.groupby("FAIXA_ATRASO", observed=True)["DETRATOR"]
              .mean().mul(100).reindex(ORD_ATRASO))

    pop = dist.groupby("DELAY_DEPARTURE_RANGE")["PERC_PAX"].sum()
    pop = (pop / pop.sum() * 100).reindex(ORD_ATRASO)
    amo = df["FAIXA_ATRASO"].value_counts(normalize=True).mul(100).reindex(ORD_ATRASO)
    razao = (amo / pop).reindex(ORD_ATRASO)

    # Paineis mais estreitos que um grafico de largura plena: o rotulo da
    # ultima faixa quebra em duas linhas para nao colidir com o vizinho.
    rotulos_x = LAB_ATRASO[:-1] + [LAB_ATRASO[-1].replace(" 120", "\n120")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.patch.set_facecolor("#FAF9F6")
    fig.subplots_adjust(top=0.72, bottom=0.16, wspace=0.32)

    _barras_painel(ax1, base, rotulos_x, AZ_CLA, AZ_ESC, 0, 1, "%")
    _cabecalho_painel(ax1, "01", "Associação: taxa de detratores",
                       "% de respondentes classificados como detratores, por faixa de atraso")

    _barras_painel(ax2, razao, rotulos_x, ROXO_CLA, ROXO_ESC, 1, 2, "×")
    ax2.axhline(1, color=VERM, lw=1.4, ls="--", zorder=1)
    ax2.text(-0.11, 1, "1,0×", transform=ax2.get_yaxis_transform(), color=VERM,
              fontsize=10, fontweight="bold", ha="right", va="center")
    subtitulo = _cabecalho_painel(ax2, "02", "Viés: sobre e sub-representação",
                                   "Participação na pesquisa ÷ participação na população de PAX ·")
    fig.canvas.draw()
    caixa = subtitulo.get_window_extent(renderer=fig.canvas.get_renderer())
    caixa_eixo = caixa.transformed(ax2.transAxes.inverted())
    ax2.text(caixa_eixo.x1, 1.075, " tracejado = sem viés", transform=ax2.transAxes,
              fontsize=9.5, color=VERM, va="bottom", fontweight="bold")

    return fig


# ------------------------------------------------- G0: evolucao trimestral (item e)
def g0_serie_temporal(df: pd.DataFrame):
    """Serie trimestral da detracao contra a incidencia de atrasos.

    O contraste entre as duas series e o que sustenta a leitura de efeito de
    periodo: a detracao sobe acima do que a operacao explica. As anotacoes
    apontam diretamente o pico de detratores e o piso de atrasos, e o
    preenchimento entre as series troca de cor conforme qual delas esta por
    cima, para que a distancia entre elas (o efeito nao explicado pela
    operacao) fique legivel sem depender so da legenda.
    """
    t = (df.groupby("TRIMESTRE", observed=True)
           .agg(Detratores=("DETRATOR", "mean"),
                Atrasos=("FAIXA_ATRASO", lambda s: (s != ORD_ATRASO[0]).mean()))
           .mul(100).reset_index())

    anos = t["TRIMESTRE"].str[:4]
    tris = t["TRIMESTRE"].str[4:]
    x = np.arange(len(t))
    det = t["Detratores"].to_numpy()
    atr = t["Atrasos"].to_numpy()

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    fig.patch.set_facecolor("#FAF9F6")
    ax.set_facecolor("#FAF9F6")
    fig.subplots_adjust(top=0.62, bottom=0.175, left=0.06, right=0.90)

    # separadores verticais tracejados entre anos
    limites_ano = np.where(anos.to_numpy()[:-1] != anos.to_numpy()[1:])[0]
    for lim in limites_ano:
        ax.axvline(lim + 0.5, color="#C9C6BE", lw=1.0, ls=(0, (2, 2)), zorder=1.5)

    # faixa vertical destacando o trimestre de pico de detratores
    i_pico = int(np.argmax(det))
    ax.axvspan(i_pico - 0.5, i_pico + 0.5, color=LARANJA, alpha=0.12, zorder=0)

    # preenchimento condicional: quem esta por cima muda a cor da area,
    # porque a distancia entre as series e o que sustenta a leitura de
    # efeito de periodo, nao o nivel absoluto de nenhuma delas.
    ax.fill_between(x, atr, det, where=(atr >= det), interpolate=True,
                     color=AZ_CLA, alpha=0.18, lw=0, zorder=1)
    ax.fill_between(x, atr, det, where=(det >= atr), interpolate=True,
                     color=VERM, alpha=0.14, lw=0, zorder=1)

    sns.lineplot(x=x, y=det, color=VERM, lw=2.4, marker="o", markersize=6,
                 zorder=3, ax=ax)
    sns.lineplot(x=x, y=atr, color=AZ_CLA, lw=2.4, marker="o", markersize=6,
                 zorder=3, ax=ax)

    # rotulo direto no fim de cada linha, no lugar de legenda; se os valores
    # finais estiverem proximos, afasta os dois rotulos pra nao colidirem.
    fim = sorted([(det[-1], "Detratores", VERM), (atr[-1], "Atrasos", AZ_CLA)])
    vao_min = 1.8
    if fim[1][0] - fim[0][0] < vao_min:
        centro = (fim[0][0] + fim[1][0]) / 2
        posicoes = [centro - vao_min / 2, centro + vao_min / 2]
    else:
        posicoes = [fim[0][0], fim[1][0]]
    for (_, rotulo, cor), y in zip(fim, posicoes):
        ax.text(x[-1] + 0.25, y, rotulo, color=cor, fontsize=10.5,
                fontweight="bold", va="center")

    # marcador vazado + anotacao no pico de detratores
    ax.scatter([i_pico], [det[i_pico]], s=64, facecolor="#FAF9F6",
               edgecolor=VERM, linewidth=2, zorder=4)
    ax.annotate(f"{virgula(det[i_pico], 1, '%')} de detratores",
                (i_pico, det[i_pico]), xytext=(i_pico - 0.3, det[i_pico] + 5.5),
                fontsize=10.5, fontweight="bold", color=VERM, ha="left")
    ax.annotate(f"{NOME_TRI[tris.iloc[i_pico]]} de {anos.iloc[i_pico]}, o maior da série",
                (i_pico, det[i_pico]), xytext=(i_pico - 0.3, det[i_pico] + 3.8),
                fontsize=9, color="#666", ha="left")

    # marcador vazado + anotacao no piso de atrasos
    i_piso = int(np.argmin(atr))
    ax.scatter([i_piso], [atr[i_piso]], s=64, facecolor="#FAF9F6",
               edgecolor=AZ_CLA, linewidth=2, zorder=4)
    ax.annotate(f"Atrasos no piso: {virgula(atr[i_piso], 1, '%')}",
                (i_piso, atr[i_piso]), xytext=(i_piso - 0.3, atr[i_piso] - 3.2),
                fontsize=10.5, fontweight="bold", color=AZ_CLA, ha="left")
    ax.annotate(f"e a detração segue em {virgula(det[i_piso], 1, '%')}",
                (i_piso, atr[i_piso]), xytext=(i_piso - 0.3, atr[i_piso] - 4.9),
                fontsize=9, color="#666", ha="left")

    ax.set_xlim(-0.6, len(t) - 1 + 1.9)
    topo, piso = float(max(det.max(), atr.max())), float(min(det.min(), atr.min()))
    ax.set_ylim(piso - 5, topo + 9)
    ax.set(xlabel="", ylabel="")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#E4E1DA", lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", labelsize=9.5, colors="#555", length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(_fmt(0, "%"))

    ax.set_xticks(x)
    ax.set_xticklabels([NOME_TRI[q] for q in tris], fontsize=9, color="#333")
    ax.get_xticklabels()[i_pico].set_fontweight("bold")
    ax.get_xticklabels()[i_pico].set_color("#111")

    # segunda linha de rotulo, abaixo da primeira: o ano, centralizado em
    # cada grupo de trimestres.
    for grupo in np.split(x, limites_ano + 1):
        ax.text(grupo.mean(), -0.135, anos.iloc[grupo[0]],
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9.5, fontweight="bold", color="#333")

    _cabecalho_kicker(
        fig, 0.06, "PERCENTUAL POR TRIMESTRE",
        "A detração cresce além do que os atrasos explicam",
        "Cada ponto reúne três meses. Detratores = % de respondentes\n"
        "insatisfeitos; atrasos = % de voos com atraso na saída no mesmo\n"
        "período. A área sombreada é a distância entre as duas séries.")
    return fig


# ---------------------------------------------- G4: detracao por tier x faixa atraso
def g4_detracao_por_tier(df: pd.DataFrame):
    """Uma linha por faixa de atraso, percorrendo os tiers de fidelidade.

    Versao anterior era um heatmap; a interacao entre fidelizacao e falha
    operacional fica visivel do mesmo jeito nas quatro curvas (nenhuma e
    plana), e o formato de linha deixa o efeito por faixa comparavel direto,
    sem exigir que o leitor varra 20 celulas isoladas.
    """
    tiers = [t for t in TIERS if t in set(df["TIER_VIAGEM"].dropna())]
    tiers_lab = [TIERS_LAB[TIERS.index(t)] for t in tiers]
    h = (df[df["TIER_VIAGEM"].isin(tiers)]
         .pivot_table(index="TIER_VIAGEM", columns="FAIXA_ATRASO",
                      values="DETRATOR", aggfunc="mean", observed=True)
         .reindex(index=tiers, columns=ORD_ATRASO) * 100)
    x = np.arange(len(tiers))

    fig, ax = plt.subplots(figsize=(13, 8.6))
    fig.patch.set_facecolor("#FAF9F6")
    ax.set_facecolor("#FAF9F6")
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.06, right=0.80)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D8D5CF")
        spine.set_linewidth(1.0)
    for xi in x:
        ax.axvline(xi, color="#E9E6DF", lw=0.9, zorder=0)

    deltas = {}
    for faixa, cor, rotulo in zip(ORD_ATRASO, PALETA_SEVERIDADE, LAB_ATRASO_LEGENDA):
        y = h[faixa].to_numpy()
        sns.lineplot(x=x, y=y, color=cor, lw=2.6, marker="o", markersize=7,
                     zorder=3, ax=ax)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + h.to_numpy().max() * 0.022, virgula(yi, 1),
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=cor)
        deltas[faixa] = round(float(y[-1] - y[0]), 1)
        ax.text(x[-1] + 0.25, y[-1], rotulo, color=cor, fontsize=11.5,
                fontweight="bold", va="bottom")
        ax.text(x[-1] + 0.25, y[-1], f"\n{virgula(deltas[faixa], 1)} pts do 1º ao último tier",
                color="#666", fontsize=9.3, va="top")

    ax.set_xlim(-0.4, len(tiers) - 1 + 2.55)
    ax.set_ylim(0, h.to_numpy().max() * 1.14)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers_lab, fontsize=10, color="#333")
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.get_xticklabels()[-1].set_color("#111")
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", labelsize=10, colors="#555", length=0)
    ax.yaxis.set_major_formatter(_fmt(0, "%"))
    ax.grid(axis="y", visible=False)
    ax.set_ylabel("% DE CLIENTES DETRATORES", fontsize=8.5, color=CINZA,
                  fontweight="bold", labelpad=14)

    # seta indicando o sentido crescente de fidelidade, sob o eixo x
    centro = (x[0] + x[-1]) / 2
    seta = FancyArrowPatch((centro - 1.3, -0.145), (centro + 1.3, -0.145),
                            transform=ax.get_xaxis_transform(), color=CINZA,
                            arrowstyle="-|>", mutation_scale=12, lw=1.1, clip_on=False)
    ax.add_patch(seta)
    ax.text(centro, -0.185, "tier de fidelidade crescente",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9.5, color="#777")

    # a frase final aponta pra onde o efeito do tier e maior, calculado dos
    # dados, e nao fixo: se o padrao mudar entre os extremos e o meio, o
    # texto acompanha.
    maior_delta = max(deltas.values())
    faixas_maiores = [LAB_ATRASO[ORD_ATRASO.index(f)].replace("\n", " ")
                       for f, d in deltas.items() if d == maior_delta]
    texto_rodape = textwrap.fill(
        f"O efeito do tier é maior justamente nas faixas intermediárias "
        f"({virgula(maior_delta, 1)} pontos em {' e em '.join(faixas_maiores)}) "
        f"e menor nos extremos, onde a detração já está baixa ou já está saturada.",
        width=100)
    fig.text(0.06, 0.075, texto_rodape, fontsize=10.5, color="#333",
             va="top", linespacing=1.5)

    _cabecalho_kicker(
        fig, 0.06, "TAXA DE DETRATORES POR TIER E FAIXA DE ATRASO",
        "Em toda faixa de atraso, o tier mais alto detrata mais",
        "Cada linha é uma faixa de atraso, percorrendo os tiers do menos fidelizado ao mais\n"
        "fidelizado. As quatro sobem: nenhuma faixa escapa do agravamento.")
    return fig


# ------------------------------------------------------- G6: correlacao de Spearman
def _painel_correlacao_alvo(ax, corr_alvo: pd.Series) -> None:
    """Barras horizontais, da correlacao mais forte com o alvo a mais fraca."""
    s = corr_alvo.reindex(ORDEM_VAR).sort_values(ascending=True)
    y = np.arange(len(s))
    valores = s.to_numpy()
    cores = [VERM if v == valores.max() else SALMAO if v >= 0.20 else CINZA_BARRA
             for v in valores]
    barras = pd.DataFrame({"pos": y, "valor": valores, "var": list(s.index)})
    sns.barplot(data=barras, x="valor", y="pos", orient="h", hue="var",
                palette=dict(zip(s.index, cores)), legend=False, width=0.6,
                native_scale=True, zorder=2, ax=ax)
    ax.set(xlabel="", ylabel="")
    ax.set_yticks(y)
    ax.set_yticklabels([NOMES_VAR[k] for k in s.index], fontsize=10, color="#333")
    for i, v in enumerate(valores):
        if v >= 0.20:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color("#111")

    for yi, v in zip(y, valores):
        destaque = v >= 0.20
        deslocamento = (6, 0) if v >= 0 else (-6, 0)
        ax.annotate(virgula(v, 2), (v, yi), xytext=deslocamento,
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=10.5,
                    fontweight="bold" if destaque else "normal",
                    color=VERM if destaque else "#555")

    # piso com folga fixa, nao proporcional: se alguma correlacao sair
    # negativa o rotulo do valor nunca fica colado no nome da variavel.
    ax.set_xlim(min(0, valores.min() - 0.05), 0.5)
    ax.set_ylim(-0.7, len(s) - 0.3)
    ax.set_facecolor("#FAF9F6")
    ax.grid(axis="x", color="#E4E1DA", lw=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, labelsize=9, colors="#777")
    ax.xaxis.set_major_formatter(_fmt(1))
    ax.text(0, -0.22, "A escala vai até 0,5; a correlação de Spearman pode chegar a 1,0.",
            transform=ax.get_xaxis_transform(), fontsize=8.3, color="#888", va="top")


def _painel_correlacao_explicativas(ax, c: pd.DataFrame) -> None:
    """Matriz triangular inferior das explicativas entre si, numerada.

    A grade e a anotacao das celulas sao do `sns.heatmap`, que ja resolve a
    mascara do triangulo superior, a escala divergente centrada em zero e o
    contraste do texto sobre cada celula. Fica pro `matplotlib` so o que o
    heatmap nao abstrai: os rotulos das variaveis fora da grade (uma coluna
    de numeros mais um nome por linha) e a legenda de cor como barra
    horizontal, por isso `cbar=False`. Sem celula quadrada (`square=False`):
    forcar o aspecto encolhe o eixo e esmaga a largura dos rotulos de linha.
    """
    ordem = ORDEM_VAR
    n = len(ordem)
    cmap = plt.get_cmap("RdBu_r")

    m = c.reindex(index=ordem, columns=ordem)
    # triangulo inferior estrito: a diagonal nao informa e a metade de cima
    # repete a de baixo.
    mascara = ~np.tril(np.ones((n, n), dtype=bool), k=-1)
    anotacoes = pd.DataFrame([[virgula(v, 2) for v in linha] for linha in m.to_numpy()],
                             index=m.index, columns=m.columns)

    sns.heatmap(m, mask=mascara, annot=anotacoes, fmt="", cmap=cmap, center=0,
                vmin=-1, vmax=1, cbar=False, square=False,
                linewidths=1.2, linecolor="#FAF9F6",
                annot_kws={"fontsize": 9.3, "fontweight": "bold"}, ax=ax)

    # o heatmap desenha a celula (i, j) no intervalo [j, j+1] x [i, i+1], entao
    # o centro fica em +0,5; os rotulos abaixo seguem essa convencao.
    ax.set_xlim(-0.1, n - 0.9)
    ax.set_ylim(n + 0.1, -0.25)
    ax.axis("off")

    trans_linha = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i in range(n):
        ax.text(-0.30, i + 0.5, f"{i + 1}", transform=trans_linha, ha="right",
                va="center", fontsize=9, color=CINZA)
        ax.text(-0.26, i + 0.5, NOMES_VAR[ordem[i]], transform=trans_linha,
                ha="left", va="center", fontsize=9.3, color="#333")
        if i == 0:
            ax.text(0.30, i + 0.5, "primeira variável da ordem",
                    transform=trans_linha, ha="left", va="center",
                    fontsize=8.8, color="#999", style="italic")

    trans_coluna = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for j in range(n - 1):
        ax.text(j + 0.5, 1.01, f"{j + 1}", transform=trans_coluna, ha="center",
                va="bottom", fontsize=9, color=CINZA)

    cax = ax.inset_axes([0.58, -0.11, 0.42, 0.045])
    grad = np.linspace(-1, 1, 256).reshape(1, -1)
    cax.imshow(grad, cmap=cmap, aspect="auto", extent=[-1, 1, 0, 1])
    cax.set_xticks([])
    cax.set_yticks([])
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D8D5CF")
    cax.text(-1, -0.9, "−1 inversa", ha="left", va="top", fontsize=8.5, color="#666")
    cax.text(1, -0.9, "+1 direta", ha="right", va="top", fontsize=8.5, color="#666")


def g6_correlacao(df: pd.DataFrame):
    """Dois paineis: correlacao de cada variavel com o alvo, e das explicativas entre si.

    Spearman e nao Pearson porque todas as numericas apresentam forte
    assimetria positiva e o alvo e binario. Separar o alvo (painel a) das
    explicativas entre si (painel b) evita a linha/coluna assimetrica que
    a matriz unica tinha, com o alvo misturado as sete variaveis de
    redundancia operacional; aqui cada painel responde a uma pergunta.
    """
    cols = ["DETRATOR"] + ORDEM_VAR
    c_total = df[cols].corr(method="spearman")
    corr_alvo = c_total["DETRATOR"].drop("DETRATOR")
    c = c_total.drop(index="DETRATOR", columns="DETRATOR")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("#FAF9F6")
    ax1.set_facecolor("#FAF9F6")
    ax2.set_facecolor("#FAF9F6")
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.05, right=0.98, wspace=0.38)

    _painel_correlacao_alvo(ax1, corr_alvo)
    _cabecalho_subpainel(ax1, "a", "Correlação de cada variável com a detração",
                         "Ordenada da mais forte para a mais fraca. Em vermelho,\n"
                         "acima de 0,20; em cinza, associação praticamente nula.")

    _painel_correlacao_explicativas(ax2, c)
    _cabecalho_subpainel(ax2, "b", "Correlação entre as variáveis explicativas",
                         "As colunas repetem a ordem das linhas, por isso só o triângulo inferior aparece.\n"
                         "Valores altos indicam redundância.")

    fig.text(0.05, 0.055,
             "As duas medidas de atraso lideram, mas nenhuma passa de "
             f"{virgula(corr_alvo.max(), 2)} — a detração não é explicada por uma única "
             "variável operacional.", fontsize=10, color="#333")

    _cabecalho_kicker(fig, 0.05, "CORRELAÇÃO DE SPEARMAN",
                      "O atraso é o único ligado à detração; o resto das variáveis é redundante entre si")
    return fig


# ------------------------------------------------------------ G2: limiar de atraso
def g2_limiar_atraso(df: pd.DataFrame):
    """Curva de risco por minuto de atraso e o impacto marginal entre faixas.

    Responde a pergunta 5 do escopo da Azul. A composicao com proporcoes
    assimetricas (2,2 para 1) e feita com gridspec do matplotlib, porque o
    painel inferior e leitura de apoio e nao tem o mesmo peso visual.

    As duas marcas seguem o modelo de tendencia/comparacao/proporcao/relacao/
    distribuicao adotado para padronizar os graficos da secao 4.2.1, e cada
    painel responde a uma pergunta diferente:
    - Painel superior, linha: TENDENCIA. O eixo e uma variavel continua
      (minutos de atraso) e o que importa e o formato da curva, isto e, onde
      ela acelera. E a inclinacao que carrega a informacao, nao cada ponto
      isolado.
    - Painel inferior, barra: COMPARACAO. Cada barra e a diferenca marginal
      entre uma faixa e a anterior, um numero discreto comparado contra o
      limiar fixo de LIMIAR_MARGINAL p.p. A pergunta aqui e binaria, "esta
      faixa passou do limiar ou nao", por isso barra e nao linha.
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})

    faixa = pd.cut(df["ESTATISTICA_ATRASOSAIDA"], BINS_LIMIAR, labels=LAB_LIMIAR)
    r = (df.groupby(faixa, observed=False)["DETRATOR"].mean().mul(100)
           .reset_index())
    r.columns = ["faixa", "taxa"]
    r = r.dropna(subset=["taxa"]).reset_index(drop=True)
    r["marginal"] = r["taxa"].diff()

    sns.lineplot(data=r, x="faixa", y="taxa", marker="o", color=VERM,
                 lw=2.6, markersize=7, ax=ax)
    ax.fill_between(range(len(r)), r["taxa"], color=VERM, alpha=0.10)

    media = df["DETRATOR"].mean() * 100
    ax.axhline(media, color=CINZA, ls=":", lw=1.5)
    ax.text(0.3, media + 2.5, f"média geral {virgula(media, 1, '%')}",
            color="#666", fontsize=8.5, ha="left")

    # Janela de inflexao localizada pelo rotulo, nao por posicao fixa.
    # Ponto de inflexao: primeira faixa cujo custo marginal ultrapassa o limiar.
    # O texto reusa o proprio rotulo do bin (ex.: "21-30"), em vez de recalcular
    # o intervalo a partir de BINS_LIMIAR: os dois sao equivalentes, mas o corte
    # do pd.cut e (20, 30], que em minutos inteiros comeca em 21, e nao em 20.
    # Reusar o rotulo elimina essa segunda fonte de verdade, que divergia do
    # eixo x em uma unidade.
    acelera = r.index[r["marginal"] >= LIMIAR_MARGINAL]
    if len(acelera):
        i = int(acelera[0])
        ax.axvspan(i - 0.5, i + 0.5, color=LARANJA, alpha=0.16, zorder=0)
        ax.annotate(f"Ponto de inflexão\n{r.loc[i, 'faixa']} min",
                    (i, float(r.loc[i, "taxa"])),
                    xytext=(i + 2, max(media - 13, 4)), fontsize=10,
                    fontweight="bold", color=LARANJA,
                    arrowprops=dict(arrowstyle="->", color=LARANJA, lw=1.6))

    ax.set(xlabel="", ylabel="Taxa de detratores")
    ax.yaxis.set_major_formatter(_fmt(0, "%"))
    ax.set_title("Limiar de atraso: curva de risco e impacto marginal", pad=12)

    rotulo_acima = f"Acima de {LIMIAR_MARGINAL} p.p."
    rotulo_ate = f"Até {LIMIAR_MARGINAL} p.p."
    r["acelera"] = np.where(r["marginal"] >= LIMIAR_MARGINAL, rotulo_acima, rotulo_ate)
    sns.barplot(data=r, x="faixa", y="marginal", hue="acelera",
                palette={rotulo_ate: AZ_CLA, rotulo_acima: LARANJA},
                dodge=False, width=0.6, ax=ax2)
    ax2.axhline(LIMIAR_MARGINAL, color=LARANJA, ls="--", lw=1.2)
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
    ax2.set(ylabel="Variação em p.p. vs.\nfaixa anterior",
            xlabel="Atraso na saída (minutos)")
    ax2.yaxis.set_major_formatter(_fmt(0))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    _rodape(ax2, "Até 15 min o custo marginal é estável, na ordem de 2 p.p. por "
                 "faixa. A partir de 20 min ele dobra e segue acelerando até 90 min.",
            y=-0.85)
    return fig


# ------------------------------------------- G3: antecedencia do aviso de cancelamento
def g3_antecedencia_cancelamento(df: pd.DataFrame):
    """Detracao e NPS medio por antecedencia do aviso, apenas voos cancelados.

    Mantem o evento negativo constante e varia so a comunicacao, o que isola o
    efeito do aviso.

    Marca escolhida pelo modelo de tendencia/comparacao/proporcao/relacao/
    distribuicao adotado para padronizar os graficos da secao 4.2.1: BARRA,
    nao linha, porque o eixo x sao janelas operacionais distintas ("mesmo
    dia", "1 dia", "2 a 3 dias"...), nao uma escala continua e uniforme como
    minutos de atraso. A distancia entre "mesmo dia" e "1 dia" nao e
    comparavel, para a decisao da companhia, a distancia entre "31 a 60
    dias" e "mais de 60 dias" — mesmo a taxa caindo de forma monotonica, a
    pergunta e "quanto cada janela custa em detracao", categoria contra
    categoria, e nao o formato de uma curva continua. O NPS medio, no eixo
    secundario, usa linha para nao competir visualmente com as barras que ja
    ocupam esse canal, nao porque responda a uma pergunta de tendencia.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    c = df[df["CANCELAMENTO_VOO"].astype(bool)].copy()
    # Faixas vindas de clean.py, para grafico e tabela nao poderem divergir.
    c["FX"] = faixa_antecedencia(c["ANTECEDENCIA_CANCELAMENTO"])
    s = (c.groupby("FX", observed=True)
           .agg(taxa=("DETRATOR", "mean"), nps=("NPS_PRINCIPAL", "mean"))
           .reset_index())
    s["taxa"] *= 100

    # RdYlGn "puro": vermelho na pior janela (mesmo dia), verde na melhor
    # (mais de 60 dias). Antes esta linha invertia a paleta e revertia a
    # inversao (RdYlGn_r + [::-1]), o que da o mesmo resultado por
    # simetria, mas obriga o leitor a refazer essa conta para confirmar
    # que nao e um bug escondido.
    sns.barplot(data=s, x="FX", y="taxa", hue="FX", legend=False, dodge=False,
                width=0.6, palette=sns.color_palette("RdYlGn", len(s)), ax=ax)
    for i, v in enumerate(s["taxa"]):
        ax.text(i, v + 1.5, virgula(v, 1, "%"), ha="center",
                fontweight="bold", fontsize=10)

    ax.set(xlabel="", ylabel="Taxa de detratores", ylim=(0, 80))
    ax.yaxis.set_major_formatter(_fmt(0, "%"))
    plt.setp(ax.get_xticklabels(), rotation=20)

    # matplotlib: eixo secundario para o NPS medio, escala e sinal distintos.
    ax2 = ax.twinx()
    ax2.grid(False)
    sns.lineplot(x=range(len(s)), y=s["nps"], marker="o", color=AZ_ESC,
                 lw=2.4, markersize=7, ax=ax2)
    ax2.axhline(0, color=CINZA, ls=":", lw=1.2)
    ax2.set(ylabel="NPS médio", ylim=(-60, 60))
    ax2.yaxis.set_major_formatter(_fmt(0))
    ax2.yaxis.label.set_color(AZ_ESC)
    ax2.tick_params(axis="y", colors=AZ_ESC)

    ax.set_title("Cancelamento: efeito da antecedência do aviso sobre a detração", pad=12)
    n_cancel = f"{len(c):,}".replace(",", ".")
    _rodape(ax, f"Apenas voos cancelados (n = {n_cancel}). Barras: taxa de "
                "detratores. Linha: NPS médio.", y=-0.26)
    return fig


# ---------------------------------------------------------------- G5: sazonalidade
def g5_sazonalidade(df: pd.DataFrame):
    """Pequenos multiplos da detracao mensal, um painel por faixa de atraso.

    Controlar por faixa e o que separa sazonalidade propria de composicao
    operacional.

    Marca escolhida pelo modelo de tendencia/comparacao/proporcao/relacao/
    distribuicao adotado para padronizar os graficos da secao 4.2.1: LINHA,
    sem ambiguidade. Mes e uma progressao temporal ciclica, e a pergunta e o
    formato da curva se repetindo entre os quatro paineis (dezembro alto,
    agosto baixo), nao a magnitude isolada de um mes contra outro. Pequenos
    multiplos de linha comparam a forma de quatro tendencias ao mesmo tempo,
    o que uma unica figura ou uma barra nao fariam.

    ATENCAO ao interpretar: sharey=False, entao cada eixo y comeca perto do
    proprio minimo, nao de zero. Isso revela o padrao interno de cada faixa,
    mas torna a amplitude visual entre paineis nao comparavel a olho nu.
    """
    d = df.copy()
    d["Mês"] = d["DATA_STD"].dt.month
    d["Faixa"] = pd.Categorical(d["FAIXA_ATRASO"], categories=ORD_ATRASO, ordered=True)
    agr = (d.groupby(["Faixa", "Mês"], observed=True)["DETRATOR"]
             .mean().mul(100).reset_index())

    g = sns.relplot(data=agr, x="Mês", y="DETRATOR", col="Faixa", hue="Faixa",
                    kind="line", marker="o", lw=2.2, markersize=5,
                    palette=PALETA_AZUL, legend=False,
                    facet_kws={"sharey": False}, height=3.4, aspect=0.95)

    meses = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    for ax, chave, titulo, cor in zip(g.axes.flat, ORD_ATRASO, LAB_ATRASO, PALETA_AZUL):
        sub = agr[agr["Faixa"] == chave]
        ax.fill_between(sub["Mês"], sub["DETRATOR"].min() - 1, sub["DETRATOR"],
                        color=cor, alpha=0.12)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(meses, fontsize=8)
        ax.set_title(titulo.replace("\n", " "), fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(_fmt(0, "%"))
        ax.set_xlabel("")
    g.axes.flat[0].set_ylabel("Taxa de detratores")
    g.figure.suptitle("Sazonalidade da detração, controlada por faixa de atraso",
                      fontweight="bold", y=1.05)
    g.figure.text(0.09, -0.06, "O padrão de dezembro alto e agosto baixo persiste "
                  "dentro de todas as faixas: não é apenas composição operacional.",
                  fontsize=8.5, color="#555")
    return g.figure


# ------------------------------- A.1: histogramas do teste de normalidade (Anexos)
def milhar(v: float) -> str:
    """Separador de milhar com ponto, conforme a norma adotada no documento."""
    return f"{int(v):,}".replace(",", ".")


def _fmt_milhar() -> FuncFormatter:
    return FuncFormatter(lambda v, _pos: milhar(v))


def a1_histograma_normalidade(serie: pd.Series, titulo: str, rotulo_x: str,
                              isolar_zero: bool = False, inteiros: bool = False):
    """Histograma de uma variavel quantitativa para a secao A.1.1.

    Fora de FIGURAS de proposito: recebe uma Serie, e nao o DataFrame de
    `clean.pkl` que o modo script consome, porque a base do anexo e a analitica
    de `data/processed`. Quem a chama e o notebook do anexo.

    Tres decisoes de desenho, porque sem elas a figura nao sustenta o texto que
    a descreve. O eixo de frequencia e logaritmico: em escala linear a barra
    dominante achata o resto contra o eixo e as tres variaveis ficam
    visualmente identicas. O eixo horizontal para no percentil 99, com os
    registros omitidos declarados no rodape, para a area do grafico nao ser
    tomada por valores extremos isolados. E, quando `isolar_zero`, o valor zero
    ganha barra propria: com intervalos uniformes ele se mistura aos valores
    baixos e a barra deixa de corresponder a proporcao citada no texto.
    """
    serie = serie.dropna()
    p99 = serie.quantile(0.99)
    acima_p99 = int((serie > p99).sum())

    if inteiros:
        # Variavel de contagem: um intervalo por valor inteiro, senao os
        # intervalos fracionarios criam vaos que nao existem nos dados.
        limites = np.arange(serie.min(), p99 + 2) - 0.5
    else:
        limites = np.linspace(0 if isolar_zero else serie.min(), p99, 41)
    largura = limites[1] - limites[0]

    fig, ax = plt.subplots(figsize=(9, 4.8))

    if isolar_zero:
        zeros = int((serie == 0).sum())
        ax.bar(-largura / 2, zeros, width=largura * 0.92, color=VERM,
               edgecolor="white", linewidth=0.4, zorder=2)
        ax.hist(serie[(serie > 0) & (serie <= p99)], bins=limites, color=AZ_CLA,
                edgecolor="white", linewidth=0.4, zorder=2)
        ax.set_xlim(-largura * 1.4, p99)
        ax.annotate(f"barra isolada do valor zero:\n{milhar(zeros)} registros "
                    f"({virgula(zeros / len(serie) * 100, 1, '%')})",
                    xy=(-largura / 2, zeros), xytext=(0.30, 0.80),
                    textcoords="axes fraction", fontsize=9, color=VERM,
                    arrowprops=dict(arrowstyle="->", color=VERM, lw=1.2))
    else:
        ax.hist(serie[serie <= p99], bins=limites, color=AZ_CLA,
                edgecolor="white", linewidth=0.4, zorder=2)

    ax.set_yscale("log")
    ax.set(xlabel=rotulo_x, ylabel="Frequência (escala logarítmica)")
    ax.yaxis.set_major_formatter(_fmt_milhar())
    ax.xaxis.set_major_formatter(_fmt_milhar())
    ax.set_title(titulo, pad=12)
    _rodape(ax, f"Eixo horizontal cortado no percentil 99; {milhar(acima_p99)} "
                "registros acima não exibidos.", y=-0.22)
    return fig


# ------------------------------- G10 e G11: curvas do modelo candidato (#107)
#
# As duas nao entram em FIGURAS: o laco de `__main__` chama cada funcao com a
# base limpa, e estas leem alvo e probabilidade da particao de teste, que so
# existem depois do modelo treinado, no notebook de modelagem.
#
# As duas recebem o limiar, e nao o par de coordenadas, para que o ponto
# marcado seja o mesmo nas duas figuras por construcao. Passar as coordenadas
# prontas deixaria as duas marcarem pontos diferentes sem nada acusar.


def _ponto_de_operacao(y: np.ndarray, score: np.ndarray, limiar: float) -> dict:
    """Cobertura, precisao e taxa de falso positivo no limiar da operacao."""
    selecionado = score >= limiar
    vp = int((selecionado & (y == 1)).sum())
    fp = int((selecionado & (y == 0)).sum())
    positivos, negativos = int((y == 1).sum()), int((y == 0).sum())
    return {
        "cobertura": vp / positivos if positivos else 0.0,
        "precisao": vp / (vp + fp) if vp + fp else 0.0,
        "tfp": fp / negativos if negativos else 0.0,
        "n_selecionados": int(selecionado.sum()),
    }


def g10_curva_roc(y_verdadeiro, score, limiar: float | None = None):
    """Curva ROC do candidato contra a diagonal do classificador aleatorio.

    A diagonal entra tracejada e em cinza porque e referencia, e nao resultado.
    O contraste entre as duas usa estilo de linha alem da cor, para a figura
    continuar legivel impressa em escala de cinza.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.asarray(y_verdadeiro).astype(int)
    score = np.asarray(score, dtype=float)
    tfp, tvp, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.6, color=CINZA, zorder=1,
            label="Aleatório (AUC 0,50)")
    ax.plot(tfp, tvp, lw=2.4, color=AZ_CLA, zorder=3,
            label=f"Candidato (AUC {virgula(auc, 3)})")

    if limiar is not None:
        p = _ponto_de_operacao(y, score, limiar)
        ax.scatter([p["tfp"]], [p["cobertura"]], s=90, facecolor="#FAF9F6",
                   edgecolor=VERM, linewidth=2.2, zorder=4)
        ax.annotate(
            f"Limiar da operação ({virgula(limiar, 4)})",
            (p["tfp"], p["cobertura"]),
            xytext=(p["tfp"] + 0.06, p["cobertura"] - 0.12),
            fontsize=9.5, fontweight="bold", color=VERM,
            arrowprops={"arrowstyle": "-", "color": VERM, "lw": 1.2},
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Taxa de falsos positivos", fontsize=10.5)
    ax.set_ylabel("Cobertura dos Detratores", fontsize=10.5)
    ax.xaxis.set_major_formatter(_fmt(1))
    ax.yaxis.set_major_formatter(_fmt(1))
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.set_title("Curva ROC do modelo candidato", fontsize=12, loc="left")
    _rodape(ax, "Partição de teste. A ROC é otimista em base desbalanceada: mede a "
                "ordenação, não a\ndensidade de Detratores no topo da fila, que é a "
                "leitura de operação da Figura 11.\nFonte: Autoria própria.",
            y=-0.24)
    fig.tight_layout()
    return fig


def g11_precisao_cobertura(y_verdadeiro, score, limiar: float | None = None):
    """Precisao contra cobertura, com a prevalencia como piso do aleatorio.

    E a figura que sustenta a leitura de operacao: a ROC sobe rapido mesmo
    quando o topo da fila tem pouca densidade de Detratores, e e aqui que a
    queda da precisao conforme a fila cresce fica visivel.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y = np.asarray(y_verdadeiro).astype(int)
    score = np.asarray(score, dtype=float)
    precisao, cobertura, _ = precision_recall_curve(y, score)
    ap = average_precision_score(y, score)
    prevalencia = float(y.mean())

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.axhline(prevalencia, ls="--", lw=1.6, color=CINZA, zorder=1,
               label=f"Lista aleatória ({virgula(100 * prevalencia, 2, '%')})")
    ax.plot(cobertura, precisao, lw=2.4, color=AZ_ESC, zorder=3,
            label=f"Candidato (precisão média {virgula(ap, 3)})")

    if limiar is not None:
        p = _ponto_de_operacao(y, score, limiar)
        fila = f"{p['n_selecionados']:,}".replace(",", ".")
        ax.scatter([p["cobertura"]], [p["precisao"]], s=90, facecolor="#FAF9F6",
                   edgecolor=VERM, linewidth=2.2, zorder=4)
        ax.annotate(
            f"Limiar da operação: fila de {fila} contatos",
            (p["cobertura"], p["precisao"]),
            xytext=(p["cobertura"] + 0.05, p["precisao"] + 0.14),
            fontsize=9.5, fontweight="bold", color=VERM,
            arrowprops={"arrowstyle": "-", "color": VERM, "lw": 1.2},
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Cobertura dos Detratores", fontsize=10.5)
    ax.set_ylabel("Precisão no topo da fila", fontsize=10.5)
    ax.xaxis.set_major_formatter(_fmt(1))
    ax.yaxis.set_major_formatter(_fmt(1))
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_title("Precisão contra cobertura do modelo candidato",
                 fontsize=12, loc="left")
    _rodape(ax, "Partição de teste. A linha tracejada é a precisão que uma fila "
                "sorteada ao acaso teria,\ne é o piso contra o qual o ganho do "
                "modelo se mede.\nFonte: Autoria própria.", y=-0.24)
    fig.tight_layout()
    return fig


FIGURAS = {
    "g0_serie_temporal": g0_serie_temporal,
    "g1_atraso_e_detracao": g1_atraso_dose_resposta,
    "g2_limiar_atraso": g2_limiar_atraso,
    "g3_antecedencia_cancelamento": g3_antecedencia_cancelamento,
    "g4_detracao_por_tier": g4_detracao_por_tier,
    "g5_sazonalidade": g5_sazonalidade,
    "g6_correlacao": g6_correlacao,
}


if __name__ == "__main__":
    import pickle

    import matplotlib
    matplotlib.use("Agg")   # apenas em execucao por script, nunca no notebook

    from clean import carregar_populacao

    with open("out/clean.pkl", "rb") as fh:
        base = pickle.load(fh)
    populacao = carregar_populacao()

    os.makedirs(OUT_DIR, exist_ok=True)
    for nome, fn in FIGURAS.items():
        fig = fn(base, populacao) if nome.startswith("g1_") else fn(base)
        fig.savefig(os.path.join(OUT_DIR, f"{nome}.png"))
        plt.close(fig)
        print(f"{nome} ok")

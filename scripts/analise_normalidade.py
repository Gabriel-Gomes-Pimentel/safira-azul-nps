"""Analise de normalidade sem SciPy.

Le ``data/base_preprocessada.parquet``, aplica Jarque--Bera manualmente em
amostras de 2.000 observacoes e salva os histogramas em ``assets``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARQUIVO = Path("data/processed/base_analitica.parquet")
TAMANHO_AMOSTRA = 2_000
ALFA = 0.05


def jarque_bera_manual(dados):
    """Retorna estatistica JB e p-valor (qui-quadrado com 2 g.l.)."""
    dados = np.asarray(dados, dtype=float)
    n = len(dados)
    media = dados.mean()
    desvio = dados.std(ddof=0)
    if n < 2 or desvio == 0:
        return np.nan, np.nan
    skew = np.mean(((dados - media) / desvio) ** 3)
    kurt = np.mean(((dados - media) / desvio) ** 4)
    jb = (n / 6) * (skew**2 + ((kurt - 3) ** 2) / 4)
    # Para X ~ chi2(2), P(X >= jb) = exp(-jb / 2).
    p_valor = np.exp(-jb / 2)
    return jb, p_valor


def selecionar_variaveis(df):
    """Resolve o nome de cada variavel analisada na base atual.

    Cada variavel de destino tem uma lista de nomes de coluna aceitos,
    todos semanticamente equivalentes a ela (nome usado no enunciado do
    modulo e nome usado na base analitica documentada do projeto). Nao
    ha aliases entre variaveis distintas: se nenhum candidato for
    encontrado, o script falha explicitamente em vez de cair em uma
    coluna de significado diferente.
    """
    mapa_variaveis = {
        "TEMPO_VOO": ["tempo_espera", "TEMPO_VOO"],
        "ATRASO_CHEGADA": ["numero_atrasos", "ATRASO_CHEGADA"],
        "QTDE_VIAGENS_12M": ["qtde_viagens_12m", "QTDE_VIAGENS_12M"],
    }
    escolhidas = []
    for destino, candidatos in mapa_variaveis.items():
        coluna = next((c for c in candidatos if c in df.columns), None)
        if coluna is None:
            raise KeyError(
                f"Nao encontrei nenhuma coluna equivalente a '{destino}' "
                f"(candidatos verificados: {candidatos}). "
                "Ajuste a lista de candidatos no script; nao substitua "
                "por uma coluna de outra variavel."
            )
        escolhidas.append(coluna)
    return escolhidas


def comentario_histograma(serie):
    assimetria = serie.skew()
    if abs(assimetria) < 0.2:
        return "O formato e aproximadamente simetrico; visualmente e compativel com normalidade."
    direcao = "positiva, com cauda longa a direita" if assimetria > 0 else "negativa, com cauda longa a esquerda"
    return f"O formato apresenta assimetria {direcao}; visualmente nao reforca normalidade."


def comentario_media_mediana(media, mediana):
    diferenca = abs(media - mediana)
    if np.isclose(media, mediana, rtol=0.05, atol=1e-12):
        return "A diferenca e pequena em relacao a escala da variavel, compativel com simetria."
    return "A diferenca e relevante em relacao a escala da variavel, reforcando assimetria e nao normalidade."


def main():
    df = pd.read_parquet(ARQUIVO)
    variaveis = selecionar_variaveis(df)
    Path("assets").mkdir(exist_ok=True)

    resultados = []
    resumo_central = []
    comentarios_histogramas = []
    for variavel in variaveis:
        dados = pd.to_numeric(df[variavel], errors="coerce").dropna()
        if len(dados) < TAMANHO_AMOSTRA:
            raise ValueError(f"{variavel} possui menos de {TAMANHO_AMOSTRA} valores validos.")
        amostra = dados.sample(n=TAMANHO_AMOSTRA, random_state=42)
        jb, p_valor = jarque_bera_manual(amostra)
        conclusao = "Rejeita-se H0: evidencia contra normalidade." if p_valor < ALFA else "Nao se rejeita H0."
        resultados.append({
            "Variavel": variavel,
            "Estatistica JB": jb,
            "p-valor": p_valor,
            "Conclusao (alpha=0,05)": conclusao,
        })

        media, mediana = dados.mean(), dados.median()
        resumo_central.append({
            "Variavel": variavel,
            "Media": media,
            "Mediana": mediana,
            "Diferenca absoluta": abs(media - mediana),
            "Comentario": comentario_media_mediana(media, mediana),
        })

        plt.figure(figsize=(8, 5))
        plt.hist(dados, bins=30, edgecolor="black")
        plt.title(f"Histograma de {variavel}")
        plt.xlabel(variavel)
        plt.ylabel("Frequencia")
        plt.tight_layout()
        arquivo_imagem = Path("assets") / f"histograma_{variavel.lower()}.png"
        plt.savefig(arquivo_imagem, dpi=150)
        plt.close()
        comentarios_histogramas.append({"Variavel": variavel, "Comentario": comentario_histograma(dados)})

    tabela_jb = pd.DataFrame(resultados)
    tabela_central = pd.DataFrame(resumo_central)
    print("\n### Item c — teste de Jarque-Bera\n")
    print(tabela_jb.to_markdown(index=False, floatfmt=".6g"))
    print("\n### Item d — comentarios dos histogramas\n")
    print(pd.DataFrame(comentarios_histogramas).to_markdown(index=False))
    print("\n### Item e — media versus mediana\n")
    print(tabela_central.to_markdown(index=False, floatfmt=".6g"))


if __name__ == "__main__":
    main()
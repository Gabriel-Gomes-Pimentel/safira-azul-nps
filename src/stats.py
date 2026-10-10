"""
SAFIRA | Secao 4.2.1 - Classificacao das colunas, estatistica descritiva e
medidas de associacao.

Produz as tres tabelas da secao 4.2.1: numericas, categoricas e V de Cramer
contra o alvo binarizado. Inclui tambem o comparativo entre a taxa de detracao
bruta e a pos-estratificada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

from clean import faixa_antecedencia

# Ordem das tabelas da secao 4.2.1, itens (c) e (d).
NUMERICAS = [
    "TEMPO_VOO", "ESTATISTICA_ATRASOSAIDA", "ATRASO_CHEGADA",
    "ANTECEDENCIA_CANCELAMENTO", "QTDE_VIAGENS_12M", "QTDE_VIAGENS_24M",
    "QTDE_VIAGENS_36M", "N_TRECHOS",
]

CATEGORICAS = [
    "CLASSE_NPS", "VOO_TIPO", "TIPO_ENTRETENIMENTO", "CANAL_COMPRA", "SEGMENTO",
    "TIER_VIAGEM", "SUB_FIL_MOTIVOVIAGEM", "SUB_FIL_FREQUENCIAAZUL",
    "CANCELAMENTO_VOO", "SUB_ENTRETENIMENTO1", "SUB_ENTRETENIMENTO2",
    "FAIXA_ATRASO", "AEROPORTO_ORIGEM", "EQUIPAMENTO_TIPO", "BASE_AIRPORTLEG",
    "ASSENTOS", "VOO_NUMERO",
]

# Preditoras candidatas para a medida de associacao com o alvo. Ficam de fora
# CLASSE_NPS (e o proprio alvo) e os campos de altissima cardinalidade.
CAT_ASSOCIACAO = [
    "FAIXA_ATRASO", "CANCELAMENTO_VOO", "SUB_FIL_FREQUENCIAAZUL",
    "TIPO_ENTRETENIMENTO", "VOO_TIPO", "TIER_VIAGEM", "SUB_FIL_MOTIVOVIAGEM",
    "AEROPORTO_ORIGEM", "SEGMENTO", "CANAL_COMPRA",
]


def classificar_colunas(df, limite_binario: int = 2):
    """Separa as colunas em numericas, categoricas e temporais.

    O criterio nao e apenas o dtype: campos numericos com no maximo
    ``limite_binario`` valores distintos, como flags, sao tratados como
    categoricos, porque a media deles nao tem interpretacao de escala.

    As temporais saem em lista propria em vez de serem descartadas em silencio:
    elas nao entram nas tabelas descritivas, mas precisam ser visiveis na
    classificacao. Retorna (numericas, categoricas, temporais).
    """
    num, cat, temporais = [], [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            temporais.append(c)
        elif pd.api.types.is_bool_dtype(s):
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            (cat if s.nunique(dropna=True) <= limite_binario else num).append(c)
        else:
            cat.append(c)
    return num, cat, temporais


def tabela_numericas(df, cols=None) -> pd.DataFrame:
    """describe com percentis, nulidade e assimetria."""
    cols = cols or NUMERICAS
    t = df[cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    t["nulos"] = df[cols].isna().mean() * 100
    t["assimetria"] = df[cols].skew()
    t = t[["count", "mean", "50%", "std", "min", "max", "95%", "nulos", "assimetria"]]
    t.columns = ["n", "Média", "Mediana", "Desvio", "Mín", "Máx", "P95",
                 "% Nulo", "Assimetria"]
    return t.round(2)


def tabela_categoricas(df, cols=None) -> pd.DataFrame:
    """Contagem de categorias, moda, frequencia da moda e nulidade."""
    cols = cols or CATEGORICAS
    linhas = []
    for c in cols:
        vc = df[c].value_counts(dropna=True)
        linhas.append({
            "Variável": c,
            "Categorias": int(df[c].nunique()),
            "Moda": str(vc.index[0]),
            "Freq. moda": int(vc.iloc[0]),
            "% moda": round(vc.iloc[0] / len(df) * 100, 2),
            "% Nulo": round(df[c].isna().mean() * 100, 2),
        })
    return pd.DataFrame(linhas)


ROTULO_NULO = "Não informado"


def cramers_v(x: pd.Series, y: pd.Series, nulo_como_categoria: bool = True) -> float:
    """V de Cramer (CRAMER, 1946) entre duas variaveis nominais.

    Mede associacao onde o coeficiente de correlacao nao se aplica. Varia de
    0 (independencia) a 1 (associacao perfeita).

    Por padrao o nulo entra como categoria propria, e nao e descartado. A
    diferenca nao e cosmetica: em TIPO_ENTRETENIMENTO, cujos 29,5% de nulos sao
    exatamente os voos de conexao, descartar o nulo derruba o V de 0,105 para
    0,041 e esconde justamente o sinal que o item (d) da secao 4.2.1 identifica.
    Nulidade estrutural e informacao, nao ausencia dela.
    """
    if nulo_como_categoria:
        x = x.astype("object").where(x.notna(), ROTULO_NULO)
    tab = pd.crosstab(x, y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(tab)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def tabela_cramer(df, cols=None, alvo: str = "DETRATOR") -> pd.DataFrame:
    """Forca de associacao de cada categorica com o alvo, em ordem decrescente."""
    cols = cols or CAT_ASSOCIACAO
    linhas = [{"Variável": c, "V de Cramér": round(cramers_v(df[c], df[alvo]), 3)}
              for c in cols]
    return (pd.DataFrame(linhas)
              .sort_values("V de Cramér", ascending=False)
              .reset_index(drop=True))


def bruto_vs_ponderado(df, cortes=("FAIXA_ATRASO", "CANAL_COMPRA", "TIER_VIAGEM", "VOO_TIPO")):
    """Taxa de detracao bruta e pos-estratificada, por corte."""
    saida = {}
    for c in cortes:
        linhas = []
        for chave, g in df.groupby(c, observed=True):
            pond = (np.average(g["DETRATOR"], weights=g["PESO_POP"]) * 100
                    if g["PESO_POP"].sum() > 0 else np.nan)
            linhas.append({c: chave, "n": len(g),
                           "bruto": g["DETRATOR"].mean() * 100, "pond": pond})
        saida[c] = pd.DataFrame(linhas).set_index(c).round(2)
    return saida


def diagnostico_pesos(df, coluna: str = "PESO_POP") -> pd.Series:
    """Dispersao dos pesos e tamanho amostral efetivo de Kish.

    O n efetivo, (soma dos pesos)^2 / soma dos pesos ao quadrado, responde a
    pergunta que importa: a quantas observacoes sem peso equivale esta amostra
    ponderada. Quanto mais desiguais os pesos, menor ele fica, e maior a
    incerteza das metricas ponderadas.
    """
    w = df[coluna].to_numpy(dtype=float)
    if not np.isfinite(w).all():
        raise ValueError(f"{coluna} contem valor nao finito; corrija antes de diagnosticar.")
    n_efetivo = w.sum() ** 2 / np.square(w).sum()
    return pd.Series({
        "n": len(w),
        "Mínimo": w.min(),
        "P1": np.quantile(w, 0.01),
        "Mediana": np.median(w),
        "P99": np.quantile(w, 0.99),
        "Máximo": w.max(),
        "Razão P99/P1": np.quantile(w, 0.99) / np.quantile(w, 0.01),
        "n efetivo (Kish)": n_efetivo,
        "Perda de eficiência (%)": (1 - n_efetivo / len(w)) * 100,
    }).round(4)


def media_ponderada_ic(y, w, confianca: float = 0.95):
    """Media ponderada com intervalo de confianca por linearizacao.

    Usa o estimador de Hajek, com variancia
    soma(w^2 (y - media)^2) / (soma w)^2. Trata os pesos como fixos e ignora o
    desenho estratificado, portanto e aproximacao. Retorna (media, inferior,
    superior, erro padrao).
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    soma = w.sum()
    if soma <= 0:
        raise ValueError("A soma dos pesos e nula; nao ha estimador ponderado.")
    media = float(np.sum(w * y) / soma)
    erro = float(np.sqrt(np.sum(np.square(w) * np.square(y - media)) / soma ** 2))
    z = float(norm.ppf(0.5 + confianca / 2))
    return media, media - z * erro, media + z * erro, erro


def media_simples_ic(y, confianca: float = 0.95):
    """Media simples com intervalo de confianca, para comparar com a ponderada."""
    y = np.asarray(y, dtype=float)
    media = float(y.mean())
    erro = float(y.std(ddof=1) / np.sqrt(len(y)))
    z = float(norm.ppf(0.5 + confianca / 2))
    return media, media - z * erro, media + z * erro, erro


def comparar_bruto_ponderado(df, metricas=("DETRATOR", "NPS_PRINCIPAL"),
                             peso: str = "PESO_POP", confianca: float = 0.95) -> pd.DataFrame:
    """Metrica bruta e pos-estratificada, cada uma com intervalo de confianca.

    Sem o intervalo nao da para dizer se a diferenca entre bruto e ponderado e
    deslocamento real de composicao ou ruido amostral.
    """
    linhas = []
    for m in metricas:
        escala = 100 if m == "DETRATOR" else 1
        bruto = media_simples_ic(df[m], confianca)
        pond = media_ponderada_ic(df[m], df[peso], confianca)
        linhas.append({
            "Métrica": "Taxa de detratores (%)" if m == "DETRATOR" else m,
            "Bruto": bruto[0] * escala,
            "IC bruto inf": bruto[1] * escala,
            "IC bruto sup": bruto[2] * escala,
            "Ponderado": pond[0] * escala,
            "IC pond. inf": pond[1] * escala,
            "IC pond. sup": pond[2] * escala,
        })
    return pd.DataFrame(linhas).set_index("Métrica").round(3)


def ic_wilson(sucessos: int, n: int, confianca: float = 0.95):
    """Intervalo de Wilson para uma proporcao.

    Preferido ao intervalo normal porque se comporta bem em faixas pequenas e
    com proporcoes proximas de 0 ou 1, situacao das faixas extremas de
    antecedencia.
    """
    if n == 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.5 + confianca / 2))
    p = sucessos / n
    denominador = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / denominador
    margem = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominador
    return (centro - margem) * 100, (centro + margem) * 100


def tabela_antecedencia(df, confianca: float = 0.95) -> pd.DataFrame:
    """Detracao e NPS por faixa de antecedencia do aviso, com n e IC de Wilson.

    Reportar o n de cada faixa e obrigatorio: sem ele, a leitura da amplitude
    entre extremos ignora que as faixas nao tem o mesmo suporte amostral.
    """
    c = df[df["CANCELAMENTO_VOO"].astype(bool)].copy()
    c["FAIXA_ANTECEDENCIA"] = faixa_antecedencia(c["ANTECEDENCIA_CANCELAMENTO"])

    linhas = []
    for faixa, grupo in c.groupby("FAIXA_ANTECEDENCIA", observed=True):
        n = len(grupo)
        k = int(grupo["DETRATOR"].sum())
        inf, sup = ic_wilson(k, n, confianca)
        linhas.append({
            "Faixa": faixa,
            "n": n,
            "Taxa de detratores (%)": round(k / n * 100, 2),
            "IC 95%": f"[{inf:.2f}; {sup:.2f}]".replace(".", ","),
            "NPS médio": round(grupo["NPS_PRINCIPAL"].mean(), 1),
        })
    return pd.DataFrame(linhas).set_index("Faixa")


CONTROLES_ANTECEDENCIA = ("TIER_VIAGEM", "TRIMESTRE", "AEROPORTO_ORIGEM",
                          "VOO_TIPO", "CANAL_COMPRA")


def sensibilidade_antecedencia(df, controles=CONTROLES_ANTECEDENCIA,
                               minimo_por_estrato: int = 30) -> pd.DataFrame:
    """Diferenca entre aviso no mesmo dia e com mais de 60 dias, bruta e por estrato.

    Compara a diferenca observada com a media das diferencas calculadas DENTRO
    de cada nivel de uma variavel de controle, ponderada pelo tamanho do estrato.
    Se a diferenca encolhe muito ao controlar, parte da associacao vinha de
    composicao. Se persiste, a composicao naquela variavel nao a explica.

    Isto nao estabelece causalidade: controla apenas o que foi observado, e a
    causa do cancelamento, o fator de confusao mais provavel, nao esta na base.
    """
    c = df[df["CANCELAMENTO_VOO"].astype(bool)].copy()
    c["FAIXA_ANTECEDENCIA"] = faixa_antecedencia(c["ANTECEDENCIA_CANCELAMENTO"])
    extremos = c[c["FAIXA_ANTECEDENCIA"].isin(["Mesmo dia", "Mais de 60 d"])]

    def diferenca(bloco):
        taxas = bloco.groupby("FAIXA_ANTECEDENCIA", observed=True)["DETRATOR"].mean()
        if {"Mesmo dia", "Mais de 60 d"} - set(taxas.index):
            return None
        return (taxas["Mesmo dia"] - taxas["Mais de 60 d"]) * 100

    bruta = diferenca(extremos)
    linhas = [{"Controle": "Nenhum (diferença bruta)", "Estratos usados": 1,
               "n coberto": len(extremos), "Diferença (p.p.)": round(bruta, 2)}]

    for controle in controles:
        pesos, valores, n_total = [], [], 0
        for _, bloco in extremos.groupby(controle, observed=True):
            if len(bloco) < minimo_por_estrato:
                continue
            d = diferenca(bloco)
            if d is None:
                continue
            pesos.append(len(bloco))
            valores.append(d)
            n_total += len(bloco)
        if not valores:
            continue
        linhas.append({
            "Controle": controle,
            "Estratos usados": len(valores),
            "n coberto": n_total,
            "Diferença (p.p.)": round(float(np.average(valores, weights=pesos)), 2),
        })
    return pd.DataFrame(linhas).set_index("Controle")


def resumo_geral(df) -> dict:
    """Numeros de referencia citados no texto da secao 4.2.1."""
    return {
        "registros": len(df),
        "clientes_distintos": int(df["ID_GOLDENRECORD"].nunique()),
        "detracao_bruta": df["DETRATOR"].mean() * 100,
        "detracao_ponderada": np.average(df["DETRATOR"], weights=df["PESO_POP"]) * 100,
        "nps_bruto": df["NPS_PRINCIPAL"].mean(),
        "nps_ponderado": np.average(df["NPS_PRINCIPAL"], weights=df["PESO_POP"]),
    }


if __name__ == "__main__":
    import pickle

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 200)

    with open("out/clean.pkl", "rb") as fh:
        base = pickle.load(fh)

    print("=== TABELA NUMERICAS ===")
    print(tabela_numericas(base).to_string())
    print("\n=== TABELA CATEGORICAS ===")
    print(tabela_categoricas(base).to_string(index=False))
    print("\n=== V DE CRAMER CONTRA O ALVO ===")
    print(tabela_cramer(base).to_string(index=False))
    print("\n=== BRUTO vs PONDERADO ===")
    for corte, tabela in bruto_vs_ponderado(base).items():
        print(f"\n{corte}")
        print(tabela.to_string())
    print("\n=== DIAGNOSTICO DOS PESOS ===")
    print(diagnostico_pesos(base).to_string())
    print("\n=== BRUTO vs PONDERADO, COM INTERVALO DE CONFIANCA ===")
    print(comparar_bruto_ponderado(base).to_string())

"""Odds ratio da Regressao Logistica por feature original (card #212).

O coeficiente de uma Regressao Logistica e legivel, mas nao no espaco em que o
modelo o produz. Entre a feature do contrato e o coeficiente existem tres
transformacoes que mudam o que o numero significa, e ignorar qualquer uma delas
devolve um odds ratio valido e errado:

- **o `OneHotEncoder` quebra uma categorica em uma coluna por nivel.** Reportar
  `categoricas__CANAL_COMPRA_OTHER` sem dizer que a feature e `CANAL_COMPRA` e o
  nivel e `OTHER` entrega ao leitor o nome que a biblioteca inventou, nao a
  variavel que a operacao conhece;
- **o `SimpleImputer` acrescenta um indicador por coluna com ausencia.** Esse
  indicador nao mede a variavel: mede o fato de ela nao ter sido medida. Em
  `ANTECEDENCIA_CANCELAMENTO`, que so existe em voo cancelado, o indicador de
  ausencia e na pratica "o voo nao foi cancelado", e le-lo como se fosse a
  antecedencia inverteria o sentido;
- **o `RobustScaler` divide pelo IQR do treino.** `exp(coef)` passa a ser o odds
  ratio por IQR, e nao por unidade natural. Nesta base o IQR e 9 minutos em
  `ESTATISTICA_ATRASOSAIDA` e 140 em `TEMPO_VOO`; nas outras seis numericas ele
  e zero e o `scikit-learn` cai no fallback de 1,0, entao ali `exp(coef)` ja e
  por unidade. Usar a mesma leitura para as oito e o erro que passa despercebido.

**Ordenar por `exp(coef)` e o mesmo erro numa terceira forma.** Aquela coluna
mistura efeito por IQR, por minuto, por dia e por nivel, e comparar numeros em
unidades diferentes nao e ranking de efeito. `ANTECEDENCIA_CANCELAMENTO` e medida
em dias e vai de 0 a 400: um odds ratio de 0,98 por dia parece desprezivel e vale
0,29 no intervalo em que a variavel de fato varia. Por isso a tabela traz
`odds_ratio_comparavel`, o efeito de percorrer o intervalo interdecil observado no
treino (p10 a p90), e e por ele que a ordenacao acontece. O interdecil, e nao o
IQR, porque em quatro das oito numericas o IQR e zero e reduziria o efeito delas
a 1,0 por construcao.

Nao ha categoria de referencia: o `OneHotEncoder` do contrato e `drop=None`,
entao todos os niveis entram. O coeficiente de um nivel nao compara contra um
nivel-base ausente, e a comparacao que significa algo e entre niveis da mesma
variavel.

O mapeamento nao e feito por parsing do nome de saida. Ele e reconstruido da
estrutura ajustada do `ColumnTransformer` e depois **conferido** contra
`get_feature_names_out()`: se o `scikit-learn` mudar a convencao de nome, a
funcao levanta em vez de rotular a coluna errada em silencio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PASSO_PREPARO = "preparo"
PASSO_MODELO = "modelo"
PREFIXO_INDICADOR = "missingindicator_"


def _nome_esperado(bloco: str, feature: str, tipo: str, nivel: str | None) -> str:
    if tipo == "ausencia":
        return f"{bloco}__{PREFIXO_INDICADOR}{feature}"
    if tipo == "categorica":
        return f"{bloco}__{feature}_{nivel}"
    return f"{bloco}__{feature}"


def mapear_colunas(preprocessador) -> pd.DataFrame:
    """Cada coluna da matriz de volta para (feature original, tipo, nivel).

    Devolve uma linha por coluna de saida, na mesma ordem de
    `get_feature_names_out()`, com `feature_original`, `tipo` em
    {numerica, ausencia, categorica} e `nivel` preenchido so nas categoricas.
    """
    linhas = []
    for bloco, transformador, colunas in preprocessador.transformers_:
        if transformador == "drop" or not len(colunas):
            continue
        passos = getattr(transformador, "named_steps", {})
        codificador = passos.get("codificar")
        imputador = passos.get("imputar")

        if codificador is not None:
            for coluna, niveis in zip(colunas, codificador.categories_):
                for nivel in niveis:
                    linhas.append((bloco, coluna, "categorica", str(nivel)))
            continue

        linhas.extend((bloco, coluna, "numerica", None) for coluna in colunas)
        # O indicador vem depois das colunas imputadas, e so para as que tinham
        # ausencia no treino: `indicator_.features_` traz os indices dessas.
        if imputador is not None and getattr(imputador, "indicator_", None) is not None:
            linhas.extend((bloco, colunas[i], "ausencia", "valor ausente")
                          for i in imputador.indicator_.features_)

    tabela = pd.DataFrame(linhas, columns=["bloco", "feature_original", "tipo", "nivel"])
    esperado = [_nome_esperado(*linha) for linha in linhas]
    saida = list(preprocessador.get_feature_names_out())
    if esperado != saida:
        divergentes = [(e, s) for e, s in zip(esperado, saida) if e != s]
        raise ValueError(
            "o mapeamento nao reproduz get_feature_names_out(): "
            f"{len(saida)} colunas na saida, {len(esperado)} reconstruidas, "
            f"primeiras divergencias {divergentes[:3]}"
        )
    tabela.insert(0, "coluna_matriz", saida)
    return tabela.drop(columns="bloco")


def _suporte(x_treino, feature: str, tipo: str, nivel: str | None,
             preenchimento: str | None = None) -> int:
    """Linhas do treino que sustentam a coluna: nao nulas, ou nulas, ou do nivel.

    O nivel que o `SimpleImputer` inventa para as categoricas (`NAO_INFORMADO`)
    nao existe na coluna crua, entao contar ocorrencias dele devolveria zero e a
    tabela diria que o nivel nao tem suporte nenhum. As linhas que o sustentam sao
    justamente as ausentes.
    """
    coluna = x_treino[feature]
    if tipo == "ausencia":
        return int(coluna.isna().sum())
    if tipo == "categorica":
        if preenchimento is not None and str(nivel) == str(preenchimento):
            return int(coluna.isna().sum())
        return int((coluna.astype("string") == str(nivel)).sum())
    return int(coluna.notna().sum())


def _preenchimento_categorico(preprocessador) -> str | None:
    """Valor que o imputador das categoricas usa no lugar da ausencia."""
    for _, transformador, _ in preprocessador.transformers_:
        passos = getattr(transformador, "named_steps", {})
        if "codificar" in passos and "imputar" in passos:
            return getattr(passos["imputar"], "fill_value", None)
    return None


def _interdecil(x_treino, colunas) -> dict[str, float]:
    """Amplitude p10 a p90 dos valores observados, por coluna numerica.

    E a escala em que dois efeitos de variaveis diferentes se comparam: percorrer
    o intervalo em que a variavel varia de fato no treino. Amplitude zero cai em
    1,0, e ai o comparavel coincide com o efeito por unidade.
    """
    faixas = {}
    for coluna in colunas:
        observado = x_treino[coluna].dropna()
        p10, p90 = observado.quantile([0.10, 0.90]) if len(observado) else (0.0, 0.0)
        faixas[coluna] = float(p90 - p10) or 1.0
    return faixas


def _escalas(preprocessador) -> dict[str, float]:
    """Divisor que o `RobustScaler` aplicou, por coluna numerica.

    O campo na tabela se chama `escala_do_scaler`, e nao `iqr_do_treino`, porque
    nem sempre e o IQR: quando o interquartil da coluna e zero, o `scikit-learn`
    substitui por 1,0. Chamar de IQR esconderia justamente as colunas em que o
    escalonamento nao aconteceu.
    """
    for bloco, transformador, colunas in preprocessador.transformers_:
        passos = getattr(transformador, "named_steps", {})
        escalador = passos.get("escalar")
        if escalador is not None:
            # `scale_` cobre a saida do imputador, que traz as colunas e depois
            # os indicadores; so as primeiras correspondem as numericas.
            return dict(zip(colunas, escalador.scale_[:len(colunas)]))
    return {}


def odds_ratio(pipeline, x_treino, passo_preparo: str = PASSO_PREPARO,
               passo_modelo: str = PASSO_MODELO) -> pd.DataFrame:
    """Tabela de odds ratio do pipeline ajustado, por feature original.

    `x_treino` e a particao de treino **crua**, a mesma que ajustou o pipeline.
    Ela entra por duas coisas que a tabela nao teria como saber sozinha: a
    amplitude interdecil de cada numerica, que da a escala comparavel, e quantas
    linhas sustentam cada coluna.

    Tres leituras de efeito, e confundi-las e o erro que esta funcao existe para
    evitar:

    - `odds_ratio` e `exp(coef)`, o efeito de uma unidade **da matriz**;
    - `odds_ratio_por_unidade` desfaz o escalonamento e da o efeito de uma unidade
      **da variavel original**, so nas numericas;
    - `odds_ratio_comparavel` e o efeito de percorrer o intervalo p10 a p90 do
      treino nas numericas, e o proprio `odds_ratio` nas colunas 0/1. **E o unico
      dos tres que se compara entre variaveis**, e e por ele que a tabela ordena.

    `n_observado` traz o suporte: quantas linhas do treino tem valor naquela
    coluna, ou pertencem aquele nivel. Odds ratio grande sobre poucas linhas nao
    sustenta a mesma afirmacao que um sobre a base inteira.

    A ordenacao e por distancia de 1,0 em escala logaritmica, entao efeito forte
    de aumento e efeito forte de reducao aparecem juntos no topo.
    """
    preprocessador = pipeline.named_steps[passo_preparo]
    estimador = pipeline.named_steps[passo_modelo]
    coeficientes = np.asarray(estimador.coef_).ravel()

    tabela = mapear_colunas(preprocessador)
    if len(coeficientes) != len(tabela):
        raise ValueError(
            f"{len(coeficientes)} coeficientes para {len(tabela)} colunas mapeadas"
        )

    e_numerica = tabela["tipo"] == "numerica"
    numericas = tabela.loc[e_numerica, "feature_original"].unique()
    escalas = _escalas(preprocessador)
    faixas = _interdecil(x_treino, numericas)

    tabela["coeficiente"] = coeficientes
    tabela["escala_do_scaler"] = [escalas.get(f) if t == "numerica" else np.nan
                                  for f, t in zip(tabela["feature_original"], tabela["tipo"])]
    tabela["faixa_p10_p90"] = [faixas.get(f) if t == "numerica" else np.nan
                               for f, t in zip(tabela["feature_original"], tabela["tipo"])]
    preenchimento = _preenchimento_categorico(preprocessador)
    tabela["n_observado"] = [_suporte(x_treino, f, t, n, preenchimento) for f, t, n in
                             zip(tabela["feature_original"], tabela["tipo"], tabela["nivel"])]
    tabela["odds_ratio"] = np.exp(coeficientes)
    por_unidade = np.exp(coeficientes / tabela["escala_do_scaler"].fillna(1.0))
    tabela["odds_ratio_por_unidade"] = np.where(e_numerica, por_unidade, np.nan)
    tabela["odds_ratio_comparavel"] = np.where(
        e_numerica,
        np.exp(np.log(por_unidade) * tabela["faixa_p10_p90"].fillna(1.0)),
        tabela["odds_ratio"],
    )
    return (tabela.assign(_ordem=np.abs(np.log(tabela["odds_ratio_comparavel"])))
            .sort_values("_ordem", ascending=False)
            .drop(columns="_ordem").reset_index(drop=True))


def odds_ratio_de_cenario(pipeline, linhas) -> np.ndarray:
    """Odds ratio de cada linha contra a primeira, pelo `decision_function`.

    Somar coeficientes a mao para estimar o efeito de um bloco de colunas erra em
    silencio, e errou aqui. O `RobustScaler` **centra** cada numerica na mediana
    do treino, entao a coluna vale zero quando a variavel esta na mediana, e nao
    quando ela vale zero. Em `ANTECEDENCIA_CANCELAMENTO`, que o imputador preenche
    com a mediana em todo voo nao cancelado, somar `coef x mediana` conta duas
    vezes um deslocamento que o centro ja absorveu.

    Passar linhas inteiras pelo pipeline evita a conta a mao: o proprio objeto
    aplica imputacao, centro, escala e codificacao antes de somar. A primeira
    linha e a referencia e sai com 1,0.

    Para isolar um bloco de colunas, as linhas precisam ser iguais em todo o
    resto. Colunas que o contrato apaga junto com a mudanca, como as de voo num
    cancelamento, devem estar na mediana imputada **dos dois lados**, senao a
    razao carrega tambem a diferenca delas.
    """
    logitos = pipeline.decision_function(linhas)
    return np.exp(logitos - logitos[0])


def efeito_acumulado(odds_ratio_por_unidade: float, unidades: float) -> float:
    """Efeito de `unidades` da variavel original, composto multiplicativamente.

    Uma variavel cujo odds ratio por minuto e 1,0028 nao tem "efeito nenhum":
    tem 1,18 por hora de atraso. Reportar so o valor unitario de uma variavel
    medida em minutos esconde o efeito na escala em que a operacao decide.
    """
    return float(odds_ratio_por_unidade ** unidades)

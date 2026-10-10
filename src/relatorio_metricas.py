"""Relatorio de metricas do notebook integrado e comparacao com benchmarks.

O notebook `notebooks/comparacao_modelos.ipynb` mede seis modelos na particao de
validacao e calibra o Gradient Boosting. Este modulo recebe o que o notebook
exporta (ver `tests/test_notebook_integrado.py`) e compara os numeros com quatro
benchmarks, cada um respondendo a uma pergunta diferente:

1. **Metas de negocio** (Secao 4.1.3 da documentacao): cada candidato atinge a
   meta de cada metrica que tem meta?
2. **Piso de referencia** (Classe Majoritaria): cada candidato supera o piso em
   todas as metricas?
3. **Referencia constante** (probabilidade igual a prevalencia do treino): o
   Gradient Boosting calibrado tem Brier menor que ela?
4. **Registro anterior** (base real): os numeros reproduzem os valores
   registrados, dentro da tolerancia? E o teste de regressao do notebook.

O modulo nao treina modelo nem le dado: recebe dicionarios e devolve tabelas e
texto, o que o deixa testavel com numeros sinteticos. As metas e o sentido de
leitura de cada metrica nao sao redeclarados aqui; chegam do dicionario
`METRICAS` do proprio notebook, a fonte unica dessas definicoes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

TOLERANCIA_PADRAO = 5e-4

# Rotulos das linhas da tabela da Secao 6.1 do notebook.
NOME_GB = "Gradient Boosting"
NOME_GB_CALIBRADO = "Gradient Boosting calibrado"
NOME_REFERENCIA = "probabilidade constante"


def maior_e_melhor(sentido: str) -> bool:
    """Traduz o sentido de leitura de `METRICAS` ("maior é melhor" ou "menor é melhor")."""
    if sentido.startswith("maior"):
        return True
    if sentido.startswith("menor"):
        return False
    raise ValueError(f"sentido de leitura desconhecido: {sentido!r}")


def comparar_com_metas(resultados: dict, metricas: dict, nome_piso: str) -> pd.DataFrame:
    """Uma linha por candidato e metrica com meta: valor, meta, se atinge e a distancia.

    O piso de referencia fica de fora: ele nao e candidato e nao disputa meta.
    Metricas sem meta (o Brier) tambem ficam de fora.
    """
    linhas = []
    for modelo, valores in resultados.items():
        if modelo == nome_piso:
            continue
        for coluna, (nome, sentido, meta) in metricas.items():
            if meta is None:
                continue
            valor = valores[coluna]
            atinge = valor >= meta if maior_e_melhor(sentido) else valor <= meta
            linhas.append({"modelo": modelo, "metrica": nome, "valor": valor, "meta": meta,
                           "atinge_meta": bool(atinge), "distancia": valor - meta})
    return pd.DataFrame(linhas, columns=["modelo", "metrica", "valor", "meta", "atinge_meta",
                                         "distancia"])


def comparar_com_piso(resultados: dict, metricas: dict, nome_piso: str) -> pd.DataFrame:
    """Uma linha por candidato e metrica: valor, valor do piso e se o candidato o supera."""
    piso = resultados[nome_piso]
    linhas = []
    for modelo, valores in resultados.items():
        if modelo == nome_piso:
            continue
        for coluna, (nome, sentido, _) in metricas.items():
            valor, referencia = valores[coluna], piso[coluna]
            supera = valor > referencia if maior_e_melhor(sentido) else valor < referencia
            linhas.append({"modelo": modelo, "metrica": nome, "valor": valor,
                           "piso": referencia, "supera_piso": bool(supera)})
    return pd.DataFrame(linhas, columns=["modelo", "metrica", "valor", "piso", "supera_piso"])


def comparar_com_registro(
    medido: dict,
    registrado: dict,
    tolerancia: float = TOLERANCIA_PADRAO,
) -> pd.DataFrame:
    """Uma linha por modelo e metrica registrados, com a diferenca e se ela cabe na tolerancia.

    Um modelo ou metrica registrado que nao aparece no medido conta como fora da
    tolerancia: sumir da tabela tambem e uma regressao.
    """
    linhas = []
    for modelo, valores in registrado.items():
        for coluna, valor_registrado in valores.items():
            valor_medido = medido.get(modelo, {}).get(coluna)
            diferenca = None if valor_medido is None else valor_medido - valor_registrado
            linhas.append({
                "modelo": modelo, "metrica": coluna, "registrado": valor_registrado,
                "medido": valor_medido, "diferenca": diferenca,
                "dentro_da_tolerancia": diferenca is not None and abs(diferenca) <= tolerancia,
            })
    tabela = pd.DataFrame(linhas, columns=["modelo", "metrica", "registrado", "medido", "diferenca",
                                           "dentro_da_tolerancia"])
    # O pandas guardaria o valor ausente como NaN, que o JSON nao aceita e o relatorio
    # mostraria como "nan"; `None` mantem a ausencia explicita.
    for coluna in ("medido", "diferenca"):
        tabela[coluna] = tabela[coluna].astype(object).where(tabela[coluna].notna(), None)
    return tabela


def montar_relatorio(exportado: dict, registro: dict | None = None) -> dict:
    """Junta o que o notebook exportou e as quatro comparacoes num dicionario serializavel.

    `registro` e o benchmark versionado da base real (`tests/benchmarks/`). Com a
    base sintetica ele nao se aplica, porque os numeros sinteticos nao tem valor
    de referencia.
    """
    resultados, metricas, piso = exportado["resultados"], exportado["metricas"], exportado["nome_piso"]
    calibracao = exportado["calibracao"]
    comparacao_registro = None
    if registro is not None:
        medido = {**resultados, **{nome: valores for nome, valores in calibracao.items()
                                   if nome != NOME_GB}}
        tabela = comparar_com_registro(medido, registro["valores"], registro["tolerancia"])
        comparacao_registro = {
            "registrado_em": registro["registrado_em"],
            "tolerancia": registro["tolerancia"],
            "todos_dentro_da_tolerancia": bool(tabela["dentro_da_tolerancia"].all()),
            "linhas": tabela.to_dict(orient="records"),
        }

    return {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "base": "sintética" if exportado["base_sintetica"] else "real",
        "caminho_base": exportado["caminho_base"],
        "linhas": exportado["linhas"],
        "prevalencia": exportado["prevalencia"],
        "metrica_principal": exportado["metrica_principal"],
        "metricas": metricas,
        "nome_piso": piso,
        "ordem": exportado["ordem"],
        "resultados": resultados,
        "metas": comparar_com_metas(resultados, metricas, piso).to_dict(orient="records"),
        "piso": comparar_com_piso(resultados, metricas, piso).to_dict(orient="records"),
        "calibracao": {
            "tabela": calibracao,
            "brier_melhorou": exportado["brier_melhorou"],
            "ordem_preservada": exportado["ordem_preservada"],
            "score_medio": exportado["score_medio"],
            "supera_referencia_constante": bool(
                calibracao[NOME_GB_CALIBRADO]["brier"] < calibracao[NOME_REFERENCIA]["brier"]
            ),
        },
        "registro": comparacao_registro,
    }


def _numero(valor: float, casas: int = 4, sinal: bool = False) -> str:
    """Formata com virgula decimal, como o restante da documentacao do projeto.

    Um valor que arredonda para zero sai como `0,0000`, sem sinal: `-0,0000` sugeriria
    um desvio que nao existe.
    """
    if round(valor, casas) == 0:
        valor, sinal = 0.0, False
    return f"{valor:{'+' if sinal else ''}.{casas}f}".replace(".", ",")


def _milhar(valor: int) -> str:
    """Separador de milhar com ponto: 48301 vira 48.301."""
    return f"{valor:,}".replace(",", ".")


def _sim_nao(valor: bool) -> str:
    return "sim" if valor else "não"


def relatorio_em_markdown(relatorio: dict) -> str:
    """Relatorio legivel para revisores e parceiros, com as quatro comparacoes em tabelas."""
    metricas = relatorio["metricas"]
    nomes = {coluna: nome for coluna, (nome, _, _) in metricas.items()}
    piso = relatorio["nome_piso"]
    linhas_validacao = relatorio["linhas"]["validacao"]
    prevalencia = relatorio["prevalencia"]["validacao"]

    partes = [
        "# Relatório de métricas do notebook integrado",
        "",
        f"Gerado em {relatorio['gerado_em']} a partir de `notebooks/comparacao_modelos.ipynb`, "
        f"na base {relatorio['base']} (`{relatorio['caminho_base']}`). As métricas são da "
        f"partição de validação: {_milhar(linhas_validacao)} respostas, com prevalência de "
        f"Detrator de {_numero(prevalencia)}.",
    ]
    if relatorio["base"] == "sintética":
        partes += ["", "**Base sintética:** os números verificam a execução e não valem para "
                   "comparar modelos."]

    cabecalho = " | ".join(nomes[coluna] for coluna in metricas)
    partes += ["", "## Métricas por modelo", "",
               f"| Modelo | Papel | {cabecalho} |",
               "| --- | --- | " + " | ".join("---" for _ in metricas) + " |"]
    for modelo in relatorio["ordem"]:
        papel = "piso de referência" if modelo == piso else "candidato"
        valores = " | ".join(_numero(relatorio["resultados"][modelo][c]) for c in metricas)
        partes.append(f"| {modelo} | {papel} | {valores} |")

    partes += ["", "## Comparação com as metas de negócio", "",
               "| Modelo | Métrica | Valor | Meta | Atinge | Distância |",
               "| --- | --- | --- | --- | --- | --- |"]
    for linha in relatorio["metas"]:
        partes.append(f"| {linha['modelo']} | {linha['metrica']} | {_numero(linha['valor'])} | "
                      f"{_numero(linha['meta'], 2)} | {_sim_nao(linha['atinge_meta'])} | "
                      f"{_numero(linha['distancia'], sinal=True)} |")

    partes += ["", f"## Comparação com o piso de referência ({piso})", "",
               "| Modelo | Métrica | Valor | Piso | Supera o piso |",
               "| --- | --- | --- | --- | --- |"]
    for linha in relatorio["piso"]:
        partes.append(f"| {linha['modelo']} | {linha['metrica']} | {_numero(linha['valor'])} | "
                      f"{_numero(linha['piso'])} | {_sim_nao(linha['supera_piso'])} |")

    calibracao = relatorio["calibracao"]
    partes += ["", "## Calibração do Gradient Boosting", "",
               f"| Versão | {cabecalho} |",
               "| --- | " + " | ".join("---" for _ in metricas) + " |"]
    for versao, valores in calibracao["tabela"].items():
        partes.append(f"| {versao} | " + " | ".join(_numero(valores[c]) for c in metricas) + " |")
    score = calibracao["score_medio"]
    partes += [
        "",
        f"- Brier melhorou com a calibração: {_sim_nao(calibracao['brier_melhorou'])}.",
        f"- Ordem dos scores preservada (mesma fila): {_sim_nao(calibracao['ordem_preservada'])}.",
        f"- Brier calibrado abaixo da probabilidade constante: "
        f"{_sim_nao(calibracao['supera_referencia_constante'])}.",
        f"- Score médio: {_numero(score['sem_calibracao'])} sem calibração e "
        f"{_numero(score['calibrado'])} calibrado, para prevalência de {_numero(prevalencia)}.",
    ]

    partes += ["", "## Regressão contra o benchmark registrado", ""]
    registro = relatorio["registro"]
    if registro is None:
        partes.append("Não se aplica: o benchmark registrado vale só para a base real.")
    else:
        partes += [
            f"Valores registrados em {registro['registrado_em']}, com tolerância de "
            f"{_numero(registro['tolerancia'])}. Todos dentro da tolerância: "
            f"{_sim_nao(registro['todos_dentro_da_tolerancia'])}.",
            "",
            "| Modelo | Métrica | Registrado | Medido | Diferença | Dentro da tolerância |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for linha in registro["linhas"]:
            medido = "ausente" if linha["medido"] is None else _numero(linha["medido"])
            diferenca = "" if linha["diferenca"] is None else _numero(linha["diferenca"], 6, True)
            partes.append(f"| {linha['modelo']} | {nomes.get(linha['metrica'], linha['metrica'])} | "
                          f"{_numero(linha['registrado'])} | {medido} | {diferenca} | "
                          f"{_sim_nao(linha['dentro_da_tolerancia'])} |")
    return "\n".join(partes) + "\n"


def escrever_relatorio(relatorio: dict, pasta: Path | str, nome: str) -> tuple[Path, Path]:
    """Grava o relatorio em Markdown (leitura) e JSON (maquina) e devolve os dois caminhos."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_md, caminho_json = pasta / f"{nome}.md", pasta / f"{nome}.json"
    caminho_md.write_text(relatorio_em_markdown(relatorio), encoding="utf-8")
    caminho_json.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return caminho_md, caminho_json

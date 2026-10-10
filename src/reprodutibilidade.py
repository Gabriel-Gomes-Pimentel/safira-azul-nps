"""Reprodutibilidade do notebook integrado: log de metricas e comparacao entre execucoes.

Atende ao card "Testar reprodutibilidade dos modelos candidatos no notebook
integrado" (ART.7). Duas execucoes consecutivas do notebook, com a mesma
semente, precisam produzir:

- CR01: metricas identicas;
- CR02: logs de metricas iguais linha a linha;
- CR03: variacao nunca superior a 0,01% em nenhuma metrica de avaliacao.

O modulo nao executa o notebook. Recebe o que cada execucao exportou (ver
`tests/execucao_notebook.py`), escreve o log de cada uma, compara as execucoes e
monta o relatorio de comparacao. Por isso e testavel com numeros sinteticos.
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# CR03, em pontos percentuais: 0,01% da propria metrica.
LIMITE_PERCENTUAL = 0.01

# Tabelas de metricas exportadas pelo notebook: Secao 6 e Secao 6.1.
TABELAS = ("resultados", "calibracao")

DESCRICAO_CRITERIOS = {
    "CR01": "Métricas idênticas em execuções consecutivas com a mesma semente",
    "CR02": "Logs de métricas iguais linha a linha em todas as execuções",
    "CR03": f"Nenhuma variação superior a {LIMITE_PERCENTUAL:.2f}% nas métricas".replace(".", ","),
}


def linhas_de_log(exportado: dict) -> list[str]:
    """Log de metricas de uma execucao: uma linha por valor, em ordem fixa.

    Os floats saem com `repr`, que preserva todas as casas: duas execucoes so
    produzem a mesma linha se o valor for exatamente o mesmo. O log nao tem data
    nem hora, para que execucoes identicas gerem arquivos identicos (CR02).
    """
    base = "sintética" if exportado["base_sintetica"] else "real"
    linhas = [f"base | {base} | {exportado['caminho_base']}", f"semente | {exportado['semente']}"]
    for particao in ("treino", "validacao", "teste"):
        linhas.append(f"linhas | {particao} | {exportado['linhas'][particao]}")
        linhas.append(f"prevalencia | {particao} | {exportado['prevalencia'][particao]!r}")
    linhas.append("ordem | " + " > ".join(exportado["ordem"]))
    for tabela in TABELAS:
        for modelo in sorted(exportado[tabela]):
            for metrica in sorted(exportado[tabela][modelo]):
                linhas.append(f"{tabela} | {modelo} | {metrica} | "
                              f"{exportado[tabela][modelo][metrica]!r}")
    for versao in ("sem_calibracao", "calibrado"):
        linhas.append(f"score_medio | {versao} | {exportado['score_medio'][versao]!r}")
    return linhas


def _valores_comparaveis(exportado: dict) -> dict[tuple[str, str, str], float]:
    valores = {
        (tabela, modelo, metrica): valor
        for tabela in TABELAS
        for modelo, metricas in exportado[tabela].items()
        for metrica, valor in metricas.items()
    }
    for versao, valor in exportado["score_medio"].items():
        valores[("score_medio", "Gradient Boosting", versao)] = valor
    return valores


def comparar_execucoes(execucoes: list[dict]) -> pd.DataFrame:
    """Uma linha por metrica, com o valor de cada execucao e a variacao entre elas.

    A variacao percentual e a amplitude (maximo menos minimo) dividida pelo valor
    da primeira execucao. Uma metrica ausente em alguma execucao conta como
    variacao infinita: sumir tambem e nao reproduzir.
    """
    if len(execucoes) < 2:
        raise ValueError("a comparação exige pelo menos duas execuções")
    por_execucao = [_valores_comparaveis(execucao) for execucao in execucoes]
    chaves = sorted(set().union(*por_execucao))
    linhas = []
    for tabela, modelo, metrica in chaves:
        valores = [valores_execucao.get((tabela, modelo, metrica)) for valores_execucao in por_execucao]
        linha = {"tabela": tabela, "modelo": modelo, "metrica": metrica}
        linha.update({f"execucao_{i}": valor for i, valor in enumerate(valores, start=1)})
        if any(valor is None for valor in valores):
            linha.update(diferenca_absoluta=float("inf"), diferenca_percentual=float("inf"),
                         identico=False)
        else:
            amplitude = max(valores) - min(valores)
            referencia = abs(valores[0])
            percentual = 0.0 if amplitude == 0 else (
                float("inf") if referencia == 0 else amplitude / referencia * 100)
            linha.update(diferenca_absoluta=amplitude, diferenca_percentual=percentual,
                         identico=all(valor == valores[0] for valor in valores))
        linhas.append(linha)
    return pd.DataFrame(linhas)


def avaliar_criterios(tabela: pd.DataFrame, logs: list[list[str]],
                      limite_percentual: float = LIMITE_PERCENTUAL) -> dict:
    """Aplica CR01, CR02 e CR03 a comparacao e aos logs das execucoes."""
    return {
        "CR01": bool(tabela["identico"].all()),
        "CR02": all(log == logs[0] for log in logs[1:]),
        "CR03": bool((tabela["diferenca_percentual"] <= limite_percentual).all()),
        "maior_diferenca_percentual": float(tabela["diferenca_percentual"].max()),
        "limite_percentual": limite_percentual,
    }


def diferencas_entre_logs(logs: list[list[str]]) -> list[str]:
    """Diff unificado de cada log contra o da primeira execucao (vazio se todos iguais)."""
    diferencas = []
    for i, log in enumerate(logs[1:], start=2):
        diferencas += difflib.unified_diff(logs[0], log, fromfile="execucao_1.log",
                                           tofile=f"execucao_{i}.log", lineterm="")
    return diferencas


def _atende(valor: bool) -> str:
    return "atende" if valor else "não atende"


def _decimal(valor: float, casas: int) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def relatorio_em_markdown(execucoes: list[dict], logs: list[list[str]], tabela: pd.DataFrame,
                          criterios: dict, gerado_em: str | None = None) -> str:
    """Relatorio de comparacao para anexar ao card: criterios, sementes, metricas e diff."""
    primeira = execucoes[0]
    base = "sintética" if primeira["base_sintetica"] else "real"
    gerado_em = gerado_em or datetime.now().isoformat(timespec="seconds")
    partes = [
        "# Relatório de reprodutibilidade do notebook integrado",
        "",
        f"Gerado em {gerado_em}. O notebook `notebooks/comparacao_modelos.ipynb` foi executado "
        f"{len(execucoes)} vezes consecutivas, cada uma num kernel novo, na base {base} "
        f"(`{primeira['caminho_base']}`), com a semente `SEMENTE = {primeira['semente']}`.",
    ]
    if primeira["base_sintetica"]:
        partes += ["", "**Base sintética:** o teste verifica a reprodutibilidade da execução; "
                   "os valores das métricas não valem para comparar modelos."]

    maior = criterios["maior_diferenca_percentual"]
    partes += ["", "## Critérios de aceite", "", "| Critério | Descrição | Resultado |",
               "| --- | --- | --- |"]
    for codigo, descricao in DESCRICAO_CRITERIOS.items():
        detalhe = f" (maior variação: {_decimal(maior, 6)}%)" if codigo == "CR03" else ""
        partes.append(f"| {codigo} | {descricao} | {_atende(criterios[codigo])}{detalhe} |")

    partes += ["", "## Sementes dos modelos", "", "| Modelo | `random_state` |", "| --- | --- |"]
    for modelo, semente in primeira["random_state"].items():
        if modelo == primeira["nome_piso"]:
            valor = f"não se aplica (estratégia `{primeira['estrategia_piso']}`, determinística)"
        else:
            valor = str(semente)
        partes.append(f"| {modelo} | {valor} |")

    colunas = [f"execucao_{i}" for i in range(1, len(execucoes) + 1)]
    cabecalho = " | ".join(f"Execução {i}" for i in range(1, len(execucoes) + 1))
    partes += ["", "## Métricas por execução", "",
               f"| Tabela | Modelo | Métrica | {cabecalho} | Diferença (%) | Idêntico |",
               "| --- | --- | --- | " + " | ".join("---" for _ in colunas) + " | --- | --- |"]
    for linha in tabela.to_dict(orient="records"):
        valores = " | ".join("ausente" if linha[c] is None or pd.isna(linha[c]) else repr(linha[c])
                             for c in colunas)
        partes.append(f"| {linha['tabela']} | {linha['modelo']} | {linha['metrica']} | {valores} | "
                      f"{_decimal(linha['diferenca_percentual'], 6)} | "
                      f"{'sim' if linha['identico'] else 'não'} |")

    diferencas = diferencas_entre_logs(logs)
    partes += ["", "## Diferenças entre os logs", ""]
    if diferencas:
        partes += ["```diff", *diferencas, "```"]
    else:
        partes.append(f"Nenhuma: os {len(logs)} logs são idênticos linha a linha "
                      f"({len(logs[0])} linhas cada).")
    return "\n".join(partes) + "\n"


def escrever_evidencias(pasta: Path | str, execucoes: list[dict], logs: list[list[str]],
                        tabela: pd.DataFrame, criterios: dict) -> dict[str, Path]:
    """Grava um log por execucao, a comparacao em JSON e o relatorio em Markdown."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = {}
    for i, log in enumerate(logs, start=1):
        caminho = pasta / f"execucao_{i}.log"
        caminho.write_text("\n".join(log) + "\n", encoding="utf-8")
        caminhos[f"execucao_{i}"] = caminho
    caminhos["comparacao"] = pasta / "comparacao.json"
    caminhos["comparacao"].write_text(json.dumps(
        {"criterios": criterios, "metricas": tabela.to_dict(orient="records")},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    caminhos["relatorio"] = pasta / "relatorio_reprodutibilidade.md"
    caminhos["relatorio"].write_text(relatorio_em_markdown(execucoes, logs, tabela, criterios),
                                     encoding="utf-8")
    return caminhos

"""SAFIRA | Validação do corte de leakage temporal (#198).

Divide a base com `split.dividir`, a mesma função que a modelagem usa, e
confere nas partições resultantes as duas garantias contra vazamento:

- CR01: toda data de um conjunto posterior é estritamente maior que toda data
  do conjunto anterior (treino < validação < teste).
- CR02: nenhum `RESPONDENT_ID` nem `ID_GOLDENRECORD` aparece em dois conjuntos.

Diferente de `split.conferir`, que para na primeira falha, aqui todas as
verificações rodam e o resultado de cada uma vai para o relatório (CR03). A
conferência é feita sobre as datas e IDs reais das partições, e não sobre os
parâmetros de corte, pelo mesmo motivo registrado em `conferir`: um filtro
errado deixaria o parâmetro coerente e a partição vazada.

Por padrão roda sobre a base dummy, que não contém dado do parceiro e por isso
o relatório pode ser versionado. Com `--base` roda sobre qualquer base
analítica, inclusive a real.

Executar com:  python scripts/validar_leakage.py
"""
from __future__ import annotations

import sys
from datetime import date
from itertools import combinations
from pathlib import Path

import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
for _pasta in ("src", "scripts"):
    _caminho = str(_RAIZ / _pasta)
    if _caminho not in sys.path:
        sys.path.insert(0, _caminho)

import split  # noqa: E402
from gerar_dummy import BASE_ANALITICA_DUMMY, SAIDA_PADRAO, gerar  # noqa: E402

BASE_PADRAO = _RAIZ / SAIDA_PADRAO / BASE_ANALITICA_DUMMY
RELATORIO_PADRAO = _RAIZ / "documents" / "extras" / "relatorio-validacao-leakage.md"
# Os mesmos cortes da política de particionamento temporal do projeto.
CORTE_VALIDACAO, CORTE_TESTE = "2025-07-01", "2026-01-01"
COLUNAS_ID = (split.COLUNA_ORDEM, split.COLUNA_CLIENTE)


def verificar(particoes: dict[str, pd.DataFrame],
              coluna_data: str = split.COLUNA_DATA,
              colunas_id: tuple[str, ...] = COLUNAS_ID) -> list[dict]:
    """Devolve uma linha por verificação: critério, par de conjuntos, ok e detalhe."""
    resultados = []
    datas = {n: pd.to_datetime(p[coluna_data], errors="coerce").dropna()
             for n, p in particoes.items()}

    # combinations segue a ordem de PARTICOES, então `a` é sempre o anterior.
    for a, b in combinations(split.PARTICOES, 2):
        if datas[a].empty or datas[b].empty:
            ok, detalhe = False, f"sem data para comparar em {a if datas[a].empty else b}"
        else:
            fim, inicio = datas[a].max(), datas[b].min()
            ok = fim < inicio
            detalhe = f"max({a}) = {fim.date()} {'<' if ok else '>='} min({b}) = {inicio.date()}"
        resultados.append({"criterio": "CR01", "verificacao": f"{a} antes de {b}",
                           "ok": ok, "detalhe": detalhe})

    for coluna in colunas_id:
        for a, b in combinations(split.PARTICOES, 2):
            comuns = set(particoes[a][coluna].dropna()) & set(particoes[b][coluna].dropna())
            resultados.append({"criterio": "CR02", "verificacao": f"{coluna}: {a} x {b}",
                               "ok": not comuns,
                               "detalhe": f"{len(comuns)} ID(s) em comum"})
    return resultados


def relatorio(particoes: dict[str, pd.DataFrame], metadados: dict,
              resultados: list[dict], origem: str) -> str:
    """Relatório em Markdown com a composição dos conjuntos e cada verificação."""
    aprovado = all(r["ok"] for r in resultados)
    tabela = split.resumo(particoes, metadados=metadados)
    linhas = [
        "# Relatório de validação do corte de leakage",
        "",
        f"Gerado por `scripts/validar_leakage.py` em {date.today()} (card #198).",
        "",
        f"- Base: `{origem}` ({metadados['linhas_totais']} linhas)",
        f"- Cortes: validação a partir de {metadados['corte_validacao']}, "
        f"teste a partir de {metadados['corte_teste']}",
        f"- Resultado: **{'APROVADO' if aprovado else 'REPROVADO'}** "
        f"({sum(r['ok'] for r in resultados)}/{len(resultados)} verificações)",
        "",
        "## Composição dos conjuntos",
        "",
        "| Conjunto | n | Clientes | Início | Fim | Taxa de detração (%) |",
        "|---|---|---|---|---|---|",
        *(f"| {n} | {r.n} | {r.clientes} | {r.data_inicio} | {r.data_fim} "
          f"| {r.get('taxa_alvo_pct', '-')} |" for n, r in tabela.iterrows()),
        "",
        f"Fora dos conjuntos: {metadados['linhas_removidas_por_recorrencia']} linha(s) "
        f"por Cliente recorrente, {metadados['linhas_sem_cliente_excluidas']} sem Cliente "
        f"e {metadados['linhas_sem_data_excluidas']} sem data.",
        "",
        "## Verificações",
        "",
        "CR01: a maior data do conjunto anterior deve ser menor que a menor data do "
        "posterior. CR02: a interseção de IDs entre conjuntos deve ser vazia.",
        "",
        "| Critério | Verificação | Resultado | Detalhe |",
        "|---|---|---|---|",
        *(f"| {r['criterio']} | {r['verificacao']} | {'ok' if r['ok'] else 'FALHOU'} "
          f"| {r['detalhe']} |" for r in resultados),
        "",
    ]
    return "\n".join(linhas)


def validar(base: Path | str = BASE_PADRAO, saida: Path | str = RELATORIO_PADRAO,
            corte_validacao: str = CORTE_VALIDACAO,
            corte_teste: str = CORTE_TESTE) -> bool:
    """Divide a base, verifica, escreve o relatório e devolve se passou."""
    base = Path(base)
    if not base.exists() and base == BASE_PADRAO:
        gerar(base.parent)
    particoes, metadados = split.dividir(pd.read_parquet(base), corte_validacao, corte_teste)
    resultados = verificar(particoes)
    origem = base.relative_to(_RAIZ) if base.is_relative_to(_RAIZ) else base
    Path(saida).write_text(relatorio(particoes, metadados, resultados, str(origem)),
                           encoding="utf-8")
    return all(r["ok"] for r in resultados)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default=BASE_PADRAO, type=Path)
    p.add_argument("--saida", default=RELATORIO_PADRAO, type=Path)
    args = p.parse_args()

    ok = validar(args.base.resolve(), args.saida)
    print(f"{'APROVADO' if ok else 'REPROVADO'}: relatório em {args.saida}")
    raise SystemExit(0 if ok else 1)

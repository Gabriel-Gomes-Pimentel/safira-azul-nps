"""Busca aleatoria do Gradient Boosting com validacao agrupada por Cliente (#190).

**O que este modulo resolve.** O #188 entregou o pipeline do Gradient Boosting
(`HistGradientBoostingClassifier`, decidido pelo #185) sobre o pre-processador do
contrato, com os hiperparametros padrao da biblioteca, e o #186 declarou o espaco
em que eles podem variar. Este modulo junta os dois numa `RandomizedSearchCV` e
grava os hiperparametros vencedores de um jeito que possa ser versionado e
reconstruido: o modelo treinado nao vai para o git, por tamanho, e so e
comparavel entre duplas se qualquer pessoa conseguir remonta-lo a partir de um
arquivo. Quem remonta e `ensembles.melhor_gradient_boosting`, a funcao que a dupla
de Metricas e Decisoes consome.

**A trava que justifica o card: a validacao e a do contrato, agrupada por
Cliente.** A base tem 407.139 `ID_GOLDENRECORD` em 484.915 respostas, entao o
mesmo Cliente aparece em varios voos. Passar `cv` como inteiro seria o atalho
mais comodo e o mais perigoso: num classificador, o scikit-learn converte o
inteiro num particionador estratificado que ignora `groups`, e o mesmo Cliente
cairia no ajuste e na validacao do mesmo fold. O modelo seria premiado por
reconhecer a pessoa, e os hiperparametros vencedores seriam os que mais
memorizam. E o mesmo motivo do #189. Por isso `criar_busca_gradient_boosting`
recusa `cv` inteiro e so aceita os folds de `validacao.criar_folds`, conferidos
por `validacao.conferir_folds` antes da busca, e `executar_busca` repassa
`groups` ao `fit` (CR02).

O que este modulo deliberadamente **nao** faz:

- **nao declara o espaco de busca.** Ele e o `ESPACO_GRADIENT_BOOSTING_HISTGB`
  do #186, e entra como argumento. Redeclara-lo aqui criaria uma segunda
  definicao livre para divergir;
- **nao define a metrica.** `scoring` e obrigatorio e sem padrao: e o
  `scorer_f2` do #242, o criterio de busca que a dupla de Metricas e Decisoes
  fixou no #238. Um padrao aqui seria a porta para cada dupla buscar por um
  criterio diferente;
- **nao mede o numero oficial.** Quem produz o numero reportado e `avaliar`, do
  #241, sobre o estimador reconstruido pelo JSON (CR05).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV

# O modulo localiza os proprios vizinhos em vez de depender de quem o importa,
# pelo mesmo motivo de `validacao.py`: sob pytest o conftest prepara o caminho,
# no Colab e na execucao direta nao ha ninguem para preparar.
_RAIZ = Path(__file__).resolve().parents[1]
for _pasta in ("src", "scripts"):
    _caminho = str(_RAIZ / _pasta)
    if _caminho not in sys.path:
        sys.path.insert(0, _caminho)

import matriz  # noqa: E402
import validacao  # noqa: E402
from ensembles import (  # noqa: E402
    ARQUIVO_HIPERPARAMETROS_GRADIENT_BOOSTING,
    PASSO_MODELO,
    SEMENTE_PADRAO,
    criar_pipeline_gradient_boosting,
)
from espaco_busca_logistica import grade_com_prefixo  # noqa: E402

# O card pede no minimo 40 iteracoes. Com seis eixos no espaco do #186, menos do
# que isso cobre tao pouco do espaco que o vencedor diria mais sobre a semente do
# sorteio do que sobre o Gradient Boosting.
N_ITER_MINIMO = 40
N_ITER_PADRAO = N_ITER_MINIMO

# Artefatos versionados. Ficam em `documents/extras/resultados/`, ao lado de
# `hiperparametros_candidato.json` do #103, e sao JSON, e nao CSV: o `.gitignore`
# do projeto proibe `*.csv` por compromisso com o parceiro, e a regra vale mesmo
# para um arquivo que so tem parametros e metricas. O dos hiperparametros e o
# mesmo que `ensembles.melhor_gradient_boosting` le: um caminho so, declarado la.
ARQUIVO_HIPERPARAMETROS = ARQUIVO_HIPERPARAMETROS_GRADIENT_BOOSTING
ARQUIVO_RESUMO_CV = ARQUIVO_HIPERPARAMETROS.parent / "cv_resultados_gradient_boosting.json"

# Colunas de `cv_results_` que vao para o resumo. Sao so parametros e agregados
# por combinacao: nenhuma linha de Cliente, nenhum indice de fold e nenhuma
# previsao entram no arquivo versionado.
COLUNAS_RESUMO = ("mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time")

# Fracao da amplitude de um intervalo continuo dentro da qual o vencedor conta
# como "no limite". Cinco por cento de 0,0 a 2,0 em `l2_regularization` e 0,1:
# um vencedor em 1,95 ja diz que a busca empurrava para o teto.
FRACAO_BORDA = 0.05


def criar_busca_gradient_boosting(
    preprocessador,
    espaco: dict,
    folds: list[tuple[np.ndarray, np.ndarray]],
    scoring,
    n_iter: int = N_ITER_PADRAO,
    random_state: int = SEMENTE_PADRAO,
    n_jobs: int | None = None,
) -> RandomizedSearchCV:
    """Monta a `RandomizedSearchCV` sobre o pipeline do #188 e o espaco do #186.

    `folds` precisa ser a lista de `validacao.criar_folds`. Um inteiro e recusado
    com `TypeError`, pelo motivo da docstring do modulo: viraria um particionador
    estratificado que ignora o Cliente. Qualquer outro iteravel de pares
    `(treino, validacao)` e aceito, mas e o chamador quem responde por ele ter
    vindo do contrato; `executar_busca` confere os folds antes de ajustar.

    `n_iter` abaixo de `N_ITER_MINIMO` e recusado (CR01). `random_state` fixa o
    sorteio das combinacoes **e** a semente do estimador, para que duas execucoes
    da mesma busca sorteiem as mesmas combinacoes e ajustem os mesmos ensembles.

    `refit=True` reajusta o vencedor no treino inteiro, e e esse ajuste que o
    notebook compara com o estimador reconstruido pelo JSON.

    `n_jobs` e o paralelismo **da busca**: cada processo ajusta um par
    combinacao e fold inteiro, e o resultado nao depende da ordem em que eles
    terminam.
    """
    if isinstance(folds, (int, np.integer)):
        raise TypeError(
            "cv inteiro recusado: num classificador ele vira um particionador "
            "estratificado que ignora o Cliente. Passe validacao.criar_folds(...)."
        )
    if n_iter < N_ITER_MINIMO:
        raise ValueError(f"n_iter={n_iter} abaixo do minimo de {N_ITER_MINIMO} do card #190.")
    if scoring is None:
        raise ValueError("scoring e obrigatorio: use o scorer_f2 do #242.")

    pipeline = criar_pipeline_gradient_boosting(preprocessador, random_state=random_state)
    return RandomizedSearchCV(
        pipeline,
        param_distributions=grade_com_prefixo(PASSO_MODELO, espaco),
        n_iter=n_iter,
        scoring=scoring,
        cv=list(folds),
        refit=True,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=False,
    )


def executar_busca(busca: RandomizedSearchCV, x_treino, y_treino, groups) -> dict[str, object]:
    """Confere os folds contra o Cliente, ajusta a busca e relata o tempo total.

    A conferencia roda antes do `fit` porque e barata e a busca nao: descobrir
    depois de minutos que um fold misturava Cliente jogaria a busca inteira fora.
    `groups` vai ao `fit` como o card pede (CR02). Com `cv` ja materializado em
    pares de indices o scikit-learn nao precisa dele para particionar, mas
    passa-lo mantem a chamada correta se alguem trocar a lista por um
    particionador agrupado.

    `x_treino` e a matriz **crua** do treino: e o pipeline que transforma,
    dentro de cada fold. `groups` e `preparo["grupos"]["treino"]`, o
    `ID_GOLDENRECORD` alinhado a `x_treino`.
    """
    validacao.conferir_folds(busca.cv, groups)

    inicio = time.perf_counter()
    busca.fit(x_treino, y_treino, groups=groups)
    tempo_total_s = time.perf_counter() - inicio

    n_combinacoes = len(busca.cv_results_["params"])
    return {
        "tempo_total_s": tempo_total_s,
        "n_combinacoes": n_combinacoes,
        "n_folds": busca.n_splits_,
        "n_ajustes": n_combinacoes * busca.n_splits_ + 1,
        "melhor_score_medio": float(busca.best_score_),
    }


def _nativo(valor):
    """Converte tipos do numpy para o que o `json` serializa sem `default=str`.

    `default=str` gravaria `np.int64(412)` como a string `"412"`, e o estimador
    reconstruido receberia texto onde esperava inteiro.

    `NaN` vira `None`. Ao montar o `DataFrame` do `cv_results_`, o pandas troca o
    `class_weight=None` sorteado por `NaN`, e o `json` gravaria `NaN`, que nem e
    JSON valido e seria lido como "sem valor" em vez de "sem reponderacao".
    """
    if isinstance(valor, np.generic):
        valor = valor.item()
    if isinstance(valor, float) and np.isnan(valor):
        return None
    return valor


def hiperparametros_vencedores(busca: RandomizedSearchCV) -> dict[str, object]:
    """Os parametros do vencedor sem o prefixo do passo, prontos para o JSON."""
    prefixo = f"{PASSO_MODELO}__"
    return {
        chave.removeprefix(prefixo): _nativo(valor)
        for chave, valor in sorted(busca.best_params_.items())
    }


def resumir_cv_results(busca: RandomizedSearchCV) -> pd.DataFrame:
    """Uma linha por combinacao, so com parametros e agregados, ordenada pelo rank.

    Sao descartadas as colunas por fold (`split0_test_score`, ...) e os tempos de
    pontuacao: nao carregam dado de Cliente, mas tambem nao ajudam a leitura, e
    manter o arquivo pequeno facilita revisar o diff.
    """
    resultados = pd.DataFrame(busca.cv_results_)
    prefixo = f"param_{PASSO_MODELO}__"
    colunas_param = [c for c in resultados.columns if c.startswith(prefixo)]
    resumo = resultados[colunas_param + list(COLUNAS_RESUMO)].copy()
    resumo.columns = [c.removeprefix(prefixo) for c in colunas_param] + list(COLUNAS_RESUMO)
    return resumo.sort_values("rank_test_score", kind="stable").reset_index(drop=True)


def salvar_resultados(
    busca: RandomizedSearchCV,
    relato: dict[str, object],
    random_state: int = SEMENTE_PADRAO,
    caminho_hiperparametros=None,
    caminho_resumo=None,
    *,
    cortes: dict[str, str],
) -> tuple[Path, Path]:
    """Grava os hiperparametros vencedores e o resumo do `cv_results_` em JSON.

    O arquivo de hiperparametros carrega tambem o `random_state`, o `n_iter`, o
    numero de folds e o tempo total: sem eles, o JSON reconstruiria o estimador
    mas nao diria de que busca ele saiu.

    `cortes` e obrigatorio e vem de `matriz.cortes_do_preparo(preparo)`: grava as
    duas datas da matriz em que a busca rodou. Sem elas, quem reconstroi o
    vencedor nao tem como saber se os hiperparametros foram escolhidos na mesma
    matriz em que ele vai ser medido, e a secao 9.1 do notebook confere isso.

    Sem caminho explicito, os arquivos vao para `ARQUIVO_HIPERPARAMETROS` e
    `ARQUIVO_RESUMO_CV`, lidos na hora da chamada, e nao fixados na definicao da
    funcao, para que o notebook e esta funcao nunca apontem para arquivos
    diferentes.
    """
    caminho_hiperparametros = Path(caminho_hiperparametros or ARQUIVO_HIPERPARAMETROS)
    caminho_resumo = Path(caminho_resumo or ARQUIVO_RESUMO_CV)
    caminho_hiperparametros.parent.mkdir(parents=True, exist_ok=True)

    registro = {
        "hiperparametros": hiperparametros_vencedores(busca),
        "random_state": int(random_state),
        "n_iter": int(busca.n_iter),
        "n_folds": int(relato["n_folds"]),
        "melhor_score_medio": float(relato["melhor_score_medio"]),
        "tempo_total_s": round(float(relato["tempo_total_s"]), 1),
        **matriz.conferir_cortes(cortes),
    }
    caminho_hiperparametros.write_text(
        json.dumps(registro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    resumo = resumir_cv_results(busca)
    linhas = [{chave: _nativo(valor) for chave, valor in linha.items()} for linha in resumo.to_dict("records")]
    caminho_resumo.write_text(
        json.dumps(linhas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return caminho_hiperparametros, caminho_resumo


def parametros_no_limite(
    hiperparametros: dict[str, object],
    espaco: dict,
    tolerancia: float = FRACAO_BORDA,
) -> list[str]:
    """Nomes dos parametros vencedores que ficaram na borda do intervalo buscado.

    Um vencedor no limite precisa ser comentado: se o melhor `max_iter` e o teto
    do intervalo, a busca provavelmente queria ir alem dele, e o intervalo do
    #186 fica sob suspeita.

    Distribuicao discreta do scipy (`randint`) conta como borda so no piso ou no
    teto exatos. Distribuicao continua (`uniform`) nunca sorteia a borda exata,
    entao conta como borda quem cair a menos de `tolerancia` da amplitude de um
    dos extremos. Em `loguniform`, a do `learning_rate`, a amplitude e medida na
    escala log, a mesma em que o #186 sorteia: na escala linear, 5% de 0,01 a
    0,3 poria 0,024 "no piso", quando ele ja esta a quase um terco do caminho
    em ordens de grandeza. Listas so tem borda se forem de numeros; categorias
    como `class_weight` ficam de fora, porque nao ha "alem" de uma categoria.
    """
    no_limite = []
    for nome, valor in hiperparametros.items():
        distribuicao = espaco.get(nome)
        if distribuicao is None or valor is None or isinstance(valor, (str, bool)):
            continue
        if hasattr(distribuicao, "support"):
            piso, teto = (float(v) for v in distribuicao.support())
            if isinstance(distribuicao.dist, type(stats.loguniform)):
                piso, teto, valor = np.log(piso), np.log(teto), np.log(valor)
            margem = 0.0 if isinstance(distribuicao.dist, stats.rv_discrete) else tolerancia * (teto - piso)
        else:
            numericos = [v for v in distribuicao if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not numericos:
                continue
            piso, teto, margem = float(min(numericos)), float(max(numericos)), 0.0
        if valor <= piso + margem or valor >= teto - margem:
            no_limite.append(nome)
    return no_limite

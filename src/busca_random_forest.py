"""Busca aleatoria do Random Forest com validacao agrupada por Cliente (card 08B, #189).

**O que este modulo resolve.** O #187 entregou o pipeline do Random Forest sobre
o pre-processador do contrato, com os hiperparametros padrao da biblioteca, e o
#186 declarou o espaco em que eles podem variar. Este modulo junta os dois numa
`RandomizedSearchCV` e devolve os hiperparametros vencedores de um jeito que
possa ser versionado e reconstruido: o modelo treinado nao vai para o git, por
tamanho, e so e comparavel entre duplas se qualquer pessoa conseguir remonta-lo a
partir de um arquivo.

**A trava que justifica o card: a validacao e a do contrato, agrupada por
Cliente.** A base tem 407.139 `ID_GOLDENRECORD` em 484.915 respostas, entao o
mesmo Cliente aparece em varios voos. Passar `cv` como inteiro seria o atalho
mais comodo e o mais perigoso: num classificador, o scikit-learn converte o
inteiro num particionador estratificado que ignora `groups`, e o mesmo Cliente
cairia no ajuste e na validacao do mesmo fold. O modelo seria premiado por
reconhecer a pessoa, e os hiperparametros vencedores seriam os que mais
memorizam. Por isso `criar_busca_random_forest` recusa `cv` inteiro e so aceita
os folds de `validacao.criar_folds`, conferidos por `validacao.conferir_folds`
antes da busca, e `executar_busca` repassa `groups` ao `fit` (CR02).

O que este modulo deliberadamente **nao** faz:

- **nao declara o espaco de busca.** Ele e o `ESPACO_RANDOM_FOREST` do #186, e
  entra como argumento. Redeclara-lo aqui criaria uma segunda definicao livre
  para divergir;
- **nao define a metrica.** `scoring` e obrigatorio e sem padrao: e o
  `scorer_f2` do card 05A.2 (#242). Um padrao aqui seria a porta para cada dupla
  buscar por um criterio diferente;
- **nao mede o numero oficial.** Quem produz o numero reportado e `avaliar`, do
  card 05A.1 (#241), sobre o estimador reconstruido pelo JSON (CR04).
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
# pelo mesmo motivo de `ensembles.py`: sob pytest o conftest prepara o caminho,
# no Colab e na execucao direta nao ha ninguem para preparar.
_RAIZ = Path(__file__).resolve().parents[1]
for _pasta in ("src", "scripts"):
    _caminho = str(_RAIZ / _pasta)
    if _caminho not in sys.path:
        sys.path.insert(0, _caminho)

import matriz  # noqa: E402
import validacao  # noqa: E402
from ensembles import PASSO_MODELO, SEMENTE_PADRAO, criar_pipeline_random_forest  # noqa: E402
from espaco_busca_logistica import grade_com_prefixo  # noqa: E402

# O card pede no minimo 40 iteracoes. Abaixo disso, com cinco eixos no espaco
# do #186, a busca cobre tao pouco do espaco que o vencedor diria mais sobre a
# semente do sorteio do que sobre o Random Forest.
N_ITER_MINIMO = 40
N_ITER_PADRAO = N_ITER_MINIMO

# Artefatos versionados. Ficam em `documents/extras/resultados/`, ao lado de
# `hiperparametros_candidato.json` do #103, e sao JSON, e nao CSV: o `.gitignore`
# do projeto proibe `*.csv` por compromisso com o parceiro, e a regra vale mesmo
# para um arquivo que so tem parametros e metricas.
_RESULTADOS = _RAIZ / "documents" / "extras" / "resultados"
ARQUIVO_HIPERPARAMETROS = _RESULTADOS / "hiperparametros_random_forest.json"
ARQUIVO_RESUMO_CV = _RESULTADOS / "cv_resultados_random_forest.json"

# Colunas de `cv_results_` que vao para o resumo. Sao so parametros e
# agregados por combinacao: nenhuma linha de Cliente, nenhum indice de fold e
# nenhuma previsao entram no arquivo versionado.
COLUNAS_RESUMO = ("mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time")

# Fracao da amplitude de um intervalo continuo dentro da qual o vencedor conta
# como "no limite". Cinco por cento de 0,3 a 1,0 em `max_features` e 0,035: um
# vencedor em 0,97 ja diz que a busca empurrava para o teto.
FRACAO_BORDA = 0.05


def criar_busca_random_forest(
    preprocessador,
    espaco: dict,
    folds: list[tuple[np.ndarray, np.ndarray]],
    scoring,
    n_iter: int = N_ITER_PADRAO,
    random_state: int = SEMENTE_PADRAO,
    n_jobs: int | None = None,
) -> RandomizedSearchCV:
    """Monta a `RandomizedSearchCV` sobre o pipeline do #187 e o espaco do #186.

    `folds` precisa ser a lista de `validacao.criar_folds`. Um inteiro e recusado
    com `TypeError`, pelo motivo da docstring do modulo: viraria um particionador
    estratificado que ignora o Cliente. Qualquer outro iteravel de pares
    `(treino, validacao)` e aceito, mas e o chamador quem responde por ele ter
    vindo do contrato; `executar_busca` confere os folds antes de ajustar.

    `n_iter` abaixo de `N_ITER_MINIMO` e recusado (CR01). `random_state` fixa o
    sorteio das combinacoes **e** a semente do estimador, para que duas execucoes
    da mesma busca sorteiem as mesmas combinacoes e ajustem as mesmas florestas.

    `refit=True` reajusta o vencedor no treino inteiro, e e esse ajuste que o
    notebook compara com o estimador reconstruido pelo JSON.

    `n_jobs` e o paralelismo **da busca**: cada processo ajusta um par
    combinacao e fold inteiro, e o resultado nao depende da ordem em que eles
    terminam. A floresta de cada ajuste roda com o `n_jobs=1` do #187, porque
    threads dentro dela mudariam o ultimo bit de `predict_proba` e, num empate
    em 0,5, o rotulo. No Colab, `n_jobs=-1` usa os nucleos da sessao sem abrir
    mao da reprodutibilidade.
    """
    if isinstance(folds, (int, np.integer)):
        raise TypeError(
            "cv inteiro recusado: num classificador ele vira um particionador "
            "estratificado que ignora o Cliente. Passe validacao.criar_folds(...)."
        )
    if n_iter < N_ITER_MINIMO:
        raise ValueError(f"n_iter={n_iter} abaixo do minimo de {N_ITER_MINIMO} do card #189.")
    if scoring is None:
        raise ValueError("scoring e obrigatorio: use o scorer_f2 do card 05A.2 (#242).")

    pipeline = criar_pipeline_random_forest(preprocessador, random_state=random_state)
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


def executar_busca(busca: RandomizedSearchCV, x_treino, y_treino, grupos_treino) -> dict[str, object]:
    """Confere os folds contra o Cliente, ajusta a busca e relata o tempo total.

    A conferencia roda antes do `fit` porque e barata e a busca nao: descobrir
    depois de horas que um fold misturava Cliente jogaria a busca inteira fora.
    `groups` vai ao `fit` como o card pede (CR02). Com `cv` ja materializado em
    pares de indices o scikit-learn nao precisa dele para particionar, mas
    passa-lo mantem a chamada correta se alguem trocar a lista por um
    particionador agrupado.

    `x_treino` e a matriz **crua** do treino: e o pipeline que transforma,
    dentro de cada fold.
    """
    validacao.conferir_folds(busca.cv, grupos_treino)

    inicio = time.perf_counter()
    busca.fit(x_treino, y_treino, groups=grupos_treino)
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
    """
    if isinstance(valor, np.generic):
        return valor.item()
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
    return resumo.sort_values("rank_test_score").reset_index(drop=True)


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
    funcao: assim o notebook, que confere `busca_rf.ARQUIVO_HIPERPARAMETROS`, e
    esta funcao nunca apontam para arquivos diferentes.
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


def carregar_hiperparametros(caminho=None) -> dict[str, object]:
    """Le o registro gravado por `salvar_resultados`, no mesmo caminho padrao dela."""
    return json.loads(Path(caminho or ARQUIVO_HIPERPARAMETROS).read_text(encoding="utf-8"))


def reconstruir_pipeline(preprocessador, registro: dict[str, object]):
    """Remonta o pipeline vencedor a partir do JSON, sem ajustar.

    Usa `criar_pipeline_random_forest` do #187, e nao um `Pipeline` montado
    aqui, para que o reconstruido seja o mesmo objeto que a busca varreu. A
    semente vem do registro: com outra, a floresta reconstruida seria outra e o
    CR04 nao teria como bater.
    """
    return criar_pipeline_random_forest(
        preprocessador,
        random_state=registro["random_state"],
        **registro["hiperparametros"],
    )


def parametros_no_limite(
    hiperparametros: dict[str, object],
    espaco: dict,
    tolerancia: float = FRACAO_BORDA,
) -> list[str]:
    """Nomes dos parametros vencedores que ficaram na borda do intervalo buscado.

    O "Como revisar" do card pede que um vencedor no limite seja comentado: se o
    melhor `max_depth` e o teto do intervalo, a busca provavelmente queria ir alem
    dele, e o intervalo do #186 fica sob suspeita.

    Distribuicao discreta do scipy (`randint`) conta como borda so no piso ou no
    teto exatos. Distribuicao continua (`uniform`) nunca sorteia a borda exata,
    entao conta como borda quem cair a menos de `tolerancia` da amplitude de um
    dos extremos. Listas so tem borda se forem de numeros; categorias como
    `class_weight` ficam de fora, porque nao ha "alem" de uma categoria.
    """
    no_limite = []
    for nome, valor in hiperparametros.items():
        distribuicao = espaco.get(nome)
        if distribuicao is None or valor is None or isinstance(valor, (str, bool)):
            continue
        if hasattr(distribuicao, "support"):
            piso, teto = (float(v) for v in distribuicao.support())
            margem = 0.0 if isinstance(distribuicao.dist, stats.rv_discrete) else tolerancia * (teto - piso)
        else:
            numericos = [v for v in distribuicao if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not numericos:
                continue
            piso, teto, margem = float(min(numericos)), float(max(numericos)), 0.0
        if valor <= piso + margem or valor >= teto - margem:
            no_limite.append(nome)
    return no_limite

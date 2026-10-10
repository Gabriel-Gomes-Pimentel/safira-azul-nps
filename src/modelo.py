"""Primeiro modelo candidato do score de detracao: gradient boosting sobre arvores.

**Por que arvores e nao um modelo aditivo.** A secao 4.2.1 mediu que nenhuma
variavel operacional isolada tem correlacao forte com a detracao, sendo a maior a
de `ATRASO_CHEGADA`, com 0,296. A hipotese 5 da secao 4.2.3 mediu a outra metade
do argumento: a interacao entre fidelizacao e falha operacional existe e foi
confirmada (p = 0,0048 na comparacao direta entre `DIAMANTE` e `SEM CADASTRO`).
Um modelo linear representa cada variavel por um peso fixo e so alcanca essa
combinacao se alguem escrever o termo de interacao a mao. O boosting sobre
arvores a encontra por construcao, e e exatamente isso que este candidato precisa
mostrar contra o piso logistico da secao 2: se ele nao superar a logistica, a
interacao ou nao existe ou nao foi captada, e a escolha do algoritmo perde o
argumento.

**Por que o `HistGradientBoostingClassifier`.** E o gradient boosting do proprio
scikit-learn, ja declarado no `requirements.txt`, entao nao acrescenta dependencia
ao projeto nem uma segunda biblioteca para o Colab instalar. Ele discretiza cada
variavel em histogramas antes de crescer as arvores, o que faz o custo depender do
numero de bins e nao do numero de linhas, e por isso ele treina em segundos sobre
as 341.962 linhas do treino.

**Nenhum hiperparametro implicito.** O DoD do #103 exige que os hiperparametros
estejam explicitos, e `HIPERPARAMETROS_CANDIDATO` existe para isso: ele declara
tambem os que coincidem com o padrao da biblioteca, porque padrao nao declarado e
decisao que ninguem tomou. Dois deles mudam o resultado e estao comentados um a
um la embaixo, com destaque para `early_stopping=False`: no padrao `"auto"` a
biblioteca separaria sozinha uma fatia **aleatoria** do ajuste para parar cedo, e
essa fatia ignoraria o agrupamento por Cliente que a secao 3 construiu, colocando
respostas da mesma pessoa nos dois lados. A validacao deste projeto sao os folds
do #102, e nenhuma outra.

A avaliacao vive aqui junto com o modelo porque as duas decisoes andam juntas: um
numero de validacao so e comparavel se o candidato e os pisos passarem pelos
mesmos folds, com o pre-processador reajustado dentro de cada um.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

# O modulo localiza os proprios vizinhos em vez de depender de quem o importa,
# pelo mesmo motivo de `matriz.py` e `validacao.py`: sob pytest o conftest prepara
# o caminho, no Colab e na execucao direta nao ha ninguem para preparar.
_RAIZ = Path(__file__).resolve().parents[1]
for _pasta in ("src", "scripts"):
    _caminho = str(_RAIZ / _pasta)
    if _caminho not in sys.path:
        sys.path.insert(0, _caminho)

SEMENTE_PADRAO = 42

# Metrica principal da comparacao. A precisao media resume a curva de precisao
# contra revocacao, que e a leitura certa quando so um em cada cinco Clientes e
# Detrator: a acuracia premia quem nunca preve a classe rara, como a secao 2.2
# mostrou no piso trivial.
METRICA_PRINCIPAL = "precisao_media"

HIPERPARAMETROS_CANDIDATO = {
    # Passo curto, metade do padrao de 0,1: cada arvore corrige menos, o modelo
    # erra menos por excesso de confianca numa unica particao do espaco, e o
    # custo e precisar de mais iteracoes, compensado logo abaixo.
    "learning_rate": 0.05,
    # 300 arvores, tres vezes o padrao de 100, para compensar o passo curto. Sem
    # parada antecipada este numero e o unico limite do ensemble, entao ele e uma
    # decisao e nao um teto de seguranca.
    "max_iter": 300,
    # Padrao da biblioteca, declarado: com 31 folhas cada arvore ja representa
    # interacao de varias ordens, que e a razao de ter escolhido arvores.
    "max_leaf_nodes": 31,
    # Sem limite de profundidade porque quem limita o tamanho da arvore aqui e
    # `max_leaf_nodes`; fixar os dois esconderia qual dos dois esta agindo.
    "max_depth": None,
    # Cinco vezes o padrao de 20. Com 341 mil linhas, folha de 20 observacoes
    # descreve ruido de um punhado de respostas, e o modelo memoriza Cliente em
    # vez de aprender o fenomeno.
    "min_samples_leaf": 100,
    # Regularizacao L2 ligada, contra o padrao 0.0, pela mesma razao: penaliza
    # folha com pouca evidencia por tras.
    "l2_regularization": 1.0,
    # **Decisao critica.** No padrao "auto" a biblioteca liga a parada antecipada
    # sozinha acima de 10 mil linhas e separa uma fatia aleatoria do ajuste para
    # medir. Essa fatia nao respeita `ID_GOLDENRECORD`, entao respostas da mesma
    # pessoa cairiam no ajuste e na medicao interna, que e o vazamento que a
    # secao 3 existe para evitar. A validacao deste projeto sao os folds do #102.
    "early_stopping": False,
    # Sem reponderacao de classe, ao contrario da logistica da secao 2.3. La ela
    # era necessaria para o piso nao colapsar na classe majoritaria; aqui as duas
    # metricas usadas dependem so da ordenacao do score, que a reponderacao nao
    # melhora, e ela deslocaria a probabilidade predita para longe da frequencia
    # observada, estragando o escore de Brier da secao 6 e o limiar por
    # capacidade da secao 7.
    "class_weight": None,
}


def criar_candidato(
    semente: int = SEMENTE_PADRAO,
    **ajustes: object,
) -> HistGradientBoostingClassifier:
    """Devolve o candidato com os hiperparametros declarados e a semente fixada.

    `ajustes` sobrescreve `HIPERPARAMETROS_CANDIDATO` sem alterar o dicionario
    original, e existe para a busca da secao 5 (#104) variar um parametro por vez
    sem reescrever a configuracao inteira aqui. A semente entra sempre, tambem
    quando a chamada vem da busca, porque duas execucoes do mesmo ajuste precisam
    dar o mesmo numero para que a comparacao entre ajustes signifique alguma coisa.
    """
    desconhecidos = set(ajustes) - set(HIPERPARAMETROS_CANDIDATO)
    if desconhecidos:
        raise ValueError(
            "ajuste de hiperparametro fora da configuracao declarada: "
            f"{sorted(desconhecidos)}. Acrescente-o a HIPERPARAMETROS_CANDIDATO "
            "com a justificativa, em vez de passa-lo solto."
        )
    return HistGradientBoostingClassifier(
        **{**HIPERPARAMETROS_CANDIDATO, **ajustes},
        random_state=semente,
    )


def avaliar_nos_folds(
    fabrica: Callable[[], object],
    x_treino: pd.DataFrame,
    y_treino: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    preprocessador: object,
) -> pd.DataFrame:
    """Mede um modelo nos folds do #102 e devolve uma linha por fold.

    Recebe uma **fabrica**, e nao um modelo pronto, porque cada fold precisa de um
    estimador novo: reaproveitar a mesma instancia faria o segundo fold comecar do
    ajuste do primeiro, e a media dos cinco mediria um modelo que ja viu quase
    todo o treino.

    O pre-processador e clonado e **reajustado dentro de cada fold**, junto com o
    modelo. O `preparar_matriz` da secao 1.4 ajusta o pre-processador na particao
    de treino inteira, o que e o certo para o modelo final, mas usar aquela versao
    aqui deixaria a mediana da imputacao e a escala carregarem informacao das
    linhas que o fold usa como validacao. O efeito seria pequeno e invisivel, que
    e a pior combinacao possivel num numero que decide hiperparametro.

    Os indices dos folds sao posicionais, como todo `split` do scikit-learn, entao
    a selecao e por `.iloc`. As particoes de `preparar_matriz` preservam o indice
    original da base, que nao e um `RangeIndex`, e trocar um pelo outro
    selecionaria silenciosamente as linhas erradas.
    """
    if not folds:
        raise ValueError("nenhum fold recebido: chame validacao.criar_folds antes")
    if len(x_treino) != len(y_treino):
        raise ValueError(
            "x_treino e y_treino precisam ter o mesmo tamanho: "
            f"{len(x_treino)} contra {len(y_treino)}"
        )

    linhas = []
    for numero, (ajuste, validacao) in enumerate(folds, start=1):
        pipeline = Pipeline([
            ("preparo", clone(preprocessador)),
            ("modelo", fabrica()),
        ])
        pipeline.fit(x_treino.iloc[ajuste], y_treino.iloc[ajuste])

        y_validacao = y_treino.iloc[validacao]
        score = pipeline.predict_proba(x_treino.iloc[validacao])[:, 1]
        linhas.append({
            "fold": numero,
            "n_ajuste": len(ajuste),
            "n_validacao": len(validacao),
            METRICA_PRINCIPAL: float(average_precision_score(y_validacao, score)),
            "roc_auc": float(roc_auc_score(y_validacao, score)),
        })
    return pd.DataFrame(linhas).set_index("fold")


def comparar_nos_folds(
    fabricas: dict[str, Callable[[], object]],
    x_treino: pd.DataFrame,
    y_treino: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    preprocessador: object,
) -> pd.DataFrame:
    """Media e desvio por modelo, todos medidos nos mesmos folds.

    O desvio entre folds entra ao lado da media porque uma diferenca de media
    menor do que a variacao entre folds nao sustenta a afirmacao de que um modelo
    e melhor do que o outro, e e essa comparacao, e nao a media sozinha, que a
    secao 4.3 precisa reportar.
    """
    linhas = []
    for nome, fabrica in fabricas.items():
        por_fold = avaliar_nos_folds(fabrica, x_treino, y_treino, folds, preprocessador)
        linhas.append({
            "modelo": nome,
            METRICA_PRINCIPAL: por_fold[METRICA_PRINCIPAL].mean(),
            f"{METRICA_PRINCIPAL}_desvio": por_fold[METRICA_PRINCIPAL].std(),
            "roc_auc": por_fold["roc_auc"].mean(),
            "roc_auc_desvio": por_fold["roc_auc"].std(),
        })
    return pd.DataFrame(linhas).set_index("modelo")


def conferir_ganho_sobre_os_pisos(
    comparacao: pd.DataFrame,
    candidato: str,
    pisos: list[str],
    metrica: str = METRICA_PRINCIPAL,
) -> None:
    """Interrompe a execucao se o candidato nao superar todos os pisos.

    E o CR02 do #103 virando trava: sem ela, um candidato pior do que a regressao
    logistica seguiria para as secoes de ajuste e de metricas sem que nada no
    caminho recusasse, e a tabela comparativa da secao 9 so denunciaria o problema
    no fim. A conferencia e na media dos folds, nunca no teste, que continua
    intocado ate a secao 6.
    """
    faltando = [nome for nome in [candidato, *pisos] if nome not in comparacao.index]
    if faltando:
        raise KeyError(f"modelos ausentes da comparacao: {faltando}")

    valor_candidato = comparacao.loc[candidato, metrica]
    perdeu_para = [
        nome for nome in pisos if valor_candidato <= comparacao.loc[nome, metrica]
    ]
    if perdeu_para:
        raise AssertionError(
            f"o candidato nao supera {perdeu_para} em {metrica} na validacao: "
            f"{valor_candidato:.4f} contra "
            + ", ".join(f"{nome}={comparacao.loc[nome, metrica]:.4f}" for nome in pisos)
            + ". Rever a escolha do algoritmo antes de seguir para a secao 5."
        )


# Grade da busca do #104. Pequena e fatorial de proposito: com tres eixos e doze
# combinacoes cada efeito e visivel isoladamente e a tabela ainda cabe no
# documento, enquanto uma busca aleatoria maior daria um numero melhor sem deixar
# claro qual parametro o produziu, que e o oposto do que a secao 4.4 vai cobrar.
# A configuracao do #103 (0.05 / 31 / 300) e uma das celulas, e nao um ponto de
# fora: sem ela na mesma tabela, medida nos mesmos folds, o ganho da busca nao
# seria comparavel.
GRADE_HIPERPARAMETROS = {
    # O padrao da biblioteca e 0,1 e o #103 escolheu metade disso. Os dois entram
    # porque passo e numero de arvores se compensam, e testar um sem o outro
    # mediria a interacao dos dois como se fosse efeito de um so.
    "learning_rate": [0.05, 0.10],
    # 31 e o padrao ja declarado no #103; 63 dobra a interacao que cada arvore
    # representa. Este e o eixo de "profundidade" do card: quem limita o tamanho
    # da arvore aqui e o numero de folhas, nao `max_depth`, que segue em None.
    "max_leaf_nodes": [31, 63],
    # Sem parada antecipada este numero e o unico limite do ensemble. Metade,
    # o valor do #103 e o dobro: e o eixo que responde se 300 arvores ja
    # esgotaram o metodo nesta base ou se o ganho ainda estava subindo.
    "max_iter": [150, 300, 600],
}


def combinacoes_da_grade(grade: dict[str, list]) -> list[dict]:
    """Produto cartesiano da grade, em ordem estavel.

    A ordem e determinista para que duas execucoes produzam a tabela na mesma
    sequencia e o diff do notebook mostre so o que mudou de fato.
    """
    from itertools import product

    nomes = list(grade)
    return [dict(zip(nomes, valores)) for valores in product(*(grade[n] for n in nomes))]


def buscar_hiperparametros(
    grade: dict[str, list],
    x_treino: pd.DataFrame,
    y_treino: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    preprocessador: object,
    semente: int = SEMENTE_PADRAO,
) -> pd.DataFrame:
    """Mede cada combinacao da grade nos folds e devolve **uma linha por combinacao**.

    Devolver a tabela inteira, e nao so o vencedor, e o CR02 do #104: a decisao
    precisa ser reconstruivel sem reexecutar a busca, que custa cerca de setenta e
    cinco segundos por combinacao nesta base.

    **O teste nao entra aqui, por construcao.** A funcao nao tem parametro por onde
    receber `x_teste`, e os folds vem de `validacao.criar_folds`, que particiona
    apenas o treino agrupando por Cliente. Nenhum caminho deste modulo alcanca a
    particao de teste, que segue intocada ate a secao 6.

    Cada combinacao passa pelos **mesmos** folds, com o pre-processador reajustado
    dentro de cada um por `avaliar_nos_folds`. Sem isso a diferenca entre duas
    linhas da tabela misturaria efeito de hiperparametro com efeito de particao.
    """
    # `product()` sem eixos devolve uma combinacao vazia, e nao nenhuma, entao a
    # conferencia e na grade e nao no resultado: sem ela uma grade vazia rodaria
    # um unico ajuste com a configuracao de partida e a tabela sairia com uma
    # linha, parecendo busca.
    if not grade:
        raise ValueError("grade vazia: nada a buscar")
    combinacoes = combinacoes_da_grade(grade)

    linhas = []
    for ajustes in combinacoes:
        por_fold = avaliar_nos_folds(
            lambda a=ajustes: criar_candidato(semente=semente, **a),
            x_treino, y_treino, folds, preprocessador,
        )
        linhas.append({
            **ajustes,
            METRICA_PRINCIPAL: por_fold[METRICA_PRINCIPAL].mean(),
            f"{METRICA_PRINCIPAL}_desvio": por_fold[METRICA_PRINCIPAL].std(),
            "roc_auc": por_fold["roc_auc"].mean(),
        })
    return pd.DataFrame(linhas)


def escolher_configuracao(
    tabela: pd.DataFrame,
    base: dict | None = None,
    metrica: str = METRICA_PRINCIPAL,
) -> dict:
    """Escolhe a configuracao final e diz **por que**, nao so qual.

    A regra e a mesma que o #103 usou para afirmar que o candidato supera a
    logistica: um ganho menor do que o desvio entre folds nao sustenta a
    afirmacao de que uma configuracao e melhor do que a outra. Aqui ela vira
    criterio de escolha, e nao so de leitura, porque o "Como revisar" do #104
    pede exatamente que um ganho marginal nao seja apresentado como relevante.

    Se o melhor da grade nao superar a configuracao de partida por mais do que o
    desvio entre folds dela, **a de partida permanece**: trocar a configuracao
    para perseguir a terceira casa decimal seria escolher ruido, e o modelo final
    ficaria diferente do que o #103 documentou sem nada ter melhorado de fato.
    """
    if base is None:
        base = {n: HIPERPARAMETROS_CANDIDATO[n] for n in GRADE_HIPERPARAMETROS}

    eixos = [c for c in tabela.columns if c in base]
    mascara = pd.Series(True, index=tabela.index)
    for eixo in eixos:
        mascara &= tabela[eixo] == base[eixo]
    if not mascara.any():
        raise ValueError(
            f"a configuracao de partida {base} nao esta na grade; sem ela na mesma "
            "tabela o ganho da busca nao e comparavel"
        )

    linha_base = tabela[mascara].iloc[0]
    melhor = tabela.loc[tabela[metrica].idxmax()]
    ganho = float(melhor[metrica] - linha_base[metrica])
    desvio = float(linha_base[f"{metrica}_desvio"])
    relevante = ganho > desvio

    escolhida = melhor if relevante else linha_base
    return {
        "configuracao": {eixo: escolhida[eixo] for eixo in eixos},
        "e_a_de_partida": not relevante,
        "metrica_escolhida": float(escolhida[metrica]),
        "metrica_da_partida": float(linha_base[metrica]),
        "ganho_sobre_a_partida": ganho,
        "desvio_entre_folds_da_partida": desvio,
        "ganho_supera_o_desvio": relevante,
    }


# --- Metricas de ordenacao do score (#105) -----------------------------------
#
# As tres metricas desta secao respondem a perguntas diferentes sobre o mesmo
# score, e a escolha das tres vem do uso real do modelo na Azul: ele ordena uma
# fila de contato pos-viagem, nao emite um veredito por passageiro.
#
# - **Precisao media.** Area sob a curva de precisao contra cobertura. E a
#   leitura certa quando a classe de interesse e minoritaria (um em cada cinco
#   Clientes e Detrator): ela mede o que a equipe encontra ao descer a fila a
#   partir do topo, que e exatamente o gesto que o time de Experiencia do
#   Cliente faz. A acuracia, na mesma situacao, premia quem nunca preve a classe
#   rara, como o piso trivial da secao 2.2 mostrou.
# - **ROC-AUC.** Depende so da ordenacao e nao da prevalencia, entao permite
#   comparar modelos entre si e contra o 0,5 de quem nao ordena nada, sem que a
#   proporcao de Detratores da particao interfira no numero.
# - **Escore de Brier.** As duas anteriores enxergam apenas a **ordem** do score.
#   Nenhuma delas muda se todas as probabilidades forem divididas por dois, e o
#   modelo passaria a dizer "risco de 10%" onde o risco e de 20% sem que
#   nenhuma das duas reclamasse. O Brier mede o erro quadratico da probabilidade
#   contra o desfecho observado, e e a unica das tres que responde se o numero
#   emitido pode ser lido como risco. Isso importa porque a saida vai para a
#   equipe como probabilidade, e nao como rotulo, e porque a secao 7 escolhe o
#   limiar operacional em cima dela. **Menor e melhor**, ao contrario das outras
#   duas, e ele mede calibracao, nao taxa de acerto.


def metricas_de_ordenacao(
    y_verdadeiro: pd.Series,
    score: np.ndarray,
) -> dict[str, float]:
    """As tres metricas do #105 a partir do alvo e da **probabilidade**.

    Recusa um score que pareca saida de `predict()` em vez de
    `predict_proba()[:, 1]`. Passar o rotulo binario no lugar da probabilidade e
    o erro que o "Como revisar" do card aponta, e ele degrada em silencio: a
    precisao media e o Brier pioram sem que nada quebre, e o numero errado
    parece apenas um modelo pior. Um score degenerado de valor unico, como o do
    piso trivial da secao 2.2, continua aceito, porque ali a constante e a
    probabilidade de fato.
    """
    from sklearn.metrics import brier_score_loss

    valores = np.unique(np.asarray(score))
    if set(valores.tolist()) == {0.0, 1.0}:
        raise ValueError(
            "o score recebido tem apenas os valores 0 e 1, o que indica saida de "
            "predict() em vez de predict_proba()[:, 1]. As tres metricas desta "
            "secao leem probabilidade, nao rotulo."
        )
    if valores.min() < 0.0 or valores.max() > 1.0:
        raise ValueError(
            f"score fora de [0, 1] (min {valores.min()}, max {valores.max()}): o "
            "escore de Brier so tem sentido sobre probabilidade."
        )

    return {
        METRICA_PRINCIPAL: float(average_precision_score(y_verdadeiro, score)),
        "roc_auc": float(roc_auc_score(y_verdadeiro, score)),
        "brier": float(brier_score_loss(y_verdadeiro, score)),
    }


def metricas_no_teste(
    modelos: dict[str, object],
    x_teste_transformado: object,
    y_teste: pd.Series,
) -> pd.DataFrame:
    """Uma linha por modelo, todas medidas na particao de teste.

    A funcao chama `predict_proba` por conta propria em vez de receber o score
    pronto. E a mesma ideia da fabrica em `avaliar_nos_folds`: o erro deixa de
    depender de quem escreve a celula. Nao ha por onde passar `predict()` aqui,
    entao a trava de `metricas_de_ordenacao` vira rede de seguranca de segunda
    linha, e nao a unica.
    """
    linhas = []
    for nome, modelo in modelos.items():
        score = modelo.predict_proba(x_teste_transformado)[:, 1]
        linhas.append({"modelo": nome, **metricas_de_ordenacao(y_teste, score)})
    return pd.DataFrame(linhas).set_index("modelo")


# --- Limiar operacional e matriz de confusao (#106) --------------------------
#
# O ponto de corte nao sai de maximizar uma metrica agregada. Sai da capacidade
# de contato da equipe de Experiencia do Cliente: a lista util e a que cabe no
# dia de trabalho, e um limiar que produza uma fila maior do que a operacao
# consegue percorrer descreve um processo que nao existe. Por isso a entrada
# aqui e `k`, o numero de contatos, e o limiar e consequencia dele.
#
# `0,5` nunca e candidato. Ele so faria sentido se o custo de contatar quem nao
# detrataria fosse igual ao de deixar passar um Detrator, e a secao 4.1.3 do
# documento registra que a propria Azul considera o segundo mais caro.


def limiar_por_capacidade(score: np.ndarray, k: int) -> float:
    """Menor score que ainda entra numa fila de `k` contatos.

    Devolve o k-esimo maior valor. Empates no limiar fazem a selecao por
    `score >= limiar` passar de `k`, e quem chama precisa reportar o numero
    efetivamente selecionado em vez de assumir `k`.
    """
    valores = np.asarray(score)
    if not 0 < k <= len(valores):
        raise ValueError(
            f"k precisa estar entre 1 e {len(valores)} (recebido {k}): a fila nao "
            "pode ser vazia nem maior que a propria particao"
        )
    return float(np.sort(valores)[::-1][k - 1])


def metricas_no_topo(
    y_verdadeiro: pd.Series,
    score: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Precisao, cobertura e F1 no topo de uma fila de `k` contatos.

    `precisao_no_topo` responde a pergunta da operacao: de cada cem pessoas que
    a equipe liga, quantas eram Detratoras. `cobertura` responde a do negocio:
    de todos os Detratores do periodo, quantos a fila alcancou. As duas se
    movem em sentidos opostos conforme `k` cresce, e e por isso que a escolha de
    `k` e uma decisao de operacao, e nao de modelagem.
    """
    y = np.asarray(y_verdadeiro).astype(int)
    limiar = limiar_por_capacidade(score, k)
    selecionado = np.asarray(score) >= limiar

    vp = int((selecionado & (y == 1)).sum())
    fp = int((selecionado & (y == 0)).sum())
    fn = int((~selecionado & (y == 1)).sum())
    vn = int((~selecionado & (y == 0)).sum())

    precisao = vp / (vp + fp) if vp + fp else 0.0
    cobertura = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * cobertura / (precisao + cobertura) if precisao + cobertura else 0.0
    return {
        "k_pedido": int(k),
        "n_selecionados": int(selecionado.sum()),
        "limiar": limiar,
        "precisao_no_topo": precisao,
        "cobertura": cobertura,
        "f1": f1,
        "vp": vp, "fp": fp, "fn": fn, "vn": vn,
    }


def matriz_de_confusao(
    y_verdadeiro: pd.Series,
    score: np.ndarray,
    limiar: float,
) -> pd.DataFrame:
    """Matriz 2x2 rotulada pela acao da operacao, e nao por 0 e 1.

    Os rotulos falam de fila de contato porque os quatro quadrantes tem custos
    diferentes e nomeados: um falso positivo gasta minutos de analista, um falso
    negativo perde um Detrator que ninguem procurou.
    """
    y = np.asarray(y_verdadeiro).astype(int)
    selecionado = np.asarray(score) >= limiar
    return pd.DataFrame(
        [[int((~selecionado & (y == 0)).sum()), int((selecionado & (y == 0)).sum())],
         [int((~selecionado & (y == 1)).sum()), int((selecionado & (y == 1)).sum())]],
        index=pd.Index(["não Detrator", "Detrator"], name="desfecho observado"),
        columns=pd.Index(["fora da fila", "na fila de contato"], name="decisão da operação"),
    )


def tabela_de_capacidade(
    y_verdadeiro: pd.Series,
    score: np.ndarray,
    capacidades_diarias: list[int],
    dias: int,
) -> pd.DataFrame:
    """Uma linha por premissa de capacidade, para a premissa poder ser revista.

    A capacidade diaria e suposicao do grupo, nao numero fornecido pela Azul.
    Entregar a tabela inteira em vez de um unico ponto faz com que a confirmacao
    do numero real pela companhia atualize a leitura sem refazer a analise.
    """
    linhas = []
    for capacidade in capacidades_diarias:
        m = metricas_no_topo(y_verdadeiro, score, capacidade * dias)
        linhas.append({"contatos_por_dia": capacidade, **m})
    return pd.DataFrame(linhas).set_index("contatos_por_dia")
# --- Tabela comparativa entre candidato e pisos (#108) -----------------------
#
# A secao 6 mede as tres metricas de ordenacao e a secao 7 escolhe o limiar. Esta
# tabela junta as duas leituras num lugar so, porque separadas elas permitem o
# erro que o card quer impedir: apresentar o numero do candidato sem escala de
# comparacao, ou comparar metrica dependente de corte entre modelos medidos em
# filas de tamanhos diferentes.


def tabela_comparativa(
    modelos: dict[str, object],
    x_teste_transformado: object,
    y_teste: pd.Series,
    k: int,
) -> pd.DataFrame:
    """Uma linha por modelo, com as metricas de ordenacao e as de fila.

    Os tres modelos passam pela **mesma particao de teste** e pela **mesma
    capacidade `k`**, e nao pelo mesmo valor numerico de limiar. Igualar o
    numero seria a comparacao errada: cada modelo emite score numa escala
    propria, e o corte de 0,29 do candidato nao significa nada na escala da
    logistica. O que a operacao tem de fato igual entre modelos e quantas
    pessoas cabem na fila do dia.

    `n_na_fila` entra como coluna, e nao como pressuposto, porque um score
    degenerado nao produz fila nenhuma: o piso trivial emite a mesma
    probabilidade para toda linha, o limiar cai sobre essa constante e
    `score >= limiar` seleciona a particao inteira. Sem a coluna, a precisao
    dele apareceria ao lado das outras como se viesse de uma fila de `k`
    contatos, quando vem de 53 mil.

    `fila_comparavel` diz, por linha, se o orcamento de `k` contatos foi de fato
    respeitado. Pedir o mesmo `k` para todos **nao** garante capacidade efetiva
    igual, e por isso a igualdade nao pode ser afirmada no texto a partir do
    parametro: ela precisa ser lida da tabela. Onde a coluna e falsa, as cinco
    metricas dependentes de corte daquela linha nao sao comparaveis as das
    demais sob este orcamento, porque descrevem uma fila de outro tamanho.

    Nao ha desempate embutido aqui de proposito. Qualquer regra que cortasse a
    fila degenerada em exatamente `k` teria de escolher quais empatados entram,
    e qualquer escolha dessas em um score constante e arbitraria: o resultado
    passaria a depender da ordem das linhas, nao do modelo. Preferir a coluna a
    um desempate silencioso deixa a limitacao visivel em vez de produzir um
    numero comparavel por construcao e sem significado.
    """
    if not modelos:
        raise ValueError("nenhum modelo recebido: a tabela comparativa precisa de pelo menos um")

    linhas = []
    for nome, modelo in modelos.items():
        score = modelo.predict_proba(x_teste_transformado)[:, 1]
        topo = metricas_no_topo(y_teste, score, k)
        linhas.append({
            "modelo": nome,
            **metricas_de_ordenacao(y_teste, score),
            "limiar": topo["limiar"],
            "n_na_fila": topo["n_selecionados"],
            "fila_comparavel": topo["n_selecionados"] == k,
            "precisao_no_topo": topo["precisao_no_topo"],
            "cobertura": topo["cobertura"],
            "f1": topo["f1"],
        })
    return pd.DataFrame(linhas).set_index("modelo")


def ganhos_do_candidato(
    tabela: pd.DataFrame,
    candidato: str,
    pisos: list[str],
    metrica: str = METRICA_PRINCIPAL,
) -> pd.DataFrame:
    """Ganho do candidato sobre cada piso, em valor absoluto e em razao.

    Os dois juntos, e nao um deles: o absoluto sozinho nao diz se 0,02 e muito,
    e a razao sozinha infla qualquer diferenca quando o piso e proximo de zero.
    E o CR04 do #108, que pede que a relevancia do ganho possa ser julgada, e
    nao apenas o sinal dele.
    """
    faltando = [nome for nome in [candidato, *pisos] if nome not in tabela.index]
    if faltando:
        raise KeyError(f"modelos ausentes da tabela: {faltando}")
    if metrica not in tabela.columns:
        raise KeyError(f"metrica ausente da tabela: {metrica}")

    valor = float(tabela.loc[candidato, metrica])
    linhas = []
    for piso in pisos:
        base = float(tabela.loc[piso, metrica])
        linhas.append({
            "piso": piso,
            "valor_do_piso": base,
            "valor_do_candidato": valor,
            "ganho_absoluto": valor - base,
            "razao": valor / base if base else float("inf"),
        })
    return pd.DataFrame(linhas).set_index("piso")

"""Modelos de ensemble: espacos de busca (#186) e custo de ajuste (#185).

Este modulo e a casa dos dois cards da dupla de Modelos de Ensemble, e eles se
completam: o #186 declara **em que intervalo** cada hiperparametro pode variar, e
o #185 mede **quanto custa** um ajuste de cada biblioteca candidata, que e o que
diz se a busca sobre aqueles intervalos cabe numa sessao do Colab. Separar os
dois em arquivos diferentes obrigaria quem le a abrir os dois para responder uma
pergunta so, e faria a semente e os nomes de estimador existirem em duplicata.

A primeira metade do arquivo, ate `ESTIMADOR_POR_ESPACO`, e a do #186; a segunda,
de `INTERVALO_AMOSTRAGEM_S` em diante, e a do #185. As docstrings originais dos
dois cards seguem abaixo, na ordem.

---

Espacos de busca aleatoria do Random Forest e do Gradient Boosting (#186).

**Por que RandomizedSearchCV e nao a grade fatorial do #104.** O #104 buscou tres
eixos do candidato unico com doze combinacoes, numero que ainda cabe inteiro no
documento. Random Forest e Gradient Boosting tem cinco a sete eixos cada, e uma
grade fatorial dessa dimensao custaria centenas de ajustes sobre os 341.962
registros do treino para cobrir uma fracao pequena do espaco. A busca aleatoria
troca cobertura exaustiva por cobertura proporcional ao numero de amostras
sorteadas, que e a troca certa quando o objetivo e encontrar uma vizinhanca boa
do espaco, e nao mapear cada aresta dele.

**#185 ainda nao decidiu a biblioteca de boosting.** O card pede para manter as
duas versoes, `HistGradientBoostingClassifier` e `XGBClassifier`, ate a decisao
do #185 e remover a descartada no mesmo dia em que ela sair. Como o #185 segue
aberto, `ESPACO_GRADIENT_BOOSTING_HISTGB` e `ESPACO_GRADIENT_BOOSTING_XGBOOST`
convivem aqui, cada um com o proprio custo por ajuste ainda por medir: os
intervalos abaixo vem da literatura de cada biblioteca e da mesma logica de
regularizacao do #103, nao de um perfil de tempo de execucao que o #185 ainda
nao produziu. Quando a decisao sair, o espaco da biblioteca descartada e o bloco
correspondente deste modulo saem no mesmo commit.

**Os numeros que ancoram os intervalos.** A base analitica tem 484.915 registros
e 20,44% de Detratores (secao 4.2.1 da documentacao); o treino usado pela busca,
apos o split por Cliente do #102, tem 341.962 linhas. O contrato de features do
card 01 (`FEATURE_SET_V1`, em `scripts/preprocessamento_nps.py`) declara 11
features brutas. Esse numero e pequeno para os padroes que motivam os valores
padrao de `max_features` (`"sqrt"` e `"log2"` datam de bases com centenas de
colunas) e por isso o espaco do Random Forest usa fracao continua em vez desses
atalhos, como o comentario de `ESPACO_RANDOM_FOREST` detalha.

**scipy.stats onde o eixo e continuo, lista onde e categorico.** `RandomizedSearchCV`
aceita as duas formas no mesmo dicionario: distribuicoes com `.rvs()` para eixos
numericos, e listas para amostragem uniforme quando o eixo e uma escolha (a
biblioteca de reponderacao, por exemplo, nao tem escala). Forcar uma distribuicao
scipy sobre uma escolha categorica exigiria uma `rv_discrete` a mais so para
imitar `random.choice`, sem nenhum ganho sobre a lista.

---

Custo de ajuste das duas bibliotecas de boosting candidatas (card 04A, #185).

**O que este modulo resolve.** A dupla de Modelos de Ensemble precisa escolher
entre o `HistGradientBoostingClassifier`, que ja vem no scikit-learn, e o
`XGBClassifier`, que acrescenta uma dependencia ao projeto e a sessao do Colab.
A escolha nao e de gosto: os cards seguintes (#186, #187 e #188) rodam uma busca
aleatoria de 40 iteracoes sobre os folds do contrato, e uma sessao do Colab cai
depois de algumas horas. Se um ajuste unico custa tempo demais, a busca nao cabe
na sessao e o card seguinte trava. Este modulo produz o numero que sustenta a
decisao: quanto custa **um** ajuste completo de cada biblioteca sobre a matriz
que o contrato entrega.

**Por que a medicao e de um ajuste so, sem busca e sem validacao cruzada.** O que
se quer estimar e o custo unitario; a busca e a validacao sao multiplicadores
conhecidos, e `estimar_busca` faz essa conta. Medir com busca embutida misturaria
o custo da biblioteca com o custo do protocolo e ainda criaria, aqui, uma segunda
particao alem da do contrato, que e exatamente o que a secao 3 proibe. Por isso
este modulo nunca instancia particionador nem validador proprio: ele recebe a
matriz de treino ja transformada e o alvo, e nada mais.

**Por que os hiperparametros sao iguais nos dois lados.** Comparar tempo de
bibliotecas com configuracoes diferentes nao mede biblioteca, mede configuracao.
`HIPERPARAMETROS_COMPARACAO` declara o par equivalente termo a termo, com mesmo
passo, mesmo numero de arvores, mesmo numero de folhas, mesmo numero de bins,
mesma semente e histograma nos dois; os nomes de cada biblioteca sao traduzidos
logo abaixo. Os valores vem de `modelo.HIPERPARAMETROS_CANDIDATO`, para que o
tempo medido seja o tempo do modelo que o projeto realmente treina, e nao o de um
modelo de brinquedo.

**Por que a memoria e medida por RSS amostrado, e nao por `tracemalloc`.** As duas
bibliotecas alocam a maior parte da memoria em codigo nativo, fora do alocador do
Python; `tracemalloc` enxerga so o lado Python e devolveria um numero pequeno e
errado. O `psutil` le o RSS do processo, que inclui as alocacoes nativas, e uma
amostragem em thread separada captura o pico durante o ajuste.

**O pipeline do Random Forest (card 08A, #187)** vive no fim do modulo:
`criar_pipeline_random_forest` encadeia o pre-processador do contrato ao
estimador, e `medir_linha_de_base` produz a referencia que a busca do #189 vai
tentar superar, medida pela funcao `avaliar` do card 05 (#241).

**O pipeline do Gradient Boosting (card 11, #188)** vem logo depois:
`criar_pipeline_gradient_boosting` faz o mesmo com o
`HistGradientBoostingClassifier`, e a linha de base dele, medida pela mesma
`medir_linha_de_base`, e a referencia da busca do #190.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# O modulo localiza os proprios vizinhos em vez de depender de quem o importa,
# pelo mesmo motivo de `matriz.py` e `modelo.py`: sob pytest o conftest prepara o
# caminho, no Colab e na execucao direta nao ha ninguem para preparar.
_RAIZ = Path(__file__).resolve().parents[1]
for _pasta in ("src", "scripts"):
    _caminho = str(_RAIZ / _pasta)
    if _caminho not in sys.path:
        sys.path.insert(0, _caminho)

import modelo  # noqa: E402

SEMENTE_PADRAO = modelo.SEMENTE_PADRAO

# Pacotes cuja versao exata sustenta a medicao deste modulo. A conferencia de
# `conferir_versoes` e restrita a eles porque sao os unicos em que uma minor
# diferente muda o numero medido: os demais do `requirements.txt` sao piso
# (`>=`) de proposito.
PACOTES_TRAVADOS = ("scikit-learn", "scipy", "xgboost")
REQUIREMENTS_PADRAO = _RAIZ / "requirements.txt"

# --------------------------------------------------------------------------
# Espacos de busca aleatoria (#186)
# --------------------------------------------------------------------------
# Proporcao negativos/positivos da base (385.775 / 99.140), usada para calibrar
# os eixos de reponderacao do Random Forest e do XGBoost. Documentada aqui, e
# nao recalculada a cada busca, para que os dois espacos usem o mesmo numero.
RAZAO_DESBALANCEAMENTO = 385_775 / 99_140

ESPACO_RANDOM_FOREST = {
    # 200 a 600 arvores. Com 341.962 linhas no treino, a variancia de cada
    # arvore ja e pequena; acima de algumas centenas o ganho de agregar mais
    # arvores fica marginal e o custo segue linear no numero delas, ao
    # contrario do Gradient Boosting, onde mais iteracoes tambem mudam o vies.
    "n_estimators": stats.randint(200, 601),
    # 3 a 20 folhas de profundidade. O teto vem de log2(341.962) ~= 18,4: uma
    # arvore balanceada raramente precisa de mais niveis do que isso para
    # separar o treino inteiro, e permitir profundidade ilimitada (`None`)
    # arriscaria arvores memorizando Cliente em vez de aprender o fenomeno, o
    # mesmo risco que a secao de `min_samples_leaf` do #103 documenta.
    "max_depth": stats.randint(3, 21),
    # 1 a 100 observacoes por folha. Cem e o valor que o #103 escolheu para o
    # Gradient Boosting nesta mesma base, pela mesma razao: folha menor do que
    # isso descreve ruido de um punhado de respostas. O piso em 1 mantem a
    # ponta nao regularizada do espaco disponivel para a busca comparar.
    "min_samples_leaf": stats.randint(1, 101),
    # Fracao continua de 0,3 a 1,0 das colunas por split, e nao `"sqrt"` ou
    # `"log2"`. As duas strings datam de bases com centenas de features; aqui o
    # contrato do card 01 declara 11 features brutas, `"sqrt"` amostraria
    # cerca de 3 delas por split e jogaria fora a maior parte do sinal
    # disponivel a cada arvore. Uma fracao ampla preserva a decorrelacao entre
    # arvores sem sufocar o sinal.
    "max_features": stats.uniform(0.3, 0.7),
    # Sem reponderacao, reponderacao pelas classes observadas neste ajuste, e
    # reponderacao por arvore (efeito parecido com sub-amostrar a classe
    # majoritaria a cada bootstrap). As tres entram porque a taxa de 20,44% de
    # detracao e desbalanceamento moderado, e ao contrario do candidato unico
    # do #103 (onde a reponderacao so atrapalharia a probabilidade calibrada
    # de um `HistGradientBoostingClassifier`), o Random Forest nao usa a saida
    # como probabilidade de risco por construcao: cada arvore vota, e a
    # reponderacao pode ajudar a fronteira sem o mesmo custo de calibracao.
    # A busca decide empiricamente, nesta base, qual das tres funciona melhor.
    "class_weight": ["balanced", "balanced_subsample", None],
}


ESPACO_GRADIENT_BOOSTING_HISTGB = {
    # Escala log de 0,01 a 0,3. O #103 fixou 0,05 (metade do padrao de 0,1) e
    # a grade do #104 testou 0,05 e 0,10; a busca aleatoria amplia essa faixa
    # para os dois lados. Log-uniforme, e nao uniforme, porque o efeito do
    # passo e multiplicativo sobre o numero de iteracoes necessario: 0,01 para
    # 0,02 muda o ajuste tanto quanto 0,15 para 0,30.
    "learning_rate": stats.loguniform(0.01, 0.3),
    # 15 a 127 folhas por arvore. O #103 usa o padrao de 31 e a grade do #104
    # testou 31 e 63; o teto de 127 (2^7 - 1) da a busca uma arvore ainda mais
    # profunda para comparar sem passar a ordem de grandeza de `max_iter`.
    "max_leaf_nodes": stats.randint(15, 128),
    # 100 a 600 iteracoes, cobrindo os tres pontos da grade do #104 (150, 300 e
    # 600) e as bordas em torno deles. Sem parada antecipada (fixa em `False`
    # em `criar_candidato`, por causa do vazamento por Cliente que o #103
    # documenta), este numero segue sendo o unico limite do ensemble.
    "max_iter": stats.randint(100, 601),
    # 0,0 a 2,0 de regularizacao L2. O #103 fixou 1,0 contra o padrao 0,0 da
    # biblioteca; o intervalo cobre o padrao original e o dobro da escolha do
    # #103 para a busca decidir se penalizar mais ou menos folhas com pouca
    # evidencia melhora a precisao media.
    "l2_regularization": stats.uniform(0.0, 2.0),
    # 20 a 200 observacoes por folha, com o mesmo raciocinio de
    # `ESPACO_RANDOM_FOREST`: folha pequena sobre 341.962 linhas memoriza
    # Cliente. O piso de 20 e o padrao da biblioteca, mantido para a busca
    # tambem considerar o extremo pouco regularizado.
    "min_samples_leaf": stats.randint(20, 201),
    # Sem reponderacao e reponderacao balanceada. O #103 documenta por que
    # `class_weight=None` foi a escolha do candidato unico: as duas metricas
    # de seleção dependem so da ordenacao do score, e reponderar deslocaria a
    # probabilidade predita para longe da frequencia observada, o que piora o
    # Brier da secao 6 e o limiar por capacidade da secao 7. O eixo entra na
    # busca mesmo assim para que a comparacao empirica, e nao so o argumento
    # teorico, sustente a escolha final.
    "class_weight": [None, "balanced"],
}


ESPACO_GRADIENT_BOOSTING_XGBOOST = {
    # Mesma faixa e mesma razao log-uniforme do HistGB: o efeito do passo
    # sobre o numero de arvores necessario e multiplicativo nas duas
    # bibliotecas, por serem as duas gradient boosting sobre arvores.
    "learning_rate": stats.loguniform(0.01, 0.3),
    # 3 a 10 niveis. O XGBoost limita a arvore por profundidade, e nao por
    # numero de folhas como o HistGB; 10 niveis já produzem ate 2^10 folhas,
    # bem acima do teto de 127 usado no espaco irmao, entao o eixo comparavel
    # entre as duas bibliotecas e a ordem de interacao alcancada, nao o numero
    # bruto do parametro.
    "max_depth": stats.randint(3, 11),
    # 100 a 600 arvores, no mesmo intervalo de `max_iter` do HistGB, para que
    # o custo por ajuste medido no #185 seja comparavel entre as duas
    # bibliotecas sob o mesmo orcamento de iteracoes.
    "n_estimators": stats.randint(100, 601),
    # 1 a 20, o equivalente do XGBoost a `min_samples_leaf`: soma minima de
    # peso das observacoes numa folha antes dela poder ser criada. O teto de
    # 20 acompanha a mesma logica de regularizacao contra folha memorizando
    # Cliente, numa escala distinta da do HistGB porque o parametro pesa
    # gradiente e nao conta linhas diretamente.
    "min_child_weight": stats.randint(1, 21),
    # 0,0 a 2,0 de regularizacao L2, mesma faixa de `l2_regularization` do
    # HistGB, pelo nome equivalente no XGBoost.
    "reg_lambda": stats.uniform(0.0, 2.0),
    # 0,6 a 1,0 das linhas por arvore. Sub-amostrar linhas e o mecanismo de
    # regularizacao do XGBoost que nao tem par direto no HistGB (que discretiza
    # em histogramas em vez de amostrar), entao o piso de 0,6 evita perder
    # tanta informacao por arvore que o ensemble precisasse de muito mais
    # iteracoes para compensar.
    "subsample": stats.uniform(0.6, 0.4),
    # 0,5 a 1,0 das colunas por arvore, equivalente a `max_features` do Random
    # Forest e sujeito ao mesmo argumento: com 11 features brutas no contrato
    # do card 01, um piso baixo deixaria cada arvore enxergar poucas colunas.
    "colsample_bytree": stats.uniform(0.5, 0.5),
    # 1,0 (sem reponderacao) a RAZAO_DESBALANCEAMENTO (~3,89, a proporcao
    # negativos/positivos da base). O XGBoost nao tem versao "balanced" pronta
    # como o scikit-learn; `scale_pos_weight` e o parametro que multiplica o
    # gradiente da classe positiva, e o teto no valor exato da proporcao
    # observada e o ponto em que a reponderacao neutraliza o desbalanceamento
    # por completo.
    "scale_pos_weight": stats.uniform(1.0, RAZAO_DESBALANCEAMENTO - 1.0),
}


# Agrupa os dois espacos de Gradient Boosting pela biblioteca a que pertencem,
# para que o notebook e os testes iterem sem repetir os dois nomes soltos. A
# chave e o nome da biblioteca, nao um rotulo livre, porque e exatamente essa
# chave que sai do dicionario no dia em que o #185 decidir qual descartar.
ESPACOS_GRADIENT_BOOSTING = {
    "hist_gradient_boosting": ESPACO_GRADIENT_BOOSTING_HISTGB,
    "xgboost": ESPACO_GRADIENT_BOOSTING_XGBOOST,
}


# Espaco -> estimador que o consome, para os testes instanciarem cada amostra
# sorteada sem repetir a associacao em cada teste.
ESTIMADOR_POR_ESPACO = {
    "random_forest": (RandomForestClassifier, ESPACO_RANDOM_FOREST),
    "gradient_boosting_hist_gradient_boosting": (
        HistGradientBoostingClassifier, ESPACO_GRADIENT_BOOSTING_HISTGB,
    ),
    "gradient_boosting_xgboost": (XGBClassifier, ESPACO_GRADIENT_BOOSTING_XGBOOST),
}


# --------------------------------------------------------------------------
# Custo de ajuste das bibliotecas candidatas (#185)
# --------------------------------------------------------------------------
# Intervalo da amostragem de memoria. Curto o bastante para nao perder o pico de
# um ajuste de poucos segundos, longo o bastante para o proprio amostrador nao
# disputar CPU com a biblioteca que esta sendo medida.
INTERVALO_AMOSTRAGEM_S = 0.05

# Numero de bins do histograma. E o padrao do `HistGradientBoostingClassifier` e
# esta declarado dos dois lados porque e ele, e nao o numero de linhas, que manda
# no custo de um boosting por histograma: deixar o padrao de cada biblioteca
# valer sozinho compararia duas discretizacoes diferentes.
BINS_HISTOGRAMA = 255

# Par equivalente termo a termo. A chave e o conceito; a traducao para o nome de
# cada biblioteca esta em `_parametros_hist` e `_parametros_xgb`. Os tres
# primeiros valores sao lidos de `modelo.HIPERPARAMETROS_CANDIDATO`, e nao
# copiados: e isso que sustenta a frase do notebook de que o tempo medido e o do
# modelo que o projeto treina. O teste
# `test_comparacao_usa_os_hiperparametros_do_candidato` trava a igualdade.
HIPERPARAMETROS_COMPARACAO = {
    "passo": modelo.HIPERPARAMETROS_CANDIDATO["learning_rate"],
    "arvores": modelo.HIPERPARAMETROS_CANDIDATO["max_iter"],
    "folhas": modelo.HIPERPARAMETROS_CANDIDATO["max_leaf_nodes"],
    "bins": BINS_HISTOGRAMA,
}


def ler_versoes_travadas(caminho=REQUIREMENTS_PADRAO) -> dict[str, str]:
    """Le do `requirements.txt` os pacotes fixados com `==` e devolve nome -> versao.

    Linhas com piso (`>=`), comentarios e linhas em branco ficam de fora: so a
    versao exata e promessa de reproducibilidade.
    """
    travadas = {}
    for linha in Path(caminho).read_text(encoding="utf-8").splitlines():
        linha = linha.split("#", 1)[0].strip()
        casamento = re.fullmatch(r"([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", linha)
        if casamento:
            travadas[casamento.group(1).lower()] = casamento.group(2)
    return travadas


def conferir_versoes(
    caminho=REQUIREMENTS_PADRAO,
    pacotes: tuple[str, ...] = PACOTES_TRAVADOS,
) -> dict[str, str]:
    """Interrompe a execucao se a versao carregada divergir da fixada no requirements.

    Imprimir a versao e confiar que quem le vai reparar na divergencia nao basta:
    no Colab, uma copia pre-instalada pode ficar na frente do `pip install`, a
    celula roda sem erro e a medicao das secoes seguintes passa a ser de outra
    pilha. Falhar aqui transforma essa divergencia silenciosa num erro na
    primeira celula. Tambem falha se algum pacote de `pacotes` nao estiver
    fixado com `==`, porque sem versao exata nao ha contra o que conferir.

    Devolve nome -> versao conferida, para a celula imprimir o que foi validado.
    """
    travadas = ler_versoes_travadas(caminho)
    nao_fixados = [nome for nome in pacotes if nome not in travadas]
    if nao_fixados:
        raise AssertionError(
            f"{nao_fixados} sem versao exata (==) em {Path(caminho).name}; "
            "a medicao deste notebook exige a pilha fixada."
        )
    divergentes = {
        nome: (travadas[nome], metadata.version(nome))
        for nome in pacotes
        if metadata.version(nome) != travadas[nome]
    }
    if divergentes:
        detalhe = "; ".join(
            f"{nome}: requirements {esperada}, carregada {carregada}"
            for nome, (esperada, carregada) in divergentes.items()
        )
        raise AssertionError(
            f"Versao carregada diverge do requirements.txt ({detalhe}). "
            "No Colab, reinicie o ambiente de execucao depois do pip install."
        )
    return {nome: travadas[nome] for nome in pacotes}


def _parametros_hist(random_state: int) -> dict[str, object]:
    return {
        "learning_rate": HIPERPARAMETROS_COMPARACAO["passo"],
        "max_iter": HIPERPARAMETROS_COMPARACAO["arvores"],
        "max_leaf_nodes": HIPERPARAMETROS_COMPARACAO["folhas"],
        "max_bins": HIPERPARAMETROS_COMPARACAO["bins"],
        # Mesma razao do #103: no padrao "auto" a biblioteca separaria sozinha
        # uma fatia aleatoria do ajuste, ignorando o agrupamento por Cliente. Uma
        # medicao de tempo com parada antecipada ligada tambem mediria menos
        # arvores do que as declaradas, e a comparacao perderia o sentido.
        "early_stopping": False,
        "random_state": random_state,
    }


def _parametros_xgb(random_state: int) -> dict[str, object]:
    return {
        "learning_rate": HIPERPARAMETROS_COMPARACAO["passo"],
        "n_estimators": HIPERPARAMETROS_COMPARACAO["arvores"],
        # O XGBoost cresce por profundidade no padrao; `lossguide` com
        # `max_depth=0` e o modo que corresponde ao crescimento por folhas do
        # scikit-learn, que e o que torna `max_leaves` comparavel a
        # `max_leaf_nodes`.
        "grow_policy": "lossguide",
        "max_depth": 0,
        "max_leaves": HIPERPARAMETROS_COMPARACAO["folhas"],
        "max_bin": HIPERPARAMETROS_COMPARACAO["bins"],
        "tree_method": "hist",
        "random_state": random_state,
    }


def construir_candidatos(random_state: int = SEMENTE_PADRAO) -> dict[str, object]:
    """Devolve as duas bibliotecas configuradas com o par equivalente.

    A chave e o nome pelo qual a decisao sera escrita, e e ela que aparece na
    tabela de medicao e no markdown do notebook.
    """
    return {
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            **_parametros_hist(random_state)
        ),
        "XGBClassifier": XGBClassifier(**_parametros_xgb(random_state)),
    }


class _AmostradorDeMemoria:
    """Le o RSS do processo em intervalos fixos e guarda o maior valor visto."""

    def __init__(self, intervalo_s: float = INTERVALO_AMOSTRAGEM_S) -> None:
        self._intervalo_s = intervalo_s
        self._processo = psutil.Process()
        self._parar = threading.Event()
        self._thread = threading.Thread(target=self._amostrar, daemon=True)
        self.inicial_bytes = self._processo.memory_info().rss
        self.pico_bytes = self.inicial_bytes

    def _amostrar(self) -> None:
        while not self._parar.is_set():
            self.pico_bytes = max(self.pico_bytes, self._processo.memory_info().rss)
            self._parar.wait(self._intervalo_s)

    def __enter__(self) -> "_AmostradorDeMemoria":
        self._thread.start()
        return self

    def __exit__(self, *_excecao) -> None:
        self._parar.set()
        self._thread.join()
        # Uma ultima leitura fecha a janela entre a penultima amostra e o fim do
        # ajuste, onde o pico pode acontecer sem nenhuma amostra o ver.
        self.pico_bytes = max(self.pico_bytes, self._processo.memory_info().rss)


def medir_ajuste(
    estimador,
    matriz_treino,
    y_treino,
    random_state: int = SEMENTE_PADRAO,
) -> dict[str, object]:
    """Ajusta o estimador **uma** vez e devolve tempo, pico e delta de memoria.

    O estimador chega clonado para que a medicao nunca aproveite um ajuste
    anterior: um modelo reaproveitado mediria um tempo menor do que o real, que e
    justamente o erro que faria a decisao da biblioteca sair errada.
    """
    candidato = clone(estimador)
    with _AmostradorDeMemoria() as amostrador:
        inicio = time.perf_counter()
        candidato.fit(matriz_treino, y_treino)
        tempo_s = time.perf_counter() - inicio

    mb = 1024 * 1024
    return {
        "tempo_ajuste_s": tempo_s,
        "pico_memoria_mb": amostrador.pico_bytes / mb,
        "delta_memoria_mb": (amostrador.pico_bytes - amostrador.inicial_bytes) / mb,
        "random_state": random_state,
        "n_linhas": int(np.shape(matriz_treino)[0]),
        "n_colunas": int(np.shape(matriz_treino)[1]),
        "estimador": candidato,
    }


def medir_bibliotecas(
    matriz_treino,
    y_treino,
    random_state: int = SEMENTE_PADRAO,
    candidatos: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Tabela de medicao do CR04: uma linha por biblioteca.

    Recebe a matriz **ja transformada** pelo pre-processador do contrato. Passar
    aqui as features cruas e reajustar qualquer coisa seria criar um segundo
    pre-processamento fora do contrato, e a comparacao deixaria de medir so a
    biblioteca.
    """
    candidatos = candidatos or construir_candidatos(random_state)
    linhas = []
    for nome, estimador in candidatos.items():
        medicao = medir_ajuste(estimador, matriz_treino, y_treino, random_state=random_state)
        medicao.pop("estimador")
        linhas.append({"biblioteca": nome, **medicao})
    return pd.DataFrame(linhas).set_index("biblioteca")


def estimar_busca(
    tempo_ajuste_s: float,
    n_iteracoes: int = 40,
    n_folds: int = 5,
) -> float:
    """Horas estimadas para a busca dos cards seguintes, em ajustes sequenciais.

    A conta e deliberadamente otimista: conta apenas os ajustes e ignora
    transformacao, calculo de metrica e o fato de cada fold custar quase o treino
    inteiro. Se nem essa estimativa couber na sessao do Colab, a busca com
    certeza nao cabe.
    """
    if n_iteracoes <= 0 or n_folds <= 0:
        raise ValueError("A busca precisa de pelo menos uma iteracao e um fold.")
    return tempo_ajuste_s * n_iteracoes * n_folds / 3600


def escolher_biblioteca(
    tabela: pd.DataFrame,
    orcamento_horas: float = 3.0,
    n_iteracoes: int = 40,
    n_folds: int = 5,
) -> dict[str, object]:
    """Aplica o criterio do CR05 sobre a tabela de medicao.

    O criterio tem uma ordem, e ela importa: primeiro descarta quem nao cabe no
    orcamento de sessao, porque uma biblioteca que estoura a sessao nao e uma
    opcao mais lenta, e uma opcao que nao existe. Entre as que cabem vence a mais
    barata de manter, e ai o `HistGradientBoostingClassifier` leva por nao
    acrescentar dependencia; ele so perde se for lento a ponto de tirar a busca
    do orcamento.
    """
    estimativas = {
        nome: estimar_busca(linha["tempo_ajuste_s"], n_iteracoes, n_folds)
        for nome, linha in tabela.iterrows()
    }
    cabem = {nome: horas for nome, horas in estimativas.items() if horas <= orcamento_horas}

    if not cabem:
        escolhida = min(estimativas, key=estimativas.get)
        motivo = (
            f"Nenhuma das duas cabe em {orcamento_horas:.1f}h de sessao com "
            f"{n_iteracoes} iteracoes vezes {n_folds} folds; a decisao sobe para a "
            f"dupla de Metricas e Decisoes reduzir a busca. A menos cara e "
            f"{escolhida}, com {estimativas[escolhida]:.2f}h."
        )
    elif "HistGradientBoostingClassifier" in cabem:
        escolhida = "HistGradientBoostingClassifier"
        motivo = (
            f"Cabe em {cabem[escolhida]:.2f}h, dentro do orcamento de "
            f"{orcamento_horas:.1f}h, e nao acrescenta dependencia ao projeto nem "
            f"instalacao a sessao do Colab."
        )
    else:
        escolhida = min(cabem, key=cabem.get)
        motivo = (
            f"O HistGradientBoostingClassifier nao cabe no orcamento de "
            f"{orcamento_horas:.1f}h; {escolhida} cabe, com {cabem[escolhida]:.2f}h, "
            f"e a dependencia extra se paga por viabilizar a busca."
        )

    return {
        "escolhida": escolhida,
        "estimativas_horas": estimativas,
        "orcamento_horas": orcamento_horas,
        "n_iteracoes": n_iteracoes,
        "n_folds": n_folds,
        "justificativa": motivo,
    }


# ---------------------------------------------------------------------------
# Pipeline do Random Forest (card 08A, #187)
# ---------------------------------------------------------------------------
#
# O Random Forest entra como um unico objeto que encadeia o pre-processador do
# contrato e o estimador, pelo mesmo motivo do pipeline da logistica (#208):
# enquanto o `ColumnTransformer` vive solto, quem ajusta decide onde ajustar, e a
# escolha mais comoda, ajustar uma vez no treino inteiro e so transformar dentro
# dos folds, e a que vaza. Dentro de um `Pipeline`, a busca do D3 chama `fit` no
# objeto inteiro a cada fold e o pre-processador e reajustado ali.
#
# Arvore nao precisa de escalonamento, e seria tentador dispensar o
# `RobustScaler` aqui. Nao se dispensa: o CR01 pede o pre-processador do
# contrato sem alteracao, e a comparacao entre as quatro familias so mede
# algoritmo se todas recebem a mesma matriz. O escalonamento monotono nao muda
# nenhum corte que a arvore escolheria, entao o custo de mante-lo e zero.

# Nomes dos passos. Sao os mesmos do `pipeline_logistica` (#208) de proposito: a
# busca enderecada hiperparametro por `passo__parametro`, e a Fernanda (#188)
# monta o do boosting com estes mesmos nomes para que as duplas escrevam as
# grades do mesmo jeito.
PASSO_PREPARO = "preparo"
PASSO_MODELO = "modelo"


def criar_pipeline_random_forest(
    preprocessador,
    random_state: int = SEMENTE_PADRAO,
    **hiperparametros,
) -> Pipeline:
    """Encadeia o pre-processador do contrato e o `RandomForestClassifier`.

    `preprocessador` e `preparo["preprocessador"]`, o `ColumnTransformer` que
    `matriz.preparar_matriz` monta a partir da allowlist. Ele entra **clonado**:
    mesma especificacao, sem o estado que `preparar_matriz` ja ajustou no
    treino. Remontar o `ColumnTransformer` aqui criaria uma segunda definicao do
    contrato; aproveitar o objeto ja ajustado faria o primeiro `fit` do fold
    partir de medianas e quartis que viram as linhas de validacao daquele fold.

    Sem `hiperparametros`, o estimador nasce no padrao da biblioteca, que e o que
    o card pede para a linha de base: quem escolhe profundidade, numero de
    arvores e `class_weight` e a busca do D3 (#189), nao esta funcao. O argumento
    existe para a busca e os testes montarem uma combinacao sem escrever o
    `Pipeline` a mao.

    `random_state` e obrigatorio no sentido pratico, e nao so um padrao: o Random
    Forest sorteia a amostra bootstrap e as colunas de cada divisao, e sem
    semente fixa dois ajustes identicos devolvem florestas diferentes, o que
    tornaria qualquer diferenca de metrica na busca indistinguivel de sorte.
    `n_jobs` fica em 1 por padrao, e nao em -1, pelo mesmo motivo. Com varias
    threads as arvores continuam as mesmas, porque cada uma recebe a semente
    derivada de `random_state` antes da distribuicao, mas `predict_proba` soma os
    votos na ordem em que as threads terminam, e a ordem muda o ultimo bit do
    ponto flutuante. Num empate exato em 0,5 isso vira o rotulo, e dois ajustes
    identicos passam a dar F2 diferente. O paralelismo que nao custa
    reprodutibilidade fica um nivel acima, na busca do #189, que distribui os
    ajustes inteiros entre processos. Quem quiser threads mesmo assim passa
    `n_jobs` em `hiperparametros`.
    """
    parametros = {"n_jobs": 1, **hiperparametros, "random_state": random_state}
    return Pipeline([
        (PASSO_PREPARO, clone(preprocessador)),
        (PASSO_MODELO, RandomForestClassifier(**parametros)),
    ])


def medir_linha_de_base(
    pipeline: Pipeline,
    x_treino,
    y_treino,
    x_avaliacao,
    y_avaliacao,
    avaliar: Callable,
) -> dict[str, object]:
    """Ajusta o pipeline no treino, pontua a avaliacao e devolve o que `avaliar` calcular.

    E a referencia dos cards #187 e #188: o desempenho do Random Forest e do
    Gradient Boosting **antes** de qualquer busca. Sem ela, o ganho que o #189 e
    o #190 reportarem nao tem contra o que ser lido.

    `x_treino` e `preparo["x"]["treino"]`, a matriz **crua**, e nao
    `preparo["matrizes"]["treino"]`: passar a matriz ja transformada faria o
    `ColumnTransformer` do pipeline ser ajustado sobre a saida de outro, sem erro
    nenhum e com colunas que a base nao tem.

    `avaliar` e a funcao do card 05 (#241), e entra como argumento por dois
    motivos: o modulo continua importavel enquanto aquele card nao esta em
    `develop`, e nenhuma metrica e nem particao e definida aqui. O dicionario de
    metricas volta como `avaliar` o devolveu, sem renomear chave, porque a tabela
    comparativa do card 18A.1 depende de as quatro duplas falarem a mesma
    lingua. `y_proba` e a coluna 1 de `predict_proba`, a probabilidade de
    Detrator: Precisao Media e ROC-AUC sao calculadas sobre o score continuo, e a
    coluna errada devolveria um numero valido e errado.
    """
    if not callable(avaliar):
        raise TypeError(
            "avaliar precisa ser a funcao do card 05 (#241), com assinatura "
            "avaliar(y_true, y_pred, y_proba)"
        )

    inicio = time.perf_counter()
    pipeline.fit(x_treino, y_treino)
    y_pred = pipeline.predict(x_avaliacao)
    y_proba = pipeline.predict_proba(x_avaliacao)[:, 1]
    metricas = avaliar(y_avaliacao, y_pred, y_proba)
    tempo_total_s = time.perf_counter() - inicio

    return {
        "metricas": metricas,
        "tempo_total_s": tempo_total_s,
        "random_state": pipeline.named_steps[PASSO_MODELO].random_state,
        "n_treino": int(len(y_treino)),
        "n_avaliacao": int(len(y_avaliacao)),
    }


# ---------------------------------------------------------------------------
# Pipeline do Gradient Boosting (card 11, #188)
# ---------------------------------------------------------------------------
#
# O card 11 foi dividido para liberar o D3 inteiro para a busca do #190: o
# Gradient Boosting entra aqui como um unico objeto que encadeia o
# pre-processador do contrato e o estimador, pela mesma razao do pipeline da
# logistica (#208) e do Random Forest (#187). Enquanto o `ColumnTransformer`
# vive solto, quem ajusta decide onde ajustar, e a escolha mais comoda, ajustar
# uma vez no treino inteiro e so transformar dentro dos folds, e a que vaza.
# Dentro de um `Pipeline`, a busca do #190 chama `fit` no objeto inteiro a cada
# fold e o pre-processador e reajustado ali.
#
# A biblioteca e o `HistGradientBoostingClassifier`: o #185 decidiu por ele em
# vez do `XGBClassifier` porque e o gradient boosting do proprio scikit-learn,
# ja fixado no requirements, sem dependencia extra para o Colab instalar.
#
# Os nomes dos passos e `medir_linha_de_base` sao os do Random Forest, logo
# acima: a busca do #190 enderecada hiperparametro por `passo__parametro` do
# mesmo jeito, e a linha de base e medida pela mesma funcao.


def criar_pipeline_gradient_boosting(
    preprocessador,
    random_state: int = SEMENTE_PADRAO,
    **hiperparametros,
) -> Pipeline:
    """Encadeia o pre-processador do contrato e o `HistGradientBoostingClassifier`.

    `preprocessador` e `preparo["preprocessador"]`, o `ColumnTransformer` que
    `matriz.preparar_matriz` monta a partir da allowlist. Ele entra **clonado**:
    mesma especificacao, sem o estado que `preparar_matriz` ja ajustou no
    treino. Remontar o `ColumnTransformer` aqui criaria uma segunda definicao do
    contrato; aproveitar o objeto ja ajustado faria o primeiro `fit` do fold
    partir de medianas e quartis que viram as linhas de validacao daquele fold.

    Sem `hiperparametros`, o estimador nasce no padrao da biblioteca em todo
    eixo de busca (`learning_rate`, `max_iter`, `max_leaf_nodes`,
    `l2_regularization`, `min_samples_leaf`), que e o que o card pede para a
    linha de base: quem escolhe esses valores e a busca do D3 (#190), nao esta
    funcao. `**hiperparametros` existe para a busca e os testes montarem uma
    combinacao especifica sem escrever o `Pipeline` a mao.

    O unico padrao que esta funcao nao deixa passar e `early_stopping`. No
    padrao `"auto"` da biblioteca, ele liga sozinho acima de 10 mil linhas e
    separa uma fatia **aleatoria** do ajuste para medir a parada antecipada;
    essa fatia nao respeita `ID_GOLDENRECORD`, entao respostas do mesmo Cliente
    cairiam dos dois lados, o mesmo vazamento que `modelo.HIPERPARAMETROS_CANDIDATO`
    (#103) documenta e evita. A validacao deste projeto sao os folds do
    contrato, e nenhuma outra; por isso `early_stopping=False` e fixado aqui, e
    nao deixado como argumento de `hiperparametros`, para que nenhuma chamada
    religue-o por engano.

    `random_state` e obrigatorio no sentido pratico: o `HistGradientBoostingClassifier`
    sorteia o subamostra dos bins do histograma, e sem semente fixa dois ajustes
    identicos produzem arvores diferentes, o que tornaria qualquer diferenca de
    metrica na busca do #190 indistinguivel de sorte.
    """
    parametros = {**hiperparametros, "early_stopping": False, "random_state": random_state}
    return Pipeline([
        (PASSO_PREPARO, clone(preprocessador)),
        (PASSO_MODELO, HistGradientBoostingClassifier(**parametros)),
    ])


# ---------------------------------------------------------------------------
# Melhor Gradient Boosting, reconstruido pelo JSON da busca (#190)
# ---------------------------------------------------------------------------
#
# A busca do #190 (`busca_gradient_boosting.py`) grava os hiperparametros
# vencedores em `documents/extras/resultados/`, e nao o modelo ajustado: o
# modelo nao vai para o git, por tamanho, e um pickle so abriria na mesma versao
# do scikit-learn. O caminho vive aqui, e nao no modulo da busca, porque e esta
# funcao que a dupla de Metricas e Decisoes importa, e ela nao pode depender de
# `sklearn.model_selection` (ver `test_pipeline_nao_importa_selecao_de_modelo`).
ARQUIVO_HIPERPARAMETROS_GRADIENT_BOOSTING = (
    Path(__file__).resolve().parents[1]
    / "documents" / "extras" / "resultados" / "hiperparametros_gradient_boosting.json"
)


def melhor_gradient_boosting(preprocessador, caminho=None) -> Pipeline:
    """Remonta, **sem ajustar**, o pipeline vencedor da busca do #190.

    E o ponto de entrada da dupla de Metricas e Decisoes: recebe
    `preparo["preprocessador"]` e devolve o mesmo `Pipeline` que a busca elegeu,
    pronto para `fit` no treino e para `avaliar` (#241). Entregar o pipeline nao
    ajustado, e nao o `best_estimator_`, e o que deixa quem compara os modelos
    decidir onde ajustar, com o mesmo contrato das outras tres duplas.

    O pipeline sai de `criar_pipeline_gradient_boosting` (#188), e nao de um
    `Pipeline` montado aqui, para que o reconstruido seja o mesmo objeto que a
    busca varreu, com o mesmo `early_stopping=False`. A semente vem do JSON, e nao
    de `SEMENTE_PADRAO`: com outra semente o ensemble reconstruido seria outro, e a
    metrica medida sobre ele deixaria de bater com a registrada no notebook.
    Registro sem `random_state` e recusado, pelo mesmo motivo.

    `caminho` existe para os testes apontarem para um JSON temporario; sem ele,
    le `ARQUIVO_HIPERPARAMETROS_GRADIENT_BOOSTING`, resolvido na hora da chamada.
    """
    caminho = Path(caminho or ARQUIVO_HIPERPARAMETROS_GRADIENT_BOOSTING)
    registro = json.loads(caminho.read_text(encoding="utf-8"))

    faltando = {"hiperparametros", "random_state"} - set(registro)
    if faltando:
        raise ValueError(
            f"{caminho.name} sem {sorted(faltando)}: sem a semente e os "
            "hiperparametros da busca do #190 o pipeline nao e o vencedor"
        )

    return criar_pipeline_gradient_boosting(
        preprocessador,
        random_state=int(registro["random_state"]),
        **registro["hiperparametros"],
    )

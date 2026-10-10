"""Tabela comparativa final das quatro duplas de modelagem (card 14A, #249).

**O que este modulo estende, e por que nao substitui.** A Secao 4.3.2 (#108) ja
construiu `modelo.tabela_comparativa`: uma linha por modelo, medida na mesma
particao de teste e no mesmo `k` de fila, com o limiar tirado da capacidade de
contato e nao de uma metrica maximizada. Essa e exatamente a comparacao que o
ART.7 precisa entre os quatro candidatos novos (Regressao Logistica #211,
Arvore #232, Random Forest #189, Gradient Boosting #190) e os dois pisos
(DummyClassifier #197/#220 e o primeiro candidato da Secao 4.3.2). Reescrever a
busca do limiar ou o laco de montagem duplicaria uma decisao ja tomada e correria
o risco de divergir dela sem ninguem perceber. Por isso `consolidar_tabela_final`
chama `modelo.tabela_comparativa` para o limiar, a fila e a comparabilidade
(CR02, CR03), e so acrescenta as colunas que faltam.

**Por que as colunas nao vem de `tabela_comparativa` direto.** Aquela funcao e
de 2026-09-08 (#108), antes de o protocolo de avaliacao (#241) existir: ela
chama `precisao_media`/`roc_auc`/`cobertura`, nomes proprios, calculados a mao.
A partir do #241 o projeto tem uma unica fonte para essas metricas,
`avaliacao.avaliar(y_true, y_pred, y_proba)`, e a tabela do card 18A.1 depende
de todas as duplas falarem essa lingua. Por isso este modulo recalcula
Sensibilidade, Precisao Media e ROC-AUC chamando `avaliar` sobre o **mesmo**
`y_pred` que o limiar de `tabela_comparativa` produziria — o numero bate com o
de `cobertura`/`precisao_media`/`roc_auc` a menos de arredondamento, a diferenca
e so de qual funcao e a fonte de verdade dali para frente.

**F2 e um valor diferente dos outros tres, de proposito.** A Descricao do card
pede F2 "so como registro do criterio de busca": e o F2 que a busca de cada
modelo maximizou durante a validacao cruzada (por exemplo a primeira linha de
`documents/extras/resultados/hiperparametros_logistica.json`), nao um F2 recalculado no limiar
operacional. Calcula-lo no limiar de capacidade misturaria duas perguntas
diferentes — "qual configuracao a busca escolheu" e "como o vencedor se sai na
fila que a operacao consegue perseguir" — que o CR02 pede para manter
separadas. Por isso `f2_da_busca` entra como argumento, um numero por modelo, e
nao e recalculado aqui.

**Modelo ausente nao e excecao.** No dia em que este card foi escrito, a Arvore
(#232) e o Random Forest (#189) ainda nao tinham vencedor versionado. A funcao
aceita `None` no lugar do estimador: a linha entra na tabela com todas as
metricas em `nan` e `disponivel=False`, em vez de a chamada inteira falhar por
causa de um modelo que ainda nao chegou. E o mesmo padrao que `avaliar` usa para
o lote sem Detrator e que `escolher_melhor_ensemble` (#191) usa para o ensemble
sem numero: a tabela nunca inventa metrica para preencher uma lacuna.

**Cada modelo recebe a matriz que ele mesmo espera, e nao uma unica matriz
comum.** A primeira versao deste modulo passava uma so `x_teste_transformado`
para todos os candidatos, herdando a suposicao de `modelo.tabela_comparativa`
(#108): a de que todo `modelo` recebido e um estimador desnudo, ajustado sobre a
matriz ja transformada, como era o unico candidato da Secao 4.3.2. Essa suposicao
nao vale mais a partir do #187/#208: os quatro candidatos novos (e o vencedor da
Regressao Logistica, #211) sao `Pipeline`s que **incluem o proprio
`ColumnTransformer`** e esperam o `DataFrame` cru do contrato, com as colunas
nomeadas que o pre-processador seleciona por string. Passar a matriz ja
transformada a um desses pipelines faz o `ColumnTransformer` de dentro tentar
selecionar coluna por nome num `ndarray` sem nomes, e falhar com
`ValueError: Specifying the columns using strings is only supported for
dataframes` — o defeito que a revisao da !120 relatou. Por isso `x_teste` entra
como um dicionario por modelo: o primeiro candidato da Secao 4.3.2 (um estimador
desnudo) recebe a matriz transformada, e cada `Pipeline` novo recebe o `DataFrame`
cru. A funcao chama `modelo.tabela_comparativa` uma vez por modelo, cada vez com
a matriz que aquele modelo espera, em vez de uma unica chamada com uma matriz
comum — o reaproveitamento da logica de limiar (CR03) continua o mesmo, so a
matriz que a acompanha muda por linha.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# O modulo localiza os proprios vizinhos em vez de depender de quem o importa,
# pelo mesmo motivo de `ensembles.py` e `explicabilidade.py`: sob pytest o
# conftest prepara o caminho, no Colab e na execucao direta nao ha ninguem para
# preparar.
_RAIZ = Path(__file__).resolve().parents[1]
for _pasta in ("src", "scripts"):
    _caminho = str(_RAIZ / _pasta)
    if _caminho not in sys.path:
        sys.path.insert(0, _caminho)

import modelo  # noqa: E402
from avaliacao import avaliar as avaliar_padrao  # noqa: E402

COLUNAS_METRICAS = ("F2", "Sensibilidade", "Precisão Média", "ROC-AUC")


def consolidar_tabela_final(
    modelos: dict[str, object | None],
    x_teste: dict[str, object],
    y_teste: pd.Series,
    k: int,
    f2_da_busca: dict[str, float] | None = None,
    avaliar: Callable = avaliar_padrao,
) -> pd.DataFrame:
    """Uma linha por modelo, com F2 (registro), Sensibilidade, Precisão Média e ROC-AUC.

    `modelos` mapeia o nome do modelo (inclui os quatro candidatos, o baseline
    dummy e o primeiro candidato da Seção 4.3.2) ao estimador já ajustado, com
    `predict_proba`. Um valor `None` marca um modelo sem vencedor versionado
    ainda: a linha dele entra com as métricas em `nan` e `disponivel=False`, sem
    lançar exceção — a tabela pode ser montada antes de as quatro duplas
    terminarem, e ganha linha real conforme cada uma chega.

    `x_teste` mapeia o **mesmo** nome de modelo à matriz que **aquele** modelo
    espera receber — não é uma matriz comum para todos. Um `Pipeline` que
    encadeia `ColumnTransformer` e estimador (os quatro candidatos novos e o
    vencedor da Regressão Logística, #211) precisa do `DataFrame` **cru** do
    contrato; um estimador desnudo, ajustado direto sobre a matriz já
    transformada (o primeiro candidato da Seção 4.3.2), precisa da matriz
    transformada. Passar a matriz errada não é detectado aqui: quem chama
    precisa saber que tipo de objeto cada `modelos[nome]` é. Todas as entradas
    de `x_teste` e de `y_teste` descrevem a **mesma** partição de teste, só a
    representação (crua ou transformada) muda por modelo.

    `k` é o tamanho da fila de contato, **fixo entre todos os modelos**: é isso
    que faz `limiar_por_capacidade` devolver um corte por capacidade
    operacional, e não por métrica (CR02).

    `f2_da_busca` é opcional, nome do modelo -> F2 que a respectiva busca de
    hiperparâmetros registrou (ex.: a primeira linha de
    `documents/extras/resultados/hiperparametros_logistica.json`). Sem entrada para um modelo, a
    coluna `F2` fica `nan` para ele: é um registro externo, não uma métrica
    recalculada aqui.
    """
    if not modelos:
        raise ValueError("nenhum modelo recebido: a tabela final precisa de pelo menos um")

    f2_da_busca = f2_da_busca or {}
    disponiveis = {nome: est for nome, est in modelos.items() if est is not None}
    ausentes = [nome for nome, est in modelos.items() if est is None]

    faltando_x = [nome for nome in disponiveis if nome not in x_teste]
    if faltando_x:
        raise KeyError(
            f"x_teste sem entrada para {faltando_x}: cada modelo disponivel precisa "
            "da propria matriz (crua para Pipeline, transformada para estimador desnudo)."
        )

    linhas: dict[str, dict[str, object]] = {}

    for nome, estimador in disponiveis.items():
        x_do_modelo = x_teste[nome]
        base = modelo.tabela_comparativa({nome: estimador}, x_do_modelo, y_teste, k)
        score = estimador.predict_proba(x_do_modelo)[:, 1]
        limiar = base.loc[nome, "limiar"]
        y_pred = (np.asarray(score) >= limiar).astype(int)
        metricas = avaliar(y_teste, y_pred, score)
        linhas[nome] = {
            "disponivel": True,
            "F2": f2_da_busca.get(nome, float("nan")),
            "Sensibilidade": metricas["Sensibilidade"],
            "Precisão Média": metricas["Precisão Média"],
            "ROC-AUC": metricas["ROC-AUC"],
            "limiar": limiar,
            "n_na_fila": int(base.loc[nome, "n_na_fila"]),
            "fila_comparavel": bool(base.loc[nome, "fila_comparavel"]),
        }

    for nome in ausentes:
        linhas[nome] = {
            "disponivel": False,
            "F2": f2_da_busca.get(nome, float("nan")),
            "Sensibilidade": float("nan"),
            "Precisão Média": float("nan"),
            "ROC-AUC": float("nan"),
            "limiar": float("nan"),
            "n_na_fila": None,
            "fila_comparavel": False,
        }

    tabela = pd.DataFrame.from_dict(
        {nome: linhas[nome] for nome in modelos}, orient="index"
    )
    tabela.index.name = "modelo"
    return tabela

"""Matriz de modelagem: particao em tres conjuntos, features e pre-processamento.

Este modulo e a ponte entre a base analitica e o modelo. Ele encadeia tres
etapas que precisam acontecer nesta ordem, e a ordem e o ponto:

1. particionar por data e por Cliente (`split.dividir`);
2. selecionar apenas as features disponiveis no momento da predicao
   (`selecionar_features_score_pos_viagem`);
3. ajustar o pre-processador **somente no treino** e aplica-lo aos demais.

Inverter 2 e 3 seria inofensivo. Inverter 1 e 3 nao: ajustar imputacao,
escalonamento ou codificacao sobre a base inteira faria a mediana do treino
carregar informacao de 2026, e a metrica de teste passaria a medir um modelo que
ja viu o periodo que deveria prever.

Por que nao reaproveitar `criar_preprocessador_modelagem` de
`scripts/preprocessamento_nps.py`: aquela funcao entrega dois conjuntos, e o
artefato exige treino, validacao e teste; alem disso ela chama uma divisao que
levanta excecao quando alguma linha nao tem data, situacao que ja ocorreu na
base analitica por um bug de concatenacao (ver a docstring de `dividir`, em
`split.py`, ja corrigido). O que se reaproveita aqui e o que continua correto
la: a allowlist de features e o desenho do pre-processador.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# O modulo localiza os proprios vizinhos em vez de depender de quem o importa.
# Sem isso, funciona sob pytest, que ja prepara o caminho pelo conftest, e falha
# no Colab e em qualquer execucao direta.
_RAIZ = Path(__file__).resolve().parents[1]
for _pasta in ("src", "scripts"):
    _caminho = str(_RAIZ / _pasta)
    if _caminho not in sys.path:
        sys.path.insert(0, _caminho)

import features  # noqa: E402
import split  # noqa: E402

# A allowlist vive em scripts/ desde o pre-processamento e continua sendo a
# fonte unica: duplica-la aqui abriria a porta para as duas listas divergirem.
from preprocessamento_nps import (  # noqa: E402
    COLUNA_ALVO,
    FEATURE_SET_V1,
    selecionar_features_score_pos_viagem,
    validar_contrato_dados_score_pos_viagem,
)

PREFIXOS_PROIBIDOS = ("NPS_", "SUB_")
# As features de historico (src/features.py) nao entram em FEATURE_SET_V1
# porque essa allowlist e o contrato canonico do score pos-viagem, compartilhado
# com o pre-processamento e testado em outros arquivos; historico e um bloco a
# parte, com sua propria trava de anterioridade, por isso a permissao e local.
COLUNAS_HISTORICO_PERMITIDAS = frozenset(features.FEATURES_HISTORICO)


def _classificar_colunas(x: pd.DataFrame, selecionadas: list[str]) -> tuple[list[str], list[str]]:
    categoricas = [
        coluna for coluna in selecionadas
        if pd.api.types.is_bool_dtype(x[coluna])
        or pd.api.types.is_object_dtype(x[coluna])
        or pd.api.types.is_string_dtype(x[coluna])
        or isinstance(x[coluna].dtype, pd.CategoricalDtype)
    ]
    return [c for c in selecionadas if c not in categoricas], categoricas


def _montar_preprocessador(numericas: list[str], categoricas: list[str]) -> ColumnTransformer:
    """Imputacao, escalonamento robusto e codificacao one-hot.

    `add_indicator` preserva a informacao de que o valor era ausente, em vez de
    deixar a mediana se passar por medicao real. O escalonamento e robusto
    porque as tres variaveis quantitativas tem assimetria alta e cauda longa,
    conforme o anexo A.1: com media e desvio, um unico voo de 72 horas
    deslocaria a escala inteira. `handle_unknown='ignore'` evita que uma
    categoria vista so na validacao ou no teste interrompa a transformacao.
    """
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("escalar", RobustScaler()),
        ]), numericas),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="NAO_INFORMADO")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categoricas),
    ], remainder="drop")


def conferir_contrato_da_matriz(
    x: pd.DataFrame,
    preprocessador=None,
    colunas_extras_permitidas: frozenset[str] = frozenset(),
) -> None:
    """Recusa a matriz se uma coluna de pesquisa entrar nela, ou se faltar alguma da allowlist.

    O modelo pontua o risco na janela entre o voo e a resposta a pesquisa, entao
    qualquer coluna `NPS_` ou `SUB_` so existe depois do instante da predicao.
    Usa-la seria prever a resposta com a propria resposta. A conferencia e feita
    nas duas pontas porque elas pegam falhas diferentes: na entrada, uma
    allowlist alterada; na saida, uma transformacao que reintroduza a coluna sob
    outro nome.

    A conferencia de excesso, sozinha, nao bastava: `selecionar_features_score_pos_viagem`
    apenas registra uma feature ausente na fonte e segue adiante, entao um nome
    defasado na allowlist — como `PERFIL_TUDOAZUL` no lugar de `TIER_VIAGEM`,
    corrigido nesta mesma MR — treinava com uma feature a menos, em silencio,
    sem que nada aqui recusasse a matriz incompleta. A conferencia de falta
    fecha essa lacuna.

    `colunas_extras_permitidas` estende a allowlist de excesso para blocos de
    feature que tem sua propria trava de anterioridade, como o historico de
    Cliente (ver `features.adicionar_historico`), sem misturar as duas listas.
    A conferencia de falta continua restrita a `FEATURE_SET_V1`: o
    historico e opcional (`incluir_historico=False` o omite), entao exigi-lo
    aqui quebraria esse caso.
    """
    permitidas = set(FEATURE_SET_V1) | set(colunas_extras_permitidas)
    fora_da_allowlist = set(x.columns) - permitidas
    if fora_da_allowlist:
        raise AssertionError(
            "A matriz usa colunas fora da allowlist do score pos-viagem: "
            f"{sorted(fora_da_allowlist)}"
        )

    faltando_da_allowlist = set(FEATURE_SET_V1) - set(x.columns)
    if faltando_da_allowlist:
        raise AssertionError(
            "Features da allowlist do score pos-viagem ausentes na matriz: "
            f"{sorted(faltando_da_allowlist)}"
        )

    entrada_proibida = [c for c in x.columns if c.startswith(PREFIXOS_PROIBIDOS)]
    if entrada_proibida:
        raise AssertionError(
            f"Coluna de pesquisa entrou na matriz do modelo: {entrada_proibida}"
        )

    if preprocessador is not None:
        saida_proibida = [
            c for c in preprocessador.get_feature_names_out()
            if any(p in c for p in PREFIXOS_PROIBIDOS)
        ]
        if saida_proibida:
            raise AssertionError(
                "Coluna de pesquisa apareceu na saida do pre-processador: "
                f"{saida_proibida}"
            )


def preparar_matriz(
    df: pd.DataFrame,
    corte_validacao: str,
    corte_teste: str,
    sem_data: str = "treino",
    verificar_anterioridade: bool = True,
    incluir_historico: bool = True,
) -> dict[str, object]:
    """Devolve as tres particoes, as matrizes transformadas e os metadados.

    `verificar_anterioridade` so faz sentido quando `sem_data='treino'` e
    controla apenas se **esta funcao** relata o resultado da verificacao em
    `metadados['anterioridade_sem_data']`; a garantia de seguranca em si nao
    depende deste parametro. `split.dividir` verifica a anterioridade por
    conta propria, de forma incondicional, sempre que `sem_data='treino'` e
    existe alguma linha sem data — desligar este parametro nao abre uma
    brecha, so faz `preparar_matriz` nao repetir o calculo para relata-lo.

    `incluir_historico` acrescenta as features de Cliente e historico (ver
    `features.adicionar_historico`) a matriz. O calculo acontece sobre o `df`
    inteiro, antes do particionamento: a anterioridade de cada linha depende
    apenas da ordem de respostas do proprio Cliente, nao da particao em que ela
    cai, entao calcular por particao separadamente so repetiria o mesmo
    resultado com mais codigo.
    """
    if COLUNA_ALVO not in df.columns:
        raise KeyError(f"A modelagem exige a coluna-alvo {COLUNA_ALVO}.")

    anterioridade = None
    if sem_data == "treino" and verificar_anterioridade:
        anterioridade = split.verificar_anterioridade_sem_data(df)

    if incluir_historico:
        df = features.adicionar_historico(df)

    contrato_dados = validar_contrato_dados_score_pos_viagem(df, exigir_alvo=True)

    particoes, metadados = split.dividir(
        df, corte_validacao, corte_teste, sem_data=sem_data,
    )
    split.conferir(particoes, metadados=metadados)
    linhas_sem_cliente_split = metadados.get("linhas_sem_cliente_excluidas")
    if linhas_sem_cliente_split is None:
        raise ValueError("O split não informou a contagem de linhas sem ID_GOLDENRECORD.")
    if linhas_sem_cliente_split != contrato_dados["linhas_sem_cliente"]:
        raise ValueError("A contagem de linhas sem ID_GOLDENRECORD divergiu entre o contrato e o split.")

    x_bruto, selecionadas, ausentes = selecionar_features_score_pos_viagem(df)
    numericas, categoricas = _classificar_colunas(x_bruto, selecionadas)

    colunas_historico: tuple[str, ...] = ()
    if incluir_historico:
        colunas_historico = features.FEATURES_HISTORICO
        x_bruto = pd.concat([x_bruto, df.loc[x_bruto.index, list(colunas_historico)]], axis=1)
        numericas = numericas + list(colunas_historico)

    x = {nome: x_bruto.loc[p.index] for nome, p in particoes.items()}
    y = {nome: p[COLUNA_ALVO] for nome, p in particoes.items()}
    grupos = {nome: p[split.COLUNA_CLIENTE] for nome, p in particoes.items()}

    conferir_contrato_da_matriz(x["treino"], colunas_extras_permitidas=COLUNAS_HISTORICO_PERMITIDAS)

    preprocessador = _montar_preprocessador(numericas, categoricas)
    # O ajuste ve apenas o treino; validacao e teste sao somente transformados.
    matrizes = {"treino": preprocessador.fit_transform(x["treino"])}
    for nome in ("validacao", "teste"):
        matrizes[nome] = preprocessador.transform(x[nome])

    conferir_contrato_da_matriz(
        x["treino"], preprocessador, colunas_extras_permitidas=COLUNAS_HISTORICO_PERMITIDAS
    )

    cobertura_historico = features.cobertura_do_historico(df) if incluir_historico else None

    metadados = {
        **metadados,
        "features_usadas": selecionadas + list(colunas_historico),
        "features_ausentes_na_fonte": ausentes,
        "numericas": numericas,
        "categoricas": categoricas,
        "colunas_da_matriz": int(matrizes["treino"].shape[1]),
        "anterioridade_sem_data": anterioridade,
        "cobertura_historico": cobertura_historico,
        "contrato_dados": contrato_dados,
    }
    return {
        "particoes": particoes,
        "x": x,
        "y": y,
        "grupos": grupos,
        "matrizes": matrizes,
        "preprocessador": preprocessador,
        "metadados": metadados,
    }


CHAVES_DE_CORTE = ("corte_validacao", "corte_teste")


def cortes_do_preparo(preparo: dict[str, object]) -> dict[str, str]:
    """As duas datas de corte com que `preparar_matriz` montou as particoes.

    Sao elas que dizem de que matriz um numero saiu. Uma busca de hiperparametros
    e a medicao do vencedor so falam da mesma coisa se usaram os mesmos cortes, e
    o nome `"validacao"` sozinho nao garante isso: toda matriz tem uma validacao.
    """
    metadados = preparo["metadados"]
    return {chave: str(metadados[chave]) for chave in CHAVES_DE_CORTE}


def conferir_cortes(cortes: dict[str, str]) -> dict[str, str]:
    """Recusa um registro de cortes incompleto e devolve so as duas chaves, como texto."""
    faltando = [chave for chave in CHAVES_DE_CORTE if chave not in cortes]
    if faltando:
        raise ValueError(f"cortes sem {faltando}: use matriz.cortes_do_preparo(preparo).")
    return {chave: str(cortes[chave]) for chave in CHAVES_DE_CORTE}


def resumo_da_matriz(preparo: dict[str, object]) -> pd.DataFrame:
    """Tabela por particao com n, Clientes, prevalencia e forma da matriz."""
    linhas = []
    for nome in split.PARTICOES:
        y = preparo["y"][nome]
        linhas.append({
            "particao": nome,
            "n": len(y),
            "clientes": int(preparo["grupos"][nome].nunique()),
            "prevalencia_pct": round(float(y.mean()) * 100, 2) if len(y) else 0.0,
            "colunas_matriz": int(preparo["matrizes"][nome].shape[1]),
        })
    return pd.DataFrame(linhas).set_index("particao")

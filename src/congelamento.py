"""Congelamento e carregamento unico das particoes de treino, validacao e teste.

A divisao de `src/split.dividir` e temporal e por Cliente, logo deterministica:
a mesma base com os mesmos cortes sempre produz as mesmas particoes.
Determinismo, porem, nao e congelamento. Basta a base analitica ser regerada com
uma linha a mais, ou alguem chamar `dividir` com uma data de corte ligeiramente
diferente, para que os conjuntos mudem sem que ninguem perceba — e as metricas
que os cards #103 a #110 comparam entre si na Secao 4.4 deixam de medir a mesma
coisa.

Este modulo fecha essa porta. Ele persiste os indices das tres particoes como
artefato local, calcula um hash sobre o CONTEUDO desses indices e registra ao
lado a semente fixada, a impressao digital da base de origem e os parametros de
corte usados. Ao carregar, confere as tres coisas: hash contra o conteudo,
impressao digital contra a base atual e parametros contra os pedidos. Qualquer
divergencia interrompe a execucao em vez de devolver uma divisao que ja nao e a
acordada.

O artefato NAO e versionado. Fica sob `data/`, ja coberto pelo `.gitignore`,
conforme o Termo de Abertura, que veda publicar dados do parceiro: indices sao
posicoes de linhas de uma base que nao pode sair do ambiente local.

Ponto de entrada unico: `obter_particoes`. Os cards de modelagem chamam essa
funcao em vez de chamarem `dividir` por conta propria; recriar a divisao em cada
card anularia o congelamento, que e exatamente o que este modulo existe para
impedir.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from split import PARTICOES, dividir

# Semente do projeto. A divisao em si nao a utiliza, por ser temporal, mas a
# validacao cruzada por Cliente, as buscas de hiperparametros e qualquer
# amostragem dos cards seguintes precisam partir do mesmo numero. Fica aqui para
# que exista uma unica fonte, e nao uma constante repetida em cada notebook.
SEMENTE = 42

# Versao do formato do artefato. Se o payload mudar de forma, esta constante
# sobe e o carregamento recusa artefatos antigos, em vez de interpretar campo
# ausente como valor valido.
VERSAO_ARTEFATO = 1

# Sob data/, portanto ignorado pelo git. O caminho e relativo a raiz do projeto,
# resolvida a partir da localizacao deste arquivo, para que notebook, script e
# pytest cheguem ao mesmo lugar independentemente do diretorio de trabalho.
RAIZ = Path(__file__).resolve().parents[1]
ARTEFATO_PADRAO = RAIZ / "data" / "processed" / "particoes_modelagem.json"


def hash_indices(indices_por_particao: dict[str, list[int]]) -> str:
    """Hash sobre o conteudo dos indices, nunca sobre o caminho do arquivo.

    Duas maquinas guardam o artefato em pastas diferentes e precisam chegar ao
    mesmo hash; hash de caminho tornaria a comparacao entre elas impossivel. O
    digest cobre os valores dos indices na ordem em que definem cada particao,
    com o nome da particao junto, para que trocar validacao por teste nao
    produza o mesmo resultado.
    """
    corpo = json.dumps(
        {nome: [int(i) for i in indices_por_particao[nome]] for nome in PARTICOES},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(corpo.encode("utf-8")).hexdigest()


def impressao_digital_base(caminho: str | Path) -> str:
    """Identifica a versao da base de origem pelo conteudo do arquivo.

    Data de modificacao e tamanho mudam sem que os dados mudem, e vice-versa. O
    digest e lido em blocos porque a base analitica nao cabe confortavelmente em
    memoria duas vezes.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Base de origem nao encontrada: {caminho}")
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def congelar(
    particoes: dict[str, pd.DataFrame],
    *,
    versao_base: str,
    parametros: dict[str, object],
    metadados: dict[str, object] | None = None,
    semente: int = SEMENTE,
    caminho: str | Path = ARTEFATO_PADRAO,
) -> dict:
    """Grava o artefato de indices das tres particoes e devolve o registro."""
    faltando = [nome for nome in PARTICOES if nome not in particoes]
    if faltando:
        raise KeyError(f"particoes ausentes no congelamento: {faltando}")

    indices = {nome: [int(i) for i in particoes[nome].index] for nome in PARTICOES}
    registro = {
        "versao_artefato": VERSAO_ARTEFATO,
        "semente": int(semente),
        "versao_base": versao_base,
        "parametros": dict(parametros),
        "hash_indices": hash_indices(indices),
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadados_divisao": metadados or {},
        "indices": indices,
    }
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    # default=str: os metadados vem de preparar_matriz e carregam Timestamp e
    # tipos do numpy, que o json nao serializa. Eles sao registro de
    # procedencia, nao entram no hash, entao a forma textual basta.
    caminho.write_text(
        json.dumps(registro, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return registro


def carregar_registro(caminho: str | Path = ARTEFATO_PADRAO) -> dict:
    """Le o artefato e recusa o que nao for integro.

    A conferencia do hash na leitura e o que da sentido ao congelamento: um
    artefato editado a mao, truncado por uma copia interrompida ou gerado por
    uma versao anterior do formato precisa parar a execucao, e nao seguir como
    se fosse a divisao acordada.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Artefato de indices nao encontrado em {caminho}. "
            "Gere-o com obter_particoes(df, caminho_base=...) antes de treinar."
        )
    registro = json.loads(caminho.read_text(encoding="utf-8"))

    if registro.get("versao_artefato") != VERSAO_ARTEFATO:
        raise ValueError(
            f"Artefato na versao {registro.get('versao_artefato')}; esta execucao "
            f"exige a versao {VERSAO_ARTEFATO}. Regere o artefato a partir da base."
        )
    esperado = hash_indices(registro["indices"])
    if esperado != registro["hash_indices"]:
        raise ValueError(
            "Hash do artefato nao confere com os indices que ele carrega: registrado "
            f"{registro['hash_indices'][:12]}, calculado {esperado[:12]}. O arquivo "
            "foi alterado; regere o artefato a partir da base."
        )
    return registro


def _conferir_procedencia(
    registro: dict, *, versao_base: str, parametros: dict[str, object]
) -> None:
    """Recusa um artefato gerado sobre outra base ou com outra politica de corte.

    Separado de `carregar_registro` porque aquela funcao responde por integridade
    do arquivo, que independe de quem o le, enquanto estas duas conferencias
    dependem do que a execucao atual esta pedindo.
    """
    if registro["versao_base"] != versao_base:
        raise ValueError(
            "A divisao congelada pertence a outra versao da base: artefato gerado "
            f"sobre {registro['versao_base'][:12]}, base atual {versao_base[:12]}. "
            "Rode com regerar=True se a troca de base for intencional."
        )
    if registro["parametros"] != parametros:
        raise ValueError(
            "A divisao congelada usou outros parametros de corte: artefato com "
            f"{registro['parametros']}, execucao pedindo {parametros}. Mudar a "
            "politica de particionamento exige regerar=True e nova rodada dos cards."
        )


def obter_particoes(
    df: pd.DataFrame,
    *,
    caminho_base: str | Path,
    corte_validacao: str,
    corte_teste: str,
    sem_data: str = "treino",
    caminho: str | Path = ARTEFATO_PADRAO,
    regerar: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Funcao unica de carregamento das particoes congeladas.

    Na primeira execucao o artefato ainda nao existe e e gerado por `dividir`.
    Nas seguintes ele e lido e a divisao NAO e refeita: e esse o congelamento.
    Recriar a divisao em cada card faria duas pessoas rodando o notebook em
    maquinas diferentes chegarem a conjuntos distintos sem aviso, e as metricas
    comparadas na Secao 4.4 deixariam de significar a mesma coisa.

    O retorno tem a mesma forma do de `dividir`, de modo que `conferir` e
    `resumo` continuem valendo sobre as particoes carregadas.
    """
    caminho = Path(caminho)
    versao_base = impressao_digital_base(caminho_base)
    parametros = {
        "corte_validacao": str(pd.Timestamp(corte_validacao).date()),
        "corte_teste": str(pd.Timestamp(corte_teste).date()),
        "sem_data": sem_data,
    }

    if regerar or not caminho.exists():
        particoes, metadados = dividir(
            df,
            corte_validacao=corte_validacao,
            corte_teste=corte_teste,
            sem_data=sem_data,
        )
        registro = congelar(
            particoes,
            versao_base=versao_base,
            parametros=parametros,
            metadados=metadados,
            caminho=caminho,
        )
        print(f"particoes congeladas em {caminho} | hash {registro['hash_indices'][:12]}")
        return particoes, registro

    registro = carregar_registro(caminho)
    _conferir_procedencia(registro, versao_base=versao_base, parametros=parametros)

    ausentes = {
        nome: sorted(set(registro["indices"][nome]) - set(df.index))
        for nome in PARTICOES
    }
    total_ausente = sum(len(v) for v in ausentes.values())
    if total_ausente:
        raise ValueError(
            f"{total_ausente} indice(s) do artefato nao existem no DataFrame recebido "
            f"({ {n: len(v) for n, v in ausentes.items()} }). A base foi reordenada ou "
            "filtrada depois do congelamento; regere o artefato."
        )

    particoes = {nome: df.loc[registro["indices"][nome]] for nome in PARTICOES}
    print(f"particoes carregadas de {caminho} | hash {registro['hash_indices'][:12]}")
    return particoes, registro


def descrever(registro: dict) -> str:
    """Bloco de procedencia para imprimir no notebook.

    A Secao 4.3 precisa mostrar semente, versao da base e hash na propria
    execucao. O notebook e versionado sem saidas de celula, entao esse texto tem
    de ser gerado por codigo, e nao copiado a mao para o Markdown, onde
    envelheceria em silencio.
    """
    linhas = [
        f"semente fixada:     {registro['semente']}",
        f"versao da base:     {registro['versao_base'][:16]}",
        f"hash dos indices:   {registro['hash_indices'][:16]}",
        f"artefato gerado em: {registro['gerado_em']}",
    ]
    linhas += [f"{chave}: {valor}" for chave, valor in registro["parametros"].items()]
    linhas += [
        f"linhas em {nome}: {len(registro['indices'][nome]):,}".replace(",", ".")
        for nome in PARTICOES
    ]
    return "\n".join(linhas)


def congelar_ou_conferir(
    particoes: dict[str, pd.DataFrame],
    *,
    caminho_base: str | Path,
    parametros: dict[str, object],
    metadados: dict[str, object] | None = None,
    caminho: str | Path = ARTEFATO_PADRAO,
    regerar: bool = False,
) -> dict:
    """Congela as particoes que a matriz ja produziu, ou confere que nao mudaram.

    `obter_particoes` serve a quem parte do zero: ele mesmo chama `dividir`. No
    notebook de modelagem a divisao ja aconteceu, dentro de `preparar_matriz`, e
    reparti-la aqui produziria a mesma divisao duas vezes no mesmo notebook —
    exatamente o que a Secao 1.5 do notebook recusa fazer. Esta funcao recebe as
    particoes prontas e trata o artefato como o que ele e: registro do que foi
    acordado, e nao uma segunda fonte de verdade.

    Na primeira execucao grava o artefato. Nas seguintes, le o que esta gravado e
    compara o hash dos indices atuais com o registrado. Divergiu, para: a base
    foi regerada, reordenada ou filtrada desde o congelamento, e as metricas que
    os cards #103 a #110 comparam na Secao 4.4 deixariam de medir a mesma coisa
    sem que ninguem percebesse.
    """
    caminho = Path(caminho)
    versao_base = impressao_digital_base(caminho_base)
    parametros = dict(parametros)

    if regerar or not caminho.exists():
        registro = congelar(
            particoes,
            versao_base=versao_base,
            parametros=parametros,
            metadados=metadados,
            caminho=caminho,
        )
        print(f"particoes congeladas em {caminho} | hash {registro['hash_indices'][:12]}")
        return registro

    registro = carregar_registro(caminho)
    _conferir_procedencia(registro, versao_base=versao_base, parametros=parametros)

    faltando = [nome for nome in PARTICOES if nome not in particoes]
    if faltando:
        raise KeyError(f"particoes ausentes na conferencia: {faltando}")

    atual = {nome: [int(i) for i in particoes[nome].index] for nome in PARTICOES}
    hash_atual = hash_indices(atual)
    if hash_atual != registro["hash_indices"]:
        tamanhos_artefato = {n: len(registro["indices"][n]) for n in PARTICOES}
        tamanhos_atuais = {n: len(atual[n]) for n in PARTICOES}
        raise ValueError(
            "As particoes desta execucao nao sao as congeladas: artefato com hash "
            f"{registro['hash_indices'][:12]} e tamanhos {tamanhos_artefato}, execucao "
            f"com hash {hash_atual[:12]} e tamanhos {tamanhos_atuais}. A base mudou "
            "desde o congelamento; regere com regerar=True se a mudanca for intencional."
        )

    print(f"particoes conferidas contra {caminho} | hash {registro['hash_indices'][:12]}")
    return registro

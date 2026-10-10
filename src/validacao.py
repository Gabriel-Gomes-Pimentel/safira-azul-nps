"""Validacao cruzada por Cliente dentro do treino, com GroupKFold de cinco folds.

O artefato exige um conjunto de validacao alem do treino e do teste. Ele poderia
ser um terceiro bloco temporal, mas aqui e um GroupKFold de cinco folds dentro do
treino, e a escolha tem motivo: a particao temporal ja consome os meses finais da
base no teste, e recortar mais um bloco encolheria o treino justamente na faixa em
que o historico de Cliente comeca a existir. Com os folds, cada linha do treino
serve de validacao exatamente uma vez, sem tirar nenhuma do ajuste.

O agrupamento por `ID_GOLDENRECORD` e o ponto, nao um detalhe de implementacao.
A hipotese 4 da secao 4.2.4 mediu que quem detratou uma vez volta a detratar com
chance 4,61 vezes maior, e que o efeito sobrevive ao controle por atraso e
cancelamento. Ou seja: as respostas de um mesmo Cliente nao sao independentes.
Um `KFold` comum colocaria duas respostas da mesma pessoa em folds diferentes, e o
modelo acertaria a segunda por reconhecer a primeira. A metrica de validacao
subiria sem que nada tivesse melhorado, e a decisao de hiperparametro tomada em
cima dela seria tomada sobre um numero inflado.

E a mesma razao que ja governa a divisao treino/teste em `split.dividir`. Aqui ela
e aplicada um nivel abaixo, dentro do treino, porque o vazamento entre folds nao e
barrado pelo corte temporal: os folds nao sao temporais.

A divisao treino/teste continua sendo a temporal por Cliente. Este modulo nao
reparticiona nada: recebe o treino que `matriz.preparar_matriz` ja produziu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# O modulo localiza os proprios vizinhos em vez de depender de quem o importa,
# pelo mesmo motivo de `matriz.py`: sob pytest o conftest prepara o caminho, no
# Colab e na execucao direta nao ha ninguem para preparar.
_RAIZ = Path(__file__).resolve().parents[1]
for _pasta in ("src", "scripts"):
    _caminho = str(_RAIZ / _pasta)
    if _caminho not in sys.path:
        sys.path.insert(0, _caminho)

import split  # noqa: E402

# A geracao dos folds vive no pre-processamento desde a secao 4.2.2 e continua
# sendo a fonte unica: reimplementar o GroupKFold aqui criaria duas versoes da
# mesma decisao de metodo, que poderiam divergir sem ninguem perceber.
from preprocessamento_nps import criar_folds_validacao_por_cliente  # noqa: E402

N_FOLDS = 5
ALVO = "DETRATOR"


def criar_folds(
    x_treino: pd.DataFrame,
    grupos_treino: pd.Series,
    n_splits: int = N_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materializa os folds numa lista de pares (posicoes de treino, de validacao).

    `criar_folds_validacao_por_cliente` devolve o gerador do `GroupKFold`, que se
    esgota na primeira passagem. Conferir os folds e depois treinar sobre eles sao
    duas passagens, entao a lista e necessaria: sem ela, a conferencia consumiria
    os folds e o treino receberia um gerador vazio, falha que nao levanta erro e
    so aparece como metrica estranha la na frente.

    Os indices sao **posicionais**, como todo `split` do scikit-learn: use
    `.iloc[...]`, nunca `.loc[...]`. As particoes de `preparar_matriz` preservam o
    indice original da base, que nao e um `RangeIndex`, entao trocar um pelo outro
    silenciosamente seleciona as linhas erradas.

    A divisao e deterministica: o `GroupKFold` nao embaralha, entao duas execucoes
    sobre a mesma entrada produzem os mesmos folds, que e o que torna comparaveis
    as metricas de dois candidatos.
    """
    if len(x_treino) != len(grupos_treino):
        raise ValueError(
            "x_treino e grupos_treino precisam ter o mesmo tamanho: "
            f"{len(x_treino)} contra {len(grupos_treino)}"
        )
    folds = [
        (np.asarray(treino), np.asarray(validacao))
        for treino, validacao in criar_folds_validacao_por_cliente(
            x_treino, grupos_treino, n_splits=n_splits
        )
    ]
    if len(folds) != n_splits:
        raise AssertionError(
            f"esperava {n_splits} folds e o GroupKFold devolveu {len(folds)}"
        )
    return folds


def conferir_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    grupos_treino: pd.Series,
) -> None:
    """Verifica por assercao que os folds nao vazam Cliente. Falha alto se vazarem.

    Sao quatro conferencias, e cada uma pega uma falha diferente:

    1. **Cliente em duas validacoes.** Duas respostas da mesma pessoa avaliadas em
       folds diferentes fazem a media dos cinco parecer melhor do que e.
    2. **Cliente no treino e na validacao do mesmo fold.** E o vazamento direto: o
       modelo ve a pessoa no ajuste e e cobrado sobre ela na medicao.
    3. **Cobertura.** A uniao das validacoes tem que ser o treino inteiro, sem
       repetir posicao. Fold vazio ou linha avaliada duas vezes distorce a media.
    4. **Grupo sem identificacao.** Cliente nulo impede saber de quem e a linha, e
       trata-los como pessoas distintas reintroduz o vazamento que o agrupamento
       existe para evitar.

    O `GroupKFold` garante 1 e 2 por construcao. A conferencia existe porque a
    garantia depende de o vetor de grupos ser o certo: passar o indice, a data ou
    um `grupos` de outra particao produz folds tecnicamente validos e
    cientificamente errados, e nada no resultado denuncia isso.
    """
    if grupos_treino.isna().any():
        raise AssertionError(
            f"{int(grupos_treino.isna().sum())} linha(s) do treino sem "
            f"{split.COLUNA_CLIENTE}: a validacao por grupo exige o identificador"
        )

    clientes_por_fold = [
        set(grupos_treino.iloc[validacao]) for _, validacao in folds
    ]
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            comum = clientes_por_fold[i] & clientes_por_fold[j]
            if comum:
                raise AssertionError(
                    f"{len(comum)} Cliente(s) na validacao dos folds {i} e {j} ao "
                    "mesmo tempo: a validacao agrupada impede que a mesma pessoa "
                    "seja avaliada em dois folds"
                )

    for numero, (treino, validacao) in enumerate(folds):
        vazados = set(grupos_treino.iloc[treino]) & set(grupos_treino.iloc[validacao])
        if vazados:
            raise AssertionError(
                f"fold {numero}: {len(vazados)} Cliente(s) presentes no ajuste e na "
                "validacao ao mesmo tempo"
            )

    posicoes = np.concatenate([validacao for _, validacao in folds])
    repetidas = len(posicoes) - len(set(posicoes.tolist()))
    if repetidas:
        raise AssertionError(
            f"{repetidas} posicao(oes) do treino aparecem mais de uma vez nas "
            "validacoes, no mesmo fold ou em folds diferentes"
        )
    if len(posicoes) != len(grupos_treino):
        raise AssertionError(
            f"a uniao das validacoes cobre {len(posicoes)} linha(s) e o treino tem "
            f"{len(grupos_treino)}: ha linha fora de toda validacao"
        )


def resumo_dos_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    grupos_treino: pd.Series,
    y_treino: pd.Series | None = None,
) -> pd.DataFrame:
    """Tabela por fold com tamanho, Clientes e prevalencia do alvo.

    A prevalencia entra ao lado do tamanho porque folds de tamanho parecido podem
    ter taxa de detracao diferente: como o agrupamento e por Cliente, e Clientes
    recorrentes nao se distribuem por igual, um fold pode concentrar detratores. E
    essa diferenca, e nao o tamanho, que explica metrica instavel entre folds.
    """
    linhas = []
    for numero, (treino, validacao) in enumerate(folds, start=1):
        linha = {
            "fold": numero,
            "n_ajuste": len(treino),
            "n_validacao": len(validacao),
            "clientes_ajuste": int(grupos_treino.iloc[treino].nunique()),
            "clientes_validacao": int(grupos_treino.iloc[validacao].nunique()),
        }
        if y_treino is not None:
            linha["prevalencia_validacao_pct"] = round(
                float(y_treino.iloc[validacao].mean()) * 100, 2
            )
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("fold")


def metadados_da_divisao(preparo: dict[str, object]) -> dict[str, object]:
    """Reune o que a Secao 4.3 precisa relatar sobre a divisao, sem recalcula-la.

    Os numeros ja existem em `preparo['metadados']`, produzidos por
    `split.dividir`. Recalcula-los no notebook criaria uma segunda contagem que
    poderia divergir da primeira depois de qualquer ajuste no corte.
    """
    metadados = preparo["metadados"]
    teste = preparo["particoes"]["teste"]
    datas_teste = pd.to_datetime(teste[split.COLUNA_DATA], errors="coerce").dropna()
    return {
        "mes_inicial_teste": (
            str(datas_teste.min().to_period("M")) if not datas_teste.empty else None
        ),
        "corte_validacao": metadados["corte_validacao"],
        "corte_teste": metadados["corte_teste"],
        "linhas_removidas_por_recorrencia": metadados["linhas_removidas_por_recorrencia"],
        "clientes_treino": metadados["clientes_treino"],
        "clientes_validacao": metadados["clientes_validacao"],
        "clientes_teste": metadados["clientes_teste"],
    }

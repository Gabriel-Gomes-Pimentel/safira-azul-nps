"""Teste de fumaça do ajuste único da Regressão Logística sobre o contrato (#206).

O que este arquivo protege não é o número medido — tempo de ajuste varia com a
máquina e não se testa. O que se protege é **de onde a medição tira os dados**.

O card #206 existe para medir o custo de um ajuste antes de a busca de
hiperparâmetros do card #207 ser dimensionada. A tentação, ao medir, é montar um
escalonamento rápido "só para o teste rodar": bastaria isso para o número medido
deixar de descrever o pipeline real, e para a comparação da Seção 4.4 passar a
comparar a Regressão Logística contra ensembles que consumiram outra matriz.

Por isso os testes se dividem em dois grupos. Os de comportamento confirmam que
`medir_ajuste_unico` descreve a matriz que recebeu e que a não convergência é
reportada em vez de engolida. Os de origem leem o código-fonte do módulo e
recusam qualquer sinal de que ele reparticione a base ou ajuste pré-processamento
próprio — o mesmo recurso usado em `tests/test_vazamento_alvo.py`, e pela mesma
razão: a garantia precisa valer a cada `pytest`, não a cada revisão de diff.

Tudo roda sobre fixture sintética. Nenhum teste lê `data/`, em linha com o Termo
de Abertura, que veda versionar ou publicar base do parceiro.

Executar com:  pytest tests/test_fumaca_regressao_logistica.py -v
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fumaca_logistica
from fumaca_logistica import medir_ajuste_unico, menor_max_iter_que_converge
from matriz import preparar_matriz

CORTES = {"corte_validacao": "2025-07-01", "corte_teste": "2026-01-01"}


@pytest.fixture
def base():
    """Base sintética com a allowlist completa e os três períodos cobertos.

    Mesma forma da fixture de `tests/test_matriz.py`: um Cliente por período,
    para que o desempate por recorrência não encolha as partições, e datas
    crescentes dentro de cada período, para que a verificação de anterioridade
    não recuse uma base que na prática está ordenada.
    """
    n = 60
    datas = (
        list(pd.date_range("2024-03-01", periods=30).astype(str))
        + list(pd.date_range("2025-09-01", periods=15).astype(str))
        + list(pd.date_range("2026-03-01", periods=15).astype(str))
    )
    return pd.DataFrame({
        "RESPONDENT_ID": range(1, n + 1),
        "ID_GOLDENRECORD": range(101, 101 + n),
        "DATA_STD": datas,
        "DETRATOR": ([1, 0] * (n // 2)),
        "TIER_VIAGEM": ["DIAMANTE", "SAFIRA"] * (n // 2),
        "VOO_TIPO": ["DIRETO", "CONEXAO"] * (n // 2),
        "TIPO_ENTRETENIMENTO": ["TELA"] * n,
        "CANAL_COMPRA": ["WEB", "AGENCIA"] * (n // 2),
        "SEGMENTO": ["CORPORATIVO", "LAZER"] * (n // 2),
        "ESTATISTICA_ATRASOSAIDA": np.arange(n, dtype=float),
        "ATRASO_CHEGADA": np.arange(n, dtype=float) * 2,
        "CANCELAMENTO_VOO": [False, True] * (n // 2),
        "ANTECEDENCIA_CANCELAMENTO": np.arange(n, dtype=float),
        "TEMPO_VOO": np.arange(n, dtype=float) + 60,
        "N_TRECHOS": (np.arange(n) % 3) + 1,
        "NPS_PRINCIPAL": [-100, 100] * (n // 2),
        "SUB_NOTA_TRIPULACAO": [1, 5] * (n // 2),
    })


@pytest.fixture
def contrato(base):
    """As saídas do contrato, exatamente como o notebook as consome."""
    return preparar_matriz(base, **CORTES)


FONTE = Path(fumaca_logistica.__file__).read_text(encoding="utf-8")
ARVORE = ast.parse(FONTE)


def _modulos_importados() -> set[str]:
    """Nomes de módulo que `fumaca_logistica.py` importa, em qualquer das formas."""
    modulos: set[str] = set()
    for no in ast.walk(ARVORE):
        if isinstance(no, ast.Import):
            modulos.update(alias.name for alias in no.names)
        elif isinstance(no, ast.ImportFrom) and no.module:
            modulos.add(no.module)
    return modulos


def _nomes_usados() -> set[str]:
    """Identificadores e atributos que o módulo de fato chama ou referencia.

    Lê da árvore sintática, e não do texto, porque as docstrings do módulo citam
    `RobustScaler` e `ColumnTransformer` de propósito, para explicar que a escala
    das features já vem resolvida pelo contrato. Uma busca textual confundiria a
    explicação com o uso e proibiria justamente a documentação que deixa a
    fronteira clara.
    """
    nomes: set[str] = set()
    for no in ast.walk(ARVORE):
        if isinstance(no, ast.Name):
            nomes.add(no.id)
        elif isinstance(no, ast.Attribute):
            nomes.add(no.attr)
    return nomes


# ------------------------------------------------------- consumo do contrato

def test_medicao_descreve_a_matriz_que_recebeu_do_contrato(contrato):
    """n_linhas e n_colunas têm que ser os da matriz do contrato, não outros.

    É a checagem que denuncia uma medição feita sobre uma amostra reduzida ou
    sobre a matriz crua: nos dois casos a forma reportada divergiria da que o
    contrato entregou, e o tempo medido subestimaria a busca do card #207.
    """
    medicao = medir_ajuste_unico(contrato["matrizes"]["treino"], contrato["y"]["treino"])

    esperado = contrato["matrizes"]["treino"].shape
    assert (medicao["n_linhas"], medicao["n_colunas"]) == esperado
    assert medicao["segundos"] > 0
    assert medicao["max_iter"] == fumaca_logistica.MAX_ITER_PADRAO


def test_medicao_usa_o_pre_processador_ja_ajustado_no_contrato(contrato):
    """A largura da matriz tem que bater com a saída do pré-processador do contrato.

    Se o módulo reajustasse a codificação por conta própria, o número de colunas
    deixaria de coincidir com `get_feature_names_out` do transformador que o
    contrato ajustou no treino — que é a assinatura do vazamento que a Seção 4.3
    fechou e que a Seção 4.4 não pode reabrir.
    """
    largura_do_contrato = len(contrato["preprocessador"].get_feature_names_out())
    medicao = medir_ajuste_unico(contrato["matrizes"]["treino"], contrato["y"]["treino"])

    assert medicao["n_colunas"] == largura_do_contrato


def test_grupos_do_contrato_acompanham_as_linhas_da_matriz(contrato):
    """`grupos` vem do contrato alinhado a X, e é o que o card #210 vai agrupar.

    Este card não agrupa nada: só confirma que a chave de Cliente chegou junto e
    na mesma ordem. Um desalinhamento aqui só apareceria lá na frente, como
    métrica otimista, e sem pista de onde nasceu.
    """
    x_treino = contrato["x"]["treino"]
    grupos_treino = contrato["grupos"]["treino"]

    assert list(grupos_treino.index) == list(x_treino.index)
    assert len(grupos_treino) == contrato["matrizes"]["treino"].shape[0]
    assert grupos_treino.notna().all()


# ------------------------------------------------------------- convergência

def test_ajuste_que_para_no_limite_reporta_nao_convergencia(contrato):
    """Com uma iteração o ajuste trunca, e isso precisa aparecer no resultado.

    O aviso é capturado, não silenciado, porque o card #212 lê os coeficientes
    deste modelo como odds ratio: coeficiente de ajuste truncado é provisório, e
    interpretá-lo como efeito da operação seria ler ruído de otimização.
    """
    medicao = medir_ajuste_unico(
        contrato["matrizes"]["treino"], contrato["y"]["treino"], max_iter=1
    )

    assert medicao["convergiu"] is False
    assert medicao["aviso"]
    assert medicao["n_iter"] == 1


def test_menor_max_iter_devolve_a_tentativa_que_converge(contrato):
    """A tentativa escolhida é a primeira que converge, e a trilha fica registrada.

    A trilha importa para o card #207: só o tempo do ajuste que converge entra na
    conta de custo da grade. Somar os truncados subestimaria a busca.
    """
    resultado = menor_max_iter_que_converge(
        contrato["matrizes"]["treino"], contrato["y"]["treino"], escala=(1, 100, 200)
    )

    assert resultado["convergiu"] is True
    assert resultado["escolhido"]["convergiu"] is True
    assert resultado["escolhido"] is resultado["tentativas"][-1]
    assert resultado["tentativas"][0]["max_iter"] == 1


def test_escala_esgotada_reporta_em_vez_de_levantar(contrato):
    """Não convergir é achado a registrar no notebook, não erro de execução."""
    resultado = menor_max_iter_que_converge(
        contrato["matrizes"]["treino"], contrato["y"]["treino"], escala=(1, 2)
    )

    assert resultado["convergiu"] is False
    assert len(resultado["tentativas"]) == 2


# ------------------------------------------------------- origem dos dados

def test_fumaca_nao_importa_selecao_de_modelo():
    """O módulo não pode tocar `sklearn.model_selection`.

    É de lá que sairia qualquer repartição ou validação cruzada própria, e este
    card não cria nenhuma das duas: a divisão já veio congelada do contrato e a
    validação cruzada por Cliente é do card #210. Barrar o pacote inteiro é mais
    firme do que barrar nome por nome, porque cobre também o que for adicionado
    lá depois.
    """
    assert not [m for m in _modulos_importados() if "model_selection" in m]


def test_fumaca_nao_ajusta_pre_processamento_proprio():
    """Nenhum transformador é instanciado ou ajustado dentro do módulo.

    O pré-processamento pertence ao contrato, que o ajusta apenas no treino.
    Recriá-lo aqui produziria uma segunda matriz, parecida o bastante para passar
    despercebida e diferente o bastante para invalidar a comparação da Seção 4.4.
    """
    pacotes_de_transformacao = ("sklearn.preprocessing", "sklearn.compose", "sklearn.impute")
    importados = _modulos_importados()
    assert not [m for m in importados if m.startswith(pacotes_de_transformacao)]

    usados = _nomes_usados()
    assert "fit_transform" not in usados
    assert not [n for n in usados if n.endswith(("Scaler", "Encoder", "Transformer", "Imputer"))]


def test_fumaca_nao_le_a_base_do_parceiro():
    """O módulo recebe matriz em memória; ele não abre arquivo nenhum.

    Se ele lesse `data/` por conta própria, a medição deixaria de descrever a
    matriz que o notebook realmente usa e passaria a depender de um arquivo que
    não está versionado — irreproduzível para quem revisa.
    """
    usados = _nomes_usados()
    for proibido in ("read_parquet", "read_csv", "read_excel", "open", "Path"):
        assert proibido not in usados, f"{proibido} nao pode ser usado em fumaca_logistica.py"

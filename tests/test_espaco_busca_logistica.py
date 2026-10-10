"""Testes da grade de busca da Regressão Logística (#207).

O que estes testes protegem é a **validade da declaração**, não o resultado da
busca. A grade do card #207 é escrita antes de o card #210 executá-la, e uma
combinação inválida só apareceria lá, no meio de uma busca que já gastou folds:
`lbfgs` com L1 levanta `ValueError` e derruba o `GridSearchCV` inteiro depois de
horas de execução. Rodar cada combinação uma vez sobre uma matriz sintética
minúscula custa milissegundos e move essa descoberta para antes da busca.

O segundo grupo de testes cobre a conta de custo. Ela é o que decide se a busca
cabe na sessão do Colab, e uma conta que esquece os folds ou o `refit` erraria a
decisão para menos, que é justamente o lado perigoso.

Nada aqui lê `data/`: a fixture é sintética, em linha com o Termo de Abertura.

Executar com:  pytest tests/test_espaco_busca_logistica.py -v
"""
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from espaco_busca_logistica import (
    GRADE_LOGISTICA,
    MAX_ITER,
    PESOS_DE_CLASSE,
    SEGUNDOS_POR_AJUSTE,
    VALORES_C,
    VALORES_L1_RATIO,
    combinacoes,
    custo_estimado,
    grade_com_prefixo,
    medir_ajuste,
)

# Pares que o scikit-learn 1.9 aceita, lidos da tabela de compatibilidade da
# documentação do `solver`. `saga` é o único que aceita valor intermediário, e
# por isso é o único caso em que a checagem é um intervalo e não uma lista.
L1_RATIO_ACEITO = {
    "lbfgs": [0.0],
    "newton-cg": [0.0],
    "newton-cholesky": [0.0],
    "sag": [0.0],
    "liblinear": [0.0, 1.0],
}


@pytest.fixture
def dados():
    """Matriz sintética binária, pequena e separável o bastante para convergir.

    Duas colunas informativas e duas de ruído: o suficiente para a L1 ter o que
    zerar quando a regularização aperta, sem que o ajuste custe tempo de teste.
    """
    aleatorio = np.random.RandomState(42)
    matriz = aleatorio.randn(400, 4)
    alvo = (matriz[:, 0] + 0.5 * matriz[:, 1] > 0).astype(int)
    return matriz, alvo


def test_toda_combinacao_da_grade_ajusta_sem_erro_e_sem_aviso(dados):
    """CR03: nenhuma combinação declarada é inválida no scikit-learn.

    O aviso também é barrado, e não só o erro: `penalty` sai na versão 1.10, e
    uma grade que hoje passa emitindo `FutureWarning` é uma grade que quebra na
    próxima atualização do `requirements.txt`.
    """
    matriz, alvo = dados

    for parametros in combinacoes():
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always", FutureWarning)
            LogisticRegression(random_state=42, **parametros).fit(matriz, alvo)

        avisos = [str(a.message) for a in capturados if issubclass(a.category, FutureWarning)]
        assert not avisos, f"{parametros} emitiu {avisos}"


def test_grade_nao_declara_penalty():
    """A penalidade é expressa por `l1_ratio`, não pelo parâmetro depreciado."""
    for subgrade in GRADE_LOGISTICA:
        assert "penalty" not in subgrade


def test_cada_solver_so_aparece_com_l1_ratio_que_ele_aceita():
    """A separação em subgrades existe justamente para impedir o par inválido.

    Um produto cartesiano único de `l1_ratio` por `solver` produziria `lbfgs`
    com L1. O teste falha na declaração, e não na execução da busca.
    """
    for parametros in combinacoes():
        solver, l1_ratio = parametros["solver"], parametros["l1_ratio"]
        if solver == "saga":
            assert 0.0 <= l1_ratio <= 1.0
        else:
            assert l1_ratio in L1_RATIO_ACEITO[solver], f"{solver} não aceita l1_ratio={l1_ratio}"


def test_contagem_de_combinacoes_bate_com_o_produto_dos_eixos():
    """Duas subgrades, cada uma com `C` x `class_weight`, sem sobreposição.

    É este número que multiplica os folds na conta de custo: se a expansão
    surpreender, a conta que dimensiona a busca surpreende junto.
    """
    esperado = len(VALORES_L1_RATIO) * len(VALORES_C) * len(PESOS_DE_CLASSE)

    assert len(combinacoes()) == esperado == 16
    assert len({tuple(sorted(c.items(), key=str)) for c in combinacoes()}) == esperado


def test_toda_combinacao_fixa_o_max_iter_que_converge():
    """O limite precisa cobrir o pior caso da grade, não o do card #206.

    O #206 mediu 534 iterações no ponto de partida da biblioteca e concluiu que
    800 bastava. A medição deste card, nos extremos da grade, achou 838 em
    `C=10.0` com `class_weight="balanced"`: com 800 essa combinação truncaria, e
    é o coeficiente dela que o card #212 leria como odds ratio.
    """
    assert MAX_ITER >= 838
    assert {c["max_iter"] for c in combinacoes()} == {MAX_ITER}


def test_todo_solver_da_grade_tem_tempo_medido():
    """Sem tempo medido para um solver, a conta de custo do card não fecha.

    O teste existe para o dia em que alguém acrescentar uma subgrade com `saga`
    ou `newton-cholesky` e esquecer de medir: a falha aparece aqui, e não numa
    busca que já começou.
    """
    assert {c["solver"] for c in combinacoes()} <= set(SEGUNDOS_POR_AJUSTE)
    assert all(segundos > 0 for segundos in SEGUNDOS_POR_AJUSTE.values())


def test_custo_soma_todos_os_folds_de_cada_combinacao_e_o_refit():
    """A conta é combinações x folds x tempo do solver, mais um ajuste de `refit`.

    Os tempos aqui são fictícios de propósito: o que se testa é a aritmética,
    não a máquina. Oito combinações por solver, cinco folds: 8 x 5 x 10 s de
    `lbfgs`, mais 8 x 5 x 2 s de `liblinear`, mais o `refit` no solver mais caro.
    """
    custo = custo_estimado({"lbfgs": 10.0, "liblinear": 2.0}, n_folds=5)

    assert custo["por_solver"] == {"lbfgs": 400.0, "liblinear": 80.0}
    assert custo["refit"] == 10.0
    assert custo["segundos"] == 490.0
    assert custo["ajustes"] == 16 * 5 + 1
    assert custo["combinacoes"] == 16


def test_custo_recusa_solver_sem_tempo_medido():
    """Sem medição não há conta: estimativa de catálogo não dimensiona busca."""
    with pytest.raises(ValueError, match="liblinear"):
        custo_estimado({"lbfgs": 10.0}, n_folds=5)


def test_custo_recusa_numero_de_folds_sem_sentido():
    with pytest.raises(ValueError, match="n_folds"):
        custo_estimado({"lbfgs": 10.0, "liblinear": 2.0}, n_folds=1)


def test_prefixo_do_pipeline_preserva_a_expansao():
    """O card #210 consome a mesma grade, só que com nome de passo do `Pipeline`."""
    prefixada = grade_com_prefixo("modelo")

    assert len(list(combinacoes(prefixada))) == len(combinacoes())
    for parametros in combinacoes(prefixada):
        assert all(chave.startswith("modelo__") for chave in parametros)


def test_prefixo_vazio_e_recusado():
    with pytest.raises(ValueError, match="passo"):
        grade_com_prefixo("")


def test_medicao_reporta_ajuste_que_para_no_limite(dados):
    """Uma combinação que trunca precisa aparecer como tal na conta de custo.

    O ajuste truncado para antes e custa menos, então tomá-lo como tempo de
    referência subestimaria a busca, além de medir um modelo que ninguém quer.
    """
    matriz, alvo = dados

    medicao = medir_ajuste(matriz, alvo, {"solver": "lbfgs", "l1_ratio": 0.0, "max_iter": 1})

    assert medicao["convergiu"] is False
    assert medicao["aviso"] is not None
    assert medicao["segundos"] > 0


def test_medicao_conta_coeficientes_zerados_pela_l1(dados):
    """Com a L1 apertada, parte dos coeficientes vai a zero.

    É o efeito que justifica o eixo `l1_ratio` na grade: feature com coeficiente
    zerado sai da leitura de odds ratio do card #212, e isso é resultado da
    busca, não detalhe de implementação.
    """
    matriz, alvo = dados
    apertado = {"solver": "liblinear", "l1_ratio": 1.0, "C": 0.001, "max_iter": MAX_ITER}

    medicao = medir_ajuste(matriz, alvo, apertado)

    assert medicao["coeficientes_nulos"] > 0
    assert medicao["parametros"] == apertado

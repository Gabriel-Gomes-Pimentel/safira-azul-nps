"""Testes da validação cruzada por Cliente da seção 4.3.

O que estes testes protegem não é a contagem de folds, e sim a garantia que dá
sentido a ela: nenhum Cliente pode ser avaliado num fold e ter servido de ajuste
em outro. É o mesmo vazamento que `split.py` evita entre treino e teste, um nível
abaixo, e ele é mais fácil de deixar passar aqui porque o corte temporal não
protege nada dentro do treino.

Um fold que vaza não quebra: ele devolve uma métrica melhor. Por isso o que se
testa é a trava **disparando**, e não o caminho feliz.

Todos usam dados sintéticos, sem tocar nas bases da Azul.

Executar com:  pytest tests/test_validacao.py -v
"""
import numpy as np
import pandas as pd
import pytest

from validacao import (N_FOLDS, conferir_folds, criar_folds,
                       metadados_da_divisao, resumo_dos_folds)


@pytest.fixture
def treino():
    """30 respostas de 10 Clientes, três respostas cada.

    Clientes recorrentes são o caso que importa: com um Cliente por linha,
    qualquer divisão passaria e o teste não provaria nada sobre agrupamento.
    """
    clientes = [100 + i // 3 for i in range(30)]
    x = pd.DataFrame({"TEMPO_VOO": np.arange(30, dtype=float)},
                     index=range(500, 530))
    grupos = pd.Series(clientes, index=x.index, name="ID_GOLDENRECORD")
    y = pd.Series([1, 0, 0] * 10, index=x.index, name="DETRATOR")
    return x, grupos, y


def test_cria_cinco_folds_que_cobrem_o_treino_inteiro(treino):
    x, grupos, _ = treino
    folds = criar_folds(x, grupos)
    assert len(folds) == N_FOLDS
    assert sum(len(validacao) for _, validacao in folds) == len(x)


def test_nenhum_cliente_aparece_na_validacao_de_dois_folds(treino):
    """CR02: é a garantia que o agrupamento existe para dar."""
    x, grupos, _ = treino
    folds = criar_folds(x, grupos)
    vistos = set()
    for _, validacao in folds:
        clientes = set(grupos.iloc[validacao])
        assert not (clientes & vistos)
        vistos |= clientes


def test_ajuste_e_validacao_do_mesmo_fold_nao_compartilham_cliente(treino):
    x, grupos, _ = treino
    for ajuste, validacao in criar_folds(x, grupos):
        assert not set(grupos.iloc[ajuste]) & set(grupos.iloc[validacao])


def test_conferir_folds_aceita_a_divisao_do_groupkfold(treino):
    x, grupos, _ = treino
    conferir_folds(criar_folds(x, grupos), grupos)


def test_conferir_folds_dispara_quando_um_cliente_vaza_entre_folds(treino):
    """Folds forjados: a mesma pessoa avaliada duas vezes precisa ser recusada."""
    _, grupos, _ = treino
    posicoes = np.arange(len(grupos))
    do_cliente_100 = posicoes[grupos.to_numpy() == 100]
    outras = posicoes[grupos.to_numpy() != 100]
    metade = len(outras) // 2
    folds = [
        (np.concatenate([outras[metade:], do_cliente_100[1:]]),
         np.concatenate([outras[:metade], do_cliente_100[:1]])),
        (np.concatenate([outras[:metade], do_cliente_100[:1]]),
         np.concatenate([outras[metade:], do_cliente_100[1:]])),
    ]
    with pytest.raises(AssertionError, match="dois folds"):
        conferir_folds(folds, grupos)


def test_conferir_folds_dispara_quando_o_cliente_esta_no_ajuste_e_na_validacao(treino):
    _, grupos, _ = treino
    posicoes = np.arange(len(grupos))
    folds = [(posicoes, posicoes)]
    with pytest.raises(AssertionError, match="ajuste e na validacao"):
        conferir_folds(folds, grupos)


def test_conferir_folds_dispara_quando_alguma_linha_fica_fora_da_validacao(treino):
    """Um único fold cobrindo 9 das 30 linhas: os três primeiros Clientes.

    O corte cai na fronteira de Cliente de propósito, para que a falta de
    cobertura seja a única coisa errada e a trava certa dispare.
    """
    _, grupos, _ = treino
    posicoes = np.arange(len(grupos))
    folds = [(posicoes[9:], posicoes[:9])]
    with pytest.raises(AssertionError, match="fora de toda validacao"):
        conferir_folds(folds, grupos)


def test_conferir_folds_dispara_com_posicao_repetida_no_mesmo_fold(treino):
    """A repeticao dentro de um mesmo fold e o caso que so esta trava pega.

    A posicao 14 sai da validacao do primeiro fold e a 13 entra no lugar dela.
    As duas sao do Cliente 104, entao nenhum Cliente passa a ser avaliado em
    dois folds e nenhum aparece no ajuste e na validacao do mesmo fold: as duas
    travas anteriores continuam caladas. A uniao das validacoes tambem segue com
    30 posicoes, o tamanho do treino, entao a conferencia de cobertura nao acusa
    nada. A linha 14 fica fora de toda validacao em silencio, e so a contagem de
    posicoes repetidas denuncia.
    """
    _, grupos, _ = treino
    posicoes = np.arange(len(grupos))
    validacao_primeiro = np.concatenate([posicoes[:14], [13]])
    validacao_segundo = posicoes[15:]
    folds = [
        (posicoes[15:], validacao_primeiro),
        (posicoes[:15], validacao_segundo),
    ]
    with pytest.raises(AssertionError, match="mais de uma vez"):
        conferir_folds(folds, grupos)


def test_conferir_folds_dispara_com_cliente_nulo(treino):
    x, grupos, _ = treino
    folds = criar_folds(x, grupos)
    com_nulo = grupos.copy()
    com_nulo.iloc[0] = np.nan
    with pytest.raises(AssertionError, match="sem ID_GOLDENRECORD"):
        conferir_folds(folds, com_nulo)


def test_recusa_grupos_de_tamanho_diferente_da_matriz(treino):
    x, grupos, _ = treino
    with pytest.raises(ValueError, match="mesmo tamanho"):
        criar_folds(x, grupos.iloc[:-1])


def test_recusa_quando_ha_menos_clientes_que_folds(treino):
    x, grupos, _ = treino
    poucos = pd.Series([1, 2, 3] * 10, index=grupos.index)
    with pytest.raises(ValueError, match="Clientes insuficientes"):
        criar_folds(x, poucos)


def test_divisao_e_deterministica(treino):
    """Duas execuções têm que dar os mesmos folds, senão nenhuma comparação vale."""
    x, grupos, _ = treino
    primeira = criar_folds(x, grupos)
    segunda = criar_folds(x, grupos)
    for (a_ajuste, a_val), (b_ajuste, b_val) in zip(primeira, segunda):
        assert np.array_equal(a_ajuste, b_ajuste)
        assert np.array_equal(a_val, b_val)


def test_resumo_tem_uma_linha_por_fold_com_tamanho_e_prevalencia(treino):
    x, grupos, y = treino
    tabela = resumo_dos_folds(criar_folds(x, grupos), grupos, y)
    assert len(tabela) == N_FOLDS
    assert tabela["n_validacao"].sum() == len(x)
    assert {"n_ajuste", "clientes_validacao", "prevalencia_validacao_pct"} <= set(tabela.columns)


def test_metadados_da_divisao_reaproveitam_o_que_o_split_ja_contou():
    preparo = {
        "metadados": {
            "corte_validacao": "2025-07-01", "corte_teste": "2026-01-01",
            "linhas_removidas_por_recorrencia": 7,
            "clientes_treino": 30, "clientes_validacao": 10, "clientes_teste": 12,
        },
        "particoes": {"teste": pd.DataFrame({"DATA_STD": ["2026-03-05", "2026-04-01"]})},
    }
    metadados = metadados_da_divisao(preparo)
    assert metadados["mes_inicial_teste"] == "2026-03"
    assert metadados["linhas_removidas_por_recorrencia"] == 7
    assert metadados["clientes_teste"] == 12

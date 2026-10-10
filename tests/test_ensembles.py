"""Teste de fumaca das bibliotecas de boosting candidatas (card 04A, #185).

Este arquivo nao mede qualidade de modelo: a metrica sai no notebook, sobre a
base real. O que ele protege e a validade da **medicao de custo** que decide qual
biblioteca o projeto vai usar nos cards #186, #187 e #188. Sao quatro riscos, e
todos eles devolveriam um tempo menor que o real, nunca um erro:

1. Medir sobre uma matriz que nao veio do contrato. Um pre-processamento proprio,
   mais simples que o do contrato, produziria menos colunas e um ajuste mais
   rapido do que o que a busca vai enfrentar.
2. Criar particao ou validacao cruzada aqui. Alem de contrariar a secao 3, uma
   fatia aleatoria menor que o treino do contrato tambem mediria menos.
3. Reaproveitar um modelo ja ajustado entre medicoes. A segunda biblioteca
   comecaria de um estado pronto e pareceria mais barata.
4. Comparar as duas com configuracoes diferentes. O tempo passaria a medir
   hiperparametro, e nao biblioteca.

A partir do card 08A (#187) o arquivo cobre tambem o pipeline do Random Forest:
que ele e montado sobre o pre-processador do contrato sem remonta-lo, que dois
ajustes com a mesma semente dao previsoes identicas (CR03) e que a linha de base
so e medida pela funcao `avaliar` recebida, sem metrica nem particao propria.

Todos usam dados sinteticos, sem tocar nas bases da Azul.

A segunda metade do arquivo e a do #186: que uma combinacao sorteada de cada
espaco de busca e aceita pelo estimador que vai consumi-la, e que a amostragem
se repete com a mesma semente.

A ultima parte cobre `criar_pipeline_gradient_boosting` e
`medir_linha_de_base` do card #188. Tudo roda sobre a fixture sintetica
`base_gb`/`contrato_gb`: o card 05 (#241) ainda nao esta em `develop`, entao os
testes de `medir_linha_de_base` usam um `avaliar` de mentira so para conferir
que os vetores certos chegam ate ele, sem depender da metrica real nem da
base do parceiro.

Por fim, a parte do #190 cobre `melhor_gradient_boosting` (CR04): a funcao que
a dupla de Metricas e Decisoes importa precisa devolver o pipeline vencedor
**nao ajustado** e com a semente fixada pelo JSON versionado.

Executar com:  pytest tests/test_ensembles.py -v
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import ensembles
import modelo
from sklearn.model_selection import ParameterSampler

from ensembles import (
    ARQUIVO_HIPERPARAMETROS_GRADIENT_BOOSTING,
    ESPACO_GRADIENT_BOOSTING_HISTGB,
    ESPACO_GRADIENT_BOOSTING_XGBOOST,
    ESPACO_RANDOM_FOREST,
    ESPACOS_GRADIENT_BOOSTING,
    ESTIMADOR_POR_ESPACO,
    HIPERPARAMETROS_COMPARACAO,
    RAZAO_DESBALANCEAMENTO,
    SEMENTE_PADRAO,
    PACOTES_TRAVADOS,
    PASSO_MODELO,
    PASSO_PREPARO,
    REQUIREMENTS_PADRAO,
    conferir_versoes,
    construir_candidatos,
    criar_pipeline_gradient_boosting,
    criar_pipeline_random_forest,
    escolher_biblioteca,
    estimar_busca,
    ler_versoes_travadas,
    medir_ajuste,
    medir_bibliotecas,
    medir_linha_de_base,
    melhor_gradient_boosting,
)
from matriz import preparar_matriz

RAIZ = Path(__file__).resolve().parents[1]

CORTE_VALIDACAO = "2025-06-01"
CORTE_TESTE = "2025-12-01"

# Os nomes proibidos sao montados por partes de proposito. O CR03 pede que a
# busca por eles no diff volte vazia, e um arquivo de teste que os escrevesse por
# extenso apareceria nessa busca e derrubaria a propria evidencia que deveria
# produzir.
PARTICIONADORES_PROIBIDOS = (
    "K" + "Fold",
    "Stratified" + "K" + "Fold",
    "Group" + "K" + "Fold",
    "train" + "_test_" + "split",
    # CR02 do #187: a linha de base e medida so por `avaliar`, entao validacao
    # cruzada propria tambem conta como particao fora do contrato.
    "cross" + "_val_" + "score",
)

ARQUIVOS_DO_CARD = (
    RAIZ / "src" / "ensembles.py",
    Path(__file__),
)


@pytest.fixture(scope="module")
def base_sintetica():
    """Base com as colunas que o contrato exige, em escala de teste.

    Os clientes se repetem para que o agrupamento por `ID_GOLDENRECORD` tenha o
    que agrupar, e as datas cobrem os dois cortes para que as tres particoes
    existam.
    """
    gerador = np.random.default_rng(7)
    n = 1200
    datas = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        gerador.integers(0, 900, size=n), unit="D"
    )
    return pd.DataFrame({
        "RESPONDENT_ID": np.arange(n),
        "ID_GOLDENRECORD": gerador.integers(1, 400, size=n),
        "DATA_STD": datas.strftime("%Y-%m-%d"),
        "DETRATOR": (gerador.random(n) < 0.2).astype(int),
        "TIER_VIAGEM": gerador.choice(["Diamante", "Safira", "Sem cadastro"], n),
        "VOO_TIPO": gerador.choice(["Direto", "Conexao"], n),
        "TIPO_ENTRETENIMENTO": gerador.choice(["Tela", "Streaming", "Sem"], n),
        "CANAL_COMPRA": gerador.choice(["Web", "Mobile", "Agency"], n),
        "SEGMENTO": gerador.choice(["Lazer", "Corporativo"], n),
        "ESTATISTICA_ATRASOSAIDA": gerador.integers(0, 300, n),
        "ATRASO_CHEGADA": gerador.integers(0, 300, n),
        "CANCELAMENTO_VOO": gerador.random(n) < 0.05,
        "ANTECEDENCIA_CANCELAMENTO": gerador.integers(0, 10, n),
        "TEMPO_VOO": gerador.integers(40, 600, n),
        "N_TRECHOS": gerador.integers(1, 4, n),
    })


@pytest.fixture(scope="module")
def contrato(base_sintetica):
    """A unica origem de `x`, `y`, `grupos` e do `ColumnTransformer` nos testes."""
    return preparar_matriz(base_sintetica, CORTE_VALIDACAO, CORTE_TESTE)


@pytest.fixture(scope="module")
def candidatos_rapidos():
    """As duas bibliotecas com poucas arvores, so para o caminho ser exercitado.

    O numero de arvores da comparacao real esta em `HIPERPARAMETROS_COMPARACAO` e
    vale no notebook; aqui ele so faria a suite demorar sem proteger nada a mais.
    """
    candidatos = construir_candidatos()
    candidatos["HistGradientBoostingClassifier"].set_params(max_iter=10)
    candidatos["XGBClassifier"].set_params(n_estimators=10)
    return candidatos


# ------------------------------------------------------- origem da matriz medida
def test_matriz_medida_vem_inteira_do_contrato(contrato):
    """`x`, `y`, `grupos` e a matriz transformada saem todos de `preparar_matriz`."""
    assert {"x", "y", "grupos", "matrizes", "preprocessador"} <= set(contrato)

    treino = contrato["matrizes"]["treino"]
    assert treino.shape[0] == len(contrato["y"]["treino"]) == len(contrato["grupos"]["treino"])
    assert treino.shape[1] == contrato["metadados"]["colunas_da_matriz"]
    # O agrupamento por Cliente existe de fato: sem repeticao, o `grupos` do
    # contrato nao teria nada a agrupar e a medicao estaria olhando outra base.
    assert contrato["grupos"]["treino"].nunique() < len(contrato["grupos"]["treino"])


def test_medicao_nao_reajusta_o_pre_processador_do_contrato(contrato, candidatos_rapidos):
    """A transformacao da validacao e identica antes e depois da medicao.

    Se alguma etapa do modulo reajustasse o `ColumnTransformer`, a mediana da
    imputacao e a escala passariam a carregar informacao de fora do treino e a
    saida abaixo mudaria.
    """
    preprocessador = contrato["preprocessador"]
    antes = preprocessador.transform(contrato["x"]["validacao"])

    medir_bibliotecas(
        contrato["matrizes"]["treino"],
        contrato["y"]["treino"],
        candidatos=candidatos_rapidos,
    )

    depois = preprocessador.transform(contrato["x"]["validacao"])
    np.testing.assert_array_equal(antes, depois)


# ------------------------------------------------------------ tabela de medicao
def test_tabela_traz_tempo_pico_e_semente_das_duas_bibliotecas(contrato, candidatos_rapidos):
    """O que o CR04 pede, por biblioteca, na tabela que vai para o notebook."""
    tabela = medir_bibliotecas(
        contrato["matrizes"]["treino"],
        contrato["y"]["treino"],
        candidatos=candidatos_rapidos,
    )

    assert list(tabela.index) == ["HistGradientBoostingClassifier", "XGBClassifier"]
    assert {"tempo_ajuste_s", "pico_memoria_mb", "random_state"} <= set(tabela.columns)
    assert (tabela["tempo_ajuste_s"] > 0).all()
    assert (tabela["pico_memoria_mb"] > 0).all()
    assert tabela["random_state"].nunique() == 1
    assert (tabela["n_linhas"] == contrato["matrizes"]["treino"].shape[0]).all()


def test_medicao_nao_reaproveita_ajuste_anterior(contrato, candidatos_rapidos):
    """O estimador recebido continua sem ajuste; quem treina e a copia."""
    original = candidatos_rapidos["HistGradientBoostingClassifier"]

    medicao = medir_ajuste(original, contrato["matrizes"]["treino"], contrato["y"]["treino"])

    check_is_fitted(medicao["estimador"])
    with pytest.raises(NotFittedError):
        check_is_fitted(original)


def test_hiperparametros_sao_equivalentes_nos_dois_lados():
    """Mesmo passo, mesmas arvores, mesmas folhas, mesmos bins e mesma semente."""
    candidatos = construir_candidatos(random_state=7)
    hist = candidatos["HistGradientBoostingClassifier"].get_params()
    xgb = candidatos["XGBClassifier"].get_params()

    assert hist["learning_rate"] == xgb["learning_rate"] == HIPERPARAMETROS_COMPARACAO["passo"]
    assert hist["max_iter"] == xgb["n_estimators"] == HIPERPARAMETROS_COMPARACAO["arvores"]
    assert hist["max_leaf_nodes"] == xgb["max_leaves"] == HIPERPARAMETROS_COMPARACAO["folhas"]
    assert hist["max_bins"] == xgb["max_bin"] == HIPERPARAMETROS_COMPARACAO["bins"]
    assert hist["random_state"] == xgb["random_state"] == 7
    # Sem parada antecipada dos dois lados: com ela, o tempo medido seria o de
    # menos arvores do que as declaradas.
    assert hist["early_stopping"] is False
    assert xgb["early_stopping_rounds"] is None


def test_comparacao_usa_os_hiperparametros_do_candidato():
    """O tempo medido e o do modelo que o projeto treina, e nao o de uma copia.

    Revisao da !94: o notebook afirma que os valores vem de
    `modelo.HIPERPARAMETROS_CANDIDATO`, e a celula usa
    `ensembles.HIPERPARAMETROS_COMPARACAO`. Os dois nomes existem, mas o segundo
    e derivado do primeiro; este teste falha se alguem trocar a derivacao por
    uma copia que depois divirja.
    """
    candidato = modelo.HIPERPARAMETROS_CANDIDATO
    assert HIPERPARAMETROS_COMPARACAO["passo"] == candidato["learning_rate"]
    assert HIPERPARAMETROS_COMPARACAO["arvores"] == candidato["max_iter"]
    assert HIPERPARAMETROS_COMPARACAO["folhas"] == candidato["max_leaf_nodes"]

    medido = construir_candidatos()["HistGradientBoostingClassifier"].get_params()
    treinado = modelo.criar_candidato().get_params()
    for chave in ("learning_rate", "max_iter", "max_leaf_nodes", "early_stopping", "random_state"):
        assert medido[chave] == treinado[chave], chave


# ------------------------------------------------------- conferencia de versoes
def _requirements(tmp_path, conteudo):
    caminho = tmp_path / "requirements.txt"
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


def test_requirements_do_projeto_fixa_os_tres_pacotes_da_medicao():
    """CR01: sem versao exata, `conferir_versoes` nao teria contra o que comparar."""
    travadas = ler_versoes_travadas(REQUIREMENTS_PADRAO)
    assert set(PACOTES_TRAVADOS) <= set(travadas)


def test_leitura_ignora_piso_comentario_e_linha_em_branco(tmp_path):
    caminho = _requirements(tmp_path, (
        "# comentario\n"
        "pandas>=2.0\n"
        "\n"
        "scipy==1.18.1  # com comentario na linha\n"
        "Scikit-Learn == 1.9.1\n"
    ))
    assert ler_versoes_travadas(caminho) == {"scipy": "1.18.1", "scikit-learn": "1.9.1"}


def test_conferencia_passa_quando_as_versoes_carregadas_batem(tmp_path):
    from importlib import metadata

    conteudo = "".join(f"{nome}=={metadata.version(nome)}\n" for nome in PACOTES_TRAVADOS)
    conferidas = conferir_versoes(_requirements(tmp_path, conteudo))
    assert set(conferidas) == set(PACOTES_TRAVADOS)


def test_conferencia_falha_quando_a_versao_carregada_diverge(tmp_path):
    """A divergencia vira erro na primeira celula, e nao um print que ninguem le."""
    from importlib import metadata

    conteudo = "".join(f"{nome}=={metadata.version(nome)}\n" for nome in PACOTES_TRAVADOS)
    conteudo = conteudo.replace(f"scipy=={metadata.version('scipy')}", "scipy==0.0.1")
    with pytest.raises(AssertionError, match="scipy: requirements 0.0.1"):
        conferir_versoes(_requirements(tmp_path, conteudo))


def test_conferencia_falha_quando_um_pacote_nao_tem_versao_exata(tmp_path):
    caminho = _requirements(tmp_path, "scikit-learn>=1.3\nscipy==1.18.1\nxgboost==3.4.1\n")
    with pytest.raises(AssertionError, match="sem versao exata"):
        conferir_versoes(caminho)


# ------------------------------------------------------- estimativa e criterio
def test_estimativa_multiplica_iteracoes_por_folds():
    assert estimar_busca(3600, n_iteracoes=1, n_folds=1) == pytest.approx(1.0)
    assert estimar_busca(36, n_iteracoes=40, n_folds=5) == pytest.approx(2.0)


@pytest.mark.parametrize("n_iteracoes,n_folds", [(0, 5), (40, 0), (-1, 5)])
def test_estimativa_recusa_busca_vazia(n_iteracoes, n_folds):
    with pytest.raises(ValueError):
        estimar_busca(10, n_iteracoes=n_iteracoes, n_folds=n_folds)


def test_criterio_prefere_a_biblioteca_sem_dependencia_extra_quando_ela_cabe():
    tabela = pd.DataFrame(
        {"tempo_ajuste_s": [10.0, 5.0]},
        index=["HistGradientBoostingClassifier", "XGBClassifier"],
    )

    decisao = escolher_biblioteca(tabela, orcamento_horas=3.0)

    assert decisao["escolhida"] == "HistGradientBoostingClassifier"
    assert "dependencia" in decisao["justificativa"]


def test_criterio_descarta_quem_estoura_o_orcamento_da_sessao():
    """Estourar a sessao nao e ser mais lento; e nao ser uma opcao."""
    tabela = pd.DataFrame(
        {"tempo_ajuste_s": [600.0, 5.0]},
        index=["HistGradientBoostingClassifier", "XGBClassifier"],
    )

    decisao = escolher_biblioteca(tabela, orcamento_horas=3.0)

    assert decisao["escolhida"] == "XGBClassifier"
    assert decisao["estimativas_horas"]["HistGradientBoostingClassifier"] > 3.0


def test_criterio_devolve_a_decisao_para_o_grupo_quando_nenhuma_cabe():
    tabela = pd.DataFrame(
        {"tempo_ajuste_s": [900.0, 600.0]},
        index=["HistGradientBoostingClassifier", "XGBClassifier"],
    )

    decisao = escolher_biblioteca(tabela, orcamento_horas=3.0)

    assert decisao["escolhida"] == "XGBClassifier"
    assert "Metricas e Decisoes" in decisao["justificativa"]


# --------------------------------------------------------------------- travas
@pytest.mark.parametrize("arquivo", ARQUIVOS_DO_CARD, ids=lambda p: p.name)
def test_card_nao_cria_particao_nem_validacao_propria(arquivo):
    """CR03: nenhum particionador fora do contrato aparece no codigo do card."""
    fonte = arquivo.read_text(encoding="utf-8")
    encontrados = [nome for nome in PARTICIONADORES_PROIBIDOS if nome in fonte]
    assert not encontrados, (
        f"{arquivo.name} instancia particionador proprio: {encontrados}. "
        "A validacao deste projeto sao os folds do contrato, e nenhuma outra."
    )


# ------------------------------------------- pipeline do random forest (#187)
# Poucas arvores bastam para exercitar o caminho; a floresta padrao da linha de
# base roda no notebook, sobre a base real.
ARVORES_TESTE = 10


def _avaliar_falso(chamadas):
    """Substituto de `avaliar(y_true, y_pred, y_proba)` que so registra o que recebeu.

    O card 05 (#241) ainda nao esta em `develop`; o que se protege aqui e o
    contrato da chamada, nao a metrica.
    """
    def avaliar(y_true, y_pred, y_proba):
        chamadas.append({"y_true": y_true, "y_pred": y_pred, "y_proba": y_proba})
        return {"metrica_falsa": 0.5}
    return avaliar


def test_pipeline_encadeia_o_pre_processador_do_contrato_e_o_random_forest(contrato):
    """CR01: o primeiro passo e o `ColumnTransformer` do contrato, clonado e virgem."""
    original = contrato["preprocessador"]

    pipeline = criar_pipeline_random_forest(original, random_state=7)

    assert [nome for nome, _ in pipeline.steps] == [PASSO_PREPARO, PASSO_MODELO]
    preparo = pipeline.named_steps[PASSO_PREPARO]
    assert isinstance(preparo, ColumnTransformer)
    assert preparo is not original
    # Mesma especificacao: mesmos blocos, mesmas colunas, mesmos passos internos.
    assert [(n, c) for n, _, c in preparo.transformers] == [
        (n, c) for n, _, c in original.transformers
    ]
    assert preparo.get_params(deep=True).keys() == original.get_params(deep=True).keys()
    # Nasce sem o ajuste que `preparar_matriz` ja fez no treino.
    with pytest.raises(NotFittedError):
        check_is_fitted(preparo)

    modelo_rf = pipeline.named_steps[PASSO_MODELO]
    assert isinstance(modelo_rf, RandomForestClassifier)
    assert modelo_rf.random_state == 7


def test_pipeline_nasce_nos_hiperparametros_padrao_da_biblioteca(contrato):
    """A linha de base nao escolhe hiperparametro; quem escolhe e a busca do #189."""
    padrao = RandomForestClassifier().get_params()
    modelo_rf = criar_pipeline_random_forest(contrato["preprocessador"]).named_steps[PASSO_MODELO]
    montado = modelo_rf.get_params()

    diferentes = {
        chave for chave in padrao
        if chave not in {"random_state", "n_jobs"} and montado[chave] != padrao[chave]
    }
    assert not diferentes


def test_hiperparametros_repassados_nao_sobrescrevem_a_semente(contrato):
    pipeline = criar_pipeline_random_forest(
        contrato["preprocessador"], random_state=3, n_estimators=ARVORES_TESTE
    )
    modelo_rf = pipeline.named_steps[PASSO_MODELO]
    assert modelo_rf.n_estimators == ARVORES_TESTE
    assert modelo_rf.random_state == 3


def test_mesma_semente_produz_previsoes_identicas(contrato):
    """CR03: dois ajustes independentes com o mesmo `random_state` sao o mesmo modelo."""
    probabilidades = []
    for _ in range(2):
        pipeline = criar_pipeline_random_forest(
            contrato["preprocessador"], random_state=11, n_estimators=ARVORES_TESTE
        )
        pipeline.fit(contrato["x"]["treino"], contrato["y"]["treino"])
        probabilidades.append(pipeline.predict_proba(contrato["x"]["validacao"]))

    np.testing.assert_array_equal(probabilidades[0], probabilidades[1])


def test_sementes_diferentes_produzem_florestas_diferentes(contrato):
    """Contraprova do CR03: se a semente nao chegasse ao estimador, o teste acima
    passaria igual, por sorte ou por acaso de implementacao."""
    probabilidades = []
    for semente in (11, 12):
        pipeline = criar_pipeline_random_forest(
            contrato["preprocessador"], random_state=semente, n_estimators=ARVORES_TESTE
        )
        pipeline.fit(contrato["x"]["treino"], contrato["y"]["treino"])
        probabilidades.append(pipeline.predict_proba(contrato["x"]["validacao"]))

    assert not np.array_equal(probabilidades[0], probabilidades[1])


def test_ajuste_do_pipeline_nao_mexe_no_pre_processador_do_contrato(contrato):
    """O `fit` do pipeline ajusta a copia; o objeto do contrato fica como estava."""
    preprocessador = contrato["preprocessador"]
    antes = preprocessador.transform(contrato["x"]["validacao"])

    pipeline = criar_pipeline_random_forest(preprocessador, n_estimators=ARVORES_TESTE)
    pipeline.fit(contrato["x"]["treino"], contrato["y"]["treino"])

    np.testing.assert_array_equal(antes, preprocessador.transform(contrato["x"]["validacao"]))


def test_linha_de_base_repassa_os_vetores_certos_para_avaliar(contrato):
    """CR02: a metrica vem de `avaliar`, que recebe rotulo, predicao e score de Detrator."""
    chamadas = []
    pipeline = criar_pipeline_random_forest(
        contrato["preprocessador"], random_state=5, n_estimators=ARVORES_TESTE
    )

    resultado = medir_linha_de_base(
        pipeline,
        contrato["x"]["treino"],
        contrato["y"]["treino"],
        contrato["x"]["validacao"],
        contrato["y"]["validacao"],
        avaliar=_avaliar_falso(chamadas),
    )

    assert len(chamadas) == 1
    recebido = chamadas[0]
    y_validacao = contrato["y"]["validacao"]
    assert recebido["y_true"] is y_validacao
    np.testing.assert_array_equal(
        recebido["y_pred"], pipeline.predict(contrato["x"]["validacao"])
    )
    # Coluna 1 de `predict_proba`: a probabilidade de Detrator, nao a de Neutro/Promotor.
    np.testing.assert_array_equal(
        recebido["y_proba"], pipeline.predict_proba(contrato["x"]["validacao"])[:, 1]
    )
    assert pipeline.classes_[1] == 1

    assert resultado["metricas"] == {"metrica_falsa": 0.5}
    assert resultado["random_state"] == 5
    assert resultado["tempo_total_s"] > 0
    assert resultado["n_treino"] == len(contrato["y"]["treino"])
    assert resultado["n_avaliacao"] == len(y_validacao)


def test_linha_de_base_recusa_avaliar_que_nao_e_funcao(contrato):
    pipeline = criar_pipeline_random_forest(contrato["preprocessador"], n_estimators=ARVORES_TESTE)
    with pytest.raises(TypeError, match="#241"):
        medir_linha_de_base(
            pipeline,
            contrato["x"]["treino"],
            contrato["y"]["treino"],
            contrato["x"]["validacao"],
            contrato["y"]["validacao"],
            avaliar=None,
        )


# --------------------------------------------- espacos de busca (#186)
N_AMOSTRAS_MINIMO = 40


def test_espacos_gradient_boosting_mantem_as_duas_bibliotecas_ate_o_185():
    """O card exige as duas versoes vivas enquanto o #185 nao decide.

    Quando a decisao sair, a chave da biblioteca descartada some deste
    dicionario no mesmo commit, e este teste e o primeiro a acusar se alguem
    esquecer de atualizar `ESTIMADOR_POR_ESPACO` junto.
    """
    assert set(ESPACOS_GRADIENT_BOOSTING) == {"hist_gradient_boosting", "xgboost"}
    assert ESPACOS_GRADIENT_BOOSTING["hist_gradient_boosting"] is ESPACO_GRADIENT_BOOSTING_HISTGB
    assert ESPACOS_GRADIENT_BOOSTING["xgboost"] is ESPACO_GRADIENT_BOOSTING_XGBOOST


@pytest.mark.parametrize("nome", sorted(ESTIMADOR_POR_ESPACO))
def test_amostras_do_espaco_sao_aceitas_pelo_estimador(nome):
    """CR02: 40 amostras sorteadas com semente fixa, cada uma vira um estimador."""
    estimador_cls, espaco = ESTIMADOR_POR_ESPACO[nome]
    amostras = list(
        ParameterSampler(espaco, n_iter=N_AMOSTRAS_MINIMO, random_state=SEMENTE_PADRAO)
    )

    assert len(amostras) == N_AMOSTRAS_MINIMO

    for parametros in amostras:
        modelo = estimador_cls(**parametros, random_state=SEMENTE_PADRAO)
        for nome_parametro, valor in parametros.items():
            assert modelo.get_params()[nome_parametro] == valor


@pytest.mark.parametrize("nome", sorted(ESTIMADOR_POR_ESPACO))
def test_amostragem_e_reprodutivel_com_a_mesma_semente(nome):
    """Duas buscas com a mesma semente precisam sortear as mesmas combinacoes."""
    _, espaco = ESTIMADOR_POR_ESPACO[nome]
    primeira = list(ParameterSampler(espaco, n_iter=N_AMOSTRAS_MINIMO, random_state=SEMENTE_PADRAO))
    segunda = list(ParameterSampler(espaco, n_iter=N_AMOSTRAS_MINIMO, random_state=SEMENTE_PADRAO))
    assert primeira == segunda


def test_max_features_do_random_forest_e_fracao_dimensionada_pelo_contrato_do_card_01():
    """CR ligado ao DoR: `max_features` precisa caber no contrato de 11 features.

    O espaco usa fracao continua, entao o teste confere que a distribuicao so
    produz valores dentro de (0, 1], nunca uma contagem absoluta de colunas
    que passaria das 11 do contrato.
    """
    amostras = ESPACO_RANDOM_FOREST["max_features"].rvs(
        size=200, random_state=SEMENTE_PADRAO
    )
    assert amostras.min() > 0.0
    assert amostras.max() <= 1.0


def test_scale_pos_weight_do_xgboost_nao_passa_da_razao_observada():
    """O teto do eixo e a proporcao negativos/positivos da base (~3,89)."""
    amostras = ESPACO_GRADIENT_BOOSTING_XGBOOST["scale_pos_weight"].rvs(
        size=200, random_state=SEMENTE_PADRAO
    )
    assert amostras.min() >= 1.0
    assert amostras.max() <= RAZAO_DESBALANCEAMENTO


def test_nenhum_espaco_tem_fit_na_importacao():
    """CR01: os espacos sao dicionarios de distribuicoes, nunca um estimador ajustado."""
    for _, espaco in ESTIMADOR_POR_ESPACO.values():
        assert isinstance(espaco, dict)
        assert not hasattr(espaco, "fit")


# ---------------------------------------------------------------------------
# Pipeline do Gradient Boosting (#188)
# ---------------------------------------------------------------------------

CORTES = {"corte_validacao": "2025-07-01", "corte_teste": "2026-01-01"}


@pytest.fixture
def base_gb():
    """Base sintetica com a allowlist completa e os tres periodos cobertos.

    Mesma forma da fixture de `tests/test_pipeline_logistica.py`: um Cliente por
    periodo, para que o desempate por recorrencia nao encolha as particoes, e
    datas crescentes dentro de cada periodo. As duas ultimas colunas sao de
    pesquisa, para que a trava de vazamento tenha o que recusar.
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
def contrato_gb(base_gb):
    """As saidas do contrato, exatamente como o notebook as consome."""
    return preparar_matriz(base_gb, **CORTES)


@pytest.fixture
def treino(contrato_gb):
    """`X` cru e `y` da particao de treino, que e o que o pipeline recebe."""
    return contrato_gb["x"]["treino"], contrato_gb["y"]["treino"]


@pytest.fixture
def avaliacao(contrato_gb):
    """`X` cru e `y` da particao de validacao, usada como referencia da linha de base."""
    return contrato_gb["x"]["validacao"], contrato_gb["y"]["validacao"]


def test_criar_pipeline_nao_ajusta_o_preprocessador_que_recebeu(contrato_gb):
    """CR01: o `ColumnTransformer` do contrato entra clonado, nunca remontado.

    Se o `clone` sair da funcao, o objeto que `preparar_matriz` ja ajustou sobre
    o treino inteiro passa a ser o mesmo de dentro do pipeline, e cada `fit` de
    fold o reajusta por baixo, sem aviso.
    """
    do_contrato = contrato_gb["preprocessador"]
    pipeline = criar_pipeline_gradient_boosting(do_contrato)

    assert pipeline.named_steps[PASSO_PREPARO] is not do_contrato

    x, y = contrato_gb["x"]["treino"], contrato_gb["y"]["treino"]
    pipeline.fit(x, y)

    assert pipeline.named_steps[PASSO_PREPARO] is not contrato_gb["preprocessador"]


def test_preprocessador_do_pipeline_nasce_nao_ajustado(contrato_gb):
    """O `clone` copia a especificacao e descarta o estado ja ajustado."""
    pipeline = criar_pipeline_gradient_boosting(contrato_gb["preprocessador"])

    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps[PASSO_PREPARO])


def test_pipeline_nasce_com_hiperparametros_padrao_exceto_early_stopping(contrato_gb):
    """Linha de base do card: nenhum eixo de busca e tocado, so `early_stopping`.

    `early_stopping=False` nao e escolha de tuning, e a mesma trava de
    vazamento por Cliente que `modelo.HIPERPARAMETROS_CANDIDATO` (#103) ja
    documenta: no padrao `"auto"` a biblioteca separaria uma fatia aleatoria do
    ajuste que ignora `ID_GOLDENRECORD`.
    """
    padrao = HistGradientBoostingClassifier()
    estimador = criar_pipeline_gradient_boosting(
        contrato_gb["preprocessador"]
    ).named_steps[PASSO_MODELO]

    assert estimador.early_stopping is False
    assert estimador.learning_rate == padrao.learning_rate
    assert estimador.max_iter == padrao.max_iter
    assert estimador.max_leaf_nodes == padrao.max_leaf_nodes
    assert estimador.l2_regularization == padrao.l2_regularization
    assert estimador.min_samples_leaf == padrao.min_samples_leaf


def test_semente_chega_ao_estimador_e_nao_fica_implicita(contrato_gb):
    """Semente ausente e o defeito silencioso: o resultado muda sem nada quebrar."""
    estimador = criar_pipeline_gradient_boosting(
        contrato_gb["preprocessador"]
    ).named_steps[PASSO_MODELO]
    assert estimador.random_state == SEMENTE_PADRAO

    outro = criar_pipeline_gradient_boosting(contrato_gb["preprocessador"], random_state=7)
    assert outro.named_steps[PASSO_MODELO].random_state == 7


def test_pipeline_nao_importa_selecao_de_modelo():
    """CR02: o modulo nao pode tocar `sklearn.model_selection`.

    E de la que sairia qualquer particionador ou divisao aleatoria (a lista
    esta em `PARTICIONADORES_PROIBIDOS`, montada por partes). A validacao
    deste projeto sao os folds do contrato; um splitter proprio aqui seria uma
    terceira particao competindo com as que o grupo ja acordou.
    """
    fonte = Path(ensembles.__file__).read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    modulos = {
        alias.name
        for no in ast.walk(arvore)
        if isinstance(no, ast.Import)
        for alias in no.names
    } | {
        no.module
        for no in ast.walk(arvore)
        if isinstance(no, ast.ImportFrom) and no.module
    }
    assert not [m for m in modulos if "model_selection" in m]


def test_criar_pipeline_gradient_boosting_nao_monta_pre_processamento_proprio():
    """CR01: o pre-processamento e so o do contrato, clonado."""
    fonte = Path(ensembles.__file__).read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    modulos = {
        alias.name
        for no in ast.walk(arvore)
        if isinstance(no, ast.Import)
        for alias in no.names
    } | {
        no.module
        for no in ast.walk(arvore)
        if isinstance(no, ast.ImportFrom) and no.module
    }
    pacotes = ("sklearn.preprocessing", "sklearn.compose", "sklearn.impute")
    assert not [m for m in modulos if m.startswith(pacotes)]


def test_duas_construcoes_com_a_mesma_semente_dao_a_mesma_previsao(treino, contrato_gb):
    """CR03: mesma semente e mesma particao tem que devolver o mesmo resultado.

    Os dois pipelines sao construidos do zero, e nao reaproveitados: e isso que
    a busca do #190 faz a cada ponto sorteado, e o que quem revisa faz ao tentar
    repetir o resultado do notebook.
    """
    x, y = treino

    primeiro = criar_pipeline_gradient_boosting(contrato_gb["preprocessador"]).fit(x, y)
    segundo = criar_pipeline_gradient_boosting(contrato_gb["preprocessador"]).fit(x, y)

    assert np.array_equal(primeiro.predict_proba(x), segundo.predict_proba(x))
    assert np.array_equal(primeiro.predict(x), segundo.predict(x))


def test_medir_linha_de_base_repassa_a_probabilidade_da_classe_positiva(treino, avaliacao, contrato_gb):
    """`y_proba` tem que ser a coluna 1 de `predict_proba`, a de Detrator.

    O `avaliar` real e do card 05 (#241), que ainda nao esta em `develop`: o
    espiao abaixo fica no lugar dele so para conferir que `medir_linha_de_base`
    passa os tres vetores certos adiante, sem calcular metrica nenhuma aqui.
    """
    x_treino, y_treino = treino
    x_avaliacao, y_avaliacao = avaliacao
    recebido = {}

    def avaliar_espiao(y_true, y_pred, y_proba):
        recebido.update(y_true=y_true, y_pred=y_pred, y_proba=y_proba)
        return {"F2": 0.5}

    pipeline = criar_pipeline_gradient_boosting(contrato_gb["preprocessador"])
    devolvido = medir_linha_de_base(
        pipeline, x_treino, y_treino, x_avaliacao, y_avaliacao, avaliar_espiao
    )

    ajustado = pipeline  # medir_linha_de_base ajusta o pipeline recebido
    assert np.array_equal(recebido["y_proba"], ajustado.predict_proba(x_avaliacao)[:, 1])
    assert np.array_equal(recebido["y_pred"], ajustado.predict(x_avaliacao))
    assert recebido["y_true"] is y_avaliacao
    # O dicionario volta como `avaliar` devolveu, sem chave renomeada nem
    # metrica acrescentada: quem define o vocabulario da tabela do 18A.1 e o
    # card #241.
    assert devolvido["metricas"] == {"F2": 0.5}
    assert devolvido["n_treino"] == len(y_treino)
    assert devolvido["n_avaliacao"] == len(y_avaliacao)
    assert devolvido["random_state"] == SEMENTE_PADRAO


def test_medir_linha_de_base_recusa_avaliar_que_nao_e_funcao(treino, avaliacao, contrato_gb):
    """Passar o dicionario de metricas no lugar da funcao e o engano provavel."""
    x_treino, y_treino = treino
    x_avaliacao, y_avaliacao = avaliacao
    pipeline = criar_pipeline_gradient_boosting(contrato_gb["preprocessador"])

    with pytest.raises(TypeError, match="card 05"):
        medir_linha_de_base(
            pipeline, x_treino, y_treino, x_avaliacao, y_avaliacao, {"F2": 0.5}
        )


def test_medir_linha_de_base_nao_precisa_do_avaliar_real_do_card_05(treino, avaliacao, contrato_gb):
    """Fallback do card: o pipeline e a medicao funcionam com um `avaliar` sintetico.

    Se o card 05 (#241) atrasar, este e o teste que prova que o pipeline e a
    medicao continuam commitaveis no D2: nenhuma importacao de `avaliacao` real
    e feita neste modulo.
    """
    x_treino, y_treino = treino
    x_avaliacao, y_avaliacao = avaliacao
    pipeline = criar_pipeline_gradient_boosting(contrato_gb["preprocessador"])

    resultado = medir_linha_de_base(
        pipeline, x_treino, y_treino, x_avaliacao, y_avaliacao,
        lambda y_true, y_pred, y_proba: {"metrica_sintetica": 1.0},
    )

    assert resultado["metricas"] == {"metrica_sintetica": 1.0}


# ---------------------------------------------------------------------------
# Melhor Gradient Boosting pelo JSON versionado (#190, CR04)
# ---------------------------------------------------------------------------

# Registro no formato que `busca_gradient_boosting.salvar_resultados` grava, com
# valores pequenos para o ajuste dos testes caber em segundos. A semente e 7, e
# nao `SEMENTE_PADRAO`, para que um teste perceba se a funcao ignorar o JSON.
REGISTRO_TESTE = {
    "hiperparametros": {
        "class_weight": "balanced",
        "l2_regularization": 0.5,
        "learning_rate": 0.1,
        "max_iter": 12,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 5,
    },
    "random_state": 7,
    "n_iter": 40,
    "n_folds": 5,
    "melhor_score_medio": 0.5,
    "tempo_total_s": 1.0,
}


@pytest.fixture
def json_vencedor(tmp_path):
    caminho = tmp_path / "hiperparametros_gradient_boosting.json"
    caminho.write_text(json.dumps(REGISTRO_TESTE), encoding="utf-8")
    return caminho


def test_melhor_gradient_boosting_devolve_pipeline_nao_ajustado(contrato_gb, json_vencedor):
    """CR04: quem decide onde ajustar e a dupla de Metricas, nao esta funcao."""
    pipeline = melhor_gradient_boosting(contrato_gb["preprocessador"], caminho=json_vencedor)

    assert [nome for nome, _ in pipeline.steps] == [PASSO_PREPARO, PASSO_MODELO]
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps[PASSO_MODELO])
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps[PASSO_PREPARO])
    # O pre-processador do contrato entra clonado, como em todo pipeline do #188.
    assert pipeline.named_steps[PASSO_PREPARO] is not contrato_gb["preprocessador"]


def test_melhor_gradient_boosting_fixa_a_semente_do_json(contrato_gb, json_vencedor):
    """CR04: `random_state` fixo, e o da busca, nao o padrao do modulo."""
    estimador = melhor_gradient_boosting(
        contrato_gb["preprocessador"], caminho=json_vencedor
    ).named_steps[PASSO_MODELO]

    assert estimador.random_state == REGISTRO_TESTE["random_state"]
    assert estimador.random_state != SEMENTE_PADRAO


def test_melhor_gradient_boosting_aplica_os_hiperparametros_do_json(contrato_gb, json_vencedor):
    """Cada eixo vencedor chega ao estimador, e a trava do #188 continua valendo."""
    estimador = melhor_gradient_boosting(
        contrato_gb["preprocessador"], caminho=json_vencedor
    ).named_steps[PASSO_MODELO]

    for nome, valor in REGISTRO_TESTE["hiperparametros"].items():
        assert estimador.get_params()[nome] == valor
    assert estimador.early_stopping is False


def test_melhor_gradient_boosting_e_reprodutivel(treino, avaliacao, contrato_gb, json_vencedor):
    """Duas reconstrucoes ajustadas no mesmo treino preveem igual, bit a bit."""
    x_treino, y_treino = treino
    x_avaliacao, _ = avaliacao

    primeiro = melhor_gradient_boosting(contrato_gb["preprocessador"], caminho=json_vencedor).fit(x_treino, y_treino)
    segundo = melhor_gradient_boosting(contrato_gb["preprocessador"], caminho=json_vencedor).fit(x_treino, y_treino)

    assert np.array_equal(primeiro.predict_proba(x_avaliacao), segundo.predict_proba(x_avaliacao))


def test_melhor_gradient_boosting_recusa_json_sem_semente(contrato_gb, tmp_path):
    """Sem semente o ensemble reconstruido seria outro a cada chamada."""
    caminho = tmp_path / "sem_semente.json"
    registro = {chave: valor for chave, valor in REGISTRO_TESTE.items() if chave != "random_state"}
    caminho.write_text(json.dumps(registro), encoding="utf-8")

    with pytest.raises(ValueError, match="random_state"):
        melhor_gradient_boosting(contrato_gb["preprocessador"], caminho=caminho)


def test_json_versionado_reconstroi_o_vencedor_da_busca(contrato_gb):
    """O arquivo em `documents/extras/resultados/` e o que a dupla de Metricas
    consome sem argumento.

    Confere que ele existe, que a semente e a do notebook (42), que a busca
    cumpriu o minimo de 40 iteracoes e que os eixos vencedores sao exatamente os
    do espaco do #186: um eixo a mais ou a menos diria que o JSON saiu de outra
    busca.
    """
    registro = json.loads(ARQUIVO_HIPERPARAMETROS_GRADIENT_BOOSTING.read_text(encoding="utf-8"))
    assert registro["random_state"] == SEMENTE_PADRAO
    assert registro["n_iter"] >= N_AMOSTRAS_MINIMO
    assert set(registro["hiperparametros"]) == set(ESPACO_GRADIENT_BOOSTING_HISTGB)

    pipeline = melhor_gradient_boosting(contrato_gb["preprocessador"])
    estimador = pipeline.named_steps[PASSO_MODELO]
    assert estimador.random_state == SEMENTE_PADRAO
    with pytest.raises(NotFittedError):
        check_is_fitted(estimador)

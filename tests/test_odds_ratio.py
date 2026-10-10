"""Testes do mapeamento de coeficiente para feature original (card #212).

O que se testa aqui e o lugar onde o calculo de odds ratio erra em silencio: a
volta da coluna da matriz para a variavel do contrato. Um mapeamento deslocado
em uma posicao nao levanta erro nenhum, produz uma tabela inteira com numeros
plausiveis, e so aparece quando alguem estranha o sinal de uma variavel.

O pre-processador vem de `matriz._montar_preprocessador`, o mesmo que o contrato
usa, e nao de uma copia montada aqui: uma copia passaria a concordar consigo
mesma no dia em que o contrato mudasse de estrutura.

Executar com:  pytest tests/test_odds_ratio.py -v
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from matriz import _montar_preprocessador
from odds_ratio import (efeito_acumulado, mapear_colunas, odds_ratio,
                        odds_ratio_de_cenario)

NUMERICAS = ["ATRASO", "TEMPO"]
CATEGORICAS = ["CANAL", "TIER"]


@pytest.fixture
def dados():
    """Duas numericas, uma delas com ausencia, e duas categoricas."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "ATRASO": rng.integers(0, 120, n).astype("float64"),
        # so esta tem ausencia: o indicador tem que sair so para ela
        "TEMPO": pd.Series(rng.integers(40, 300, n), dtype="float64").mask(rng.random(n) < 0.2),
        "CANAL": rng.choice(["WEB", "MOBILE", "AGENCY"], n),
        "TIER": rng.choice(["SAFIRA", "DIAMANTE"], n),
    })


@pytest.fixture
def ajustado(dados):
    rng = np.random.default_rng(7)
    y = pd.Series(rng.integers(0, 2, len(dados)))
    pipeline = Pipeline([
        ("preparo", _montar_preprocessador(NUMERICAS, CATEGORICAS)),
        ("modelo", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(dados, y)
    return pipeline


@pytest.fixture
def mapa(ajustado):
    return mapear_colunas(ajustado.named_steps["preparo"])


def test_mapa_cobre_todas_as_colunas_da_matriz(ajustado, mapa):
    """Uma linha por coluna de saida, na mesma ordem."""
    nomes = list(ajustado.named_steps["preparo"].get_feature_names_out())
    assert list(mapa["coluna_matriz"]) == nomes


def test_cada_nivel_categorico_vira_uma_linha(ajustado, mapa):
    """`drop=None`, entao todos os niveis entram, e cada um aponta para sua variavel."""
    codificador = (ajustado.named_steps["preparo"]
                   .named_transformers_["categoricas"].named_steps["codificar"])
    for coluna, niveis in zip(CATEGORICAS, codificador.categories_):
        linhas = mapa[(mapa["feature_original"] == coluna) & (mapa["tipo"] == "categorica")]
        assert sorted(linhas["nivel"]) == sorted(str(n) for n in niveis)


def test_indicador_de_ausencia_aponta_para_a_variavel_certa(mapa):
    """O indicador mede a ausencia de TEMPO, e nao a ausencia de ATRASO."""
    ausencias = mapa[mapa["tipo"] == "ausencia"]
    assert list(ausencias["feature_original"]) == ["TEMPO"]
    assert ausencias["coluna_matriz"].str.contains("missingindicator_TEMPO").all()


def test_nenhum_nome_gerado_pelo_columntransformer_vaza(mapa):
    """CR01: a coluna de feature traz a variavel do contrato, nunca o nome da saida."""
    assert not mapa["feature_original"].str.contains("__").any()
    assert not mapa["feature_original"].str.startswith("missingindicator_").any()
    assert set(mapa["feature_original"]) == set(NUMERICAS + CATEGORICAS)


def test_mapeamento_deslocado_levanta_em_vez_de_rotular_errado(ajustado, monkeypatch):
    """A trava existe porque o erro desta funcao e silencioso por natureza."""
    preparo = ajustado.named_steps["preparo"]
    nomes = list(preparo.get_feature_names_out())
    monkeypatch.setattr(type(preparo), "get_feature_names_out",
                        lambda self, input_features=None: np.array(nomes[1:] + ["extra"]))
    with pytest.raises(ValueError, match="get_feature_names_out"):
        mapear_colunas(preparo)


def test_odds_ratio_e_a_exponencial_do_coeficiente(ajustado, dados):
    tabela = odds_ratio(ajustado, dados)
    assert len(tabela) == ajustado.named_steps["modelo"].coef_.size
    assert np.allclose(tabela["odds_ratio"], np.exp(tabela["coeficiente"]))


def test_escala_do_scaler_e_o_iqr_do_dado_imputado(ajustado, dados):
    """A escala guardada tem que ser o IQR de verdade, calculado por fora.

    Conferir a coluna contra a formula que a construiu nao provaria nada: os dois
    lados errariam juntos se o mapeamento pegasse o divisor da coluna errada. Por
    isso o IQR e recalculado aqui a partir do dado que o `RobustScaler` viu, que e
    a saida do imputador e nao o dado cru.
    """
    preparo = ajustado.named_steps["preparo"]
    imputador = preparo.named_transformers_["numericas"].named_steps["imputar"]
    imputado = pd.DataFrame(imputador.transform(dados[NUMERICAS])[:, :len(NUMERICAS)],
                            columns=NUMERICAS)
    tabela = odds_ratio(ajustado, dados)
    for coluna in NUMERICAS:
        esperado = imputado[coluna].quantile(0.75) - imputado[coluna].quantile(0.25)
        guardado = tabela.loc[(tabela["feature_original"] == coluna)
                              & (tabela["tipo"] == "numerica"), "escala_do_scaler"].iloc[0]
        assert guardado == pytest.approx(esperado or 1.0)


def test_odds_ratio_por_unidade_desfaz_o_escalonamento(ajustado, dados):
    """`exp(coef)` e por IQR; a coluna por unidade divide de volta."""
    tabela = odds_ratio(ajustado, dados)
    numericas = tabela[tabela["tipo"] == "numerica"]
    assert numericas["escala_do_scaler"].notna().all()
    # A formula em si, agora contra uma escala que o teste acima conferiu por fora:
    # `exp(coef)` e por unidade da matriz, e dividir pela escala devolve o efeito
    # de uma unidade da variavel original.
    assert np.allclose(numericas["odds_ratio_por_unidade"],
                       np.exp(numericas["coeficiente"] / numericas["escala_do_scaler"]))
    # Categoricas e indicadores ja sao 0/1: as duas leituras coincidiriam.
    assert tabela.loc[tabela["tipo"] != "numerica", "odds_ratio_por_unidade"].isna().all()


def test_ordenacao_usa_a_escala_comparavel_e_nao_o_odds_ratio_cru(ajustado, dados):
    """Ordenar por `odds_ratio` compara minuto com dia com nivel, e nao e ranking.

    A tabela ordena por distancia de 1,0 do `odds_ratio_comparavel`, que percorre
    o intervalo p10 a p90 de cada numerica. Uma variavel de faixa larga e efeito
    pequeno por unidade tem que aparecer acima de uma de faixa estreita com o
    mesmo `odds_ratio` cru.
    """
    tabela = odds_ratio(ajustado, dados)
    ordem = np.abs(np.log(tabela["odds_ratio_comparavel"]))
    assert ordem.is_monotonic_decreasing

    numericas = tabela[tabela["tipo"] == "numerica"]
    assert (numericas["faixa_p10_p90"] > 0).all()
    assert np.allclose(numericas["odds_ratio_comparavel"],
                       numericas["odds_ratio_por_unidade"] ** numericas["faixa_p10_p90"])
    # Nas colunas 0/1 o comparavel e o proprio odds ratio: percorrer a faixa e ir de 0 a 1.
    zero_um = tabela[tabela["tipo"] != "numerica"]
    assert np.allclose(zero_um["odds_ratio_comparavel"], zero_um["odds_ratio"])


def test_suporte_conta_as_linhas_que_sustentam_cada_coluna(ajustado, dados):
    """Sem o suporte, o leitor nao sabe se um OR vem da base inteira ou de 2%."""
    tabela = odds_ratio(ajustado, dados)
    ausencia = tabela[tabela["tipo"] == "ausencia"]
    assert (ausencia["n_observado"] == dados["TEMPO"].isna().sum()).all()

    for coluna in CATEGORICAS:
        linhas = tabela[(tabela["feature_original"] == coluna)
                        & (tabela["tipo"] == "categorica")]
        assert linhas["n_observado"].sum() == len(dados)

    numericas = tabela[tabela["tipo"] == "numerica"]
    for linha in numericas.itertuples():
        assert linha.n_observado == dados[linha.feature_original].notna().sum()


def test_efeito_acumulado_compoe_multiplicativamente(ajustado):
    """Um odds ratio por minuto so vira leitura na escala em que a operacao decide."""
    assert efeito_acumulado(1.0028, 60) == pytest.approx(1.0028 ** 60)
    assert efeito_acumulado(1.0, 999) == pytest.approx(1.0)


def test_cenario_mede_o_bloco_que_a_soma_de_coeficientes_erra(ajustado, dados):
    """O centro do `RobustScaler` e o que a soma a mao esquece.

    A coluna escalonada vale zero quando a variavel esta na **mediana** do treino,
    nao quando vale zero. Entao o efeito de levar uma variavel de `mediana` para
    `mediana + d` nao e `exp(coef * (mediana + d))`, e sim `exp(coef * d / escala)`.
    Somar coeficientes a mao confunde os dois e erra sem levantar nada.
    """
    preparo = ajustado.named_steps["preparo"]
    escalador = preparo.named_transformers_["numericas"].named_steps["escalar"]
    mediana = escalador.center_[NUMERICAS.index("ATRASO")]
    escala = escalador.scale_[NUMERICAS.index("ATRASO")]

    base = dados.iloc[[0]].copy()
    base["ATRASO"] = mediana
    deslocada = base.copy()
    deslocada["ATRASO"] = mediana + escala   # exatamente uma unidade da matriz

    razao = odds_ratio_de_cenario(ajustado, pd.concat([base, deslocada]))
    assert razao[0] == pytest.approx(1.0)

    tabela = odds_ratio(ajustado, dados)
    esperado = tabela.loc[(tabela["feature_original"] == "ATRASO")
                          & (tabela["tipo"] == "numerica"), "odds_ratio"].iloc[0]
    assert razao[1] == pytest.approx(esperado, rel=1e-6)

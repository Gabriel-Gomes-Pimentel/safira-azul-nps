"""Testes do congelamento das partições da seção 4.3.

O que se prova aqui não é que a divisão funciona — isso é de `test_split.py` — e
sim que ela **para de mudar**. Um artefato que se deixa reler depois de editado,
um hash que muda de máquina para máquina, ou uma divisão refeita em silêncio a
cada execução tornariam incomparáveis entre si as métricas que os cards #103 a
#110 produzem. Cada teste abaixo fecha uma dessas portas.

Tudo roda sobre dados sintéticos e `tmp_path`. Nenhum teste lê `data/` nem
qualquer base da Azul.

Executar com:  pytest tests/test_congelamento.py -v
"""
import json

import pandas as pd
import pytest

import congelamento
from congelamento import (ARTEFATO_PADRAO, RAIZ, SEMENTE, VERSAO_ARTEFATO,
                          carregar_registro, congelar, congelar_ou_conferir,
                          descrever, hash_indices, impressao_digital_base,
                          obter_particoes)
from split import PARTICOES, conferir, dividir, resumo

CORTES = {"corte_validacao": "2025-07-01", "corte_teste": "2026-01-01"}


@pytest.fixture
def base():
    """Oito respostas cobrindo os três períodos, com Cliente recorrente.

    O Cliente 10 aparece no treino e no teste, o que exercita o desempate por
    recorrência: o artefato precisa congelar a divisão já desempatada, e não a
    ingênua por data.
    """
    return pd.DataFrame({
        "RESPONDENT_ID": [1, 2, 3, 4, 5, 6, 7, 8],
        "ID_GOLDENRECORD": [10, 10, 20, 30, 40, 50, 60, 70],
        "DATA_STD": ["2024-03-01", "2026-02-01", "2024-05-01", "2024-07-01",
                     "2025-09-01", "2025-11-01", "2026-03-01", "2024-09-01"],
        "DETRATOR": [1, 0, 1, 0, 1, 0, 1, 0],
    })


@pytest.fixture
def arquivo_base(tmp_path):
    """Um arquivo no lugar da base analítica: aqui só o conteúdo importa."""
    caminho = tmp_path / "base_analitica.parquet"
    caminho.write_bytes(b"conteudo da base v1")
    return caminho


@pytest.fixture
def artefato(tmp_path):
    return tmp_path / "particoes_modelagem.json"


# -------------------------------------------------------------- hash e conteúdo
def test_mesma_base_e_mesmos_cortes_produzem_o_mesmo_hash(base, arquivo_base, tmp_path):
    """CR01: duas pessoas partindo da mesma base chegam aos mesmos conjuntos."""
    _, primeiro = obter_particoes(base, caminho_base=arquivo_base,
                                  caminho=tmp_path / "a.json", **CORTES)
    _, segundo = obter_particoes(base, caminho_base=arquivo_base,
                                 caminho=tmp_path / "b.json", **CORTES)

    assert primeiro["hash_indices"] == segundo["hash_indices"]
    assert primeiro["indices"] == segundo["indices"]


def test_hash_cobre_o_conteudo_e_nao_o_caminho_do_arquivo(base, arquivo_base, tmp_path):
    """Hash de caminho mudaria de máquina para máquina e não congelaria nada."""
    particoes, registro = obter_particoes(base, caminho_base=arquivo_base,
                                          caminho=tmp_path / "pasta_a" / "p.json",
                                          **CORTES)
    outro = congelar(particoes, versao_base="v1", parametros=registro["parametros"],
                     caminho=tmp_path / "pasta_b" / "outro_nome.json")

    assert outro["hash_indices"] == registro["hash_indices"]


def test_hash_muda_quando_um_unico_indice_muda(base, arquivo_base, artefato):
    _, registro = obter_particoes(base, caminho_base=arquivo_base,
                                  caminho=artefato, **CORTES)

    adulterado = {nome: list(indices) for nome, indices in registro["indices"].items()}
    adulterado["treino"] = adulterado["treino"][:-1]

    assert hash_indices(adulterado) != registro["hash_indices"]


def test_hash_distingue_as_particoes_entre_si():
    """Trocar validação por teste não pode produzir o mesmo digest."""
    original = {"treino": [1, 2], "validacao": [3], "teste": [4]}
    trocado = {"treino": [1, 2], "validacao": [4], "teste": [3]}

    assert hash_indices(original) != hash_indices(trocado)


def test_impressao_digital_muda_com_o_conteudo_da_base(tmp_path):
    primeiro, segundo = tmp_path / "v1.parquet", tmp_path / "v2.parquet"
    primeiro.write_bytes(b"conteudo da base v1")
    segundo.write_bytes(b"conteudo da base v2")

    assert impressao_digital_base(primeiro) != impressao_digital_base(segundo)


# --------------------------------------------------------- o congelamento congela
def test_segunda_execucao_carrega_sem_refazer_a_divisao(base, arquivo_base,
                                                        artefato, monkeypatch):
    """CR04: refazer a divisão em cada card anularia o congelamento."""
    _, primeiro = obter_particoes(base, caminho_base=arquivo_base,
                                  caminho=artefato, **CORTES)

    def recusar(*args, **kwargs):
        raise AssertionError("A divisão foi refeita apesar de o artefato existir.")

    monkeypatch.setattr(congelamento, "dividir", recusar)
    particoes, segundo = obter_particoes(base, caminho_base=arquivo_base,
                                         caminho=artefato, **CORTES)

    assert segundo["hash_indices"] == primeiro["hash_indices"]
    for nome in PARTICOES:
        assert list(particoes[nome].index) == primeiro["indices"][nome]


def test_particoes_carregadas_passam_na_conferencia_do_split(base, arquivo_base,
                                                             artefato):
    """O retorno tem a forma de `dividir`, então `conferir` e `resumo` valem nele."""
    obter_particoes(base, caminho_base=arquivo_base, caminho=artefato, **CORTES)
    particoes, registro = obter_particoes(base, caminho_base=arquivo_base,
                                          caminho=artefato, **CORTES)

    conferir(particoes, metadados=registro["metadados_divisao"])
    tabela = resumo(particoes, metadados=registro["metadados_divisao"])
    assert list(tabela.index) == list(PARTICOES)


def test_regerar_so_acontece_quando_pedido_explicitamente(base, arquivo_base, artefato):
    """O escape existe, mas nunca é acionado por omissão."""
    _, primeiro = obter_particoes(base, caminho_base=arquivo_base,
                                  caminho=artefato, **CORTES)
    _, segundo = obter_particoes(base, caminho_base=arquivo_base, caminho=artefato,
                                 regerar=True, **CORTES)

    assert segundo["hash_indices"] == primeiro["hash_indices"]   # base igual, split igual


# ------------------------------------------------------------- recusas na leitura
def test_recusa_artefato_com_indices_adulterados(base, arquivo_base, artefato):
    """Editar o JSON à mão é o caminho mais curto para quebrar a comparação."""
    obter_particoes(base, caminho_base=arquivo_base, caminho=artefato, **CORTES)

    registro = json.loads(artefato.read_text(encoding="utf-8"))
    registro["indices"]["treino"].append(999)
    artefato.write_text(json.dumps(registro), encoding="utf-8")

    with pytest.raises(ValueError, match="Hash do artefato nao confere"):
        carregar_registro(artefato)


def test_recusa_artefato_gerado_sobre_outra_versao_da_base(base, arquivo_base,
                                                           artefato, tmp_path):
    """A divisão congelada não vale para uma base cujo conteúdo mudou."""
    obter_particoes(base, caminho_base=arquivo_base, caminho=artefato, **CORTES)

    outra_base = tmp_path / "base_analitica_v2.parquet"
    outra_base.write_bytes(b"conteudo da base v2")

    with pytest.raises(ValueError, match="outra versao da base"):
        obter_particoes(base, caminho_base=outra_base, caminho=artefato, **CORTES)


def test_recusa_execucao_com_outros_parametros_de_corte(base, arquivo_base, artefato):
    """Mudar a política de particionamento é decisão do #127, não efeito colateral."""
    obter_particoes(base, caminho_base=arquivo_base, caminho=artefato, **CORTES)

    with pytest.raises(ValueError, match="outros parametros de corte"):
        obter_particoes(base, caminho_base=arquivo_base, caminho=artefato,
                        corte_validacao="2025-08-01", corte_teste="2026-01-01")


def test_recusa_artefato_de_formato_anterior(base, arquivo_base, artefato):
    obter_particoes(base, caminho_base=arquivo_base, caminho=artefato, **CORTES)

    registro = json.loads(artefato.read_text(encoding="utf-8"))
    registro["versao_artefato"] = VERSAO_ARTEFATO - 1
    artefato.write_text(json.dumps(registro), encoding="utf-8")

    with pytest.raises(ValueError, match="Regere o artefato"):
        carregar_registro(artefato)


def test_recusa_indice_que_nao_existe_mais_no_dataframe(base, arquivo_base, artefato):
    """Base reordenada ou filtrada depois do congelamento tem que parar."""
    obter_particoes(base, caminho_base=arquivo_base, caminho=artefato, **CORTES)

    with pytest.raises(ValueError, match="nao existem no DataFrame"):
        obter_particoes(base.iloc[:4], caminho_base=arquivo_base,
                        caminho=artefato, **CORTES)


def test_recusa_carregamento_sem_artefato(artefato):
    with pytest.raises(FileNotFoundError, match="obter_particoes"):
        carregar_registro(artefato)


def test_recusa_base_de_origem_inexistente(base, tmp_path, artefato):
    with pytest.raises(FileNotFoundError, match="Base de origem"):
        obter_particoes(base, caminho_base=tmp_path / "nao_existe.parquet",
                        caminho=artefato, **CORTES)


def test_recusa_congelar_sem_as_tres_particoes(base, arquivo_base, artefato):
    particoes, registro = obter_particoes(base, caminho_base=arquivo_base,
                                          caminho=artefato, **CORTES)
    del particoes["validacao"]

    with pytest.raises(KeyError, match="validacao"):
        congelar(particoes, versao_base="v1", parametros=registro["parametros"],
                 caminho=artefato)


# --------------------------------------------------- procedência e versionamento
def test_artefato_padrao_fica_sob_data_coberto_pelo_gitignore():
    """CR03: nenhum índice da base do parceiro pode ser versionado."""
    relativo = ARTEFATO_PADRAO.relative_to(RAIZ)

    assert relativo.parts[0] == "data"
    assert "data/" in (RAIZ / ".gitignore").read_text(encoding="utf-8")


def test_registro_traz_semente_versao_da_base_hash_e_cortes(base, arquivo_base,
                                                            artefato):
    """CR02: os quatro precisam estar no artefato para chegarem ao notebook."""
    _, registro = obter_particoes(base, caminho_base=arquivo_base,
                                  caminho=artefato, **CORTES)

    assert registro["semente"] == SEMENTE
    assert registro["versao_base"] == impressao_digital_base(arquivo_base)
    assert len(registro["hash_indices"]) == 64
    assert registro["parametros"]["corte_validacao"] == "2025-07-01"
    assert registro["parametros"]["corte_teste"] == "2026-01-01"


def test_descricao_expoe_o_que_a_secao_4_3_precisa_relatar(base, arquivo_base,
                                                           artefato):
    _, registro = obter_particoes(base, caminho_base=arquivo_base,
                                  caminho=artefato, **CORTES)
    texto = descrever(registro)

    assert str(SEMENTE) in texto
    assert registro["hash_indices"][:16] in texto
    assert registro["versao_base"][:16] in texto
    assert "2025-07-01" in texto
    for nome in PARTICOES:
        assert nome in texto


def test_metadados_da_divisao_sao_preservados_no_artefato(base, arquivo_base,
                                                          artefato):
    """Sem eles o artefato diria quais linhas, mas não por que foram essas."""
    _, registro = obter_particoes(base, caminho_base=arquivo_base,
                                  caminho=artefato, **CORTES)
    metadados = registro["metadados_divisao"]

    assert metadados["linhas_removidas_por_recorrencia"] == 1   # o Cliente 10
    assert metadados["politica_sem_data"] == "treino"


# ------------------------------------- congelamento sobre particoes ja prontas
# `congelar_ou_conferir` existe para o caminho do notebook, em que a divisao ja
# aconteceu dentro de `preparar_matriz`. O que estes testes fecham e a porta de
# a matriz e o artefato descreverem conjuntos diferentes sem ninguem notar.
@pytest.fixture
def particoes_prontas(base):
    particoes, metadados = dividir(base, **CORTES, sem_data="treino")
    return particoes, metadados


def test_congela_as_particoes_recebidas_sem_chamar_dividir(
    particoes_prontas, arquivo_base, artefato, monkeypatch
):
    """A funcao nao reparticiona: recebe pronto, e e isso que a distingue."""
    particoes, metadados = particoes_prontas

    def recusar(*args, **kwargs):
        raise AssertionError("congelar_ou_conferir nao pode chamar dividir.")

    monkeypatch.setattr(congelamento, "dividir", recusar)

    registro = congelar_ou_conferir(
        particoes,
        caminho_base=arquivo_base,
        parametros=CORTES,
        metadados=metadados,
        caminho=artefato,
    )

    assert artefato.exists()
    assert registro["hash_indices"] == hash_indices(
        {nome: [int(i) for i in particoes[nome].index] for nome in PARTICOES}
    )


def test_segunda_execucao_confere_as_mesmas_particoes(
    particoes_prontas, arquivo_base, artefato
):
    particoes, metadados = particoes_prontas
    primeiro = congelar_ou_conferir(
        particoes, caminho_base=arquivo_base, parametros=CORTES,
        metadados=metadados, caminho=artefato,
    )
    segundo = congelar_ou_conferir(
        particoes, caminho_base=arquivo_base, parametros=CORTES, caminho=artefato,
    )

    assert segundo["hash_indices"] == primeiro["hash_indices"]
    assert segundo["gerado_em"] == primeiro["gerado_em"], "o artefato foi reescrito"


def test_recusa_particoes_que_nao_sao_as_congeladas(
    particoes_prontas, arquivo_base, artefato
):
    """O caso que motiva a funcao: a base mudou e a matriz nao percebeu."""
    particoes, metadados = particoes_prontas
    congelar_ou_conferir(
        particoes, caminho_base=arquivo_base, parametros=CORTES,
        metadados=metadados, caminho=artefato,
    )

    alteradas = dict(particoes)
    alteradas["treino"] = particoes["treino"].iloc[1:]

    with pytest.raises(ValueError, match="nao sao as congeladas"):
        congelar_ou_conferir(
            alteradas, caminho_base=arquivo_base, parametros=CORTES, caminho=artefato,
        )


def test_conferencia_recusa_outra_versao_da_base(
    particoes_prontas, arquivo_base, artefato, tmp_path
):
    particoes, metadados = particoes_prontas
    congelar_ou_conferir(
        particoes, caminho_base=arquivo_base, parametros=CORTES,
        metadados=metadados, caminho=artefato,
    )

    outra = tmp_path / "base_analitica_v2.parquet"
    outra.write_bytes(b"conteudo da base v2")

    with pytest.raises(ValueError, match="outra versao da base"):
        congelar_ou_conferir(
            particoes, caminho_base=outra, parametros=CORTES, caminho=artefato,
        )


def test_conferencia_recusa_outros_parametros_de_corte(
    particoes_prontas, arquivo_base, artefato
):
    particoes, metadados = particoes_prontas
    congelar_ou_conferir(
        particoes, caminho_base=arquivo_base, parametros=CORTES,
        metadados=metadados, caminho=artefato,
    )

    with pytest.raises(ValueError, match="outros parametros de corte"):
        congelar_ou_conferir(
            particoes,
            caminho_base=arquivo_base,
            parametros={"corte_validacao": "2025-08-01", "corte_teste": "2026-01-01"},
            caminho=artefato,
        )


def test_regerar_reescreve_o_artefato_com_as_particoes_atuais(
    particoes_prontas, arquivo_base, artefato
):
    particoes, metadados = particoes_prontas
    congelar_ou_conferir(
        particoes, caminho_base=arquivo_base, parametros=CORTES,
        metadados=metadados, caminho=artefato,
    )

    alteradas = dict(particoes)
    alteradas["treino"] = particoes["treino"].iloc[1:]
    registro = congelar_ou_conferir(
        alteradas, caminho_base=arquivo_base, parametros=CORTES,
        caminho=artefato, regerar=True,
    )

    assert len(registro["indices"]["treino"]) == len(particoes["treino"]) - 1


def test_conferencia_exige_as_tres_particoes(
    particoes_prontas, arquivo_base, artefato
):
    particoes, metadados = particoes_prontas
    congelar_ou_conferir(
        particoes, caminho_base=arquivo_base, parametros=CORTES,
        metadados=metadados, caminho=artefato,
    )

    with pytest.raises(KeyError, match="particoes ausentes"):
        congelar_ou_conferir(
            {"treino": particoes["treino"]},
            caminho_base=arquivo_base, parametros=CORTES, caminho=artefato,
        )


def test_artefato_aceita_metadados_que_o_json_nao_serializa(
    particoes_prontas, arquivo_base, artefato
):
    """preparar_matriz devolve Timestamp e tipos do numpy nos metadados."""
    particoes, metadados = particoes_prontas
    metadados = {**metadados, "gerado_para_teste_em": pd.Timestamp("2026-01-01")}

    congelar_ou_conferir(
        particoes, caminho_base=arquivo_base, parametros=CORTES,
        metadados=metadados, caminho=artefato,
    )

    gravado = json.loads(artefato.read_text(encoding="utf-8"))
    assert gravado["metadados_divisao"]["gerado_para_teste_em"].startswith("2026-01-01")

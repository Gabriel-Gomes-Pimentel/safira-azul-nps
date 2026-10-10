"""
SAFIRA | Secao 4.2.1 - Integracao e limpeza das bases da Azul.

Le os cinco arquivos disponibilizados pelo parceiro, aplica os filtros de
qualidade, constroi as variaveis derivadas e calcula os pesos de
pos-estratificacao. Produz um DataFrame unico com granularidade de voo
avaliado.

As bases NAO sao versionadas (restricao do TAPI). Aponte DATA_DIR para o
diretorio local ou para a pasta montada do Google Drive.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("SAFIRA_DATA_DIR", "data/raw")

CHAVE = "RESPONDENT_ID"

# Divergencia de conteudo aceita entre copias da mesma chave no NPS. A secao
# 4.2.1 aprova exatamente este caso: o registro 49088377 aparece duas vezes com
# TEMPO_VOO de 340 e 1.084 minutos, e a decisao registrada e manter a primeira
# ocorrencia. Qualquer outra coluna divergente bloqueia a execucao.
DIVERGENCIA_APROVADA_NPS = ("TEMPO_VOO",)

# Taxonomia de atraso usada pela Azul. A ordem importa: e a mesma chave da
# base populacional DISTRIBUICAO_PAX_NORMALIZADO.
ORD_ATRASO = ["a. Sem Atraso", "b. 15m - 60m", "c. 61m - 120m", "d. >120m"]

# Faixas de antecedencia do aviso de cancelamento, em dias. Definidas aqui para
# que grafico e tabela usem o mesmo corte e nao possam divergir.
BINS_ANTECEDENCIA = [-1, 0, 1, 3, 7, 15, 30, 60, 1000]
LAB_ANTECEDENCIA = ["Mesmo dia", "1 dia", "2 a 3 d", "4 a 7 d", "8 a 15 d",
                    "16 a 30 d", "31 a 60 d", "Mais de 60 d"]


def faixa_antecedencia(dias: pd.Series) -> pd.Categorical:
    """Discretiza a antecedencia do aviso de cancelamento."""
    return pd.cut(dias, BINS_ANTECEDENCIA, labels=LAB_ANTECEDENCIA)


def _logger(log=None):
    """Devolve uma funcao que imprime e, se houver lista, acumula no log."""
    registrar = log.append if log is not None else (lambda s: None)

    def L(msg):
        registrar(msg)
        print(msg)

    return L


def carregar_bases(data_dir: str | None = None):
    """Le as tres bases transacionais. Retorna (nps, perfil, viagem)."""
    d = data_dir or DATA_DIR
    nps = pd.concat(
        [pd.read_csv(f, low_memory=False)
         for f in sorted(glob.glob(os.path.join(d, "PROJETO_INTELI.NPS_0*.csv")))],
        ignore_index=True,
    )
    perfil = pd.concat(
        [pd.read_csv(f, low_memory=False)
         for f in sorted(glob.glob(os.path.join(d, "PROJETO_INTELI.PERFIL_CLIENTE_0*.csv")))],
        ignore_index=True,
    )
    viagem = pd.read_csv(
        os.path.join(d, "PROJETO_INTELI.INFORMACAO_VIAGEM.csv"), low_memory=False
    )
    return nps, perfil, viagem


def carregar_populacao(data_dir: str | None = None) -> pd.DataFrame:
    """Le a base agregada de distribuicao populacional de passageiros."""
    d = data_dir or DATA_DIR
    dist = pd.read_csv(os.path.join(d, "PROJETO_INTELI.DISTRIBUICAO_PAX_NORMALIZADO.csv"))
    dist["MES_ANO"] = pd.to_datetime(dist["MES"])
    return dist


def faixa_atraso(minutos: pd.Series) -> pd.Categorical:
    """Discretiza o atraso na saida na taxonomia da Azul."""
    banda = np.select(
        [minutos < 15, minutos <= 60, minutos <= 120],
        ORD_ATRASO[:3],
        default=ORD_ATRASO[3],
    )
    return pd.Categorical(banda, categories=ORD_ATRASO, ordered=True)


def duplicatas_divergentes(df, chave: str = CHAVE) -> dict:
    """Mapeia chaves repetidas para as colunas em que as copias discordam.

    Chaves repetidas cujas linhas sao integralmente iguais nao aparecem aqui:
    ja foram eliminadas por drop_duplicates. O que sobra e divergencia real de
    conteudo, que exige decisao explicita.
    """
    repetidas = df[df.duplicated(chave, keep=False)]
    achados = {}
    for valor, grupo in repetidas.groupby(chave, observed=True):
        divergem = [c for c in grupo.columns
                    if c != chave and grupo[c].nunique(dropna=False) > 1]
        if divergem:
            achados[valor] = divergem
    return achados


def deduplicar(df, nome: str, divergencia_aprovada=(), chave: str = CHAVE, log=None):
    """Remove copias integralmente identicas e bloqueia divergencias nao aprovadas.

    O descarte cego pela chave e o que esta regra evita: manter a primeira
    ocorrencia sem olhar o conteudo pode jogar fora dado valido em silencio e
    invalidar a contagem final de registros.
    """
    L = _logger(log)

    n0 = len(df)
    df = df.drop_duplicates()
    L(f"[F1:{nome}] Linhas integralmente duplicadas removidas: {n0 - len(df)}")

    achados = duplicatas_divergentes(df, chave)
    aprovadas = set(divergencia_aprovada)
    bloqueio = {k: v for k, v in achados.items() if set(v) - aprovadas}
    if bloqueio:
        exemplos = list(bloqueio.items())[:5]
        raise ValueError(
            f"{nome}: {len(bloqueio)} chave(s) {chave} repetida(s) com divergencia "
            f"fora do que foi aprovado. Exemplos (chave -> colunas): {exemplos}. "
            "Defina e documente a regra de deduplicacao antes de prosseguir."
        )

    if achados:
        n0 = len(df)
        df = df.drop_duplicates(subset=chave, keep="first")
        L(f"[F2:{nome}] {len(achados)} chave(s) com divergencia aprovada em "
          f"{sorted(aprovadas)}: {n0 - len(df)} linha(s) removida(s), mantida a primeira")
        for valor, colunas in list(achados.items())[:10]:
            L(f"          {chave}={valor} divergia em {colunas}")

    return df


def conferir_cobertura(nps, perfil, viagem, chave: str = CHAVE,
                       exigir_integral: bool = True, log=None) -> None:
    """Confere a cobertura da chave antes da juncao interna.

    A secao 4.2.1 afirma relacao 1:1 com cobertura integral. Sem esta checagem,
    um inner join descartaria respostas sem correspondencia sem qualquer aviso,
    e o total final de registros deixaria de significar o que a documentacao diz.
    """
    L = _logger(log)
    ids_nps = set(nps[chave])
    auxiliares = {"perfil": set(perfil[chave]), "viagem": set(viagem[chave])}

    ausentes = {}
    for nome, ids in auxiliares.items():
        faltam = ids_nps - ids
        ausentes[nome] = faltam
        L(f"[F3a] Cobertura do NPS em {nome}: "
          f"{(1 - len(faltam) / len(ids_nps)) * 100:.2f}% "
          f"({len(faltam)} chave(s) sem correspondencia)")
        sobra = ids - ids_nps
        if sobra:
            L(f"[F3a] {nome} tem {len(sobra)} chave(s) sem resposta de pesquisa correspondente")

    if exigir_integral and any(ausentes.values()):
        detalhe = {nome: len(f) for nome, f in ausentes.items() if f}
        raise ValueError(
            f"Cobertura incompleta antes da juncao: {detalhe}. A secao 4.2.1 documenta "
            "cobertura integral, e a juncao interna descartaria essas respostas em silencio."
        )


def _conferir_coluna_redundante(esquerda, direita, coluna: str, chave: str = CHAVE) -> None:
    """Garante que a coluna repetida nas duas tabelas de fato coincide.

    Descartar a copia sem comparar assume equivalencia em vez de verificar.
    """
    par = esquerda[[chave, coluna]].merge(
        direita[[chave, coluna]], on=chave, how="inner", suffixes=("_esq", "_dir"))
    esq, dir_ = par[f"{coluna}_esq"], par[f"{coluna}_dir"]
    iguais = esq.eq(dir_) | (esq.isna() & dir_.isna())
    if not iguais.all():
        raise ValueError(
            f"A coluna {coluna} diverge entre as tabelas em {int((~iguais).sum())} "
            "registro(s). O descarte da copia foi bloqueado para investigacao."
        )


def integrar(nps, perfil, viagem, log=None, exigir_cobertura_integral: bool = True):
    """Aplica os filtros F1 a F4 e devolve a base analitica integrada."""
    L = _logger(log)
    L(f"[0] Bruto: NPS={len(nps)} PERFIL={len(perfil)} VIAGEM={len(viagem)}")

    # F1 e F2: por tabela. Apenas o NPS tem divergencia aprovada.
    nps = deduplicar(nps, "nps", DIVERGENCIA_APROVADA_NPS, log=log)
    perfil = deduplicar(perfil, "perfil", log=log)
    viagem = deduplicar(viagem, "viagem", log=log)

    # F3a: a cobertura tem que valer antes de qualquer inner join.
    conferir_cobertura(nps, perfil, viagem, log=log,
                       exigir_integral=exigir_cobertura_integral)

    # ID_GOLDENRECORD existe nas duas tabelas. So descarta a copia depois de
    # comprovar que ela coincide com a original.
    _conferir_coluna_redundante(nps, perfil, "ID_GOLDENRECORD")

    # F3: validate garante a cardinalidade 1:1 que a documentacao afirma.
    # Se ela nao valer, o pandas interrompe em vez de duplicar linhas.
    df = (nps
          .merge(perfil.drop(columns=["ID_GOLDENRECORD"]), on=CHAVE,
                 how="inner", validate="one_to_one")
          .merge(viagem, on=CHAVE, how="inner", validate="one_to_one"))
    L(f"[F3] Apos juncao 1:1 validada: {len(df)} linhas x {df.shape[1]} colunas")

    esperado = len(nps)
    if len(df) != esperado:
        raise ValueError(
            f"A juncao devolveu {len(df)} linhas, mas o NPS deduplicado tem {esperado}. "
            "Registros foram perdidos ou multiplicados na integracao."
        )

    # F4: colunas sem poder discriminativo.
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const)
    L(f"[F4] Colunas constantes removidas: {const}")
    return df


def derivar(df, log=None):
    """Constroi alvo e variaveis derivadas do item (f) da secao 4.2.1."""
    L = _logger(log)
    df = df.copy()
    df["DATA_STD"] = pd.to_datetime(df["DATA_STD"])
    df["DETRATOR"] = (df["NPS_PRINCIPAL"] == -100).astype(int)
    df["CLASSE_NPS"] = df["NPS_PRINCIPAL"].map({100: "Promotor", 0: "Neutro", -100: "Detrator"})

    # CANCELAMENTO_VOO costuma vir como booleano, mas se o CSV trouxer texto o
    # astype(bool) marcaria a string 'False' como verdadeira. Normaliza uma vez so.
    if not pd.api.types.is_bool_dtype(df["CANCELAMENTO_VOO"]):
        df["CANCELAMENTO_VOO"] = (df["CANCELAMENTO_VOO"].astype(str).str.strip()
                                  .str.lower().isin(["true", "1", "sim", "s"]))

    # ATENCAO: ASSENTOS traz um assento POR TRECHO do mesmo passageiro
    # (ex.: '20A/17A/28A' para o itinerario FOR/UDI/CNF/POA). Nao mede tamanho
    # de grupo. Verificado por tabulacao cruzada contra N_TRECHOS (rho = 0,984).
    # Mantida apenas como variavel de auditoria, fora do conjunto de preditores.
    df["N_ASSENTOS_AUDIT"] = df["ASSENTOS"].str.count("/").add(1)

    df["N_TRECHOS"] = df["BASE_AIRPORTLEG"].str.count("/")  # trechos = aeroportos - 1
    df["AEROPORTO_ORIGEM"] = df["BASE_AIRPORTLEG"].str.split("/").str[0]
    df["AEROPORTO_DESTINO"] = df["BASE_AIRPORTLEG"].str.split("/").str[-1]
    df["FAIXA_ATRASO"] = faixa_atraso(df["ESTATISTICA_ATRASOSAIDA"])
    df["TRIMESTRE"] = df["DATA_STD"].dt.to_period("Q").astype(str)
    df["MES_ANO"] = df["DATA_STD"].dt.to_period("M").dt.to_timestamp()
    df["N_RESP_CLIENTE"] = df["ID_GOLDENRECORD"].map(df["ID_GOLDENRECORD"].value_counts())

    L("[F5] Derivadas: N_TRECHOS, AEROPORTO_ORIGEM/DESTINO, FAIXA_ATRASO, "
      "MES_ANO, N_RESP_CLIENTE")
    L("[F5b] N_ASSENTOS descartada como preditor: redundante com N_TRECHOS (rho=0,984)")
    return df


def pesos_pos_estratificacao(df, dist, log=None, exigir_cobertura_integral: bool = True):
    """Peso = proporcao populacional / proporcao amostral, por mes x faixa x canal."""
    L = _logger(log)

    pop = (dist.groupby(["MES_ANO", "DELAY_DEPARTURE_RANGE", "CANAL_COMPRA"])
               ["PERC_PAX"].sum().rename("p_pop").reset_index())
    pop["p_pop"] = pop.groupby("MES_ANO")["p_pop"].transform(lambda x: x / x.sum())

    amo = (df.groupby(["MES_ANO", "FAIXA_ATRASO", "CANAL_COMPRA"], observed=True)
             .size().rename("n").reset_index())
    amo["p_amo"] = amo.groupby("MES_ANO")["n"].transform(lambda x: x / x.sum())

    m = amo.merge(
        pop,
        left_on=["MES_ANO", "FAIXA_ATRASO", "CANAL_COMPRA"],
        right_on=["MES_ANO", "DELAY_DEPARTURE_RANGE", "CANAL_COMPRA"],
        how="left",
    )
    m["PESO_POP"] = m["p_pop"] / m["p_amo"]

    # Estrato da amostra sem contrapartida populacional. Antes isso virava peso
    # zero por fillna(0), o que remove o respondente do estimador ponderado sem
    # avisar e desloca a taxa de detracao. Agora e condicao de parada.
    estratos = ["MES_ANO", "FAIXA_ATRASO", "CANAL_COMPRA"]
    sem_peso = m[m["PESO_POP"].isna() | (m["PESO_POP"] <= 0)]
    if len(sem_peso):
        n_respostas = int(sem_peso["n"].sum())
        L(f"[F6] ATENCAO: {len(sem_peso)} estrato(s) sem peso valido, "
          f"cobrindo {n_respostas} resposta(s)")
        if exigir_cobertura_integral:
            amostra = sem_peso[estratos + ["n"]].head(10).to_string(index=False)
            raise ValueError(
                f"Pos-estratificacao incompleta: {len(sem_peso)} estrato(s) "
                f"mes x faixa de atraso x canal sem peso valido, cobrindo "
                f"{n_respostas} resposta(s). A secao 4.2.1 documenta cobertura "
                f"integral da chave. Estratos ausentes:\n{amostra}"
            )

    df = df.merge(
        m[estratos + ["PESO_POP"]], on=estratos, how="left")

    orfaos = df["PESO_POP"].isna()
    if orfaos.any():
        if exigir_cobertura_integral:
            chaves = df.loc[orfaos, estratos].drop_duplicates().head(10).to_string(index=False)
            raise ValueError(
                f"{int(orfaos.sum())} resposta(s) ficaram sem PESO_POP apos a juncao "
                f"com a tabela de estratos. Chaves sem correspondencia:\n{chaves}"
            )
        L(f"[F6] {int(orfaos.sum())} resposta(s) sem peso receberam 0 e saem do estimador ponderado")
        df["PESO_POP"] = df["PESO_POP"].fillna(0)

    L(f"[F6] Peso de pos-estratificacao calculado e validado em {len(m)} estratos "
      f"(mes x faixa_atraso x canal), cobrindo "
      f"{(df['PESO_POP'] > 0).mean() * 100:.2f}% das respostas")
    return df


def pipeline(data_dir: str | None = None, exigir_cobertura_integral: bool = True):
    """Executa o pipeline completo. Retorna (df, log)."""
    log: list[str] = []
    nps, perfil, viagem = carregar_bases(data_dir)
    df = integrar(nps, perfil, viagem, log,
                  exigir_cobertura_integral=exigir_cobertura_integral)
    df = derivar(df, log)
    df = pesos_pos_estratificacao(df, carregar_populacao(data_dir), log,
                                  exigir_cobertura_integral=exigir_cobertura_integral)
    _logger(log)(f"[FINAL] {len(df)} linhas x {df.shape[1]} colunas")
    return df, log


if __name__ == "__main__":
    import pickle

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)

    base, registro = pipeline()
    os.makedirs("out", exist_ok=True)
    with open("out/clean.pkl", "wb") as fh:
        pickle.dump(base, fh)
    with open("out/log_limpeza.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(registro))

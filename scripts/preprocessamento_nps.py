"""Pré-processamento reprodutível da base NPS.

O módulo separa o diagnóstico (feito sobre a base integrada) das transformações
para modelagem. Imputação, codificação e escala são ajustadas somente no treino.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


PADROES = {
    "nps": re.compile(r"NPS[_ .-]", re.IGNORECASE),
    "perfil": re.compile(r"PERFIL_CLIENTE", re.IGNORECASE),
    "viagem": re.compile(r"INFORMACAO_VIAGEM", re.IGNORECASE),
}
EXTENSOES = {".csv", ".xlsx", ".parquet"}
PRIORIDADE_FORMATO = {".parquet": 0, ".csv": 1, ".xlsx": 2}

# Contrato da base integrada descrita em documents/documentacao.md. Qualquer
# mudança de schema precisa ser uma decisão explícita, não um efeito colateral
# de uma nova coluna na origem.
QUANTIDADE_COLUNAS_INTEGRADAS_ESPERADA = 46

# O modelo atual pontua a jornada depois de seu encerramento operacional e
# antes da resposta NPS. Esta allowlist é deliberadamente restritiva: campos
# técnicos, identificadores, respostas da pesquisa e rota/equipamento bruto não
# entram por inferência de tipo/cardinalidade. A disponibilidade temporal em
# t_score continua sendo pré-condição da fonte de dados.
#
# TIER_VIAGEM substitui PERFIL_TUDOAZUL: ambos representam fidelidade, mas o
# primeiro pertence ao perfil associado à viagem e evita carregar duas versões
# semanticamente redundantes do mesmo atributo. N_TRECHOS é a única derivação
# de rota admitida na V1; tem baixa cardinalidade e já possuía regra de negócio
# implementada na exploração.
#
# QTDE_VIAGENS_12M permanece fora enquanto a fonte não fornecer uma contagem
# reconstruída com corte estrito em t_score. A coluna disponível hoje é
# referenciada ao momento da resposta e, portanto, não satisfaz o contrato
# temporal do score.
FEATURE_SET_V1 = [
    "TIER_VIAGEM",
    "VOO_TIPO",
    "TIPO_ENTRETENIMENTO",
    "CANAL_COMPRA",
    "SEGMENTO",
    "ESTATISTICA_ATRASOSAIDA",
    "ATRASO_CHEGADA",
    "CANCELAMENTO_VOO",
    "ANTECEDENCIA_CANCELAMENTO",
    "TEMPO_VOO",
    "N_TRECHOS",
]

FEATURES_CATEGORICAS_V1 = frozenset({
    "TIER_VIAGEM",
    "VOO_TIPO",
    "TIPO_ENTRETENIMENTO",
    "CANAL_COMPRA",
    "SEGMENTO",
    "CANCELAMENTO_VOO",
})

FEATURES_NUMERICAS_V1 = frozenset({
    "ESTATISTICA_ATRASOSAIDA",
    "ATRASO_CHEGADA",
    "ANTECEDENCIA_CANCELAMENTO",
    "TEMPO_VOO",
    "N_TRECHOS",
})

# Quando a jornada é cancelada, o contrato abre a janela de score no registro
# do cancelamento. Informações que dependem da execução/encerramento da jornada
# ainda não existem nesse instante e precisam chegar ao modelo como ausentes,
# mesmo que uma extração retrospectiva as tenha preenchido depois.
FEATURES_POS_ENCERRAMENTO_JORNADA = frozenset({
    "ESTATISTICA_ATRASOSAIDA",
    "ATRASO_CHEGADA",
    "TEMPO_VOO",
    "N_TRECHOS",
})

# Nome anterior preservado como alias imutável para não quebrar notebooks e
# consumidores existentes. Novas implementações devem importar FEATURE_SET_V1.
FEATURES_SCORE_POS_VIAGEM = tuple(FEATURE_SET_V1)

# Campos coletados na pesquisa que origina o target. A trava por prefixo cobre
# também novos NPS_* que apareçam em futuras cargas sem depender deste catálogo.
COLUNAS_PESQUISA_PROIBIDAS = frozenset({
    "NPS_PRINCIPAL",
    "DETRATOR",
    "CATEGORIA_NPS",
    "CLASSE_NPS",
    "SUB_ENTRETENIMENTO1",
    "SUB_ENTRETENIMENTO2",
    "SUB_FIL_MOTIVOVIAGEM",
    "SUB_FIL_FREQUENCIAAZUL",
})

COLUNAS_IDENTIFICADORAS_PROIBIDAS = frozenset({
    "RESPONDENT_ID",
    "ID_GOLDENRECORD",
    "CLIENTE_RECORDLOCATOR",
    "RECORD_LOCATOR",
    "VOO_NUMERO",
})

COLUNA_RESPONDENTE = "RESPONDENT_ID"
COLUNA_CLIENTE = "ID_GOLDENRECORD"
COLUNA_ALVO = "DETRATOR"


def _feature_proibida_por_leakage(coluna: str) -> bool:
    normalizada = coluna.strip().upper()
    return normalizada.startswith("NPS_") or normalizada in COLUNAS_PESQUISA_PROIBIDAS


def _validar_contrato_feature_set_v1() -> None:
    duplicadas = sorted({
        coluna for coluna in FEATURE_SET_V1 if FEATURE_SET_V1.count(coluna) > 1
    })
    proibidas = sorted(coluna for coluna in FEATURE_SET_V1 if _feature_proibida_por_leakage(coluna))
    identificadores = sorted(
        coluna for coluna in FEATURE_SET_V1
        if coluna.strip().upper() in COLUNAS_IDENTIFICADORAS_PROIBIDAS
    )
    tipadas = FEATURES_CATEGORICAS_V1 | FEATURES_NUMERICAS_V1
    sem_tipo = sorted(set(FEATURE_SET_V1) - tipadas)
    tipos_excedentes = sorted(tipadas - set(FEATURE_SET_V1))
    temporais_excedentes = sorted(FEATURES_POS_ENCERRAMENTO_JORNADA - set(FEATURE_SET_V1))
    if duplicadas:
        raise RuntimeError(f"FEATURE_SET_V1 possui feature(s) duplicada(s): {duplicadas}.")
    if proibidas:
        raise RuntimeError(f"FEATURE_SET_V1 contém feature(s) com leakage: {proibidas}.")
    if identificadores:
        raise RuntimeError(f"FEATURE_SET_V1 contém identificador(es): {identificadores}.")
    if sem_tipo or tipos_excedentes or temporais_excedentes:
        raise RuntimeError(
            "O contrato de tipos do FEATURE_SET_V1 está inconsistente: "
            f"sem tipo={sem_tipo}; fora da V1={tipos_excedentes}; "
            f"temporais fora da V1={temporais_excedentes}."
        )


_validar_contrato_feature_set_v1()

# ``NPS_PRINCIPAL`` não guarda a nota bruta de 0 a 10 nesta fonte. A Azul a
# entrega já classificada: -100 (detrator), 0 (neutro) e 100 (promotor).
# A validação explícita impede que uma mudança de codificação na origem
# transforme silenciosamente valores desconhecidos em não detratores.
CODIGOS_NPS_PRINCIPAL_VALIDOS = frozenset((-100, 0, 100))


def criar_target_detrator(df: pd.DataFrame) -> pd.Series:
    """Cria o alvo binário a partir da classificação NPS principal.

    ``1`` representa detrator (``NPS_PRINCIPAL == -100``); ``0`` reúne neutro
    (``0``) e promotor (``100``). A função falha para nulos ou códigos fora da
    escala contratada, evitando classificá-los silenciosamente como classe 0.
    """
    coluna = "NPS_PRINCIPAL"
    if coluna not in df.columns:
        raise KeyError(f"A criação do target exige a coluna {coluna}.")

    nulos = int(df[coluna].isna().sum())
    if nulos:
        raise ValueError(f"{coluna} possui {nulos} valor(es) nulo(s).")

    inesperados = sorted(set(df.loc[
        ~df[coluna].isin(CODIGOS_NPS_PRINCIPAL_VALIDOS), coluna
    ].tolist()))
    if inesperados:
        raise ValueError(
            f"{coluna} possui código(s) fora da escala esperada "
            f"{sorted(CODIGOS_NPS_PRINCIPAL_VALIDOS)}: {inesperados}."
        )

    return (df[coluna] == -100).astype("int8").rename("DETRATOR")


def normalizar_data_std(df: pd.DataFrame, origem: Path) -> pd.DataFrame:
    """Uniformiza ``DATA_STD`` em cada fonte, antes de qualquer concatenação.

    A primeira partição de NPS é Excel e chega ao pandas como ``Timestamp``;
    as demais são CSV e chegam como texto. Converter somente depois do
    ``concat`` deixa uma coluna ``object`` com representações mistas e pode
    transformar datas válidas em ``NaT``. A normalização por arquivo mantém
    uma única representação temporal na integração e falha explicitamente se
    a fonte contiver uma data ausente ou inválida.
    """
    if "DATA_STD" not in df.columns:
        return df

    convertido = pd.to_datetime(df["DATA_STD"], format="mixed", errors="coerce")
    invalidos = convertido.isna()
    if invalidos.any():
        exemplos = df.loc[invalidos, "DATA_STD"].head(5).tolist()
        raise ValueError(
            f"{origem.name} possui {int(invalidos.sum())} valor(es) inválido(s) "
            f"em DATA_STD. Exemplos: {exemplos}."
        )
    df = df.copy()
    df["DATA_STD"] = convertido
    return df


def ler_arquivo(caminho: Path) -> pd.DataFrame:
    """Lê CSV, Excel ou Parquet e uniformiza a data na própria fonte."""
    if caminho.suffix.lower() == ".csv":
        df = pd.read_csv(caminho)
    elif caminho.suffix.lower() == ".xlsx":
        # Calamine é substancialmente mais rápido em planilhas grandes. O
        # fallback mantém compatibilidade com ambientes que só têm openpyxl.
        try:
            df = pd.read_excel(caminho, engine="calamine")
        except ImportError:
            df = pd.read_excel(caminho, engine="openpyxl")
    elif caminho.suffix.lower() == ".parquet":
        validar_parquet(caminho)
        try:
            df = pd.read_parquet(caminho)
        except Exception as erro:
            raise ValueError(f"Parquet inválido ou ilegível: {caminho.name}.") from erro
    else:
        raise ValueError(f"Formato não suportado: {caminho}")
    return normalizar_data_std(df, caminho)


def validar_parquet(caminho: Path) -> None:
    """Confere metadados do Parquet antes que ele entre na integração."""
    try:
        from pyarrow import parquet as pq
    except ImportError as erro:
        raise RuntimeError(
            "A validação de Parquet exige pyarrow; instale as dependências do projeto."
        ) from erro

    try:
        arquivo = pq.ParquetFile(caminho)
        metadados = arquivo.metadata
        if metadados is None or metadados.num_columns == 0:
            raise ValueError("arquivo sem metadados ou sem colunas")
    except Exception as erro:
        raise ValueError(f"Parquet inválido ou corrompido: {caminho.name}.") from erro


def descobrir_fontes(diretorio: str | Path = "data/raw") -> dict[str, list[Path]]:
    """Descobre as fontes e prefere um formato por parte lógica.

    Quando existem CSV e XLSX da mesma parte, ambos são comparados antes de o
    formato prioritário ser usado. Assim, não há concatenação duplicada.
    """
    diretorio = Path(diretorio)
    # Path.iterdir() não garante ordem alguma, e a ordem obtida aqui define a ordem em que as
    # partições são concatenadas e, por consequência, a ordem das linhas da base analítica. Sem
    # ordenar, a mesma base sai com linhas em posições diferentes em cada máquina, e qualquer
    # artefato que dependa de posição muda em silêncio. Ordenar pelo nome torna a base idêntica
    # em qualquer ambiente.
    arquivos = sorted(
        (p for p in diretorio.iterdir() if p.suffix.lower() in EXTENSOES),
        key=lambda caminho: caminho.name.upper(),
    )
    grupos = {nome: [] for nome in PADROES}
    for arquivo in arquivos:
        nome = arquivo.name.upper()
        if "DISTRIBUICAO_PAX_NORMALIZADO" in nome:
            continue
        for grupo, padrao in PADROES.items():
            if padrao.search(nome):
                grupos[grupo].append(arquivo)
                break

    for grupo, encontrados in grupos.items():
        if not encontrados:
            raise FileNotFoundError(f"Nenhum arquivo encontrado para o grupo {grupo}.")
    return grupos


def _chave_parte(caminho: Path) -> str:
    return caminho.stem.replace(" ", "_").replace(".", "_").upper()


def selecionar_partes(arquivos: list[Path]) -> tuple[list[Path], list[Path]]:
    """Compara cópias da mesma parte e retorna arquivos usados e ignorados."""
    por_parte: dict[str, list[Path]] = {}
    for arquivo in arquivos:
        por_parte.setdefault(_chave_parte(arquivo), []).append(arquivo)

    usados, ignorados = [], []
    for parte, alternativas in por_parte.items():
        alternativas.sort(key=lambda p: PRIORIDADE_FORMATO[p.suffix.lower()])
        escolhido = alternativas[0]
        if len(alternativas) == 1:
            usados.append(escolhido)
            continue
        referencia = ler_arquivo(escolhido)
        for alternativa in alternativas[1:]:
            comparacao = ler_arquivo(alternativa)
            if not referencia.equals(comparacao):
                raise ValueError(
                    f"Cópias divergentes para {parte}: {escolhido.name} e "
                    f"{alternativa.name}. Decida qual deve ser usada."
                )
            ignorados.append(alternativa)
        usados.append(escolhido)
    return usados, ignorados


def relatorio_chave(df: pd.DataFrame, nome: str) -> dict[str, object]:
    if "RESPONDENT_ID" not in df:
        raise KeyError(f"{nome} não possui RESPONDENT_ID.")
    return {
        "tabela": nome,
        "linhas": len(df),
        "colunas": len(df.columns),
        "ids_unicos": df["RESPONDENT_ID"].nunique(),
        "ids_nulos": int(df["RESPONDENT_ID"].isna().sum()),
        "ids_duplicados": int(df["RESPONDENT_ID"].duplicated().sum()),
    }


def consolidar_duplicidades_tempo_voo(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Consolida somente cópias iguais exceto por ``TEMPO_VOO``.

    A regra é específica ao caso aprovado: uma única resposta por
    ``RESPONDENT_ID`` e duas medições divergentes de duração. O valor final é a
    média aritmética; qualquer outra divergência permanece um erro bloqueante.
    """
    if "TEMPO_VOO" not in df.columns:
        raise KeyError("A regra de consolidação exige a coluna TEMPO_VOO.")

    resultado = df.copy()
    resultado["TEMPO_VOO_CONSOLIDADO"] = 0
    ids_duplicados = resultado.loc[
        resultado["RESPONDENT_ID"].duplicated(keep=False), "RESPONDENT_ID"
    ].dropna().unique()
    colunas_comparacao = [
        coluna for coluna in resultado.columns
        if coluna not in {"RESPONDENT_ID", "TEMPO_VOO", "TEMPO_VOO_CONSOLIDADO"}
    ]
    remover = []

    for respondent_id in ids_duplicados:
        indices = resultado.index[resultado["RESPONDENT_ID"] == respondent_id]
        grupo = resultado.loc[indices]
        divergentes = [
            coluna for coluna in colunas_comparacao
            if grupo[coluna].nunique(dropna=False) > 1
        ]
        if divergentes:
            raise ValueError(
                f"RESPONDENT_ID {respondent_id} possui divergências além de TEMPO_VOO: "
                f"{divergentes}. A integração foi bloqueada."
            )
        valores_tempo = pd.to_numeric(grupo["TEMPO_VOO"], errors="coerce")
        if valores_tempo.notna().sum() != len(grupo):
            raise ValueError(
                f"RESPONDENT_ID {respondent_id} tem TEMPO_VOO nulo ou inválido; "
                "não é possível aplicar a média com segurança."
            )
        indice_manter = indices[0]
        resultado.loc[indice_manter, "TEMPO_VOO"] = valores_tempo.mean()
        resultado.loc[indice_manter, "TEMPO_VOO_CONSOLIDADO"] = 1
        remover.extend(indices[1:])

    resultado = resultado.drop(index=remover).reset_index(drop=True)
    return resultado, len(ids_duplicados)


def remover_colunas_redundantes(
    esquerda: pd.DataFrame, direita: pd.DataFrame, nome_direita: str
) -> tuple[pd.DataFrame, list[str]]:
    """Remove colunas repetidas somente quando forem iguais por respondente."""
    repetidas = (set(esquerda.columns) & set(direita.columns)) - {"RESPONDENT_ID"}
    remover = []
    for coluna in sorted(repetidas):
        comparacao = esquerda[["RESPONDENT_ID", coluna]].merge(
            direita[["RESPONDENT_ID", coluna]],
            on="RESPONDENT_ID",
            how="inner",
            validate="one_to_one",
            suffixes=("_esquerda", "_direita"),
        )
        esquerda_valor = comparacao[f"{coluna}_esquerda"]
        direita_valor = comparacao[f"{coluna}_direita"]
        equivalentes = (esquerda_valor.eq(direita_valor) |
                         (esquerda_valor.isna() & direita_valor.isna()))
        if not equivalentes.all():
            raise ValueError(
                f"A coluna repetida {coluna} diverge entre NPS e {nome_direita}; "
                "a integração foi bloqueada para investigação."
            )
        remover.append(coluna)
    if remover:
        print(f"{nome_direita}: colunas redundantes comprovadamente equivalentes e removidas: {remover}")
    return direita.drop(columns=remover), remover


def validar_base_integrada(df: pd.DataFrame) -> None:
    """Impede persistência de uma integração com schema inesperado."""
    if df.columns.duplicated().any():
        duplicadas = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"A base integrada possui colunas duplicadas: {duplicadas}.")
    if len(df.columns) != QUANTIDADE_COLUNAS_INTEGRADAS_ESPERADA:
        raise ValueError(
            "A base integrada possui "
            f"{len(df.columns)} colunas; eram esperadas "
            f"{QUANTIDADE_COLUNAS_INTEGRADAS_ESPERADA}."
        )


def integrar_bases(diretorio: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Integra as bases com consolidação auditável do caso aprovado de duplicidade."""
    grupos = descobrir_fontes(diretorio)
    tabelas: dict[str, pd.DataFrame] = {}
    relatorios = []
    for grupo, arquivos in grupos.items():
        usados, ignorados = selecionar_partes(arquivos)
        print(f"{grupo}: usados={[p.name for p in usados]}; ignorados={[p.name for p in ignorados]}")
        tabela = pd.concat([ler_arquivo(p) for p in usados], ignore_index=True)
        duplicados_antes = int(tabela["RESPONDENT_ID"].duplicated().sum())
        consolidados = 0
        if grupo == "nps" and duplicados_antes:
            tabela, consolidados = consolidar_duplicidades_tempo_voo(tabela)
            print(f"nps: {consolidados} RESPONDENT_ID duplicado(s) consolidado(s) pela média de TEMPO_VOO.")
        relatorio = relatorio_chave(tabela, grupo)
        relatorio["ids_duplicados_antes"] = duplicados_antes
        relatorio["ids_consolidados_tempo_voo"] = consolidados
        relatorios.append(relatorio)
        if tabela["RESPONDENT_ID"].duplicated().any():
            raise ValueError(
                f"{grupo} possui RESPONDENT_ID duplicado. O merge one_to_one foi bloqueado; "
                "aprove uma regra de deduplicação antes de persistir a base final."
            )
        tabelas[grupo] = tabela

    perfil_sem_redundancias, removidas_perfil = remover_colunas_redundantes(
        tabelas["nps"], tabelas["perfil"], "perfil"
    )
    integrado = tabelas["nps"].merge(perfil_sem_redundancias, on="RESPONDENT_ID", validate="one_to_one")
    viagem_sem_redundancias, removidas_viagem = remover_colunas_redundantes(
        integrado, tabelas["viagem"], "viagem"
    )
    integrado = integrado.merge(viagem_sem_redundancias, on="RESPONDENT_ID", validate="one_to_one")
    validar_base_integrada(integrado)
    print(f"Colunas redundantes removidas após comparação: {removidas_perfil + removidas_viagem}")
    return integrado, pd.DataFrame(relatorios)


def perfil_qualidade(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "coluna": df.columns,
        "tipo": df.dtypes.astype(str).to_numpy(),
        "nulos": df.isna().sum().to_numpy(),
        "percentual_nulos": (df.isna().mean() * 100).round(2).to_numpy(),
        "valores_unicos": df.nunique(dropna=True).to_numpy(),
    }).sort_values(["percentual_nulos", "valores_unicos"], ascending=False)


def identificar_outliers_iqr(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    registros = []
    for coluna in colunas:
        serie = pd.to_numeric(df[coluna], errors="coerce").dropna()
        q1, q3 = serie.quantile([0.25, 0.75])
        iqr = q3 - q1
        limite_superior = q3 + 1.5 * iqr
        limite_inferior = q1 - 1.5 * iqr
        registros.append({
            "variavel": coluna, "q1": q1, "q3": q3, "p95": serie.quantile(.95),
            "p99": serie.quantile(.99), "min": serie.min(), "max": serie.max(),
            "outliers_iqr": int(((serie < limite_inferior) | (serie > limite_superior)).sum()),
        })
    return pd.DataFrame(registros).round(2)


def padronizar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza espaços e caixa sem preencher nulos ou inventar categorias."""
    resultado = df.copy()
    for coluna in resultado.select_dtypes(include=["object", "string"]):
        resultado[coluna] = resultado[coluna].str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    return resultado


def preparar_base_analitica(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica somente correções semanticamente justificadas e preserva nulos."""
    df = padronizar_categoricas(normalizar_data_std(df_raw, Path("base_analitica")))
    df["DATA_STD_CONVERTIDA"] = df["DATA_STD"]
    df["MES_ANO"] = df["DATA_STD_CONVERTIDA"].dt.to_period("M").astype("string")
    df["DETRATOR"] = criar_target_detrator(df)
    df["CATEGORIA_NPS"] = df["NPS_PRINCIPAL"].map({
        -100: "DETRATOR", 0: "NEUTRO", 100: "PROMOTOR"
    }).astype("string")
    df["TEMPO_VOO_INVALIDO"] = (df["TEMPO_VOO"] <= 0).astype("int8")
    df.loc[df["TEMPO_VOO"] <= 0, "TEMPO_VOO"] = np.nan
    # N_TRECHOS e exigida pelo FEATURE_SET_V1 e e derivada aqui, e nao so sob
    # demanda em selecionar_features_score_pos_viagem, para que o parquet salvo
    # seja autossuficiente: quem carrega a base pronta nao deveria precisar
    # saber que uma feature do contrato ainda depende de materializacao tardia.
    # `materializar_features_v1` e idempotente (retorna df sem mudanca se
    # N_TRECHOS ja existir), entao a chamada tardia downstream continua segura.
    df = materializar_features_v1(df)
    return df


def dividir_treino_teste_temporal_por_cliente(
    df: pd.DataFrame,
    proporcao_teste: float = .2,
) -> tuple[pd.Index, pd.Index, dict[str, object]]:
    """Reserva os meses mais recentes para teste sem repetir Clientes.

    O corte é temporal, em vez de aleatório: o teste contém os últimos meses
    disponíveis. Para impedir vazamento entre respostas recorrentes, todo
    ``ID_GOLDENRECORD`` que ocorre no teste é retirado também do treino.

    Os registros sem ``ID_GOLDENRECORD`` ficam fora da validação. O nulo ocorre
    ao mesmo tempo na pesquisa e no perfil, então não há como saber se duas
    dessas linhas são do mesmo Cliente; tratá-las como grupos unitários
    reintroduziria justamente o vazamento que a divisão agrupada impede. São
    0,02% da base, e a exclusão fica registrada em log e nos metadados.
    """
    colunas_obrigatorias = {"DATA_STD_CONVERTIDA", "ID_GOLDENRECORD"}
    ausentes = colunas_obrigatorias - set(df.columns)
    if ausentes:
        raise KeyError(f"A divisão exige as colunas: {sorted(ausentes)}.")
    if not 0 < proporcao_teste < 1:
        raise ValueError("proporcao_teste deve estar entre 0 e 1.")

    datas = pd.to_datetime(df["DATA_STD_CONVERTIDA"], errors="coerce")
    if datas.isna().any():
        raise ValueError("DATA_STD_CONVERTIDA possui valores inválidos; corrija-os antes do split.")
    grupos = df["ID_GOLDENRECORD"]
    sem_cliente = grupos.isna()
    if sem_cliente.all():
        raise ValueError("ID_GOLDENRECORD é nulo em toda a base; não há o que agrupar por Cliente.")
    if sem_cliente.any():
        print(f"divisão: {int(sem_cliente.sum())} registro(s) sem ID_GOLDENRECORD "
              f"({sem_cliente.mean():.2%} da base) excluído(s) do treino e do teste.")

    meses = datas.dt.to_period("M")
    meses_ordenados = meses.sort_values().unique()
    meses_teste = max(1, int(np.ceil(len(meses_ordenados) * proporcao_teste)))
    primeiro_mes_teste = meses_ordenados[-meses_teste]
    mascara_teste = (meses >= primeiro_mes_teste) & ~sem_cliente
    clientes_teste = set(grupos.loc[mascara_teste])
    mascara_treino = (meses < primeiro_mes_teste) & ~sem_cliente & ~grupos.isin(clientes_teste)

    if not mascara_treino.any() or not mascara_teste.any():
        raise ValueError("A divisão temporal não produziu treino e teste não vazios.")
    if set(grupos.loc[mascara_treino]) & clientes_teste:
        raise AssertionError("Há Cliente presente simultaneamente no treino e no teste.")

    metadados = {
        "primeiro_mes_teste": str(primeiro_mes_teste),
        "ultimo_mes_teste": str(meses.max()),
        "linhas_removidas_por_recorrencia": int(((meses < primeiro_mes_teste) & ~sem_cliente
                                                 & grupos.isin(clientes_teste)).sum()),
        "registros_sem_cliente_excluidos": int(sem_cliente.sum()),
        "clientes_treino": int(grupos.loc[mascara_treino].nunique()),
        "clientes_teste": int(grupos.loc[mascara_teste].nunique()),
    }
    return df.index[mascara_treino], df.index[mascara_teste], metadados


def criar_folds_validacao_por_cliente(
    x_treino: pd.DataFrame, grupos_treino: pd.Series, n_splits: int = 5,
):
    """Gera folds de validação sem que um Cliente apareça em dois folds."""
    if grupos_treino.isna().any():
        raise ValueError("A validação por grupo exige ID_GOLDENRECORD sem nulos.")
    if grupos_treino.nunique() < n_splits:
        raise ValueError("Há Clientes insuficientes para a quantidade de folds solicitada.")
    return GroupKFold(n_splits=n_splits).split(x_treino, groups=grupos_treino)


def derivar_n_trechos(df: pd.DataFrame) -> pd.Series:
    """Deriva a quantidade de trechos da sequência de aeroportos da jornada.

    ``BASE_AIRPORTLEG`` usa o contrato ``ORIGEM[/CONEXAO...]/DESTINO``. A
    função valida esse contrato sem imprimir valores da base e conta os
    separadores; assim, uma jornada ``AAA/BBB`` possui um trecho.
    """
    origem = "BASE_AIRPORTLEG"
    if origem not in df.columns:
        raise KeyError(f"A derivação de N_TRECHOS exige a coluna {origem}.")

    itinerarios = df[origem].astype("string").str.strip()
    invalidos = itinerarios.notna() & (
        ~itinerarios.str.contains("/", regex=False, na=False)
        | itinerarios.str.startswith("/", na=False)
        | itinerarios.str.endswith("/", na=False)
        | itinerarios.str.contains("//", regex=False, na=False)
    )
    if invalidos.any():
        raise ValueError(
            f"{origem} possui {int(invalidos.sum())} itinerário(s) fora do formato contratado."
        )
    return itinerarios.str.count("/").astype("Int64").rename("N_TRECHOS")


def materializar_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """Materializa somente derivações já aprovadas para o Feature Set V1."""
    if "N_TRECHOS" in df.columns or "BASE_AIRPORTLEG" not in df.columns:
        return df
    resultado = df.copy()
    resultado["N_TRECHOS"] = derivar_n_trechos(resultado)
    return resultado


def conferir_features_v1_na_base(df: pd.DataFrame) -> pd.DataFrame:
    """Materializa as derivações da V1 e só então confere a allowlist.

    Um parquet gravado antes do fix(#177) não traz ``N_TRECHOS``, embora traga
    ``BASE_AIRPORTLEG``, de onde ela sai. Conferir ``df.columns`` direto contra
    a allowlist levantava ``KeyError`` para uma feature que o próprio pipeline
    sabe derivar; a conferência agora olha a base já materializada e só falha
    quando a feature não existe nem pode ser derivada.

    Devolve a base materializada, para que o chamador siga com ela.
    """
    base = materializar_features_v1(df)
    ausentes = [coluna for coluna in FEATURE_SET_V1 if coluna not in base.columns]
    if ausentes:
        raise KeyError(
            f"Features da allowlist ausentes na base: {ausentes}. "
            "N_TRECHOS só é derivada quando BASE_AIRPORTLEG está presente; "
            "regere a base com a seção 13 de notebooks/pre-processamento.ipynb."
        )
    return base


def validar_schema_features_v1(df: pd.DataFrame) -> None:
    """Falha para ausência ou mudança de tipo em qualquer feature obrigatória."""
    ausentes = sorted(set(FEATURE_SET_V1) - set(df.columns))
    if ausentes:
        raise KeyError(f"Feature(s) obrigatória(s) ausente(s) no FEATURE_SET_V1: {ausentes}.")

    erros_tipo = []
    for coluna in FEATURE_SET_V1:
        serie = df[coluna]
        if coluna == "CANCELAMENTO_VOO":
            tipo_valido = pd.api.types.is_bool_dtype(serie)
        elif coluna in FEATURES_CATEGORICAS_V1:
            tipo_valido = (
                pd.api.types.is_object_dtype(serie)
                or pd.api.types.is_string_dtype(serie)
                or isinstance(serie.dtype, pd.CategoricalDtype)
            )
        else:
            tipo_valido = (
                pd.api.types.is_numeric_dtype(serie)
                and not pd.api.types.is_bool_dtype(serie)
            )
        if not tipo_valido:
            erros_tipo.append(f"{coluna}={serie.dtype}")

    if erros_tipo:
        raise TypeError(
            "Feature(s) com dtype incompatível com o contrato V1: "
            f"{erros_tipo}."
        )
    if df["CANCELAMENTO_VOO"].isna().any():
        raise ValueError("CANCELAMENTO_VOO não pode ser nulo no contrato temporal do score.")

    infinitas = []
    for coluna in FEATURES_NUMERICAS_V1:
        valores = df[coluna].dropna().astype("float64")
        if not np.isfinite(valores).all():
            infinitas.append(coluna)
    if infinitas:
        raise ValueError(f"Feature(s) numérica(s) possui(em) valor infinito: {sorted(infinitas)}.")


def validar_contrato_dados_score_pos_viagem(
    df: pd.DataFrame, *, exigir_alvo: bool = False
) -> dict[str, int]:
    """Valida a estrutura da base de treino ou da entrada do score.

    A ausência de ``ID_GOLDENRECORD`` não invalida a base inteira: essas linhas
    ficam fora da validação agrupada e precisam ser contabilizadas. Já um
    ``RESPONDENT_ID`` nulo ou repetido quebra a unidade de análise e interrompe
    o pipeline antes do split ou do ajuste de modelo.

    ``DETRATOR`` é obrigatório apenas na base de treino e avaliação, indicada
    por ``exigir_alvo=True``. A entrada de uma jornada a pontuar não exige o
    target nem a chave de agrupamento e deve passar pela mesma validação com o
    padrão falso.
    """
    obrigatorias = {COLUNA_RESPONDENTE}
    if exigir_alvo:
        obrigatorias.update({COLUNA_CLIENTE, COLUNA_ALVO})
    ausentes = sorted(obrigatorias - set(df.columns))
    if ausentes:
        raise KeyError(f"O contrato de dados exige as colunas: {ausentes}.")

    respondentes = df[COLUNA_RESPONDENTE]
    nulos_respondente = int(respondentes.isna().sum())
    if nulos_respondente:
        raise ValueError(
            f"{COLUNA_RESPONDENTE} possui {nulos_respondente} valor(es) nulo(s); "
            "a unidade de análise exige uma resposta identificável."
        )
    duplicados_respondente = int(respondentes.duplicated().sum())
    if duplicados_respondente:
        raise ValueError(
            f"{COLUNA_RESPONDENTE} possui {duplicados_respondente} valor(es) "
            "duplicado(s); a cardinalidade da base analítica deve ser 1:1."
        )

    if exigir_alvo:
        alvo = df[COLUNA_ALVO]
        # isin considera True == 1 e 1.0 == 1; o target contratado é inteiro,
        # sem coerção implícita de booleanos, floats ou colunas object.
        if not pd.api.types.is_integer_dtype(alvo.dtype):
            raise TypeError(
                f"{COLUNA_ALVO} exige dtype inteiro (0 ou 1); recebido {alvo.dtype}."
            )
        invalidos_alvo = alvo.isna() | ~alvo.isin((0, 1))
        if invalidos_alvo.any():
            raise ValueError(
                f"{COLUNA_ALVO} possui {int(invalidos_alvo.sum())} valor(es) fora do "
                "domínio binário {0, 1}."
            )
        if "NPS_PRINCIPAL" in df.columns:
            esperado = criar_target_detrator(df)
            divergentes = int((alvo != esperado).sum())
            if divergentes:
                raise ValueError(
                    f"{COLUNA_ALVO} diverge de NPS_PRINCIPAL em "
                    f"{divergentes} linha(s)."
                )

    # Materializa apenas a derivação aprovada antes de conferir tipos e allowlist.
    validar_schema_features_v1(materializar_features_v1(df))
    return {
        "linhas_sem_cliente": (
            int(df[COLUNA_CLIENTE].isna().sum()) if COLUNA_CLIENTE in df.columns else 0
        ),
    }


def aplicar_contrato_temporal_score_pos_viagem(df: pd.DataFrame) -> pd.DataFrame:
    """Remove informação indisponível no instante de score de cancelamentos."""
    canceladas = df["CANCELAMENTO_VOO"]
    if not canceladas.any():
        return df

    resultado = df.copy()
    for coluna in FEATURES_POS_ENCERRAMENTO_JORNADA:
        resultado[coluna] = resultado[coluna].mask(canceladas)
    return resultado


def selecionar_features_score_pos_viagem(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Seleciona exclusivamente as features do score pós-viagem V1.

    A função não tenta deduzir segurança por cardinalidade. Toda feature da V1
    é obrigatória e possui dtype contratado; mudanças de schema interrompem o
    pipeline em vez de produzir silenciosamente outro modelo.
    """
    _validar_contrato_feature_set_v1()
    base_features = materializar_features_v1(df)
    validar_schema_features_v1(base_features)
    base_features = aplicar_contrato_temporal_score_pos_viagem(base_features)
    selecionadas = list(FEATURE_SET_V1)
    return base_features.loc[:, selecionadas].copy(), selecionadas, []


def criar_preprocessador_modelagem(df: pd.DataFrame):
    """Cria e ajusta o pré-processador somente nos dados de treino.

    Nulos estruturais recebem categoria explícita apenas na matriz do modelo;
    a base analítica permanece com os nulos originais.
    """
    alvo = "DETRATOR"
    if alvo not in df.columns:
        raise KeyError(f"A modelagem exige a coluna-alvo {alvo}.")
    x, selecionadas, ausentes = selecionar_features_score_pos_viagem(df)
    print(f"features do score pós-viagem: {selecionadas}; ausentes na fonte: {ausentes}")
    categoricas = [coluna for coluna in selecionadas if coluna in FEATURES_CATEGORICAS_V1]
    numericas = [coluna for coluna in selecionadas if coluna in FEATURES_NUMERICAS_V1]
    y = df[alvo]
    indices_treino, indices_teste, metadados_divisao = dividir_treino_teste_temporal_por_cliente(df)
    x_treino, x_teste = x.loc[indices_treino], x.loc[indices_teste]
    y_treino, y_teste = y.loc[indices_treino], y.loc[indices_teste]
    pipeline_numerico = Pipeline([
        ("imputar", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True,
        )),
        ("escalar", RobustScaler()),
    ])
    pipeline_categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="CATEGORIA_AUSENTE")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessador = ColumnTransformer([
        ("numericas", pipeline_numerico, numericas),
        ("categoricas", pipeline_categorico, categoricas),
    ], remainder="drop")
    x_treino_transformado = preprocessador.fit_transform(x_treino)
    x_teste_transformado = preprocessador.transform(x_teste)
    return preprocessador, (x_treino, x_teste, y_treino, y_teste), (x_treino_transformado, x_teste_transformado), (numericas, categoricas), metadados_divisao

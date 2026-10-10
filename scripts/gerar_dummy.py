"""SAFIRA | Baseline dummy da base analítica.

Produz um único artefato, `data/dummy/base_analitica_dummy.parquet`, com as
mesmas 52 colunas, na mesma ordem e com os mesmos tipos de
`data/processed/base_analitica.parquet`, em 10% do volume e com valores
inteiramente sintéticos. Serve para exercitar o que consome a base já
preprocessada — particionamento, features, matriz, modelo — sem tocar no dado do
parceiro, que não é versionável.

Nada é lido de `data/raw/`, e nenhum CSV é escrito. O gerador monta em memória
as 46 colunas que a integração entregaria e chama `preparar_base_analitica`
sobre elas, exatamente como o pipeline faz com o dado real. As seis colunas
derivadas — `DATA_STD_CONVERTIDA`, `MES_ANO`, `DETRATOR`, `CATEGORIA_NPS`,
`TEMPO_VOO_INVALIDO` e `N_TRECHOS` — não são escritas aqui: saem da mesma função
que as produz na base real, e por isso não podem divergir dela. A caixa alta das
categóricas também vem de lá, via `padronizar_categoricas`.

Nenhum valor real é copiado. `RESPONDENT_ID` e `ID_GOLDENRECORD` vêm de faixas
próprias, acima de qualquer chave do dado real, e as demais colunas são
sorteadas a partir de vocabulários públicos (siglas IATA, tipos de aeronave,
canais de compra) ou de distribuições paramétricas ajustadas às marginais da
base analítica.

O que o gerador preserva de propósito, porque quem consome a base depende disso:

- `RESPONDENT_ID` único e crescente com `DATA_STD`, de que dependem a
  anterioridade em `features.adicionar_historico` e o corte temporal de `split`.
- Reincidência de clientes em `ID_GOLDENRECORD`, sem a qual as features de
  histórico ficariam vazias e o corte por cliente não teria o que remover.
- Coerência entre `BASE_AIRPORTLEG`, `EQUIPAMENTO_TIPO`, `VOO_NUMERO`,
  `VOO_TIPO` e `ASSENTOS`: um aeroporto a mais que trechos, e uma aeronave, um
  número de voo e um assento por trecho. É de `BASE_AIRPORTLEG` que o pipeline
  deriva `N_TRECHOS`.
- As ausências que definem o tipo lido: `ID_GOLDENRECORD`, `TEMPO_VOO` e
  `QTDE_VIAGENS_*` saem float64 porque têm nulo, como no real.
- A evolução da série: os tiers `Azul One` e `Diamante Unique` só aparecem a
  partir de 2025-10-24, como na base real, o que os deixa inteiramente do lado
  do teste em relação ao corte de validação. As respostas sem cliente se
  concentram no miolo.

O que este baseline não cobre, por depender só da base analítica: a integração
(`clean.integrar`, `integrar_bases`) e a pós-estratificação
(`clean.pesos_pos_estratificacao`), que leem as fontes e a tabela populacional.

Executar com:  python scripts/gerar_dummy.py
Conferir com:  python scripts/gerar_dummy.py --conferir
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessamento_nps import preparar_base_analitica

SAIDA_PADRAO = Path("data/dummy")
BASE_ANALITICA_REAL = Path("data/processed/base_analitica.parquet")
# Nome próprio, e não o mesmo da real em outra pasta: um parquet dummy que se
# chama igual ao de produção é indistinguível assim que alguém o copia de lugar.
BASE_ANALITICA_DUMMY = "base_analitica_dummy.parquet"
FRACAO = 0.10
SEMENTE = 42

# Trechos da série, em ordem cronológica: (linhas no real, primeira data, última
# data, ausência em ID_GOLDENRECORD). O recorte é o que faz a base dummy
# envelhecer como a real, com bolsões de resposta sem cliente em vez de ausência
# uniforme de ponta a ponta.
PERIODOS = (
    (120000, "2023-07-01", "2024-01-06", 0.0000),
    (120000, "2024-01-06", "2024-07-28", 0.0005),
    (120000, "2024-07-28", "2025-05-13", 0.0003),
    (124915, "2025-05-13", "2026-06-30", 0.0000),
)

# Primeira ocorrência de AZUL ONE e DIAMANTE UNIQUE na base analítica real. Os
# dois tiers saem desta data, e não do índice do período: o quarto período começa
# em 2025-05-13 e atravessa o corte de validação de 2025-07-01, então marcar o
# período inteiro colocaria no treino um nível que na base real só existe no
# teste. A largura da matriz do contrato depende disso: com os dois níveis no
# treino o OneHotEncoder aprende sete e a matriz sai com 40 colunas, onde a real
# tem cinco e 38, e o caminho de categoria não vista deixa de ser exercitado.
PRIMEIRO_TIER_NOVO = "2025-10-24"
LINHAS_REAIS = sum(periodo[0] for periodo in PERIODOS)

# Faixas próprias das chaves, acima das do dado real: um identificador dummy não
# pode apontar para respondente ou cliente que exista.
PRIMEIRO_ID = 100_000_000
PRIMEIRO_CLIENTE = 300_000_000
ULTIMO_CLIENTE = 400_000_000
FRACAO_CLIENTES = 0.85          # clientes distintos por resposta; no real, ~0,85

AEROPORTOS = ("VCP", "CGH", "GRU", "CNF", "SDU", "REC", "POA", "CWB", "BSB",
              "FOR", "SSA", "MAO", "BEL", "VIX", "NAT", "GYN", "FLN", "CGB",
              "MCZ", "UDI", "JPA", "THE", "SLZ", "AJU", "PMW", "IGU", "LDB",
              "JOI", "NVT", "IOS")
EQUIPAMENTOS = ("32N", "32Q", "295", "E95", "AT9", "32I", "339")
# As proporções abaixo são as marginais da base analítica real. A caixa alta
# fica por conta de `padronizar_categoricas`, chamada pelo preparo.
CANAIS = ("Web", "Mobile", "Agency", "Callcenter", "Aeroporto", "Other")
PESO_CANAL = (0.2921, 0.2512, 0.4278, 0.0190, 0.0030, 0.0069)
SEGMENTOS = ("Demais Clientes", "Azul Viagens", "Corporativo")
PESO_SEGMENTO = (0.8941, 0.0207, 0.0852)
TIERS_ANTIGOS = ("Sem cadastro", "Azul Fidelidade", "Topazio", "Safira", "Diamante")
TIERS_NOVOS = TIERS_ANTIGOS + ("Azul One", "Diamante Unique")
# Proporções de TIER_VIAGEM medidas na base real, em dois vetores separados pela
# data acima. São dois e não um porque o peso dos tiers novos é condicional à
# janela: aplicar a marginal global (0,0028) dentro de uma janela que é 15% da
# série produz cinco vezes menos `Azul One` e `Diamante Unique` do que o real tem.
PESO_TIER_ANTES = (0.1213, 0.5064, 0.1190, 0.1014, 0.1519)
PESO_TIER_DEPOIS = (0.1138, 0.4871, 0.0994, 0.1131, 0.1686, 0.0025, 0.0155)
TIPOS_VOO = ("Direto", "Conexão", "Escala")
PESO_TIPO_VOO = (0.7024, 0.2949, 0.0027)
# Distribuição de trechos entre os voos que não são diretos.
TRECHOS_COM_PARADA = (2, 3, 4, 5, 6)
PESO_TRECHOS = (0.8270, 0.1660, 0.0050, 0.0015, 0.0005)
NOTAS_NPS = (100, 0, -100)
PESO_NPS_PRINCIPAL = (0.6498, 0.1457, 0.2044)

# Itens de NPS: nota em {-100, 0, 100} e a ausência medida no dado real, que é o
# que separa um item respondido de um item não apresentado ao respondente.
ITENS_NPS = {
    "NPS_ATRASO": 0.8522, "NPS_BAGAGEM": 0.6561, "NPS_BAGMAO": 0.2008,
    "NPS_CANCELAMENTO24H": 0.9860, "NPS_CKBALCAO": 0.8871, "NPS_CKMOBILE": 0.4565,
    "NPS_CKTOTEM": 0.9976, "NPS_CKWEB": 0.8578, "NPS_COMISSARIOS": 0.2166,
    "NPS_CONFORTO": 0.2671, "NPS_EMBARQUE": 0.1594, "NPS_ENTRETENIMENTO": 0.6455,
    "NPS_LIMPEZA": 0.2451, "NPS_PILOTOS": 0.2339, "NPS_RESAGENCIA": 0.6243,
    "NPS_RESWEB": 0.5301, "NPS_SNACKS": 0.3480, "NPS_AZULFID": 0.7039,
    "NPS_WIFI": 0.9002,
}

# As 46 colunas que a integração entrega ao preparo, na ordem da base real. As
# outras seis são derivadas por `preparar_base_analitica` e não aparecem aqui de
# propósito: repeti-las seria criar uma segunda definição das mesmas colunas,
# livre para divergir da do pipeline no dia em que a regra mudar.
COLUNAS_INTEGRADA = (
    "RESPONDENT_ID", "ID_GOLDENRECORD", "DATA_STD", "EQUIPAMENTO_TIPO",
    "BASE_AIRPORTLEG", "VOO_TIPO", "VOO_NUMERO", "TIPO_ENTRETENIMENTO",
    "NPS_PRINCIPAL", *ITENS_NPS, "SUB_ENTRETENIMENTO1", "SUB_ENTRETENIMENTO2",
    "SUB_FIL_MOTIVOVIAGEM", "SUB_FIL_FREQUENCIAAZUL", "VOO_INTERNACIONAL",
    "TEMPO_VOO", "CANAL_COMPRA", "TEMPO_VOO_CONSOLIDADO", "SEGMENTO",
    "TIER_VIAGEM", "QTDE_VIAGENS_12M", "QTDE_VIAGENS_24M", "QTDE_VIAGENS_36M",
    "CANCELAMENTO_VOO", "ANTECEDENCIA_CANCELAMENTO", "ATRASO_CHEGADA",
    "ESTATISTICA_ATRASOSAIDA", "ASSENTOS",
)


def _ausentar(rng, valores, taxa):
    """Devolve a série com `taxa` das posições sorteadas trocadas por ausência."""
    s = pd.Series(list(valores))
    return s.mask(rng.random(len(s)) < taxa) if taxa else s


def _categorica(rng, n, valores, pesos=None, ausencia=0.0):
    return _ausentar(rng, rng.choice(valores, n, p=pesos), ausencia)


def _pesos(valores):
    """Renormaliza para somar 1, para poder recortar um prefixo da tupla."""
    p = np.asarray(valores, dtype=float)
    return p / p.sum()


def _juntar(partes):
    """Concatena por linha as listas de trechos, no formato 'A/B/C'."""
    return ["/".join(p) for p in partes]


def _esqueleto(rng, fracao):
    """Monta a espinha dorsal: chave, cliente, data, tipo de voo e nº de trechos.

    Sai daqui tudo que as outras colunas precisam enxergar para ficarem coerentes
    entre si, em vez de cada bloco sortear por conta própria e o itinerário
    acabar discordando do tipo de voo ou do número de assentos.
    """
    tamanhos = [max(1, round(linhas * fracao)) for linhas, *_ in PERIODOS]
    inicios = np.cumsum([0] + tamanhos[:-1])
    total = sum(tamanhos)

    # Chave única e crescente: a ordem por RESPONDENT_ID tem que reproduzir a
    # cronologia, de que dependem a anterioridade do histórico e o corte temporal.
    resp = PRIMEIRO_ID + np.cumsum(rng.integers(1, 40, total))

    # Clientes sorteados de um conjunto menor que o de respostas, para que haja
    # reincidência e as features de histórico tenham o que enxergar.
    clientes = rng.choice(np.arange(PRIMEIRO_CLIENTE, ULTIMO_CLIENTE),
                          max(1, round(total * FRACAO_CLIENTES)), replace=False)
    golden = pd.Series(rng.choice(clientes, total), dtype="float64")

    datas = []
    sem_cliente = np.zeros(total, dtype=bool)
    for (_, ini, fim, taxa_nula), n, pos in zip(PERIODOS, tamanhos, inicios):
        ini, fim = pd.Timestamp(ini), pd.Timestamp(fim)
        datas.append(ini + pd.to_timedelta(
            np.sort(rng.integers(0, (fim - ini).days + 1, n)), unit="D"))
        if taxa_nula:
            sem_cliente[pos:pos + n] = rng.random(n) < taxa_nula
    golden = golden.mask(sem_cliente)

    data_std = pd.DatetimeIndex(np.concatenate(datas)).as_unit("us")
    # Da data, nunca do período: ver PRIMEIRO_TIER_NOVO.
    tier_novo = data_std >= pd.Timestamp(PRIMEIRO_TIER_NOVO)

    # Voo direto tem um trecho; conexão e escala têm de dois a seis, na proporção
    # condicional que N_TRECHOS tem na base analítica real.
    tipo = rng.choice(TIPOS_VOO, total, p=_pesos(PESO_TIPO_VOO))
    trechos = np.where(tipo == TIPOS_VOO[0], 1,
                       rng.choice(TRECHOS_COM_PARADA, total, p=_pesos(PESO_TRECHOS)))

    return pd.DataFrame({
        "RESPONDENT_ID": resp,
        "ID_GOLDENRECORD": golden,
        "DATA_STD": data_std,
        "VOO_TIPO": tipo,
        "N_TRECHOS": trechos,
        "TIER_NOVO": tier_novo,
    })


def _integrada(rng, base):
    """Devolve as 46 colunas que a integração entregaria ao preparo."""
    n = len(base)
    trechos = base["N_TRECHOS"].to_numpy()
    sem_cliente = base["ID_GOLDENRECORD"].isna().to_numpy()

    # Tempo de voo cresce com o número de trechos e inclui a espera em conexão.
    por_trecho = np.clip(rng.lognormal(4.45, 0.45, n), 35, 700)
    tempo = np.clip(por_trecho * trechos + 60 * (trechos - 1), 35, 4320).round()

    # 80% dos voos chegam sem atraso; quando atrasam, o atraso na saída explica a
    # maior parte do atraso na chegada.
    chegada = np.where(rng.random(n) < 0.796, 0,
                       np.clip(rng.lognormal(3.5, 1.3, n), 1, 4319)).round()
    saida = np.where(
        chegada == 0,
        np.where(rng.random(n) < 0.60, 0, np.clip(rng.lognormal(2.0, 0.8, n), 1, 60)),
        chegada * rng.uniform(0.6, 1.1, n),
    ).round()
    cancelado = rng.random(n) < 0.089

    # Contagem de viagens ausente onde o cliente também é: sem cadastro, sem
    # histórico. É a mesma ausência conjunta que a base real tem.
    v12 = rng.poisson(3, n) + rng.binomial(1, 0.1, n) * rng.poisson(12, n)
    v24 = v12 + rng.poisson(4, n)
    v36 = v24 + rng.poisson(5, n)

    def contagem(v):
        return pd.Series(v, dtype="float64").mask(sem_cliente)

    # Os tiers novos só existem a partir de PRIMEIRO_TIER_NOVO; antes disso os
    # pesos dos cinco originais são renormalizados entre si.
    tier = np.where(base["TIER_NOVO"].to_numpy(),
                    rng.choice(TIERS_NOVOS, n, p=_pesos(PESO_TIER_DEPOIS)),
                    rng.choice(TIERS_ANTIGOS, n, p=_pesos(PESO_TIER_ANTES)))

    # Uma única resposta com TEMPO_VOO consolidado, como no real: é a marca que a
    # integração deixa ao resolver o par duplicado aprovado de NPS_04.
    consolidado = np.zeros(n, dtype="int64")
    consolidado[0] = 1

    df = pd.DataFrame({
        "RESPONDENT_ID": base["RESPONDENT_ID"].to_numpy(),
        "ID_GOLDENRECORD": base["ID_GOLDENRECORD"].to_numpy(),
        "DATA_STD": base["DATA_STD"].to_numpy(),
        "EQUIPAMENTO_TIPO": _juntar(rng.choice(EQUIPAMENTOS, t) for t in trechos),
        "BASE_AIRPORTLEG": _juntar(rng.choice(AEROPORTOS, t + 1, replace=False)
                                   for t in trechos),
        "VOO_TIPO": base["VOO_TIPO"].to_numpy(),
        "VOO_NUMERO": _juntar(rng.integers(2100, 9900, t).astype(str) for t in trechos),
        "TIPO_ENTRETENIMENTO": _categorica(
            rng, n, ("AO VIVO", "GRAVADO", "NÃO TEM ENTRETENIMENTO"),
            (0.55, 0.20, 0.25), 0.2949),
        "NPS_PRINCIPAL": rng.choice(NOTAS_NPS, n, p=_pesos(PESO_NPS_PRINCIPAL)),
    })
    for item, ausencia in ITENS_NPS.items():
        df[item] = _ausentar(
            rng, rng.choice(NOTAS_NPS, n, p=(0.72, 0.12, 0.16)), ausencia
        ).astype("float64")
    df["SUB_ENTRETENIMENTO1"] = _categorica(
        rng, n, ("Sim", "Não", "Não assisti"), (0.45, 0.40, 0.15), 0.6904)
    df["SUB_ENTRETENIMENTO2"] = _categorica(
        rng, n, ("Monitor", "Áudio", "Controle", "Sinal Ruim",
                 "Não tinha canal ao vivo"), None, 0.9336)
    df["SUB_FIL_MOTIVOVIAGEM"] = _categorica(
        rng, n, ("Lazer", "Trabalho", "Pessoal", "Trabalho combinado com lazer"),
        (0.55, 0.25, 0.14, 0.06), 0.0061)
    df["SUB_FIL_FREQUENCIAAZUL"] = _categorica(
        rng, n, ("De 2 a 5 vezes por ano", "De 6 a 10 vezes por ano",
                 "Esta foi a primeira vez",
                 "Realizo pelo menos 1 viagem por mês pela Azul"),
        (0.48, 0.20, 0.22, 0.10), 0.0081)
    df["VOO_INTERNACIONAL"] = "Domestic"
    df["TEMPO_VOO"] = _ausentar(rng, tempo, 0.0005).astype("float64")
    df["CANAL_COMPRA"] = rng.choice(CANAIS, n, p=_pesos(PESO_CANAL))
    df["TEMPO_VOO_CONSOLIDADO"] = consolidado
    df["SEGMENTO"] = rng.choice(SEGMENTOS, n, p=_pesos(PESO_SEGMENTO))
    df["TIER_VIAGEM"] = tier
    df["QTDE_VIAGENS_12M"] = contagem(v12)
    df["QTDE_VIAGENS_24M"] = contagem(v24)
    df["QTDE_VIAGENS_36M"] = contagem(v36)
    df["CANCELAMENTO_VOO"] = cancelado
    # Só voo cancelado tem antecedência de aviso.
    df["ANTECEDENCIA_CANCELAMENTO"] = pd.Series(
        rng.exponential(12, n).round().clip(0, 400)).where(cancelado)
    df["ATRASO_CHEGADA"] = chegada.astype("int64")
    df["ESTATISTICA_ATRASOSAIDA"] = saida.astype("int64")
    df["ASSENTOS"] = _ausentar(rng, _juntar(
        [f"{f}{c}" for f, c in zip(rng.integers(1, 31, t),
                                   rng.choice(tuple("ABCDEF"), t))]
        for t in trechos), 0.0012)
    return df[list(COLUNAS_INTEGRADA)]


def gerar(saida: Path | str = SAIDA_PADRAO, fracao: float = FRACAO,
          semente: int = SEMENTE) -> dict[str, int]:
    """Escreve o parquet da base analítica dummy e devolve {arquivo: nº de linhas}.

    As colunas derivadas não são escritas aqui: `preparar_base_analitica` é
    chamada sobre o quadro sintético, igual ao que o pipeline faz com o dado
    real. É o que mantém alvo, categorias e `N_TRECHOS` idênticos aos da base
    real mesmo quando a regra que os produz mudar.
    """
    rng = np.random.default_rng(semente)
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)

    base = preparar_base_analitica(_integrada(rng, _esqueleto(rng, fracao)))
    base.to_parquet(saida / BASE_ANALITICA_DUMMY, index=False)
    return {BASE_ANALITICA_DUMMY: len(base)}


def conferir(real: Path | str = BASE_ANALITICA_REAL,
             dummy: Path | str = SAIDA_PADRAO, fracao: float = FRACAO) -> bool:
    """Compara colunas, tipos e volume do dummy contra a base analítica real.

    É a evidência dos critérios de aceite, e não um teste da suíte: exige o dado
    do parceiro em disco.
    """
    real, dummy = Path(real), Path(dummy)
    if not real.exists():
        print(f"Sem contraparte em {real}: nada a conferir.")
        return False

    r = pd.read_parquet(real)
    d = pd.read_parquet(dummy / BASE_ANALITICA_DUMMY)
    colunas = list(r.columns) == list(d.columns)
    tipos = {c: (str(r[c].dtype), str(d[c].dtype))
             for c in r.columns if colunas and r[c].dtype != d[c].dtype}
    # Tolerância de meio ponto percentual: o arredondamento por período e as
    # linhas que o preparo descarta afastam o volume da fração exata.
    volume = abs(len(d) / len(r) - fracao) < 0.005
    print(f"{BASE_ANALITICA_DUMMY:30s} {len(d):6d}/{len(r):7d} = {len(d) / len(r):6.2%}"
          f"  volume={'ok' if volume else 'DIVERGE'}"
          f"  colunas={'ok' if colunas else 'DIVERGEM'}"
          f"  tipos={'ok' if not tipos else tipos}")
    if not colunas:
        print(f"  só no real : {[c for c in r.columns if c not in d.columns]}")
        print(f"  só no dummy: {[c for c in d.columns if c not in r.columns]}")

    ok = volume and colunas and not tipos
    print("CONFERE" if ok else "NÃO CONFERE")
    return ok


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--saida", default=SAIDA_PADRAO, type=Path)
    p.add_argument("--conferir", action="store_true",
                   help="compara o dummy já gerado com a base analítica real e sai")
    p.add_argument("--fracao", default=FRACAO, type=float)
    p.add_argument("--semente", default=SEMENTE, type=int)
    args = p.parse_args()

    if args.conferir:
        raise SystemExit(0 if conferir(dummy=args.saida, fracao=args.fracao) else 1)

    for nome, linhas in gerar(args.saida, args.fracao, args.semente).items():
        print(f"{nome:30s} {linhas:7d} linha(s)")
    print(f"Use com:  pd.read_parquet('{args.saida / BASE_ANALITICA_DUMMY}')")

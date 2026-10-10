# Contrato de Dados do Score Pós-Viagem

## Finalidade

Este documento fixa o contrato da base analítica usada pelo score pós-viagem do SAFIRA. A validação de schema é executável em `scripts/preprocessamento_nps.py`; o split temporal agrupado e o ajuste do pré-processador somente no treino são aplicados em `src/split.py` e `src/matriz.py`.

## Unidade e Chaves

- Cada linha representa uma resposta ou jornada identificada por `RESPONDENT_ID`.
- `RESPONDENT_ID` é obrigatório, não pode ser nulo e deve ser único na base analítica.
- `ID_GOLDENRECORD` identifica o Cliente e é usado exclusivamente para agrupamento na validação. Ele é obrigatório no treino e na avaliação, mas não na entrada de score.
- Registros de treino e avaliação sem `ID_GOLDENRECORD` não são corrigidos nem transformados em grupos unitários. Eles ficam fora das partições agrupadas, com a quantidade registrada em metadados.
- Fontes transacionais unidas por `RESPONDENT_ID` devem obedecer cardinalidade 1:1; qualquer multiplicação de linhas interrompe a integração.

## Target

- Target de modelagem: `DETRATOR`.
- Classe positiva: `NPS_PRINCIPAL == -100`.
- Classe negativa: `NPS_PRINCIPAL` igual a `0` ou `100`.
- `criar_target_detrator` em `scripts/preprocessamento_nps.py` valida a escala de `NPS_PRINCIPAL` e deriva `DETRATOR`.
- Na base de treino e avaliação, `DETRATOR` deve ter dtype inteiro (inclusive `Int64` nulável, desde que sem nulos) e pertencer ao domínio binário `{0, 1}`. Booleanos, floats e colunas `object` não são aceitos, mesmo quando equivalem numericamente a `0` ou `1`; o target nunca compõe `X`.
- Na entrada de uma jornada a pontuar, `DETRATOR` não existe e não é exigido pelo contrato.

## Feature Set V1

A allowlist implementada contém exatamente 11 features:

`TIER_VIAGEM`, `VOO_TIPO`, `TIPO_ENTRETENIMENTO`, `CANAL_COMPRA`, `SEGMENTO`, `ESTATISTICA_ATRASOSAIDA`, `ATRASO_CHEGADA`, `CANCELAMENTO_VOO`, `ANTECEDENCIA_CANCELAMENTO`, `TEMPO_VOO` e `N_TRECHOS`.

- `TIER_VIAGEM` substitui `PERFIL_TUDOAZUL` para evitar redundância semântica.
- `N_TRECHOS` é a única derivação de rota aprovada e deriva de `BASE_AIRPORTLEG`.
- `QTDE_VIAGENS_12M` fica fora até que exista uma contagem reconstruída com corte estrito em `t_score`.

## Leakage e Disponibilidade Temporal

- Nenhuma feature pode estar disponível depois de `t_score`.
- `NPS_PRINCIPAL`, `DETRATOR`, `CATEGORIA_NPS`, qualquer `NPS_*`, qualquer `SUB_*` e identificadores não entram como preditores.
- Para voos cancelados, `ESTATISTICA_ATRASOSAIDA`, `ATRASO_CHEGADA`, `TEMPO_VOO` e `N_TRECHOS` são mascaradas como ausentes, pois dependem do encerramento da jornada.
- Missing estrutural é preservado. O pipeline não aplica `fillna(0)` global.

## Validação

- `validar_contrato_dados_score_pos_viagem` valida `RESPONDENT_ID`, o dtype e domínio de `DETRATOR` no treino e na avaliação, a chave de agrupamento no treino e na avaliação, e o schema da allowlist. Quando `NPS_PRINCIPAL` está presente no treino ou na avaliação, também valida sua escala `{-100, 0, 100}` e a coerência linha a linha com `DETRATOR` (`1` se e somente se `NPS_PRINCIPAL == -100`).
- `split.dividir` aplica o split temporal agrupado por `ID_GOLDENRECORD`, exclui linhas sem Cliente e impede que um Cliente exista em mais de uma partição.
- `preparar_matriz` ajusta imputação, encoding e escalonamento somente no treino.
- `validar_schema_features_v1` e os testes em `tests/test_contrato_dados.py` fazem schema inválido, duplicidade de `RESPONDENT_ID`, target fora do domínio e dtypes incompatíveis falharem explicitamente.

## Segurança

Testes utilizam fixtures sintéticas. Dados reais, credenciais e outputs de notebooks não podem ser versionados.

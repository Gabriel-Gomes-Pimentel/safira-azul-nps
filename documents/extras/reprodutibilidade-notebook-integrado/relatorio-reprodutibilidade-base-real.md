# Relatório de reprodutibilidade do notebook integrado

Gerado em 2026-09-25T11:49:50. O notebook `notebooks/comparacao_modelos.ipynb` foi executado 2 vezes consecutivas, cada uma num kernel novo, na base real (`data/processed/base_analitica.parquet`), com a semente `SEMENTE = 42`.

## Critérios de aceite

| Critério | Descrição | Resultado |
| --- | --- | --- |
| CR01 | Métricas idênticas em execuções consecutivas com a mesma semente | atende |
| CR02 | Logs de métricas iguais linha a linha em todas as execuções | atende |
| CR03 | Nenhuma variação superior a 0,01% nas métricas | atende (maior variação: 0,000000%) |

## Sementes dos modelos

| Modelo | `random_state` |
| --- | --- |
| Extra Trees | 42 |
| Regressão Logística | 42 |
| Árvore de Decisão | 42 |
| Random Forest | 42 |
| Gradient Boosting | 42 |
| Classe Majoritária | não se aplica (estratégia `most_frequent`, determinística) |

## Métricas por execução

| Tabela | Modelo | Métrica | Execução 1 | Execução 2 | Diferença (%) | Idêntico |
| --- | --- | --- | --- | --- | --- | --- |
| calibracao | Gradient Boosting | brier | 0.18327336776388378 | 0.18327336776388378 | 0,000000 | sim |
| calibracao | Gradient Boosting | precisao_media | 0.5173965896494139 | 0.5173965896494139 | 0,000000 | sim |
| calibracao | Gradient Boosting | roc_auc | 0.7330095901545676 | 0.7330095901545676 | 0,000000 | sim |
| calibracao | Gradient Boosting calibrado | brier | 0.14047486762990488 | 0.14047486762990488 | 0,000000 | sim |
| calibracao | Gradient Boosting calibrado | precisao_media | 0.5173965896494139 | 0.5173965896494139 | 0,000000 | sim |
| calibracao | Gradient Boosting calibrado | roc_auc | 0.7330095901545676 | 0.7330095901545676 | 0,000000 | sim |
| calibracao | probabilidade constante | brier | 0.1694812994224446 | 0.1694812994224446 | 0,000000 | sim |
| calibracao | probabilidade constante | precisao_media | 0.21599966874391835 | 0.21599966874391835 | 0,000000 | sim |
| calibracao | probabilidade constante | roc_auc | 0.5 | 0.5 | 0,000000 | sim |
| resultados | Classe Majoritária | brier | 0.21599966874391835 | 0.21599966874391835 | 0,000000 | sim |
| resultados | Classe Majoritária | precisao_media | 0.21599966874391835 | 0.21599966874391835 | 0,000000 | sim |
| resultados | Classe Majoritária | roc_auc | 0.5 | 0.5 | 0,000000 | sim |
| resultados | Extra Trees | brier | 0.19359582636319558 | 0.19359582636319558 | 0,000000 | sim |
| resultados | Extra Trees | precisao_media | 0.48578382481608084 | 0.48578382481608084 | 0,000000 | sim |
| resultados | Extra Trees | roc_auc | 0.7232473538742756 | 0.7232473538742756 | 0,000000 | sim |
| resultados | Gradient Boosting | brier | 0.18327336776388378 | 0.18327336776388378 | 0,000000 | sim |
| resultados | Gradient Boosting | precisao_media | 0.5173965896494139 | 0.5173965896494139 | 0,000000 | sim |
| resultados | Gradient Boosting | roc_auc | 0.7330095901545676 | 0.7330095901545676 | 0,000000 | sim |
| resultados | Random Forest | brier | 0.1585727841163558 | 0.1585727841163558 | 0,000000 | sim |
| resultados | Random Forest | precisao_media | 0.42708279834126234 | 0.42708279834126234 | 0,000000 | sim |
| resultados | Random Forest | roc_auc | 0.6716429297992469 | 0.6716429297992469 | 0,000000 | sim |
| resultados | Regressão Logística | brier | 0.18771627084363351 | 0.18771627084363351 | 0,000000 | sim |
| resultados | Regressão Logística | precisao_media | 0.5010808304230082 | 0.5010808304230082 | 0,000000 | sim |
| resultados | Regressão Logística | roc_auc | 0.7272930642829576 | 0.7272930642829576 | 0,000000 | sim |
| resultados | Árvore de Decisão | brier | 0.19467799956613308 | 0.19467799956613308 | 0,000000 | sim |
| resultados | Árvore de Decisão | precisao_media | 0.4415485144019479 | 0.4415485144019479 | 0,000000 | sim |
| resultados | Árvore de Decisão | roc_auc | 0.6799828250627618 | 0.6799828250627618 | 0,000000 | sim |
| score_medio | Gradient Boosting | calibrado | 0.1890217138862885 | 0.1890217138862885 | 0,000000 | sim |
| score_medio | Gradient Boosting | sem_calibracao | 0.4190245542611301 | 0.4190245542611301 | 0,000000 | sim |

## Diferenças entre os logs

Nenhuma: os 2 logs são idênticos linha a linha (38 linhas cada).

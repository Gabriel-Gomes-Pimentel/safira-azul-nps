# Relatório de validação do corte de leakage

Gerado por `scripts/validar_leakage.py` em 2026-09-23 (card #198).

- Base: `data\dummy\base_analitica_dummy.parquet` (48492 linhas)
- Cortes: validação a partir de 2025-07-01, teste a partir de 2026-01-01
- Resultado: **APROVADO** (9/9 verificações)

## Composição dos conjuntos

| Conjunto | n | Clientes | Início | Fim | Taxa de detração (%) |
|---|---|---|---|---|---|
| treino | 28529 | 18860 | 2023-07-01 | 2025-06-30 | 20.58 |
| validacao | 4911 | 4606 | 2025-07-01 | 2025-12-31 | 20.99 |
| teste | 5427 | 5064 | 2026-01-01 | 2026-06-30 | 20.03 |

Fora dos conjuntos: 9617 linha(s) por Cliente recorrente, 8 sem Cliente e 0 sem data.

## Verificações

CR01: a maior data do conjunto anterior deve ser menor que a menor data do posterior. CR02: a interseção de IDs entre conjuntos deve ser vazia.

| Critério | Verificação | Resultado | Detalhe |
|---|---|---|---|
| CR01 | treino antes de validacao | ok | max(treino) = 2025-06-30 < min(validacao) = 2025-07-01 |
| CR01 | treino antes de teste | ok | max(treino) = 2025-06-30 < min(teste) = 2026-01-01 |
| CR01 | validacao antes de teste | ok | max(validacao) = 2025-12-31 < min(teste) = 2026-01-01 |
| CR02 | RESPONDENT_ID: treino x validacao | ok | 0 ID(s) em comum |
| CR02 | RESPONDENT_ID: treino x teste | ok | 0 ID(s) em comum |
| CR02 | RESPONDENT_ID: validacao x teste | ok | 0 ID(s) em comum |
| CR02 | ID_GOLDENRECORD: treino x validacao | ok | 0 ID(s) em comum |
| CR02 | ID_GOLDENRECORD: treino x teste | ok | 0 ID(s) em comum |
| CR02 | ID_GOLDENRECORD: validacao x teste | ok | 0 ID(s) em comum |

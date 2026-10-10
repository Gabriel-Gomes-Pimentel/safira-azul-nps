# Protocolo de avaliação do Artefato 7 — referência técnica do time

Consulta rápida para as quatro duplas de modelagem (Regressão Logística, Árvore de Decisão,
Random Forest e Gradient Boosting): o que usar para tunar e medir o seu modelo, e onde está o
código. Este arquivo **consolida**; não decide nada de novo e não é o texto da Seção 4.4. A
redação acadêmica, com citação ABNT, é o card #244.

Esta referência se apoia em dois artefatos: `documents/extras/protocolo-avaliacao-artefato7.md`,
que registra a escolha do F2 (card #238), e `src/scorer_f2.py`, que o implementa (card #242).

## Resumo

| Item | O que usar | Onde está |
|---|---|---|
| Critério de busca de hiperparâmetro | F2 (F-beta, beta = 2), como `scoring` do `GridSearchCV`/`RandomizedSearchCV` | `src/scorer_f2.py` (`scorer_f2`), card #242 |
| Métricas de negócio reportadas | Sensibilidade (Recall), Precisão Média (AP) e ROC-AUC, com metas 0,70 / 0,40 / 0,75 | Seção 4.3.2 |
| Partição treino / validação / teste | Temporal por `DATA_STD` e agrupada por Cliente; cortes 01/07/2025 e 01/01/2026 | `split.dividir`, `documents/extras/politica-de-particionamento-temporal.md` |
| Validação cruzada dentro do treino | `GroupKFold` com **5 folds** (`N_FOLDS`) | `validacao.criar_folds` |
| Chave de agrupamento | `ID_GOLDENRECORD` | `split.COLUNA_CLIENTE` |
| Semente | 42 | `congelamento.SEMENTE` |
| Limiar operacional | Definido por capacidade de contato (50 contatos/dia), depois do tuning; nunca entra na busca | Seção 4.3.2, card #247 |

## Como usar

1. Monte o `Pipeline` (pré-processamento + estimador) sobre o contrato de dados. O ajuste do
   pré-processador acontece dentro de cada fold, nunca fora.
2. Gere os folds com `validacao.criar_folds` e passe a lista em `cv=`. Não use `KFold`,
   `StratifiedKFold` nem `train_test_split` soltos: eles colocam respostas do mesmo Cliente em
   lados diferentes e inflam a métrica.
3. Passe `scoring=scorer_f2`. O F2 só orienta a escolha dos hiperparâmetros.
4. Ao reportar o modelo, use Sensibilidade, Precisão Média e ROC-AUC, os mesmos nomes da
   Seção 4.3.2. O F2 não substitui essas métricas na tabela comparativa.
5. Não escolha limiar na busca. O `scorer_f2` avalia o corte do próprio estimador; o limiar
   operacional é derivado da capacidade de contato depois que os quatro modelos estão tunados.

## Onde cada decisão está justificada

- Por que F2 e não outra métrica, e por que acurácia fica de fora:
  `documents/extras/protocolo-avaliacao-artefato7.md` (card #238).
- Por que o particionamento é temporal e por Cliente: Seção 4.2.3 e
  `documents/extras/politica-de-particionamento-temporal.md` (card #127).
- Por que a validação interna é `GroupKFold` por Cliente: docstring de `src/validacao.py`
  (card #102).
